# Bare-Metal MNIST Inference on a Custom RISC-V + INT8 Matmul Accelerator

End-to-end inference on a custom RV32IM core with a hardware matmul accelerator, verified in cycle-accurate Verilator simulation and pushed through a Sky130 OpenLane flow.

## Latest Verified Result

Current local run (`make run`, Feb 18, 2026):

```text
Loaded 11619 lines from inference.hex
*** PASSED *** after 1999813 simulation cycles
Total cycles: 1999813
```

| Metric | Value |
|---|---|
| ISA | RV32IM (+ Zicsr in toolchain flags) |
| CPU pipeline | 4-stage (IF, ID, EX, WB) |
| Accelerator | Decoupled 4x4 INT8 outer-product matmul engine, DMA-fed |
| Workload | 100 MNIST images, 2-layer MLP (784 -> 128 -> 10) |
| Accuracy | 100/100 correct in current test set |
| Cycles | 1,999,813 total (~19,998 cycles/image) |
| Speedup vs unoptimized baseline (48,886,157 cycles) | 24.45x |
| Binary size | ~190 KB ELF / ~182 KB raw bin |

## What Is In This Repo vs Sibling Repos

This repo (`/Users/peter/verilator`) is the integration and simulation workspace:
- `inference_bare.c`: bare-metal runtime and accelerator driver.
- `riscv-accelerator/MatmulAccelerator.sv`: custom accelerator RTL.
- `sim_main.cpp`: Verilator harness that loads `inference.hex` at reset PC base `0x2000`.
- `OpenLane/`: local OpenLane flow checkout used for Sky130 implementation experiments.

Sibling repos used by this flow:
- `/Users/peter/asic-project-fa25-golden-gates`: CPU + SoC RTL (`src/Riscv151.v`, `src/riscv_top.v`, pipeline stages, memory subsystem).

Also vendored into this repo:
- `riscv-ml-inference/`: model training/export pipeline (`train/train_and_export.py`, `runtime/weights.h`, `runtime/test_images.h`).

## Architecture Snapshot

### CPU
- 4-stage pipeline with clear stage split so decode, execute, and writeback are no longer collapsed.
- IF stage handles PC generation, instruction fetch, and branch prediction lookups.
- ID stage handles decode, register read, hazard checks, and forwarding select decisions.
- EX stage handles ALU, branch resolution, and RV32M multiply/divide execution.
- WB stage handles memory-return path and final register writeback.
- Hazard behavior includes forwarding for common RAW cases, one-cycle load-use bubbles, and a mispredict flush path.

### Accelerator (`riscv-accelerator/MatmulAccelerator.sv`)
- MMIO base: `0x80000000`.
- Computes a 4x4 output tile (`int32` accumulators) from packed INT8 vectors.
- Uses outer-product accumulation over K in 4-element chunks: `C_tile += A_step(4x1) x B_step(1x4)`.
- Each compute step performs up to 16 INT8 multiplies and accumulates into 16 INT32 cells.
- Reads 128-bit chunks through DMA interface.
- Command FIFO decouples software launch from execution.
- Result registers: `0x18` to `0x54`.
- Added control for strided input traversal:
  - `0x58`: `X_STRIDE`
  - `0x5C`: `K_ROW_LEN`

## End-to-End Execution Path

1. PyTorch script in `riscv-ml-inference/train/train_and_export.py` trains and exports quantized INT8 weights/test vectors.
2. `inference_bare.c` compiles into a flat bare-metal image (`inference.hex`) with weights in `.rodata`.
3. `sim_main.cpp` loads the image into simulated SRAM at reset base `0x2000`.
4. CPU executes the inference loop and launches accelerator commands through MMIO.
5. Accelerator DMA reads packed activations/weights, performs 4x4 outer-product accumulation, and exposes tile results via MMIO.
6. Runtime applies bias/rescale/argmax and reports PASS/FAIL through CSR `0x51e`.

### Software Runtime (`inference_bare.c`)
- No OS, no libc startup (`-nostdlib -nostartfiles`).
- Packed INT8 data layout for accelerator-friendly access.
- Batch size 4 inference loop.
- Pass/fail signaling via CSR `0x51e` (`tohost` convention in testbench).

## Performance Progression (Consistent Timeline)

Historical cycle counts from improvement logs and current run:

| Stage | Cycles | Speedup vs baseline |
|---|---:|---:|
| Baseline (unoptimized SW) | 48,886,157 | 1.00x |
| + Software optimizations (`-O3`, fused ops, reciprocal scaling) | 44,830,519 | 1.09x |
| + Pipeline/branch fixes | 44,830,278 | 1.09x |
| + V1 accelerator (shared 32-bit dcache path) | 6,698,993 | 7.30x |
| + Separate 128-bit DMA + batched inference | 2,914,655 | 16.77x |
| + 4-stage pipeline refactor | 3,007,447 | 16.26x |
| + Phase-2 accelerator/MMIO/driver fixes (current) | 1,999,813 | 24.45x |

Derived throughput at the current point:
- Total MACs: ~10.16M (100 images x (100,352 + 1,280)).
- Effective throughput: ~0.197 cycles/MAC.

Detailed logs live in:
- `improvements/2026-02-15.md`
- `improvements/2026-02-16.md`
- `improvements/2026-02-18.md`

## Why The Latest Jump Happened

The latest acceleration came from correctness and interface fixes, not just adding more hardware:
- MMIO map synchronization (results moved to `0x18` and kept separate from config regs).
- Packed-weight bug fix (`w0..w3` packing correctness and casting discipline).
- Correct block addressing in software (`blk * K * 4` style offsets).
- Correct `K` dimension passed to accelerator (previously truncated behavior).
- Driver hardening and queue-safe launch/poll behavior.

## Open-Source Toolchain Used

This project was built entirely with open-source tooling:
- Verilator for cycle-accurate RTL simulation.
- RISC-V GNU toolchain (`riscv64-unknown-elf-gcc/objcopy/objdump`) for bare-metal binaries.
- OpenLane flow (local checkout in this repo) for RTL-to-GDS flow orchestration.
- OpenLane components: OpenROAD, Yosys, Magic, Netgen, KLayout, CVC, SPEF extractor.
- Sky130 PDK for physical-design experiments.

## Local Docker-Based Sky130 Flow

OpenLane is run locally in Docker containers:

```bash
cd OpenLane
make mount
./flow.tcl -design riscv_top -to synthesis
```

Full place-and-route attempt:

```bash
cd OpenLane
make mount
./flow.tcl -design riscv_top
```

## Sky130 Status (Current, Honest State)

From recent OpenLane PAR artifacts in this workspace:

| Metric | Current observation |
|---|---|
| Run | `sram_run_par` |
| Die area | 1.44 mm^2 (1200 um x 1200 um floorplan) |
| Core area | 1,399,932.65 um^2 |
| OpenDP utilization | 22.9% |
| Synthesized cell count | 21,507 |
| Total cells after fill/tap/decap | 115,402 |
| RCX SPEF timing | WNS 0.00 ns, TNS 0.00 ns at 20 ns clock |
| Suggested frequency from this run | ~50 MHz |
| Flow status | Marked failed at Magic GDS pointer stage (`gds_ptrs`), after routing/signoff reports were generated |

Additional exploratory run:
- `RUN_2026.02.18_07.35.00` kept the same 1.44 mm^2 area with higher density (OpenDP 28.45%).
- RCX SPEF timing there was near-close at 17 ns target with `spef_wns = -0.36 ns` and `spef_tns = -1.79 ns`.
- The same late Magic `gds_ptrs` step failure marked the flow as failed.

Interpretation:
- The design is functionally strong in RTL/simulation.
- Physical implementation quality is significantly better than the old 4 mm^2 snapshot and now reaches completed routing and signoff report generation in 1.44 mm^2 runs.
- Remaining blockers are late flow robustness issues and final timing margin at tighter clock targets.

## Reproduce The Simulation Result

1. Ensure toolchain dependencies are installed:
   - Verilator
   - `riscv64-unknown-elf-*`
   - Python 3
2. Check local path settings in `Makefile`:
   - `RTL_DIR`
   - `INFER_DIR`
   - `START_S`
3. Build and run:

```bash
make clean
make run
```

Expected: `*** PASSED *** after 1999813 simulation cycles` (or very close, depending on exact binary/image revision).

## Acknowledgment

Special thanks to Tingyao Huang for major CPU-core development contributions used in this end-to-end flow.
