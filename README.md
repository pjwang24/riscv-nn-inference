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
| Accelerator | 4x4 INT8 outer-product engine, DMA-fed |
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
- `/Users/peter/riscv-ml-inference`: model training/export pipeline (`train/train_and_export.py`, `runtime/weights.h`, `runtime/test_images.h`).

## Architecture Snapshot

### CPU
- 4-stage pipeline: IF -> ID -> EX -> WB.
- Branch prediction (BHT/BTB), forwarding, load-use hazard handling.
- RV32M multiply/divide support.

### Accelerator (`riscv-accelerator/MatmulAccelerator.sv`)
- MMIO base: `0x80000000`.
- Computes a 4x4 output tile (`int32` accumulators) from packed INT8 vectors.
- Reads 128-bit chunks through DMA interface.
- Command FIFO decouples software launch from execution.
- Result registers: `0x18` to `0x54`.
- Added control for strided input traversal:
  - `0x58`: `X_STRIDE`
  - `0x5C`: `K_ROW_LEN`

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

From the latest documented OpenLane analysis in this workspace:

| Metric | Current observation |
|---|---|
| Logic cells | ~65,912 |
| Flip-flops | 15,570 |
| Floorplan used in experiment | 2mm x 2mm (4 mm^2 die) |
| Core utilization | ~24% |
| Timing snapshot | WNS ~ -64.62 ns (target 10 ns) |
| Estimated Fmax (from that snapshot) | ~13.4 MHz |
| PAR status | Failed at detailed routing due to congestion/memory pressure |

Interpretation:
- The design is functionally strong in RTL/simulation.
- Physical implementation is currently limited by dense distributed structures (notably predictor/register resources) and would benefit from more macro-centric memory architecture to close timing/routing.

## Reproduce The Simulation Result

1. Ensure toolchain dependencies are installed:
   - Verilator
   - `riscv64-unknown-elf-*`
   - Python 3
2. Check absolute paths in `Makefile`:
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
