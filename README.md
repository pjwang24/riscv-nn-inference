# Bare-Metal Neural Network Inference on a Custom RISC-V Processor

> End-to-end ML inference on a 3-stage pipelined RV32IM core, verified via cycle-accurate Verilator simulation.

## Key Results

| Metric | Value |
|--------|-------|
| ISA | RV32IM (Integer + Multiply/Divide) |
| Simulation cycles | **48,886,157** (~48.9M) |
| Test images | 100 (MNIST digits) |
| Inference result | **PASSED** (all predictions correct) |
| Binary size | ~178 KB (780 B code + 177 KB weights/data) |

---

## Project Overview

This project demonstrates a quantized two-layer neural network (MLP) running entirely as bare-metal C on a custom RISC-V processor. The pipeline spans:

1. **Model Training** — PyTorch MLP trained on MNIST, exported with INT8 weight quantization
2. **Bare-Metal C** — Fixed-point inference code compiled with `-nostdlib` for the RV32IM target
3. **RTL Design** — 3-stage pipelined CPU with hardware multiply, branch prediction, and CSR support
4. **Verilator Simulation** — Cycle-accurate verification of the full hardware–software stack

---

## Processor Architecture

### 3-Stage Pipeline

```
┌──────────┐    ┌──────────────────┐    ┌───────────────────┐
│  Fetch   │───▶│  Decode/Execute  │───▶│  Memory/Writeback │
│  (F)     │    │  (D/X)           │    │  (M/W)            │
└──────────┘    └──────────────────┘    └───────────────────┘
  • PC gen        • ALU / Multiplier     • Data cache R/W
  • I-cache read  • Branch resolution    • Reg writeback
  • BHT + BTB     • Forwarding           • WB mux (ALU/Mem/PC+4)
```

### Branch Prediction

- **128-entry BHT** with 2-bit saturating counters
- **128-entry BTB** for target address prediction
- Mispredict correction from the execute stage

### M-Extension (Hardware Multiply/Divide)

Single-cycle combinational unit supporting all 8 RV32M instructions:

| Instruction | Operation |
|-------------|-----------|
| `MUL` | Lower 32 bits of signed × signed |
| `MULH` / `MULHSU` / `MULHU` | Upper 32 bits (signed, mixed, unsigned) |
| `DIV` / `DIVU` | Signed / unsigned division |
| `REM` / `REMU` | Signed / unsigned remainder |

The M-extension is critical for this workload — matrix multiplication dominates inference time. Without hardware multiply, each `mul` would require multi-cycle software emulation.

### Data Forwarding

A `forward_logic` module bypasses RAW hazards between execute and writeback stages, eliminating most pipeline stalls.

### CSR Support

The `tohost` register (CSR `0x51E`) communicates results to the testbench:
- **Write 1** → PASSED
- **Write >1** → FAILED (value encodes error count)

---

## Memory Subsystem

### No-Cache Mode (used in this project)

| Parameter | Value |
|-----------|-------|
| Data width | 128 bits (4 × 32-bit words) |
| Depth | 2M entries |
| Capacity | 32 MB per instance |
| Access latency | **1 cycle** (guaranteed) |
| Instances | 2 (icache + dcache) |

Separate instruction and data memories with no stalls. This provides a performance **upper bound** by eliminating all memory latency.

### Cache Mode (alternative configuration)

The RTL also supports a cache-based hierarchy with an arbiter and external memory model. Cache misses incur multi-cycle penalties. This mode is more realistic for ASIC targets but was not used here.

### Why No-Cache Matters for ML

Neural network weights (~177 KB) are accessed repeatedly during matmul. The 128×784 FC1 weight matrix alone (~98 KB) would likely exceed typical L1 cache sizes, causing frequent capacity misses. No-cache mode isolates the CPU pipeline's throughput from memory effects.

---

## Neural Network

### Architecture

```
Input (784) ──▶ FC1 (128) ──▶ ReLU ──▶ Rescale ──▶ FC2 (10) ──▶ Argmax ──▶ digit 0–9
                 INT8×INT8     INT32     INT32→INT8   INT8×INT8
```

| Layer | Shape | Parameters | Storage |
|-------|-------|------------|---------|
| FC1 weight | 128 × 784 | 100,352 | ~98 KB (INT8) |
| FC1 bias | 128 | 128 | 512 B (INT32) |
| FC2 weight | 10 × 128 | 1,280 | ~1.3 KB (INT8) |
| FC2 bias | 10 | 10 | 40 B (INT32) |
| Test images | 100 × 784 | 78,400 | ~76.5 KB (INT8) |

### Quantization

- All weights and activations are **INT8**
- Multiply-accumulate produces **INT32** intermediates (no overflow)
- Dynamic rescaling between layers using `max_val`-based normalization
- No floating-point hardware required

### Bare-Metal Implementation

- Compiled with `-nostdlib -nostartfiles` (no OS, no libc)
- Custom `memset` provided in-source
- Results reported via inline assembly CSR writes
- Inner matmul loop unrolled 4× for ILP

---

## Build & Run

### Prerequisites

- [Verilator](https://verilator.org) ≥ 5.0
- [RISC-V GNU Toolchain](https://github.com/riscv-collab/riscv-gnu-toolchain) (`riscv64-unknown-elf-gcc`)
- Python 3 (for `bin2hex.py`)

### Quick Start

```bash
# Build everything and run simulation
make clean && make

# Or run steps individually:
make inference.hex        # Cross-compile C → ELF → binary → hex
make obj_dir/Vriscv_top   # Build Verilator simulation
make run                  # Execute simulation
```

### Expected Output

```
Loaded 11386 lines from inference.hex
*** PASSED *** after 48886157 simulation cycles
Total cycles: 48886157
```

---

## Performance Analysis

At ~48.9M total cycles for 100 images:

| Metric | Value |
|--------|-------|
| MACs per image | ~101,632 (FC1: 100,352 + FC2: 1,280) |
| Total MACs | ~10.16M |
| Effective throughput | **~4.8 cycles/MAC** |
| `mul` instructions in binary | 14 |
| `div` instructions in binary | 1 |

The ~4.8 cycles/MAC includes all overhead: loop control, data loads/stores, bias addition, ReLU, rescaling, and argmax.

---

## Project Structure

```
verilator/
├── Makefile              # Build orchestration
├── sim_main.cpp          # Verilator testbench (hex loading, reset, CSR monitoring)
├── inference_bare.c      # Bare-metal MLP inference
├── inference.ld          # Linker script (base address 0x2000)
├── bin2hex.py            # Binary-to-hex converter (128-bit width)
└── README.md

asic-project-fa25-golden-gates/src/
├── riscv_top.v           # Top-level module
├── Riscv151.v            # CPU core (pipeline integration)
├── Fetch.sv              # Fetch stage + branch prediction
├── Execute.sv            # Execute stage (ALU, multiplier, forwarding)
├── Writeback.sv          # Writeback stage
├── Control_logic.sv      # Instruction decoder
├── Memory151.v           # Memory subsystem (cache/no-cache mux)
├── no_cache_mem.v        # Direct-mapped SRAM memory
├── Multiplier.sv         # RV32M multiply/divide unit
├── ALU.v / ALUdec.v      # ALU and ALU decoder
├── RegFile.sv            # Register file
├── BranchPrediction.v    # BHT + BTB
├── Branch_Comp.v         # Branch comparator
└── const.vh / Opcode.vh  # Constants and opcode definitions

riscv-ml-inference/runtime/
├── weights.h             # Quantized INT8 weights and INT32 biases
└── test_images.h         # 100 MNIST test images + expected labels
```

---

## Significance

### RISC-V for Edge ML

This project shows that a **minimal RISC-V core** — no FPU, no OS, no ML accelerator — can run real neural network inference. The open ISA enables:
- **Custom extensions** — MAC instructions, SIMD, or tightly-coupled accelerators
- **Area optimization** — remove unused features, add domain-specific hardware
- **Full-stack verification** — processor + memory + software in one simulation

### Memory Architecture as a Design Variable

The no-cache vs. cache configuration directly impacts ML performance. No-cache mode provides deterministic single-cycle access but requires large on-chip SRAM. Cache mode is area-efficient but introduces miss penalties that disproportionately affect weight-dominated workloads.

---

## Future Work

### Hardware
- **Fused MAC instruction** — `mul` + `add` in one cycle (~2× matmul throughput)
- **INT8 SIMD** — 4 parallel INT8 multiplies per 32-bit register (~4× throughput)
- **Systolic array** — small matrix engine for 16–64× dense matmul acceleration
- **Performance counters** — cycle/instruction/misprediction/miss CSRs

### Memory
- **Cache mode benchmarking** — measure real miss rates for this workload
- **Scratchpad + DMA** — software-managed tiling with overlapped data movement

### Software
- **Weight tiling** — cache-friendly 16×16 block matmul
- **Fused kernels** — merge ReLU/rescaling into matmul output loop
- **Deeper unrolling** — 8× or 16× with register blocking

### Model
- **CNNs** — 2D convolution via im2col + matmul
- **CIFAR-10** — more complex image classification
- **Mixed precision** — INT16 first layer, INT8 deeper layers
