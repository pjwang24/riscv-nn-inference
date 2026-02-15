# Bare-Metal Neural Network Inference on a Custom RISC-V Processor

> End-to-end ML inference on a 3-stage pipelined RV32IM core with a custom matmul accelerator, verified via cycle-accurate Verilator simulation.

## Key Results

| Metric | Value |
|--------|-------|
| ISA | RV32IM (Integer + Multiply/Divide) |
| Accelerator | 4-lane DMA-based INT8 matmul engine |
| Simulation cycles | **6,698,993** (~6.7M) |
| Speedup vs. unoptimized baseline | **7.30×** (from 48.9M) |
| Test images | 100 (MNIST handwritten digits) |
| Inference result | **PASSED** (all predictions correct, 97% accuracy) |
| Binary size | ~178 KB (code + weights + test data) |

---

## How It Works: From Training to Silicon

This section explains how a neural network trained on a laptop ends up running on a custom CPU — something that looks like magic until you see the pipeline.

### The Big Picture

```
 ┌─────────────┐     ┌──────────────┐     ┌────────────────┐     ┌─────────────────┐
 │  1. TRAIN   │────▶│  2. EXPORT   │────▶│  3. COMPILE    │────▶│  4. SIMULATE    │
 │  (PyTorch)  │     │  (Python→C)  │     │  (RISC-V GCC)  │     │  (Verilator)    │
 │             │     │              │     │                │     │                 │
 │ float32     │     │ int8 arrays  │     │ flat binary    │     │ clock-by-clock  │
 │ model.pth   │     │ weights.h    │     │ inference.hex  │     │ CPU execution   │
 └─────────────┘     └──────────────┘     └────────────────┘     └─────────────────┘
```

### Step 1: Train the Model (Python / PyTorch)

A simple 2-layer MLP is trained on the MNIST handwritten digit dataset using standard PyTorch:

```
Input (784 pixels) → FC1 (128 neurons) → ReLU → FC2 (10 classes) → Argmax → digit 0–9
```

At this point, the model's "knowledge" lives in its **weight matrices** — two big tables of floating-point numbers (e.g., `fc1.weight` is a 128×784 matrix of `float32` values). These weights encode the patterns the network learned during training.

### Step 2: Quantize and Export (Python → C Header Files)

Our RISC-V CPU has **no floating-point hardware** — it only understands integers. So we need to convert the model:

1. **Quantize**: Each float32 weight is scaled and rounded to `int8` (−128 to +127). This loses a tiny bit of precision but reduces storage by 4× and enables fast integer math.

2. **Export as C arrays**: The quantized weights are written directly into C header files as constant arrays:

   ```c
   // weights.h (auto-generated)
   const int8_t fc1_weight[100352] = { -5, 0, -3, -2, 3, -1, 1, ... };
   const int32_t fc1_bias[128] = { -170, 308, 38, ... };
   ```

   Similarly, 100 test images and their expected labels are exported to `test_images.h`.

> **Key insight:** The weights aren't "loaded" at runtime from a file system — there is no file system. They are **baked directly into the program binary** as constant data, just like hardcoded lookup tables.

### Step 3: Cross-Compile (RISC-V GCC → Hex File)

The C inference code (`inference_bare.c`) `#include`s the weight headers. When compiled, the weights become part of the program's `.rodata` section:

```
inference_bare.c  ─┐
weights.h          ├──▶ riscv-gcc ──▶ inference.elf ──▶ inference.bin ──▶ inference.hex
test_images.h      │         ▲
start.s           ─┘         │
                       linker script (base address 0x2000)
```

The linker script places everything — code, weights, and test images — into a single contiguous memory region starting at address `0x2000`. The final `.hex` file is a flat dump of this memory image, formatted as 128-bit hex lines.

**What's in the binary:**
| Section | Content | Size |
|---------|---------|------|
| `.text` | Inference code (matmul, ReLU, argmax, etc.) | ~780 B |
| `.rodata` | FC1 weights (128×784) | ~98 KB |
| `.rodata` | FC2 weights (10×128), biases | ~2 KB |
| `.rodata` | 100 test images (100×784) | ~77 KB |

### Step 4: Simulate (Verilator)

The Verilator testbench (`sim_main.cpp`) acts like a "bootloader" — it writes the hex file directly into the processor's simulated SRAM:

```cpp
// Pseudocode from sim_main.cpp
for each line in inference.hex:
    memory[addr] = parse_128bit_hex(line);  // icache + dcache
    addr++;
```

After loading, the CPU boots from address `0x2000` and begins executing the inference code. From the CPU's perspective, the weights are just bytes sitting in memory at known addresses — the same way an embedded system would have firmware data burned into ROM.

### Why This Works

There's no operating system, no file I/O, no dynamic memory allocation. The entire "deployment" is:

1. Trained model weights → constant C arrays → compiled into the binary → loaded into RAM
2. CPU reads weights from RAM addresses, multiplies them with input pixels, and produces predictions
3. The matmul accelerator reads the same weight data via DMA from the same RAM

The key realization: **a neural network is just arithmetic on arrays.** Once you have the weights in memory and code that knows the array dimensions, you can run inference on anything — from a datacenter GPU to a bare-metal RISC-V core.

---

## Project Overview

This project demonstrates a quantized two-layer neural network (MLP) running entirely as bare-metal C on a custom RISC-V processor with a purpose-built matmul accelerator. The pipeline spans:

1. **Model Training** — PyTorch MLP trained on MNIST, exported with INT8 weight quantization
2. **Bare-Metal C** — Fixed-point inference code compiled with `-nostdlib` for the RV32IM target
3. **HW Accelerator** — DMA-based 4-lane INT8 matmul engine with MMIO interface
4. **RTL Design** — 3-stage pipelined CPU with hardware multiply, branch prediction, and CSR support
5. **Verilator Simulation** — Cycle-accurate verification of the full hardware–software stack

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

- **256-entry BHT** with 2-bit saturating counters
- **256-entry BTB** for target address prediction
- Trains on both branches and jumps (JAL/JALR)
- Mispredict correction from the execute stage

### M-Extension (Hardware Multiply/Divide)

Single-cycle combinational unit supporting all 8 RV32M instructions:

| Instruction | Operation |
|-------------|-----------|
| `MUL` | Lower 32 bits of signed × signed |
| `MULH` / `MULHSU` / `MULHU` | Upper 32 bits (signed, mixed, unsigned) |
| `DIV` / `DIVU` | Signed / unsigned division |
| `REM` / `REMU` | Signed / unsigned remainder |

### Matmul Accelerator

A DMA-based, 4-lane, weight-stationary INT8 matmul engine mapped at `0x80000000`:

- **4 MAC lanes** with 256-word weight SRAMs — process 4 output neurons in parallel
- **Packed INT8 dot-product** — 4 byte multiplies per clock per lane (16 MACs/cycle)
- **DMA engine** — reads weights and inputs directly from data memory
- **CPU integration** — shares dcache port via address-decoded mux; CPU core is unmodified

The CPU writes weight/input addresses and dimensions to MMIO registers, triggers the accelerator, polls for completion, and reads the 4 accumulated INT32 results.

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
- Matmul offloaded to hardware accelerator via MMIO

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
Loaded 11398 lines from inference.hex
*** PASSED *** after 6698993 simulation cycles
Total cycles: 6698993
```

---

## Performance Analysis

### Optimization History

| Stage | Cycles | Speedup |
|-------|--------|---------|
| Original baseline (unoptimized) | 48,886,157 | 1.00× |
| + Software optimizations (reciprocal multiply, loop unrolling, fused kernels, `-O3`) | 44,830,519 | 1.09× |
| + Pipeline improvements (jump bubble fix, larger BHT/BTB) | 44,830,278 | 1.09× |
| + **Matmul accelerator** | **6,698,993** | **7.30×** |

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for detailed change-by-change analysis.

### Cycle Breakdown

At ~6.7M total cycles for 100 images (~67K cycles/image):

| Metric | Value |
|--------|-------|
| MACs per image | ~101,632 (FC1: 100,352 + FC2: 1,280) |
| Total MACs | ~10.16M |
| Effective throughput | **~0.66 cycles/MAC** |

---

## Project Structure

```
verilator/
├── Makefile              # Build orchestration
├── sim_main.cpp          # Verilator testbench (hex loading, reset, CSR monitoring)
├── inference_bare.c      # Bare-metal MLP inference (MMIO accelerator calls)
├── inference.ld          # Linker script (base address 0x2000)
├── bin2hex.py            # Binary-to-hex converter (128-bit width)
├── IMPROVEMENTS.md       # Optimization log with cycle measurements
├── README.md
└── riscv-accelerator/
    └── MatmulAccelerator.sv  # 4-lane DMA matmul accelerator

asic-project-fa25-golden-gates/src/
├── riscv_top.v           # Top-level module (CPU + memory + accelerator mux)
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

riscv-ml-inference/
├── train/
│   └── train_and_export.py  # PyTorch training + INT8 quantization + C export
└── runtime/
    ├── weights.h             # Quantized INT8 weights and INT32 biases
    └── test_images.h         # 100 MNIST test images + expected labels
```

---

## Significance

### RISC-V for Edge ML

This project shows that a **minimal RISC-V core** with a purpose-built accelerator can run real neural network inference at high throughput. The open ISA enables:
- **Custom accelerators** — the matmul engine provides a 7.3× speedup without modifying the CPU core
- **Area optimization** — remove unused features, add domain-specific hardware
- **Full-stack verification** — processor + accelerator + memory + software in one simulation

### Co-Design in Action

The optimization journey (48.9M → 6.7M cycles) demonstrates the value of hardware-software co-design:
- Software optimizations alone yielded only 8% improvement
- Pipeline changes contributed negligibly for this workload
- The accelerator — designed specifically for the bottleneck — delivered an **85% reduction**

---

## Future Work

### Accelerator
- **Pipelined DMA** — overlap weight loading with computation (1 cycle/word instead of 2)
- **Wider MAC array** — 8 or 16 lanes for larger tile sizes
- **Double-buffered SRAMs** — load next tile's weights while computing current tile
- **Systolic array** — 2D dataflow for higher utilization

### Hardware
- **Performance counters** — cycle/instruction/misprediction CSRs for profiling
- **Cache mode benchmarking** — measure real miss rates for this workload

### Software
- **Tiled FC2** — handle non-multiple-of-4 output dimensions more efficiently
- **Activation pipelining** — overlap bias/ReLU/rescale with next tile's matmul

### Model
- **CNNs** — 2D convolution via im2col + matmul on the accelerator
- **CIFAR-10** — more complex image classification
- **Mixed precision** — INT16 first layer, INT8 deeper layers

---

## Acknowledgments

Special thanks to **Tingyao Huang** for his contributions to the design and development of the RISC-V CPU core used in this project.


