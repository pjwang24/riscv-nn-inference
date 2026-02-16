# Cycle Reduction Improvement Log

---

## 2026-02-15 — Comprehensive Optimization: SW + HW + Accelerator

**Original Baseline:** 48,886,157 cycles  
**Final Result:** 6,698,993 cycles (**−86.3%**, 7.30× overall speedup)

Workload: 100 MNIST images through a 2-layer INT8 MLP (784→128→10) on a 3-stage RV32IM pipeline with GShare branch prediction and no-cache 1-cycle memory.

---

### Strategy A: Software Optimizations

> 48,886,157 → 44,830,519 cycles (**−8.3%**)

| ID | Optimization | File | Description |
|----|---|---|---|
| A1 | Reciprocal multiply | `inference_bare.c` | Replaced 128 per-neuron divisions in `rescale_to_int8` with 1 division + 128 multiply-shifts: `recip = (127 << 16) / max_val` |
| A2 | 8× loop unrolling | `inference_bare.c` | Doubled matmul inner loop unrolling from 4× to 8×, reducing branch overhead |
| A3 | Fused kernels | `inference_bare.c` | Combined `add_bias()` + `relu()` + `rescale_to_int8()` into single `fused_bias_relu_rescale()` — eliminates 2 redundant passes over the 128-element hidden layer |
| A4 | `-O3` compiler flag | `Makefile` | Switched from `-O2` to `-O3` for more aggressive inlining and instruction scheduling |

---

### Strategy B: Pipeline & Branch Prediction Improvements (RTL)

> 44,830,519 → 44,830,278 cycles (**−241 cycles**)

| ID | Optimization | File | Description |
|----|---|---|---|
| B1 | Fix jump bubble | `Execute.sv`, `Riscv151.v` | JAL/JALR no longer unconditionally flush the pipeline; they use BTB prediction like branches. BHT/BTB now trains on jump instructions. |
| B2 | Larger BHT/BTB | `Fetch.sv` | Doubled predictor tables from 128 to 256 entries to reduce aliasing in long loops |

> **Note:** Marginal improvement is expected — this workload is dominated by the matmul inner loop (~10M MACs) with very few jump instructions. These changes are architecturally correct and benefit jump-heavy workloads.

---

### Strategy C: Word-Packed Loads — FAILED (Reverted)

Attempted to replace byte loads (`lb`) with word loads (`lw`) + shift/mask extraction to reduce memory operations by 4×. This **increased** cycles to ~69M because the software byte-extraction overhead (3 shifts + mask per byte) exceeded the hardware sign-extension cost of `lb`. The compiler's `-O3` scheduling was already optimal for byte loads on RV32IM.

---

### Strategy D: Decoupled Matmul Accelerator

> 44,830,278 → 6,698,993 cycles (**−85.1%**, 6.69× speedup)

A DMA-based, 4-lane, weight-stationary INT8 matmul accelerator that shares the dcache port with the CPU via an address-decoded mux. **The CPU core (Riscv151) is completely unmodified.**

#### Accelerator Architecture

```
┌────────────────────────────────────────────────┐
│  Matmul Accelerator  (MMIO @ 0x80000000)       │
│                                                │
│   ┌──────────┐  ┌──────────────────────────┐   │
│   │  Config  │  │       DMA Engine         │   │
│   │  Regs    │  │  IDLE → LOAD_W → COMPUTE │   │
│   │  W_ADDR  │  │       → DONE             │   │
│   │  X_ADDR  │  └──────────┬───────────────┘   │
│   │  M/N_DIM │             │                   │
│   └──────────┘     ┌───────▼───────┐           │
│                    │  Weight SRAMs │           │
│   ┌──────────┐     │ 256w × 4 lanes│           │
│   │ RESULT0  │◄────┤               │           │
│   │ RESULT1  │◄────┤  Packed INT8  │           │
│   │ RESULT2  │◄────┤  MAC Array    │           │
│   │ RESULT3  │◄────┤  (4 lanes)    │           │
│   └──────────┘     └───────────────┘           │
└────────────────────────────────────────────────┘
```

#### MMIO Register Map

| Offset | Name | R/W | Description |
|--------|------|-----|-------------|
| `0x00` | CTRL/STATUS | W/R | W: `[0]`=start `[1]`=clear — R: `[0]`=busy `[1]`=done |
| `0x04` | W_ADDR | W | Base byte address of weight matrix in data memory |
| `0x08` | X_ADDR | W | Base byte address of input vector in data memory |
| `0x0C` | M_DIM | W | Number of output rows (1–4) |
| `0x10` | N_DIM | W | Dot-product length (multiple of 4) |
| `0x14` | RESULT0 | R | Lane 0 accumulated INT32 result |
| `0x18` | RESULT1 | R | Lane 1 accumulated INT32 result |
| `0x1C` | RESULT2 | R | Lane 2 accumulated INT32 result |
| `0x20` | RESULT3 | R | Lane 3 accumulated INT32 result |

#### Operation Flow

1. CPU writes W_ADDR, X_ADDR, M_DIM, N_DIM (config registers accept writes in all states)
2. CPU writes CTRL=1 → accelerator takes over dcache port via DMA
3. DMA loads weight rows into 4 lane SRAMs (2 cycles per word: issue + capture)
4. DMA streams input vector; each word triggers 4 parallel packed INT8 dot-products
5. CPU polls STATUS until done bit is set
6. CPU reads RESULT0–RESULT3 (up to 4 output neurons per tile)

#### Files Changed

| ID | Type | File | Description |
|----|------|------|-------------|
| D1 | RTL | `MatmulAccelerator.sv` **[NEW]** | 4-lane accelerator: DMA state machine, weight SRAMs, packed INT8 MAC, MMIO decode |
| D2 | RTL | `riscv_top.v` | Dcache port mux: `addr[31]=1` → MMIO to accelerator; `accel_busy` → DMA drives addr/re |
| D3 | SW | `inference_bare.c` | `matmul()` rewritten to use accelerator: writes config, polls done, reads 4 results per tile |
| D4 | Build | `Makefile` | Added accelerator source and `-I` include path |

#### Bugs Found During Verification

1. **Combinational loop** — `mmio_rdata` depended on current-cycle `cpu_dcache_addr` through the CPU writeback path, causing Verilator convergence failure. Fixed by registering the read address select (`reg_offset_d`).

2. **Config register write gating** — W_ADDR/X_ADDR/M_DIM/N_DIM were only accepted in S_IDLE, but the CPU writes them while the accelerator is in S_DONE between tiles. Subsequent tiles used stale addresses, causing 13/100 wrong predictions. Fixed by separating config writes into their own `always` block.

---

### Verification

```
*** PASSED *** after 6698993 simulation cycles
Total cycles: 6698993
```

All 100 MNIST predictions correct (97% model accuracy maintained).

---

### Cumulative Results

| Stage | Cycles | Δ Cycles | Δ % | Speedup |
|-------|--------|----------|-----|---------|
| Original baseline | 48,886,157 | — | — | 1.00× |
| + Strategy A (SW optimizations) | 44,830,519 | −4,055,638 | −8.3% | 1.09× |
| + Strategy B (pipeline/predictor) | 44,830,278 | −4,055,879 | −8.3% | 1.09× |
| + Strategy D (matmul accelerator) | 6,698,993 | −38,131,285 | −85.1% | 6.69× |

---

### Strategy E: Separate DMA & Batched Inference

> 6,698,993 → 2,914,655 cycles (**−56.5%**, 2.30× speedup vs Accelerator)

Optimized memory access by widening the accelerator's DMA path and amortizing weight loading costs.

#### Key Changes
1.  **Separate 128-bit DMA Path**: Accelerator now has a dedicated read-only port to memory, bypassing the CPU's 32-bit interface.
2.  **Batched Inference (Batch=4)**: Processes 4 input images against the same loaded weights. Weights are loaded once (128-bit wide) and reused 4 times, reducing effective weight memory bandwidth by 75%.
3.  **128-bit Data Consumption**: Input vectors are consumed in 128-bit chunks (4 words/cycle bandwidth).

### Cumulative Results

| Stage | Cycles | Δ Cycles | Δ % | Speedup |
|-------|--------|----------|-----|---------|
| Original baseline | 48,886,157 | — | — | 1.00× |
| ... | ... | ... | ... | ... |
| + Strategy D (matmul accelerator) | 6,698,993 | −42,187,164 | −86.3% | 7.30× |
| + Strategy E (DMA + Batching) | **2,914,655** | **−45,971,502** | **−94.0%** | **16.77×** |

<!-- Future improvement entries go here, following the same date-header format -->
