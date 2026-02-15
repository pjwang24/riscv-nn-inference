# Cycle Reduction Improvement Log

Tracking optimizations applied to reduce simulation cycle count for the RISC-V ML inference workload (100 MNIST images, 2-layer INT8 MLP on a 3-stage RV32IM pipeline).

---

## 2026-02-15 — Strategy A: Software Optimizations

**Baseline:** 48,886,157 cycles  
**Result:** 44,830,519 cycles (**−8.3%**, −4,055,638 cycles)

### Changes

| ID | Optimization | File | Description |
|----|---|---|---|
| A1 | Reciprocal multiply | `inference_bare.c` | Replaced 128 per-neuron divisions in `rescale_to_int8` with 1 division + 128 multiply-shifts using `recip = (127 << 16) / max_val` |
| A2 | 8× loop unrolling | `inference_bare.c` | Doubled matmul inner loop unrolling from 4× to 8×, reducing loop overhead |
| A3 | Fused kernels | `inference_bare.c` | Combined `add_bias()` + `relu()` + `rescale_to_int8()` into single `fused_bias_relu_rescale()` — eliminates 2 redundant passes over the 128-element hidden layer |
| A4 | `-O3` flag | `Makefile` | Switched compiler optimization from `-O2` to `-O3` for more aggressive inlining/scheduling |

### Verification

```
*** PASSED *** after 44830519 simulation cycles
Total cycles: 44830519
```

All 100 MNIST predictions correct.

---

---

## 2026-02-15 — Strategy B: Hardware Optimizations (RTL)

**Baseline:** 44,830,519 cycles (post–Strategy A)  
**Result:** 44,830,278 cycles (**−241 cycles**, −0.0005%)

> **Note:** The marginal improvement is expected — this workload is dominated by the matmul inner loop (~10M MACs) which executes very few jump instructions. The branch predictor was already performing well on the tight loops. These changes are still architecturally correct and will benefit workloads with more function calls.

### Changes

| ID | Optimization | File | Description |
|----|---|---|---|
| B1 | Fix jump bubble | `Execute.sv`, `Riscv151.v` | Jumps (JAL/JALR) no longer unconditionally flush; they use BTB prediction like branches. BHT/BTB now trains on jumps too. |
| B2 | Larger BHT/BTB | `Fetch.sv` | Doubled predictor tables from 128 to 256 entries to reduce aliasing |

### Verification

```
*** PASSED *** after 44830278 simulation cycles
Total cycles: 44830278
```

All 100 MNIST predictions correct.

---

## Cumulative Results

| Stage | Cycles | Δ from Original |
|---|---|---|
| Original baseline | 48,886,157 | — |
| + Strategy A (SW) | 44,830,519 | −4,055,638 (−8.3%) |
| + Strategy B (HW) | 44,830,278 | −4,055,879 (−8.3%) |

---

<!-- Future improvement entries go here, following the same date-header format -->
