// inference_bare.c — Bare-metal fixed-point MLP inference
// Updated for Batched Inference and Interleaved DMA Accelerator

#include "test_images.h"
#include "weights.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define csr_tohost(val)                                                        \
  {                                                                            \
    asm volatile("csrw 0x51e, %[v]" ::[v] "r"(val));                           \
  }

void *memset(void *s, int c, size_t n) {
  unsigned char *p = (unsigned char *)s;
  while (n--)
    *p++ = (unsigned char)c;
  return s;
}

void *memcpy(void *dest, const void *src, size_t n) {
  unsigned char *d = (unsigned char *)dest;
  const unsigned char *s = (const unsigned char *)src;
  while (n--)
    *d++ = *s++;
  return dest;
}

// =============================================================
// Matmul Accelerator MMIO
// =============================================================
#define ACCEL_BASE 0x80000000
#define ACCEL_CTRL (*(volatile uint32_t *)(ACCEL_BASE + 0x00))
#define ACCEL_STATUS (*(volatile uint32_t *)(ACCEL_BASE + 0x00))
#define ACCEL_W_ADDR (*(volatile uint32_t *)(ACCEL_BASE + 0x04))
#define ACCEL_X_ADDR (*(volatile uint32_t *)(ACCEL_BASE + 0x08))
#define ACCEL_M_DIM (*(volatile uint32_t *)(ACCEL_BASE + 0x0C))
#define ACCEL_N_DIM (*(volatile uint32_t *)(ACCEL_BASE + 0x10))
#define ACCEL_RESULT0 (*(volatile int32_t *)(ACCEL_BASE + 0x14))
#define ACCEL_RESULT1 (*(volatile int32_t *)(ACCEL_BASE + 0x18))
#define ACCEL_RESULT2 (*(volatile int32_t *)(ACCEL_BASE + 0x1C))
#define ACCEL_RESULT3 (*(volatile int32_t *)(ACCEL_BASE + 0x20))

// 4-lane accelerator requires interleaved weights:
// [Row0_W0, Row1_W0, Row2_W0, Row3_W0, Row0_W1...]
// We copy weights to RAM and interleave them.
// FC1: 128x784 (exact multiple of 4 rows)
// FC2: 10x128 (pad to 12 rows)
int8_t __attribute__((aligned(16))) fc1_weights_hw[128 * 784];
int8_t
    __attribute__((aligned(16))) fc2_weights_hw[12 * 128]; // Padded to 12 rows

static void interleave_weights(const int8_t *src, int8_t *dst, int rows,
                               int cols) {
  // Process in blocks of 4 rows (Lanes)
  for (int r = 0; r < rows; r += 4) {
    int rows_in_block = (rows - r >= 4) ? 4 : (rows - r);
    // Process columns in blocks of 4 (one 32-bit word per lane)
    for (int c = 0; c < cols; c += 4) {
      // For each lane (row), output 4 sequential column values
      for (int i = 0; i < 4; i++) {   // Lane 0..3
        for (int j = 0; j < 4; j++) { // Column c..c+3
          if (i < rows_in_block && (c + j) < cols) {
            *dst++ = src[(r + i) * cols + (c + j)];
          } else {
            *dst++ = 0; // Padding
          }
        }
      }
    }
  }
}

// Driver: Compute matmul for a BATCH of inputs against SAME weight tile
// skip_load=false: Load weights, then compute 1st input
// skip_load=true: Reuse weights, compute input
static void
matmul_tile_batch(const int8_t *W_tile_addr,
                  const int8_t **inputs, // Array of pointers to inputs
                  int32_t **outputs,     // Array of pointers to outputs
                  int start_row,         // Current output row offset
                  int num_rows,          // Rows in this tile (1-4)
                  int N, int batch_size) {
  // 1. Load Weights + Compute Batch[0]
  ACCEL_W_ADDR = (uint32_t)W_tile_addr; // Already interleaved
  ACCEL_X_ADDR = (uint32_t)inputs[0];
  ACCEL_M_DIM = num_rows;
  ACCEL_N_DIM = N;
  ACCEL_CTRL = 1; // Start (Load + Compute)

  while (!(ACCEL_STATUS & 0x2))
    ; // Wait Done

  outputs[0][start_row] = ACCEL_RESULT0;
  if (num_rows > 1)
    outputs[0][start_row + 1] = ACCEL_RESULT1;
  if (num_rows > 2)
    outputs[0][start_row + 2] = ACCEL_RESULT2;
  if (num_rows > 3)
    outputs[0][start_row + 3] = ACCEL_RESULT3;

  // 2. Compute remaining batches (Skip Load)
  for (int b = 1; b < batch_size; b++) {
    ACCEL_X_ADDR = (uint32_t)inputs[b];
    // Keep W_ADDR, M, N same.
    // Set Bit 0 (Start) AND Bit 2 (Skip Load) -> 0x5
    ACCEL_CTRL = 5;

    while (!(ACCEL_STATUS & 0x2))
      ;

    outputs[b][start_row] = ACCEL_RESULT0;
    if (num_rows > 1)
      outputs[b][start_row + 1] = ACCEL_RESULT1;
    if (num_rows > 2)
      outputs[b][start_row + 2] = ACCEL_RESULT2;
    if (num_rows > 3)
      outputs[b][start_row + 3] = ACCEL_RESULT3;
  }
}

// Full Layer Batch Matmul
static void layer_matmul_batch(const int8_t *W_interleaved,
                               const int8_t **inputs, int32_t **outputs, int M,
                               int N, int batch_size) {
  // Iterate over output rows in blocks of 4
  // Since W is interleaved, W pointer advances by 4*N bytes per tile
  const int8_t *w_ptr = W_interleaved;

  for (int i = 0; i < M; i += 4) {
    int rows_this_tile = (M - i >= 4) ? 4 : (M - i);
    matmul_tile_batch(w_ptr, inputs, outputs, i, rows_this_tile, N, batch_size);

    w_ptr += 4 * N; // Advance to next interleaved block
  }
}

// Division helper for quantization scaling without relying on DIV instruction.
static int32_t soft_div(int32_t numer, int32_t denom) {
  if (denom == 0) {
    return 0;
  }

  const bool neg = ((numer < 0) ^ (denom < 0));
  uint64_t a = (numer < 0) ? (uint64_t)(-(int64_t)numer) : (uint64_t)numer;
  const uint64_t b =
      (denom < 0) ? (uint64_t)(-(int64_t)denom) : (uint64_t)denom;

  uint32_t q = 0;
  for (int i = 31; i >= 0; i--) {
    const uint64_t shifted = b << i;
    if (shifted <= a) {
      a -= shifted;
      q |= (1u << i);
    }
  }

  return neg ? -(int32_t)q : (int32_t)q;
}

static void fused_bias_relu_rescale(int32_t *raw, const int32_t *bias,
                                    int8_t *out, int size) {
  int32_t max_val = 0;
  for (int i = 0; i < size; i++) {
    int32_t v = raw[i] + bias[i];
    v = v & ~(v >> 31);
    raw[i] = v;
    if (v > max_val)
      max_val = v;
  }
  if (max_val == 0) {
    for (int i = 0; i < size; i++)
      out[i] = 0;
    return;
  }
  // Replace hardware division with soft_div
  int32_t recip = soft_div((127 << 16), max_val);
  for (int i = 0; i < size; i++) {
    out[i] = (int8_t)((raw[i] * recip) >> 16);
  }
}

static void add_bias(int32_t *out, const int32_t *bias, int size) {
  for (int i = 0; i < size; i++)
    out[i] += bias[i];
}

static int argmax(int32_t *x, int size) {
  int max_idx = 0;
  int32_t max_val = x[0];
  for (int i = 1; i < size; i++) {
    if (x[i] > max_val) {
      max_val = x[i];
      max_idx = i;
    }
  }
  return max_idx;
}

// Buffers for Batch=4
#define BATCH_SIZE 4
int32_t l1_raw[BATCH_SIZE][HIDDEN_SIZE];
int8_t l1_q[BATCH_SIZE][HIDDEN_SIZE];
int32_t l2_raw[BATCH_SIZE][OUTPUT_SIZE];

// Aligned input buffer for batch
int8_t __attribute__((aligned(16))) batch_inputs_aligned[BATCH_SIZE][784];

int main(void) {
  // 1. Prepare Weights
  interleave_weights(fc1_weight, fc1_weights_hw, 128, 784);
  interleave_weights(fc2_weight, fc2_weights_hw, 10, 128);

  int correct = 0;
  for (int i = 0; i < NUM_TEST_IMAGES; i += BATCH_SIZE) {
    int batch = (NUM_TEST_IMAGES - i >= BATCH_SIZE) ? BATCH_SIZE
                                                    : (NUM_TEST_IMAGES - i);

    // Pointers for batch
    const int8_t *batch_inputs[BATCH_SIZE];
    int32_t *batch_l1_raw[BATCH_SIZE];
    const int8_t *batch_l1_q[BATCH_SIZE];
    int32_t *batch_l2_raw[BATCH_SIZE];

    for (int b = 0; b < batch; b++) {
      // Copy to aligned buffer
      memcpy(batch_inputs_aligned[b], test_images[i + b], 784);
      batch_inputs[b] = batch_inputs_aligned[b];

      batch_l1_raw[b] = l1_raw[b];
      batch_l1_q[b] = l1_q[b];
      batch_l2_raw[b] = l2_raw[b];
    }

    // Layer 1: FC1
    layer_matmul_batch(fc1_weights_hw, batch_inputs, batch_l1_raw, HIDDEN_SIZE,
                       INPUT_SIZE, batch);

    // Activation
    for (int b = 0; b < batch; b++) {
      fused_bias_relu_rescale(l1_raw[b], fc1_bias, l1_q[b], HIDDEN_SIZE);
    }

    // Layer 2: FC2
    // Note: l1_q is array of int8_t, but layer_matmul expects array of
    // pointers. We set up batch_l1_q above.
    layer_matmul_batch(fc2_weights_hw, batch_l1_q, batch_l2_raw, OUTPUT_SIZE,
                       HIDDEN_SIZE, batch);

    // Bias + Argmax
    for (int b = 0; b < batch; b++) {
      add_bias(l2_raw[b], fc2_bias, OUTPUT_SIZE);
      int pred = argmax(l2_raw[b], OUTPUT_SIZE);
      if (pred == expected_labels[i + b]) {
        correct++;
      }
    }
  }

  int num_wrong = NUM_TEST_IMAGES - correct;
  if (num_wrong == 0) {
    csr_tohost(1);
  } else {
    csr_tohost(num_wrong + 1);
  }

  for (;;)
    asm volatile("nop");

  return 0;
}
