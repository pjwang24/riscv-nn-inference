// inference_bare.c — Bare-metal fixed-point MLP inference for EECS 151 RISC-V
// core No OS, no putchar, no syscalls. Results reported via CSR tohost.
//
// CSR 0x51e (tohost):
//   Write 1 = PASSED (all predictions correct)
//   Write >1 = FAILED (value = number of wrong predictions + 1)

#include "test_images.h"
#include "weights.h"
#include <stdint.h>

// Write to tohost CSR to signal pass/fail to testbench
#define csr_tohost(val)                                                        \
  {                                                                            \
    asm volatile("csrw 0x51e, %[v]" ::[v] "r"(val));                           \
  }

#include <stddef.h>
void *memset(void *s, int c, size_t n) {
  unsigned char *p = (unsigned char *)s;
  while (n--) {
    *p++ = (unsigned char)c;
  }
  return s;
}

// =============================================================
// Matmul Accelerator MMIO interface
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

static void matmul(const int8_t *W, const int8_t *x, int32_t *out, int M,
                   int N) {
  // Process 4 output neurons at a time using the accelerator
  for (int i = 0; i < M; i += 4) {
    int rows_this_tile = (M - i >= 4) ? 4 : (M - i);

    ACCEL_W_ADDR = (uint32_t)&W[i * N];
    ACCEL_X_ADDR = (uint32_t)x;
    ACCEL_M_DIM = rows_this_tile;
    ACCEL_N_DIM = N;
    ACCEL_CTRL = 1; // Start

    // Poll until done (bit 1 of STATUS)
    while (!(ACCEL_STATUS & 0x2))
      ;

    out[i] = ACCEL_RESULT0;
    if (rows_this_tile > 1)
      out[i + 1] = ACCEL_RESULT1;
    if (rows_this_tile > 2)
      out[i + 2] = ACCEL_RESULT2;
    if (rows_this_tile > 3)
      out[i + 3] = ACCEL_RESULT3;
  }
}

static void add_bias(int32_t *out, const int32_t *bias, int size) {
  for (int i = 0; i < size; i++) {
    out[i] += bias[i];
  }
}

static void relu(int32_t *x, int size) {
  for (int i = 0; i < size; i++) {
    x[i] = x[i] & ~(x[i] >> 31);
  }
}

// A3: Fused bias + ReLU + rescale kernel (replaces separate add_bias, relu,
// rescale_to_int8 for the hidden layer). Two passes: one to compute
// bias+ReLU+find-max, one to scale using reciprocal multiplication (A1).
static void fused_bias_relu_rescale(int32_t *raw, const int32_t *bias,
                                    int8_t *out, int size) {
  int32_t max_val = 0;
  for (int i = 0; i < size; i++) {
    int32_t v = raw[i] + bias[i]; // bias
    v = v & ~(v >> 31);           // ReLU (branchless)
    raw[i] = v;                   // store back for scaling pass
    if (v > max_val)
      max_val = v;
  }
  if (max_val == 0) {
    for (int i = 0; i < size; i++)
      out[i] = 0;
    return;
  }
  // A1: one division to get reciprocal, then multiply+shift
  int32_t recip = (127 << 16) / max_val;
  for (int i = 0; i < size; i++) {
    out[i] = (int8_t)((raw[i] * recip) >> 16);
  }
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

// =============================================================
// Full forward pass
// =============================================================
static int predict(const int8_t *image) {
  int32_t hidden_raw[HIDDEN_SIZE];
  int8_t hidden_q[HIDDEN_SIZE];
  int32_t output[OUTPUT_SIZE];

  matmul(fc1_weight, image, hidden_raw, HIDDEN_SIZE, INPUT_SIZE);
  // A3: fused bias + ReLU + rescale (replaces 3 separate calls)
  fused_bias_relu_rescale(hidden_raw, fc1_bias, hidden_q, HIDDEN_SIZE);
  matmul(fc2_weight, hidden_q, output, OUTPUT_SIZE, HIDDEN_SIZE);
  add_bias(output, fc2_bias, OUTPUT_SIZE);

  return argmax(output, OUTPUT_SIZE);
}

// =============================================================
// Main — run inference on test images, report via CSR
// =============================================================
void main() {
  int correct = 0;

  for (int i = 0; i < NUM_TEST_IMAGES; i++) {
    int prediction = predict(test_images[i]);
    if (prediction == expected_labels[i]) {
      correct++;
    }
  }

  // Report results via CSR tohost
  // 1 = all correct (PASSED)
  // >1 = number wrong + 1 (FAILED, but we can decode the count)
  int num_wrong = NUM_TEST_IMAGES - correct;
  if (num_wrong == 0) {
    csr_tohost(1); // PASSED - all correct
  } else {
    csr_tohost(num_wrong + 1); // FAILED - tohost tells us how many wrong
  }

  // Spin forever (testbench will catch the CSR write and stop)
  for (;;) {
    asm volatile("nop");
  }
}