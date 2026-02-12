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
// Fixed-point neural network operations
// =============================================================

static void matmul(const int8_t *W, const int8_t *x, int32_t *out, int M,
                   int N) {
  for (int i = 0; i < M; i++) {
    int32_t acc = 0;
    const int8_t *row = &W[i * N];
    int j = 0;
    for (; j <= N - 4; j += 4) {
      acc += (int32_t)row[j] * (int32_t)x[j];
      acc += (int32_t)row[j + 1] * (int32_t)x[j + 1];
      acc += (int32_t)row[j + 2] * (int32_t)x[j + 2];
      acc += (int32_t)row[j + 3] * (int32_t)x[j + 3];
    }
    for (; j < N; j++) {
      acc += (int32_t)row[j] * (int32_t)x[j];
    }
    out[i] = acc;
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

static void rescale_to_int8(int32_t *in, int8_t *out, int size) {
  int32_t max_val = 0;
  for (int i = 0; i < size; i++) {
    int32_t abs_val = in[i] < 0 ? -in[i] : in[i];
    if (abs_val > max_val)
      max_val = abs_val;
  }
  if (max_val == 0) {
    for (int i = 0; i < size; i++)
      out[i] = 0;
    return;
  }
  for (int i = 0; i < size; i++) {
    out[i] = (int8_t)((in[i] * 127) / max_val);
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
  add_bias(hidden_raw, fc1_bias, HIDDEN_SIZE);
  relu(hidden_raw, HIDDEN_SIZE);
  rescale_to_int8(hidden_raw, hidden_q, HIDDEN_SIZE);
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