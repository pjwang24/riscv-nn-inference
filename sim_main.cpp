#include "Vriscv_top.h"
#include "Vriscv_top___024root.h"
#include "verilated.h"
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Clock period doesn't matter for Verilator (we manually toggle)
// We just count cycles.

uint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

static bool parse_hex_line(const char *line, uint32_t words[4]) {
  const size_t len = strlen(line);
  if (len == 0 || len > 32) {
    return false;
  }

  for (int i = 0; i < 4; i++) {
    words[i] = 0;
  }

  // Input is big-endian hex text. We build 4 little-endian 32-bit words.
  for (int i = 0; i < static_cast<int>(len); i++) {
    const char c = line[len - 1 - i];
    if (!std::isxdigit(static_cast<unsigned char>(c))) {
      return false;
    }

    int nibble = 0;
    if (c >= '0' && c <= '9') {
      nibble = c - '0';
    } else if (c >= 'a' && c <= 'f') {
      nibble = c - 'a' + 10;
    } else {
      nibble = c - 'A' + 10;
    }

    const int word_idx = i / 8;
    const int bit_pos = (i % 8) * 4;
    words[word_idx] |= (nibble << bit_pos);
  }

  return true;
}

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);

  // Parse arguments
  const char *hex_file = nullptr;
  uint64_t max_cycles = 50000000; // 50M cycles default
  for (int i = 1; i < argc; i++) {
    if (strncmp(argv[i], "+loadmem=", 9) == 0) {
      hex_file = argv[i] + 9;
    } else if (strncmp(argv[i], "+max-cycles=", 12) == 0) {
      max_cycles = atoll(argv[i] + 12);
    }
  }

  if (!hex_file) {
    fprintf(stderr, "Usage: %s +loadmem=<hexfile> [+max-cycles=N]\n",
            argv[0]);
    return 1;
  }

  // Create model
  Vriscv_top *top = new Vriscv_top;

  // ---------------------------------------------------------------
  // Reset phase FIRST — this triggers Verilator's `initial` blocks
  // which zero all RAM in no_cache_mem. We must do this BEFORE
  // loading the hex file, or the initial block will wipe our data.
  // ---------------------------------------------------------------
  top->reset = 1;
  top->clk = 0;
  for (int i = 0; i < 100; i++) {
    top->clk = !top->clk;
    top->eval();
    main_time++;
  }
  top->reset = 0;

  // ---------------------------------------------------------------
  // NOW load hex file into memory (after initial blocks have run)
  // ---------------------------------------------------------------
  FILE *f = fopen(hex_file, "r");
  if (!f) {
    fprintf(stderr, "ERROR: Cannot open hex file: %s\n", hex_file);
    return 1;
  }

  // objcopy -O binary strips the base address. The binary starts at VMA 0x2000
  // (PC_RESET). no_cache_mem uses 128-bit (16-byte) indexed RAM, so
  // 0x2000 / 16 = 512. We must load at that offset to match the CPU.
  const int mem_depth =
      sizeof(top->rootp->riscv_top__DOT__mem__DOT__icache__DOT__ram) /
      sizeof(top->rootp->riscv_top__DOT__mem__DOT__icache__DOT__ram[0]);

  char line[256];
  int addr = 0x2000 / 16; // must match PC_RESET / (MEM_DATA_BITS/8)
  int line_no = 0;
  while (fgets(line, sizeof(line), f)) {
    line_no++;
    line[strcspn(line, "\r\n")] = 0;
    if (strlen(line) == 0)
      continue;

    uint32_t words[4] = {0, 0, 0, 0};
    if (!parse_hex_line(line, words)) {
      fprintf(stderr, "ERROR: Invalid hex line %d in %s\n", line_no, hex_file);
      fclose(f);
      delete top;
      return 1;
    }
    if (addr >= mem_depth) {
      fprintf(stderr,
              "ERROR: Hex image exceeds memory depth at line %d (addr=%d)\n",
              line_no, addr);
      fclose(f);
      delete top;
      return 1;
    }

    // Store as 128-bit value in memory model
    // Verilator stores wide signals as arrays of uint32_t
    // For WData (wide data), index [0] is LSB
    top->rootp->riscv_top__DOT__mem__DOT__icache__DOT__ram[addr][0] = words[0];
    top->rootp->riscv_top__DOT__mem__DOT__icache__DOT__ram[addr][1] = words[1];
    top->rootp->riscv_top__DOT__mem__DOT__icache__DOT__ram[addr][2] = words[2];
    top->rootp->riscv_top__DOT__mem__DOT__icache__DOT__ram[addr][3] = words[3];

    top->rootp->riscv_top__DOT__mem__DOT__dcache__DOT__ram[addr][0] = words[0];
    top->rootp->riscv_top__DOT__mem__DOT__dcache__DOT__ram[addr][1] = words[1];
    top->rootp->riscv_top__DOT__mem__DOT__dcache__DOT__ram[addr][2] = words[2];
    top->rootp->riscv_top__DOT__mem__DOT__dcache__DOT__ram[addr][3] = words[3];

    addr++;
  }
  fclose(f);
  fprintf(stderr, "Loaded %d lines from %s\n", addr - 0x2000 / 16, hex_file);

  // Run simulation
  uint64_t cycle_count = 0;

  while (cycle_count < max_cycles) {
    // Rising edge
    top->clk = 1;
    top->eval();
    main_time++;
    cycle_count++;

    // Check CSR (tohost)
    uint32_t csr_val = top->csr;

    if (csr_val == 1 && cycle_count > 10) {
      fprintf(stderr, "*** PASSED *** after %llu simulation cycles\n",
              cycle_count);
      break;
    }
    if (csr_val > 1 && cycle_count > 10) {
      fprintf(stderr,
              "*** FAILED *** (tohost = %d) after %llu simulation cycles\n",
              csr_val, cycle_count);
      break;
    }

    // Falling edge
    top->clk = 0;
    top->eval();
    main_time++;
  }

  if (cycle_count >= max_cycles) {
    fprintf(stderr, "*** TIMEOUT *** after %llu simulation cycles\n",
            cycle_count);
  }

  fprintf(stderr, "Total cycles: %llu\n", cycle_count);

  top->final();
  delete top;
  return 0;
}
