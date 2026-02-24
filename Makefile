# Makefile for Verilator simulation of RISC-V ML Inference
# Assumes: RTL source in $(RTL_DIR), inference code in $(INFER_DIR)

# ---- Paths (adjust these to your setup) ----
RTL_DIR     = /Users/peter/asic-project-fa25-golden-gates/src
INFER_DIR   = $(CURDIR)/riscv-ml-inference
START_S     = $(CURDIR)/benchmarks/bmark/start.s
LINKER_LD   = $(CURDIR)/inference.ld

# ---- RISC-V toolchain ----
RISCV       = riscv64-unknown-elf
GCC_OPTS    = -march=rv32im_zicsr -mabi=ilp32 -static -mcmodel=medany \
              -fvisibility=hidden -nostdlib -nostartfiles -Wl,--build-id=none -O3
CFLAGS_EXTRA ?=

# ---- Verilator settings ----
VERILATOR   = verilator
ENABLE_DUAL_ISSUE ?= 1
VFLAGS      = --cc --exe --build --trace \
              -Wno-fatal \
              -Dno_cache_mem \
              $(if $(filter 1,$(ENABLE_DUAL_ISSUE)),-DENABLE_DUAL_ISSUE,) \
              $(EXTRA_VFLAGS) \
              --top-module riscv_top \
              -I$(RTL_DIR) \
              -I$(CURDIR)/riscv-accelerator

# ---- RTL source files ----
RTL_SRCS = \
	$(RTL_DIR)/riscv_top.v \
	$(RTL_DIR)/Riscv151.v \
	$(RTL_DIR)/Memory151.v \
	$(RTL_DIR)/no_cache_mem.v \
	$(RTL_DIR)/ExtMemModel.v \
	$(RTL_DIR)/Execute.sv \
	$(RTL_DIR)/Fetch.sv \
	$(RTL_DIR)/Writeback.sv \
	$(RTL_DIR)/Control_logic.sv \
	$(RTL_DIR)/RegFile.sv \
	$(RTL_DIR)/ALU.v \
	$(RTL_DIR)/ALUdec.v \
	$(RTL_DIR)/Branch_Comp.v \
	$(RTL_DIR)/BranchPrediction.v \
	$(RTL_DIR)/riscv_arbiter.v \
	$(RTL_DIR)/Multiplier.sv \
	$(RTL_DIR)/HazardUnit.sv \
	$(RTL_DIR)/Opcode.vh \
	$(CURDIR)/riscv-accelerator/MatmulAccelerator.sv

# ---- Targets ----
.PHONY: all compile_rtl compile_inference run run_bench clean clean_bench

all: run

# Step 1: Compile inference_bare.c -> hex file
inference.hex: inference_bare.c $(INFER_DIR)/runtime/weights.h $(INFER_DIR)/runtime/test_images.h $(START_S) $(LINKER_LD)
	$(RISCV)-gcc $(GCC_OPTS) $(CFLAGS_EXTRA) -I$(INFER_DIR)/runtime -T $(LINKER_LD) $(START_S) inference_bare.c -o inference.elf
	$(RISCV)-objdump -D -Mnumeric inference.elf > inference.dump
	$(RISCV)-objcopy inference.elf -O binary inference.bin
	python3 bin2hex.py -w 128 inference.bin inference.hex

# Step 2: Build Verilator simulation
obj_dir/Vriscv_top: sim_main.cpp $(RTL_SRCS)
	$(VERILATOR) $(VFLAGS) \
		$(RTL_SRCS) \
		sim_main.cpp

# Step 3: Run
run: obj_dir/Vriscv_top inference.hex
	./obj_dir/Vriscv_top +loadmem=inference.hex +max-cycles=500000000 $(EXTRA_FLAGS)

# Generic benchmark compile/run path (independent from inference_bare.c)
BENCH_NAME ?= bmark
BENCH_SRC  ?= $(CURDIR)/benchmarks/bmark/fib.c
BENCH_LD   ?= $(CURDIR)/benchmarks/bmark/common.ld
BENCH_START ?= $(CURDIR)/benchmarks/bmark/start.s
MAX_CYCLES ?= 500000000
BENCH_ELF  := $(BENCH_NAME).elf
BENCH_BIN  := $(BENCH_NAME).bin
BENCH_HEX  := $(BENCH_NAME).hex
BENCH_DUMP := $(BENCH_NAME).dump

$(BENCH_HEX): $(BENCH_SRC) $(BENCH_START) $(BENCH_LD)
	$(RISCV)-gcc $(GCC_OPTS) $(CFLAGS_EXTRA) -I$(INFER_DIR)/runtime -T $(BENCH_LD) $(BENCH_START) $(BENCH_SRC) -o $(BENCH_ELF)
	$(RISCV)-objdump -D -Mnumeric $(BENCH_ELF) > $(BENCH_DUMP)
	$(RISCV)-objcopy $(BENCH_ELF) -O binary $(BENCH_BIN)
	python3 bin2hex.py -w 128 $(BENCH_BIN) $(BENCH_HEX)

run_bench: obj_dir/Vriscv_top $(BENCH_HEX)
	./obj_dir/Vriscv_top +loadmem=$(BENCH_HEX) +max-cycles=$(MAX_CYCLES) $(EXTRA_FLAGS)

run_ilp_single:
	$(MAKE) clean
	$(MAKE) run ENABLE_DUAL_ISSUE=0 CFLAGS_EXTRA="-DILP_MICROBENCH"

run_ilp_dual:
	$(MAKE) clean
	$(MAKE) run ENABLE_DUAL_ISSUE=1 CFLAGS_EXTRA="-DILP_MICROBENCH"

clean:
	rm -rf obj_dir inference.elf inference.bin inference.hex inference.dump

clean_bench:
	rm -f *.elf *.bin *.hex *.dump
