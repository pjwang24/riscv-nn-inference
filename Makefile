# Makefile for Verilator simulation of RISC-V ML Inference
# Assumes: RTL source in $(RTL_DIR), inference code in $(INFER_DIR)

# ---- Paths (adjust these to your setup) ----
RTL_DIR     = /Users/peter/asic-project-fa25-golden-gates/src
INFER_DIR   = /Users/peter/riscv-ml-inference
START_S     = /Users/peter/asic-project-fa25-golden-gates/tests/bmark/start.s
LINKER_LD   = /Users/peter/verilator/inference.ld

# ---- RISC-V toolchain ----
RISCV       = riscv64-unknown-elf
GCC_OPTS    = -march=rv32im_zicsr -mabi=ilp32 -static -mcmodel=medany \
              -fvisibility=hidden -nostdlib -nostartfiles -Wl,--build-id=none -O3

# ---- Verilator settings ----
VERILATOR   = verilator
VFLAGS      = --cc --exe --build --trace \
              -Wno-fatal \
              -Dno_cache_mem \
              --top-module riscv_top \
              -I$(RTL_DIR) \
              -I/Users/peter/verilator/riscv-accelerator

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
	$(RTL_DIR)/Opcode.vh \
	/Users/peter/verilator/riscv-accelerator/MatmulAccelerator.sv

# ---- Targets ----
.PHONY: all compile_rtl compile_inference run clean

all: run

# Step 1: Compile inference_bare.c -> hex file
inference.hex: inference_bare.c $(INFER_DIR)/runtime/weights.h $(INFER_DIR)/runtime/test_images.h $(START_S) $(LINKER_LD)
	$(RISCV)-gcc $(GCC_OPTS) -I$(INFER_DIR)/runtime -T $(LINKER_LD) $(START_S) inference_bare.c -o inference.elf
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
	./obj_dir/Vriscv_top +loadmem=inference.hex +max-cycles=500000000

clean:
	rm -rf obj_dir inference.elf inference.bin inference.hex inference.dump