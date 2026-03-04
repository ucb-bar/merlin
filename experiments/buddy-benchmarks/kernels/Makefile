# Makefile for Buddy-MLIR Gemmini kernel benchmarks
#
# Builds MLIR kernels through buddy-opt -> buddy-translate -> buddy-llc,
# then links with C harnesses against gemmini-rocc-tests baremetal runtime.
#
# Targets:
#   all               - Build all kernel benchmarks
#   run-all           - Run all on Spike
#   <name>-baremetal  - Build a specific benchmark
#   run-<name>        - Run a specific benchmark on Spike
#   clean             - Remove build artifacts

# ============== Paths ==============
RISCV ?= /home/eecs/ashvin.verma/toolchains/riscv
BUDDY ?= /scratch/ashvin/buddy-mlir/build/bin
PK ?= /scratch/ashvin/riscv-pk/build/pk
SPIKE ?= $(RISCV)/bin/spike

GEMMINI_ROOT := /scratch/ashvin/chipyard/generators/gemmini/software/gemmini-rocc-tests
BENCH_COMMON := $(GEMMINI_ROOT)/riscv-tests/benchmarks/common
GEMMINI_INCLUDE := $(GEMMINI_ROOT)/include
MLP_DIR := $(GEMMINI_ROOT)/mlps

# ============== Compilers ==============
CC := $(RISCV)/bin/riscv64-unknown-elf-gcc

# ============== Flags ==============
CFLAGS := \
	-DPREALLOCATE=1 \
	-DMULTITHREAD=1 \
	-mcmodel=medany \
	-std=gnu99 \
	-O2 \
	-ffast-math \
	-fno-common \
	-fno-builtin-printf \
	-fno-tree-loop-distribute-patterns \
	-march=rv64gc -Wa,-march=rv64gc \
	-I$(GEMMINI_ROOT)/riscv-tests \
	-I$(GEMMINI_ROOT)/riscv-tests/env \
	-I$(GEMMINI_ROOT) \
	-I$(BENCH_COMMON) \
	-I$(GEMMINI_INCLUDE) \
	-I$(MLP_DIR) \
	-Wno-incompatible-pointer-types

CFLAGS_BAREMETAL := \
	$(CFLAGS) \
	-nostdlib \
	-nostartfiles \
	-static \
	-T $(BENCH_COMMON)/test.ld \
	-DBAREMETAL=1

LIBS := -lm -lgcc

# Benchmark common sources
BENCH_SRCS := $(wildcard $(BENCH_COMMON)/*.c) $(wildcard $(BENCH_COMMON)/*.S)

# ============== Buddy MLIR passes ==============
BUDDY_OPT_FLAGS := \
	-lower-gemmini \
	-convert-scf-to-cf \
	-convert-arith-to-llvm \
	-convert-func-to-llvm \
	-llvm-legalize-for-export

BUDDY_LLC_FLAGS := \
	-O3 \
	-filetype=obj \
	-mtriple=riscv64-unknown-elf \
	-mattr=+buddyext,+d,+f,+c \
	-float-abi=hard \
	-code-model=medium

# ============== Benchmark definitions ==============
# Each benchmark: (name, mlir-dir, mlir-file, c-file, func-name)
BENCHMARKS := conv conv-with-pool mlp2 mlp2-os mlp1 softmax-matmul igelu-matmul

# Build directory for intermediate artifacts
BUILD := build

# ============== Targets ==============
.PHONY: all clean run-all $(addprefix run-,$(BENCHMARKS))

all: $(addsuffix -baremetal,$(BENCHMARKS))

# ---- Generic MLIR compilation rules ----
# Pattern: build/<name>.llvm.mlir from <dir>/<file>.mlir
$(BUILD)/%.llvm.mlir: | $(BUILD)
	$(BUDDY)/buddy-opt $< $(BUDDY_OPT_FLAGS) -o $@

$(BUILD)/%.ll: $(BUILD)/%.llvm.mlir
	$(BUDDY)/buddy-translate $< --buddy-to-llvmir -o $@

$(BUILD)/%.o: $(BUILD)/%.ll
	$(BUDDY)/buddy-llc $(BUDDY_LLC_FLAGS) $< -o $@

$(BUILD):
	mkdir -p $(BUILD)

# ---- conv ----
$(BUILD)/conv-buddy.llvm.mlir: conv/conv-buddy.mlir | $(BUILD)
	$(BUDDY)/buddy-opt $< $(BUDDY_OPT_FLAGS) -o $@

$(BUILD)/conv-buddy.ll: $(BUILD)/conv-buddy.llvm.mlir
	$(BUDDY)/buddy-translate $< --buddy-to-llvmir -o $@

$(BUILD)/conv-buddy.o: $(BUILD)/conv-buddy.ll
	$(BUDDY)/buddy-llc $(BUDDY_LLC_FLAGS) $< -o $@

conv-baremetal: conv/conv-buddy.c $(BUILD)/conv-buddy.o
	$(CC) $(CFLAGS_BAREMETAL) $< $(BUILD)/conv-buddy.o $(BENCH_SRCS) $(LIBS) -o $@

# ---- conv-with-pool ----
$(BUILD)/conv-with-pool-buddy.llvm.mlir: conv-with-pool/conv-with-pool-buddy.mlir | $(BUILD)
	$(BUDDY)/buddy-opt $< $(BUDDY_OPT_FLAGS) -o $@

$(BUILD)/conv-with-pool-buddy.ll: $(BUILD)/conv-with-pool-buddy.llvm.mlir
	$(BUDDY)/buddy-translate $< --buddy-to-llvmir -o $@

$(BUILD)/conv-with-pool-buddy.o: $(BUILD)/conv-with-pool-buddy.ll
	$(BUDDY)/buddy-llc $(BUDDY_LLC_FLAGS) $< -o $@

conv-with-pool-baremetal: conv-with-pool/conv-with-pool-buddy.c $(BUILD)/conv-with-pool-buddy.o
	$(CC) $(CFLAGS_BAREMETAL) $< $(BUILD)/conv-with-pool-buddy.o $(BENCH_SRCS) $(LIBS) -o $@

# ---- mlp2 (weight-stationary) ----
$(BUILD)/mlp2-buddy.llvm.mlir: mlp2/mlp2-buddy.mlir | $(BUILD)
	$(BUDDY)/buddy-opt $< $(BUDDY_OPT_FLAGS) -o $@

$(BUILD)/mlp2-buddy.ll: $(BUILD)/mlp2-buddy.llvm.mlir
	$(BUDDY)/buddy-translate $< --buddy-to-llvmir -o $@

$(BUILD)/mlp2-buddy.o: $(BUILD)/mlp2-buddy.ll
	$(BUDDY)/buddy-llc $(BUDDY_LLC_FLAGS) $< -o $@

mlp2-baremetal: mlp2/mlp2-buddy.c $(BUILD)/mlp2-buddy.o
	$(CC) $(CFLAGS_BAREMETAL) $< $(BUILD)/mlp2-buddy.o $(BENCH_SRCS) $(LIBS) -o $@

# ---- mlp2-os (output-stationary) ----
$(BUILD)/mlp2-buddy-os.llvm.mlir: mlp2/mlp2-buddy-os.mlir | $(BUILD)
	$(BUDDY)/buddy-opt $< $(BUDDY_OPT_FLAGS) -o $@

$(BUILD)/mlp2-buddy-os.ll: $(BUILD)/mlp2-buddy-os.llvm.mlir
	$(BUDDY)/buddy-translate $< --buddy-to-llvmir -o $@

$(BUILD)/mlp2-buddy-os.o: $(BUILD)/mlp2-buddy-os.ll
	$(BUDDY)/buddy-llc $(BUDDY_LLC_FLAGS) $< -o $@

mlp2-os-baremetal: mlp2/mlp2-buddy.c $(BUILD)/mlp2-buddy-os.o
	$(CC) $(CFLAGS_BAREMETAL) $< $(BUILD)/mlp2-buddy-os.o $(BENCH_SRCS) $(LIBS) -o $@

# ---- mlp1 (6-layer) ----
$(BUILD)/mlp1-buddy.llvm.mlir: mlp1/mlp1-buddy.mlir | $(BUILD)
	$(BUDDY)/buddy-opt $< $(BUDDY_OPT_FLAGS) -o $@

$(BUILD)/mlp1-buddy.ll: $(BUILD)/mlp1-buddy.llvm.mlir
	$(BUDDY)/buddy-translate $< --buddy-to-llvmir -o $@

$(BUILD)/mlp1-buddy.o: $(BUILD)/mlp1-buddy.ll
	$(BUDDY)/buddy-llc $(BUDDY_LLC_FLAGS) $< -o $@

mlp1-baremetal: mlp1/mlp1-buddy.c $(BUILD)/mlp1-buddy.o
	$(CC) $(CFLAGS_BAREMETAL) $< $(BUILD)/mlp1-buddy.o $(BENCH_SRCS) $(LIBS) -o $@

# ---- softmax-matmul ----
$(BUILD)/softmax-matmul-buddy.llvm.mlir: softmax-matmul/softmax-matmul-buddy.mlir | $(BUILD)
	$(BUDDY)/buddy-opt $< $(BUDDY_OPT_FLAGS) -o $@

$(BUILD)/softmax-matmul-buddy.ll: $(BUILD)/softmax-matmul-buddy.llvm.mlir
	$(BUDDY)/buddy-translate $< --buddy-to-llvmir -o $@

$(BUILD)/softmax-matmul-buddy.o: $(BUILD)/softmax-matmul-buddy.ll
	$(BUDDY)/buddy-llc $(BUDDY_LLC_FLAGS) $< -o $@

softmax-matmul-baremetal: softmax-matmul/softmax-matmul-buddy.c $(BUILD)/softmax-matmul-buddy.o
	$(CC) $(CFLAGS_BAREMETAL) $< $(BUILD)/softmax-matmul-buddy.o $(BENCH_SRCS) $(LIBS) -o $@

# ---- igelu-matmul ----
$(BUILD)/igelu-matmul-buddy.llvm.mlir: igelu-matmul/igelu-matmul-buddy.mlir | $(BUILD)
	$(BUDDY)/buddy-opt $< $(BUDDY_OPT_FLAGS) -o $@

$(BUILD)/igelu-matmul-buddy.ll: $(BUILD)/igelu-matmul-buddy.llvm.mlir
	$(BUDDY)/buddy-translate $< --buddy-to-llvmir -o $@

$(BUILD)/igelu-matmul-buddy.o: $(BUILD)/igelu-matmul-buddy.ll
	$(BUDDY)/buddy-llc $(BUDDY_LLC_FLAGS) $< -o $@

igelu-matmul-baremetal: igelu-matmul/igelu-matmul-buddy.c $(BUILD)/igelu-matmul-buddy.o
	$(CC) $(CFLAGS_BAREMETAL) $< $(BUILD)/igelu-matmul-buddy.o $(BENCH_SRCS) $(LIBS) -o $@

# ============== Run targets ==============
run-conv: conv-baremetal
	$(SPIKE) --extension=gemmini $<

run-conv-with-pool: conv-with-pool-baremetal
	$(SPIKE) --extension=gemmini $<

run-mlp2: mlp2-baremetal
	$(SPIKE) --extension=gemmini $<

run-mlp2-os: mlp2-os-baremetal
	$(SPIKE) --extension=gemmini $<

run-mlp1: mlp1-baremetal
	$(SPIKE) --extension=gemmini $<

run-softmax-matmul: softmax-matmul-baremetal
	$(SPIKE) --extension=gemmini $<

run-igelu-matmul: igelu-matmul-baremetal
	$(SPIKE) --extension=gemmini $<

run-all: $(addsuffix -baremetal,$(BENCHMARKS))
	@for bench in $(BENCHMARKS); do \
		echo "=== Running $$bench ==="; \
		$(SPIKE) --extension=gemmini $${bench}-baremetal 2>&1; \
		echo ""; \
	done

# ============== Clean ==============
clean:
	rm -rf $(BUILD)
	rm -f $(addsuffix -baremetal,$(BENCHMARKS))
