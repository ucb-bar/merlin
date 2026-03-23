# Makefile for ResNet50 Gemmini vs Buddy-MLIR comparison
#
# Targets:
#   conv1-gemmini-baremetal  - Gemmini C reference (single layer)
#   conv1-buddy-baremetal    - Buddy MLIR (single layer)
#   run-gemmini              - Run Gemmini C on Spike
#   run-buddy                - Run Buddy on Spike
#   compare                  - Run both and compare checksums

# ============== Paths ==============
RISCV ?= /home/eecs/ashvin.verma/toolchains/riscv
BUDDY ?= /scratch/ashvin/buddy-mlir/build/bin
PK ?= /scratch/ashvin/riscv-pk/build/pk
SPIKE ?= $(RISCV)/bin/spike

GEMMINI_ROOT := /scratch/ashvin/chipyard/generators/gemmini/software/gemmini-rocc-tests
BENCH_COMMON := $(GEMMINI_ROOT)/riscv-tests/benchmarks/common
GEMMINI_INCLUDE := $(GEMMINI_ROOT)/include
IMAGENET_DIR := $(GEMMINI_ROOT)/imagenet

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
	-I$(IMAGENET_DIR) \
	-Wno-incompatible-pointer-types

CFLAGS_BAREMETAL := \
	$(CFLAGS) \
	-nostdlib \
	-nostartfiles \
	-static \
	-T $(BENCH_COMMON)/test.ld \
	-DBAREMETAL=1

CFLAGS_PK := \
	$(CFLAGS) \
	-static \
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

# ============== Targets ==============
.PHONY: all clean run-gemmini run-buddy run-bad compare validate conv2-validate

all: conv1-gemmini-baremetal conv1-buddy-baremetal conv1-bad-buddy-baremetal

conv2: conv2-gemmini-baremetal conv2-buddy-baremetal

# ---- Gemmini C Reference ----
conv1-gemmini-baremetal: conv1-gemmini.c
	$(CC) $(CFLAGS_BAREMETAL) $< $(BENCH_SRCS) $(LIBS) -o $@

conv1-gemmini-pk: conv1-gemmini.c
	$(CC) $(CFLAGS_PK) $< $(LIBS) -o $@

# ---- Buddy MLIR Path ----
# Step 1: Lower MLIR to LLVM dialect
conv1-buddy.llvm.mlir: conv1-buddy.mlir
	$(BUDDY)/buddy-opt $< $(BUDDY_OPT_FLAGS) -o $@

# Step 2: Translate to LLVM IR
conv1-buddy.ll: conv1-buddy.llvm.mlir
	$(BUDDY)/buddy-translate $< --buddy-to-llvmir -o $@

# Step 3: Compile to object file
conv1-buddy.o: conv1-buddy.ll
	$(BUDDY)/buddy-llc $(BUDDY_LLC_FLAGS) $< -o $@

# Step 4: Link with C harness (baremetal)
conv1-buddy-baremetal: conv1-buddy.c conv1-buddy.o
	$(CC) $(CFLAGS_BAREMETAL) $< conv1-buddy.o $(BENCH_SRCS) $(LIBS) -o $@

# Step 4 (alternate): Link with C harness (pk)
conv1-buddy-pk: conv1-buddy.c conv1-buddy.o
	$(CC) $(CFLAGS_PK) $< conv1-buddy.o $(LIBS) -o $@

# ---- BAD Buddy MLIR Path (intentionally wrong for validation) ----
conv1-bad-buddy.llvm.mlir: conv1-bad-buddy.mlir
	$(BUDDY)/buddy-opt $< $(BUDDY_OPT_FLAGS) -o $@

conv1-bad-buddy.ll: conv1-bad-buddy.llvm.mlir
	$(BUDDY)/buddy-translate $< --buddy-to-llvmir -o $@

conv1-bad-buddy.o: conv1-bad-buddy.ll
	$(BUDDY)/buddy-llc $(BUDDY_LLC_FLAGS) $< -o $@

conv1-bad-buddy-baremetal: conv1-bad-buddy.c conv1-bad-buddy.o
	$(CC) $(CFLAGS_BAREMETAL) $< conv1-bad-buddy.o $(BENCH_SRCS) $(LIBS) -o $@

# ---- Conv2 (1x1 matmul) ----
conv2-gemmini-baremetal: conv2-gemmini.c
	$(CC) $(CFLAGS_BAREMETAL) $< $(BENCH_SRCS) $(LIBS) -o $@

conv2-buddy.llvm.mlir: conv2-buddy.mlir
	$(BUDDY)/buddy-opt $< $(BUDDY_OPT_FLAGS) -o $@

conv2-buddy.ll: conv2-buddy.llvm.mlir
	$(BUDDY)/buddy-translate $< --buddy-to-llvmir -o $@

conv2-buddy.o: conv2-buddy.ll
	$(BUDDY)/buddy-llc $(BUDDY_LLC_FLAGS) $< -o $@

conv2-buddy-baremetal: conv2-buddy.c conv2-buddy.o
	$(CC) $(CFLAGS_BAREMETAL) $< conv2-buddy.o $(BENCH_SRCS) $(LIBS) -o $@

# ============== Run targets ==============
run-gemmini: conv1-gemmini-baremetal
	$(SPIKE) --extension=gemmini $<

run-gemmini-pk: conv1-gemmini-pk
	$(SPIKE) --extension=gemmini $(PK) $<

run-buddy: conv1-buddy-baremetal
	$(SPIKE) --extension=gemmini $<

run-buddy-pk: conv1-buddy-pk
	$(SPIKE) --extension=gemmini $(PK) $<

run-bad: conv1-bad-buddy-baremetal
	$(SPIKE) --extension=gemmini $<

run-conv2-gemmini: conv2-gemmini-baremetal
	$(SPIKE) --extension=gemmini $<

run-conv2-buddy: conv2-buddy-baremetal
	$(SPIKE) --extension=gemmini $<

conv2-validate: conv2-gemmini-baremetal conv2-buddy-baremetal
	@echo "========================================"
	@echo "     Conv2 (1x1 matmul) Validation     "
	@echo "========================================"
	@echo ""
	@echo "--- Gemmini C Reference ---"
	@$(SPIKE) --extension=gemmini conv2-gemmini-baremetal 2>&1 | tee conv2-gemmini.log
	@echo ""
	@echo "--- Buddy MLIR ---"
	@$(SPIKE) --extension=gemmini conv2-buddy-baremetal 2>&1 | tee conv2-buddy.log
	@echo ""
	@echo "=== Conv2 Comparison ==="
	@GEMMINI_CKSUM=$$(grep 'Conv2 output checksum:' conv2-gemmini.log | awk '{print $$4}'); \
	BUDDY_CKSUM=$$(grep 'Conv2 output checksum:' conv2-buddy.log | awk '{print $$4}'); \
	echo "Gemmini C checksum: $$GEMMINI_CKSUM"; \
	echo "Buddy checksum:     $$BUDDY_CKSUM"; \
	if [ "$$GEMMINI_CKSUM" = "$$BUDDY_CKSUM" ]; then \
		echo "[PASS] Conv2 checksums match"; \
	else \
		echo "[FAIL] Conv2 checksums do NOT match!"; \
	fi

compare: conv1-gemmini-baremetal conv1-buddy-baremetal
	@echo "=== Running Gemmini C Reference ==="
	@$(SPIKE) --extension=gemmini conv1-gemmini-baremetal 2>&1 | tee gemmini.log
	@echo ""
	@echo "=== Running Buddy MLIR ==="
	@$(SPIKE) --extension=gemmini conv1-buddy-baremetal 2>&1 | tee buddy.log
	@echo ""
	@echo "=== Comparison ==="
	@echo "Gemmini output checksum: $$(grep 'Output checksum' gemmini.log)"
	@echo "Buddy output checksum:   $$(grep 'Output checksum' buddy.log)"

# Full validation including intentional failure case
validate: conv1-gemmini-baremetal conv1-buddy-baremetal conv1-bad-buddy-baremetal
	@echo "========================================"
	@echo "     Conv1 Validation Test Suite       "
	@echo "========================================"
	@echo ""
	@echo "--- Test 1: Gemmini C Reference ---"
	@$(SPIKE) --extension=gemmini conv1-gemmini-baremetal 2>&1 | tee gemmini.log
	@GEMMINI_CKSUM=$$(grep 'Output checksum:' gemmini.log | awk '{print $$3}'); \
	echo "Reference checksum: $$GEMMINI_CKSUM" > validation_result.txt
	@echo ""
	@echo "--- Test 2: Buddy MLIR (correct) ---"
	@$(SPIKE) --extension=gemmini conv1-buddy-baremetal 2>&1 | tee buddy.log
	@echo ""
	@echo "--- Test 3: Buddy MLIR (INTENTIONAL BAD - wrong stride) ---"
	@$(SPIKE) --extension=gemmini conv1-bad-buddy-baremetal 2>&1 | tee bad.log
	@echo ""
	@echo "========================================"
	@echo "          VALIDATION RESULTS           "
	@echo "========================================"
	@GEMMINI_CKSUM=$$(grep 'Output checksum:' gemmini.log | awk '{print $$3}'); \
	BUDDY_CKSUM=$$(grep 'Output checksum:' buddy.log | awk '{print $$3}'); \
	BAD_CKSUM=$$(grep 'Output checksum:' bad.log | awk '{print $$3}'); \
	echo "Gemmini C reference checksum: $$GEMMINI_CKSUM"; \
	echo "Buddy MLIR checksum:          $$BUDDY_CKSUM"; \
	echo "BAD Buddy checksum:           $$BAD_CKSUM"; \
	echo ""; \
	if [ "$$GEMMINI_CKSUM" = "$$BUDDY_CKSUM" ]; then \
		echo "[PASS] Buddy MLIR matches Gemmini C reference"; \
	else \
		echo "[FAIL] Buddy MLIR does NOT match Gemmini C reference!"; \
	fi; \
	if [ "$$GEMMINI_CKSUM" != "$$BAD_CKSUM" ]; then \
		echo "[PASS] BAD test correctly produces different checksum (validation works)"; \
	else \
		echo "[FAIL] BAD test unexpectedly matches reference (validation broken!)"; \
	fi

# ============== Clean ==============
clean:
	rm -f *.o *.ll *.llvm.mlir *.log validation_result.txt
	rm -f conv1-gemmini-baremetal conv1-gemmini-pk
	rm -f conv1-buddy-baremetal conv1-buddy-pk
	rm -f conv1-bad-buddy-baremetal
	rm -f conv2-gemmini-baremetal conv2-buddy-baremetal
