# Builds libsaturn_opu.so — a Spike runtime extension for the Saturn OPU
# (custom OP-V 0x57 opcodes: vopacc, opmvinbcast, vmv.vr, vmv.rv).
#
# Pairs with chipyard's libgemmini.so so a single spike invocation can
# decode both Gemmini (custom-3 0x7B) and OPU (OP-V 0x57) instructions
# emitted by merlin_hetero_runner ELFs.
#
# Usage:
#   export CHIPYARD_ROOT=/path/to/chipyard
#   make install
#   spike --extension=gemmini --extension=saturn_opu -p2 zephyr.elf
#
# Configurable variables:
#   CHIPYARD_ROOT  chipyard checkout (must contain toolchains/riscv-tools/riscv-isa-sim)
#   RISCV          riscv-tools install dir; defaults to $CHIPYARD_ROOT/.conda-env/riscv-tools
#   PREFIX         install prefix for `make install`; defaults to $RISCV

ifndef CHIPYARD_ROOT
$(error CHIPYARD_ROOT is unset — point it at your chipyard checkout)
endif

SPIKE_SRC := $(CHIPYARD_ROOT)/toolchains/riscv-tools/riscv-isa-sim
SPIKE_BUILD := $(SPIKE_SRC)/build

RISCV ?= $(CHIPYARD_ROOT)/.conda-env/riscv-tools
PREFIX ?= $(RISCV)

# The two header dirs that aren't installed under $(RISCV)/include but
# saturn_opu.cc needs: spike's source riscv/ subdir (insn_macros.h,
# decode_macros.h) and the configure-time generated config.h in build/.
INCLUDES := \
	-I $(RISCV)/include \
	-I $(SPIKE_SRC)/riscv \
	-I $(SPIKE_SRC)/softfloat \
	-I $(SPIKE_SRC)/fesvr \
	-I $(SPIKE_BUILD)

CXX ?= g++
CXXFLAGS := -std=c++17 -fPIC -O3 -Wall -Wno-unused-parameter
LDFLAGS := -L $(RISCV)/lib -Wl,-rpath,$(RISCV)/lib -shared

.PHONY: default install clean test verify-spike-build

default: libsaturn_opu.so

libsaturn_opu.so: saturn_opu.cc
	@if [ ! -f $(SPIKE_BUILD)/config.h ]; then \
		echo "ERROR: $(SPIKE_BUILD)/config.h not found — chipyard's spike must be built first."; \
		echo "       Try: cd $(CHIPYARD_ROOT) && ./scripts/build-toolchains.sh"; \
		exit 1; \
	fi
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $^

install: libsaturn_opu.so
	install -d $(PREFIX)/lib
	install -m 0755 libsaturn_opu.so $(PREFIX)/lib/

test: libsaturn_opu.so
	./tests/smoke.sh

clean:
	rm -f libsaturn_opu.so
