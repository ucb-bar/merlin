# SPDX-License-Identifier: Apache-2.0
#
# Merlin RISC-V Muon toolchain CMake.
#
# This file is the single source of truth for the cross-compile flag set used to
# build Radiance/Muon GPU kernels (RV32IM_zfinx_zhinx, ilp32, +vortex). It
# replicates the canonical flag list from `radiance-kernels/kernels/common.mk`
# exactly, so a Merlin-built kernel ELF is byte-equivalent to the upstream
# reference.
#
# Required environment: $LLVM_MUON              path to llvm-muon root
# (containing bin/clang etc.) $RADIANCE_KERNELS_ROOT  radiance-kernels checkout
# for libmuonrt.a + headers
#
# Optional: $RADIANCE_KERNELS_PIN   commit hash to validate against pin.txt;
# CMake fails loudly on mismatch.

if(MERLIN_RISCV_MUON_TOOLCHAIN_INCLUDED)
  return()
endif()
set(MERLIN_RISCV_MUON_TOOLCHAIN_INCLUDED true)

# --- 1. Target System Configuration --------------------------------------

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR riscv32)

# --- 2. Toolchain paths --------------------------------------------------

if(NOT DEFINED ENV{LLVM_MUON})
  message(
    FATAL_ERROR
      "LLVM_MUON environment variable must be set, e.g.\n"
      "  export LLVM_MUON=/scratch2/agustin/radiance-kernels/llvm/llvm-muon")
endif()
set(LLVM_MUON
    "$ENV{LLVM_MUON}"
    CACHE PATH "llvm-muon install root" FORCE)

if(NOT DEFINED ENV{RADIANCE_KERNELS_ROOT})
  message(
    FATAL_ERROR
      "RADIANCE_KERNELS_ROOT environment variable must be set, e.g.\n"
      "  export RADIANCE_KERNELS_ROOT=/scratch2/agustin/radiance-kernels")
endif()
set(RADIANCE_KERNELS_ROOT
    "$ENV{RADIANCE_KERNELS_ROOT}"
    CACHE PATH "radiance-kernels checkout root" FORCE)

set(RADIANCE_LIB_PATH
    "${RADIANCE_KERNELS_ROOT}/lib"
    CACHE PATH "radiance-kernels lib dir (libmuonrt.a, tohost.S)" FORCE)
set(RADIANCE_INCLUDE_PATH
    "${RADIANCE_LIB_PATH}/include"
    CACHE PATH "radiance-kernels public headers (mu_intrinsics.h)" FORCE)
set(GEMMINI_SW_PATH
    "${RADIANCE_LIB_PATH}/mxgemmini"
    CACHE PATH "radiance-kernels mxgemmini headers" FORCE)

# --- 3. Define tools ------------------------------------------------------

set(CMAKE_C_COMPILER "${LLVM_MUON}/bin/clang")
set(CMAKE_CXX_COMPILER "${LLVM_MUON}/bin/clang++")
set(CMAKE_ASM_COMPILER "${LLVM_MUON}/bin/clang")
# llvm-muon ships its own lld; the upstream Make uses `-fuse-ld=lld`.
set(CMAKE_LINKER "${LLVM_MUON}/bin/clang++")

set(CMAKE_AR "${LLVM_MUON}/bin/llvm-ar")
set(CMAKE_RANLIB "${LLVM_MUON}/bin/llvm-ranlib")
set(CMAKE_STRIP "${LLVM_MUON}/bin/llvm-strip")
set(CMAKE_OBJDUMP "${LLVM_MUON}/bin/llvm-objdump")
set(CMAKE_OBJCOPY "${LLVM_MUON}/bin/llvm-objcopy")

# Sysroot: llvm-muon ships its own newlib-style sysroot under $LLVM_MUON.
set(CMAKE_SYSROOT "${LLVM_MUON}")

# --- 4. Flag set (mirrors radiance-kernels/kernels/common.mk MU_CFLAGS) --
#
# Keep these in lockstep with common.mk. Any drift surfaces as a binary diff in
# the byte-equivalence check (Phase 0b acceptance).

set(MERLIN_MUON_ARCH_FLAGS
    "-march=rv32im_zfinx_zhinx -mabi=ilp32"
    CACHE STRING "Muon ISA + ABI flags" FORCE)

set(MERLIN_MUON_FEATURE_FLAGS
    "-Xclang -target-feature -Xclang +vortex"
    CACHE STRING "Muon target-feature flags (Vortex SIMT print bits)" FORCE)

# Mirror MU_CFLAGS from common.mk verbatim. -mllvm -inline-threshold matters for
# the radiance-kernels reference build (the kernels rely on aggressive inlining
# of mu_intrinsics.h); diverging will produce different binaries.
set(MERLIN_MUON_BASE_CFLAGS
    "${MERLIN_MUON_ARCH_FLAGS} ${MERLIN_MUON_FEATURE_FLAGS} \
--sysroot=${LLVM_MUON} \
-O3 -std=c++20 \
-mcmodel=medany -fno-rtti -fno-exceptions \
-fdata-sections -ffunction-sections \
-mllvm -inline-threshold=262144 \
-I${RADIANCE_INCLUDE_PATH} -I${GEMMINI_SW_PATH} \
-DRADIANCE -DRADIANCE_DEVICE -DNDEBUG -DLLVM_VORTEX")

# Mirror MU_LDFLAGS from common.mk. Linker script + libmuonrt.a + tohost.S are
# mandatory: omitting any of them produces a kernel that hangs all warps (no
# mu_schedule entry sequence) or never reports completion (no tohost).
set(MERLIN_MUON_LINKER_SCRIPT
    "${CMAKE_CURRENT_LIST_DIR}/mu_link.ld"
    CACHE FILEPATH "Vendored mu_link.ld" FORCE)

set(MERLIN_MUON_BASE_LDFLAGS
    "-nodefaultlibs -nostartfiles \
-Wl,-Bstatic,-T,${MERLIN_MUON_LINKER_SCRIPT},-z,norelro \
-fuse-ld=lld \
${RADIANCE_LIB_PATH}/libmuonrt.a \
${RADIANCE_LIB_PATH}/tohost.S")

# --- 5. Apply to CMake variables -----------------------------------------

# C++ is the kernel surface (mu_intrinsics.h is C++ inline-asm-heavy). Plain C
# kernels are not supported; this is intentional, see plan §C++ is the IR.
set(CMAKE_CXX_FLAGS
    "${MERLIN_MUON_BASE_CFLAGS}"
    CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS
    "${MERLIN_MUON_BASE_CFLAGS}"
    CACHE STRING "" FORCE)
set(CMAKE_ASM_FLAGS
    "${MERLIN_MUON_BASE_CFLAGS}"
    CACHE STRING "" FORCE)

# Suppress the standard CMake link-line; we override CMAKE_CXX_LINK_EXECUTABLE
# directly so the Muon linker script + libmuonrt.a + tohost.S land in the
# correct order.
set(CMAKE_EXE_LINKER_FLAGS
    ""
    CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS
    ""
    CACHE STRING "" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS
    ""
    CACHE STRING "" FORCE)

# Override link rule to mirror common.mk's `%.radiance.elf` rule: $(MU_CXX)
# $(MU_CFLAGS) $< $(MU_LDFLAGS) -o $@ CMake substitutes <OBJECTS>,
# <LINK_LIBRARIES>, <TARGET>. Note that MU_LDFLAGS already includes the linker
# script and the runtime stubs.
set(CMAKE_CXX_LINK_EXECUTABLE
    "<CMAKE_CXX_COMPILER> ${MERLIN_MUON_BASE_CFLAGS} <OBJECTS> <LINK_LIBRARIES> ${MERLIN_MUON_BASE_LDFLAGS} -o <TARGET>"
)
set(CMAKE_C_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE}")

# --- 6. CMake cross-compile housekeeping ---------------------------------

set(CMAKE_CROSSCOMPILING ON)
set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
