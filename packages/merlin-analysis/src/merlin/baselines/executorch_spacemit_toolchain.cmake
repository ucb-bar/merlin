# CMake toolchain file: cross-compile ExecuTorch's executor_runner + XNNPACK for the
# SpacemiT K1 (rv64gcv glibc Linux) using the SpacemiT clang-19 toolchain.
#
# This REPLACES examples/riscv/riscv64-linux-gnu-toolchain.cmake (which assumes the Ubuntu
# gcc-riscv64-linux-gnu apt package and does NOT enable the vector extension). We force
# -march=rv64gcv -mabi=lp64d on the WHOLE build so:
#   * the ExecuTorch core + portable kernels are compiled +v (auto-vectorizable), and
#   * rvv_audit.enforce_rvv_march is satisfied (no scalar-only binary slips through).
# XNNPACK's own RVV microkernels additionally carry -march=rv64gcv via its CMakeLists
# SET_PROPERTY; those are the hand-written RVV ukernels this arm is meant to exercise.
#
# The toolchain root is taken from MERLIN_K1_TOOLCHAIN_ROOT (set by baselines/executorch.py
# from merlin.rvvgen.k1._toolchain_root()), so we do not hard-code the /scratch2 path here.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

if(NOT DEFINED ENV{MERLIN_K1_TOOLCHAIN_ROOT})
  message(FATAL_ERROR "MERLIN_K1_TOOLCHAIN_ROOT env must point at the SpacemiT toolchain prefix "
                      "(the dir containing bin/clang)")
endif()
set(_TC "$ENV{MERLIN_K1_TOOLCHAIN_ROOT}")

set(CMAKE_C_COMPILER   "${_TC}/bin/clang"   CACHE FILEPATH "SpacemiT RISC-V C compiler")
set(CMAKE_CXX_COMPILER "${_TC}/bin/clang++" CACHE FILEPATH "SpacemiT RISC-V C++ compiler")
set(CMAKE_AR      "${_TC}/bin/llvm-ar"      CACHE FILEPATH "archiver")
set(CMAKE_RANLIB  "${_TC}/bin/llvm-ranlib"  CACHE FILEPATH "ranlib")
set(CMAKE_STRIP   "${_TC}/bin/llvm-strip"   CACHE FILEPATH "strip")

# The SpacemiT clang defaults to --target=riscv64-unknown-linux-gnu, but be explicit and
# pin the RVV march/abi on every translation unit (C, C++, ASM).
set(_MARCH "rv64gcv")
set(_MABI  "lp64d")
set(CMAKE_C_FLAGS_INIT   "--target=riscv64-unknown-linux-gnu -march=${_MARCH} -mabi=${_MABI}")
set(CMAKE_CXX_FLAGS_INIT "--target=riscv64-unknown-linux-gnu -march=${_MARCH} -mabi=${_MABI}")
set(CMAKE_ASM_FLAGS_INIT "--target=riscv64-unknown-linux-gnu -march=${_MARCH} -mabi=${_MABI}")

# The board runs its own glibc; the toolchain ships a matching sysroot. Search libs/includes in
# the toolchain sysroot, but find build programs (llvm tools) on the host.
set(CMAKE_SYSROOT "${_TC}/sysroot")
set(CMAKE_FIND_ROOT_PATH "${_TC}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
