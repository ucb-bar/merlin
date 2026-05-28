# SPDX-License-Identifier: Apache-2.0
#
# Wrapper around `radiance-kernels/kernels/common.mk`. Provides a single source
# of truth for the Muon kernel build (toolchain flags + link recipe) and a CMake
# helper for compiling user kernel sources into a `kernel.radiance.elf` matching
# the upstream reference build.
#
# Usage: include(.../build_tools/radiance/common.cmake)
# merlin_radiance_pin_check()                  # validates pin.txt
# merlin_radiance_kernel_executable(           # builds kernel.radiance.elf NAME
# vecadd KERNEL    kernel.cpp [HOST     host.cpp] [DEPS     mu_lib_a.cpp
# mu_lib_b.cpp] )
#
# This module assumes the riscv_muon.toolchain.cmake has already been loaded
# (CMAKE_C_COMPILER etc. point at llvm-muon clang).

include_guard(GLOBAL)

if(NOT DEFINED LLVM_MUON OR NOT DEFINED RADIANCE_KERNELS_ROOT)
  message(
    FATAL_ERROR
      "merlin/build_tools/radiance/common.cmake requires LLVM_MUON and "
      "RADIANCE_KERNELS_ROOT to be set; load riscv_muon.toolchain.cmake first.")
endif()

# --- Pin validation ------------------------------------------------------

function(merlin_radiance_pin_check)
  set(_pin_file "${CMAKE_CURRENT_LIST_DIR}/pin.txt")
  if(NOT EXISTS "${_pin_file}")
    message(
      WARNING "merlin radiance: ${_pin_file} not found; skipping pin check.")
    return()
  endif()
  file(READ "${_pin_file}" _pin_text)
  string(REGEX MATCH "radiance_kernels_commit *= *([0-9a-f]+)" _match
               "${_pin_text}")
  if(NOT _match)
    message(
      WARNING
        "merlin radiance: pin.txt has no radiance_kernels_commit; skipping.")
    return()
  endif()
  set(_pinned_commit "${CMAKE_MATCH_1}")

  find_program(GIT_EXE git)
  if(NOT GIT_EXE)
    message(WARNING "merlin radiance: git not found; cannot verify pin.")
    return()
  endif()
  execute_process(
    COMMAND ${GIT_EXE} -C "${RADIANCE_KERNELS_ROOT}" rev-parse HEAD
    OUTPUT_VARIABLE _actual_commit
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0)
    message(
      WARNING
        "merlin radiance: ${RADIANCE_KERNELS_ROOT} is not a git checkout; "
        "skipping pin verification.")
    return()
  endif()

  if(NOT _actual_commit STREQUAL _pinned_commit)
    message(
      WARNING "merlin radiance: radiance-kernels checkout drift!\n"
              "  expected (pin.txt):  ${_pinned_commit}\n"
              "  actual (HEAD):       ${_actual_commit}\n"
              "  Bump pin.txt after re-validating the byte-equivalence test.")
  else()
    message(STATUS "merlin radiance: pin OK (${_actual_commit})")
  endif()
endfunction()

# --- Kernel executable helper -------------------------------------------

# merlin_radiance_kernel_executable( NAME       <basename>     # e.g. "vecadd"
# -> kernel.radiance.elf KERNEL     <src.cpp>      # entrypoint (must call
# mu_schedule) [HOST      <src.cpp>]     # optional RV64 host stub for soc.elf
# (deferred) [DEPS      <a.cpp> ...]   # additional MU_SRC_DEPS )
#
# Produces an executable target named `<NAME>_radiance` whose output filename is
# `<NAME>.radiance.elf`, compiled+linked with the canonical Muon toolchain.
#
# Example: merlin_radiance_kernel_executable( NAME    vecadd KERNEL
# ${RADIANCE_KERNELS_ROOT}/kernels/vecadd/kernel.cpp)
function(merlin_radiance_kernel_executable)
  set(options)
  set(oneValueArgs NAME KERNEL HOST)
  set(multiValueArgs DEPS)
  cmake_parse_arguments(MR "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        ${ARGN})

  if(NOT MR_NAME OR NOT MR_KERNEL)
    message(
      FATAL_ERROR
        "merlin_radiance_kernel_executable: NAME and KERNEL are required")
  endif()

  set(_target "${MR_NAME}_radiance")
  add_executable(${_target} ${MR_KERNEL} ${MR_DEPS})
  set_target_properties(
    ${_target}
    PROPERTIES OUTPUT_NAME "${MR_NAME}.radiance"
               SUFFIX ".elf"
               LINKER_LANGUAGE CXX)

  # Headers from radiance-kernels are already on the global `-I` path via
  # MERLIN_MUON_BASE_CFLAGS. Allow per-kernel `data/` directories alongside the
  # kernel source to be picked up automatically (vecadd uses this for `data`
  # blob includes).
  get_filename_component(_kernel_dir "${MR_KERNEL}" DIRECTORY)
  if(IS_DIRECTORY "${_kernel_dir}")
    target_include_directories(${_target} PRIVATE "${_kernel_dir}")
  endif()
endfunction()
