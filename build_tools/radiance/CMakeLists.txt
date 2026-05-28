# SPDX-License-Identifier: Apache-2.0
#
# Driver CMakeLists for the Merlin Radiance/Muon kernel build.
#
# This is *not* an IREE build — it's a thin wrap around the radiance-kernels
# reference Makefile. Loaded by `./merlin build --profile radiance_muon`.
#
# Required configure-time args:
# -DCMAKE_TOOLCHAIN_FILE=.../riscv_muon.toolchain.cmake
# -DMERLIN_RADIANCE_KERNEL_DIR=<absolute path to a dir containing kernel.cpp>
#
# Optional: -DMERLIN_RADIANCE_KERNEL_NAME=<basename>  (default: derived from dir
# name)
#
# Output: <kernel_name>.radiance.elf at the CMake build root.

cmake_minimum_required(VERSION 3.20)
project(merlin_radiance_muon LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/common.cmake")

# Pin verification (warn-only; doesn't block the build).
merlin_radiance_pin_check()

# --- Resolve the kernel source dir --------------------------------------

if(NOT DEFINED MERLIN_RADIANCE_KERNEL_DIR)
  message(
    FATAL_ERROR
      "MERLIN_RADIANCE_KERNEL_DIR must be set, e.g.:\n"
      "  -DMERLIN_RADIANCE_KERNEL_DIR=$RADIANCE_KERNELS_ROOT/kernels/vecadd")
endif()

if(NOT IS_DIRECTORY "${MERLIN_RADIANCE_KERNEL_DIR}")
  message(
    FATAL_ERROR
      "MERLIN_RADIANCE_KERNEL_DIR does not exist: ${MERLIN_RADIANCE_KERNEL_DIR}"
  )
endif()

if(NOT DEFINED MERLIN_RADIANCE_KERNEL_NAME)
  get_filename_component(MERLIN_RADIANCE_KERNEL_NAME
                         "${MERLIN_RADIANCE_KERNEL_DIR}" NAME)
endif()

set(_kernel_src "${MERLIN_RADIANCE_KERNEL_DIR}/kernel.cpp")
if(NOT EXISTS "${_kernel_src}")
  message(
    FATAL_ERROR
      "${MERLIN_RADIANCE_KERNEL_DIR}/kernel.cpp not found.\n"
      "  The radiance-kernels convention requires kernel.cpp at the kernel "
      "directory root.")
endif()

# Phase-2 manifest mode: when MERLIN_RADIANCE_KERNEL_BODY_OBJ is provided, we
# link the precompiled kernel-body .o (from tools/kernels/precompile.py manifest
# mode) alongside the wrapper. The wrapper template's main() declares the body
# as `extern "C"` and calls mu_schedule on it; the link step satisfies the
# symbol from the .o.
set(MERLIN_RADIANCE_KERNEL_BODY_OBJ
    ""
    CACHE FILEPATH
          "Precompiled Muon kernel body .o (manifest mode); empty in Phase 1.")

set(_host_src "${MERLIN_RADIANCE_KERNEL_DIR}/host.cpp")
if(NOT EXISTS "${_host_src}")
  set(_host_src "")
endif()

message(STATUS "merlin radiance: kernel name = ${MERLIN_RADIANCE_KERNEL_NAME}")
message(STATUS "merlin radiance: kernel src  = ${_kernel_src}")
if(_host_src)
  message(
    STATUS
      "merlin radiance: host src    = ${_host_src} (informational; .soc.elf is deferred)"
  )
endif()

# --- Compile + link the kernel ------------------------------------------

merlin_radiance_kernel_executable(NAME "${MERLIN_RADIANCE_KERNEL_NAME}" KERNEL
                                  "${_kernel_src}")

# Manifest-mode: link the precompiled body .o into the executable.
if(NOT MERLIN_RADIANCE_KERNEL_BODY_OBJ STREQUAL "")
  if(NOT EXISTS "${MERLIN_RADIANCE_KERNEL_BODY_OBJ}")
    message(
      FATAL_ERROR
        "MERLIN_RADIANCE_KERNEL_BODY_OBJ does not exist: ${MERLIN_RADIANCE_KERNEL_BODY_OBJ}"
    )
  endif()
  message(
    STATUS
      "merlin radiance: linking body .o = ${MERLIN_RADIANCE_KERNEL_BODY_OBJ}")
  target_link_libraries("${MERLIN_RADIANCE_KERNEL_NAME}_radiance"
                        PRIVATE "${MERLIN_RADIANCE_KERNEL_BODY_OBJ}")
endif()

# Convenience: dump the ELF after link, mirroring the upstream `%.radiance.dump:
# %.radiance.elf` rule. Useful for byte-equivalence debugging.
add_custom_command(
  TARGET "${MERLIN_RADIANCE_KERNEL_NAME}_radiance"
  POST_BUILD
  COMMAND
    "${CMAKE_OBJDUMP}" -D
    "$<TARGET_FILE:${MERLIN_RADIANCE_KERNEL_NAME}_radiance>" >
    "${MERLIN_RADIANCE_KERNEL_NAME}.radiance.dump"
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
  COMMENT "Dumping ${MERLIN_RADIANCE_KERNEL_NAME}.radiance.elf disassembly"
  VERBATIM)
