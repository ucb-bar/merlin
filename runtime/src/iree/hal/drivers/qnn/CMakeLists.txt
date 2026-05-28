# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions. See
# https://llvm.org/LICENSE.txt for license information. SPDX-License-Identifier:
# Apache-2.0 WITH LLVM-exception

# Merlin-level QNN runtime HAL driver — wraps Qualcomm's QNN SDK so an IREE
# process can load pre-built QNN context binaries (.bin blobs from
# qairt-converter) as hal.executable.objects and dispatch them via
# QnnGraph_execute. Runtime-only path: there is no IREE codegen for QNN, so the
# matching compile-time piece is the kernel embedding pipeline with source_lang:
# qnn-context-binary.
#
# Lives at runtime/src/iree/hal/drivers/qnn/ alongside the radiance driver (the
# established Merlin pattern for vendor HAL plugins). Gated under
# MERLIN_RUNTIME_ENABLE_HAL_QNN; the IREE-fork-side init.c picks us up via
# IREE_HAVE_HAL_QNN_DRIVER_MODULE=1.

set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree/hal/drivers/qnn")

if(NOT MERLIN_RUNTIME_ENABLE_HAL_QNN)
  return()
endif()

# Auto-detect the QNN SDK if QNN_SDK_ROOT isn't already set in the env or CMake
# cache. The merlin default sits at /scratch2/dima/misc_sw/qualcomm/ but
# ultimately this should come from a user-controlled env var.
if(NOT DEFINED QNN_SDK_ROOT AND NOT "$ENV{QNN_SDK_ROOT}" STREQUAL "")
  set(QNN_SDK_ROOT
      "$ENV{QNN_SDK_ROOT}"
      CACHE PATH "QNN SDK root")
endif()
if(NOT DEFINED QNN_SDK_ROOT)
  set(QNN_SDK_ROOT
      "/scratch2/dima/misc_sw/qualcomm/qairt/2.45.0.260326"
      CACHE PATH "QNN SDK root")
endif()

if(NOT EXISTS "${QNN_SDK_ROOT}/include/QNN/QnnInterface.h")
  message(
    WARNING
      "MERLIN_RUNTIME_ENABLE_HAL_QNN=ON but QnnInterface.h not found under "
      "${QNN_SDK_ROOT}. Disabling the QNN HAL driver.")
  return()
endif()

# QNN_SDK_LIB_DIR points at the per-target lib subdir. For aarch64-linux
# (QRB5165) that's lib/aarch64-oe-linux-gcc9.3 — the libs there expose
# QnnInterface_getProviders + dlopen'd backend-specific stubs. The actual
# backend libs are loaded at runtime by name; CMake just needs to find the
# headers + the libQnnSystem entry point.
set(QNN_SDK_LIB_DIR_AARCH64 "${QNN_SDK_ROOT}/lib/aarch64-oe-linux-gcc9.3")
set(QNN_SDK_LIB_DIR_X86_64 "${QNN_SDK_ROOT}/lib/x86_64-linux-clang")

# Pick the lib dir matching the build's processor architecture.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(_QNN_LIB_DIR "${QNN_SDK_LIB_DIR_AARCH64}")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  set(_QNN_LIB_DIR "${QNN_SDK_LIB_DIR_X86_64}")
else()
  message(
    WARNING
      "QNN HAL driver: unknown CMAKE_SYSTEM_PROCESSOR='${CMAKE_SYSTEM_PROCESSOR}'; "
      "skipping.")
  return()
endif()

if(NOT EXISTS "${_QNN_LIB_DIR}/libQnnSystem.so")
  message(
    WARNING "QNN HAL driver: libQnnSystem.so not found under ${_QNN_LIB_DIR}; "
            "skipping.")
  return()
endif()

iree_cc_library(
  NAME
  qnn_driver
  HDRS
  "qnn_command_buffer.h"
  "qnn_device.h"
  "qnn_driver.h"
  "qnn_executable.h"
  "qnn_executable_cache.h"
  "qnn_graph_builder.h"
  "qnn_semaphore.h"
  SRCS
  "qnn_command_buffer.c"
  "qnn_device.c"
  "qnn_driver.c"
  "qnn_executable.c"
  "qnn_executable_cache.c"
  "qnn_graph_builder.c"
  "qnn_semaphore.c"
  DEPS
  iree::async
  iree::async::util::proactor_pool
  iree::base
  iree::base::core_headers
  iree::base::internal
  iree::base::internal::dynamic_library
  iree::hal
  iree::hal::local
  iree::hal::utils::deferred_command_buffer
  iree::hal::utils::file_transfer
  iree::hal::utils::files
  iree::hal::utils::queue_emulation
  iree::hal::utils::resource_set
  iree::schemas::executable_debug_info_c_fbs
  INCLUDES
  "${QNN_SDK_ROOT}/include/QNN"
  COPTS
  "-DQNN_HAL_LIB_DIR=\"${_QNN_LIB_DIR}\""
  PUBLIC)

add_subdirectory(registration)

# Surface a compile def to the runtime-plugin scope so the HAL driver
# enumeration in registration/init.c picks up the QNN driver.
target_compile_definitions(iree_hal_drivers_qnn_qnn_driver
                           PUBLIC IREE_HAVE_HAL_QNN_DRIVER_MODULE=1)
