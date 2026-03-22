# Copyright 2026 UCB-BAR
#
# Runtime executable loader for cuda_tile (CTL1 FlatBuffer format).
# This is NOT a full HAL driver — it provides the executable loading layer
# that plugs into the existing CUDA HAL driver.

iree_cc_library(
  NAME
    native_executable
  HDRS
    "native_executable.h"
  SRCS
    "native_executable.c"
  DEPS
    iree::base::api
    iree::base::internal::flatcc::parsing
    iree::base::tracing
    iree::hal::api
    iree::hal::drivers::cuda::cuda_dynamic_symbols
    iree::hal::drivers::cuda::cuda_status_util
    iree::hal::utils::executable_debug_info
    iree::hal::utils::executable_header
    iree::schemas::cuda_tile_executable_def_c_fbs
  PUBLIC
)
