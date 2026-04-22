# Copyright 2026 UCB-BAR

iree_cc_library(
  NAME
    registration
  HDRS
    "driver_module.h"
  SRCS
    "driver_module.c"
  DEPS
    iree::base
    iree::base::tooling::flags
    iree::hal
    iree::hal::drivers::cuda_tile
  DEFINES
    "IREE_HAVE_HAL_CUDA_TILE_DRIVER_MODULE=1"
  PUBLIC
)
