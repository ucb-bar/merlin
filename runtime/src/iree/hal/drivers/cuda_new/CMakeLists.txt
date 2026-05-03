set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree/hal/drivers/cuda_new")

iree_add_all_subdirs()

iree_cc_library(
  NAME
    dynamic_symbols
  HDRS
    "dynamic_symbols.h"
    "status_util.h"
  TEXTUAL_HDRS
    "dynamic_symbol_table.h"
  SRCS
    "dynamic_symbols.c"
    "headers.h"
    "status_util.c"
  DEPS
    iree::base
    iree::base::internal::dynamic_library
    iree_cuda::headers
  PUBLIC
)

iree_cc_library(
  NAME
    cuda_new
  HDRS
    "api.h"
  SRCS
    "api.h"
    "driver.c"
    "driver.h"
    "physical_device.c"
    "physical_device.h"
    "logical_device.c"
    "logical_device.h"
    "target_caps.c"
    "target_caps.h"
  DEPS
    ::dynamic_symbols
    iree::base
    iree::hal
  PUBLIC
)
