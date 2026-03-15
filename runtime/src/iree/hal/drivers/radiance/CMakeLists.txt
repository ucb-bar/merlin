set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "iree/hal/drivers/radiance")

if(NOT DEFINED MERLIN_HAL_RADIANCE_BUILD_TESTS)
  set(MERLIN_HAL_RADIANCE_BUILD_TESTS ON)
endif()
if(NOT DEFINED MERLIN_HAL_RADIANCE_ENABLE_RPC_COMPAT)
  set(MERLIN_HAL_RADIANCE_ENABLE_RPC_COMPAT ON)
endif()
if(NOT DEFINED MERLIN_HAL_RADIANCE_ENABLE_DIRECT_SUBMIT)
  set(MERLIN_HAL_RADIANCE_ENABLE_DIRECT_SUBMIT ON)
endif()
if(NOT DEFINED MERLIN_HAL_RADIANCE_ENABLE_KMOD)
  set(MERLIN_HAL_RADIANCE_ENABLE_KMOD ON)
endif()

set(_RADIANCE_DEFINES)
if(MERLIN_HAL_RADIANCE_ENABLE_RPC_COMPAT)
  list(APPEND _RADIANCE_DEFINES "MERLIN_RADIANCE_ENABLE_RPC_COMPAT=1")
else()
  list(APPEND _RADIANCE_DEFINES "MERLIN_RADIANCE_ENABLE_RPC_COMPAT=0")
endif()
if(MERLIN_HAL_RADIANCE_ENABLE_DIRECT_SUBMIT)
  list(APPEND _RADIANCE_DEFINES "MERLIN_RADIANCE_ENABLE_DIRECT_SUBMIT=1")
else()
  list(APPEND _RADIANCE_DEFINES "MERLIN_RADIANCE_ENABLE_DIRECT_SUBMIT=0")
endif()
if(MERLIN_HAL_RADIANCE_ENABLE_KMOD)
  list(APPEND _RADIANCE_DEFINES "MERLIN_RADIANCE_ENABLE_KMOD=1")
else()
  list(APPEND _RADIANCE_DEFINES "MERLIN_RADIANCE_ENABLE_KMOD=0")
endif()

iree_cc_library(
  NAME
  radiance
  HDRS
  "api.h"
  SRCS
  "api.h"
  "allocator.c"
  "allocator.h"
  "buffer.c"
  "buffer.h"
  "command_buffer.c"
  "command_buffer.h"
  "dispatch_builder.c"
  "dispatch_builder.h"
  "device.c"
  "device.h"
  "driver.c"
  "driver.h"
  "executable.c"
  "executable.h"
  "executable_cache.c"
  "executable_cache.h"
  "occupancy.c"
  "occupancy.h"
  "packet_layout.h"
  "semaphore.c"
  "semaphore.h"
  "submission.c"
  "submission.h"
  "target_caps.c"
  "target_caps.h"
  "transport/direct_submit.c"
  "transport/kmod.c"
  "transport/rpc_compat.c"
  "transport/transport.c"
  "transport/transport.h"
  DEPS
  iree::base
  iree::base::internal
  iree::base::internal::arena
  iree::hal
  iree::hal::local
  iree::hal::local::executable_loader
  iree::hal::utils::deferred_command_buffer
  iree::hal::utils::file_transfer
  iree::hal::utils::files
  iree::hal::utils::queue_emulation
  iree::hal::utils::semaphore_base
  DEFINES
  ${_RADIANCE_DEFINES}
  PUBLIC)

add_subdirectory(registration)

if(IREE_BUILD_TESTS AND MERLIN_HAL_RADIANCE_BUILD_TESTS)
  add_subdirectory(testing)
endif()
