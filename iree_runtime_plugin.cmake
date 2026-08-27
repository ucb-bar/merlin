# Runtime plugin entry point for Merlin's xpu-rt runtime wrapper.
#
# This file is discovered by IREE via -DIREE_CMAKE_PLUGIN_PATHS=<merlin root>
# (see merlin/tools/build.py). It is included from within the IREE runtime/
# CMake context and can freely reference IREE runtime targets such as
# iree_runtime_runtime.
#
# We forward to the subdirectory that defines the xpurt_objs / xpurt targets.
#
# This path tracked a directory move and was left stale: bf8483d refactored
# xpu-rt into projects/, then 4582be0 ("Embed XPU-RT into samples for
# stand-alone export") moved it to samples/common/xpu-rt without updating this
# reference, so configuring any target that enables the runtime plugin dies
# with "add_subdirectory given source .../projects/xpu-rt which is not an
# existing directory".
#
# samples/CMakeLists.txt deliberately does NOT add common/xpu-rt -- this file
# is its only entry point, which is why the samples build cannot cover for it.
# Both SpacemiTX60 runners link `xpurt`, so the targets must exist here.

get_filename_component(MERLIN_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)

add_subdirectory("${MERLIN_SOURCE_DIR}/samples/common/xpu-rt"
                 "${IREE_BINARY_DIR}/runtime/plugins/xpu-rt")

# Merlin runtime plugin entrypoint used by IREE plugin CMake integration.

# Runtime plugin tree toggles.
option(MERLIN_RUNTIME_ENABLE_SAMPLES "Build Merlin runtime plugin samples" ON)
option(MERLIN_RUNTIME_ENABLE_BENCHMARKS
       "Build Merlin runtime plugin benchmarks" OFF)

# Radiance external HAL toggles.
option(MERLIN_RUNTIME_ENABLE_HAL_RADIANCE
       "Enable Merlin Radiance external HAL driver" OFF)
# Backward-compatibility option kept for existing build.py invocations.
option(MERLIN_ENABLE_HAL_RADIANCE
       "Legacy alias for MERLIN_RUNTIME_ENABLE_HAL_RADIANCE" OFF)
option(MERLIN_HAL_RADIANCE_BUILD_TESTS "Build Radiance runtime plugin tests" ON)
option(MERLIN_HAL_RADIANCE_ENABLE_RPC_COMPAT
       "Enable Radiance RPC-compat transport backend" ON)
option(MERLIN_HAL_RADIANCE_ENABLE_DIRECT_SUBMIT
       "Enable Radiance direct-submit transport backend" ON)
option(MERLIN_HAL_RADIANCE_ENABLE_KMOD "Enable Radiance kmod transport backend"
       ON)

if(MERLIN_ENABLE_HAL_RADIANCE)
  set(MERLIN_RUNTIME_ENABLE_HAL_RADIANCE ON)
endif()

if(MERLIN_RUNTIME_ENABLE_HAL_RADIANCE)
  iree_register_external_hal_driver(
    NAME
    radiance
    SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/runtime/src/iree/hal/drivers/radiance"
    BINARY_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/merlin/runtime/iree/hal/drivers/radiance"
    DRIVER_TARGET
    iree::hal::drivers::radiance::registration
    REGISTER_FN
    iree_hal_radiance_driver_module_register)

  if(NOT "radiance" IN_LIST IREE_EXTERNAL_HAL_DRIVERS)
    list(APPEND IREE_EXTERNAL_HAL_DRIVERS "radiance")
    set(IREE_EXTERNAL_HAL_DRIVERS
        "${IREE_EXTERNAL_HAL_DRIVERS}"
        CACHE STRING "" FORCE)
  endif()
endif()

if(MERLIN_RUNTIME_ENABLE_SAMPLES)
  add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/samples" "merlin-samples")
endif()

if(MERLIN_RUNTIME_ENABLE_BENCHMARKS)
  if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/benchmarks/CMakeLists.txt")
    add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/benchmarks" "merlin-benchmarks")
  else()
    message(
      WARNING
        "MERLIN_RUNTIME_ENABLE_BENCHMARKS=ON but benchmarks/CMakeLists.txt is missing"
    )
  endif()
endif()
