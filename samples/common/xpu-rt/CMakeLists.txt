# xpu-rt: generic dispatch runners — baseline topo-order and two-cluster
# scheduler.
#
# This CMake file is included from IREE's runtime plugin infrastructure. It
# builds: xpurt_objs        — OBJECT library (for standalone archive) xpurt —
# STATIC library (for in-tree linking)
#
# In-tree usage (samples, tests): target_link_libraries(<target> PRIVATE xpurt
# Threads::Threads)
#
# Out-of-tree usage (XPU-RT, Zephyr): Link against the standalone archive
# produced by CMakeLists_standalone.cmake: libxpurt_standalone.a   (xpu-rt
# objects + full IREE runtime in one .a) See CMakeLists_standalone.cmake for
# details.

find_package(Threads REQUIRED)

# Exported paths so CMakeLists_standalone.cmake can locate us.
set(MERLIN_XPU_RT_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}"
    CACHE INTERNAL "xpu-rt source dir for standalone lib")
set(MERLIN_XPU_RT_STANDALONE_CMAKE
    "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists_standalone.cmake"
    CACHE INTERNAL "Path to xpu-rt standalone CMake include")

# IREE deps. We can't gate on `if(TARGET ...)` because the merlin runtime plugin
# runs BEFORE IREE's runtime/src subdirectories add these targets. CMake
# resolves the names at generator time (after all add_subdirectory calls), so
# listing them unconditionally is safe — and it's how IREE-internal targets
# reference each other.
set(_XPURT_IREE_RUNTIME_TARGET "iree_runtime_impl")
set(_XPURT_IREE_EXTRA_DEPS
    iree_base_base
    iree_base_tooling_flags
    iree_hal_hal
    iree_hal_drivers_drivers
    iree_hal_drivers_local_task_task_driver
    iree_hal_drivers_local_task_registration_registration
    iree_hal_local_local
    iree_hal_local_executable_loader
    iree_hal_local_loaders_registration_registration
    iree_hal_local_loaders_embedded_elf_loader
    iree_hal_local_plugins_registration_registration
    iree_modules_hal_hal
    iree_task_task
    iree_task_api
    iree_vm_impl
    iree_vm_bytecode_module
    iree_async_util_proactor_pool
    iree_async_util_continuation
    iree_async_util_completion_pool)

add_library(xpurt_objs OBJECT baseline_runner.cc scheduler_runner.cc)
target_compile_features(xpurt_objs PRIVATE cxx_std_17)
target_include_directories(xpurt_objs PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_link_libraries(
  xpurt_objs PUBLIC merlin_dispatch Threads::Threads
                    ${_XPURT_IREE_RUNTIME_TARGET} ${_XPURT_IREE_EXTRA_DEPS})

# Newer IREE gates `iree_allocator_system()` behind the
# IREE_ALLOCATOR_SYSTEM_CTL preprocessor macro, which IREE itself only sets on
# iree_base_base's PRIVATE compile options — it doesn't propagate to downstream
# consumers. Mirror IREE's default (libc) so the runner call sites compile
# without each having to plumb a custom allocator.
if(NOT DEFINED IREE_ALLOCATOR_SYSTEM)
  set(IREE_ALLOCATOR_SYSTEM "libc")
endif()
target_compile_definitions(
  xpurt_objs
  PRIVATE
    "IREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_${IREE_ALLOCATOR_SYSTEM}_ctl")

add_library(xpurt STATIC $<TARGET_OBJECTS:xpurt_objs>)
target_include_directories(xpurt PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_link_libraries(
  xpurt PUBLIC merlin_dispatch Threads::Threads ${_XPURT_IREE_RUNTIME_TARGET}
               ${_XPURT_IREE_EXTRA_DEPS})

# Multi-device runner: full-model VMFB across N pinned local-task devices in one
# process. Closes the loop on dispatch-level data flow (Part C / C1) where
# vanilla iree-run-module cannot distinguish per-URI cpu_ids.
add_executable(merlin_multi_device_runner multi_device_runner.cc)
target_compile_features(merlin_multi_device_runner PRIVATE cxx_std_17)
target_include_directories(merlin_multi_device_runner
                           PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_compile_definitions(
  merlin_multi_device_runner
  PRIVATE
    "IREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_${IREE_ALLOCATOR_SYSTEM}_ctl")
target_link_libraries(
  merlin_multi_device_runner
  PRIVATE Threads::Threads ${_XPURT_IREE_RUNTIME_TARGET}
          ${_XPURT_IREE_EXTRA_DEPS} iree_tooling_context_util
          iree_tooling_function_io)
