# IREE runtime utilities — module caching, pinned devices, fatal state.
#
# Usage: target_link_libraries(<target> PRIVATE merlin_runtime) Then:  #include
# "runtime/module_cache.h"  etc.

add_library(merlin_runtime INTERFACE)

# Newer IREE (post 2026 refactor) split `iree_runtime_runtime` into
# `iree_runtime_impl`. Pick whichever exists.
if(TARGET iree_runtime_runtime)
  target_link_libraries(merlin_runtime INTERFACE merlin_core
                                                 iree_runtime_runtime)
elseif(TARGET iree_runtime_impl)
  target_link_libraries(merlin_runtime INTERFACE merlin_core iree_runtime_impl)
else()
  target_link_libraries(merlin_runtime INTERFACE merlin_core)
endif()

# IREE internal headers (task API, local-task driver) for pinned device
# creation. The fallback walks up from this file's location to the in-tree
# iree_bar checkout — the canonical layout in this repo. We do this
# unconditionally rather than gating on IREE_SOURCE_DIR because that variable
# isn't always in scope when this CMake runs from the runtime- plugin context.
get_filename_component(_MERLIN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../"
                       ABSOLUTE)
set(_IREE_RUNTIME_SRC "${_MERLIN_ROOT}/third_party/iree_bar/runtime/src")
if(EXISTS "${_IREE_RUNTIME_SRC}")
  target_include_directories(merlin_runtime INTERFACE ${_IREE_RUNTIME_SRC})
endif()
if(DEFINED IREE_SOURCE_DIR
   AND EXISTS "${IREE_SOURCE_DIR}/runtime/src"
   AND NOT "${IREE_SOURCE_DIR}/runtime/src" STREQUAL "${_IREE_RUNTIME_SRC}")
  target_include_directories(merlin_runtime
                             INTERFACE ${IREE_SOURCE_DIR}/runtime/src)
endif()
if(DEFINED IREE_BINARY_DIR AND EXISTS "${IREE_BINARY_DIR}/runtime/src")
  target_include_directories(merlin_runtime
                             INTERFACE ${IREE_BINARY_DIR}/runtime/src)
endif()

# dispatch_bench — minimal in-tree microbenchmark harness used to measure
# per-dispatch cost across HAL backends (CPU, QNN, ...). Built only when the
# SpacemiTX60 sample tree is enabled (same gate as the scheduler binaries) since
# both share the runtime plumbing.
if(MERLIN_BUILD_XPU_RT_RUNNER OR MERLIN_BUILD_SPACEMITX60)
  add_executable(merlin_dispatch_bench dispatch_bench.c)
  set_target_properties(merlin_dispatch_bench
                        PROPERTIES OUTPUT_NAME "merlin-dispatch-bench")
  target_compile_features(merlin_dispatch_bench PRIVATE c_std_11)
  add_executable(merlin_transfer_bench transfer_bench.c)
  set_target_properties(merlin_transfer_bench
                        PROPERTIES OUTPUT_NAME "merlin-transfer-bench")
  target_compile_features(merlin_transfer_bench PRIVATE c_std_11)
  if(NOT DEFINED IREE_ALLOCATOR_SYSTEM)
    set(IREE_ALLOCATOR_SYSTEM "libc")
  endif()
  target_compile_definitions(
    merlin_dispatch_bench
    PRIVATE
      "IREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_${IREE_ALLOCATOR_SYSTEM}_ctl")
  target_compile_definitions(
    merlin_transfer_bench
    PRIVATE
      "IREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_${IREE_ALLOCATOR_SYSTEM}_ctl")
  set(_BENCH_DEPS
      merlin_runtime
      iree_runtime_impl
      iree_base_base
      iree_hal_hal
      iree_hal_drivers_drivers
      iree_hal_drivers_local_task_task_driver
      iree_hal_drivers_local_task_registration_registration
      iree_hal_local_local
      iree_hal_local_loaders_registration_registration
      iree_hal_local_loaders_embedded_elf_loader
      iree_hal_local_plugins_registration_registration
      iree_modules_hal_hal
      iree_task_task
      iree_task_api
      iree_vm_impl
      iree_vm_bytecode_module
      iree_async_util_proactor_pool)
  target_link_libraries(merlin_transfer_bench PRIVATE ${_BENCH_DEPS})

  # dispatch_flow_runner — runs a heterogeneous schedule with REAL data flowing
  # dispatch-to-dispatch (Phase F of the heterogeneous scheduling pipeline).
  # Loads N per-dispatch VMFBs, threads outputs as next dispatch's inputs,
  # transfers across devices via host-mediated queue read/write.
  add_executable(merlin_dispatch_flow_runner dispatch_flow_runner.cc)
  set_target_properties(merlin_dispatch_flow_runner
                        PROPERTIES OUTPUT_NAME "merlin-dispatch-flow-runner")
  target_compile_features(merlin_dispatch_flow_runner PRIVATE cxx_std_17)
  target_compile_definitions(
    merlin_dispatch_flow_runner
    PRIVATE
      "IREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_${IREE_ALLOCATOR_SYSTEM}_ctl")
  target_include_directories(merlin_dispatch_flow_runner
                             PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
  target_link_libraries(merlin_dispatch_flow_runner PRIVATE ${_BENCH_DEPS})
  target_link_libraries(
    merlin_dispatch_bench
    PRIVATE merlin_runtime
            iree_runtime_impl
            iree_base_base
            iree_hal_hal
            iree_hal_drivers_drivers
            iree_hal_drivers_local_task_task_driver
            iree_hal_drivers_local_task_registration_registration
            iree_hal_local_local
            iree_hal_local_loaders_registration_registration
            iree_hal_local_loaders_embedded_elf_loader
            iree_hal_local_plugins_registration_registration
            iree_modules_hal_hal
            iree_task_task
            iree_task_api
            iree_vm_impl
            iree_vm_bytecode_module
            iree_async_util_proactor_pool)
endif()
