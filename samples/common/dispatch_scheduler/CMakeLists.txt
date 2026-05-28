# Target-neutral entry point for the dependency-aware schedule runner.
#
# The binary `merlin-dispatch-scheduler` is platform-agnostic: it loads a
# schedule.json, creates HAL devices for the requested targets (CPU_P / CPU_E
# via local-task or local-sync, QNN_GPU / QNN_HTA via the QNN HAL driver),
# spawns one worker per target, and dispatches nodes respecting their dependency
# graph.
#
# Same binary works on QRB5165 (Qualcomm SA8155 / Snapdragon 865), SpacemiT X60
# (RISC-V), and any other target that has the relevant HAL drivers compiled in.
# Target-specific build profiles (qrb5165-runtime-release,
# spacemit-merlin-release, ...) just turn different driver modules on or off.
#
# Both samples/QRB5165/dispatch_scheduler/CMakeLists.txt and
# samples/SpacemiTX60/dispatch_scheduler/CMakeLists.txt alias here so the tree
# advertises the right entry point per platform.

set(_NAME "merlin_dispatch_scheduler")
add_executable(${_NAME} "")
target_sources(${_NAME} PRIVATE main.c)

set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME
                                          "merlin-dispatch-scheduler")

target_compile_features(${_NAME} PRIVATE c_std_11)

find_package(Threads REQUIRED)

if(DEFINED IREE_DEFAULT_COPTS)
  target_compile_options(${_NAME} PRIVATE ${IREE_DEFAULT_COPTS})
endif()

target_link_libraries(${_NAME} PRIVATE xpurt Threads::Threads)
