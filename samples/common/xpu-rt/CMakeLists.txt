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

set(_XPURT_IREE_RUNTIME_TARGET "iree_runtime_runtime")
if(TARGET iree::runtime::runtime)
  set(_XPURT_IREE_RUNTIME_TARGET "iree::runtime::runtime")
elseif(TARGET iree_runtime_runtime)
  set(_XPURT_IREE_RUNTIME_TARGET "iree_runtime_runtime")
endif()

add_library(xpurt_objs OBJECT baseline_runner.cc scheduler_runner.cc)
target_compile_features(xpurt_objs PRIVATE cxx_std_17)
target_include_directories(xpurt_objs PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_link_libraries(
  xpurt_objs PUBLIC merlin_dispatch iree_tooling_function_util Threads::Threads)

add_library(xpurt STATIC $<TARGET_OBJECTS:xpurt_objs>)
target_include_directories(xpurt PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_link_libraries(xpurt PUBLIC merlin_dispatch iree_tooling_function_util
                                   Threads::Threads)
