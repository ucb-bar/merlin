# xpu-rt: IREE-based runtime wrapper library.
#
# This CMake file is included from IREE's runtime plugin infrastructure via
# merlin/iree_runtime_plugin.cmake. It builds a static library:
# libxpurt_iree_plugin.a
#
# The library links against IREE's runtime target(s) and exposes a thin C API
# for higher-level schedulers to call into.
#
# We also build an OBJECT library so that a "standalone" combined static lib
# (plugin + IREE runtime objects) can be created from runtime/src/iree/runtime
# via CMakeLists_standalone.cmake (included there after iree_runtime_impl
# exists).

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

add_library(xpurt_iree_plugin_objs OBJECT xpurt_scheduler_core.c
                                          xpurt_scheduler_backend_iree.c)
target_include_directories(xpurt_iree_plugin_objs
                           PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(xpurt_iree_plugin_objs
                      PUBLIC ${_XPURT_IREE_RUNTIME_TARGET})

add_library(xpurt_iree_plugin STATIC $<TARGET_OBJECTS:xpurt_iree_plugin_objs>)
target_include_directories(xpurt_iree_plugin
                           PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(xpurt_iree_plugin PUBLIC ${_XPURT_IREE_RUNTIME_TARGET})
