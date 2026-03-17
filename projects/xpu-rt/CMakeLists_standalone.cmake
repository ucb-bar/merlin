# Standalone xpu-rt static library: plugin + IREE runtime in one .a
#
# This file is included from IREE's runtime/src/iree/runtime/CMakeLists.txt
# after iree_runtime_impl and iree_runtime_unified are defined, so we can
# reference INTERFACE_IREE_TRANSITIVE_OBJECTS and build a single archive that
# contains both the xpu-rt plugin objects and the full IREE runtime. Consumers
# (e.g. json_dispatch_runner) can then link only this lib (+ system libs).

if(NOT TARGET xpurt_iree_plugin_objs)
  message(STATUS "xpurt standalone: xpurt_iree_plugin_objs not found, skipping")
  return()
endif()

# IREE target naming: prefer the namespaced targets but accept the raw ones too.
set(_XPURT_IREE_IMPL_TARGET "")
if(TARGET iree::runtime::impl)
  set(_XPURT_IREE_IMPL_TARGET "iree::runtime::impl")
elseif(TARGET iree_runtime_impl)
  set(_XPURT_IREE_IMPL_TARGET "iree_runtime_impl")
endif()

set(_XPURT_IREE_UNIFIED_TARGET "")
if(TARGET iree::runtime::unified)
  set(_XPURT_IREE_UNIFIED_TARGET "iree::runtime::unified")
elseif(TARGET iree_runtime_unified)
  set(_XPURT_IREE_UNIFIED_TARGET "iree_runtime_unified")
endif()

if(_XPURT_IREE_IMPL_TARGET STREQUAL "")
  message(
    STATUS "xpurt standalone: IREE runtime impl target not found, skipping")
  return()
endif()
if(_XPURT_IREE_UNIFIED_TARGET STREQUAL "")
  message(
    STATUS "xpurt standalone: IREE runtime unified target not found, skipping")
  return()
endif()

set(_XPURT_OBJS "$<TARGET_OBJECTS:xpurt_iree_plugin_objs>")
set(_IREE_OBJS
    "$<REMOVE_DUPLICATES:$<GENEX_EVAL:$<TARGET_PROPERTY:${_XPURT_IREE_IMPL_TARGET},INTERFACE_IREE_TRANSITIVE_OBJECTS>>>"
)

add_library(xpurt_iree_plugin_standalone STATIC ${_XPURT_OBJS} ${_IREE_OBJS})
target_include_directories(
  xpurt_iree_plugin_standalone
  PUBLIC
    "${MERLIN_XPU_RT_SOURCE_DIR}"
    # GENEX_EVAL is required here because some IREE targets' include dirs
    # contain nested generator expressions.
    $<GENEX_EVAL:$<TARGET_PROPERTY:${_XPURT_IREE_IMPL_TARGET},INTERFACE_INCLUDE_DIRECTORIES>>
)
# Forward link libs from IREE unified (e.g. pthread, dl) so the runner only
# needs this + system.
target_link_libraries(
  xpurt_iree_plugin_standalone
  PUBLIC
    # INTERFACE_LINK_LIBRARIES of the unified target may itself contain
    # generator expressions (including $<GENEX_EVAL:...>) which must be
    # evaluated before being used as link items on this target.
    $<GENEX_EVAL:$<TARGET_PROPERTY:${_XPURT_IREE_UNIFIED_TARGET},INTERFACE_LINK_LIBRARIES>>
)
# Output: libxpurt_iree_plugin_standalone.a (single .a for runner to link +
# system libs).
message(
  STATUS
    "xpurt standalone: added xpurt_iree_plugin_standalone (combined plugin + IREE runtime)"
)
