# Standalone xpu-rt static library: xpu-rt runners + IREE runtime in one .a
#
# This file is included from IREE's runtime/src/iree/runtime/CMakeLists.txt
# after iree_runtime_impl and iree_runtime_unified are defined, so we can
# reference INTERFACE_IREE_TRANSITIVE_OBJECTS and build a single archive that
# contains both the xpu-rt runner objects and the full IREE runtime.
#
# Downstream consumers (XPU-RT repo, Zephyr, etc.) link only this archive plus
# system libs:
#
# target_link_libraries(my_app PRIVATE -Wl,--whole-archive
# /path/to/libxpurt_standalone.a -Wl,--no-whole-archive Threads::Threads
# ${CMAKE_DL_LIBS} m)
#
# For Zephyr integration, point XPURT_STANDALONE_LIB_PATH at the archive
# produced by a cross-compilation merlin build and use Zephyr's
# zephyr_library_import_from_static() or target_link_libraries().

if(NOT TARGET xpurt_objs)
  message(STATUS "xpurt standalone: xpurt_objs not found, skipping")
  return()
endif()

# IREE target naming: prefer the namespaced targets but accept the raw ones.
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

set(_XPURT_OBJS "$<TARGET_OBJECTS:xpurt_objs>")
set(_IREE_OBJS
    "$<REMOVE_DUPLICATES:$<GENEX_EVAL:$<TARGET_PROPERTY:${_XPURT_IREE_IMPL_TARGET},INTERFACE_IREE_TRANSITIVE_OBJECTS>>>"
)

add_library(xpurt_standalone STATIC ${_XPURT_OBJS} ${_IREE_OBJS})
target_include_directories(
  xpurt_standalone
  PUBLIC
    "${MERLIN_XPU_RT_SOURCE_DIR}/.."
    # GENEX_EVAL is required here because some IREE targets' include dirs
    # contain nested generator expressions.
    $<GENEX_EVAL:$<TARGET_PROPERTY:${_XPURT_IREE_IMPL_TARGET},INTERFACE_INCLUDE_DIRECTORIES>>
)
# Forward link libs from IREE unified (e.g. pthread, dl) so downstream consumers
# only need this + system.
target_link_libraries(
  xpurt_standalone
  PUBLIC
    $<GENEX_EVAL:$<TARGET_PROPERTY:${_XPURT_IREE_UNIFIED_TARGET},INTERFACE_LINK_LIBRARIES>>
)
# Output: libxpurt_standalone.a
message(
  STATUS
    "xpurt standalone: added xpurt_standalone (combined xpu-rt + IREE runtime)")
