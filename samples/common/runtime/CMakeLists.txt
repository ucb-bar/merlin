# IREE runtime utilities — module caching, pinned devices, fatal state.
#
# Usage: target_link_libraries(<target> PRIVATE merlin_runtime) Then:  #include
# "runtime/module_cache.h"  etc.

add_library(merlin_runtime INTERFACE)

target_link_libraries(merlin_runtime INTERFACE merlin_core iree_runtime_runtime)

# IREE internal headers (task API, local-task driver) for pinned device
# creation.
if(DEFINED IREE_SOURCE_DIR)
  target_include_directories(
    merlin_runtime INTERFACE ${IREE_SOURCE_DIR}/runtime/src
                             ${IREE_BINARY_DIR}/runtime/src)
endif()
