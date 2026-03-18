# Generic utilities — no IREE dependency.
#
# Usage: target_link_libraries(<target> PRIVATE merlin_core) Then:  #include
# "core/stats.h"  etc.

add_library(merlin_core INTERFACE)

target_include_directories(merlin_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
