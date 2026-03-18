# Dispatch scheduling library — graph parsing, types, output, VMFB resolution.
#
# Usage: target_link_libraries(<target> PRIVATE merlin_dispatch) Then:  #include
# "dispatch/dispatch_types.h"  etc.

add_library(merlin_dispatch INTERFACE)

target_link_libraries(merlin_dispatch INTERFACE merlin_runtime merlin_core)
