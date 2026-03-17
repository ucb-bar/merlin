# Common header-only utilities for IREE benchmark samples.
#
# Usage: target_link_libraries(<your_target> PRIVATE merlin_iree_bench_common)
# Then:  #include "iree_bench/stats.h"  etc.

add_library(merlin_iree_bench_common INTERFACE)

target_include_directories(merlin_iree_bench_common
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(merlin_iree_bench_common INTERFACE iree_runtime_runtime)
