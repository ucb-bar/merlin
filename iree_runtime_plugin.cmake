# Merlin runtime plugin entrypoint used by IREE plugin CMake integration.
# Keep this intentionally minimal and only register Merlin-owned sample/runtime
# targets from this repository.

add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/samples" "merlin-samples")
#add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/benchmarks" "merlin-benchmarks")