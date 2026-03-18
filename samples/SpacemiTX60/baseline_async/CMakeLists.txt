set(_NAME "merlin_baseline_async")
add_executable(${_NAME} "")
target_sources(${_NAME} PRIVATE main.c runtime_dispatch_graph.h
                                runtime_dispatch_graph.cc)

set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "merlin-baseline-async")

target_compile_features(${_NAME} PRIVATE c_std_11 cxx_std_17)

find_package(Threads REQUIRED)

if(DEFINED IREE_DEFAULT_COPTS)
  target_compile_options(${_NAME} PRIVATE ${IREE_DEFAULT_COPTS})
endif()

target_link_libraries(${_NAME} PRIVATE merlin_dispatch Threads::Threads)
