# Dispatch level dual-model runtime sample:
# - Dronet runs at max throughput
# - MLP runs at a fixed periodic frequency
# - Both modules are loaded into the same IREE runtime session

set(_NAME "merlin_benchmark_baseline_dual_model_async")
add_executable(${_NAME} "")
target_sources(${_NAME}
  PRIVATE
    main.c
    runtime_dispatch_graph.h
    runtime_dispatch_graph.cc
)

set_target_properties(${_NAME} PROPERTIES
  OUTPUT_NAME "benchmark-baseline-dual-model-async-run"
)

target_compile_features(${_NAME}
  PRIVATE
    c_std_11
    cxx_std_17
)

find_package(Threads REQUIRED)

if(DEFINED IREE_DEFAULT_COPTS)
  target_compile_options(${_NAME} PRIVATE ${IREE_DEFAULT_COPTS})
endif()

target_link_libraries(${_NAME}
  PRIVATE
    iree_runtime_runtime
    Threads::Threads
)
