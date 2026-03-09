# Baseline dual-model runtime sample:
# - Dronet runs at max throughput
# - MLP runs at a fixed periodic frequency
# - Both modules are loaded into the same IREE runtime session

set(_NAME "merlin_v2_2_model_async")
add_executable(${_NAME} "")
target_sources(${_NAME}
  PRIVATE
    main.c
    runtime_scheduler.h
    runtime_scheduler.cc
)

set_target_properties(${_NAME} PROPERTIES
  OUTPUT_NAME "v2-dual-model-async-run"
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
