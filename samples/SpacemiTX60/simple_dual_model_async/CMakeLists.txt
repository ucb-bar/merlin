# Simple dual-model runtime sample:
# - Loads both VMFBs into one IREE session
# - Runs dronet then mlp sequentially (no scheduler/threads/sensors)

set(_NAME "merlin_simple_dual_model_async_run")
add_executable(${_NAME} "")
target_sources(${_NAME}
  PRIVATE
    main.c
)

set_target_properties(${_NAME} PROPERTIES
  OUTPUT_NAME "simple-dual-model-async-run"
)

target_compile_features(${_NAME}
  PRIVATE
    c_std_11
)

if(DEFINED IREE_DEFAULT_COPTS)
  target_compile_options(${_NAME} PRIVATE ${IREE_DEFAULT_COPTS})
endif()

target_link_libraries(${_NAME}
  PRIVATE
    iree_runtime_runtime
)
