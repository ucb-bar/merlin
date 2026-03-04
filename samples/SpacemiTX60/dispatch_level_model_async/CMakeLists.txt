# Dispatch-level model runtime sample:
# - Executes per-dispatch VMFBs based on dependencies described in a JSON file
# - Copies the JSON + dispatch VMFB payloads next to the produced binary

set(_NAME "merlin_dispatch_level_model_async_run")
add_executable(${_NAME} "")
target_sources(${_NAME}
  PRIVATE
    main.c
)

set_target_properties(${_NAME} PROPERTIES
  OUTPUT_NAME "dispatch-level-model-async-run"
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

# Copy dispatch graph JSON + VMFBs next to the built binary so it can be run
# directly from the build tree.
set(_DISPATCH_JSON "${CMAKE_CURRENT_SOURCE_DIR}/dronet.q.int8_dispatch_graph.json")
set(_DISPATCH_VMFBS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dispatches")

add_custom_command(TARGET ${_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${_DISPATCH_JSON}"
          "$<TARGET_FILE_DIR:${_NAME}>/dronet.q.int8_dispatch_graph.json"
  COMMAND ${CMAKE_COMMAND} -E make_directory
          "$<TARGET_FILE_DIR:${_NAME}>/dispatches"
  # NOTE: We copy the whole directory instead of individual VMFB files because
  # Ninja uses '$' for variable expansion and these dispatch VMFB filenames
  # contain literal '$' (e.g. module_dronet$async_dispatch_0_...vmfb).
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${_DISPATCH_VMFBS_DIR}"
          "$<TARGET_FILE_DIR:${_NAME}>/dispatches"
)
