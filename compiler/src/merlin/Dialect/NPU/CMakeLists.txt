if(NOT (MERLIN_ENABLE_TARGET_NPU OR MERLIN_BUILD_NPU))
  return()
endif()

add_subdirectory(IR)
add_subdirectory(Transforms)
add_subdirectory(Translation)
