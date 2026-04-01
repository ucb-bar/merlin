# merlin/samples/CMakeLists.txt

add_subdirectory(common/core)
add_subdirectory(common/runtime)
add_subdirectory(common/dispatch)

if(MERLIN_BUILD_SPACEMITX60)
  message(STATUS "Including Merlin SpacemiTX60 samples")
  add_subdirectory(SpacemiTX60)
endif()

if(MERLIN_BUILD_SATURN_OPU)
  message(STATUS "Including Merlin SaturnOPU samples")
  add_subdirectory(SaturnOPU)
endif()
