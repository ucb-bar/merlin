cmake_minimum_required(VERSION 3.16)
project(merlin_timing VERSION 0.1.0 LANGUAGES CXX)

add_library(merlin_timing INTERFACE)
target_compile_features(merlin_timing INTERFACE cxx_std_17)
target_include_directories(merlin_timing INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

include(CTest)
if(BUILD_TESTING)
  add_executable(merlin_timing_test tests/test_engine.cpp)
  target_link_libraries(merlin_timing_test PRIVATE merlin_timing)
  add_test(NAME merlin_timing_core COMMAND merlin_timing_test)
  add_executable(merlin_timing_reservations_test tests/test_reservations.cpp)
  target_link_libraries(merlin_timing_reservations_test PRIVATE merlin_timing)
  add_test(NAME merlin_timing_reservations COMMAND merlin_timing_reservations_test)
endif()

install(DIRECTORY include/ DESTINATION include)
install(TARGETS merlin_timing EXPORT merlin_timing_targets)
install(EXPORT merlin_timing_targets NAMESPACE merlin_timing:: DESTINATION lib/cmake/merlin_timing)
