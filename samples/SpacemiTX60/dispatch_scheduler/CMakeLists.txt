set(_NAME "merlin_dispatch_scheduler")
add_executable(${_NAME} "")

target_sources(${_NAME} PRIVATE main.c runtime_scheduler.h runtime_scheduler.cc)

set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME
                                          "merlin-dispatch-scheduler")

target_compile_features(${_NAME} PRIVATE c_std_11 cxx_std_17)

find_package(Threads REQUIRED)

if(DEFINED IREE_DEFAULT_COPTS)
  target_compile_options(${_NAME} PRIVATE ${IREE_DEFAULT_COPTS})
endif()

target_link_libraries(${_NAME} PRIVATE merlin_dispatch Threads::Threads)
