# merlin/samples/CMakeLists.txt

add_subdirectory(common/core)
add_subdirectory(common/runtime)
add_subdirectory(common/dispatch)
# samples/common/xpu-rt is only included by SpacemiTX60 builds because the
# baseline_runner.cc / scheduler_runner.cc files use the older IREE
# `iree_hal_driver_create_device_by_id` signature (pre-
# iree_hal_device_create_params_t). They build cleanly against the vendored
# iree_bar from before that refactor; in the SpacemiTX60 release pipeline that
# uses a pinned IREE that's still the case. For QRB5165 / newer IREE we
# currently drive the schedule from a Python-side runner
# (tools/run_schedule_on_board.py).
if(MERLIN_BUILD_XPU_RT_RUNNER)
  add_subdirectory(common/xpu-rt)
  # Target-neutral merlin-dispatch-scheduler binary. Same binary is used by
  # Qualcomm QRB5165, SpacemiT X60, and any other target whose HAL drivers are
  # linked into the build.
  add_subdirectory(common/dispatch_scheduler)
endif()

if(MERLIN_BUILD_QRB5165 OR DEFINED MERLIN_RUNTIME_ENABLE_HAL_QNN)
  add_subdirectory(QRB5165)
endif()

if(MERLIN_BUILD_SPACEMITX60)
  message(STATUS "Including Merlin SpacemiTX60 samples")
  add_subdirectory(SpacemiTX60)
endif()

if(MERLIN_BUILD_SATURN_OPU)
  message(STATUS "Including Merlin SaturnOPU samples")
  add_subdirectory(SaturnOPU)
  # Radiance / mxGemmini-VCS sample. Lives next to SaturnOPU because both target
  # the firesim profile's bare-metal RV64 toolchain. Only included once
  # SaturnOPU is on (so the IREE runtime archives + linker script are
  # available).
  add_subdirectory(Radiance)
endif()
