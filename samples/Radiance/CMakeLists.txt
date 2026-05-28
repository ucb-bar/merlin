# Radiance / mxGemmini sample subdirectories.
#
# These samples are built by the firesim profile (cross-toolchain RV64 baremetal
# newlib). The mxGemmini VCS bench compiles its embedded VMFB with the host
# iree-compile + the gemmini plugin in MMIO mode, then links the produced .vmfb
# (via .incbin) into a bare-metal ELF that
# simv-chipyard.harness-RadianceGemminiOnlyConfig can load.
#
# Built only when MERLIN_BUILD_SATURN_OPU is set so the IREE bare-metal runtime
# archives are available; the firesim profile turns this on by default.

if(MERLIN_BUILD_SATURN_OPU)
  add_subdirectory(mxgemmini_vcs_runner)
endif()
