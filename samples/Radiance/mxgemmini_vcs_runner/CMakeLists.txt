# =============================================================================
# --- mxGemmini-on-VCS: bare-metal runner for the mlp_3layer fixtures ---
# =============================================================================
# Validates the dialect-driven mxGemmini codegen by loading the .vmfb produced
# by `./merlin compile --target gemmini_mx_vcs[_fp4]` through the IREE
# local-sync HAL driver and invoking `module.mlp_3layer`. The dispatch ELF
# inside the .vmfb embeds volatile MMIO stores at GEMMINI_CTRL=0x40084000 +
# {0x10, 0x18, 0x00} produced by the gemmini-lower-intr-to-mmio pass.
#
# Two cmake targets are produced: bench_gemmini_mx_vcs_mlp_fp8  — uses
# gemmini_mx_vcs.yaml (FP8_0) bench_gemmini_mx_vcs_mlp_fp4  — uses
# gemmini_mx_vcs.yaml (hw: VCS_FP4) (FP4)
#
# Both link the same C runner, distinguished only by which embedded .vmfb is
# .incbin'd. The same deterministic 1x16 i8 input from
# tests/integration/gemmini_mx_vcs/fixtures/test_pattern.h is used.
#
# Usage: ./merlin build --profile firesim --cmake-target
# bench_gemmini_mx_vcs_mlp_fp8 ./merlin build --profile firesim --cmake-target
# bench_gemmini_mx_vcs_mlp_fp4

# Locate host iree-compile (same lookup as gemmini_spike).
if(DEFINED ENV{MERLIN_HOST_IREE_COMPILE})
  set(_MX_HOST_COMPILE "$ENV{MERLIN_HOST_IREE_COMPILE}")
elseif(EXISTS
       "${IREE_CMAKE_PLUGIN_PATHS}/build/host-merlin-debug/tools/iree-compile")
  set(_MX_HOST_COMPILE
      "${IREE_CMAKE_PLUGIN_PATHS}/build/host-merlin-debug/tools/iree-compile")
elseif(
  EXISTS
  "${IREE_CMAKE_PLUGIN_PATHS}/build/host-merlin-release/tools/iree-compile")
  set(_MX_HOST_COMPILE
      "${IREE_CMAKE_PLUGIN_PATHS}/build/host-merlin-release/tools/iree-compile")
else()
  message(
    WARNING
      "Skipping bench_gemmini_mx_vcs_mlp_*: no host iree-compile found. "
      "Run `./merlin build --profile gemmini --cmake-target iree-compile` first, "
      "or set MERLIN_HOST_IREE_COMPILE.")
  set(_MX_HOST_COMPILE "")
endif()

set(_MX_FIXTURE_FP8
    "${IREE_CMAKE_PLUGIN_PATHS}/tests/integration/gemmini_mx_vcs/fixtures/matmul_16x32x16_fp8.mlir"
)
set(_MX_FIXTURE_FP4
    "${IREE_CMAKE_PLUGIN_PATHS}/tests/integration/gemmini_mx_vcs/fixtures/mlp_3layer_fp4.mlir"
)
set(_MX_TEST_PATTERN_INC
    "${IREE_CMAKE_PLUGIN_PATHS}/tests/integration/gemmini_mx_vcs/fixtures")

# Common iree-compile flags (mirror models/gemmini_mx_vcs.yaml plus the
# embedded-elf flag the bare-metal loader requires).
set(_MX_COMPILE_COMMON_FLAGS
    --iree-hal-target-device=local
    --iree-hal-local-target-device-backends=llvm-cpu
    --iree-llvmcpu-target-triple=riscv64-unknown-elf
    --iree-llvmcpu-target-abi=lp64d
    --iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c
    --iree-llvmcpu-target-vector-width-in-bytes=8
    --iree-opt-data-tiling=false
    --iree-dispatch-creation-data-tiling=false
    --iree-llvmcpu-enable-ukernels=none
    --iree-llvmcpu-link-ukernel-bitcode=false
    --iree-execution-model=async-external
    --iree-llvmcpu-link-embedded=true
    --iree-vm-bytecode-module-strip-source-map=true
    --iree-vm-emit-polyglot-zip=false
    --iree-plugin=gemmini
    --iree-gemmini-enable
    --iree-gemmini-lower-back-to-iree=false
    --iree-gemmini-command-issue=mmio
    --iree-gemmini-mmio-base=0x40084000
    --iree-gemmini-elem-bits=8
    --iree-gemmini-acc-bits=32)

# -----------------------------------------------------------------------------
# Helper: produce one bench binary for a given format.
# -----------------------------------------------------------------------------
function(_mx_make_bench bench_name fixture mx_format)
  if(NOT _MX_HOST_COMPILE)
    return()
  endif()
  # Optional shape override via cmake variable: pass
  # MX_BATCH/MX_IN_DIM/MX_OUT_DIM as -D defines on the runner compile. If unset,
  # runner uses 16x16x16 defaults.
  set(_shape_defs)
  if(DEFINED MX_BATCH)
    list(APPEND _shape_defs "MX_BATCH=${MX_BATCH}")
  endif()
  if(DEFINED MX_IN_DIM)
    list(APPEND _shape_defs "MX_IN_DIM=${MX_IN_DIM}")
  endif()
  if(DEFINED MX_OUT_DIM)
    list(APPEND _shape_defs "MX_OUT_DIM=${MX_OUT_DIM}")
  endif()

  set(_vmfb "${CMAKE_CURRENT_BINARY_DIR}/${bench_name}.embedded.vmfb")
  set(_embed_s "${CMAKE_CURRENT_BINARY_DIR}/${bench_name}_embed.S")
  set(_embed_o "${CMAKE_CURRENT_BINARY_DIR}/${bench_name}_embed.o")

  add_custom_command(
    OUTPUT "${_vmfb}"
    COMMAND "${_MX_HOST_COMPILE}" "${fixture}" -o "${_vmfb}"
            ${_MX_COMPILE_COMMON_FLAGS} "--iree-gemmini-mx-format=${mx_format}"
    DEPENDS "${fixture}"
    COMMENT
      "[mxgemmini-vcs] compiling ${fixture} → embedded VMFB (${mx_format})")

  set(MXGEMMINI_VCS_VMFB_PATH "${_vmfb}")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/mxgemmini_vmfb_embed.S.in"
                 "${_embed_s}" @ONLY)

  add_custom_command(
    OUTPUT "${_embed_o}"
    COMMAND
      ${CMAKE_C_COMPILER} --target=riscv64-unknown-elf
      "--sysroot=${CMAKE_SYSROOT}" -march=rv64imafdc -mabi=lp64d -mcmodel=medany
      -c "${_embed_s}" -o "${_embed_o}"
    DEPENDS "${_vmfb}" "${_embed_s}"
    COMMENT "[mxgemmini-vcs] embedding ${bench_name} VMFB via .incbin")
  add_custom_target(${bench_name}_vmfb_obj DEPENDS "${_embed_o}")

  iree_cc_binary(
    NAME
    ${bench_name}
    SRCS
    "mxgemmini_vcs_runner.c"
    DEPS
    iree::base
    iree::hal
    iree::hal::drivers::local_sync::sync_driver
    iree::hal::local
    iree::hal::local::loaders::embedded_elf_loader
    iree::modules::hal
    iree::vm
    iree::vm::bytecode::module)

  set(_iree_target
      "iree_.._.._samples_Radiance_mxgemmini_vcs_runner_${bench_name}")
  target_link_libraries("${_iree_target}" PRIVATE "${_embed_o}")
  target_include_directories("${_iree_target}"
                             PRIVATE "${_MX_TEST_PATTERN_INC}")
  if(_shape_defs)
    target_compile_definitions("${_iree_target}" PRIVATE ${_shape_defs})
  endif()
  add_dependencies("${_iree_target}" ${bench_name}_vmfb_obj)
endfunction()

_mx_make_bench(bench_gemmini_mx_vcs_mlp_fp8 "${_MX_FIXTURE_FP8}" "fp8_0")
_mx_make_bench(bench_gemmini_mx_vcs_mlp_fp4 "${_MX_FIXTURE_FP4}" "fp4")
