# Full generated SmolVLA host, using glossy startup and the penguin26 console.
set(PENGUIN26_SMOLVLA_PACKAGE "" CACHE PATH "SmolVLA package containing source/")
set(PENGUIN26_SMOLVLA_BLOBS "" CACHE PATH "Extracted SmolVLA asset blobs")
set(PENGUIN26_SMOLVLA_PHASE "" CACHE STRING "Self-check one captured phase (0 through 199); empty runs the full model")
if (NOT PENGUIN26_SMOLVLA_PACKAGE)
  return()
endif()
if (NOT LINKER STREQUAL "smolvla")
  message(FATAL_ERROR "atlas-delivery-smolvla requires -DLINKER=smolvla")
endif()
if (NOT IS_DIRECTORY "${PENGUIN26_SMOLVLA_BLOBS}")
  message(FATAL_ERROR "Set PENGUIN26_SMOLVLA_BLOBS to the extracted blob directory")
endif()
if (NOT PENGUIN26_SMOLVLA_PHASE STREQUAL "")
  if (NOT PENGUIN26_SMOLVLA_PHASE MATCHES "^(0|[1-9][0-9]?|1[0-9][0-9])$")
    message(FATAL_ERROR "PENGUIN26_SMOLVLA_PHASE must be empty or an integer from 0 through 199")
  endif()
endif()

set(_smol_src "${PENGUIN26_SMOLVLA_PACKAGE}/source")
# Exact C translation units from the generator's source/build.sh; start.S is
# deliberately replaced by glossy, and atlas_resnet_graph_math.c is not used.
set(_smol_c
  main.c
  smol_host_ops.c smol_host_attention.c smol_host_softmax.c smol_host_rope.c
  smol_host_controls.c host_matrix_fp8pack.c
  smol_policy_ops.c smol_policy_block.c smol_policy_graph.c
  smol_vision_ops.c smol_vision_graph.c smol_policy_raw.c
  smol_full_driver.c smol_vision_attention.c atlas_host.c
  support/rv64_support.c)
list(TRANSFORM _smol_c PREPEND "${_smol_src}/")

add_executable(atlas-delivery-smolvla smolvla_main.c ${_smol_c} "${_smol_src}/assets.S")
target_compile_definitions(atlas-delivery-smolvla PRIVATE
  ATLAS_WORKLOAD_RETURN_TO_CALLER=1 ATLAS_RESULT_USE_PLATFORM_CONSOLE=1
  ATLAS_SMOL_LAUNCH_HOST_TIMING=1)
if (NOT PENGUIN26_SMOLVLA_PHASE STREQUAL "")
  target_compile_definitions(atlas-delivery-smolvla PRIVATE
    PENGUIN26_SMOLVLA_PHASE=${PENGUIN26_SMOLVLA_PHASE})
endif()
target_include_directories(atlas-delivery-smolvla BEFORE PRIVATE
  "${_smol_src}/support/include" "${_smol_src}")
target_compile_options(atlas-delivery-smolvla PRIVATE
  $<$<COMPILE_LANGUAGE:C>:-std=c11;-O2;-ffp-contract=off;-fno-fast-math;-ffreestanding;-fno-stack-protector;-fno-pic;-ffunction-sections;-fdata-sections;-fno-tree-vectorize;-fno-tree-slp-vectorize;-fno-tree-loop-distribute-patterns>)
# The platform headers use the GNU 'asm' keyword; only this tiny wrapper needs
# GNU syntax. Every generator translation unit retains strict -std=c11.
set_source_files_properties(smolvla_main.c PROPERTIES COMPILE_OPTIONS -std=gnu11)
# Keep glossy/HTIF's tohost/fromhost independent of the workload result.
set_source_files_properties("${_smol_src}/main.c" PROPERTIES
  COMPILE_DEFINITIONS "tohost=smol_tohost;fromhost=smol_fromhost")
set_source_files_properties("${_smol_src}/assets.S" PROPERTIES
  COMPILE_OPTIONS "-Wa,-I,${PENGUIN26_SMOLVLA_BLOBS}")
target_link_options(atlas-delivery-smolvla PRIVATE
  -Wl,--gc-sections "-Wl,-Map,${CMAKE_CURRENT_BINARY_DIR}/atlas-delivery-smolvla.map")
set_property(TARGET atlas-delivery-smolvla APPEND PROPERTY LINK_DEPENDS "${LINKER_SCRIPT}")
target_link_libraries(atlas-delivery-smolvla PRIVATE
  -L${CMAKE_BINARY_DIR}/glossy -Wl,--whole-archive glossy -Wl,--no-whole-archive
  chip-config rocketcore)

option(SMOLVLA_FAST "Skip verification-only staging and use bound immutable B programs" OFF)
set(SMOLVLA_FAST_ASSETS "" CACHE PATH "Directory with ELF-bound smol_fast_assets and smol_fast_images")
if(SMOLVLA_FAST)
  if(NOT EXISTS "${SMOLVLA_FAST_ASSETS}/smol_fast_assets.c" OR NOT EXISTS "${SMOLVLA_FAST_ASSETS}/smol_fast_images.S")
    message(FATAL_ERROR "SMOLVLA_FAST requires freshly generated ELF-bound program assets")
  endif()
  target_sources(atlas-delivery-smolvla PRIVATE "${SMOLVLA_FAST_ASSETS}/smol_fast_assets.c" "${SMOLVLA_FAST_ASSETS}/smol_fast_images.S")
  target_include_directories(atlas-delivery-smolvla PRIVATE "${SMOLVLA_FAST_ASSETS}")
  target_compile_definitions(atlas-delivery-smolvla PRIVATE ATLAS_FAST_MODE=1)
endif()
