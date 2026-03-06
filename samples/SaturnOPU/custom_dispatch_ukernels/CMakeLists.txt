# CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(RiscVGemmTest C)

# --- 1. Include Plugin Subdirectory ---

add_subdirectory(dummy_kernel)

# --- Find IREE ---
if(NOT DEFINED IREE_HOST_BIN_DIR)
  message(FATAL_ERROR "IREE_HOST_BIN_DIR is not set. Please re-run cmake with: -DIREE_HOST_BIN_DIR=/path/to/iree/install/bin")
endif()
find_program(IREE_COMPILE_EXECUTABLE iree-compile HINTS "${IREE_HOST_BIN_DIR}" REQUIRED)
message(STATUS "Using iree-compile: ${IREE_COMPILE_EXECUTABLE}")




# --- 2. Configure Model Compilation ---
set(IREE_COMPILE_FLAGS
  --iree-hal-target-backends=llvm-cpu 
  --iree-llvmcpu-target-cpu=generic-rv64
  --iree-llvmcpu-link-static=false
  --iree-llvmcpu-target-triple=riscv64-linux-gnu 
  --iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+v,+zvl512b,+zvbb
  --iree-llvmcpu-target-abi=lp64d
  --iree-dispatch-creation-data-tiling
  -mlir-disable-threading
  --dump-compilation-phases-to=/scratch2/agustin/merlin/samples/custom_dispatch_ukernels_saturn/dummy_kernel/compilation_phases
)

set(SOURCE_MODEL_MLIR "${CMAKE_CURRENT_SOURCE_DIR}/compilation_phases_fc/model_quantized_ort.mlir")

# --- 3. Custom Command to Compile the CUSTOM DISPATCH Model ---
set(COMPILED_MODEL_CUSTOM_FILENAME "qgemm_custom_riscv64.vmfb")
set(COMPILED_MODEL_CUSTOM "${CMAKE_CURRENT_BINARY_DIR}/${COMPILED_MODEL_CUSTOM_FILENAME}")

message(STATUS "Custom Spec MLIR: ${CUSTOM_SPEC_MLIR}")
message(STATUS "Source Model MLIR: ${SOURCE_MODEL_MLIR}")
message(STATUS "Current Binary Dir: ${CMAKE_CURRENT_BINARY_DIR}")

add_custom_command(
  OUTPUT ${COMPILED_MODEL_CUSTOM}
  COMMAND ${IREE_COMPILE_EXECUTABLE}
          ${IREE_COMPILE_FLAGS}
          # where to find "riscv_qgemm.o"
          --iree-hal-executable-object-search-path=${CMAKE_CURRENT_BINARY_DIR}/dummy_kernel/
          # un the match-and-replace
          --iree-preprocessing-transform-spec-filename=${CUSTOM_SPEC_MLIR}
          # input graph
          ${SOURCE_MODEL_MLIR}
          # output
          -o ${COMPILED_MODEL_CUSTOM}
  DEPENDS 
    ${IREE_COMPILE_EXECUTABLE} 
    ${CUSTOM_OP_OBJECT_FILE}  # Depends on the .o file
    ${SOURCE_MODEL_MLIR} 
    ${CUSTOM_SPEC_MLIR}       # Depends on the spec file
  VERBATIM
  COMMENT "Compiling CUSTOM DISPATCH model for riscv64"
)
add_custom_target(compile_custom_model DEPENDS ${COMPILED_MODEL_CUSTOM})


# --- 4. Build the Final Executable ---
set(_NAME "run_model")
add_executable(${_NAME} "main.c")
target_compile_features(${_NAME} PRIVATE c_std_17)

# --- 5. Link Libraries ---
target_link_libraries(${_NAME} PRIVATE iree_runtime_runtime) # (or however you link IREE)

# --- 6. Configure Executable for Custom Dispatch ---
target_compile_definitions(${_NAME} PRIVATE
  MODEL_VMFB_FILENAME="${COMPILED_MODEL_CUSTOM_FILENAME}"
)
add_dependencies(${_NAME} compile_custom_model)