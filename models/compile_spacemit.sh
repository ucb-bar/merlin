#!/bin/bash

# ==============================================================================
# Configuration & Setup
# ==============================================================================

# Exit on error, undefined vars, and pipe failures
set -euo pipefail

# ------------------------------------------------------------------------------
# 1. Selection: Define Models and Configurations
# ------------------------------------------------------------------------------

# Models to process (folder names)
TARGET_MODELS=("diffusion") 

# Configurations to build (comment out the ones you don't want)
# Options are "NPU", "RVV", "SCALAR"
TARGET_CONFIGS=("RVV")

# Global Quantization Switch
# "true"  = Look for $MODEL.q.int8.onnx and append _quant to output folder
# "false" = Use standard $MODEL.onnx
USE_QUANTIZED="false"

# ------------------------------------------------------------------------------
# 2. Toolchain Setup
# ------------------------------------------------------------------------------

# Allow overriding IREE_TOOL_DIR via environment variable
IREE_TOOL_DIR="${IREE_TOOL_DIR:-/scratch2/agustin/merlin/build-iree-host-deb-tracy/tools}"
COMPILE_TOOL="$IREE_TOOL_DIR/iree-compile"
IMPORT_TOOL="iree-import-onnx"

# Check if tools exist
if [ ! -f "$COMPILE_TOOL" ]; then
    echo "❌ Error: iree-compile not found at $COMPILE_TOOL"
    echo "   Please set IREE_TOOL_DIR environment variable to your build path."
    exit 1
fi

# Base directory is where this script is located
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="$BASE_DIR/compiled_models"

mkdir -p "$OUTPUT_ROOT"

# ==============================================================================
# 3. Flag Definitions
# ==============================================================================

# --- Base Flags (Common to ALL configurations) ---
BASE_FLAGS=(
    "--iree-hal-target-device=local"
    "--iree-hal-local-target-device-backends=llvm-cpu"
    "--iree-llvmcpu-target-triple=riscv64-unknown-linux-gnu"
    "--iree-llvmcpu-target-abi=lp64d"
    "--iree-dispatch-creation-data-tiling"
    #"--iree-global-opt-propagate-transposes=true"
    "--iree-opt-level=O3"
    "--iree-opt-data-tiling"
    #"--iree-vm-bytecode-module-strip-source-map=true"

)

# NOT IN USE YET 
# TODO: Integrate these into the compilation flags based on quantization
BASE_QUANT_FLAGS=(
    "--iree-global-opt-enable-quantized-matmul-reassociation"
    "--iree-global-opt-enable-quantized-matmul-reassociation"
    "--iree-opt-generalize-matmul=true"
    "--iree-llvmcpu-general-matmul-tile-bytes=262144"
    "--iree-llvmcpu-narrow-matmul-tile-bytes=32768"
    "--iree-llvmcpu-skip-intermediate-roundings"
)

# --- Configuration Specific Flags ---

# 1. NPU: Vectorized + Ukernels
NPU_FLAGS=(
    "${BASE_FLAGS[@]}"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c,+v,+zvl256b,+zfh,+zba,+zbb,+zbc,+zbs,+zicbom,+zicboz,+zicbop,+zihintpause"
    #"--iree-llvmcpu-link-embedded=false"
    "--iree-llvmcpu-target-vector-width-in-bytes=32"
    "--iree-llvmcpu-loop-vectorization=true"
    #"--riscv-v-fixed-length-vector-lmul-max=8"
    "--iree-llvmcpu-enable-ukernels=all"
    "--iree-llvmcpu-link-ukernel-bitcode=true"
)

# 2. RVV: Vectorized + NO Ukernels
RVV_FLAGS=(
    "${BASE_FLAGS[@]}"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c,+v,+zvl256b,+zfh,+zba,+zbb,+zbc,+zbs,+zicbom,+zicboz,+zicbop,+zihintpause"
    #"--iree-llvmcpu-link-embedded=false"
    "--iree-llvmcpu-target-vector-width-in-bytes=32"
    "--iree-llvmcpu-loop-vectorization=true"
    #"--riscv-v-fixed-length-vector-lmul-max=8"
    "--iree-llvmcpu-enable-ukernels=all"
)

# 3. SCALAR: Minimal Vectorization + NO Ukernels
SCALAR_FLAGS=(
    "${BASE_FLAGS[@]}"
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c"
    "--iree-llvmcpu-enable-ukernels=none"
)

# ==============================================================================
# Main Execution Loop
# ==============================================================================

for MODEL in "${TARGET_MODELS[@]}"; do

    # ----------------------------------------------------------------------
    # Determine Input Files based on Quantization Switch
    # ----------------------------------------------------------------------
    if [ "$USE_QUANTIZED" == "true" ]; then
        SOURCE_ONNX="$BASE_DIR/$MODEL/$MODEL.q.int8.onnx"
        MODEL_SUFFIX="_quant"
        MODE_MSG="Quantized (Int8)"
    else
        SOURCE_ONNX="$BASE_DIR/$MODEL/$MODEL.onnx"
        MODEL_SUFFIX=""
        MODE_MSG="Float (FP32)"
    fi
    
    # Define source MLIR location (if skipping ONNX import)
    SOURCE_MLIR="$BASE_DIR/${MODEL}/${MODEL}${MODEL_SUFFIX}.mlir"

    for CONFIG in "${TARGET_CONFIGS[@]}"; do
        
        echo "################################################################################"
        echo "🚀 Processing Model: $MODEL | Config: spacemit_$CONFIG | Mode: $MODE_MSG"
        echo "################################################################################"

        # ----------------------------------------------------------------------
        # Determine Flags for current Config
        # ----------------------------------------------------------------------
        case "$CONFIG" in
            "NPU")    CURRENT_FLAGS=("${NPU_FLAGS[@]}") ;;
            "RVV")    CURRENT_FLAGS=("${RVV_FLAGS[@]}") ;;
            "SCALAR") CURRENT_FLAGS=("${SCALAR_FLAGS[@]}") ;;
            *)        echo "❌ Error: Unknown configuration '$CONFIG'"; exit 1 ;;
        esac

        # ----------------------------------------------------------------------
        # Setup Output Directories (Must happen inside CONFIG loop)
        # ----------------------------------------------------------------------
        OUTPUT_DIR="$OUTPUT_ROOT/$MODEL/spacemit_${CONFIG}${MODEL_SUFFIX}"
        MLIR_OUTPUT="$OUTPUT_DIR/${MODEL}${MODEL_SUFFIX}.mlir"
        VMFB_OUTPUT="$OUTPUT_DIR/${MODEL}${MODEL_SUFFIX}.vmfb"
        GRAPH_OUT="$OUTPUT_DIR/${MODEL}${MODEL_SUFFIX}_dispatch_graph.dot"
        
        mkdir -p "$OUTPUT_DIR"

        # Host flags for dumping artifacts specific to this run
        HOST_FLAGS=(
            "${CURRENT_FLAGS[@]}"
            "--iree-hal-dump-executable-sources-to=$OUTPUT_DIR/sources/"
            "--iree-hal-dump-executable-files-to=$OUTPUT_DIR/files/"
            "--iree-hal-dump-executable-binaries-to=$OUTPUT_DIR/binaries/"
            "--iree-hal-dump-executable-configurations-to=$OUTPUT_DIR/configs/"
            "--iree-hal-dump-executable-benchmarks-to=$OUTPUT_DIR/benchmarks/"
            "--dump-compilation-phases-to=$OUTPUT_DIR/phases/"
        )

        # ----------------------------------------------------------------------
        # 4. Input Handling (ONNX -> MLIR)
        # ----------------------------------------------------------------------
        if [ -f "$MLIR_OUTPUT" ]; then
            echo "  ℹ️  MLIR file already exists at $MLIR_OUTPUT. Skipping import."
        elif [ -f "$SOURCE_ONNX" ]; then
            echo "  found ONNX file: $SOURCE_ONNX"
            
            if ! command -v "$IMPORT_TOOL" &> /dev/null; then
                echo "❌ Error: $IMPORT_TOOL not found. Activate your python venv."
                exit 1
            fi

            echo "  Importing ONNX to MLIR..."
            "$IMPORT_TOOL" "$SOURCE_ONNX" --opset-version 17 -o "$MLIR_OUTPUT"
        elif [ -f "$SOURCE_MLIR" ]; then
            echo "  found Source MLIR file (no ONNX): $SOURCE_MLIR"
            echo "  Copying to output directory..."
            cp "$SOURCE_MLIR" "$MLIR_OUTPUT"
        else
            echo "❌ Error: Could not find $MODEL.onnx or $MODEL.mlir in $BASE_DIR/$MODEL/"
            continue
        fi

        # ----------------------------------------------------------------------
        # 5. Compile Main Model (VMFB)
        # ----------------------------------------------------------------------
        
        # Add graph output flag specifically for this run
        COMPILE_FLAGS_WITH_GRAPH=(
            "${HOST_FLAGS[@]}"
            "--iree-flow-dump-dispatch-graph"
            "--iree-flow-dump-dispatch-graph-output-file=$GRAPH_OUT"
        )
        
        echo "  Compiling main model ($CONFIG)..."
        "$COMPILE_TOOL" "$MLIR_OUTPUT" \
            -o "$VMFB_OUTPUT" \
            "${COMPILE_FLAGS_WITH_GRAPH[@]}"

        echo "✅ Successfully compiled: $VMFB_OUTPUT"

        # ----------------------------------------------------------------------
        # 6. Compile Dispatches
        # ----------------------------------------------------------------------
        echo "  Compiling individual dispatch sources..."
        
        SOURCES_DIR="$OUTPUT_DIR/benchmarks"
        VMFB_DIR="$SOURCES_DIR/vmfb"

        if [ -d "$SOURCES_DIR" ]; then
            mkdir -p "$VMFB_DIR"
            
            # Use process substitution to avoid subshell variable issues if needed
            find "$SOURCES_DIR" -name "*.mlir" | sort -V | while read -r mlir_file; do
                filename=$(basename -- "$mlir_file")
                
                output_vmfb_name="${filename%.mlir}.vmfb"
                output_vmfb_path="$VMFB_DIR/$output_vmfb_name"
                
                echo "    Compiling $filename -> $output_vmfb_name"
                "$COMPILE_TOOL" "$mlir_file" -o "$output_vmfb_path" "${CURRENT_FLAGS[@]}"
            done
            
            # ------------------------------------------------------------------
            # 7. Zip Results
            # ------------------------------------------------------------------
            echo "  Zipping benchmark artifacts for $CONFIG..."
            
            ZIP_NAME="${MODEL}_spacemit_${CONFIG}${MODEL_SUFFIX}_benchmarks.zip"
            ZIP_PATH="$OUTPUT_DIR/$ZIP_NAME"
            
            # Check if VMFB files were actually created
            if ls "$VMFB_DIR"/*.vmfb >/dev/null 2>&1; then
                zip -j "$ZIP_PATH" "$SOURCES_DIR"/*.mlir "$VMFB_DIR"/*.vmfb
                echo "✅ Created Flattened Archive: $ZIP_PATH"
            else
                echo "⚠️  No VMFB files were generated to zip."
            fi
            
        else
            echo "⚠️  No benchmark directory found at $SOURCES_DIR"
        fi

        echo "✅ Completed $MODEL [spacemit_$CONFIG]"
        echo "=========================================="
    done
done

echo "=========================================="
echo "🎉 All Models and Configurations Processed."