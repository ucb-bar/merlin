# Copyright 2026 UCB-BAR
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# ============================================================================
# All Gemmini dialect + passes + plugin registration are built in this single
# CMake scope so that iree_cc_library :: target references resolve correctly.
#
# Source files live under compiler/src/merlin/Dialect/Gemmini/ per the
# Merlin repo convention.  We reference them here using absolute paths.
# ============================================================================

set(_GEMMINI_SRC_DIR "${MERLIN_COMPILER_SOURCE_DIR}/src/merlin/Dialect/Gemmini")

# --- Gemmini Dialect IR Library ---
iree_cc_library(
  NAME
    IR
  SRCS
    "${_GEMMINI_SRC_DIR}/IR/GemminiDialect.cpp"
  DEPS
    MLIRFuncDialect
    MLIRIR
    MLIRLLVMDialect
    MLIRSideEffectInterfaces
  PUBLIC
)

# --- Gemmini Transforms / Passes Library ---
iree_cc_library(
  NAME
    Transforms
  SRCS
    "${_GEMMINI_SRC_DIR}/Transforms/LegalizeForLLVMExport.cpp"
    "${_GEMMINI_SRC_DIR}/Transforms/LowerLinalgToGemmini.cpp"
    "${_GEMMINI_SRC_DIR}/Transforms/LowerGemminiPass.cpp"
    "${_GEMMINI_SRC_DIR}/Transforms/GemminiIRDumps.cpp"
  DEPS
    ::IR
    MLIRArithDialect
    MLIRArithToLLVM
    MLIRBufferizationTransforms
    MLIRFuncDialect
    MLIRFuncToLLVM
    MLIRIR
    MLIRLinalgDialect
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
    MLIRMemRefDialect
    MLIRMemRefToLLVM
    MLIRPass
    MLIRSCFDialect
    MLIRTransforms
    MLIRVectorDialect
  PUBLIC
)

# --- Gemmini Registration Library ---
iree_cc_library(
  NAME
    GemminiRegistration
  SRCS
    "${_GEMMINI_SRC_DIR}/RegisterGemmini.cpp"
  DEPS
    ::IR
    ::Transforms
    MLIRIR
    MLIRPass
  PUBLIC
)

# --- Plugin Entry Point ---
iree_cc_library(
  NAME
    registration
  SRCS
    "PluginRegistration.cpp"
  DEPS
    ::GemminiRegistration
    ::IR
    ::Transforms
    iree::compiler::PluginAPI
    MLIRIR
    MLIRPass
  PUBLIC
)

iree_compiler_register_plugin(
  PLUGIN_ID
    merlin_gemmini
  TARGET
    ::registration
)

iree_lit_test_suite(
  NAME
    test
  SRCS
    "test/gemmini_matmul_lowering.mlir"
  TOOLS
    FileCheck
    iree-compile
  LABELS
    "hostonly"
)