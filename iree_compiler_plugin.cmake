# Copyright 2026 UCB-BAR
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This file is included by IREE via -DIREE_CMAKE_PLUGIN_PATHS and executes in
# IREE's compiler plugin CMake context. Always use absolute source paths.

set(MERLIN_COMPILER_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(MERLIN_COMPILER_BINARY_ROOT "${CMAKE_CURRENT_BINARY_DIR}/merlin")

option(MERLIN_ENABLE_CORE
       "Build Merlin experimental core libraries under compiler/src/merlin."
       ON)
option(MERLIN_ENABLE_TARGET_GEMMINI "Enable Merlin Gemmini target plugin." ON)
option(MERLIN_ENABLE_TARGET_SATURN "Enable Merlin Saturn target plugin." ON)
option(MERLIN_ENABLE_TARGET_SPACEMIT
       "Enable Merlin SpacemiT target plugin."
       ON)

if((MERLIN_ENABLE_TARGET_GEMMINI OR
    MERLIN_ENABLE_TARGET_SATURN OR
    MERLIN_ENABLE_TARGET_SPACEMIT) AND
   NOT MERLIN_ENABLE_CORE)
  message(FATAL_ERROR
    "Merlin target plugins require -DMERLIN_ENABLE_CORE=ON")
endif()

if(MERLIN_ENABLE_CORE)
  add_subdirectory("${MERLIN_COMPILER_SOURCE_DIR}/src/merlin"
                   "${MERLIN_COMPILER_BINARY_ROOT}/src/merlin")
endif()

if(MERLIN_ENABLE_TARGET_GEMMINI)
  add_subdirectory("${MERLIN_COMPILER_SOURCE_DIR}/plugins/target/Gemmini"
                   "${MERLIN_COMPILER_BINARY_ROOT}/target/Gemmini")
endif()

if(MERLIN_ENABLE_TARGET_SATURN)
  add_subdirectory("${MERLIN_COMPILER_SOURCE_DIR}/plugins/target/Saturn"
                   "${MERLIN_COMPILER_BINARY_ROOT}/target/Saturn")
endif()

if(MERLIN_ENABLE_TARGET_SPACEMIT)
  add_subdirectory("${MERLIN_COMPILER_SOURCE_DIR}/plugins/target/SpacemiT"
                   "${MERLIN_COMPILER_BINARY_ROOT}/target/SpacemiT")
endif()
