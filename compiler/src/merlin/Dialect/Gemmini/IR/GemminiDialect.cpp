//===- GemminiDialect.cpp - MLIR Gemmini dialect implementation ----------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Original File sourced and modified from https://github.com/buddy-compiler/buddy-mlir
//
//===----------------------------------------------------------------------===//
//
// This file implements the Gemmini dialect and its operations.
//
//===----------------------------------------------------------------------===//

// compiler/src/merlin/Dialect/Gemmini/IR/GemminiDialect.cpp

#include "merlin/Dialect/Gemmini/IR/GemminiDialect.h"
#include "merlin/Dialect/Gemmini/IR/GemminiOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

// Ensure namespace matches
using namespace merlin::gemmini;

// TableGen generated dialect definitions
#include "merlin/Dialect/Gemmini/IR/GemminiDialect.cpp.inc"

#define GET_OP_CLASSES
#include "merlin/Dialect/Gemmini/IR/Gemmini.cpp.inc"

void GemminiDialect::initialize() {
  // Register all operations defined in Gemmini.td
  addOperations<
#define GET_OP_LIST
#include "merlin/Dialect/Gemmini/IR/Gemmini.cpp.inc"
      >();
}