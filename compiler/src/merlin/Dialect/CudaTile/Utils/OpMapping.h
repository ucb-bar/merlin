// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Shared op-name mapping tables for cuda_tile lowering. Consumed by:
//   - compiler/src/merlin/Dialect/CudaTile/Transforms/Convert*.cpp
//     (linalg/arith/math -> cuda_tile op-name attribute)
//   - compiler/plugins/target/CudaTile/CudaTileTarget.cpp
//     (codegen-time body walking)
//
// Header-only by design: the maps are pure switches over op type.

#ifndef MERLIN_DIALECT_CUDATILE_UTILS_OPMAPPING_H_
#define MERLIN_DIALECT_CUDATILE_UTILS_OPMAPPING_H_

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::iree_compiler::cuda_tile {

inline llvm::StringRef mapArithToCudaTile(Operation *op) {
  if (isa<arith::AddFOp>(op)) return "addf";
  if (isa<arith::SubFOp>(op)) return "subf";
  if (isa<arith::MulFOp>(op)) return "mulf";
  if (isa<arith::DivFOp>(op)) return "divf";
  if (isa<arith::MaximumFOp, arith::MaxNumFOp>(op)) return "maxf";
  if (isa<arith::MinimumFOp, arith::MinNumFOp>(op)) return "minf";
  if (isa<arith::NegFOp>(op)) return "negf";
  if (isa<arith::AddIOp>(op)) return "addi";
  if (isa<arith::SubIOp>(op)) return "subi";
  if (isa<arith::MulIOp>(op)) return "muli";
  if (isa<arith::AndIOp>(op)) return "andi";
  if (isa<arith::OrIOp>(op)) return "ori";
  if (isa<arith::XOrIOp>(op)) return "xori";
  if (isa<arith::MaxSIOp>(op)) return "maxi";
  if (isa<arith::MinSIOp>(op)) return "mini";
  if (isa<arith::SelectOp>(op)) return "select";
  return "";
}

inline llvm::StringRef mapMathToCudaTile(Operation *op) {
  if (isa<math::ExpOp>(op)) return "exp";
  if (isa<math::Exp2Op>(op)) return "exp2";
  if (isa<math::LogOp>(op)) return "log";
  if (isa<math::Log2Op>(op)) return "log2";
  if (isa<math::SqrtOp>(op)) return "sqrt";
  if (isa<math::RsqrtOp>(op)) return "rsqrt";
  if (isa<math::SinOp>(op)) return "sin";
  if (isa<math::CosOp>(op)) return "cos";
  if (isa<math::TanhOp>(op)) return "tanh";
  if (isa<math::CeilOp>(op)) return "ceil";
  if (isa<math::FloorOp>(op)) return "floor";
  if (isa<math::AbsFOp>(op)) return "absf";
  if (isa<math::FmaOp>(op)) return "fma";
  return "";
}

// Match a single-op reduction body. Returns "" if the body has zero or more
// than one non-yield op, or if the op is not a recognized combiner.
inline llvm::StringRef matchReduceCombiner(Region &body) {
  if (body.empty() || body.front().empty()) return "";
  Block &block = body.front();
  auto ops = block.without_terminator();
  auto it = ops.begin();
  if (it == ops.end()) return "";
  Operation *combinerOp = &*it;
  ++it;
  if (it != ops.end()) return "";

  if (isa<arith::AddFOp>(combinerOp)) return "addf";
  if (isa<arith::AddIOp>(combinerOp)) return "addi";
  if (isa<arith::MaximumFOp, arith::MaxNumFOp>(combinerOp)) return "maxf";
  if (isa<arith::MinimumFOp, arith::MinNumFOp>(combinerOp)) return "minf";
  if (isa<arith::MaxSIOp>(combinerOp)) return "maxi";
  if (isa<arith::MinSIOp>(combinerOp)) return "mini";
  if (isa<arith::MulFOp>(combinerOp)) return "mulf";
  if (isa<arith::MulIOp>(combinerOp)) return "muli";
  return "";
}

}  // namespace mlir::iree_compiler::cuda_tile

#endif  // MERLIN_DIALECT_CUDATILE_UTILS_OPMAPPING_H_
