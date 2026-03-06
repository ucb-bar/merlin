// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Splits stream.async.execute regions that contain multiple dispatch
// operations into individual execute regions, each with exactly one dispatch.
//
// Before:
//   %results, %tp = stream.async.execute {
//     %d0 = stream.async.dispatch @dispatch_0(...)
//     %d1 = stream.async.dispatch @dispatch_1(%d0, ...)
//     stream.yield %d1
//   }
//
// After:
//   %r0, %tp0 = stream.async.execute {
//     %d0 = stream.async.dispatch @dispatch_0(...)
//     stream.yield %d0
//   }
//   %r1, %tp1 = stream.async.execute await(%tp0) {
//     %d1 = stream.async.dispatch @dispatch_1(%r0, ...)
//     stream.yield %d1
//   }

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-stream-split-execute-regions"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_SPLITEXECUTEREGIONSPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

// A "chunk" is one dispatch plus all of the non-dispatch ops it transitively
// depends on (splats, constants, etc.) that live in the same execute body.
struct DispatchChunk {
  // The dispatch op that anchors this chunk.
  IREE::Stream::AsyncDispatchOp dispatchOp;
  // All ops in this chunk, in topological order. Includes the dispatch and
  // its local (non-dispatch) dependencies like splats.
  SmallVector<Operation *> ops;
  // Values from inside the original execute body that this chunk's ops
  // consume but that are produced by OTHER chunks (i.e., cross-chunk deps).
  // These become captured operands of the new execute region.
  SetVector<Value> externalInputs;
  // Values produced by this chunk that are consumed by later chunks or by
  // the yield (i.e., they escape). These become results of the new execute.
  SetVector<Value> escapingOutputs;
};

// Queries the size of a stream resource value. Uses the
// SizeAwareTypeInterface to find a size value in scope.
static Value queryValueSize(Location loc, Value value, OpBuilder &builder) {
  auto sizeValue =
      IREE::Util::SizeAwareTypeInterface::queryValueSize(loc, value, builder);
  if (sizeValue) return sizeValue;
  // Fallback: if the value itself carries its size as a dynamic dim we just
  // won't find it and this will fail verification later. In practice the
  // stream dialect always carries sizes.
  return {};
}

// Determines whether an op is a "support" op (not a dispatch) that should be
// pulled into a chunk alongside the dispatch that uses its result. Splats,
// clones, fills, etc.
static bool isSupportOp(Operation *op) {
  // Anything that is not a dispatch, yield, or concurrent is "support".
  return !isa<IREE::Stream::AsyncDispatchOp>(op) &&
         !isa<IREE::Stream::YieldOp>(op) &&
         !isa<IREE::Stream::AsyncConcurrentOp>(op);
}

// Collects all non-dispatch ops that `dispatchOp` transitively depends on
// within the same block.
static void collectLocalDeps(Operation *root, Block *block,
                             DenseSet<Operation *> &collected) {
  SmallVector<Operation *> worklist;
  worklist.push_back(root);
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    for (Value operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp || defOp->getBlock() != block) continue;
      if (isa<IREE::Stream::AsyncDispatchOp>(defOp)) continue;
      if (collected.insert(defOp).second) {
        worklist.push_back(defOp);
      }
    }
  }
}

// Builds the list of dispatch chunks from an execute region body.
static SmallVector<DispatchChunk>
buildChunks(IREE::Stream::AsyncExecuteOp executeOp) {
  auto &body = executeOp.getBody();
  assert(body.getBlocks().size() == 1);
  Block *block = &body.front();

  // Gather all dispatches in order.
  SmallVector<IREE::Stream::AsyncDispatchOp> dispatches;
  for (auto &op : *block) {
    if (auto dispatch = dyn_cast<IREE::Stream::AsyncDispatchOp>(op)) {
      dispatches.push_back(dispatch);
    }
  }

  // For each dispatch, figure out which support ops belong to it.
  // A support op belongs to the FIRST dispatch that (transitively) uses it.
  DenseMap<Operation *, int> opToChunk;
  SmallVector<DispatchChunk> chunks(dispatches.size());
  for (auto [i, dispatch] : llvm::enumerate(dispatches)) {
    chunks[i].dispatchOp = dispatch;

    DenseSet<Operation *> localDeps;
    collectLocalDeps(dispatch, block, localDeps);

    // Assign support ops to this chunk if not yet assigned.
    for (Operation *dep : localDeps) {
      if (!opToChunk.count(dep)) {
        opToChunk[dep] = i;
      }
    }
    opToChunk[dispatch] = i;
  }

  // Build the ops list per chunk in topological order.
  for (auto &op : *block) {
    if (isa<IREE::Stream::YieldOp>(op)) continue;
    auto it = opToChunk.find(&op);
    if (it != opToChunk.end()) {
      chunks[it->second].ops.push_back(&op);
    }
  }

  // Determine cross-chunk inputs and escaping outputs.
  // A value is an "external input" to chunk i if it's produced by chunk j<i
  // (or is a block argument).
  auto yieldOp = cast<IREE::Stream::YieldOp>(block->getTerminator());
  DenseSet<Value> yieldedValues(yieldOp.getResourceOperands().begin(),
                                yieldOp.getResourceOperands().end());

  for (auto [i, chunk] : llvm::enumerate(chunks)) {
    DenseSet<Operation *> chunkOpSet(chunk.ops.begin(), chunk.ops.end());

    for (Operation *op : chunk.ops) {
      for (Value operand : op->getOperands()) {
        // Block arguments are always external.
        if (isa<BlockArgument>(operand)) {
          chunk.externalInputs.insert(operand);
          continue;
        }
        Operation *defOp = operand.getDefiningOp();
        if (defOp && !chunkOpSet.contains(defOp)) {
          // Produced by another chunk — external input.
          chunk.externalInputs.insert(operand);
        }
      }
    }

    // A value escapes if it's used by a later chunk OR is yielded.
    for (Operation *op : chunk.ops) {
      for (Value result : op->getResults()) {
        bool escapes = false;
        if (yieldedValues.contains(result)) escapes = true;
        for (OpOperand &use : result.getUses()) {
          if (!chunkOpSet.contains(use.getOwner())) {
            escapes = true;
            break;
          }
        }
        if (escapes) {
          chunk.escapingOutputs.insert(result);
        }
      }
    }
  }

  return chunks;
}

// Creates a new stream.async.execute for a single chunk.
static IREE::Stream::AsyncExecuteOp
emitChunkExecute(DispatchChunk &chunk,
                 Value awaitTimepoint,
                 IREE::Stream::AffinityAttr affinityAttr,
                 IRMapping &valueMapping,
                 OpBuilder &builder) {
  auto loc = chunk.dispatchOp.getLoc();

  // Gather operands: the external inputs mapped through valueMapping.
  SmallVector<Value> operands;
  SmallVector<Value> operandSizes;
  for (Value input : chunk.externalInputs) {
    Value mapped = valueMapping.lookupOrDefault(input);
    if (!isa<IREE::Stream::ResourceType>(mapped.getType())) continue;
    operands.push_back(mapped);
    Value size = queryValueSize(loc, mapped, builder);
    if (size) operandSizes.push_back(size);
  }

  // Gather result types and sizes from escaping outputs.
  SmallVector<Type> resultTypes;
  SmallVector<Value> resultSizes;
  for (Value output : chunk.escapingOutputs) {
    resultTypes.push_back(output.getType());
    Value size = queryValueSize(loc, output, builder);
    if (size) resultSizes.push_back(size);
  }

  SmallVector<int64_t> tiedOperands;
  auto newExecuteOp = IREE::Stream::AsyncExecuteOp::create(
      builder, loc, resultTypes, resultSizes,
      awaitTimepoint, operands, operandSizes, tiedOperands);
  if (affinityAttr) {
    newExecuteOp.setAffinityAttr(affinityAttr);
  }

  // Build the body.
  auto &entryBlock = newExecuteOp.getBody().emplaceBlock();

  // Add block arguments for resource operands.
  IRMapping bodyMapping;
  unsigned argIdx = 0;
  for (Value input : chunk.externalInputs) {
    Value mapped = valueMapping.lookupOrDefault(input);
    if (!isa<IREE::Stream::ResourceType>(mapped.getType())) continue;
    auto arg = entryBlock.addArgument(mapped.getType(), loc);
    bodyMapping.map(input, arg);
    argIdx++;
  }
  // Also map non-resource external inputs (like i32 splat values).
  // These just get used directly; they aren't block arguments.
  // The clone below will pick them up via bodyMapping.lookupOrDefault.

  // Clone ops into the body.
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(&entryBlock);
  for (Operation *op : chunk.ops) {
    bodyBuilder.clone(*op, bodyMapping);
  }

  // Build the yield with escaping outputs.
  SmallVector<Value> yieldValues;
  SmallVector<Value> yieldSizes;
  for (Value output : chunk.escapingOutputs) {
    Value mapped = bodyMapping.lookupOrDefault(output);
    yieldValues.push_back(mapped);
    Value size = queryValueSize(loc, mapped, bodyBuilder);
    if (size) yieldSizes.push_back(size);
  }
  IREE::Stream::YieldOp::create(bodyBuilder, loc, yieldValues, yieldSizes);

  // Map the original escaping outputs to the results of this new execute.
  for (auto [output, result] :
       llvm::zip_equal(chunk.escapingOutputs, newExecuteOp.getResults())) {
    valueMapping.map(output, result);
  }

  return newExecuteOp;
}

// Splits a single execute op that contains multiple dispatches.
static LogicalResult
splitExecuteOp(IREE::Stream::AsyncExecuteOp executeOp) {
  auto &body = executeOp.getBody();
  if (body.getBlocks().size() != 1) return success();
  Block *block = &body.front();

  // Count dispatches.
  int dispatchCount = 0;
  for (auto &op : *block) {
    if (isa<IREE::Stream::AsyncDispatchOp>(op)) dispatchCount++;
  }
  // Nothing to split if 0 or 1 dispatches.
  if (dispatchCount <= 1) return success();

  LLVM_DEBUG(llvm::dbgs() << "Splitting execute with " << dispatchCount
                          << " dispatches\n");

  auto chunks = buildChunks(executeOp);
  if (chunks.empty()) return success();

  auto affinityAttr = executeOp.getAffinityAttr();

  // We build new execute ops right before the original one.
  OpBuilder builder(executeOp);
  IRMapping valueMapping;

  // Map original execute's block args → original operands so that the
  // new executes can capture from outside the original region.
  for (auto [blockArg, operand] :
       llvm::zip_equal(block->getArguments(),
                       executeOp.getResourceOperands())) {
    valueMapping.map(blockArg, operand);
  }

  // Track the latest timepoint for chaining.
  Value currentTimepoint = executeOp.getAwaitTimepoint();

  for (auto &chunk : chunks) {
    auto newExecuteOp = emitChunkExecute(
        chunk, currentTimepoint, affinityAttr, valueMapping, builder);
    currentTimepoint = newExecuteOp.getResultTimepoint();
  }

  // Now replace uses of the original execute's results.
  // The original yield tells us which internal values mapped to which results.
  auto yieldOp = cast<IREE::Stream::YieldOp>(block->getTerminator());
  for (auto [originalResult, yieldedValue] :
       llvm::zip_equal(executeOp.getResults(),
                       yieldOp.getResourceOperands())) {
    Value replacement = valueMapping.lookupOrDefault(yieldedValue);
    originalResult.replaceAllUsesWith(replacement);
  }
  executeOp.getResultTimepoint().replaceAllUsesWith(currentTimepoint);

  // Erase the original execute.
  executeOp.erase();

  return success();
}

//===----------------------------------------------------------------------===//
// --iree-stream-split-execute-regions
//===----------------------------------------------------------------------===//

struct SplitExecuteRegionsPass
    : public IREE::Stream::impl::SplitExecuteRegionsPassBase<
          SplitExecuteRegionsPass> {
  void runOnOperation() override {
    mlir::CallableOpInterface parentOp = getOperation();
    if (!parentOp.getCallableRegion() ||
        parentOp.getCallableRegion()->empty()) {
      return;
    }

    // Collect execute ops first to avoid modifying while iterating.
    SmallVector<IREE::Stream::AsyncExecuteOp> executeOps;
    parentOp.getCallableRegion()->walk(
        [&](IREE::Stream::AsyncExecuteOp op) {
          executeOps.push_back(op);
        });

    for (auto executeOp : executeOps) {
      if (failed(splitExecuteOp(executeOp))) {
        return signalPassFailure();
      }
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::IREE::Stream