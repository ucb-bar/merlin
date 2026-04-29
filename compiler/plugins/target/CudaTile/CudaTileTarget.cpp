// Copyright 2026 UCB-BAR
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// CudaTile target backend for IREE.
//
// Produces CUBIN binaries via NVIDIA's cuda_tile IR and tileiras assembler,
// bypassing LLVM's NVPTX backend entirely. cuda_tile kernels run on NVIDIA
// GPUs with native support for TMA, wgmma, and CTA clustering.
//
// In-process compilation: builds cuda_tile dialect ops directly via OpBuilder,
// serializes to tilebc via BytecodeWriter, then invokes tileiras for cubin.

#include "compiler/plugins/target/CudaTile/CudaTileKernelPlan.h"
#include "compiler/plugins/target/CudaTile/CudaTileOptions.h"
#include "compiler/src/merlin/Dialect/CudaTile/Transforms/Passes.h"
#include "compiler/src/merlin/Dialect/CudaTile/Utils/OpMapping.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "llvm/Support/MathExtras.h"

// cuda_tile dialect + bytecode writer (in-process compilation).
#include "cuda_tile/Bytecode/Writer/BytecodeWriter.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Utils/ExecutableDebugInfoUtils.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/compiler/Utils/StringUtils.h"
#include "iree/compiler/Utils/ToolUtils.h"
#include "iree/schemas/cuda_tile_executable_def_builder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdlib>

#define DEBUG_TYPE "cuda-tile-target"

namespace mlir::iree_compiler::IREE::HAL {

namespace {

//===----------------------------------------------------------------------===//
// tileiras Tool Invocation
//===----------------------------------------------------------------------===//

static constexpr char kTileirasCompilerName[] = "tileiras";

/// Attempts to find the tileiras assembler tool on PATH or at a user-specified
/// location.
static FailureOr<std::string>
findTileirasCompiler(const CudaTileOptions &options, std::string *message) {
  if (!options.tileirasPath.empty()) {
    if (llvm::sys::fs::exists(options.tileirasPath)) {
      return options.tileirasPath;
    }
  }

  std::string tileirasCompiler = findTool(kTileirasCompilerName);
  if (llvm::sys::fs::exists(tileirasCompiler)) {
    return tileirasCompiler;
  }

  *message = std::string(
      "Could not find tileiras assembler. Install via "
      "'conda install cuda-tileiras' or pass path explicitly with "
      "--iree-cuda-tile-tileiras-path=<path>");
  return failure();
}

/// Compiles cuda_tile bytecode (tilebc) to CUBIN using NVIDIA's tileiras
/// assembler. Follows the same pattern as CUDA's compileWithPtxas().
static FailureOr<std::string>
compileWithTileiras(StringRef tileirasCompiler, StringRef smArch,
                    StringRef tileirasParams, StringRef tilebcData,
                    std::string *message) {
  // Create temporary files.
  llvm::SmallString<64> tilebcFile, stdinFile, stdoutFile, stderrFile;
  llvm::sys::fs::createTemporaryFile("iree-tilebc", ".tilebc", tilebcFile);
  llvm::sys::fs::createTemporaryFile("tileiras-stdin", "", stdinFile);
  llvm::sys::fs::createTemporaryFile("tileiras-stdout", "", stdoutFile);
  llvm::sys::fs::createTemporaryFile("tileiras-stderr", "", stderrFile);
  std::string cubinFile = std::string(tilebcFile) + ".cubin";
  llvm::FileRemover stdinRemover(stdinFile.c_str());
  llvm::FileRemover stdoutRemover(stdoutFile.c_str());
  llvm::FileRemover stderrRemover(stderrFile.c_str());
  llvm::FileRemover binRemover(cubinFile.c_str());
  llvm::FileRemover srcRemover(tilebcFile.c_str());

  // Write tilebc data to temp file.
  {
    std::error_code ec;
    llvm::raw_fd_ostream fTilebc(tilebcFile, ec, llvm::sys::fs::OF_None);
    fTilebc.write(tilebcData.data(), tilebcData.size());
    fTilebc.close();
    if (fTilebc.has_error()) {
      *message = "Could not write tilebc to temporary file";
      return failure();
    }
  }

  // Build tileiras command line:
  //   tileiras --gpu-name sm_86 -o output.cubin input.tilebc
  std::vector<StringRef> argVector{
      StringRef(kTileirasCompilerName), StringRef("--gpu-name"),
      smArch,                           StringRef("-o"),
      StringRef(cubinFile),             StringRef(tilebcFile),
  };

  // Parse additional user-supplied parameters.
#ifdef _WIN32
  auto tokenize = llvm::cl::TokenizeWindowsCommandLine;
#else
  auto tokenize = llvm::cl::TokenizeGNUCommandLine;
#endif
  llvm::BumpPtrAllocator scratchAllocator;
  llvm::StringSaver stringSaver(scratchAllocator);
  SmallVector<const char *> rawArgs;
  tokenize(tileirasParams, stringSaver, rawArgs, /*MarkEOLs=*/false);
  for (auto rawArg : rawArgs) {
    argVector.push_back(StringRef(rawArg));
  }

  std::optional<StringRef> redirects[] = {
      stdinFile.str(),
      stdoutFile.str(),
      stderrFile.str(),
  };

  // Invoke tileiras.
  if (llvm::sys::ExecuteAndWait(unescapeCommandLineComponent(tileirasCompiler),
                                llvm::ArrayRef<llvm::StringRef>(argVector),
                                /*Env=*/std::nullopt,
                                /*Redirects=*/redirects,
                                /*SecondsToWait=*/0, /*MemoryLimit=*/0,
                                /*ErrMsg=*/message)) {
    if (message->empty()) {
      *message = std::string("Invoking tileiras failed, see: ") +
                 stderrFile.str().str() + "\n";
    }
    stderrRemover.releaseFile();
    return failure();
  }

  // Read the cubin output.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> maybeCubin =
      llvm::MemoryBuffer::getFile(cubinFile);
  if (!maybeCubin) {
    *message = "Could not read cubin file produced by tileiras";
    return failure();
  }

  return std::string(maybeCubin->get()->getBuffer());
}

static LogicalResult
dumpCudaTileKernelPlanIfRequested(Operation *anchor,
                                  const CudaTileOptions &options,
                                  const CudaTileKernelPlan &plan) {
  if (options.dumpKernelPlanTo.empty())
    return success();

  if (options.dumpKernelPlanTo == "-") {
    printCudaTileKernelPlan(plan, llvm::outs());
    return success();
  }

  std::error_code ec;
  llvm::raw_fd_ostream os(
      options.dumpKernelPlanTo, ec,
      llvm::sys::fs::OF_Append | llvm::sys::fs::OF_Text);
  if (ec)
    return anchor->emitError("failed to open cuda_tile kernel plan dump file '")
           << options.dumpKernelPlanTo << "': " << ec.message();

  printCudaTileKernelPlan(plan, os);
  os.flush();
  if (os.has_error())
    return anchor->emitError("failed to write cuda_tile kernel plan dump file '")
           << options.dumpKernelPlanTo << "'";
  return success();
}

//===----------------------------------------------------------------------===//
// Helper Utilities
//===----------------------------------------------------------------------===//

/// Compute row-major strides for a shape.
static SmallVector<int64_t> computeRowMajorStrides(ArrayRef<int64_t> shape) {
  return mlir::computeSuffixProduct(shape);
}

/// Read a DenseI64ArrayAttr from an op, or return empty.
static SmallVector<int64_t> getI64ArrayAttr(Operation *op, StringRef name) {
  if (auto attr = op->getAttrOfType<DenseI64ArrayAttr>(name))
    return SmallVector<int64_t>(attr.asArrayRef().begin(),
                                attr.asArrayRef().end());
  return {};
}

/// Round up to next power of 2 (or keep if already power of 2).
static int64_t nextPow2(int64_t v) {
  if (v <= 1) return 1;
  return static_cast<int64_t>(llvm::PowerOf2Ceil(static_cast<uint64_t>(v)));
}

/// Extract the tensor dimension that is reduced in a linalg.generic.
/// Maps from iterator-space reduction dims to tensor dims via indexing maps.
static int64_t extractReduceDim(linalg::GenericOp genOp,
                                ArrayRef<int64_t> /*srcShape*/) {
  auto iterTypes = genOp.getIteratorTypesArray();
  auto maps = genOp.getIndexingMapsArray();
  AffineMap inputMap = maps.empty() ? AffineMap() : maps[0];
  for (unsigned iterDim = 0; iterDim < iterTypes.size(); ++iterDim) {
    if (iterTypes[iterDim] != mlir::utils::IteratorType::reduction)
      continue;
    if (inputMap) {
      for (unsigned tensorDim = 0; tensorDim < inputMap.getNumResults();
           ++tensorDim) {
        if (auto dimExpr = dyn_cast<AffineDimExpr>(inputMap.getResult(tensorDim)))
          if (dimExpr.getPosition() == iterDim)
            return tensorDim;
      }
    } else {
      return iterDim;
    }
  }
  return -1;
}

// Op-name mapping tables live in the shared header so the linalg→cuda_tile
// transform passes and codegen-time body walking stay in lock-step.
using ::mlir::iree_compiler::cuda_tile::mapArithToCudaTile;
using ::mlir::iree_compiler::cuda_tile::mapMathToCudaTile;
using ::mlir::iree_compiler::cuda_tile::matchReduceCombiner;

static SmallVector<StringRef> getElementwiseOpNames(Operation *op) {
  SmallVector<StringRef> ops;
  if (auto opNameAttr = op->getAttrOfType<StringAttr>("cuda_tile.op_name")) {
    opNameAttr.getValue().split(ops, ';');
    return ops;
  }

  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp)
    return ops;

  for (auto &bodyOp : genericOp.getRegion().front().without_terminator()) {
    StringRef name = mapArithToCudaTile(&bodyOp);
    if (name.empty())
      name = mapMathToCudaTile(&bodyOp);
    if (!name.empty())
      ops.push_back(name);
  }
  return ops;
}

/// Compute tile shape for an N-D tensor. Same rank as input.
/// Last dim: tile by tileN. Second-to-last: tile by tileM.
/// All other dims: keep full (they become batch dims in the grid).
/// All tile dims are rounded to power of 2 (cuda_tile requirement).
static SmallVector<int64_t> computeTileShape(ArrayRef<int64_t> shape,
                                             int64_t tileM, int64_t tileN) {
  int64_t rank = shape.size();
  SmallVector<int64_t> tileShape(rank);
  for (int64_t i = 0; i < rank; ++i) {
    int64_t dim;
    if (i == rank - 1)
      dim = std::min(tileN, shape[i]);
    else if (i == rank - 2)
      dim = std::min(tileM, shape[i]);
    else
      dim = shape[i]; // batch dims: full
    tileShape[i] = nextPow2(dim);
  }
  return tileShape;
}

/// Compute grid dims from N-D shape and tile shape.
/// gridX = tiles along last dim, gridY = tiles along second-to-last,
/// gridZ = product of all batch dims (dims before the last 2).
static SmallVector<int64_t, 3> computeGridDims(ArrayRef<int64_t> shape,
                                                ArrayRef<int64_t> tileShape) {
  int64_t rank = std::min(shape.size(), tileShape.size());
  int64_t gridX = 1, gridY = 1, gridZ = 1;
  if (rank >= 1)
    gridX = (shape[rank - 1] + tileShape[rank - 1] - 1) / tileShape[rank - 1];
  if (rank >= 2)
    gridY = (shape[rank - 2] + tileShape[rank - 2] - 1) / tileShape[rank - 2];
  for (int64_t i = 0; i < rank - 2; ++i)
    gridZ *= (shape[i] + tileShape[i] - 1) / tileShape[i];
  return {gridX, gridY, gridZ};
}

/// Extracts a static tensor shape from plain tensors or dispatch tensor wrappers.
static SmallVector<int64_t> getStaticShapeFromType(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type)) {
    if (shaped.hasStaticShape())
      return SmallVector<int64_t>(shaped.getShape().begin(),
                                  shaped.getShape().end());
    return {};
  }
  if (auto dispatchTensor =
          dyn_cast<IREE::TensorExt::DispatchTensorType>(type)) {
    if (auto shaped = dyn_cast<ShapedType>(dispatchTensor.getBoundType())) {
      if (shaped.hasStaticShape())
        return SmallVector<int64_t>(shaped.getShape().begin(),
                                    shaped.getShape().end());
    }
  }
  return {};
}

static Type getElementTypeFromType(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type))
    return shaped.getElementType();
  if (auto dispatchTensor =
          dyn_cast<IREE::TensorExt::DispatchTensorType>(type)) {
    if (auto shaped = dyn_cast<ShapedType>(dispatchTensor.getBoundType()))
      return shaped.getElementType();
  }
  return {};
}

static int64_t getElementByteWidth(Type elemType) {
  if (elemType.isIndex())
    return 8;
  if (auto intType = dyn_cast<IntegerType>(elemType))
    return std::max<int64_t>(1, (intType.getWidth() + 7) / 8);
  if (auto floatType = dyn_cast<FloatType>(elemType))
    return std::max<int64_t>(1, (floatType.getWidth() + 7) / 8);
  return 0;
}

static bool isIdentitySlice(ArrayRef<int64_t> offsets, ArrayRef<int64_t> strides) {
  return llvm::all_of(offsets, [](int64_t v) { return v == 0; }) &&
         llvm::all_of(strides, [](int64_t v) { return v == 1; });
}

//===----------------------------------------------------------------------===//
// CudaTileOpEmitter — builds cuda_tile dialect ops in-process
//===----------------------------------------------------------------------===//

/// Builds a ::mlir::cuda_tile::ModuleOp with ops via OpBuilder, then serializes to
/// tilebc bytecode. Replaces the old text-based CudaTileTextEmitter.
class CudaTileOpEmitter {
public:
  explicit CudaTileOpEmitter(MLIRContext *ctx)
      : ctx(ctx), loc(UnknownLoc::get(ctx)), b(ctx) {}

  ~CudaTileOpEmitter() {
    if (moduleOp)
      moduleOp->erase();
  }

  // Non-copyable, movable.
  CudaTileOpEmitter(const CudaTileOpEmitter &) = delete;
  CudaTileOpEmitter &operator=(const CudaTileOpEmitter &) = delete;
  CudaTileOpEmitter(CudaTileOpEmitter &&other) noexcept
      : ctx(other.ctx), loc(other.loc), b(other.ctx),
        moduleOp(other.moduleOp), entryOp(other.entryOp),
        entryArgs(std::move(other.entryArgs)) {
    other.moduleOp = nullptr;
  }

  //===-- Module structure ------------------------------------------------===//

  /// Create a standalone cuda_tile.module (not attached to any parent).
  void beginModule(StringRef name) {
    OperationState state(loc, ::mlir::cuda_tile::ModuleOp::getOperationName());
    state.addAttribute(SymbolTable::getSymbolAttrName(),
                       StringAttr::get(ctx, name));
    state.addRegion()->emplaceBlock();
    Operation *op = Operation::create(state);
    moduleOp = cast<::mlir::cuda_tile::ModuleOp>(op);
    b.setInsertionPointToEnd(&moduleOp.getBody().front());
  }

  /// Create an entry function with numArgs pointer arguments.
  void beginEntry(StringRef name, int numArgs, Type elemType) {
    auto ptrType = ::mlir::cuda_tile::PointerType::get(ctx, elemType);
    auto tilePtrType = ::mlir::cuda_tile::TileType::get(ctx, {}, ptrType);
    SmallVector<Type> argTypes(numArgs, tilePtrType);
    auto funcType = FunctionType::get(ctx, argTypes, {});

    entryOp = b.create<::mlir::cuda_tile::EntryOp>(
        loc, StringAttr::get(ctx, name), TypeAttr::get(funcType),
        /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr,
        /*optimization_hints=*/nullptr);

    // The builder creates the region but not the entry block.
    // Use FunctionOpInterface::addEntryBlock() to create the block with
    // properly-typed arguments, then add the implicit ReturnOp terminator.
    Block *entryBlock = entryOp.addEntryBlock();
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToEnd(entryBlock);
      b.create<::mlir::cuda_tile::ReturnOp>(loc);
    }

    // Set insertion point before the terminator for body ops.
    b.setInsertionPoint(entryBlock->getTerminator());

    entryArgs.clear();
    for (auto arg : entryBlock->getArguments())
      entryArgs.push_back(arg);
  }

  Value getArg(int idx) { return entryArgs[idx]; }

  /// Finalize the entry. Insertion point moves back to module body.
  void endEntry() { b.setInsertionPointToEnd(&moduleOp.getBody().front()); }

  //===-- Constants -------------------------------------------------------===//

  /// Create a scalar i32 constant: `constant dense<val> : tile<i32>`.
  Value constI32(int64_t val) {
    auto type = ::mlir::cuda_tile::TileType::get(ctx, {}, b.getI32Type());
    auto attr = DenseElementsAttr::get(cast<ShapedType>(type),
                                       b.getI32IntegerAttr(val));
    return b.create<::mlir::cuda_tile::ConstantOp>(loc, type,
                                           cast<DenseTypedElementsAttr>(attr))
        .getResult();
  }

  /// Create a splat tile constant: `constant dense<val> : tile<shape x elem>`.
  Value constSplat(ArrayRef<int64_t> shape, Type elemType, double val) {
    auto type = ::mlir::cuda_tile::TileType::get(ctx, shape, elemType);
    Attribute elemVal;
    if (isa<FloatType>(elemType))
      elemVal = FloatAttr::get(elemType, val);
    else
      elemVal = IntegerAttr::get(elemType, static_cast<int64_t>(val));
    auto attr = DenseElementsAttr::get(cast<ShapedType>(type), elemVal);
    return b.create<::mlir::cuda_tile::ConstantOp>(loc, type,
                                           cast<DenseTypedElementsAttr>(attr))
        .getResult();
  }

  /// Broadcast a scalar tile value to an N-D tile shape.
  Value broadcastScalarToTile(Value scalar, ArrayRef<int64_t> shape,
                              Type elemType) {
    if (shape.empty())
      return scalar;
    SmallVector<int64_t> ones(shape.size(), 1);
    Value shaped = reshape(scalar, ones, elemType);
    if (ones == SmallVector<int64_t>(shape))
      return shaped;
    return broadcastTile(shaped, shape, elemType);
  }

  /// Create a per-lane 0..tileShape[dim)-1 index tile for one tile dimension.
  Value iotaForDim(ArrayRef<int64_t> tileShape, int64_t dim) {
    Type i32 = b.getI32Type();
    auto oneDType = ::mlir::cuda_tile::TileType::get(ctx, {tileShape[dim]}, i32);
    Value local =
        b.create<::mlir::cuda_tile::IotaOp>(loc, oneDType).getResult();
    if (tileShape.size() == 1)
      return local;

    SmallVector<int64_t> reshapeShape(tileShape.size(), 1);
    reshapeShape[dim] = tileShape[dim];
    local = reshape(local, reshapeShape, i32);
    if (reshapeShape == SmallVector<int64_t>(tileShape))
      return local;
    return broadcastTile(local, tileShape, i32);
  }

  /// Create a tile containing global element indices for a linalg.index dim.
  Value globalIndexTile(ArrayRef<int64_t> tileShape, int64_t dim,
                        Value tileIndex) {
    Type i32 = b.getI32Type();
    Value local = iotaForDim(tileShape, dim);
    Value tileExtent = constI32(tileShape[dim]);
    Value base = b.create<::mlir::cuda_tile::MulIOp>(
                      loc, tileIndex, tileExtent,
                      ::mlir::cuda_tile::IntegerOverflowAttr::get(
                          ctx, ::mlir::cuda_tile::IntegerOverflow::NONE))
                     .getResult();
    base = broadcastScalarToTile(base, tileShape, i32);
    return b.create<::mlir::cuda_tile::AddIOp>(
                loc, local, base,
                ::mlir::cuda_tile::IntegerOverflowAttr::get(
                    ctx, ::mlir::cuda_tile::IntegerOverflow::NONE))
        .getResult();
  }

  /// Create a scalar tile containing lane 0's global index for a dimension.
  Value globalIndexScalar(ArrayRef<int64_t> tileShape, int64_t dim,
                          Value tileIndex) {
    Value tileExtent = constI32(tileShape[dim]);
    return b.create<::mlir::cuda_tile::MulIOp>(
                loc, tileIndex, tileExtent,
                ::mlir::cuda_tile::IntegerOverflowAttr::get(
                    ctx, ::mlir::cuda_tile::IntegerOverflow::NONE))
        .getResult();
  }

  //===-- Data movement ---------------------------------------------------===//

  /// Create make_tensor_view with static shape and strides.
  Value makeTensorView(Value basePtr, ArrayRef<int64_t> shape,
                       ArrayRef<int64_t> strides, Type elemType) {
    auto tvType =
        ::mlir::cuda_tile::TensorViewType::get(ctx, elemType, shape, strides);
    return b
        .create<::mlir::cuda_tile::MakeTensorViewOp>(loc, tvType, basePtr,
                                             /*dynamicShape=*/ValueRange{},
                                             /*dynamicStrides=*/ValueRange{})
        .getResult();
  }

  /// Create make_partition_view.
  Value makePartitionView(Value tensorView, ArrayRef<int64_t> tileShape,
                          bool zeroPad = false) {
    auto tvType = cast<::mlir::cuda_tile::TensorViewType>(tensorView.getType());
    SmallVector<int32_t> tileShapeI32(tileShape.begin(), tileShape.end());
    auto tileShapeAttr = DenseI32ArrayAttr::get(ctx, tileShapeI32);
    SmallVector<int32_t> dimMap(tileShape.size());
    std::iota(dimMap.begin(), dimMap.end(), 0);
    auto padAttr =
        zeroPad ? ::mlir::cuda_tile::PaddingValueAttr::get(
                      ctx, ::mlir::cuda_tile::PaddingValue::zero)
                : ::mlir::cuda_tile::PaddingValueAttr();
    auto pvType = ::mlir::cuda_tile::PartitionViewType::get(ctx, tileShapeAttr, tvType,
                                                    dimMap, padAttr);
    return b.create<::mlir::cuda_tile::MakePartitionViewOp>(loc, pvType, tensorView)
        .getResult();
  }

  /// Create get_tile_block_id → {x, y, z}.
  std::tuple<Value, Value, Value> getTileBlockId() {
    auto i32Tile = ::mlir::cuda_tile::TileType::get(ctx, {}, b.getI32Type());
    auto op =
        b.create<::mlir::cuda_tile::GetTileBlockIdOp>(loc, i32Tile, i32Tile, i32Tile);
    return {op.getBlockIdX(), op.getBlockIdY(), op.getBlockIdZ()};
  }

  /// Create load_view_tko weak → {tile, token}.
  std::pair<Value, Value> loadViewTko(Value view, ValueRange indices,
                                      ArrayRef<int64_t> tileShape,
                                      Type elemType) {
    auto tileType = ::mlir::cuda_tile::TileType::get(ctx, tileShape, elemType);
    auto tokenType = ::mlir::cuda_tile::TokenType::get(ctx);
    auto op = b.create<::mlir::cuda_tile::LoadViewTkoOp>(
        loc, tileType, tokenType,
        ::mlir::cuda_tile::MemoryOrderingSemantics::WEAK,
        /*memory_scope=*/::mlir::cuda_tile::MemoryScopeAttr(), view, indices,
        /*token=*/Value(),
        /*optimization_hints=*/::mlir::cuda_tile::OptimizationHintsAttr());
    return {op.getTile(), op.getResultToken()};
  }

  /// Create load_ptr_tko weak → {tile, token}.
  std::pair<Value, Value> loadPtrTko(Value ptrTile,
                                     ArrayRef<int64_t> tileShape,
                                     Type elemType, Value mask = {},
                                     Value padding = {}) {
    auto tileType = ::mlir::cuda_tile::TileType::get(ctx, tileShape, elemType);
    auto tokenType = ::mlir::cuda_tile::TokenType::get(ctx);
    auto op = b.create<::mlir::cuda_tile::LoadPtrTkoOp>(
        loc, tileType, tokenType,
        ::mlir::cuda_tile::MemoryOrderingSemantics::WEAK,
        /*memory_scope=*/::mlir::cuda_tile::MemoryScopeAttr(), ptrTile, mask, padding,
        /*token=*/Value(),
        /*optimization_hints=*/::mlir::cuda_tile::OptimizationHintsAttr());
    return {op.getResult(), op.getResultToken()};
  }

  /// Create store_view_tko weak → token.
  Value storeViewTko(Value tile, Value view, ValueRange indices) {
    auto tokenType = ::mlir::cuda_tile::TokenType::get(ctx);
    auto op = b.create<::mlir::cuda_tile::StoreViewTkoOp>(
        loc, tokenType,
        ::mlir::cuda_tile::MemoryOrderingSemantics::WEAK,
        /*memory_scope=*/::mlir::cuda_tile::MemoryScopeAttr(), tile, view, indices,
        /*token=*/Value(),
        /*optimization_hints=*/::mlir::cuda_tile::OptimizationHintsAttr());
    return op.getResultToken();
  }

  //===-- Arithmetic / Contractions ---------------------------------------===//

  /// Create mmaf (float matrix multiply-accumulate).
  Value mmaf(Value lhs, Value rhs, Value acc) {
    return b.create<::mlir::cuda_tile::MmaFOp>(loc, lhs, rhs, acc).getResult();
  }

  /// Create permute op for tile dimension reordering.
  Value permute(Value source, ArrayRef<int32_t> permutation, Type resultType) {
    auto permAttr = DenseI32ArrayAttr::get(ctx, permutation);
    return b.create<::mlir::cuda_tile::PermuteOp>(loc, resultType, source, permAttr)
        .getResult();
  }

  //===-- Elementwise ops (switch-dispatched) ------------------------------===//
  //
  // To add a new op: (1) add enum entry, (2) add StringSwitch case,
  // (3) add switch case with the create<>() call.

  // clang-format off
  enum class EWOp : uint8_t {
    // Binary float (lhs, rhs, rounding)
    AddF, SubF, MulF, DivF,
    // Binary float (lhs, rhs, no rounding)
    MaxF, MinF, Pow,
    // Binary integer (lhs, rhs, overflow)
    AddI, SubI, MulI,
    // Binary integer bitwise
    AndI, OrI, XOrI,
    // Binary integer (lhs, rhs, signedness)
    MaxI, MinI,
    // Ternary
    Select, Fma,
    // Unary float (simple)
    NegF, AbsF, Exp, Exp2, Log, Log2, Sin, Cos, Tanh, Ceil, Floor,
    // Unary float (with rounding/attrs)
    Sqrt, Rsqrt,
    // Unary integer
    NegI, AbsI,
    // Unknown
    Unknown
  };
  // clang-format on

  Value emitElementwise(StringRef opName, ValueRange operands) {
    Value a = operands.size() > 0 ? operands[0] : Value();
    Value c = operands.size() > 1 ? operands[1] : Value();
    Value d = operands.size() > 2 ? operands[2] : Value();
    auto rnd = ::mlir::cuda_tile::RoundingModeAttr::get(
        ctx, ::mlir::cuda_tile::RoundingMode::NEAREST_EVEN);
    auto ovf = ::mlir::cuda_tile::IntegerOverflowAttr::get(
        ctx, ::mlir::cuda_tile::IntegerOverflow::NONE);

    auto kind = llvm::StringSwitch<EWOp>(opName)
        .Case("addf", EWOp::AddF).Case("subf", EWOp::SubF)
        .Case("mulf", EWOp::MulF).Case("divf", EWOp::DivF)
        .Case("maxf", EWOp::MaxF).Case("minf", EWOp::MinF)
        .Case("pow",  EWOp::Pow)
        .Case("addi", EWOp::AddI).Case("subi", EWOp::SubI)
        .Case("muli", EWOp::MulI)
        .Case("andi", EWOp::AndI).Case("ori", EWOp::OrI)
        .Case("xori", EWOp::XOrI)
        .Case("maxi", EWOp::MaxI).Case("mini", EWOp::MinI)
        .Case("select", EWOp::Select).Case("fma", EWOp::Fma)
        .Case("negf", EWOp::NegF).Case("absf", EWOp::AbsF)
        .Case("exp",  EWOp::Exp) .Case("exp2", EWOp::Exp2)
        .Case("log",  EWOp::Log) .Case("log2", EWOp::Log2)
        .Case("sqrt", EWOp::Sqrt).Case("rsqrt",EWOp::Rsqrt)
        .Case("sin",  EWOp::Sin) .Case("cos",  EWOp::Cos)
        .Case("tanh", EWOp::Tanh)
        .Case("ceil", EWOp::Ceil).Case("floor", EWOp::Floor)
        .Case("negi", EWOp::NegI).Case("absi", EWOp::AbsI)
        .Default(EWOp::Unknown);

    // clang-format off
    switch (kind) {
    // Binary float with rounding
    case EWOp::AddF: return b.create<::mlir::cuda_tile::AddFOp>(loc, a, c, rnd).getResult();
    case EWOp::SubF: return b.create<::mlir::cuda_tile::SubFOp>(loc, a, c, rnd).getResult();
    case EWOp::MulF: return b.create<::mlir::cuda_tile::MulFOp>(loc, a, c, rnd).getResult();
    case EWOp::DivF: return b.create<::mlir::cuda_tile::DivFOp>(loc, a, c, rnd).getResult();
    // Binary float without rounding
    case EWOp::MaxF: return b.create<::mlir::cuda_tile::MaxFOp>(loc, a, c).getResult();
    case EWOp::MinF: return b.create<::mlir::cuda_tile::MinFOp>(loc, a, c).getResult();
    case EWOp::Pow:  return b.create<::mlir::cuda_tile::PowOp>(loc, a, c).getResult();
    // Binary integer
    case EWOp::AddI: return b.create<::mlir::cuda_tile::AddIOp>(loc, a, c, ovf).getResult();
    case EWOp::SubI: return b.create<::mlir::cuda_tile::SubIOp>(loc, a, c, ovf).getResult();
    case EWOp::MulI: return b.create<::mlir::cuda_tile::MulIOp>(loc, a, c, ovf).getResult();
    case EWOp::AndI: return b.create<::mlir::cuda_tile::AndIOp>(loc, a, c).getResult();
    case EWOp::OrI:  return b.create<::mlir::cuda_tile::OrIOp>(loc, a, c).getResult();
    case EWOp::XOrI: return b.create<::mlir::cuda_tile::XOrIOp>(loc, a, c).getResult();
    case EWOp::MaxI: return b.create<::mlir::cuda_tile::MaxIOp>(loc, a.getType(), ValueRange{a, c}, ::mlir::cuda_tile::Signedness::Signed).getResult();
    case EWOp::MinI: return b.create<::mlir::cuda_tile::MinIOp>(loc, a.getType(), ValueRange{a, c}, ::mlir::cuda_tile::Signedness::Signed).getResult();
    // Ternary
    case EWOp::Select: return b.create<::mlir::cuda_tile::SelectOp>(loc, a, c, d).getResult();
    case EWOp::Fma:    return b.create<::mlir::cuda_tile::FmaOp>(loc, a, c, d, rnd).getResult();
    // Unary float (simple)
    case EWOp::NegF:  return b.create<::mlir::cuda_tile::NegFOp>(loc, a).getResult();
    case EWOp::AbsF:  return b.create<::mlir::cuda_tile::AbsFOp>(loc, a).getResult();
    case EWOp::Exp:   return b.create<::mlir::cuda_tile::ExpOp>(loc, a).getResult();
    case EWOp::Exp2:  return b.create<::mlir::cuda_tile::Exp2Op>(loc, a).getResult();
    case EWOp::Log:   return b.create<::mlir::cuda_tile::LogOp>(loc, a).getResult();
    case EWOp::Log2:  return b.create<::mlir::cuda_tile::Log2Op>(loc, a).getResult();
    case EWOp::Sin:   return b.create<::mlir::cuda_tile::SinOp>(loc, a).getResult();
    case EWOp::Cos:   return b.create<::mlir::cuda_tile::CosOp>(loc, a).getResult();
    case EWOp::Tanh:  return b.create<::mlir::cuda_tile::TanHOp>(loc, a).getResult();
    case EWOp::Ceil:  return b.create<::mlir::cuda_tile::CeilOp>(loc, a).getResult();
    case EWOp::Floor: return b.create<::mlir::cuda_tile::FloorOp>(loc, a).getResult();
    // Unary float with rounding/attrs
    case EWOp::Sqrt:  return b.create<::mlir::cuda_tile::SqrtOp>(loc, a, rnd).getResult();
    case EWOp::Rsqrt: return b.create<::mlir::cuda_tile::RsqrtOp>(loc, a).getResult();
    // Unary integer
    case EWOp::NegI:  return b.create<::mlir::cuda_tile::NegIOp>(loc, a).getResult();
    case EWOp::AbsI:  return b.create<::mlir::cuda_tile::AbsIOp>(loc, a).getResult();
    // Fallback
    case EWOp::Unknown: return a;
    }
    // clang-format on
    llvm_unreachable("unhandled EWOp");
  }

  /// Emit a cuda_tile comparison (cmpf) from an arith::CmpFPredicate.
  Value emitCmpF(arith::CmpFPredicate pred, Value lhs, Value rhs) {
    using CP = ::mlir::cuda_tile::ComparisonPredicate;
    using CO = ::mlir::cuda_tile::ComparisonOrdering;
    CP ctPred;
    CO ctOrder;
    // clang-format off
    switch (pred) {
    case arith::CmpFPredicate::OEQ: ctPred = CP::EQUAL;                 ctOrder = CO::ORDERED;   break;
    case arith::CmpFPredicate::OGT: ctPred = CP::GREATER_THAN;          ctOrder = CO::ORDERED;   break;
    case arith::CmpFPredicate::OGE: ctPred = CP::GREATER_THAN_OR_EQUAL; ctOrder = CO::ORDERED;   break;
    case arith::CmpFPredicate::OLT: ctPred = CP::LESS_THAN;             ctOrder = CO::ORDERED;   break;
    case arith::CmpFPredicate::OLE: ctPred = CP::LESS_THAN_OR_EQUAL;    ctOrder = CO::ORDERED;   break;
    case arith::CmpFPredicate::ONE: ctPred = CP::NOT_EQUAL;             ctOrder = CO::ORDERED;   break;
    case arith::CmpFPredicate::UEQ: ctPred = CP::EQUAL;                 ctOrder = CO::UNORDERED; break;
    case arith::CmpFPredicate::UGT: ctPred = CP::GREATER_THAN;          ctOrder = CO::UNORDERED; break;
    case arith::CmpFPredicate::UGE: ctPred = CP::GREATER_THAN_OR_EQUAL; ctOrder = CO::UNORDERED; break;
    case arith::CmpFPredicate::ULT: ctPred = CP::LESS_THAN;             ctOrder = CO::UNORDERED; break;
    case arith::CmpFPredicate::ULE: ctPred = CP::LESS_THAN_OR_EQUAL;    ctOrder = CO::UNORDERED; break;
    case arith::CmpFPredicate::UNE: ctPred = CP::NOT_EQUAL;             ctOrder = CO::UNORDERED; break;
    default: ctPred = CP::EQUAL; ctOrder = CO::ORDERED; break;
    }
    // clang-format on
    auto predAttr = ::mlir::cuda_tile::ComparisonPredicateAttr::get(ctx, ctPred);
    auto orderAttr = ::mlir::cuda_tile::ComparisonOrderingAttr::get(ctx, ctOrder);
    return b.create<::mlir::cuda_tile::CmpFOp>(loc, predAttr, orderAttr, lhs, rhs)
        .getResult();
  }

  /// Emit a cuda_tile comparison (cmpi) from an arith::CmpIPredicate.
  Value emitCmpI(arith::CmpIPredicate pred, Value lhs, Value rhs) {
    using CP = ::mlir::cuda_tile::ComparisonPredicate;
    using SG = ::mlir::cuda_tile::Signedness;
    CP ctPred;
    SG ctSignedness;
    // clang-format off
    switch (pred) {
    case arith::CmpIPredicate::eq:  ctPred = CP::EQUAL;                 ctSignedness = SG::Signed;   break;
    case arith::CmpIPredicate::ne:  ctPred = CP::NOT_EQUAL;             ctSignedness = SG::Signed;   break;
    case arith::CmpIPredicate::slt: ctPred = CP::LESS_THAN;             ctSignedness = SG::Signed;   break;
    case arith::CmpIPredicate::sle: ctPred = CP::LESS_THAN_OR_EQUAL;    ctSignedness = SG::Signed;   break;
    case arith::CmpIPredicate::sgt: ctPred = CP::GREATER_THAN;          ctSignedness = SG::Signed;   break;
    case arith::CmpIPredicate::sge: ctPred = CP::GREATER_THAN_OR_EQUAL; ctSignedness = SG::Signed;   break;
    case arith::CmpIPredicate::ult: ctPred = CP::LESS_THAN;             ctSignedness = SG::Unsigned; break;
    case arith::CmpIPredicate::ule: ctPred = CP::LESS_THAN_OR_EQUAL;    ctSignedness = SG::Unsigned; break;
    case arith::CmpIPredicate::ugt: ctPred = CP::GREATER_THAN;          ctSignedness = SG::Unsigned; break;
    case arith::CmpIPredicate::uge: ctPred = CP::GREATER_THAN_OR_EQUAL; ctSignedness = SG::Unsigned; break;
    }
    // clang-format on
    auto predAttr = ::mlir::cuda_tile::ComparisonPredicateAttr::get(ctx, ctPred);
    auto signAttr = ::mlir::cuda_tile::SignednessAttr::get(ctx, ctSignedness);
    return b.create<::mlir::cuda_tile::CmpIOp>(loc, predAttr, lhs, rhs, signAttr)
        .getResult();
  }

  Value emitDivI(Value lhs, Value rhs, ::mlir::cuda_tile::Signedness signedness,
                 ::mlir::cuda_tile::RoundingMode rounding) {
    auto signAttr = ::mlir::cuda_tile::SignednessAttr::get(ctx, signedness);
    auto rndAttr = ::mlir::cuda_tile::RoundingModeAttr::get(ctx, rounding);
    return b.create<::mlir::cuda_tile::DivIOp>(loc, lhs, rhs, signAttr, rndAttr)
        .getResult();
  }

  Value emitRemI(Value lhs, Value rhs, ::mlir::cuda_tile::Signedness signedness) {
    auto signAttr = ::mlir::cuda_tile::SignednessAttr::get(ctx, signedness);
    return b.create<::mlir::cuda_tile::RemIOp>(loc, lhs, rhs, signAttr).getResult();
  }

  Value emitIToF(Value source, ArrayRef<int64_t> resultShape,
                 Type floatElemType) {
    auto resultType = ::mlir::cuda_tile::TileType::get(ctx, resultShape, floatElemType);
    auto signAttr =
        ::mlir::cuda_tile::SignednessAttr::get(ctx, ::mlir::cuda_tile::Signedness::Signed);
    auto rndAttr = ::mlir::cuda_tile::RoundingModeAttr::get(
        ctx, ::mlir::cuda_tile::RoundingMode::NEAREST_EVEN);
    return b.create<::mlir::cuda_tile::IToFOp>(loc, resultType, source, signAttr,
                                       rndAttr)
        .getResult();
  }

  //===-- Reductions ------------------------------------------------------===//

  /// Create a reduce op with the given combiner.
  /// combiner is one of: "addf", "mulf", "maxf", "minf", "addi", etc.
  /// reduceDim is the dimension to reduce along.
  /// Returns the reduced tile value.
  Value reduce(Value input, int64_t reduceDim, StringRef combiner,
               ArrayRef<int64_t> resultShape, Type elemType) {
    auto resultTileType =
        ::mlir::cuda_tile::TileType::get(ctx, resultShape, elemType);

    // Determine identity value.
    Attribute identity;
    if (combiner == "addf" || combiner == "addi") {
      if (isa<FloatType>(elemType))
        identity = FloatAttr::get(elemType, 0.0);
      else
        identity = IntegerAttr::get(elemType, 0);
    } else if (combiner == "mulf" || combiner == "muli") {
      if (isa<FloatType>(elemType))
        identity = FloatAttr::get(elemType, 1.0);
      else
        identity = IntegerAttr::get(elemType, 1);
    } else if (combiner == "maxf") {
      identity = FloatAttr::get(
          elemType, -std::numeric_limits<double>::infinity());
    } else if (combiner == "minf") {
      identity = FloatAttr::get(
          elemType, std::numeric_limits<double>::infinity());
    } else if (combiner == "maxi") {
      identity = IntegerAttr::get(elemType, INT64_MIN);
    } else if (combiner == "mini") {
      identity = IntegerAttr::get(elemType, INT64_MAX);
    } else {
      // Default to additive identity.
      if (isa<FloatType>(elemType))
        identity = FloatAttr::get(elemType, 0.0);
      else
        identity = IntegerAttr::get(elemType, 0);
    }

    auto identities = b.getArrayAttr({identity});
    auto dimAttr = b.getI32IntegerAttr(reduceDim);

    auto reduceOp = b.create<::mlir::cuda_tile::ReduceOp>(
        loc, TypeRange{resultTileType}, ValueRange{input}, dimAttr,
        identities);

    // Build the combiner body region.
    Block *body = b.createBlock(&reduceOp.getBody());
    // Region args: [current_elem, prev_accum] for each operand.
    auto scalarTileType = ::mlir::cuda_tile::TileType::get(ctx, {}, elemType);
    body->addArgument(scalarTileType, loc); // current
    body->addArgument(scalarTileType, loc); // accumulator

    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToEnd(body);

    Value cur = body->getArgument(0);
    Value acc = body->getArgument(1);
    Value result = emitElementwise(combiner, {cur, acc});
    b.create<::mlir::cuda_tile::YieldOp>(loc, ValueRange{result});

    return reduceOp.getResult(0);
  }

  //===-- Shape manipulation -----------------------------------------------===//

  /// Reshape a tile to a new shape (same number of elements).
  /// E.g., tile<4xf32> → tile<4x1xf32>
  Value reshape(Value source, ArrayRef<int64_t> newShape, Type elemType) {
    auto resultType = ::mlir::cuda_tile::TileType::get(ctx, newShape, elemType);
    return b.create<::mlir::cuda_tile::ReshapeOp>(loc, resultType, source).getResult();
  }

  /// Extract a subtile from a tile.
  Value extractSubtile(Value source, ArrayRef<int64_t> resultShape,
                       Type elemType, ValueRange indices) {
    auto resultType = ::mlir::cuda_tile::TileType::get(ctx, resultShape, elemType);
    return b.create<::mlir::cuda_tile::ExtractOp>(loc, resultType, source, indices)
        .getResult();
  }

  /// Concatenate two tiles along a dimension.
  Value cat(Value lhs, Value rhs, int64_t dim, ArrayRef<int64_t> resultShape,
            Type elemType) {
    auto resultType = ::mlir::cuda_tile::TileType::get(ctx, resultShape, elemType);
    return b.create<::mlir::cuda_tile::CatOp>(loc, resultType, lhs, rhs,
                                      static_cast<uint64_t>(dim))
        .getResult();
  }

  /// Broadcast a tile: expand 1-dims to match the new shape.
  /// E.g., tile<4x1xf32> → tile<4x8xf32>
  Value broadcastTile(Value source, ArrayRef<int64_t> newShape, Type elemType) {
    auto resultType = ::mlir::cuda_tile::TileType::get(ctx, newShape, elemType);
    return b.create<::mlir::cuda_tile::BroadcastOp>(loc, resultType, source)
        .getResult();
  }

  /// Attach a same_elements assumption to a broadcast-like tile.
  Value assumeSameElements(Value value, ArrayRef<int64_t> chunkShape) {
    SmallVector<int64_t> chunk(chunkShape.begin(), chunkShape.end());
    auto attr = ::mlir::cuda_tile::SameElementsAttr::get(
        ctx, DenseI64ArrayAttr::get(ctx, chunk));
    return b.create<::mlir::cuda_tile::AssumeOp>(loc, value.getType(), value, attr)
        .getResult();
  }

  //===-- Pointer arithmetic -----------------------------------------------===//

  /// Offset a scalar pointer by a constant number of elements.
  /// result = base + offset * sizeof(element)
  Value offsetPtr(Value basePtr, int64_t offset) {
    auto offsetVal = constI32(offset);
    return b.create<::mlir::cuda_tile::OffsetOp>(loc, basePtr.getType(), basePtr,
                                         offsetVal)
        .getResult();
  }

  /// Offset a tile of pointers by a tile of element offsets.
  Value offsetPtrTile(Value ptrTile, Value offsets) {
    return b.create<::mlir::cuda_tile::OffsetOp>(loc, ptrTile.getType(), ptrTile,
                                         offsets)
        .getResult();
  }

  //===-- Control flow ----------------------------------------------------===//

  /// Create a for loop. Returns the ForOp. Caller must build the body
  /// and call endFor() afterwards.
  ::mlir::cuda_tile::ForOp beginFor(Value lb, Value ub, Value step,
                            ValueRange initArgs) {
    auto forOp = b.create<::mlir::cuda_tile::ForOp>(loc, lb, ub, step, initArgs);
    // ForOp with initArgs doesn't create a terminator automatically.
    // Add a placeholder ContinueOp that endFor() will replace.
    Block *body = forOp.getBody();
    if (!body->mightHaveTerminator()) {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToEnd(body);
      // Yield the iter args back as-is (placeholder).
      b.create<::mlir::cuda_tile::ContinueOp>(loc, forOp.getRegionIterValues());
    }
    // Set insertion point before the terminator for body ops.
    b.setInsertionPoint(body->getTerminator());
    return forOp;
  }

  ::mlir::cuda_tile::IfOp beginIf(Value condition) {
    auto ifOp = b.create<::mlir::cuda_tile::IfOp>(loc, TypeRange{}, condition);
    Block *thenBlock = new Block();
    ifOp.getThenRegion().push_back(thenBlock);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToEnd(thenBlock);
      b.create<::mlir::cuda_tile::YieldOp>(loc);
    }
    b.setInsertionPoint(thenBlock->getTerminator());
    return ifOp;
  }

  void endIf(::mlir::cuda_tile::IfOp ifOp) { b.setInsertionPointAfter(ifOp); }

  /// Finalize the for loop body: replace the implicit ContinueOp with one
  /// that yields the given values, then restore insertion point after the loop.
  void endFor(::mlir::cuda_tile::ForOp forOp, ValueRange yieldValues) {
    // Replace the implicit (empty) ContinueOp with one carrying yield values.
    auto *terminator = forOp.getBody()->getTerminator();
    b.setInsertionPoint(terminator);
    b.create<::mlir::cuda_tile::ContinueOp>(loc, yieldValues);
    terminator->erase();
    // Restore insertion point to after the for op.
    b.setInsertionPointAfter(forOp);
  }

  /// Restore the builder insertion point to just before the entry's terminator.
  /// Call this after building ops with nested regions (reduce, for) to ensure
  /// subsequent ops go in the right place.
  void restoreEntryInsertionPoint() {
    Block *entryBlock = &entryOp.getBody().front();
    b.setInsertionPoint(entryBlock->getTerminator());
  }

  void emitReturn() {
    // EntryOp already has an implicit ReturnOp terminator. Nothing to do
    // unless we want to return values (which entry functions don't).
  }

  //===-- Serialization ---------------------------------------------------===//

  /// Serialize the cuda_tile module to tilebc bytecode.
  LogicalResult serialize(std::string &output) {
    llvm::raw_string_ostream os(output);
    return ::mlir::cuda_tile::writeBytecode(os, moduleOp);
  }

  /// Get the module for inspection/debugging.
  ::mlir::cuda_tile::ModuleOp getModule() { return moduleOp; }
  OpBuilder &builder() { return b; }
  MLIRContext *getContext() { return ctx; }
  Location getLoc() { return loc; }

private:
  MLIRContext *ctx;
  Location loc;
  OpBuilder b;
  ::mlir::cuda_tile::ModuleOp moduleOp = nullptr;
  ::mlir::cuda_tile::EntryOp entryOp = nullptr;
  SmallVector<Value> entryArgs;
};

static Value offsetPtrByBytes(CudaTileOpEmitter &e, Value basePtr,
                              Type elemType, int64_t byteOffset) {
  if (byteOffset == 0)
    return basePtr;
  int64_t elemBytes = getElementByteWidth(elemType);
  if (elemBytes <= 0 || byteOffset % elemBytes != 0) {
    LLVM_DEBUG(llvm::dbgs()
               << "[cuda_tile] ignoring unsupported binding byte offset "
               << byteOffset << " for element type " << elemType << "\n");
    return basePtr;
  }
  return e.offsetPtr(basePtr, byteOffset / elemBytes);
}

static const CudaTileBindingPlan *
findBindingPlan(ArrayRef<CudaTileBindingPlan> bindings, int64_t bindingIndex) {
  for (const CudaTileBindingPlan &binding : bindings) {
    if (binding.binding == bindingIndex)
      return &binding;
  }
  if (bindingIndex >= 0 && bindingIndex < (int64_t)bindings.size())
    return &bindings[bindingIndex];
  return nullptr;
}

static const CudaTileBindingPlan *
findBindingPlanForMemref(ArrayRef<CudaTileBindingPlan> bindings, Value memref) {
  for (const CudaTileBindingPlan &binding : bindings) {
    if (binding.memref == memref)
      return &binding;
  }
  return nullptr;
}

static int64_t getNumBindingArgs(ArrayRef<CudaTileBindingPlan> bindings) {
  int64_t maxBinding = -1;
  for (const CudaTileBindingPlan &binding : bindings)
    maxBinding = std::max(maxBinding, binding.binding);
  if (maxBinding >= 0)
    return maxBinding + 1;
  return bindings.size();
}

static Value getSubspanArg(CudaTileOpEmitter &e,
                           const CudaTileBindingPlan &binding,
                           Type elemType) {
  int64_t bindingIndex = binding.binding >= 0 ? binding.binding : 0;
  Value ptr = e.getArg(bindingIndex);
  if (!binding.hasStaticByteOffset) {
    LLVM_DEBUG(llvm::dbgs()
               << "[cuda_tile] binding " << bindingIndex
               << " has dynamic byte offset; using base pointer\n");
    return ptr;
  }
  return offsetPtrByBytes(e, ptr, elemType, binding.byteOffset);
}

static Value getBindingArg(CudaTileOpEmitter &e,
                           ArrayRef<CudaTileBindingPlan> bindings,
                           int64_t bindingIndex, Type elemType) {
  Value ptr = e.getArg(bindingIndex);
  const CudaTileBindingPlan *binding =
      findBindingPlan(bindings, bindingIndex);
  if (!binding)
    return ptr;
  if (!binding->hasStaticByteOffset) {
    LLVM_DEBUG(llvm::dbgs()
               << "[cuda_tile] binding " << bindingIndex
               << " has dynamic byte offset; using base pointer\n");
    return ptr;
  }
  return offsetPtrByBytes(e, ptr, elemType, binding->byteOffset);
}

struct CudaTileGatherSource {
  Value tensor;
  int64_t binding = -1;
  int64_t byteOffset = 0;
  bool hasStaticByteOffset = true;
  SmallVector<int64_t> logicalShape;
  SmallVector<int64_t> physicalShape;
  SmallVector<int64_t> loadOffsets;
  SmallVector<int64_t> loadStrides;
  Type elemType;
};

static SmallVector<CudaTileGatherSource, 4>
buildGatherSources(const CudaTileKernelPlan &plan) {
  SmallVector<CudaTileGatherSource, 4> sources;
  for (const CudaTileOperandPlan &operand : plan.operands) {
    if (!operand.isDispatchLoad)
      continue;
    auto loadOp =
        operand.value.getDefiningOp<IREE::TensorExt::DispatchTensorLoadOp>();
    if (!loadOp)
      continue;

    const CudaTileBindingPlan *binding =
        findBindingPlanForMemref(plan.bindingShapes, loadOp.getSource());
    if (!binding)
      continue;

    CudaTileGatherSource source;
    source.tensor = operand.value;
    source.binding = binding->binding >= 0 ? binding->binding
                                           : operand.binding;
    source.byteOffset = binding->byteOffset;
    source.hasStaticByteOffset = binding->hasStaticByteOffset;
    source.logicalShape = operand.logicalShape;
    source.physicalShape = operand.physicalShape.empty()
                               ? binding->shape
                               : operand.physicalShape;
    source.loadOffsets = operand.offsets;
    source.loadStrides = operand.strides;
    source.elemType = getElementTypeFromType(operand.value.getType());
    sources.push_back(std::move(source));
  }
  return sources;
}

static std::optional<double> getScalarConstantValue(Value value) {
  auto cstOp = value.getDefiningOp<arith::ConstantOp>();
  if (!cstOp)
    return std::nullopt;
  Attribute attr = cstOp.getValue();
  if (auto fAttr = dyn_cast<FloatAttr>(attr))
    return fAttr.getValueAsDouble();
  if (auto iAttr = dyn_cast<IntegerAttr>(attr))
    return static_cast<double>(iAttr.getInt());
  return std::nullopt;
}

static Type getCudaTileConstantElementType(CudaTileOpEmitter &e, Value value,
                                           Type defaultElemType) {
  Type type = value.getType();
  if (type.isIndex())
    return e.builder().getI32Type();
  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth() > 32 ? e.builder().getI32Type() : type;
  return defaultElemType;
}

static bool resolveCudaTileBodyOperands(CudaTileOpEmitter &e,
                                        Operation *bodyOp,
                                        DenseMap<Value, Value> &bodyMap,
                                        ArrayRef<int64_t> tileShape,
                                        Type elemType,
                                        SmallVectorImpl<Value> &opInputs) {
  for (Value operand : bodyOp->getOperands()) {
    auto it = bodyMap.find(operand);
    if (it != bodyMap.end()) {
      opInputs.push_back(it->second);
      continue;
    }
    if (std::optional<double> value = getScalarConstantValue(operand)) {
      Type constElemType =
          getCudaTileConstantElementType(e, operand, elemType);
      opInputs.push_back(e.constSplat(tileShape, constElemType, *value));
      continue;
    }
    return false;
  }
  return true;
}

static const CudaTileGatherSource *
findGatherSource(ArrayRef<CudaTileGatherSource> sources, Value tensor) {
  for (const CudaTileGatherSource &source : sources) {
    if (source.tensor == tensor)
      return &source;
  }
  return nullptr;
}

static std::optional<SmallVector<int64_t>>
normalizeStaticIndexList(ArrayRef<int64_t> values, int64_t rank,
                         int64_t defaultValue) {
  SmallVector<int64_t> normalized(rank, defaultValue);
  if (values.empty())
    return normalized;
  if (values.size() != static_cast<size_t>(rank))
    return std::nullopt;
  for (auto [i, value] : llvm::enumerate(values)) {
    if (ShapedType::isDynamic(value))
      return std::nullopt;
    normalized[i] = value;
  }
  return normalized;
}

static Value resolveCudaTileIndexOperand(CudaTileOpEmitter &e, Value operand,
                                         DenseMap<Value, Value> &bodyMap,
                                         ArrayRef<int64_t> tileShape) {
  auto it = bodyMap.find(operand);
  if (it != bodyMap.end())
    return it->second;
  if (std::optional<double> value = getScalarConstantValue(operand))
    return e.constSplat(tileShape, e.builder().getI32Type(), *value);
  return {};
}

static Value andTileMasks(CudaTileOpEmitter &e, Value lhs, Value rhs) {
  if (!lhs)
    return rhs;
  if (!rhs)
    return lhs;
  return e.emitElementwise("andi", {lhs, rhs});
}

static Value buildOutputInBoundsMask(CudaTileOpEmitter &e,
                                     ArrayRef<int64_t> tileShape,
                                     ArrayRef<Value> outputTileIndices,
                                     ArrayRef<int64_t> outputShape) {
  Value mask = e.constSplat(tileShape, e.builder().getI1Type(), 1);
  if (outputShape.empty() ||
      outputTileIndices.size() < static_cast<size_t>(outputShape.size()))
    return mask;

  Type i32Type = e.builder().getI32Type();
  for (auto [dim, extent] : llvm::enumerate(outputShape)) {
    if (ShapedType::isDynamic(extent))
      continue;
    Value index = e.globalIndexTile(tileShape, dim, outputTileIndices[dim]);
    Value upper = e.constSplat(tileShape, i32Type, extent);
    Value inBounds = e.emitCmpI(arith::CmpIPredicate::ult, index, upper);
    mask = andTileMasks(e, mask, inBounds);
  }
  return mask;
}

struct CudaTileAffineIndexExpr {
  SmallVector<int64_t> coeffs;
  int64_t constant = 0;
};

static CudaTileAffineIndexExpr makeConstantAffineIndexExpr(int64_t outputRank,
                                                           int64_t value) {
  CudaTileAffineIndexExpr expr;
  expr.coeffs.assign(outputRank, 0);
  expr.constant = value;
  return expr;
}

static std::optional<int64_t> getIntegerConstantValue(Value value) {
  auto cstOp = value.getDefiningOp<arith::ConstantOp>();
  if (!cstOp)
    return std::nullopt;
  if (auto intAttr = dyn_cast<IntegerAttr>(cstOp.getValue()))
    return intAttr.getInt();
  return std::nullopt;
}

static CudaTileAffineIndexExpr addAffineIndexExpr(
    const CudaTileAffineIndexExpr &lhs,
    const CudaTileAffineIndexExpr &rhs) {
  CudaTileAffineIndexExpr result;
  result.coeffs.assign(lhs.coeffs.begin(), lhs.coeffs.end());
  for (auto [i, coeff] : llvm::enumerate(rhs.coeffs))
    result.coeffs[i] += coeff;
  result.constant = lhs.constant + rhs.constant;
  return result;
}

static CudaTileAffineIndexExpr scaleAffineIndexExpr(
    const CudaTileAffineIndexExpr &expr, int64_t scale) {
  CudaTileAffineIndexExpr result;
  result.coeffs.reserve(expr.coeffs.size());
  for (int64_t coeff : expr.coeffs)
    result.coeffs.push_back(coeff * scale);
  result.constant = expr.constant * scale;
  return result;
}

static std::optional<CudaTileAffineIndexExpr>
matchAffineIndexExpr(Value value, int64_t outputRank) {
  if (std::optional<int64_t> cst = getIntegerConstantValue(value))
    return makeConstantAffineIndexExpr(outputRank, *cst);

  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return std::nullopt;

  if (auto indexOp = dyn_cast<linalg::IndexOp>(defOp)) {
    int64_t dim = indexOp.getDim();
    if (dim < 0 || dim >= outputRank)
      return std::nullopt;
    CudaTileAffineIndexExpr expr =
        makeConstantAffineIndexExpr(outputRank, 0);
    expr.coeffs[dim] = 1;
    return expr;
  }

  if (isa<arith::IndexCastOp, arith::IndexCastUIOp>(defOp))
    return matchAffineIndexExpr(defOp->getOperand(0), outputRank);

  if (isa<arith::AddIOp>(defOp)) {
    auto lhs = matchAffineIndexExpr(defOp->getOperand(0), outputRank);
    auto rhs = matchAffineIndexExpr(defOp->getOperand(1), outputRank);
    if (!lhs || !rhs)
      return std::nullopt;
    return addAffineIndexExpr(*lhs, *rhs);
  }

  if (isa<arith::SubIOp>(defOp)) {
    auto lhs = matchAffineIndexExpr(defOp->getOperand(0), outputRank);
    auto rhs = matchAffineIndexExpr(defOp->getOperand(1), outputRank);
    if (!lhs || !rhs)
      return std::nullopt;
    return addAffineIndexExpr(*lhs, scaleAffineIndexExpr(*rhs, -1));
  }

  if (isa<arith::MulIOp>(defOp)) {
    if (std::optional<int64_t> lhsCst =
            getIntegerConstantValue(defOp->getOperand(0))) {
      auto rhs = matchAffineIndexExpr(defOp->getOperand(1), outputRank);
      if (!rhs)
        return std::nullopt;
      return scaleAffineIndexExpr(*rhs, *lhsCst);
    }
    if (std::optional<int64_t> rhsCst =
            getIntegerConstantValue(defOp->getOperand(1))) {
      auto lhs = matchAffineIndexExpr(defOp->getOperand(0), outputRank);
      if (!lhs)
        return std::nullopt;
      return scaleAffineIndexExpr(*lhs, *rhsCst);
    }
  }

  return std::nullopt;
}

static bool isFlatLinalgIndex(Value value) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return false;
  if (auto indexOp = dyn_cast<linalg::IndexOp>(defOp))
    return indexOp.getDim() == 0;
  if (isa<arith::IndexCastOp, arith::IndexCastUIOp>(defOp))
    return isFlatLinalgIndex(defOp->getOperand(0));
  return false;
}

static std::optional<CudaTileAffineIndexExpr>
matchFlat2DAffineIndexExpr(Value value, int64_t innerDim) {
  constexpr int64_t kUnflattenedRank = 2;
  if (std::optional<int64_t> cst = getIntegerConstantValue(value))
    return makeConstantAffineIndexExpr(kUnflattenedRank, *cst);

  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return std::nullopt;

  if (isa<arith::IndexCastOp, arith::IndexCastUIOp>(defOp))
    return matchFlat2DAffineIndexExpr(defOp->getOperand(0), innerDim);

  if (isa<arith::DivSIOp, arith::FloorDivSIOp>(defOp) &&
      isFlatLinalgIndex(defOp->getOperand(0))) {
    std::optional<int64_t> divisor =
        getIntegerConstantValue(defOp->getOperand(1));
    if (!divisor || *divisor != innerDim)
      return std::nullopt;
    CudaTileAffineIndexExpr expr =
        makeConstantAffineIndexExpr(kUnflattenedRank, 0);
    expr.coeffs[0] = 1;
    return expr;
  }

  if (isa<arith::RemSIOp>(defOp) && isFlatLinalgIndex(defOp->getOperand(0))) {
    std::optional<int64_t> divisor =
        getIntegerConstantValue(defOp->getOperand(1));
    if (!divisor || *divisor != innerDim)
      return std::nullopt;
    CudaTileAffineIndexExpr expr =
        makeConstantAffineIndexExpr(kUnflattenedRank, 0);
    expr.coeffs[1] = 1;
    return expr;
  }

  if (isa<arith::AddIOp>(defOp)) {
    auto lhs = matchFlat2DAffineIndexExpr(defOp->getOperand(0), innerDim);
    auto rhs = matchFlat2DAffineIndexExpr(defOp->getOperand(1), innerDim);
    if (!lhs || !rhs)
      return std::nullopt;
    return addAffineIndexExpr(*lhs, *rhs);
  }

  if (isa<arith::SubIOp>(defOp)) {
    auto lhs = matchFlat2DAffineIndexExpr(defOp->getOperand(0), innerDim);
    auto rhs = matchFlat2DAffineIndexExpr(defOp->getOperand(1), innerDim);
    if (!lhs || !rhs)
      return std::nullopt;
    return addAffineIndexExpr(*lhs, scaleAffineIndexExpr(*rhs, -1));
  }

  if (isa<arith::MulIOp>(defOp)) {
    if (std::optional<int64_t> lhsCst =
            getIntegerConstantValue(defOp->getOperand(0))) {
      auto rhs = matchFlat2DAffineIndexExpr(defOp->getOperand(1), innerDim);
      if (!rhs)
        return std::nullopt;
      return scaleAffineIndexExpr(*rhs, *lhsCst);
    }
    if (std::optional<int64_t> rhsCst =
            getIntegerConstantValue(defOp->getOperand(1))) {
      auto lhs = matchFlat2DAffineIndexExpr(defOp->getOperand(0), innerDim);
      if (!lhs)
        return std::nullopt;
      return scaleAffineIndexExpr(*lhs, *rhsCst);
    }
  }

  return std::nullopt;
}

static bool affineIndicesAreInBounds(
    ArrayRef<CudaTileAffineIndexExpr> sourceIndexExprs,
    ArrayRef<int64_t> logicalShape, ArrayRef<int64_t> outputShape) {
  if (logicalShape.size() != sourceIndexExprs.size())
    return false;
  if (llvm::any_of(logicalShape, ShapedType::isDynamic) ||
      llvm::any_of(outputShape, ShapedType::isDynamic))
    return false;

  for (auto [sourceDim, expr] : llvm::enumerate(sourceIndexExprs)) {
    int64_t minIndex = expr.constant;
    int64_t maxIndex = expr.constant;
    if (expr.coeffs.size() != outputShape.size())
      return false;
    for (auto [outDim, coeff] : llvm::enumerate(expr.coeffs)) {
      int64_t maxOutIndex = outputShape[outDim] - 1;
      if (coeff >= 0)
        maxIndex += coeff * maxOutIndex;
      else
        minIndex += coeff * maxOutIndex;
    }
    if (minIndex < 0 || maxIndex >= logicalShape[sourceDim])
      return false;
  }
  return true;
}

static Value concatenateOneElementTiles(CudaTileOpEmitter &e,
                                        SmallVector<Value> pieces,
                                        Type elemType) {
  SmallVector<std::pair<Value, int64_t>> work;
  work.reserve(pieces.size());
  for (Value piece : pieces)
    work.push_back({piece, 1});

  while (work.size() > 1) {
    SmallVector<std::pair<Value, int64_t>> next;
    for (int64_t i = 0; i < static_cast<int64_t>(work.size()); i += 2) {
      if (i + 1 == static_cast<int64_t>(work.size())) {
        next.push_back(work[i]);
        continue;
      }
      int64_t lhsSize = work[i].second;
      int64_t rhsSize = work[i + 1].second;
      int64_t resultSize = lhsSize + rhsSize;
      Value combined = e.cat(work[i].first, work[i + 1].first, 0,
                             {resultSize}, elemType);
      next.push_back({combined, resultSize});
    }
    work = std::move(next);
  }

  return work.empty() ? Value() : work.front().first;
}

static Value emitTensorExtractAffineViewGather(
    CudaTileOpEmitter &e, tensor::ExtractOp extractOp,
    const CudaTileGatherSource &source, ArrayRef<int64_t> tileShape,
    ArrayRef<Value> outputTileIndices, ArrayRef<int64_t> outputShape) {
  int64_t sourceRank = extractOp.getIndices().size();
  int64_t outputRank = outputShape.size();
  if (outputRank == 0 || tileShape.size() != static_cast<size_t>(outputRank) ||
      outputTileIndices.size() < static_cast<size_t>(outputRank))
    return {};

  SmallVector<int64_t> logicalShape = source.logicalShape;
  if (logicalShape.empty())
    logicalShape = getStaticShapeFromType(extractOp.getTensor().getType());
  SmallVector<int64_t> physicalShape =
      source.physicalShape.empty() ? logicalShape : source.physicalShape;
  if (sourceRank == 0 ||
      logicalShape.size() != static_cast<size_t>(sourceRank) ||
      physicalShape.size() != static_cast<size_t>(sourceRank))
    return {};
  if (llvm::any_of(logicalShape, ShapedType::isDynamic) ||
      llvm::any_of(physicalShape, ShapedType::isDynamic) ||
      llvm::any_of(outputShape, ShapedType::isDynamic))
    return {};

  std::optional<SmallVector<int64_t>> loadOffsets =
      normalizeStaticIndexList(source.loadOffsets, sourceRank, 0);
  std::optional<SmallVector<int64_t>> loadStrides =
      normalizeStaticIndexList(source.loadStrides, sourceRank, 1);
  if (!loadOffsets || !loadStrides)
    return {};

  SmallVector<CudaTileAffineIndexExpr> sourceIndexExprs;
  sourceIndexExprs.reserve(sourceRank);
  for (Value index : extractOp.getIndices()) {
    auto expr = matchAffineIndexExpr(index, outputRank);
    if (!expr)
      return {};
    sourceIndexExprs.push_back(std::move(*expr));
  }

  if (!affineIndicesAreInBounds(sourceIndexExprs, logicalShape, outputShape))
    return {};

  SmallVector<int64_t> physicalStrides = computeRowMajorStrides(physicalShape);
  SmallVector<int64_t> viewStrides(outputRank, 0);
  int64_t baseOffset = 0;
  for (int64_t sourceDim = 0; sourceDim < sourceRank; ++sourceDim) {
    int64_t strideScale = (*loadStrides)[sourceDim] *
                          physicalStrides[sourceDim];
    baseOffset += ((*loadOffsets)[sourceDim] +
                   (*loadStrides)[sourceDim] *
                       sourceIndexExprs[sourceDim].constant) *
                  physicalStrides[sourceDim];
    for (int64_t outDim = 0; outDim < outputRank; ++outDim)
      viewStrides[outDim] += sourceIndexExprs[sourceDim].coeffs[outDim] *
                             strideScale;
  }

  // TensorView strides are the simplest robust lowering for affine gathers, but
  // reverse/negative-stride gathers need a separate legality path.
  if (baseOffset < 0 ||
      llvm::any_of(viewStrides, [](int64_t stride) { return stride < 0; }))
    return {};

  Value basePtr = e.getArg(source.binding);
  if (source.hasStaticByteOffset) {
    basePtr =
        offsetPtrByBytes(e, basePtr, source.elemType, source.byteOffset);
  } else {
    LLVM_DEBUG(llvm::dbgs()
               << "[cuda_tile] affine gather source binding "
               << source.binding
               << " has dynamic byte offset; using base pointer\n");
  }
  if (baseOffset != 0)
    basePtr = e.offsetPtr(basePtr, baseOffset);

  auto view = e.makeTensorView(basePtr, outputShape, viewStrides,
                               source.elemType);
  auto partition = e.makePartitionView(view, tileShape, /*zeroPad=*/true);
  auto [tile, token] =
      e.loadViewTko(partition, outputTileIndices, tileShape, source.elemType);
  return tile;
}

static Value emitTensorExtractFlat2DAffineViewGather(
    CudaTileOpEmitter &e, tensor::ExtractOp extractOp,
    const CudaTileGatherSource &source, ArrayRef<int64_t> tileShape,
    ArrayRef<int64_t> outputShape) {
  if (outputShape.size() != 1 || tileShape.size() != 1)
    return {};
  int64_t outputElementCount = outputShape[0];
  int64_t flatTileCount = tileShape[0];
  if (ShapedType::isDynamic(outputElementCount) || outputElementCount <= 0 ||
      outputElementCount > flatTileCount || flatTileCount > 256)
    return {};

  int64_t sourceRank = extractOp.getIndices().size();
  SmallVector<int64_t> logicalShape = source.logicalShape;
  if (logicalShape.empty())
    logicalShape = getStaticShapeFromType(extractOp.getTensor().getType());
  SmallVector<int64_t> physicalShape =
      source.physicalShape.empty() ? logicalShape : source.physicalShape;
  if (sourceRank == 0 ||
      logicalShape.size() != static_cast<size_t>(sourceRank) ||
      physicalShape.size() != static_cast<size_t>(sourceRank))
    return {};
  if (llvm::any_of(logicalShape, ShapedType::isDynamic) ||
      llvm::any_of(physicalShape, ShapedType::isDynamic))
    return {};

  std::optional<SmallVector<int64_t>> loadOffsets =
      normalizeStaticIndexList(source.loadOffsets, sourceRank, 0);
  std::optional<SmallVector<int64_t>> loadStrides =
      normalizeStaticIndexList(source.loadStrides, sourceRank, 1);
  if (!loadOffsets || !loadStrides)
    return {};

  for (int64_t innerDim = 1; innerDim <= outputElementCount; ++innerDim) {
    if ((outputElementCount % innerDim) != 0)
      continue;
    SmallVector<int64_t> unflattenedOutputShape = {
        outputElementCount / innerDim, innerDim};

    SmallVector<CudaTileAffineIndexExpr> sourceIndexExprs;
    sourceIndexExprs.reserve(sourceRank);
    bool matched = true;
    for (Value index : extractOp.getIndices()) {
      auto expr = matchFlat2DAffineIndexExpr(index, innerDim);
      if (!expr) {
        matched = false;
        break;
      }
      sourceIndexExprs.push_back(std::move(*expr));
    }
    if (!matched)
      continue;
    if (!affineIndicesAreInBounds(sourceIndexExprs, logicalShape,
                                  unflattenedOutputShape))
      continue;

    SmallVector<int64_t> physicalStrides =
        computeRowMajorStrides(physicalShape);
    SmallVector<int64_t> viewStrides(2, 0);
    int64_t baseOffset = 0;
    for (int64_t sourceDim = 0; sourceDim < sourceRank; ++sourceDim) {
      int64_t strideScale =
          (*loadStrides)[sourceDim] * physicalStrides[sourceDim];
      baseOffset += ((*loadOffsets)[sourceDim] +
                     (*loadStrides)[sourceDim] *
                         sourceIndexExprs[sourceDim].constant) *
                    physicalStrides[sourceDim];
      for (int64_t outDim = 0; outDim < 2; ++outDim)
        viewStrides[outDim] +=
            sourceIndexExprs[sourceDim].coeffs[outDim] * strideScale;
    }
    if (baseOffset < 0 ||
        llvm::any_of(viewStrides,
                     [](int64_t stride) { return stride < 0; }))
      continue;

    SmallVector<int64_t> viewTileShape = {
        nextPow2(unflattenedOutputShape[0]),
        nextPow2(unflattenedOutputShape[1])};
    int64_t viewFlatCount = viewTileShape[0] * viewTileShape[1];
    if (viewFlatCount > 256)
      continue;

    Value basePtr = e.getArg(source.binding);
    if (source.hasStaticByteOffset) {
      basePtr =
          offsetPtrByBytes(e, basePtr, source.elemType, source.byteOffset);
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "[cuda_tile] flat affine gather source binding "
                 << source.binding
                 << " has dynamic byte offset; using base pointer\n");
    }
    if (baseOffset != 0)
      basePtr = e.offsetPtr(basePtr, baseOffset);

    auto view = e.makeTensorView(basePtr, unflattenedOutputShape,
                                 viewStrides, source.elemType);
    auto partition = e.makePartitionView(view, viewTileShape,
                                         /*zeroPad=*/true);
    auto [viewTile, token] =
        e.loadViewTko(partition, {e.constI32(0), e.constI32(0)},
                      viewTileShape, source.elemType);
    Value flatViewTile = e.reshape(viewTile, {viewFlatCount},
                                   source.elemType);

    SmallVector<Value> pieces;
    pieces.reserve(flatTileCount);
    Value zeroPiece = e.constSplat({1}, source.elemType, 0.0);
    for (int64_t lane = 0; lane < flatTileCount; ++lane) {
      if (lane >= outputElementCount) {
        pieces.push_back(zeroPiece);
        continue;
      }
      int64_t row = lane / innerDim;
      int64_t col = lane % innerDim;
      int64_t sourceLane = row * viewTileShape[1] + col;
      pieces.push_back(e.extractSubtile(flatViewTile, {1}, source.elemType,
                                        {e.constI32(sourceLane)}));
    }
    return concatenateOneElementTiles(e, std::move(pieces),
                                      source.elemType);
  }

  return {};
}

static Value emitTensorExtractGather(
    CudaTileOpEmitter &e, tensor::ExtractOp extractOp,
    DenseMap<Value, Value> &bodyMap,
    ArrayRef<int64_t> tileShape, ArrayRef<Value> outputTileIndices,
    ArrayRef<int64_t> outputShape,
    ArrayRef<CudaTileGatherSource> gatherSources) {
  const CudaTileGatherSource *source =
      findGatherSource(gatherSources, extractOp.getTensor());
  if (!source || source->binding < 0 || !source->elemType)
    return {};

  if (Value affineTile = emitTensorExtractAffineViewGather(
          e, extractOp, *source, tileShape, outputTileIndices, outputShape))
    return affineTile;
  if (Value flatAffineTile = emitTensorExtractFlat2DAffineViewGather(
          e, extractOp, *source, tileShape, outputShape))
    return flatAffineTile;

  int64_t rank = extractOp.getIndices().size();
  SmallVector<int64_t> logicalShape = source->logicalShape;
  if (logicalShape.empty())
    logicalShape = getStaticShapeFromType(extractOp.getTensor().getType());
  SmallVector<int64_t> physicalShape = source->physicalShape.empty()
                                           ? logicalShape
                                           : source->physicalShape;
  if (rank == 0 || logicalShape.size() != static_cast<size_t>(rank) ||
      physicalShape.size() != static_cast<size_t>(rank))
    return {};
  if (llvm::any_of(logicalShape, ShapedType::isDynamic) ||
      llvm::any_of(physicalShape, ShapedType::isDynamic))
    return {};

  std::optional<SmallVector<int64_t>> loadOffsets =
      normalizeStaticIndexList(source->loadOffsets, rank, 0);
  std::optional<SmallVector<int64_t>> loadStrides =
      normalizeStaticIndexList(source->loadStrides, rank, 1);
  if (!loadOffsets || !loadStrides)
    return {};

  SmallVector<Value> indexTiles;
  for (Value index : extractOp.getIndices()) {
    Value indexTile =
        resolveCudaTileIndexOperand(e, index, bodyMap, tileShape);
    if (!indexTile)
      return {};
    indexTiles.push_back(indexTile);
  }

  Type i32Type = e.builder().getI32Type();
  SmallVector<int64_t> physicalStrides = computeRowMajorStrides(physicalShape);
  int64_t baseOffset = 0;
  for (int64_t dim = 0; dim < rank; ++dim)
    baseOffset += (*loadOffsets)[dim] * physicalStrides[dim];

  Value linearOffset = e.constSplat(tileShape, i32Type, baseOffset);
  for (int64_t dim = 0; dim < rank; ++dim) {
    int64_t coeff = (*loadStrides)[dim] * physicalStrides[dim];
    if (coeff == 0)
      continue;

    Value term = indexTiles[dim];
    if (coeff != 1) {
      Value coeffTile = e.constSplat(tileShape, i32Type, coeff);
      term = e.emitElementwise("muli", {term, coeffTile});
    }
    linearOffset = e.emitElementwise("addi", {linearOffset, term});
  }

  int64_t physicalElementCount = 1;
  for (int64_t dim : physicalShape)
    physicalElementCount *= dim;
  if (physicalElementCount > 0) {
    Value minOffset = e.constSplat(tileShape, i32Type, 0.0);
    Value maxOffset =
        e.constSplat(tileShape, i32Type, physicalElementCount - 1);
    linearOffset = e.emitElementwise("maxi", {linearOffset, minOffset});
    linearOffset = e.emitElementwise("mini", {linearOffset, maxOffset});
  }

  Value basePtr = e.getArg(source->binding);
  if (source->hasStaticByteOffset) {
    basePtr =
        offsetPtrByBytes(e, basePtr, source->elemType, source->byteOffset);
  } else {
    LLVM_DEBUG(llvm::dbgs()
               << "[cuda_tile] gather source binding " << source->binding
               << " has dynamic byte offset; using base pointer\n");
  }

  auto ptrType = ::mlir::cuda_tile::PointerType::get(e.getContext(), source->elemType);
  Value ptrTile = e.broadcastScalarToTile(basePtr, tileShape, ptrType);
  Value gatherPtrs = e.offsetPtrTile(ptrTile, linearOffset);
  auto [tile, token] =
      e.loadPtrTko(gatherPtrs, tileShape, source->elemType, Value(), Value());
  return tile;
}

static Value emitElementwiseGenericBody(CudaTileOpEmitter &e,
                                        linalg::GenericOp genericOp,
                                        ArrayRef<Value> inputTiles,
                                        ArrayRef<int64_t> tileShape,
                                        Type elemType, Value fallback,
                                        ArrayRef<Value> outputTileIndices = {},
                                        ArrayRef<int64_t> outputShape = {},
                                        ArrayRef<CudaTileGatherSource>
                                            gatherSources = {}) {
  Block &body = genericOp.getRegion().front();
  DenseMap<Value, Value> bodyMap;

  int64_t numDpsInputs = genericOp.getNumDpsInputs();
  for (int64_t i = 0; i < numDpsInputs && i < (int64_t)inputTiles.size(); ++i)
    bodyMap[body.getArgument(i)] = inputTiles[i];

  Value current = fallback;
  for (Operation &op : body.without_terminator()) {
    if (auto indexOp = dyn_cast<linalg::IndexOp>(&op)) {
      int64_t dim = indexOp.getDim();
      if (dim >= 0 && dim < (int64_t)tileShape.size() &&
          dim < (int64_t)outputTileIndices.size()) {
        Value result = e.globalIndexTile(tileShape, dim,
                                         outputTileIndices[dim]);
        bodyMap[indexOp.getResult()] = result;
        current = result;
      }
      continue;
    }

    if (auto extractOp = dyn_cast<tensor::ExtractOp>(&op)) {
      if (Value result =
              emitTensorExtractGather(e, extractOp, bodyMap, tileShape,
                                      outputTileIndices, outputShape,
                                      gatherSources)) {
        bodyMap[extractOp.getResult()] = result;
        current = result;
      }
      continue;
    }

    if (auto cmpOp = dyn_cast<arith::CmpFOp>(&op)) {
      SmallVector<Value> opInputs;
      if (resolveCudaTileBodyOperands(e, &op, bodyMap, tileShape, elemType,
                                      opInputs) &&
          opInputs.size() == 2) {
        Value result = e.emitCmpF(cmpOp.getPredicate(), opInputs[0],
                                  opInputs[1]);
        bodyMap[op.getResult(0)] = result;
        current = result;
      }
      continue;
    }

    if (auto cmpOp = dyn_cast<arith::CmpIOp>(&op)) {
      SmallVector<Value> opInputs;
      if (resolveCudaTileBodyOperands(e, &op, bodyMap, tileShape, elemType,
                                      opInputs) &&
          opInputs.size() == 2) {
        Value result = e.emitCmpI(cmpOp.getPredicate(), opInputs[0],
                                  opInputs[1]);
        bodyMap[op.getResult(0)] = result;
        current = result;
      }
      continue;
    }

    if (isa<arith::IndexCastOp, arith::IndexCastUIOp>(&op)) {
      SmallVector<Value> opInputs;
      if (resolveCudaTileBodyOperands(e, &op, bodyMap, tileShape, elemType,
                                      opInputs) &&
          opInputs.size() == 1) {
        bodyMap[op.getResult(0)] = opInputs[0];
        current = opInputs[0];
      }
      continue;
    }

    if (auto sitofpOp = dyn_cast<arith::SIToFPOp>(&op)) {
      SmallVector<Value> opInputs;
      if (resolveCudaTileBodyOperands(e, &op, bodyMap, tileShape, elemType,
                                      opInputs) &&
          opInputs.size() == 1) {
        Value result = e.emitIToF(opInputs[0], tileShape, elemType);
        bodyMap[sitofpOp.getResult()] = result;
        current = result;
      }
      continue;
    }

    if (auto divOp = dyn_cast<arith::DivSIOp>(&op)) {
      SmallVector<Value> opInputs;
      if (resolveCudaTileBodyOperands(e, &op, bodyMap, tileShape, elemType,
                                      opInputs) &&
          opInputs.size() == 2) {
        Value result = e.emitDivI(opInputs[0], opInputs[1],
                                  ::mlir::cuda_tile::Signedness::Signed,
                                  ::mlir::cuda_tile::RoundingMode::ZERO);
        bodyMap[divOp.getResult()] = result;
        current = result;
      }
      continue;
    }

    if (auto divOp = dyn_cast<arith::FloorDivSIOp>(&op)) {
      SmallVector<Value> opInputs;
      if (resolveCudaTileBodyOperands(e, &op, bodyMap, tileShape, elemType,
                                      opInputs) &&
          opInputs.size() == 2) {
        Value result = e.emitDivI(opInputs[0], opInputs[1],
                                  ::mlir::cuda_tile::Signedness::Signed,
                                  ::mlir::cuda_tile::RoundingMode::NEGATIVE_INF);
        bodyMap[divOp.getResult()] = result;
        current = result;
      }
      continue;
    }

    if (auto remOp = dyn_cast<arith::RemSIOp>(&op)) {
      SmallVector<Value> opInputs;
      if (resolveCudaTileBodyOperands(e, &op, bodyMap, tileShape, elemType,
                                      opInputs) &&
          opInputs.size() == 2) {
        Value result =
            e.emitRemI(opInputs[0], opInputs[1], ::mlir::cuda_tile::Signedness::Signed);
        bodyMap[remOp.getResult()] = result;
        current = result;
      }
      continue;
    }

    StringRef name = mapArithToCudaTile(&op);
    if (name.empty())
      name = mapMathToCudaTile(&op);
    if (name.empty())
      continue;

    SmallVector<Value> opInputs;
    if (!resolveCudaTileBodyOperands(e, &op, bodyMap, tileShape, elemType,
                                     opInputs) ||
        opInputs.empty())
      continue;

    Value result = e.emitElementwise(name, opInputs);
    if (op.getNumResults() > 0)
      bodyMap[op.getResult(0)] = result;
    current = result;
  }

  if (auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator())) {
    if (!yieldOp.getOperands().empty()) {
      auto it = bodyMap.find(yieldOp.getOperand(0));
      if (it != bodyMap.end())
        current = it->second;
    }
  }
  return current;
}

//===----------------------------------------------------------------------===//
// Kernel Boilerplate Helper
//===----------------------------------------------------------------------===//

/// Build N-D partition indices from tile block IDs.
/// Last dim → bidX, second-to-last → bidY, batch dims → derived from bidZ.
static SmallVector<Value>
buildNDIndices(CudaTileOpEmitter &e, ArrayRef<int64_t> shape,
               ArrayRef<int64_t> tileShape, Value bidX, Value bidY,
               Value bidZ) {
  int64_t rank = shape.size();
  SmallVector<Value> indices(rank);
  if (rank >= 1)
    indices[rank - 1] = bidX;
  if (rank >= 2)
    indices[rank - 2] = bidY;
  if (rank == 3) {
    indices[0] = bidZ;
  } else if (rank > 3) {
    auto signAttr = ::mlir::cuda_tile::SignednessAttr::get(
        e.builder().getContext(), ::mlir::cuda_tile::Signedness::Unsigned);
    auto rndAttr = ::mlir::cuda_tile::RoundingModeAttr::get(
        e.builder().getContext(), ::mlir::cuda_tile::RoundingMode::ZERO);
    auto loc = e.builder().getUnknownLoc();
    Value remaining = bidZ;
    for (int64_t i = rank - 3; i >= 0; --i) {
      int64_t dimTiles = (shape[i] + tileShape[i] - 1) / tileShape[i];
      auto dimTilesVal = e.constI32(dimTiles);
      indices[i] =
          e.builder()
              .create<::mlir::cuda_tile::RemIOp>(loc, remaining, dimTilesVal, signAttr)
              .getResult();
      remaining =
          e.builder()
              .create<::mlir::cuda_tile::DivIOp>(loc, remaining, dimTilesVal, signAttr,
                                         rndAttr)
              .getResult();
    }
  }
  return indices;
}

struct KernelBoilerplate {
  SmallVector<Value> partViews;
  SmallVector<Value> indices;
  Value bidX, bidY, bidZ;
};

/// Emit the standard boilerplate: module, entry, tensor views, partition views,
/// block IDs, and indices for a simple element-wise-style kernel.
static KernelBoilerplate
emitKernelBoilerplate(CudaTileOpEmitter &e, StringRef kernelName,
                      int64_t numArgs, ArrayRef<int64_t> shape,
                      ArrayRef<int64_t> strides, ArrayRef<int64_t> tileShape,
                      Type elemType,
                      ArrayRef<CudaTileBindingPlan> bindings = {}) {
  KernelBoilerplate bp;

  e.beginModule(kernelName);
  e.beginEntry("main", numArgs, elemType);

  for (int64_t i = 0; i < numArgs; ++i) {
    Value ptr = bindings.empty()
                    ? e.getArg(i)
                    : getBindingArg(e, bindings, i, elemType);
    auto tv = e.makeTensorView(ptr, shape, strides, elemType);
    auto pv = e.makePartitionView(tv, tileShape);
    bp.partViews.push_back(pv);
  }

  auto [bx, by, bz] = e.getTileBlockId();
  bp.bidX = bx;
  bp.bidY = by;
  bp.bidZ = bz;

  bp.indices = buildNDIndices(e, shape, tileShape, bx, by, bz);

  return bp;
}

//===----------------------------------------------------------------------===//
// Data Movement Kernel Generators (Phase 1)
//===----------------------------------------------------------------------===//

/// Generate a contiguous copy kernel.
static std::pair<CudaTileOpEmitter, SmallVector<int64_t, 3>>
generateCopyKernel(MLIRContext *ctx, StringRef kernelName,
                   ArrayRef<int64_t> shape, Type elemType, int64_t tileM,
                   int64_t tileN) {
  CudaTileOpEmitter e(ctx);
  auto tileShape = computeTileShape(shape, tileM, tileN);
  auto strides = computeRowMajorStrides(shape);
  auto gridDims = computeGridDims(shape, tileShape);

  auto bp =
      emitKernelBoilerplate(e, kernelName, 2, shape, strides, tileShape,
                            elemType);

  auto [tile, tok] =
      e.loadViewTko(bp.partViews[0], bp.indices, tileShape, elemType);
  e.storeViewTko(tile, bp.partViews[1], bp.indices);
  e.emitReturn();
  e.endEntry();
  return {std::move(e), gridDims};
}

/// Generate a transpose kernel with arbitrary permutation.
static std::pair<CudaTileOpEmitter, SmallVector<int64_t, 3>>
generateTransposeKernel(MLIRContext *ctx, StringRef kernelName,
                        ArrayRef<int64_t> srcShape, ArrayRef<int64_t> dstShape,
                        ArrayRef<int64_t> permutation, Type elemType,
                        int64_t tileM, int64_t tileN) {
  CudaTileOpEmitter e(ctx);

  auto srcStrides = computeRowMajorStrides(srcShape);
  auto dstStrides = computeRowMajorStrides(dstShape);
  auto srcTileShape = computeTileShape(srcShape, tileM, tileN);
  auto gridDims = computeGridDims(srcShape, srcTileShape);

  // Build dst tile shape from permutation.
  SmallVector<int64_t> dstTileShape(srcTileShape.size());
  for (size_t i = 0; i < srcTileShape.size(); ++i) {
    if (i < permutation.size() &&
        permutation[i] < (int64_t)srcTileShape.size())
      dstTileShape[i] = srcTileShape[permutation[i]];
    else
      dstTileShape[i] = srcTileShape[i];
  }

  e.beginModule(kernelName);
  e.beginEntry("main", 2, elemType);

  // Source view + partition.
  auto srcView = e.makeTensorView(e.getArg(0), srcShape, srcStrides, elemType);
  auto srcPart = e.makePartitionView(srcView, srcTileShape);

  // Dest view + partition.
  auto dstView = e.makeTensorView(e.getArg(1), dstShape, dstStrides, elemType);
  auto dstPart = e.makePartitionView(dstView, dstTileShape);

  auto [bidX, bidY, bidZ] = e.getTileBlockId();

  auto srcIndices =
      buildNDIndices(e, srcShape, srcTileShape, bidX, bidY, bidZ);

  auto [tile, loadTok] =
      e.loadViewTko(srcPart, srcIndices, srcTileShape, elemType);

  // Permute the tile dimensions.
  SmallVector<int32_t> permI32(permutation.begin(), permutation.end());
  auto srcTileType = ::mlir::cuda_tile::TileType::get(ctx, srcTileShape, elemType);
  auto dstTileType = ::mlir::cuda_tile::TileType::get(ctx, dstTileShape, elemType);
  auto permuted = e.permute(tile, permI32, dstTileType);

  // Permuted indices: apply permutation to source indices.
  SmallVector<Value> dstIndices(srcIndices.size());
  for (size_t i = 0; i < permutation.size() && i < srcIndices.size(); ++i)
    dstIndices[i] = srcIndices[permutation[i]];

  e.storeViewTko(permuted, dstPart, dstIndices);
  e.emitReturn();
  e.endEntry();
  return {std::move(e), gridDims};
}

/// Generate an extract_slice kernel.
static std::pair<CudaTileOpEmitter, SmallVector<int64_t, 3>>
generateExtractSliceKernel(MLIRContext *ctx, StringRef kernelName,
                           ArrayRef<int64_t> srcShape,
                           ArrayRef<int64_t> dstShape,
                           ArrayRef<int64_t> offsets,
                           ArrayRef<int64_t> sliceStrides, Type elemType,
                           int64_t tileM, int64_t tileN) {
  CudaTileOpEmitter e(ctx);

  auto dstTileShape = computeTileShape(dstShape, tileM, tileN);
  auto gridDims = computeGridDims(dstShape, dstTileShape);
  auto srcStridesRM = computeRowMajorStrides(srcShape);
  auto dstStridesRM = computeRowMajorStrides(dstShape);

  // Compute effective strides for the slice (accounting for sliceStrides).
  SmallVector<int64_t> effectiveSrcStrides(srcStridesRM.size());
  for (size_t i = 0; i < srcStridesRM.size(); ++i)
    effectiveSrcStrides[i] = srcStridesRM[i] * sliceStrides[i];

  // Compute the linear element offset into the source buffer.
  int64_t linearOffset = 0;
  for (size_t i = 0; i < offsets.size(); ++i)
    linearOffset += offsets[i] * srcStridesRM[i];

  e.beginModule(kernelName);
  e.beginEntry("main", 2, elemType);

  Value srcPtr = e.getArg(0);

  // Offset the source pointer if needed.
  if (linearOffset > 0)
    srcPtr = e.offsetPtr(srcPtr, linearOffset);

  // Source: view over the slice region with effective strides.
  auto srcView =
      e.makeTensorView(srcPtr, dstShape, effectiveSrcStrides, elemType);
  auto srcPart = e.makePartitionView(srcView, dstTileShape);

  // Dest: contiguous view.
  auto dstView =
      e.makeTensorView(e.getArg(1), dstShape, dstStridesRM, elemType);
  auto dstPart = e.makePartitionView(dstView, dstTileShape);

  auto [bidX, bidY, bidZ] = e.getTileBlockId();
  auto indices = buildNDIndices(e, dstShape, dstTileShape, bidX, bidY, bidZ);

  auto [tile, tok] =
      e.loadViewTko(srcPart, indices, dstTileShape, elemType);
  e.storeViewTko(tile, dstPart, indices);
  e.emitReturn();
  e.endEntry();
  return {std::move(e), gridDims};
}

/// Generate an insert_slice kernel that writes the source tile into an in-place
/// destination buffer at the given offsets.
static std::pair<CudaTileOpEmitter, SmallVector<int64_t, 3>>
generateInsertSliceKernel(MLIRContext *ctx, StringRef kernelName,
                          ArrayRef<int64_t> srcShape,
                          ArrayRef<int64_t> dstShape,
                          ArrayRef<int64_t> offsets,
                          ArrayRef<int64_t> sliceStrides, Type elemType,
                          int64_t tileM, int64_t tileN) {
  CudaTileOpEmitter e(ctx);

  auto srcTileShape = computeTileShape(srcShape, tileM, tileN);
  auto gridDims = computeGridDims(srcShape, srcTileShape);
  auto srcStridesRM = computeRowMajorStrides(srcShape);
  auto dstStridesRM = computeRowMajorStrides(dstShape);

  SmallVector<int64_t> effectiveDstStrides(srcShape.size());
  for (size_t i = 0; i < srcShape.size(); ++i)
    effectiveDstStrides[i] = dstStridesRM[i] * sliceStrides[i];

  int64_t linearOffset = 0;
  for (size_t i = 0; i < offsets.size(); ++i)
    linearOffset += offsets[i] * dstStridesRM[i];

  e.beginModule(kernelName);
  e.beginEntry("main", 2, elemType);

  auto srcView =
      e.makeTensorView(e.getArg(0), srcShape, srcStridesRM, elemType);
  auto srcPart = e.makePartitionView(srcView, srcTileShape);

  Value dstPtr = e.getArg(1);
  if (linearOffset > 0)
    dstPtr = e.offsetPtr(dstPtr, linearOffset);

  auto dstView =
      e.makeTensorView(dstPtr, srcShape, effectiveDstStrides, elemType);
  auto dstPart = e.makePartitionView(dstView, srcTileShape);

  auto [bidX, bidY, bidZ] = e.getTileBlockId();
  auto indices =
      buildNDIndices(e, srcShape, srcTileShape, bidX, bidY, bidZ);

  auto [tile, tok] =
      e.loadViewTko(srcPart, indices, srcTileShape, elemType);
  e.storeViewTko(tile, dstPart, indices);
  e.emitReturn();
  e.endEntry();
  return {std::move(e), gridDims};
}

/// Generate a broadcast kernel: replicate src along broadcast_dims to fill dst.
static std::pair<CudaTileOpEmitter, SmallVector<int64_t, 3>>
generateBroadcastKernel(MLIRContext *ctx, StringRef kernelName,
                        ArrayRef<int64_t> srcShape, ArrayRef<int64_t> dstShape,
                        ArrayRef<int64_t> broadcastDims, Type elemType,
                        int64_t tileM, int64_t tileN) {
  CudaTileOpEmitter e(ctx);

  auto dstTileShape = computeTileShape(dstShape, tileM, tileN);
  auto gridDims = computeGridDims(dstShape, dstTileShape);
  auto srcStrides = computeRowMajorStrides(srcShape);
  auto dstStrides = computeRowMajorStrides(dstShape);

  llvm::SmallDenseSet<int64_t> bcastDimSet(broadcastDims.begin(),
                                           broadcastDims.end());
  SmallVector<int64_t> srcTileShape;
  int srcIdx = 0;
  for (size_t i = 0; i < dstShape.size(); ++i) {
    if (bcastDimSet.contains(i))
      continue;
    if (srcIdx < (int)srcShape.size())
      srcTileShape.push_back(dstTileShape[i]);
    srcIdx++;
  }
  if (srcTileShape.empty())
    srcTileShape.push_back(1);

  e.beginModule(kernelName);
  e.beginEntry("main", 2, elemType);

  auto srcView = e.makeTensorView(e.getArg(0), srcShape, srcStrides, elemType);
  auto srcPart = e.makePartitionView(srcView, srcTileShape);

  // Dest view: contiguous.
  auto dstView = e.makeTensorView(e.getArg(1), dstShape, dstStrides, elemType);
  auto dstPart = e.makePartitionView(dstView, dstTileShape);

  auto [bidX, bidY, bidZ] = e.getTileBlockId();
  auto indices =
      buildNDIndices(e, dstShape, dstTileShape, bidX, bidY, bidZ);

  SmallVector<Value> srcIndices;
  for (size_t i = 0; i < dstShape.size(); ++i) {
    if (!bcastDimSet.contains(i))
      srcIndices.push_back(indices[i]);
  }

  auto [tile, tok] =
      e.loadViewTko(srcPart, srcIndices, srcTileShape, elemType);
  SmallVector<int64_t> reshapeShape, broadcastShape;
  srcIdx = 0;
  for (size_t i = 0; i < dstShape.size(); ++i) {
    if (bcastDimSet.contains(i)) {
      reshapeShape.push_back(1);
    } else {
      reshapeShape.push_back(srcTileShape[srcIdx++]);
    }
    broadcastShape.push_back(dstTileShape[i]);
  }
  tile = e.reshape(tile, reshapeShape, elemType);
  if (reshapeShape != broadcastShape)
    tile = e.broadcastTile(tile, broadcastShape, elemType);
  e.storeViewTko(tile, dstPart, indices);
  e.emitReturn();
  e.endEntry();
  return {std::move(e), gridDims};
}

/// Generate a reduce kernel: reduce input along reduceDim with combiner.
/// Input shape [M, N], reduceDim=1 → output [M].
/// Grid tiles the non-reduced dimensions.
static std::pair<CudaTileOpEmitter, SmallVector<int64_t, 3>>
generateReduceKernel(MLIRContext *ctx, StringRef kernelName,
                     ArrayRef<int64_t> srcShape, ArrayRef<int64_t> dstShape,
                     ArrayRef<int64_t> reduceDims, StringRef combiner,
                     Type elemType, int64_t tileM, int64_t tileN) {
  CudaTileOpEmitter e(ctx);

  // For simplicity: single reduce dim, full tile along reduce axis.
  // The partition view handles boundary padding automatically.
  int64_t reduceDim = reduceDims.empty() ? srcShape.size() - 1 : reduceDims[0];

  // Tile shape: keep full reduce dim, tile non-reduce dims.
  // All dimensions must be power-of-2 for cuda_tile.
  SmallVector<int64_t> inputTileShape;
  for (size_t i = 0; i < srcShape.size(); ++i) {
    if ((int64_t)i == reduceDim) {
      inputTileShape.push_back(nextPow2(srcShape[i]));
    } else if (i == srcShape.size() - 1) {
      inputTileShape.push_back(nextPow2(std::min(tileN, srcShape[i])));
    } else if (i == srcShape.size() - 2) {
      inputTileShape.push_back(nextPow2(std::min(tileM, srcShape[i])));
    } else {
      inputTileShape.push_back(nextPow2(srcShape[i]));
    }
  }

  // Output tile shape: remove the reduce dim.
  SmallVector<int64_t> outputTileShape;
  for (size_t i = 0; i < inputTileShape.size(); ++i) {
    if ((int64_t)i != reduceDim)
      outputTileShape.push_back(inputTileShape[i]);
  }
  if (outputTileShape.empty())
    outputTileShape.push_back(1);

  // Grid dims: tile the non-reduced dimensions.
  auto gridDims = computeGridDims(
      dstShape.empty() ? ArrayRef<int64_t>{1} : dstShape, outputTileShape);

  LLVM_DEBUG({
    llvm::dbgs() << "[cuda_tile] reduce kernel=" << kernelName
                 << " srcShape=[";
    llvm::interleaveComma(srcShape, llvm::dbgs());
    llvm::dbgs() << "] dstShape=[";
    llvm::interleaveComma(dstShape, llvm::dbgs());
    llvm::dbgs() << "] reduceDim=" << reduceDim << " combiner=" << combiner
                 << " inputTile=[";
    llvm::interleaveComma(inputTileShape, llvm::dbgs());
    llvm::dbgs() << "] outputTile=[";
    llvm::interleaveComma(outputTileShape, llvm::dbgs());
    llvm::dbgs() << "]\n";
  });

  e.beginModule(kernelName);
  e.beginEntry("main", 2, elemType); // input + output bindings

  auto srcStrides = computeRowMajorStrides(srcShape);
  auto srcView = e.makeTensorView(e.getArg(0), srcShape, srcStrides, elemType);
  bool needReducePad = (inputTileShape[reduceDim] != srcShape[reduceDim]);
  auto srcPart = e.makePartitionView(srcView, inputTileShape, needReducePad);

  auto dstStrides = computeRowMajorStrides(
      dstShape.empty() ? ArrayRef<int64_t>{1} : dstShape);
  auto dstView = e.makeTensorView(
      e.getArg(1), dstShape.empty() ? ArrayRef<int64_t>{1} : dstShape,
      dstStrides, elemType);
  auto dstPart = e.makePartitionView(dstView, outputTileShape);

  auto [bidX, bidY, bidZ] = e.getTileBlockId();

  // Build input indices: use block IDs for non-reduced dims, 0 for reduced dim.
  SmallVector<Value> srcIndices;
  auto c0 = e.constI32(0);
  int nonReduceIdx = 0;
  for (size_t i = 0; i < srcShape.size(); ++i) {
    if ((int64_t)i == reduceDim) {
      srcIndices.push_back(c0);
    } else {
      srcIndices.push_back(nonReduceIdx == 0 ? bidX : bidY);
      nonReduceIdx++;
    }
  }

  auto [tile, tok] =
      e.loadViewTko(srcPart, srcIndices, inputTileShape, elemType);

  // Reduce along the specified dimension.
  auto reduced = e.reduce(tile, reduceDim, combiner, outputTileShape, elemType);

  // Restore insertion point to the entry block (reduce may have changed it).
  e.restoreEntryInsertionPoint();

  // Store result.
  SmallVector<Value> dstIndices;
  if (dstShape.size() >= 2)
    dstIndices = {bidY, bidX};
  else if (dstShape.size() == 1)
    dstIndices = {bidX};
  else
    dstIndices = {c0};

  e.storeViewTko(reduced, dstPart, dstIndices);
  e.emitReturn();
  e.endEntry();
  return {std::move(e), gridDims};
}

//===----------------------------------------------------------------------===//
// Direct Convolution Kernel Generator
//===----------------------------------------------------------------------===//
//
// For conv2d with non-1x1 filters where im2col fusion is not feasible.
// Loops over filter positions (kh, kw), loading shifted input patches and
// accumulating via mmaf.
//
// Input:  [N, H, W, C_in]
// Filter: [KH, KW, C_in, C_out]
// Output: [N, OH, OW, C_out]
//
// For each (kh, kw):
//   input_ptr = base_input + (kh * dilation_h * W * C_in +
//                             kw * dilation_w * C_in)
//   input_view = [N, OH, OW, C_in] with strides derived from the convolution
//   strides, then flatten each output spatial tile to [M, K]
//   filter_view = filter[kh*KW*C_in*C_out + kw*C_in*C_out] → [K, N]
//   acc += mmaf(input_slice, filter_slice, acc)
//
static std::pair<CudaTileOpEmitter, SmallVector<int64_t, 3>>
generateConv2DKernel(MLIRContext *ctx, StringRef kernelName,
                     ArrayRef<int64_t> inputShape,   // [N, H, W, C_in]
                     ArrayRef<int64_t> filterShape,  // [KH, KW, C_in, C_out]
                     ArrayRef<int64_t> outputShape,  // [N, OH, OW, C_out]
                     ArrayRef<int64_t> strides, ArrayRef<int64_t> dilations,
                     Type elemType, int64_t tileM, int64_t tileN,
                     int64_t tileK, bool isNCHW = false) {
  int64_t N = inputShape[0];
  int64_t H = inputShape[1];
  int64_t W = inputShape[2];
  int64_t Cin = inputShape[3];
  int64_t KH = filterShape[0];
  int64_t KW = filterShape[1];
  int64_t Cout = filterShape[3];
  int64_t OH = outputShape[1];
  int64_t OW = outputShape[2];
  int64_t strideH = strides[0];
  int64_t strideW = strides[1];
  int64_t dilationH = dilations[0];
  int64_t dilationW = dilations[1];

  int64_t tOH = nextPow2(std::min(tileM, OH));
  int64_t tOW = 1;
  int64_t tOC = nextPow2(std::min(tileN, Cout));
  int64_t tK = nextPow2(std::min(tileK, Cin));
  int64_t flatM = tOH * tOW;
  int64_t nK = (Cin + tK - 1) / tK;

  CudaTileOpEmitter e(ctx);
  e.beginModule(kernelName);
  e.beginEntry("main", 3, elemType); // input, filter, output

  // Output view: logical [N, OH, OW, C_out], strides depend on physical layout.
  SmallVector<int64_t> shC4 = {N, OH, OW, Cout};
  SmallVector<int64_t> stC4 =
      isNCHW ? SmallVector<int64_t>{Cout * OH * OW, OW, 1, OH * OW}
             : SmallVector<int64_t>{OH * OW * Cout, OW * Cout, Cout, 1};
  SmallVector<int64_t> tileC4 = {1, tOH, tOW, tOC};
  auto vC = e.makeTensorView(e.getArg(2), shC4, stC4, elemType);
  auto pC = e.makePartitionView(vC, tileC4);

  auto [bx, by, bz] = e.getTileBlockId();
  auto c0 = e.constI32(0);
  auto accInit = e.constSplat({flatM, tOC}, elemType, 0.0);
  int64_t ohTiles = (OH + tOH - 1) / tOH;

  auto signAttr = ::mlir::cuda_tile::SignednessAttr::get(
      ctx, ::mlir::cuda_tile::Signedness::Unsigned);
  auto rndAttr = ::mlir::cuda_tile::RoundingModeAttr::get(
      ctx, ::mlir::cuda_tile::RoundingMode::ZERO);
  auto batchDivisor = e.constI32(ohTiles);
  auto batchId =
      e.builder().create<::mlir::cuda_tile::DivIOp>(e.getLoc(), bz, batchDivisor,
                                            signAttr, rndAttr)
          .getResult();
  auto ohBlockId =
      e.builder().create<::mlir::cuda_tile::RemIOp>(e.getLoc(), bz, batchDivisor,
                                            signAttr)
          .getResult();

  Value acc = accInit;
  for (int64_t kh = 0; kh < KH; ++kh) {
    for (int64_t kw = 0; kw < KW; ++kw) {
      // Pointer offset for kernel position (kh, kw).
      // NHWC: offset = ((kh*dilH)*W + kw*dilW) * Cin
      // NCHW: offset = (kh*dilH)*W + kw*dilW
      int64_t inputOffset =
          isNCHW ? (kh * dilationH) * W + (kw * dilationW)
                 : ((kh * dilationH) * W + (kw * dilationW)) * Cin;
      Value inputPtr = e.offsetPtr(e.getArg(0), inputOffset);

      // Input view: logical [N, OH, OW, C_in], strides from physical layout.
      SmallVector<int64_t> shA4 = {N, OH, OW, Cin};
      SmallVector<int64_t> stA4 =
          isNCHW
              ? SmallVector<int64_t>{Cin * H * W, strideH * W, strideW, H * W}
              : SmallVector<int64_t>{H * W * Cin, strideH * W * Cin,
                                     strideW * Cin, 1};
      auto vA = e.makeTensorView(inputPtr, shA4, stA4, elemType);
      SmallVector<int64_t> tileA4 = {1, tOH, tOW, tK};
      bool needKPad = (Cin % tK) != 0;
      auto pA = e.makePartitionView(vA, tileA4, needKPad);

      // Filter slice at (kh, kw): logical [C_in, C_out].
      // HWIO: offset = (kh*KW+kw)*Cin*Cout, strides = [Cout, 1]
      // FCHW: offset = kh*KW+kw, strides = [KH*KW, Cin*KH*KW]
      int64_t filterOffset =
          isNCHW ? kh * KW + kw : (kh * KW + kw) * Cin * Cout;
      Value filterPtr = e.offsetPtr(e.getArg(1), filterOffset);
      SmallVector<int64_t> shB = {Cin, Cout};
      SmallVector<int64_t> stB =
          isNCHW ? SmallVector<int64_t>{KH * KW, Cin * KH * KW}
                 : SmallVector<int64_t>{Cout, 1};
      auto vB = e.makeTensorView(filterPtr, shB, stB, elemType);
      SmallVector<int64_t> tB = {tK, tOC};
      auto pB = e.makePartitionView(vB, tB, needKPad);

      // Inner loop over K (C_in) reduction.
      // For small Cin, this may be just 1 iteration.

      if (nK == 1) {
        auto [tAd, tokA] =
            e.loadViewTko(pA, {batchId, ohBlockId, by, c0}, tileA4, elemType);
        SmallVector<int64_t> reshapeA = {flatM, tK};
        Value flatA = e.reshape(tAd, reshapeA, elemType);

        auto [tBd, tokB] =
            e.loadViewTko(pB, {c0, bx}, tB, elemType);

        acc = e.mmaf(flatA, tBd, acc);
      } else {
        // Multiple K iterations.
        auto lb = e.constI32(0);
        auto ub = e.constI32(nK);
        auto step = e.constI32(1);
        auto forOp = e.beginFor(lb, ub, step, ValueRange{acc});
        Value iv = forOp.getInductionVar();
        Value iterAcc = forOp.getRegionIterValues()[0];

        auto [tAd, tokA] =
            e.loadViewTko(pA, {batchId, ohBlockId, by, iv}, tileA4, elemType);
        SmallVector<int64_t> reshapeA = {flatM, tK};
        Value flatA = e.reshape(tAd, reshapeA, elemType);

        auto [tBd, tokB] =
            e.loadViewTko(pB, {iv, bx}, tB, elemType);

        Value newAcc = e.mmaf(flatA, tBd, iterAcc);
        e.endFor(forOp, ValueRange{newAcc});
        acc = forOp.getResult(0);
      }
    }
  }

  Value acc4D = e.reshape(acc, tileC4, elemType);
  e.storeViewTko(acc4D, pC, {batchId, ohBlockId, by, bx});
  e.emitReturn();
  e.endEntry();

  SmallVector<int64_t, 3> gridDims = {(Cout + tOC - 1) / tOC,
                                      (OW + tOW - 1) / tOW,
                                      N * ohTiles};
  return {std::move(e), gridDims};
}

//===----------------------------------------------------------------------===//
// Pooling Kernel Generator
//===----------------------------------------------------------------------===//
//
// For windowed reduction operations (max_pool, avg_pool, etc.).
// Loops over window positions (kh, kw), loading shifted input patches and
// accumulating via elementwise combiner (maxf, addf, minf).
//
// Input:  [N, H, W, C]   (NHWC-normalized)
// Output: [N, OH, OW, C]
// Window: [KH, KW]
//
static std::pair<CudaTileOpEmitter, SmallVector<int64_t, 3>>
generatePoolingKernel(MLIRContext *ctx, StringRef kernelName,
                      ArrayRef<int64_t> inputShape,
                      ArrayRef<int64_t> outputShape,
                      ArrayRef<int64_t> windowShape,
                      ArrayRef<int64_t> strides, ArrayRef<int64_t> dilations,
                      StringRef combiner, Type elemType, int64_t tileM,
                      int64_t tileN, bool isNCHW) {
  int64_t N = inputShape[0];
  int64_t H = inputShape[1];
  int64_t W = inputShape[2];
  int64_t C = inputShape[3];
  int64_t OH = outputShape[1];
  int64_t OW = outputShape[2];
  int64_t KH = windowShape[0];
  int64_t KW = windowShape.size() > 1 ? windowShape[1] : 1;
  int64_t strideH = strides[0];
  int64_t strideW = strides.size() > 1 ? strides[1] : 1;
  int64_t dilH = dilations[0];
  int64_t dilW = dilations.size() > 1 ? dilations[1] : 1;

  int64_t tOH = nextPow2(std::min(tileM, OH));
  int64_t tOW = 1;
  int64_t tOC = nextPow2(std::min(tileN, C));

  CudaTileOpEmitter e(ctx);
  e.beginModule(kernelName);
  e.beginEntry("main", 2, elemType);

  SmallVector<int64_t> shOut = {N, OH, OW, C};
  SmallVector<int64_t> stOut =
      isNCHW ? SmallVector<int64_t>{C * OH * OW, OW, 1, OH * OW}
             : SmallVector<int64_t>{OH * OW * C, OW * C, C, 1};
  SmallVector<int64_t> tileOut = {1, tOH, tOW, tOC};
  auto vOut = e.makeTensorView(e.getArg(1), shOut, stOut, elemType);
  auto pOut = e.makePartitionView(vOut, tileOut);

  auto [bx, by, bz] = e.getTileBlockId();
  int64_t ohTiles = (OH + tOH - 1) / tOH;

  auto signAttr = ::mlir::cuda_tile::SignednessAttr::get(
      ctx, ::mlir::cuda_tile::Signedness::Unsigned);
  auto rndAttr = ::mlir::cuda_tile::RoundingModeAttr::get(
      ctx, ::mlir::cuda_tile::RoundingMode::ZERO);
  auto batchDivisor = e.constI32(ohTiles);
  auto batchId =
      e.builder()
          .create<::mlir::cuda_tile::DivIOp>(e.getLoc(), bz, batchDivisor, signAttr,
                                     rndAttr)
          .getResult();
  auto ohBlockId =
      e.builder()
          .create<::mlir::cuda_tile::RemIOp>(e.getLoc(), bz, batchDivisor, signAttr)
          .getResult();

  double initVal = 0.0;
  if (combiner == "maxf")
    initVal = -std::numeric_limits<float>::infinity();
  else if (combiner == "minf")
    initVal = std::numeric_limits<float>::infinity();

  SmallVector<int64_t> accShape = {tOH * tOW, tOC};
  Value acc = e.constSplat(accShape, elemType, initVal);

  for (int64_t kh = 0; kh < KH; ++kh) {
    for (int64_t kw = 0; kw < KW; ++kw) {
      int64_t inputOffset =
          isNCHW ? (kh * dilH) * W + (kw * dilW)
                 : ((kh * dilH) * W + (kw * dilW)) * C;
      Value inputPtr = e.offsetPtr(e.getArg(0), inputOffset);

      SmallVector<int64_t> shIn = {N, OH, OW, C};
      SmallVector<int64_t> stIn =
          isNCHW
              ? SmallVector<int64_t>{C * H * W, strideH * W, strideW, H * W}
              : SmallVector<int64_t>{H * W * C, strideH * W * C, strideW * C,
                                     1};
      auto vIn = e.makeTensorView(inputPtr, shIn, stIn, elemType);
      SmallVector<int64_t> tileIn = {1, tOH, tOW, tOC};
      auto pIn = e.makePartitionView(vIn, tileIn);

      auto [tile, tok] =
          e.loadViewTko(pIn, {batchId, ohBlockId, by, bx}, tileIn, elemType);
      Value flatInput = e.reshape(tile, accShape, elemType);

      acc = e.emitElementwise(combiner, {acc, flatInput});
    }
  }

  Value result4D = e.reshape(acc, tileOut, elemType);
  e.storeViewTko(result4D, pOut, {batchId, ohBlockId, by, bx});
  e.emitReturn();
  e.endEntry();

  SmallVector<int64_t, 3> gridDims = {(C + tOC - 1) / tOC,
                                      (OW + tOW - 1) / tOW, N * ohTiles};
  return {std::move(e), gridDims};
}

static std::pair<CudaTileOpEmitter, SmallVector<int64_t, 3>>
generateStridedPointwiseMatmul2DKernel(
    MLIRContext *ctx, StringRef kernelName, int64_t numBindings,
    int64_t bindA, int64_t bindB, int64_t bindC,
    ArrayRef<int64_t> sourceShape, ArrayRef<int64_t> sliceOffsets,
    ArrayRef<int64_t> sliceStrides, ArrayRef<int64_t> aShape,
    ArrayRef<int64_t> bShape, ArrayRef<int64_t> cShape, Type elemType,
    int64_t tileM, int64_t tileN, int64_t tileK) {
  bool hasBatch = aShape.size() == 4;
  int64_t batch = hasBatch ? aShape[0] : 1;
  int64_t OH = aShape[hasBatch ? 1 : 0];
  int64_t OW = aShape[hasBatch ? 2 : 1];
  int64_t K = aShape[hasBatch ? 3 : 2];
  int64_t N = cShape[hasBatch ? 3 : 2];

  auto sourceBaseStrides = computeRowMajorStrides(sourceShape);
  int64_t baseOffset = 0;
  SmallVector<int64_t> logicalStrides;
  logicalStrides.reserve(aShape.size());
  for (int64_t i = 0; i < static_cast<int64_t>(aShape.size()); ++i) {
    int64_t offset = i < static_cast<int64_t>(sliceOffsets.size())
                         ? sliceOffsets[i]
                         : 0;
    int64_t stride = i < static_cast<int64_t>(sliceStrides.size())
                         ? sliceStrides[i]
                         : 1;
    baseOffset += offset * sourceBaseStrides[i];
    logicalStrides.push_back(sourceBaseStrides[i] * stride);
  }

  int64_t tOH = nextPow2(std::min(tileM, OH));
  int64_t tOW = 1;
  int64_t tN = nextPow2(std::min(tileN, N));
  int64_t tK = nextPow2(std::min(tileK, K));
  int64_t flatM = tOH * tOW;
  int64_t nK = (K + tK - 1) / tK;

  CudaTileOpEmitter e(ctx);
  e.beginModule(kernelName);
  e.beginEntry("main", numBindings, elemType);

  SmallVector<int64_t> shC;
  SmallVector<int64_t> stC;
  SmallVector<int64_t> tileC;
  if (hasBatch) {
    shC = {batch, OH, OW, N};
    stC = {OH * OW * N, OW * N, N, 1};
    tileC = {1, tOH, tOW, tN};
  } else {
    shC = {OH, OW, N};
    stC = {OW * N, N, 1};
    tileC = {tOH, tOW, tN};
  }
  auto vC = e.makeTensorView(e.getArg(bindC), shC, stC, elemType);
  auto pC = e.makePartitionView(vC, tileC);

  Value inputPtr = e.offsetPtr(e.getArg(bindA), baseOffset);
  auto vA = e.makeTensorView(inputPtr, aShape, logicalStrides, elemType);
  SmallVector<int64_t> tileA =
      hasBatch ? SmallVector<int64_t>{1, tOH, tOW, tK}
               : SmallVector<int64_t>{tOH, tOW, tK};
  bool needKPad = (K % tK) != 0;
  auto pA = e.makePartitionView(vA, tileA, needKPad);

  SmallVector<int64_t> shB = {bShape[0], bShape[1]};
  SmallVector<int64_t> stB = {bShape[1], 1};
  SmallVector<int64_t> tileB = {tK, tN};
  auto vB = e.makeTensorView(e.getArg(bindB), shB, stB, elemType);
  auto pB = e.makePartitionView(vB, tileB, needKPad);

  auto [bx, by, bz] = e.getTileBlockId();
  Value batchId = bz;
  Value ohBlockId = by;
  if (hasBatch) {
    auto signAttr = ::mlir::cuda_tile::SignednessAttr::get(
        ctx, ::mlir::cuda_tile::Signedness::Unsigned);
    auto rndAttr = ::mlir::cuda_tile::RoundingModeAttr::get(
        ctx, ::mlir::cuda_tile::RoundingMode::ZERO);
    Value ohTiles = e.constI32((OH + tOH - 1) / tOH);
    batchId =
        e.builder()
            .create<::mlir::cuda_tile::DivIOp>(e.getLoc(), bz, ohTiles, signAttr,
                                       rndAttr)
            .getResult();
    ohBlockId =
        e.builder()
            .create<::mlir::cuda_tile::RemIOp>(e.getLoc(), bz, ohTiles, signAttr)
            .getResult();
  }
  auto accInit = e.constSplat({flatM, tN}, elemType, 0.0);
  auto lb = e.constI32(0);
  auto ub = e.constI32(nK);
  auto step = e.constI32(1);
  auto forOp = e.beginFor(lb, ub, step, ValueRange{accInit});
  Value iv = forOp.getInductionVar();
  Value iterAcc = forOp.getRegionIterValues()[0];

  SmallVector<Value> aCoords =
      hasBatch ? SmallVector<Value>{batchId, ohBlockId, by, iv}
               : SmallVector<Value>{bz, by, iv};
  auto [tAd, tokA] = e.loadViewTko(pA, aCoords, tileA, elemType);
  Value flatA = e.reshape(tAd, {flatM, tK}, elemType);
  auto [tBd, tokB] = e.loadViewTko(pB, {iv, bx}, tileB, elemType);
  Value newAcc = e.mmaf(flatA, tBd, iterAcc);
  e.endFor(forOp, ValueRange{newAcc});

  Value finalAcc = forOp.getResult(0);
  Value accTile = e.reshape(finalAcc, tileC, elemType);
  SmallVector<Value> cCoords =
      hasBatch ? SmallVector<Value>{batchId, ohBlockId, by, bx}
               : SmallVector<Value>{bz, by, bx};
  e.storeViewTko(accTile, pC, cCoords);
  e.emitReturn();
  e.endEntry();

  SmallVector<int64_t, 3> gridDims = {(N + tN - 1) / tN,
                                      (OW + tOW - 1) / tOW,
                                      (OH + tOH - 1) / tOH * batch};
  return {std::move(e), gridDims};
}

using CudaTileKernelBinary =
    std::pair<std::string, SmallVector<int64_t, 3>>;

static FailureOr<CudaTileKernelBinary>
serializeCudaTileKernel(CudaTileOpEmitter &&e,
                        SmallVector<int64_t, 3> gridDims) {
  if (std::getenv("IREE_CUDA_TILE_DUMP_TILE_IR")) {
    e.getModule().print(llvm::errs());
    llvm::errs() << "\n";
  }
  std::string tilebcData;
  if (failed(e.serialize(tilebcData)))
    return failure();
  return std::make_pair(std::move(tilebcData), std::move(gridDims));
}

static bool isDataMovementStrategy(CudaTileLoweringStrategy strategy) {
  switch (strategy) {
  case CudaTileLoweringStrategy::Copy:
  case CudaTileLoweringStrategy::ExtractSlice:
  case CudaTileLoweringStrategy::InsertSlice:
  case CudaTileLoweringStrategy::Transpose:
  case CudaTileLoweringStrategy::Broadcast:
  case CudaTileLoweringStrategy::ReshapeCopy:
    return true;
  default:
    return false;
  }
}

static FailureOr<CudaTileKernelBinary>
emitPureDataMovementDispatch(MLIRContext *ctx, StringRef kernelName,
                             const CudaTileOptions &options,
                             CudaTileKernelPlan &plan) {
  if (plan.singleLoadOp && plan.singleStoreOp && !plan.sawMultipleLoads &&
      !plan.sawMultipleStores) {
    auto loadSourceShape =
        getStaticShapeFromType(plan.singleLoadOp.getSource().getType());
    auto loadResultShape =
        SmallVector<int64_t>(plan.singleLoadOp.getType().getShape().begin(),
                             plan.singleLoadOp.getType().getShape().end());
    auto storeTargetShape =
        getStaticShapeFromType(plan.singleStoreOp.getTarget().getType());
    auto elemType = plan.singleLoadOp.getType().getElementType();
    auto loadOffsets =
        SmallVector<int64_t>(plan.singleLoadOp.getStaticOffsets());
    auto loadStrides =
        SmallVector<int64_t>(plan.singleLoadOp.getStaticStrides());
    auto storeOffsets =
        SmallVector<int64_t>(plan.singleStoreOp.getStaticOffsets());
    auto storeStrides =
        SmallVector<int64_t>(plan.singleStoreOp.getStaticStrides());

    bool loadIsSlice = !plan.singleLoadOp.isLoadOfWholeSource() ||
                       !isIdentitySlice(loadOffsets, loadStrides);
    bool storeIsSlice = !plan.singleStoreOp.isStoreToWholeTarget() ||
                        !isIdentitySlice(storeOffsets, storeStrides);

    int64_t tileM = options.tileM, tileN = options.tileN;
    if (loadIsSlice && !storeIsSlice) {
      auto [e, grid] =
          generateExtractSliceKernel(ctx, kernelName, loadSourceShape,
                                     loadResultShape, loadOffsets, loadStrides,
                                     elemType, tileM, tileN);
      return serializeCudaTileKernel(std::move(e), std::move(grid));
    }
    if (!loadIsSlice && storeIsSlice) {
      auto [e, grid] =
          generateInsertSliceKernel(ctx, kernelName, loadResultShape,
                                    storeTargetShape, storeOffsets,
                                    storeStrides, elemType, tileM, tileN);
      return serializeCudaTileKernel(std::move(e), std::move(grid));
    }
  }

  SmallVector<int64_t> copyShape = plan.copyFallbackShape;
  Type copyElemType = plan.copyFallbackElementType;
  if (copyShape.empty())
    return failure();
  auto [e, grid] = generateCopyKernel(ctx, kernelName, copyShape, copyElemType,
                                      options.tileM, options.tileN);
  return serializeCudaTileKernel(std::move(e), std::move(grid));
}

static FailureOr<CudaTileKernelBinary>
emitDataMovementKernel(MLIRContext *ctx, StringRef kernelName,
                       Operation *primaryOp,
                       CudaTileLoweringStrategy loweringStrategy,
                       ArrayRef<int64_t> srcShape, ArrayRef<int64_t> dstShape,
                       Type elemType, int64_t tileM, int64_t tileN) {
  switch (loweringStrategy) {
  case CudaTileLoweringStrategy::Copy: {
    auto [e, grid] =
        generateCopyKernel(ctx, kernelName, srcShape, elemType, tileM, tileN);
    return serializeCudaTileKernel(std::move(e), std::move(grid));
  }
  case CudaTileLoweringStrategy::Transpose: {
    auto perm = getI64ArrayAttr(primaryOp, "cuda_tile.permutation");
    if (perm.empty()) {
      for (int64_t i = srcShape.size() - 1; i >= 0; --i)
        perm.push_back(i);
    }
    auto [e, grid] = generateTransposeKernel(ctx, kernelName, srcShape,
                                             dstShape, perm, elemType, tileM,
                                             tileN);
    return serializeCudaTileKernel(std::move(e), std::move(grid));
  }
  case CudaTileLoweringStrategy::ExtractSlice:
  case CudaTileLoweringStrategy::InsertSlice: {
    auto offsets = getI64ArrayAttr(primaryOp, "cuda_tile.offsets");
    auto sliceStrides = getI64ArrayAttr(primaryOp, "cuda_tile.slice_strides");
    if (offsets.empty())
      offsets.resize(srcShape.size(), 0);
    if (sliceStrides.empty())
      sliceStrides.resize(srcShape.size(), 1);

    if (loweringStrategy == CudaTileLoweringStrategy::ExtractSlice) {
      auto [e, grid] =
          generateExtractSliceKernel(ctx, kernelName, srcShape, dstShape,
                                     offsets, sliceStrides, elemType, tileM,
                                     tileN);
      return serializeCudaTileKernel(std::move(e), std::move(grid));
    }
    auto [e, grid] =
        generateInsertSliceKernel(ctx, kernelName, srcShape, dstShape, offsets,
                                  sliceStrides, elemType, tileM, tileN);
    return serializeCudaTileKernel(std::move(e), std::move(grid));
  }
  case CudaTileLoweringStrategy::Broadcast: {
    auto bcastDims = getI64ArrayAttr(primaryOp, "cuda_tile.broadcast_dims");
    auto [e, grid] = generateBroadcastKernel(ctx, kernelName, srcShape,
                                             dstShape, bcastDims, elemType,
                                             tileM, tileN);
    return serializeCudaTileKernel(std::move(e), std::move(grid));
  }
  case CudaTileLoweringStrategy::ReshapeCopy: {
    auto [e, grid] =
        generateCopyKernel(ctx, kernelName, dstShape, elemType, tileM, tileN);
    return serializeCudaTileKernel(std::move(e), std::move(grid));
  }
  default:
    return failure();
  }
}

static FailureOr<CudaTileKernelBinary>
emitReductionKernel(MLIRContext *ctx, StringRef kernelName,
                    const CudaTileKernelPlan &plan, Operation *primaryOp,
                    ArrayRef<int64_t> srcShapeIn, ArrayRef<int64_t> dstShapeIn,
                    Type elemType, int64_t tileM, int64_t tileN) {
  LLVM_DEBUG(llvm::dbgs() << "[cuda_tile]   entering reduce path\n");

  SmallVector<int64_t> srcShape(srcShapeIn);
  SmallVector<int64_t> dstShape(dstShapeIn);
  if (auto genericOp = dyn_cast<linalg::GenericOp>(primaryOp)) {
    auto inType = dyn_cast<RankedTensorType>(
        genericOp.getDpsInputs()[0].getType());
    auto outType = dyn_cast<RankedTensorType>(
        genericOp.getDpsInits()[0].getType());
    if (inType)
      srcShape.assign(inType.getShape().begin(), inType.getShape().end());
    if (outType)
      dstShape.assign(outType.getShape().begin(), outType.getShape().end());
  }

  int64_t srcElems = 1, dstElems = 1;
  for (auto d : srcShape)
    srcElems *= d;
  for (auto d : dstShape)
    dstElems *= d;
  if (srcElems <= dstElems && srcShape.size() <= dstShape.size())
    return failure();

  auto combinerAttr =
      primaryOp->getAttrOfType<StringAttr>("cuda_tile.combiner");
  StringRef combiner = combinerAttr ? combinerAttr.getValue() : "addf";

  // Extract the tensor reduce dim from the actual IR. Iterator reduction dims
  // are not always tensor dims because indexing maps can transpose them.
  SmallVector<int64_t> reduceDims = plan.reductionDims;
  if (reduceDims.empty()) {
    if (auto genericOp = dyn_cast<linalg::GenericOp>(primaryOp)) {
      auto iterTypes = genericOp.getIteratorTypesArray();
      auto maps = genericOp.getIndexingMapsArray();
      AffineMap inputMap = maps.empty() ? AffineMap() : maps[0];

      for (unsigned iterDim = 0; iterDim < iterTypes.size(); ++iterDim) {
        if (iterTypes[iterDim] == mlir::utils::IteratorType::parallel)
          continue;
        if (inputMap) {
          for (unsigned tensorDim = 0; tensorDim < inputMap.getNumResults();
               ++tensorDim) {
            auto expr = inputMap.getResult(tensorDim);
            if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
              if (dimExpr.getPosition() == iterDim) {
                reduceDims.push_back(tensorDim);
                break;
              }
            }
          }
        } else {
          reduceDims.push_back(iterDim);
        }
      }
    } else if (auto reduceOp = dyn_cast<linalg::ReduceOp>(primaryOp)) {
      reduceDims.assign(reduceOp.getDimensions().begin(),
                        reduceOp.getDimensions().end());
    }
  }
  if (reduceDims.empty())
    reduceDims = getI64ArrayAttr(primaryOp, "cuda_tile.reduce_dims");

  // Bail out for windowed reductions where reduce dims could not be mapped.
  if (reduceDims.empty()) {
    if (auto genericOp = dyn_cast<linalg::GenericOp>(primaryOp)) {
      if (genericOp.getNumReductionLoops() > 0)
        return failure();
    }
  }

  auto [e, grid] = generateReduceKernel(ctx, kernelName, srcShape, dstShape,
                                        reduceDims, combiner, elemType, tileM,
                                        tileN);
  return serializeCudaTileKernel(std::move(e), std::move(grid));
}

static FailureOr<CudaTileKernelBinary>
emitDirectConv2DKernel(MLIRContext *ctx, StringRef kernelName,
                       const CudaTileConvPlan &convPlan, Type elemType,
                       int64_t tileM, int64_t tileN, int64_t tileK) {
  auto [e, grid] = generateConv2DKernel(
      ctx, kernelName, convPlan.inputShape, convPlan.filterShape,
      convPlan.outputShape, convPlan.strides, convPlan.dilations, elemType,
      tileM, tileN, tileK, convPlan.isNCHW);
  return serializeCudaTileKernel(std::move(e), std::move(grid));
}

static FailureOr<CudaTileKernelBinary>
emitPoolingKernel(MLIRContext *ctx, StringRef kernelName,
                  const CudaTileConvPlan &convPlan, Type elemType,
                  int64_t tileM, int64_t tileN) {
  auto [e, grid] = generatePoolingKernel(
      ctx, kernelName, convPlan.inputShape, convPlan.outputShape,
      convPlan.windowShape, convPlan.strides, convPlan.dilations,
      convPlan.combiner, elemType, tileM, tileN, convPlan.isNCHW);
  return serializeCudaTileKernel(std::move(e), std::move(grid));
}

struct IdentityAdaptiveAvgPoolPlan {
  int64_t inputBinding = -1;
  int64_t sumBinding = -1;
  int64_t countBinding = -1;
  int64_t numBindings = 0;
  int64_t inputByteOffset = 0;
  int64_t sumByteOffset = 0;
  int64_t countByteOffset = 0;

  int64_t channels = 0;
  int64_t inputH = 0;
  int64_t inputW = 0;
  int64_t outputH = 0;
  int64_t outputW = 0;

  Type elemType;
};

static bool isAllowedIdentityPoolIndexConstant(int64_t value, int64_t outputH,
                                               int64_t outputW) {
  return value == 0 || value == 1 || value == 2 || value == outputH ||
         value == outputW;
}

static bool bodyLooksLikeIdentityAdaptiveAvgPool(linalg::GenericOp genericOp,
                                                 int64_t outputH,
                                                 int64_t outputW) {
  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 2)
    return false;
  if (genericOp.getNumParallelLoops() != 2 ||
      genericOp.getNumReductionLoops() != 1)
    return false;

  bool sawIndex = false;
  bool sawExtract = false;
  bool sawCountCast = false;
  Block &body = genericOp.getRegion().front();
  for (Operation &op : body.without_terminator()) {
    sawIndex |= isa<linalg::IndexOp>(&op);
    sawExtract |= isa<tensor::ExtractOp>(&op);
    sawCountCast |= isa<arith::SIToFPOp>(&op);

    for (Value operand : op.getOperands()) {
      auto cst = operand.getDefiningOp<arith::ConstantOp>();
      if (!cst || !operand.getType().isIndex())
        continue;
      auto intAttr = dyn_cast<IntegerAttr>(cst.getValue());
      if (!intAttr)
        continue;
      int64_t value = intAttr.getInt();
      if (!isAllowedIdentityPoolIndexConstant(value, outputH, outputW))
        return false;
    }
  }

  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  return sawIndex && sawExtract && sawCountCast && yieldOp &&
         yieldOp.getNumOperands() == 2;
}

static FailureOr<IdentityAdaptiveAvgPoolPlan>
matchIdentityAdaptiveAvgPoolDispatch(const CudaTileKernelPlan &plan) {
  auto debugFailure = [](StringRef reason)
      -> FailureOr<IdentityAdaptiveAvgPoolPlan> {
    LLVM_DEBUG(llvm::dbgs()
               << "[cuda_tile]   identity adaptive avg pool miss: " << reason
               << "\n");
    return failure();
  };

  auto genericOp = dyn_cast_or_null<linalg::GenericOp>(plan.primaryOp);
  if (!genericOp)
    return debugFailure("primary op is not linalg.generic");

  SmallVector<const CudaTileOperandPlan *> loads;
  SmallVector<const CudaTileOperandPlan *> stores;
  for (const CudaTileOperandPlan &operand : plan.operands) {
    if (operand.isDispatchLoad)
      loads.push_back(&operand);
    if (operand.isDispatchStore)
      stores.push_back(&operand);
  }
  if (loads.size() != 1 || stores.size() != 2)
    return debugFailure("expected one dispatch load and two dispatch stores");

  const CudaTileOperandPlan *input = loads.front();
  const CudaTileOperandPlan *sum = nullptr;
  const CudaTileOperandPlan *count = nullptr;
  for (const CudaTileOperandPlan *store : stores) {
    if (store->logicalShape.size() == 2)
      sum = store;
    else if (store->logicalShape.size() == 1)
      count = store;
  }
  if (!sum || !count)
    return debugFailure("could not identify sum/count outputs");

  ArrayRef<int64_t> inputShape = input->physicalShape;
  ArrayRef<int64_t> sumShape = sum->logicalShape;
  ArrayRef<int64_t> countShape = count->logicalShape;
  if (inputShape.size() != 3 || sumShape.size() != 2 ||
      countShape.size() != 1)
    return debugFailure("unexpected input/output ranks");
  if (inputShape[0] != sumShape[0] || sumShape[1] != countShape[0])
    return debugFailure("sum/count shapes are not compatible");

  int64_t channels = inputShape[0];
  int64_t spatial = sumShape[1];
  int64_t inputH = inputShape[1];
  int64_t inputW = inputShape[2];
  int64_t outputH = -1;
  int64_t outputW = -1;
  if (inputH > 1 && inputW > 1 && (inputH - 1) * (inputW - 1) == spatial) {
    outputH = inputH - 1;
    outputW = inputW - 1;
  } else if (inputH * inputW == spatial) {
    outputH = inputH;
    outputW = inputW;
  } else {
    return debugFailure("output spatial size does not match identity pool");
  }

  if (!bodyLooksLikeIdentityAdaptiveAvgPool(genericOp, outputH, outputW))
    return debugFailure("body does not match identity pool shape/count pattern");

  Type elemType = getElementTypeFromType(input->value.getType());
  if (!elemType)
    elemType = getElementTypeFromType(sum->value.getType());
  if (!elemType || !isa<FloatType>(elemType))
    return debugFailure("could not infer floating-point element type");

  IdentityAdaptiveAvgPoolPlan poolPlan;
  poolPlan.inputBinding = input->binding;
  poolPlan.sumBinding = sum->binding;
  poolPlan.countBinding = count->binding;
  poolPlan.numBindings = getNumBindingArgs(plan.bindingShapes);
  if (const CudaTileBindingPlan *binding =
          findBindingPlan(plan.bindingShapes, input->binding))
    poolPlan.inputByteOffset = binding->byteOffset;
  if (const CudaTileBindingPlan *binding =
          findBindingPlan(plan.bindingShapes, sum->binding))
    poolPlan.sumByteOffset = binding->byteOffset;
  if (const CudaTileBindingPlan *binding =
          findBindingPlan(plan.bindingShapes, count->binding))
    poolPlan.countByteOffset = binding->byteOffset;
  poolPlan.channels = channels;
  poolPlan.inputH = inputH;
  poolPlan.inputW = inputW;
  poolPlan.outputH = outputH;
  poolPlan.outputW = outputW;
  poolPlan.elemType = elemType;
  LLVM_DEBUG(llvm::dbgs()
             << "[cuda_tile]   matched identity adaptive avg pool C="
             << channels << " input=[" << inputH << "," << inputW
             << "] output=[" << outputH << "," << outputW << "]\n");
  return poolPlan;
}

static std::pair<CudaTileOpEmitter, SmallVector<int64_t, 3>>
generateIdentityAdaptiveAvgPoolKernel(MLIRContext *ctx, StringRef kernelName,
                                      const IdentityAdaptiveAvgPoolPlan &plan,
                                      int64_t tileM, int64_t tileN) {
  int64_t tH = nextPow2(std::min(tileM, plan.outputH));
  int64_t tW = nextPow2(std::min(tileN, plan.outputW));
  constexpr int64_t kMaxTileElements = 16 * 1024;
  int64_t maxChannelsPerTile = std::max<int64_t>(1, kMaxTileElements / (tH * tW));
  int64_t tC = nextPow2(std::min(plan.channels, maxChannelsPerTile));

  SmallVector<int64_t> logicalShape = {plan.channels, plan.outputH,
                                       plan.outputW};
  SmallVector<int64_t> inputStrides = {plan.inputH * plan.inputW,
                                       plan.inputW, 1};
  SmallVector<int64_t> outputStrides = {plan.outputH * plan.outputW,
                                        plan.outputW, 1};
  SmallVector<int64_t> countShape = {plan.outputH, plan.outputW};
  SmallVector<int64_t> countStrides = {plan.outputW, 1};
  SmallVector<int64_t> tile3D = {tC, tH, tW};
  SmallVector<int64_t> tile2D = {tH, tW};
  SmallVector<int64_t, 3> gridDims = {
      (plan.outputW + tW - 1) / tW,
      (plan.outputH + tH - 1) / tH,
      (plan.channels + tC - 1) / tC};

  CudaTileOpEmitter e(ctx);
  e.beginModule(kernelName);
  e.beginEntry("main", plan.numBindings, plan.elemType);

  Value inputPtr = offsetPtrByBytes(e, e.getArg(plan.inputBinding),
                                    plan.elemType, plan.inputByteOffset);
  auto inputView = e.makeTensorView(inputPtr, logicalShape, inputStrides,
                                    plan.elemType);
  auto inputPart = e.makePartitionView(inputView, tile3D, /*zeroPad=*/true);

  Value sumPtr = offsetPtrByBytes(e, e.getArg(plan.sumBinding), plan.elemType,
                                  plan.sumByteOffset);
  auto sumView = e.makeTensorView(sumPtr, logicalShape, outputStrides,
                                  plan.elemType);
  auto sumPart = e.makePartitionView(sumView, tile3D);

  Value countPtr = offsetPtrByBytes(e, e.getArg(plan.countBinding),
                                    plan.elemType, plan.countByteOffset);
  auto countView = e.makeTensorView(countPtr, countShape, countStrides,
                                    plan.elemType);
  auto countPart = e.makePartitionView(countView, tile2D);

  auto [bx, by, bz] = e.getTileBlockId();
  SmallVector<Value> indices3D =
      buildNDIndices(e, logicalShape, tile3D, bx, by, bz);
  auto [tile, token] =
      e.loadViewTko(inputPart, indices3D, tile3D, plan.elemType);
  e.storeViewTko(tile, sumPart, indices3D);

  Value one = e.constSplat(tile2D, plan.elemType, 1.0);
  SmallVector<Value> countIndices = {indices3D[1], indices3D[2]};
  if (gridDims[2] == 1) {
    e.storeViewTko(one, countPart, countIndices);
  } else {
    Value isFirstChannelTile =
        e.emitCmpI(arith::CmpIPredicate::eq, bz, e.constI32(0));
    auto ifOp = e.beginIf(isFirstChannelTile);
    e.storeViewTko(one, countPart, countIndices);
    e.endIf(ifOp);
  }

  e.emitReturn();
  e.endEntry();
  return {std::move(e), gridDims};
}

struct CudaTileMatmulEmissionPlan {
  SmallVector<int64_t> lhsShape;
  SmallVector<int64_t> rhsShape;
  SmallVector<int64_t> resultShape;

  int64_t m = 1;
  int64_t n = 1;
  int64_t k = 1;

  int64_t tileM = 1;
  int64_t tileN = 1;
  int64_t tileK = 1;

  int64_t numBindings = 3;
  int64_t actualNumBindings = 3;
  int64_t lhsBinding = 0;
  int64_t rhsBinding = 1;
  int64_t resultBinding = 2;

  bool rhsTransposed = false;
  Value rhsConstant;

  bool hasSlicedLhs = false;
  SmallVector<int64_t> slicedLhsSourceShape;
  SmallVector<int64_t> slicedLhsOffsets;
  SmallVector<int64_t> slicedLhsSizes;
  SmallVector<int64_t> slicedLhsStrides;

  DenseMap<Value, int64_t> valueToBinding;
};

static CudaTileMatmulEmissionPlan buildMatmulEmissionPlan(
    Operation *innerModule, Operation *primaryOp,
    const CudaTileKernelPlan &plan,
    ArrayRef<CudaTileBindingPlan> bindingShapes, ArrayRef<int64_t> dstShape,
    int64_t tileM, int64_t tileN, int64_t tileK) {
  CudaTileMatmulEmissionPlan emissionPlan;
  CudaTileContractionPlan contractionPlan = plan.contraction;
  emissionPlan.lhsShape = contractionPlan.lhsShape;
  emissionPlan.rhsShape = contractionPlan.rhsShape;
  emissionPlan.resultShape = contractionPlan.resultShape;
  emissionPlan.rhsTransposed = contractionPlan.rhsTransposed;

  LLVM_DEBUG(llvm::dbgs() << "[cuda_tile]   matmul: valid="
                          << contractionPlan.isValid
                          << " M=" << contractionPlan.m
                          << " N=" << contractionPlan.n
                          << " K=" << contractionPlan.k
                          << " bT=" << emissionPlan.rhsTransposed << "\n");

  if (!contractionPlan.isValid) {
    for (auto operand : primaryOp->getOperands()) {
      if (auto type = dyn_cast<ShapedType>(operand.getType())) {
        if (!type.hasStaticShape())
          continue;
        if (emissionPlan.lhsShape.empty()) {
          emissionPlan.lhsShape.assign(type.getShape().begin(),
                                       type.getShape().end());
        } else if (emissionPlan.rhsShape.empty()) {
          emissionPlan.rhsShape.assign(type.getShape().begin(),
                                       type.getShape().end());
        }
      }
    }
    for (auto result : primaryOp->getResults()) {
      if (auto type = dyn_cast<ShapedType>(result.getType())) {
        if (type.hasStaticShape())
          emissionPlan.resultShape.assign(type.getShape().begin(),
                                          type.getShape().end());
      }
    }
    if (emissionPlan.resultShape.empty())
      emissionPlan.resultShape = SmallVector<int64_t>(dstShape);
  }

  emissionPlan.m = contractionPlan.isValid ? contractionPlan.m : 1;
  emissionPlan.n = contractionPlan.isValid
                       ? contractionPlan.n
                       : (emissionPlan.resultShape.empty()
                              ? 1
                              : emissionPlan.resultShape.back());
  emissionPlan.k =
      contractionPlan.isValid
          ? contractionPlan.k
          : (emissionPlan.rhsTransposed
                 ? (emissionPlan.rhsShape.empty()
                        ? 1
                        : emissionPlan.rhsShape.back())
                 : (emissionPlan.lhsShape.empty()
                        ? 1
                        : emissionPlan.lhsShape.back()));
  if (!contractionPlan.isValid) {
    for (int64_t i = 0;
         i + 1 < static_cast<int64_t>(emissionPlan.resultShape.size()); ++i)
      emissionPlan.m *= emissionPlan.resultShape[i];
    if (emissionPlan.resultShape.size() <= 1) {
      emissionPlan.m =
          emissionPlan.resultShape.empty() ? 1 : emissionPlan.resultShape[0];
    }
  }

  int64_t contractionTileM =
      contractionPlan.hasScheduleTiles ? contractionPlan.tileM : tileM;
  int64_t contractionTileN =
      contractionPlan.hasScheduleTiles ? contractionPlan.tileN : tileN;
  int64_t contractionTileK =
      contractionPlan.hasScheduleTiles ? contractionPlan.tileK : tileK;

  emissionPlan.tileM =
      nextPow2(std::min(contractionTileM, emissionPlan.m));
  emissionPlan.tileN =
      nextPow2(std::min(contractionTileN, emissionPlan.n));
  emissionPlan.tileK =
      nextPow2(std::min(contractionTileK, emissionPlan.k));

  int64_t numBindings = bindingShapes.empty()
                            ? int64_t{3}
                            : static_cast<int64_t>(bindingShapes.size());
  int64_t actualNumBindings = getNumBindingArgs(bindingShapes);
  if (actualNumBindings == 0)
    actualNumBindings = numBindings;
  emissionPlan.numBindings = numBindings;
  emissionPlan.actualNumBindings = actualNumBindings;

  innerModule->walk([&](Operation *op) {
    if (op->getName().getStringRef() !=
        "iree_tensor_ext.dispatch.tensor.load")
      return;
    if (op->getNumOperands() < 1)
      return;
    Value src = op->getOperand(0);
    for (int64_t i = 0; i < static_cast<int64_t>(bindingShapes.size()); ++i) {
      if (bindingShapes[i].memref == src) {
        emissionPlan.valueToBinding[op->getResult(0)] = i;
        break;
      }
    }
  });
  for (int64_t i = 0; i < static_cast<int64_t>(bindingShapes.size()); ++i)
    emissionPlan.valueToBinding[bindingShapes[i].memref] = i;

  int64_t bindA = contractionPlan.lhsBinding;
  int64_t bindB = contractionPlan.rhsBinding;
  int64_t bindC = contractionPlan.resultBinding;
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(primaryOp)) {
    auto dpsInputs = linalgOp.getDpsInputs();
    if (bindA < 0 && dpsInputs.size() >= 1) {
      auto it = emissionPlan.valueToBinding.find(dpsInputs[0]);
      if (it != emissionPlan.valueToBinding.end())
        bindA = it->second;
    }
    if (bindB < 0 && dpsInputs.size() >= 2) {
      auto it = emissionPlan.valueToBinding.find(dpsInputs[1]);
      if (it != emissionPlan.valueToBinding.end())
        bindB = it->second;
    }
    auto dpsInits = linalgOp.getDpsInits();
    if (bindC < 0 && !dpsInits.empty()) {
      for (int64_t i = 0; i < static_cast<int64_t>(bindingShapes.size());
           ++i) {
        auto bVal = bindingShapes[i].memref;
        if (auto subspan = bVal.getDefiningOp()) {
          if (subspan->getName().getStringRef() ==
              "hal.interface.binding.subspan") {
            auto flags =
                subspan->getAttrOfType<IntegerAttr>("descriptor_flags");
            if (flags && (flags.getInt() & 0x2))
              bindC = i;
          }
        }
      }
    }
  }
  if (bindA < 0)
    bindA = 0;
  if (bindC < 0)
    bindC = actualNumBindings - 1;

  Value weightConstant = contractionPlan.constantRhs;
  if (bindB < 0) {
    if (!weightConstant) {
      if (auto linalgOp = dyn_cast<linalg::LinalgOp>(primaryOp)) {
        auto dpsInputs = linalgOp.getDpsInputs();
        if (dpsInputs.size() >= 2) {
          Value bInput = dpsInputs[1];
          if (bInput.getDefiningOp<arith::ConstantOp>())
            weightConstant = bInput;
        }
      }
    }
    if (!weightConstant) {
      for (int64_t i = 0; i < numBindings; ++i) {
        if (i != bindA && i != bindC) {
          bindB = i;
          break;
        }
      }
      if (bindB < 0)
        bindB = 1;
    }
  }

  if (weightConstant)
    bindB = -1;

  emissionPlan.lhsBinding = bindA;
  emissionPlan.rhsBinding = bindB;
  emissionPlan.resultBinding = bindC;
  emissionPlan.rhsConstant = weightConstant;

  emissionPlan.slicedLhsSourceShape = contractionPlan.slicedLhsSourceShape;
  emissionPlan.slicedLhsOffsets = contractionPlan.slicedLhsOffsets;
  emissionPlan.slicedLhsSizes = contractionPlan.slicedLhsSizes;
  emissionPlan.slicedLhsStrides = contractionPlan.slicedLhsStrides;
  emissionPlan.hasSlicedLhs = contractionPlan.hasSlicedLhs;

  if (!emissionPlan.hasSlicedLhs) {
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(primaryOp)) {
      auto dpsInputs = linalgOp.getDpsInputs();
      if (!dpsInputs.empty()) {
        auto slicedALoadOp =
            dpsInputs[0]
                .getDefiningOp<IREE::TensorExt::DispatchTensorLoadOp>();
        if (slicedALoadOp) {
          emissionPlan.slicedLhsSourceShape =
              getStaticShapeFromType(slicedALoadOp.getSource().getType());
          emissionPlan.slicedLhsOffsets =
              SmallVector<int64_t>(slicedALoadOp.getStaticOffsets());
          emissionPlan.slicedLhsSizes =
              SmallVector<int64_t>(slicedALoadOp.getStaticSizes());
          emissionPlan.slicedLhsStrides =
              SmallVector<int64_t>(slicedALoadOp.getStaticStrides());
          emissionPlan.hasSlicedLhs =
              !slicedALoadOp.isLoadOfWholeSource() ||
              !isIdentitySlice(emissionPlan.slicedLhsOffsets,
                               emissionPlan.slicedLhsStrides);
        }
      }
    }
  }

  return emissionPlan;
}

//===----------------------------------------------------------------------===//
// buildCudaTileKernel — main codegen entry point
//===----------------------------------------------------------------------===//

/// Walk the inner module of a hal.executable.variant, extract metadata from
/// annotated ops, build a cuda_tile module with the appropriate kernel, and
/// serialize it to tilebc bytecode.
///
/// Returns {tilebcBytes, gridDims} on success.
static FailureOr<std::pair<std::string, SmallVector<int64_t, 3>>>
buildCudaTileKernel(MLIRContext *ctx, Operation *innerModule,
                    StringRef kernelName, const CudaTileOptions &options) {
  LLVM_DEBUG(llvm::dbgs() << "[cuda_tile] extracting plan...\n");
  CudaTileKernelPlan plan = extractCudaTileKernelPlan(innerModule, options);
  LLVM_DEBUG(llvm::dbgs() << "[cuda_tile] plan extracted OK\n");
  if (failed(dumpCudaTileKernelPlanIfRequested(innerModule, options, plan)))
    return failure();
  Operation *primaryOp = plan.primaryOp;
  int taggedOpCount = plan.taggedOpCount;
  CudaTileConvPlan convPlan = plan.conv;
  CudaTileSemanticKind semanticKind = plan.semanticKind;
  CudaTileLoweringStrategy loweringStrategy = plan.loweringStrategy;

  // If still no op found, this is a pure data-movement dispatch.
  // Generate a copy kernel using shapes from tensor values in the module.
  if (!primaryOp)
    return emitPureDataMovementDispatch(ctx, kernelName, options, plan);

  std::string kernelClass = plan.kernelClass;
  SmallVector<int64_t> srcShape = plan.srcShape;
  SmallVector<int64_t> dstShape = plan.dstShape;
  Type elemType = plan.elementType;
  if (srcShape.empty() || !elemType)
    return failure();

  LLVM_DEBUG({
    llvm::dbgs() << "[cuda_tile]   class=" << kernelClass << " src=[";
    for (size_t i = 0; i < srcShape.size(); ++i)
      llvm::dbgs() << (i ? "," : "") << srcShape[i];
    llvm::dbgs() << "] dst=[";
    for (size_t i = 0; i < dstShape.size(); ++i)
      llvm::dbgs() << (i ? "," : "") << dstShape[i];
    llvm::dbgs() << "] tagged=" << taggedOpCount
                 << " generic=" << plan.genericOpCount
                 << " conv=" << (plan.conv ? "valid" : "invalid")
                 << " semantic=" << (int)plan.semanticKind
                 << " lowering="
                 << stringifyCudaTileLoweringStrategy(loweringStrategy)
                 << " binding_args=" << getNumBindingArgs(plan.bindingShapes)
                 << " subspans=" << plan.bindingShapes.size() << "\n";
  });

  int64_t tileM = plan.schedule.tileM ? plan.schedule.tileM : options.tileM;
  int64_t tileN = plan.schedule.tileN ? plan.schedule.tileN : options.tileN;
  int64_t tileK = plan.schedule.tileK ? plan.schedule.tileK : options.tileK;

  auto &bindingShapes = plan.bindingShapes;
  SmallVector<CudaTileGatherSource, 4> gatherSources =
      buildGatherSources(plan);
  int genericOpCount = plan.genericOpCount;

  auto poolPlan = matchIdentityAdaptiveAvgPoolDispatch(plan);
  if (succeeded(poolPlan)) {
    auto [e, grid] = generateIdentityAdaptiveAvgPoolKernel(
        ctx, kernelName, *poolPlan, tileM, tileN);
    return serializeCudaTileKernel(std::move(e), std::move(grid));
  }

  // Multi-op fused dispatches.
  if (taggedOpCount > 1 || genericOpCount > 1) {
    SmallVector<Operation *> taggedOps(plan.taggedOps.begin(),
                                       plan.taggedOps.end());

    // Strategy 1: If a matmul-like op is present, promote it as primary.
    // The planner owns the semantic/lowering classification; this fallback
    // scan only exists for older dispatches that lack complete fused-op facts.
    {
      auto isPromotableStrategy =
          [](CudaTileLoweringStrategy strategy) -> bool {
        return strategy == CudaTileLoweringStrategy::Matmul ||
               strategy == CudaTileLoweringStrategy::PointwiseConvAsMatmul ||
               strategy == CudaTileLoweringStrategy::DirectConv2D ||
               strategy == CudaTileLoweringStrategy::Pooling;
      };

      Operation *promotedOp = nullptr;
      CudaTileLoweringStrategy promotedStrategy =
          CudaTileLoweringStrategy::Unsupported;
      CudaTileSemanticKind promotedSemantic = CudaTileSemanticKind::Unknown;
      CudaTileConvPlan promotedConvPlan;

      for (const CudaTileFusedOpPlan &fusedOp : plan.fusedOps) {
        if (!isPromotableStrategy(fusedOp.loweringStrategy))
          continue;
        promotedOp = fusedOp.op;
        promotedStrategy = fusedOp.loweringStrategy;
        promotedSemantic = fusedOp.semanticKind;
        promotedConvPlan = fusedOp.conv;
        break;
      }

      if (!promotedOp) {
        for (auto *op : taggedOps) {
          auto k = op->getAttrOfType<StringAttr>("cuda_tile.kernel_class");
          if (k && k.getValue() == "matmul") {
            // Skip multi-reduction generics (NCHW convolutions) that were
            // tagged as "matmul" by the contractions pass but can't be
            // handled by the flat matmul emitter.
            if (auto genOp = dyn_cast<linalg::GenericOp>(op)) {
              if (genOp.getNumReductionLoops() > 1) {
                auto cp = extractCudaTileConvPlan(genOp);
                if (!cp)
                  continue;
              }
            }
            promotedOp = op;
            promotedStrategy = CudaTileLoweringStrategy::Matmul;
            promotedSemantic = CudaTileSemanticKind::Contraction;
            break;
          }
        }
      }

      // Also check untagged generics for contractions. Skip multi-reduction
      // generics unless the conv plan can handle them.
      if (!promotedOp) {
        innerModule->walk([&](linalg::GenericOp genOp) {
          if (promotedOp)
            return;
          if (genOp.getNumReductionLoops() == 0 ||
              genOp.getNumDpsInputs() != 2)
            return;

          CudaTileConvPlan cp;
          if (genOp.getNumReductionLoops() > 1) {
            cp = extractCudaTileConvPlan(genOp);
            if (!cp)
              return;
          }

          // Check for mulf+addf body (contraction pattern).
          auto &body = genOp.getRegion().front();
          bool hasMul = false, hasAdd = false;
          for (auto &op : body.without_terminator()) {
            if (isa<arith::MulFOp>(&op))
              hasMul = true;
            if (isa<arith::AddFOp>(&op))
              hasAdd = true;
          }
          if (!hasMul || !hasAdd)
            return;

          promotedOp = genOp;
          promotedConvPlan = cp;
          if (cp.mode == CudaTileConvLoweringMode::DirectConv2D) {
            promotedStrategy = CudaTileLoweringStrategy::DirectConv2D;
            promotedSemantic = CudaTileSemanticKind::WindowedReduction;
          } else if (cp.mode == CudaTileConvLoweringMode::PointwiseMatmul) {
            promotedStrategy =
                CudaTileLoweringStrategy::PointwiseConvAsMatmul;
            promotedSemantic = CudaTileSemanticKind::WindowedReduction;
          } else {
            promotedStrategy = CudaTileLoweringStrategy::Matmul;
            promotedSemantic = CudaTileSemanticKind::Contraction;
          }
        });
      }

      // Also check for pooling (windowed reduction without mulf+addf).
      if (!promotedOp) {
        innerModule->walk([&](linalg::GenericOp genOp) {
          if (promotedOp)
            return;
          if (genOp.getNumReductionLoops() == 0)
            return;
          auto pp = extractPoolingPlan(genOp);
          if (!pp)
            return;
          promotedOp = genOp;
          promotedConvPlan = pp;
          promotedStrategy = CudaTileLoweringStrategy::Pooling;
          promotedSemantic = CudaTileSemanticKind::WindowedReduction;
        });
      }

      if (promotedOp) {
        // Promote the matmul-like or pooling op as primary and fall through.
        primaryOp = promotedOp;
        taggedOpCount = 1;
        if (promotedStrategy == CudaTileLoweringStrategy::DirectConv2D)
          kernelClass = "conv";
        else if (promotedStrategy == CudaTileLoweringStrategy::Pooling)
          kernelClass = "pooling";
        else
          kernelClass = "matmul";
        convPlan = promotedConvPlan;
        if (!convPlan) {
          if (auto genOp = dyn_cast<linalg::GenericOp>(primaryOp))
            convPlan = extractCudaTileConvPlan(genOp);
        }
        if (promotedStrategy == CudaTileLoweringStrategy::Unsupported) {
          if (convPlan.mode == CudaTileConvLoweringMode::DirectConv2D) {
            promotedStrategy = CudaTileLoweringStrategy::DirectConv2D;
            promotedSemantic = CudaTileSemanticKind::WindowedReduction;
          } else if (convPlan.mode ==
                     CudaTileConvLoweringMode::PointwiseMatmul) {
            promotedStrategy =
                CudaTileLoweringStrategy::PointwiseConvAsMatmul;
            promotedSemantic = CudaTileSemanticKind::WindowedReduction;
          } else if (convPlan.mode == CudaTileConvLoweringMode::Pooling) {
            promotedStrategy = CudaTileLoweringStrategy::Pooling;
            promotedSemantic = CudaTileSemanticKind::WindowedReduction;
          } else {
            promotedStrategy = CudaTileLoweringStrategy::Matmul;
            promotedSemantic = CudaTileSemanticKind::Contraction;
          }
        }

        semanticKind = promotedSemantic;
        loweringStrategy = promotedStrategy;
        if (semanticKind == CudaTileSemanticKind::Unknown) {
          semanticKind =
              loweringStrategy == CudaTileLoweringStrategy::DirectConv2D ||
                      loweringStrategy ==
                          CudaTileLoweringStrategy::PointwiseConvAsMatmul ||
                      loweringStrategy == CudaTileLoweringStrategy::Pooling
                  ? CudaTileSemanticKind::WindowedReduction
                  : CudaTileSemanticKind::Contraction;
        }

        if (auto classAttr =
                primaryOp->getAttrOfType<StringAttr>("cuda_tile.kernel_class"))
          kernelClass = classAttr.getValue().str();

        // Re-extract shapes from the promoted primary op.
        srcShape.clear();
        dstShape.clear();
        for (auto operand : primaryOp->getOperands()) {
          if (auto shaped = dyn_cast<ShapedType>(operand.getType())) {
            if (shaped.hasStaticShape() && srcShape.empty())
              srcShape.assign(shaped.getShape().begin(),
                              shaped.getShape().end());
          }
        }
        for (auto result : primaryOp->getResults()) {
          if (auto shaped = dyn_cast<ShapedType>(result.getType())) {
            if (shaped.hasStaticShape())
              dstShape.assign(shaped.getShape().begin(),
                              shaped.getShape().end());
          }
        }
        if (dstShape.empty())
          dstShape = srcShape;
        // Fall through to single-op dispatch below.
        goto singleOpDispatch;
      }
    }

    // Strategy 2: Generic fused reduction+elementwise kernel.
    // Walks ALL linalg.generic ops in the dispatch dynamically.
    // Handles softmax, layer_norm, rms_norm, batch_norm, and any
    // combination of reductions and elementwise ops in one fused dispatch.
    // Also subsumes the all-elementwise case (zero reductions).
    {
      SmallVector<linalg::GenericOp> allGenerics;
      for (const CudaTileFusedOpPlan &fusedOp : plan.fusedOps) {
        if (auto genericOp = dyn_cast<linalg::GenericOp>(fusedOp.op))
          allGenerics.push_back(genericOp);
      }

      // Check if any generic is a reduction or elementwise.
      // Skip dispatches containing multi-reduction generics (convolutions)
      // that we can't handle in the fused walker.
      bool hasReduceOrEW = false;
      bool hasUnhandledConv = false;
      for (auto genOp : allGenerics) {
        if (genOp.getNumReductionLoops() > 1) {
          auto cp = extractCudaTileConvPlan(genOp);
          if (!cp)
            hasUnhandledConv = true;
        }
        if (genOp.getNumReductionLoops() > 0 ||
            genOp.getNumParallelLoops() > 0)
          hasReduceOrEW = true;
      }
      if (hasUnhandledConv) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[cuda_tile]   bailing: unhandled conv\n");
        return failure();
      }
      LLVM_DEBUG(llvm::dbgs()
                 << "[cuda_tile]   allGenerics=" << allGenerics.size()
                 << " hasReduceOrEW=" << hasReduceOrEW << "\n");

      if (hasReduceOrEW && !allGenerics.empty()) {
        auto shape = dstShape;
        int64_t rank = shape.size();

        // Determine which tensor dims are reduced by any op.
        // Tiles must cover the full extent of reduced dims for correctness.
        llvm::DenseSet<int64_t> reducedDims;
        for (auto genOp : allGenerics) {
          if (genOp.getNumReductionLoops() == 0)
            continue;
          int64_t rd = extractReduceDim(genOp, shape);
          if (rd >= 0)
            reducedDims.insert(rd);
        }

        // Compute tile shape: full-width on reduced dims, tiled on others.
        SmallVector<int64_t> tileShape(rank);
        for (int64_t d = 0; d < rank; ++d) {
          if (reducedDims.contains(d))
            tileShape[d] = nextPow2(shape[d]); // full width
          else if (d == rank - 1)
            tileShape[d] = nextPow2(std::min(tileN, shape[d]));
          else if (d == rank - 2)
            tileShape[d] = nextPow2(std::min(tileM, shape[d]));
          else
            tileShape[d] = shape[d]; // batch
        }

        // Compute grid dims.
        SmallVector<int64_t, 3> gridDims = {1, 1, 1};
        if (rank >= 1 && !reducedDims.contains(rank - 1))
          gridDims[0] = (shape[rank - 1] + tileShape[rank - 1] - 1) /
                        tileShape[rank - 1];
        if (rank >= 2 && !reducedDims.contains(rank - 2))
          gridDims[1] = (shape[rank - 2] + tileShape[rank - 2] - 1) /
                        tileShape[rank - 2];
        if (rank > 2) {
          int64_t batchTiles = 1;
          for (int64_t d = 0; d < rank - 2; ++d)
            batchTiles *= shape[d];
          gridDims[2] = batchTiles;
        }

        int64_t numSubspans = bindingShapes.size();
        int64_t numBindings = getNumBindingArgs(bindingShapes);
        if (numBindings < 2)
          numBindings = 2;
        if (numSubspans == 0)
          numSubspans = numBindings;

        CudaTileOpEmitter e(ctx);
        e.beginModule(kernelName);
        e.beginEntry("main", numBindings, elemType);

        // Create per-binding views (handles broadcast shapes).
        SmallVector<Value> partViews;
        SmallVector<SmallVector<int64_t>> bindTileShapes;
        for (int64_t i = 0; i < numSubspans; ++i) {
          auto bShape =
              (i < (int64_t)bindingShapes.size() &&
               !bindingShapes[i].shape.empty())
                  ? SmallVector<int64_t>(bindingShapes[i].shape)
                  : SmallVector<int64_t>(shape);
          auto bStrides = computeRowMajorStrides(bShape);
          // For broadcast inputs, compute tile shape from their own shape.
          SmallVector<int64_t> bTile(bShape.size());
          for (int64_t d = 0; d < (int64_t)bShape.size(); ++d) {
            // If this dim in the output is reduced, use full width.
            // Otherwise use standard tiling.
            if (d < rank && reducedDims.contains(d))
              bTile[d] = nextPow2(bShape[d]);
            else
              bTile[d] = nextPow2(
                  std::min(d == (int64_t)bShape.size() - 1 ? tileN : tileM,
                           bShape[d]));
          }
          Value ptr = i < (int64_t)bindingShapes.size()
                          ? getSubspanArg(e, bindingShapes[i], elemType)
                          : e.getArg(i);
          auto tv = e.makeTensorView(ptr, bShape, bStrides, elemType);
          auto pv = e.makePartitionView(tv, bTile);
          partViews.push_back(pv);
          bindTileShapes.push_back(bTile);
        }

        auto [bx, by, bz] = e.getTileBlockId();
        SmallVector<Value> outIndices(rank);
        if (rank >= 1)
          outIndices[rank - 1] = bx;
        if (rank >= 2)
          outIndices[rank - 2] = by;
        if (rank >= 3)
          outIndices[0] = bz;

        // Helper: load tile from binding with broadcast support.
        DenseMap<int64_t, Value> loadedBindings;
        auto loadWithBroadcast = [&](int64_t bindIdx) -> Value {
          auto it = loadedBindings.find(bindIdx);
          if (it != loadedBindings.end())
            return it->second;
          auto &bInfo = bindingShapes[bindIdx];
          auto &bTile = bindTileShapes[bindIdx];
          int64_t bRank = bInfo.shape.size();
          SmallVector<Value> bIndices;
          if (bRank == rank)
            bIndices = SmallVector<Value>(outIndices);
          else
            for (int64_t d = 0; d < bRank; ++d)
              bIndices.push_back(outIndices[d]);
          auto [tile, tok] = e.loadViewTko(partViews[bindIdx], bIndices,
                                           bTile, elemType);
          if (bRank < rank) {
            SmallVector<int64_t> reshapeShape, broadcastShape;
            int64_t bDim = 0;
            for (int64_t d = 0; d < rank; ++d) {
              if (bDim < bRank && bInfo.shape[bDim] == shape[d]) {
                reshapeShape.push_back(bTile[bDim]);
                broadcastShape.push_back(tileShape[d]);
                bDim++;
              } else {
                reshapeShape.push_back(1);
                broadcastShape.push_back(tileShape[d]);
              }
            }
            tile = e.reshape(tile, reshapeShape, elemType);
            if (reshapeShape != broadcastShape)
              tile = e.broadcastTile(tile, broadcastShape, elemType);
          }
          loadedBindings[bindIdx] = tile;
          return tile;
        };

        // Walk all generics in program order, emitting cuda_tile ops.
        DenseMap<Value, Value> tileValueMap;
        Value current;

        for (auto genOp : allGenerics) {
          // Resolve inputs: from tileValueMap, binding, or constant.
          SmallVector<Value> inputTiles;
          for (auto input : genOp.getDpsInputs()) {
            auto mapIt = tileValueMap.find(input);
            if (mapIt != tileValueMap.end()) {
              inputTiles.push_back(mapIt->second);
              continue;
            }
            bool found = false;
            for (int64_t i = 0; i < (int64_t)bindingShapes.size(); ++i) {
              if (bindingShapes[i].memref == input) {
                inputTiles.push_back(loadWithBroadcast(i));
                found = true;
                break;
              }
              if (auto loadOp =
                      input.getDefiningOp<IREE::TensorExt::DispatchTensorLoadOp>();
                  loadOp && bindingShapes[i].memref == loadOp.getSource()) {
                inputTiles.push_back(loadWithBroadcast(i));
                found = true;
                break;
              }
            }
            if (found)
              continue;
            // Constant tensor input (e.g., LN weight/bias).
            if (auto cstOp = input.getDefiningOp<arith::ConstantOp>()) {
              SmallVector<float> cstData;
              auto cstAttr = cstOp.getValue();
              if (auto dense = dyn_cast<DenseElementsAttr>(cstAttr)) {
                for (auto v : dense.getValues<float>())
                  cstData.push_back(v);
              } else if (auto resAttr =
                             dyn_cast<DenseResourceElementsAttr>(cstAttr)) {
                auto blob = resAttr.getRawHandle().getBlob();
                if (blob) {
                  auto data = blob->getData();
                  auto *f = reinterpret_cast<const float *>(data.data());
                  cstData.assign(f, f + data.size() / sizeof(float));
                }
              }
              if (!cstData.empty()) {
                auto cstType = dyn_cast<ShapedType>(input.getType());
                auto cstShape = cstType ? cstType.getShape()
                                        : ArrayRef<int64_t>{};
                int64_t cRank = cstShape.size();
                SmallVector<int64_t> cTile;
                for (int64_t d = 0; d < cRank; ++d)
                  cTile.push_back(nextPow2(cstShape[d]));
                int64_t cElems = 1;
                for (auto d : cTile)
                  cElems *= d;
                SmallVector<float> padded(cElems, 0.0f);
                // Generic N-dimensional copy from cstData to padded.
                int64_t totalSrcElems = 1;
                for (int64_t d = 0; d < cRank; ++d)
                  totalSrcElems *= cstShape[d];
                for (int64_t flat = 0;
                     flat < totalSrcElems &&
                     flat < (int64_t)cstData.size();
                     ++flat) {
                  // Decompose flat index into N-d coordinates.
                  int64_t rem = flat;
                  SmallVector<int64_t> coords(cRank);
                  for (int64_t d = cRank - 1; d >= 0; --d) {
                    coords[d] = rem % cstShape[d];
                    rem /= cstShape[d];
                  }
                  // Compute padded flat index using cTile strides.
                  int64_t paddedIdx = 0;
                  int64_t stride = 1;
                  for (int64_t d = cRank - 1; d >= 0; --d) {
                    paddedIdx += coords[d] * stride;
                    stride *= cTile[d];
                  }
                  if (paddedIdx < cElems)
                    padded[paddedIdx] = cstData[flat];
                }
                auto tileCstType =
                    ::mlir::cuda_tile::TileType::get(ctx, cTile, elemType);
                auto tensorCstType =
                    RankedTensorType::get(cTile, elemType);
                auto attr = DenseElementsAttr::get(
                    tensorCstType,
                    ArrayRef<float>(padded.data(), padded.size()));
                Value tile =
                    e.builder()
                        .create<::mlir::cuda_tile::ConstantOp>(
                            e.builder().getUnknownLoc(), tileCstType,
                            cast<DenseTypedElementsAttr>(
                                attr.reshape(
                                    cast<ShapedType>(tileCstType))))
                        .getResult();
                // Broadcast to full tileShape if lower rank.
                if (cRank < rank) {
                  SmallVector<int64_t> rshp, bshp;
                  int64_t cDim = 0;
                  for (int64_t d = 0; d < rank; ++d) {
                    if (cDim < cRank && cstShape[cDim] == shape[d]) {
                      rshp.push_back(cTile[cDim]);
                      bshp.push_back(tileShape[d]);
                      cDim++;
                    } else {
                      rshp.push_back(1);
                      bshp.push_back(tileShape[d]);
                    }
                  }
                  tile = e.reshape(tile, rshp, elemType);
                  if (rshp != bshp)
                    tile = e.broadcastTile(tile, bshp, elemType);
                }
                inputTiles.push_back(tile);
                continue;
              }
            }
          }

          if (genOp.getNumReductionLoops() > 0) {
            // --- REDUCTION ---
            // Extract combiner and any pre-reduction elementwise ops.
            // E.g., square-sum body: mulf(x,x) then addf(sq, acc).
            StringRef combiner =
                matchReduceCombiner(genOp.getRegion());
            SmallVector<StringRef> preOps; // ops applied before reduce
            if (combiner.empty()) {
              // Multi-op body: extract pre-reduce and combiner ops.
              // Walk body ops: everything before the accumulator op is
              // a pre-reduce elementwise; the last 2-input op is the combiner.
              Block &block = genOp.getRegion().front();
              for (auto &op : block.without_terminator()) {
                StringRef name = mapArithToCudaTile(&op);
                if (name.empty())
                  name = mapMathToCudaTile(&op);
                if (!name.empty()) {
                  // If this op uses the block's second arg (accumulator),
                  // it's the combiner. Otherwise it's a pre-reduce op.
                  bool usesAcc = false;
                  for (auto operand : op.getOperands()) {
                    if (operand == block.getArgument(block.getNumArguments() - 1))
                      usesAcc = true;
                  }
                  if (usesAcc)
                    combiner = name;
                  else
                    preOps.push_back(name);
                }
              }
              if (combiner.empty())
                combiner = "addf";
            }

            int64_t reduceDim = extractReduceDim(genOp, shape);
            if (reduceDim < 0)
              reduceDim = rank - 1;

            // Compute reduced tile shape (drop the reduced dim).
            SmallVector<int64_t> reducedTileShape;
            for (int64_t d = 0; d < (int64_t)tileShape.size(); ++d) {
              if (d != reduceDim)
                reducedTileShape.push_back(tileShape[d]);
            }
            if (reducedTileShape.empty())
              reducedTileShape.push_back(1);

            Value inputTile =
                inputTiles.empty() ? current : inputTiles[0];

            // Apply pre-reduction ops (e.g., square for sum-of-squares).
            for (auto preOp : preOps)
              inputTile = e.emitElementwise(preOp, {inputTile, inputTile});

            current = e.reduce(inputTile, reduceDim, combiner,
                               reducedTileShape, elemType);
            e.restoreEntryInsertionPoint();

            // Reshape+broadcast back to full tileShape for downstream ops.
            SmallVector<int64_t> reshapeShape;
            for (int64_t d = 0; d < (int64_t)tileShape.size(); ++d)
              reshapeShape.push_back(d == reduceDim ? 1 : tileShape[d]);
            current = e.reshape(current, reshapeShape, elemType);
            if (reshapeShape != SmallVector<int64_t>(tileShape))
              current = e.broadcastTile(current, tileShape, elemType);

          } else {
            // --- ELEMENTWISE ---
            // Walk the body's SSA graph to properly route data flow.
            // Map block arguments to input tiles, constants to constSplat.
            auto &body = genOp.getRegion().front();
            DenseMap<Value, Value> bodyMap;

            // Map block args to input tiles.
            int64_t numDpsInputs = genOp.getNumDpsInputs();
            for (int64_t a = 0;
                 a < numDpsInputs && a < (int64_t)inputTiles.size(); ++a)
              bodyMap[body.getArgument(a)] = inputTiles[a];

            // Walk body ops in SSA order.
            for (auto &op : body.without_terminator()) {
              // Handle CmpFOp directly (needs predicate/ordering attrs).
              if (auto cmpOp = dyn_cast<arith::CmpFOp>(&op)) {
                SmallVector<Value> opInputs;
                bool allResolved = true;
                for (auto operand : op.getOperands()) {
                  auto it = bodyMap.find(operand);
                  if (it != bodyMap.end()) {
                    opInputs.push_back(it->second);
                  } else if (auto cstOp =
                                 operand.getDefiningOp<arith::ConstantOp>()) {
                    double val = 0.0;
                    if (auto fAttr = dyn_cast<FloatAttr>(cstOp.getValue()))
                      val = fAttr.getValueAsDouble();
                    else if (auto iAttr =
                                 dyn_cast<IntegerAttr>(cstOp.getValue()))
                      val = static_cast<double>(iAttr.getInt());
                    opInputs.push_back(
                        e.constSplat(tileShape, elemType, val));
                  } else {
                    allResolved = false;
                  }
                }
                if (allResolved && opInputs.size() == 2) {
                  Value result = e.emitCmpF(cmpOp.getPredicate(), opInputs[0],
                                            opInputs[1]);
                  bodyMap[op.getResult(0)] = result;
                }
                continue;
              }

              StringRef name = mapArithToCudaTile(&op);
              if (name.empty())
                name = mapMathToCudaTile(&op);
              if (name.empty())
                continue; // skip unknown ops

              // Resolve operands: from bodyMap or constants.
              SmallVector<Value> opInputs;
              bool allResolved = true;
              for (auto operand : op.getOperands()) {
                auto it = bodyMap.find(operand);
                if (it != bodyMap.end()) {
                  opInputs.push_back(it->second);
                } else if (auto cstOp =
                               operand.getDefiningOp<arith::ConstantOp>()) {
                  // Scalar constant → splat to tile shape.
                  double val = 0.0;
                  if (auto fAttr = dyn_cast<FloatAttr>(cstOp.getValue()))
                    val = fAttr.getValueAsDouble();
                  else if (auto iAttr =
                               dyn_cast<IntegerAttr>(cstOp.getValue()))
                    val = static_cast<double>(iAttr.getInt());
                  opInputs.push_back(
                      e.constSplat(tileShape, elemType, val));
                } else {
                  allResolved = false;
                }
              }

              if (allResolved && !opInputs.empty()) {
                Value result = e.emitElementwise(name, opInputs);
                if (op.getNumResults() > 0)
                  bodyMap[op.getResult(0)] = result;
                current = result;
              }
            }

            // Use the yielded value as the result.
            if (auto yieldOp =
                    dyn_cast<linalg::YieldOp>(body.getTerminator())) {
              if (!yieldOp.getOperands().empty()) {
                auto it = bodyMap.find(yieldOp.getOperand(0));
                if (it != bodyMap.end())
                  current = it->second;
              }
            }
          }

          // Map outputs for downstream ops.
          for (auto init : genOp.getDpsInits())
            tileValueMap[init] = current;
          for (auto result : genOp->getResults())
            tileValueMap[result] = current;
        }

        e.storeViewTko(current, partViews.back(), outIndices);
        e.emitReturn();
        e.endEntry();

        return serializeCudaTileKernel(std::move(e), gridDims);
      }
    }

    // Strategy 3: copy fallback.
    auto shape = dstShape.empty() ? srcShape : dstShape;
    auto [e, grid] =
        generateCopyKernel(ctx, kernelName, shape, elemType, tileM, tileN);
    return serializeCudaTileKernel(std::move(e), std::move(grid));
  }

  singleOpDispatch:

  //=== Phase 1: Data Movement ===//

  if (isDataMovementStrategy(loweringStrategy))
    return emitDataMovementKernel(ctx, kernelName, primaryOp, loweringStrategy,
                                  srcShape, dstShape, elemType, tileM, tileN);

  //=== Phase 3: Pooling (windowed reduction) — before generic reductions ===//

  if (loweringStrategy == CudaTileLoweringStrategy::Pooling ||
      (convPlan.mode == CudaTileConvLoweringMode::Pooling &&
       convPlan.spatialRank == 2))
    return emitPoolingKernel(ctx, kernelName, convPlan, elemType, tileM, tileN);

  //=== Phase 3b: Reductions ===//

  if (loweringStrategy == CudaTileLoweringStrategy::Reduction ||
      semanticKind == CudaTileSemanticKind::Reduction) {
    auto result = emitReductionKernel(ctx, kernelName, plan, primaryOp,
                                      srcShape, dstShape, elemType, tileM,
                                      tileN);
    if (succeeded(result))
      return result;
  }

  //=== Phase 4: Contractions ===//

  if (loweringStrategy == CudaTileLoweringStrategy::DirectConv2D)
    return emitDirectConv2DKernel(ctx, kernelName, convPlan, elemType, tileM,
                                  tileN, tileK);

  if (loweringStrategy == CudaTileLoweringStrategy::Matmul ||
      loweringStrategy == CudaTileLoweringStrategy::PointwiseConvAsMatmul ||
      semanticKind == CudaTileSemanticKind::Contraction) {

    CudaTileMatmulEmissionPlan matmulPlan = buildMatmulEmissionPlan(
        innerModule, primaryOp, plan, bindingShapes, dstShape, tileM, tileN,
        tileK);

    SmallVector<int64_t> &shA = matmulPlan.lhsShape;
    SmallVector<int64_t> &shB = matmulPlan.rhsShape;
    SmallVector<int64_t> &shC = matmulPlan.resultShape;
    int64_t M = matmulPlan.m;
    int64_t N = matmulPlan.n;
    int64_t K = matmulPlan.k;
    int64_t aTM = matmulPlan.tileM;
    int64_t aTN = matmulPlan.tileN;
    int64_t aTK = matmulPlan.tileK;
    int64_t actualNumBindings = matmulPlan.actualNumBindings;
    int64_t bindA = matmulPlan.lhsBinding;
    int64_t bindB = matmulPlan.rhsBinding;
    int64_t bindC = matmulPlan.resultBinding;
    bool bTransposed = matmulPlan.rhsTransposed;
    Value weightConstant = matmulPlan.rhsConstant;
    bool hasSlicedA = matmulPlan.hasSlicedLhs;
    SmallVector<int64_t> &slicedASourceShape =
        matmulPlan.slicedLhsSourceShape;
    SmallVector<int64_t> &slicedAOffsets = matmulPlan.slicedLhsOffsets;
    SmallVector<int64_t> &slicedASizes = matmulPlan.slicedLhsSizes;
    SmallVector<int64_t> &slicedAStrides = matmulPlan.slicedLhsStrides;
    DenseMap<Value, int64_t> &valueToBind = matmulPlan.valueToBinding;

    bool slicedPointwiseRankOK =
        (slicedASourceShape.size() == 3 && shA.size() == 3 &&
         shC.size() == 3) ||
        (slicedASourceShape.size() == 4 && shA.size() == 4 &&
         shC.size() == 4);
    if (hasSlicedA && !weightConstant && bindA >= 0 && bindB >= 0 &&
        bindC >= 0 && slicedPointwiseRankOK && shB.size() == 2 &&
        llvm::equal(slicedASizes, shA)) {
      auto [e, grid] = generateStridedPointwiseMatmul2DKernel(
          ctx, kernelName, actualNumBindings, bindA, bindB, bindC,
          slicedASourceShape, slicedAOffsets, slicedAStrides, shA, shB, shC,
          elemType, tileM, tileN, tileK);
      return serializeCudaTileKernel(std::move(e), std::move(grid));
    }

    CudaTileOpEmitter e(ctx);
    e.beginModule(kernelName);
    e.beginEntry("main", actualNumBindings, elemType);

    // Use 2D shapes for the kernel (flatten batch dims if present).
    SmallVector<int64_t> shA2 = {M, K}, shC2 = {M, N};
    SmallVector<int64_t> stA = {K, 1}, stC = {N, 1};

    auto ptrA = getBindingArg(e, plan.bindingShapes, bindA, elemType);
    auto vA = e.makeTensorView(ptrA, shA2, stA, elemType);

    // When B is transposed (PyTorch [N,K]), use physical shape [N,K]
    // and permute loaded tiles to [K,N] before mmaf.
    SmallVector<int64_t> shB2, stB, tB;
    if (bTransposed) {
      shB2 = {N, K};
      stB = {K, 1};
      tB = {aTN, aTK};
    } else {
      shB2 = {K, N};
      stB = {N, 1};
      tB = {aTK, aTN};
    }
    Value pBv; // partition view for B (or null if B is a constant)
    if (bindB >= 0) {
      auto ptrB = getBindingArg(e, plan.bindingShapes, bindB, elemType);
      auto vB = e.makeTensorView(ptrB, shB2, stB, elemType);
      pBv = e.makePartitionView(vB, tB);
    }

    auto ptrC = getBindingArg(e, plan.bindingShapes, bindC, elemType);
    auto vC = e.makeTensorView(ptrC, shC2, stC, elemType);
    SmallVector<int64_t> tA = {aTM, aTK}, tC = {aTM, aTN};
    auto pA = e.makePartitionView(vA, tA);
    auto pC = e.makePartitionView(vC, tC);
    SmallVector<Value> primaryInputs;
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(primaryOp)) {
      primaryInputs.append(linalgOp.getDpsInputs().begin(),
                           linalgOp.getDpsInputs().end());
    }
    auto findFusedOpPlanFor = [&](Operation *op)
        -> const CudaTileFusedOpPlan * {
      for (const CudaTileFusedOpPlan &fusedOp : plan.fusedOps) {
        if (fusedOp.op == op)
          return &fusedOp;
      }
      return nullptr;
    };

    auto [bx, by, bz] = e.getTileBlockId();
    auto accInit = e.constSplat(tC, elemType, 0.0);

    int64_t nK = (K + aTK - 1) / aTK;
    auto lb = e.constI32(0);
    auto ub = e.constI32(nK);
    auto step = e.constI32(1);

    auto forOp = e.beginFor(lb, ub, step, ValueRange{accInit});
    Value iv = forOp.getInductionVar();
    Value iterAcc = forOp.getRegionIterValues()[0];

    auto getDenseFloatConstant = [&](Value value, SmallVector<float> &rawData,
                                     SmallVector<int64_t> &shape) -> bool {
      auto cstOp = value.getDefiningOp<arith::ConstantOp>();
      if (!cstOp)
        return false;

      if (auto shapedType = dyn_cast<ShapedType>(value.getType()))
        shape.assign(shapedType.getShape().begin(), shapedType.getShape().end());

      auto attrVal = cstOp.getValue();
      if (auto dense = dyn_cast<DenseElementsAttr>(attrVal)) {
        for (auto val : dense.getValues<float>())
          rawData.push_back(val);
      } else if (auto resAttr = dyn_cast<DenseResourceElementsAttr>(attrVal)) {
        auto blob = resAttr.getRawHandle().getBlob();
        if (blob) {
          auto data = blob->getData();
          auto *floats = reinterpret_cast<const float *>(data.data());
          int64_t numElem = data.size() / sizeof(float);
          rawData.assign(floats, floats + numElem);
        }
      }
      return !rawData.empty();
    };

    auto emitDenseConstantTile = [&](ArrayRef<float> rawData,
                                     ArrayRef<int64_t> sourceShape,
                                     ArrayRef<int64_t> loadTileShape) -> Value {
      SmallVector<int64_t> paddedSourceShape(sourceShape);
      if (paddedSourceShape.empty())
        paddedSourceShape.push_back(1);

      SmallVector<int64_t> paddedTileShape(loadTileShape);
      if (paddedTileShape.empty())
        paddedTileShape.push_back(1);

      int64_t tileElemCount = 1;
      for (int64_t dim : paddedTileShape)
        tileElemCount *= dim;
      SmallVector<float> paddedData(tileElemCount, 0.0f);

      int64_t sourceElemCount = 1;
      for (int64_t dim : paddedSourceShape)
        sourceElemCount *= dim;
      for (int64_t flat = 0;
           flat < sourceElemCount && flat < static_cast<int64_t>(rawData.size());
           ++flat) {
        int64_t rem = flat;
        SmallVector<int64_t> coords(paddedSourceShape.size(), 0);
        for (int64_t d = paddedSourceShape.size() - 1; d >= 0; --d) {
          coords[d] = rem % paddedSourceShape[d];
          rem /= paddedSourceShape[d];
        }
        int64_t paddedIdx = 0;
        int64_t stride = 1;
        for (int64_t d = paddedTileShape.size() - 1; d >= 0; --d) {
          paddedIdx += coords[d] * stride;
          stride *= paddedTileShape[d];
        }
        if (paddedIdx < tileElemCount)
          paddedData[paddedIdx] = rawData[flat];
      }

      auto tileType = ::mlir::cuda_tile::TileType::get(ctx, paddedTileShape, elemType);
      auto tensorType = RankedTensorType::get(paddedTileShape, elemType);
      auto attr =
          DenseElementsAttr::get(tensorType,
                                 ArrayRef<float>(paddedData.data(),
                                                 paddedData.size()));
      return e.builder()
          .create<::mlir::cuda_tile::ConstantOp>(
              e.builder().getUnknownLoc(), tileType,
              cast<DenseTypedElementsAttr>(
                  attr.reshape(cast<ShapedType>(tileType))))
          .getResult();
    };

    auto computeBroadcastLoadPlan =
        [&](ArrayRef<int64_t> sourceShape, ArrayRef<int64_t> fullShape,
            ArrayRef<int64_t> targetTileShape, ArrayRef<Value> fullTileIndices,
            SmallVector<int64_t> &loadTileShape, SmallVector<Value> &loadIndices,
            SmallVector<int64_t> &reshapeShape) -> bool {
      loadTileShape.clear();
      loadIndices.clear();
      reshapeShape.clear();

      if (sourceShape.empty()) {
        loadTileShape.push_back(1);
        reshapeShape.assign(fullShape.size(), 1);
        return true;
      }

      if (sourceShape == fullShape) {
        loadTileShape.assign(targetTileShape.begin(), targetTileShape.end());
        loadIndices.assign(fullTileIndices.begin(), fullTileIndices.end());
        reshapeShape.assign(loadTileShape.begin(), loadTileShape.end());
        return true;
      }

      if (sourceShape.size() == fullShape.size()) {
        for (int64_t d = 0; d < static_cast<int64_t>(fullShape.size()); ++d) {
          if (sourceShape[d] == fullShape[d]) {
            loadTileShape.push_back(targetTileShape[d]);
            loadIndices.push_back(fullTileIndices[d]);
          } else if (sourceShape[d] == 1) {
            loadTileShape.push_back(1);
            loadIndices.push_back(e.constI32(0));
          } else {
            return false;
          }
        }
        reshapeShape.assign(loadTileShape.begin(), loadTileShape.end());
        return true;
      }

      if (sourceShape.size() == 1 && fullShape.size() == 2) {
        if (sourceShape[0] == fullShape[0]) {
          loadTileShape.push_back(targetTileShape[0]);
          loadIndices.push_back(fullTileIndices[0]);
          reshapeShape = {targetTileShape[0], 1};
          return true;
        }
        if (sourceShape[0] == fullShape[1]) {
          loadTileShape.push_back(targetTileShape[1]);
          loadIndices.push_back(fullTileIndices[1]);
          reshapeShape = {1, targetTileShape[1]};
          return true;
        }
      }

      return false;
    };

    auto emitTensorTileForSpace = [&](Value value, ArrayRef<int64_t> fullShape,
                                      ArrayRef<int64_t> targetTileShape,
                                      ArrayRef<Value> fullTileIndices) -> Value {
      int64_t bindingIndex = -1;
      auto bindingIt = valueToBind.find(value);
      if (bindingIt != valueToBind.end()) {
        bindingIndex = bindingIt->second;
      } else if (const CudaTileBindingPlan *binding =
                     findBindingPlanForMemref(bindingShapes, value)) {
        bindingIndex = binding->binding;
      }

      SmallVector<int64_t> loadTileShape;
      SmallVector<Value> loadIndices;
      SmallVector<int64_t> reshapeShape;
      if (bindingIndex >= 0) {
        const CudaTileBindingPlan *binding =
            findBindingPlan(bindingShapes, bindingIndex);
        if (!binding)
          return {};
        SmallVector<int64_t> sourceShape =
            binding->shape.empty() ? SmallVector<int64_t>(fullShape)
                                   : SmallVector<int64_t>(binding->shape);
        if (!computeBroadcastLoadPlan(sourceShape, fullShape, targetTileShape,
                                      fullTileIndices, loadTileShape,
                                      loadIndices, reshapeShape)) {
          return {};
        }
        auto strides = computeRowMajorStrides(sourceShape);
        Value ptr = getSubspanArg(e, *binding, elemType);
        auto view = e.makeTensorView(ptr, sourceShape, strides, elemType);
        auto partition = e.makePartitionView(view, loadTileShape);
        auto [tile, token] =
            e.loadViewTko(partition, loadIndices, loadTileShape, elemType);
        if (reshapeShape.size() != loadTileShape.size() ||
            !llvm::equal(reshapeShape, loadTileShape)) {
          tile = e.reshape(tile, reshapeShape, elemType);
        }
        if (!llvm::equal(reshapeShape, targetTileShape))
          tile = e.broadcastTile(tile, targetTileShape, elemType);
        return tile;
      }

      SmallVector<float> rawData;
      SmallVector<int64_t> sourceShape;
      if (!getDenseFloatConstant(value, rawData, sourceShape))
        return {};
      if (!computeBroadcastLoadPlan(sourceShape, fullShape, targetTileShape,
                                    fullTileIndices, loadTileShape, loadIndices,
                                    reshapeShape)) {
        return {};
      }
      Value tile = emitDenseConstantTile(rawData, sourceShape, loadTileShape);
      if (reshapeShape.size() != loadTileShape.size() ||
          !llvm::equal(reshapeShape, loadTileShape)) {
        tile = e.reshape(tile, reshapeShape, elemType);
      }
      if (!llvm::equal(reshapeShape, targetTileShape))
        tile = e.broadcastTile(tile, targetTileShape, elemType);
      return tile;
    };

    auto emitDirectMatmulPhysicalOperandTile = [&](int64_t operandIndex) -> Value {
      if (operandIndex == 0) {
        auto [tile, token] = e.loadViewTko(pA, {by, iv}, tA, elemType);
        return tile;
      }
      if (weightConstant) {
        auto cstOp = weightConstant.getDefiningOp<arith::ConstantOp>();
        if (!cstOp)
          return {};

        SmallVector<float> rawData;
        auto attrVal = cstOp.getValue();
        if (auto dense = dyn_cast<DenseElementsAttr>(attrVal)) {
          for (auto val : dense.getValues<float>())
            rawData.push_back(val);
        } else if (auto resAttr = dyn_cast<DenseResourceElementsAttr>(attrVal)) {
          auto blob = resAttr.getRawHandle().getBlob();
          if (blob) {
            auto data = blob->getData();
            auto *floats = reinterpret_cast<const float *>(data.data());
            int64_t numElem = data.size() / sizeof(float);
            rawData.assign(floats, floats + numElem);
          }
        }
        if (rawData.empty())
          return {};

        int64_t physRows = bTransposed ? N : K;
        int64_t physCols = bTransposed ? K : N;
        int64_t tileRows = bTransposed ? aTN : aTK;
        int64_t tileCols = bTransposed ? aTK : aTN;
        SmallVector<float> paddedData(tileRows * tileCols, 0.0f);
        for (int64_t r = 0; r < physRows && r < tileRows; ++r) {
          for (int64_t c = 0; c < physCols && c < tileCols; ++c) {
            int64_t srcIdx = r * physCols + c;
            if (srcIdx < static_cast<int64_t>(rawData.size()))
              paddedData[r * tileCols + c] = rawData[srcIdx];
          }
        }

        SmallVector<int64_t> tileBShape = {tileRows, tileCols};
        auto tileType = ::mlir::cuda_tile::TileType::get(ctx, tileBShape, elemType);
        auto tensorType = RankedTensorType::get(tileBShape, elemType);
        auto attr =
            DenseElementsAttr::get(tensorType,
                                   ArrayRef<float>(paddedData.data(),
                                                   paddedData.size()));
        return e.builder()
            .create<::mlir::cuda_tile::ConstantOp>(
                e.builder().getUnknownLoc(), tileType,
                cast<DenseTypedElementsAttr>(
                    attr.reshape(cast<ShapedType>(tileType))))
            .getResult();
      }
      auto [tile, token] =
          bTransposed ? e.loadViewTko(pBv, {bx, iv}, tB, elemType)
                      : e.loadViewTko(pBv, {iv, bx}, tB, elemType);
      return tile;
    };

    DenseMap<Value, Value> lhsPhysicalTileCache;
    DenseMap<Value, Value> rhsPhysicalTileCache;
    auto emitPhysicalOperandTile = [&](auto &&self, Value value,
                                       int64_t operandIndex) -> Value {
      auto &cache =
          operandIndex == 0 ? lhsPhysicalTileCache : rhsPhysicalTileCache;
      auto cacheIt = cache.find(value);
      if (cacheIt != cache.end())
        return cacheIt->second;

      ArrayRef<int64_t> fullShape = operandIndex == 0 ? ArrayRef<int64_t>(shA2)
                                                      : ArrayRef<int64_t>(shB2);
      ArrayRef<int64_t> targetTileShape =
          operandIndex == 0 ? ArrayRef<int64_t>(tA) : ArrayRef<int64_t>(tB);
      SmallVector<Value> tileIndices =
          operandIndex == 0 ? SmallVector<Value>{by, iv}
                            : (bTransposed ? SmallVector<Value>{bx, iv}
                                           : SmallVector<Value>{iv, bx});

      if (auto genericOp = value.getDefiningOp<linalg::GenericOp>()) {
        if (const CudaTileFusedOpPlan *fusedOp =
                findFusedOpPlanFor(genericOp.getOperation());
            fusedOp && fusedOp->role == CudaTileFusedOpRole::Prologue &&
            (fusedOp->primaryInputIndex < 0 ||
             fusedOp->primaryInputIndex == operandIndex)) {
          SmallVector<Value> inputTiles;
          for (Value input : genericOp.getDpsInputs()) {
            Value tile = self(self, input, operandIndex);
            if (!tile) {
              tile = emitTensorTileForSpace(input, fullShape, targetTileShape,
                                            tileIndices);
            }
            if (!tile)
              return {};
            inputTiles.push_back(tile);
          }
          Value fallback = inputTiles.empty()
                               ? e.constSplat(targetTileShape, elemType, 0.0)
                               : inputTiles.front();
          Value tile = emitElementwiseGenericBody(
              e, genericOp, inputTiles, targetTileShape, elemType, fallback,
              tileIndices, fullShape, gatherSources);
          cache[value] = tile;
          return tile;
        }
      }

      if (operandIndex < static_cast<int64_t>(primaryInputs.size()) &&
          value == primaryInputs[operandIndex]) {
        Value tile = emitDirectMatmulPhysicalOperandTile(operandIndex);
        cache[value] = tile;
        return tile;
      }

      Value tile =
          emitTensorTileForSpace(value, fullShape, targetTileShape, tileIndices);
      if (tile)
        cache[value] = tile;
      return tile;
    };

    auto finalizeMatmulOperandTile = [&](Value tile, int64_t operandIndex) {
      if (!tile || operandIndex != 1 || !bTransposed)
        return tile;
      auto permType = ::mlir::cuda_tile::TileType::get(ctx, {aTK, aTN}, elemType);
      return e.permute(tile, {1, 0}, permType);
    };

    Value tAd = primaryInputs.empty()
                    ? emitDirectMatmulPhysicalOperandTile(0)
                    : emitPhysicalOperandTile(emitPhysicalOperandTile,
                                              primaryInputs.front(), 0);
    Value tBd =
        primaryInputs.size() < 2
            ? emitDirectMatmulPhysicalOperandTile(1)
            : emitPhysicalOperandTile(emitPhysicalOperandTile,
                                      primaryInputs[1], 1);
    if (!tAd) {
      primaryOp->emitError("failed to materialize matmul lhs tile");
      return failure();
    }
    if (!tBd) {
      primaryOp->emitError("failed to materialize matmul rhs tile");
      return failure();
    }
    tBd = finalizeMatmulOperandTile(tBd, 1);
    // `plan.schedule.mmaKind` records IREEGPU MMA intent. The current
    // cuda_tile emitter still lowers through the generic floating MMA op until
    // we have a legal mapping from IREEGPU intrinsic attrs to cuda_tile/tileiras
    // instruction forms.
    auto newAcc = e.mmaf(tAd, tBd, iterAcc);
    e.endFor(forOp, ValueRange{newAcc});

    // The for loop result is the final accumulator.
    Value finalAcc = forOp.getResult(0);
    LLVM_DEBUG(llvm::dbgs() << "[cuda_tile]   matmul kernel built OK\n");

    // Post-matmul elementwise fusion: apply any elementwise generics that
    // follow the matmul in the dispatch (e.g., bias_add + relu).
    {
      DenseMap<Value, Value> fusedValueMap;
      for (Value result : primaryOp->getResults())
        fusedValueMap[result] = finalAcc;

      SmallVector<Value> outputTileIndices = {by, bx};
      for (const CudaTileFusedOpPlan &fusedOp : plan.fusedOps) {
        if (fusedOp.role != CudaTileFusedOpRole::Epilogue)
          continue;
        auto postOp = dyn_cast<linalg::GenericOp>(fusedOp.op);
        if (!postOp || postOp.getNumReductionLoops() != 0 ||
            postOp.getNumParallelLoops() == 0) {
          continue;
        }

        SmallVector<Value> inputTiles;
        for (Value input : postOp.getDpsInputs()) {
          auto valueIt = fusedValueMap.find(input);
          if (valueIt != fusedValueMap.end()) {
            inputTiles.push_back(valueIt->second);
            continue;
          }
          Value tile =
              emitTensorTileForSpace(input, shC2, tC, outputTileIndices);
          if (!tile)
            return failure();
          inputTiles.push_back(tile);
        }

        Value fallback = inputTiles.empty() ? finalAcc : inputTiles.front();
        finalAcc = emitElementwiseGenericBody(
            e, postOp, inputTiles, tC, elemType, fallback, outputTileIndices,
            shC2, gatherSources);
        for (Value result : postOp.getResults())
          fusedValueMap[result] = finalAcc;
      }
    }

    e.storeViewTko(finalAcc, pC, {by, bx});
    e.emitReturn();
    e.endEntry();

    return serializeCudaTileKernel(
        std::move(e),
        SmallVector<int64_t, 3>{(N + aTN - 1) / aTN, (M + aTM - 1) / aTM, 1});
  }

  //=== Phase 2: Elementwise (default fallback) ===//

  {
    CudaTileOpEmitter e(ctx);
    auto shape = dstShape;
    auto tileShape = computeTileShape(shape, tileM, tileN);
    auto strides = computeRowMajorStrides(shape);
    auto gridDims = computeGridDims(shape, tileShape);
    int64_t rank = shape.size();

    int64_t numSubspans = bindingShapes.size();
    int64_t numBindings = getNumBindingArgs(bindingShapes);
    if (numBindings == 0)
      numBindings = primaryOp->getNumOperands() + primaryOp->getNumResults();
    if (numBindings < 2)
      numBindings = 2;
    if (numSubspans == 0)
      numSubspans = numBindings;

    // Check if any input binding has a different shape (broadcast).
    bool hasBroadcast = numSubspans != numBindings;
    for (int64_t i = 0; i < (int64_t)bindingShapes.size() - 1; ++i) {
      if (!bindingShapes[i].shape.empty() &&
          bindingShapes[i].shape != SmallVector<int64_t>(shape))
        hasBroadcast = true;
    }

    if (!hasBroadcast) {
      // No broadcast: use standard boilerplate.
      auto bp = emitKernelBoilerplate(e, kernelName, numBindings, shape,
                                      strides, tileShape, elemType,
                                      bindingShapes);

      auto [tile, tok] =
          e.loadViewTko(bp.partViews[0], bp.indices, tileShape, elemType);

      Value current = tile;
      if (loweringStrategy == CudaTileLoweringStrategy::Elementwise) {
        auto genericOp = dyn_cast<linalg::GenericOp>(primaryOp);
        if (genericOp) {
          SmallVector<Value> inputTiles;
          int64_t inputCount =
              std::min<int64_t>(genericOp.getNumDpsInputs(),
                                std::max<int64_t>(numBindings - 1, 0));
          inputTiles.push_back(current);
          for (int64_t i = 1; i < inputCount; ++i) {
            auto [nextTile, nextTok] =
                e.loadViewTko(bp.partViews[i], bp.indices, tileShape, elemType);
            inputTiles.push_back(nextTile);
          }
          current = emitElementwiseGenericBody(e, genericOp, inputTiles,
                                               tileShape, elemType, current,
                                               bp.indices, shape,
                                               gatherSources);
        } else {
          SmallVector<StringRef> ops = getElementwiseOpNames(primaryOp);
          Value tile2;
          if (numBindings >= 3) {
            auto [t2, tok2] =
                e.loadViewTko(bp.partViews[1], bp.indices, tileShape, elemType);
            tile2 = t2;
          }
          for (auto opName : ops) {
            if (tile2 && opName == ops.front() && numBindings >= 3)
              current = e.emitElementwise(opName, {current, tile2});
            else
              current = e.emitElementwise(opName, {current});
          }
        }
      }

      e.storeViewTko(current, bp.partViews.back(), bp.indices);
      e.emitReturn();
      e.endEntry();

      return serializeCudaTileKernel(std::move(e), gridDims);
    }

    // Broadcast case: create per-binding views with correct shapes.
    e.beginModule(kernelName);
    e.beginEntry("main", numBindings, elemType);

    SmallVector<Value> partViews;
    SmallVector<SmallVector<int64_t>> bindTileShapes;
    for (int64_t i = 0; i < numSubspans; ++i) {
      auto bShape =
          (i < (int64_t)bindingShapes.size() && !bindingShapes[i].shape.empty())
              ? SmallVector<int64_t>(bindingShapes[i].shape)
              : SmallVector<int64_t>(shape);
      auto bStrides = computeRowMajorStrides(bShape);
      auto bTile = computeTileShape(bShape, tileM, tileN);
      Value ptr = i < (int64_t)bindingShapes.size()
                      ? getSubspanArg(e, bindingShapes[i], elemType)
                      : e.getArg(i);
      auto tv = e.makeTensorView(ptr, bShape, bStrides, elemType);
      auto pv = e.makePartitionView(tv, bTile);
      partViews.push_back(pv);
      bindTileShapes.push_back(bTile);
    }

    auto [bx, by, bz] = e.getTileBlockId();
    SmallVector<Value> outIndices(rank);
    if (rank >= 1)
      outIndices[rank - 1] = bx;
    if (rank >= 2)
      outIndices[rank - 2] = by;
    if (rank >= 3)
      outIndices[0] = bz;

    auto loadInputTile = [&](int64_t bindIdx) -> Value {
      auto bShape =
          (bindIdx < (int64_t)bindingShapes.size() &&
           !bindingShapes[bindIdx].shape.empty())
              ? SmallVector<int64_t>(bindingShapes[bindIdx].shape)
              : SmallVector<int64_t>(shape);
      int64_t bRank = bShape.size();
      if (bRank == rank) {
        auto [tile, tok] =
            e.loadViewTko(partViews[bindIdx], outIndices, tileShape, elemType);
        return tile;
      }

      SmallVector<Value> bIndices;
      SmallVector<int64_t> reshapeShape, broadcastShape;
      int64_t bDim = 0;
      for (int64_t d = 0; d < rank; ++d) {
        if (bDim < bRank && bShape[bDim] == shape[d]) {
          bIndices.push_back(outIndices[d]);
          reshapeShape.push_back(bindTileShapes[bindIdx][bDim]);
          bDim++;
        } else {
          reshapeShape.push_back(1);
        }
        broadcastShape.push_back(tileShape[d]);
      }

      auto [tile, tok] =
          e.loadViewTko(partViews[bindIdx], bIndices, bindTileShapes[bindIdx],
                        elemType);
      tile = e.reshape(tile, reshapeShape, elemType);
      if (reshapeShape != broadcastShape)
        tile = e.broadcastTile(tile, broadcastShape, elemType);
      return tile;
    };

    Value current;
    if (loweringStrategy == CudaTileLoweringStrategy::Elementwise) {
      auto genericOp = dyn_cast<linalg::GenericOp>(primaryOp);
      if (genericOp) {
        SmallVector<Value> inputTiles;
        int64_t inputCount =
            std::min<int64_t>(genericOp.getNumDpsInputs(),
                              std::max<int64_t>(numSubspans - 1, 0));
        for (int64_t i = 0; i < inputCount; ++i)
          inputTiles.push_back(loadInputTile(i));
        current = inputTiles.empty()
                      ? e.constSplat(tileShape, elemType, 0.0)
                      : inputTiles.front();
        current = emitElementwiseGenericBody(e, genericOp, inputTiles,
                                             tileShape, elemType, current,
                                             outIndices, shape,
                                             gatherSources);
      } else {
        current = loadInputTile(0);
        SmallVector<StringRef> ops = getElementwiseOpNames(primaryOp);
        Value tile2;
        if (numSubspans >= 3)
          tile2 = loadInputTile(1);
        for (auto opName : ops) {
          if (tile2 && opName == ops.front() && numSubspans >= 3)
            current = e.emitElementwise(opName, {current, tile2});
          else
            current = e.emitElementwise(opName, {current});
        }
      }
    }
    if (!current)
      current = loadInputTile(0);

    e.storeViewTko(current, partViews.back(), outIndices);
    e.emitReturn();
    e.endEntry();

    return serializeCudaTileKernel(std::move(e), gridDims);
  }
}

//===----------------------------------------------------------------------===//
// CudaTileTargetDevice
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CudaTileCodegenPass — runs in the translation pipeline
//===----------------------------------------------------------------------===//

/// Pass that runs on hal.executable.variant.
/// It performs annotation + kernel build + tilebc/cubin compilation,
/// stores the cubin as a string attribute on the variant op,
/// then strips the inner module body so IREE's later passes don't crash.
struct CudaTileCodegenPass
    : public PassWrapper<CudaTileCodegenPass,
                         OperationPass<IREE::HAL::ExecutableVariantOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CudaTileCodegenPass)

  CudaTileCodegenPass() = default;
  explicit CudaTileCodegenPass(const CudaTileOptions &opts) : options(opts) {}

  StringRef getArgument() const override { return "cuda-tile-codegen"; }
  StringRef getDescription() const override {
    return "Lower inner module to cuda_tile bytecode and compile to cubin";
  }

  void runOnOperation() override {
    auto variantOp = getOperation();
    if (variantOp.isExternal())
      return; // external variants don't need codegen
    auto innerModule = variantOp.getInnerModule();
    if (!innerModule)
      return;
    auto *ctx = variantOp->getContext();
    auto libraryName =
        variantOp->getParentOfType<IREE::HAL::ExecutableOp>().getName().str();

    // Decompose linalg.softmax into standard reduce + elementwise ops.
    // The decomposed form (max-reduce, sub, exp, sum-reduce, div) is
    // numerically stable.
    // NOTE: softmax may not reach here — IREE's dispatch formation may
    // crash on linalg.softmax before our pass runs (upstream bug).
    // The preprocessing decompose handles it when possible.
    {
      bool hasSoftmax = false;
      innerModule->walk([&](Operation *op) {
        if (op->getName().getStringRef() == "linalg.softmax")
          hasSoftmax = true;
      });
      if (hasSoftmax) {
        mlir::PassManager decomposePM(ctx);
        decomposePM.addNestedPass<func::FuncOp>(
            createDecomposeSoftmaxPass(/*useFusion=*/false));
        if (failed(decomposePM.run(innerModule))) {
          innerModule->emitError("softmax decomposition failed");
          return signalPassFailure();
        }
      }
    }

    // NOTE: im2col pass runs in preprocessing only (not here).
    // At preprocessing time, named conv ops exist and im2col can create separate
    // dispatches. At translation time, the conv is already inside a dispatch and
    // im2col would create an intermediate tensor that isn't backed by a binding.

    // Run annotation passes to tag ops with cuda_tile metadata.
    {
      CudaTile::CudaTileTransformOptions transformOptions;
      transformOptions.tileM = options.tileM;
      transformOptions.tileN = options.tileN;
      transformOptions.tileK = options.tileK;

      mlir::PassManager pm(ctx);
      pm.addNestedPass<func::FuncOp>(
          CudaTile::createConvertDataMovementToCudaTilePass(transformOptions));
      pm.addNestedPass<func::FuncOp>(
          CudaTile::createConvertElementwiseToCudaTilePass(transformOptions));
      pm.addNestedPass<func::FuncOp>(
          CudaTile::createConvertReductionsToCudaTilePass(transformOptions));
      pm.addNestedPass<func::FuncOp>(
          CudaTile::createConvertContractionsToCudaTilePass(transformOptions));
      pm.addNestedPass<func::FuncOp>(
          CudaTile::createConvertSCFToCudaTilePass(transformOptions));
      if (failed(pm.run(innerModule))) {
        innerModule->emitError("cuda_tile conversion passes failed");
        return signalPassFailure();
      }
    }

    // Build cuda_tile module in-process → tilebc.
    auto kernelName = sanitizeSymbolName(libraryName);
    LLVM_DEBUG(llvm::dbgs()
               << "[cuda_tile] Building kernel: " << kernelName << "\n");
    auto maybeTilebc =
        buildCudaTileKernel(ctx, innerModule, kernelName, options);
    if (failed(maybeTilebc)) {
      innerModule->emitError("failed to build cuda_tile kernel");
      return signalPassFailure();
    }
    auto &[tilebcData, gridDims] = maybeTilebc.value();

    // Step 3: tilebc → cubin via tileiras.
    std::string message;
    FailureOr<std::string> tileirasCompiler =
        findTileirasCompiler(options, &message);
    if (failed(tileirasCompiler)) {
      innerModule->emitError("failed to find tileiras: ") << message;
      return signalPassFailure();
    }
    FailureOr<std::string> maybeCubin = compileWithTileiras(
        tileirasCompiler.value(), options.smArch, options.tileirasParams,
        tilebcData, &message);
    if (failed(maybeCubin)) {
      innerModule->emitError("tileiras compilation failed: ") << message;
      return signalPassFailure();
    }

    // Step 4: Store cubin + grid dims as attributes on the variant op.
    auto cubinAttr = StringAttr::get(
        ctx, StringRef(maybeCubin.value().data(), maybeCubin.value().size()));
    variantOp->setAttr("cuda_tile.cubin_data", cubinAttr);
    variantOp->setAttr("cuda_tile.grid_dims",
                       Builder(ctx).getDenseI64ArrayAttr(gridDims));

    // Step 4b: Update the export op's workgroup_count body to return
    // our computed grid dims. This is what the runtime uses for cuLaunchKernel.
    for (auto exportOp : variantOp.getExportOps()) {
      if (Block *countBody = exportOp.getWorkgroupCountBody()) {
        // Clear ops but keep the existing block arguments (device + workload).
        countBody->clear();
        OpBuilder countBuilder(ctx);
        countBuilder.setInsertionPointToEnd(countBody);
        auto countLoc = countBuilder.getUnknownLoc();
        Value gx = countBuilder.create<arith::ConstantIndexOp>(
            countLoc, gridDims.size() > 0 ? gridDims[0] : 1);
        Value gy = countBuilder.create<arith::ConstantIndexOp>(
            countLoc, gridDims.size() > 1 ? gridDims[1] : 1);
        Value gz = countBuilder.create<arith::ConstantIndexOp>(
            countLoc, gridDims.size() > 2 ? gridDims[2] : 1);
        countBuilder.create<IREE::HAL::ReturnOp>(
            countLoc, ValueRange{gx, gy, gz});
      }
    }

    // Step 5: Strip the inner module body so IREE's later passes don't
    // crash on unlowered ops. The cubin is safely stored as an attribute.
    // Turn all functions into external declarations (no body) so IREE's
    // type converter can process them without hitting unlowered ops.
    innerModule->walk([](func::FuncOp funcOp) {
      if (!funcOp.isExternal()) {
        funcOp.getBody().getBlocks().clear();
        funcOp.setVisibility(SymbolTable::Visibility::Private);
      }
    });
  }

  CudaTileOptions options;
};

//===----------------------------------------------------------------------===//
// CudaTileTargetDevice
//===----------------------------------------------------------------------===//

class CudaTileTargetDevice final : public TargetDevice {
public:
  CudaTileTargetDevice(const CudaTileOptions &options) : options(options) {}

  IREE::HAL::DeviceTargetAttr
  getDefaultDeviceTarget(MLIRContext *context,
                         const TargetRegistry &targetRegistry) const override {
    Builder b(context);
    auto deviceConfigAttr = b.getDictionaryAttr({});

    SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
    targetRegistry.getTargetBackend("cuda_tile")
        ->getDefaultExecutableTargets(context, "cuda_tile", deviceConfigAttr,
                                      executableTargetAttrs);

    return IREE::HAL::DeviceTargetAttr::get(context,
                                            b.getStringAttr("cuda_tile"),
                                            deviceConfigAttr,
                                            executableTargetAttrs);
  }

private:
  const CudaTileOptions &options;
};

//===----------------------------------------------------------------------===//
// CudaTileTargetBackend
//===----------------------------------------------------------------------===//

class CudaTileTargetBackend final : public TargetBackend {
public:
  CudaTileTargetBackend(const CudaTileOptions &options) : options(options) {}

  std::string getLegacyDefaultDeviceID() const override { return "cuda_tile"; }

  void getDefaultExecutableTargets(
      MLIRContext *context, StringRef deviceID,
      DictionaryAttr deviceConfigAttr,
      SmallVectorImpl<IREE::HAL::ExecutableTargetAttr>
          &executableTargetAttrs) const override {
    executableTargetAttrs.push_back(getExecutableTarget(context));
  }

  IREE::HAL::ExecutableTargetAttr
  getExecutableTarget(MLIRContext *context) const {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;
    if (failed(options.verify(b))) {
      return nullptr;
    }

    configItems.emplace_back(b.getStringAttr("sm_arch"),
                             b.getStringAttr(options.smArch));

    return b.getAttr<IREE::HAL::ExecutableTargetAttr>(
        b.getStringAttr("cuda_tile"), b.getStringAttr("cuda-tile-fb"),
        b.getDictionaryAttr(configItems));
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    // Register the cuda_tile dialect for in-process compilation.
    registry.insert<::mlir::cuda_tile::CudaTileDialect>();
  }

  void buildConfigurationPassPipeline(
      IREE::HAL::ExecutableTargetAttr targetAttr,
      OpPassManager &passManager) override {
    // No cuda_tile-specific configuration passes needed yet.
  }

  void buildTranslationPassPipeline(
      IREE::HAL::ExecutableTargetAttr targetAttr,
      OpPassManager &passManager) override {
    // IREE flow: translate pipeline → serialize.
    // Translate runs first, so we do all codegen here (annotation + kernel
    // build + tilebc serialization), store the result as an attribute on
    // the variant, then strip the inner module body so IREE's later passes
    // (VM conversion) don't crash on unlowered ops.
    // serializeExecutable() reads the stored tilebc attribute.
    // Resolve workgroup count hints (iree_tensor_ext.dispatch.workgroup_count_from_slice).
    // Without this, VM conversion crashes on unresolved ops in the host module.
    passManager.addPass(createResolveWorkgroupCountHintsPass());

    if (options.enableCodegen) {
      passManager.addPass(
          std::make_unique<CudaTileCodegenPass>(options));
    }
  }

  LogicalResult serializeExecutable(const SerializationOptions &serOptions,
                                    IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override {
    auto libraryName =
        variantOp->getParentOfType<IREE::HAL::ExecutableOp>().getName().str();

    // Collect export operations.
    auto exportOps = llvm::to_vector_of<IREE::HAL::ExecutableExportOp>(
        variantOp.getExportOps());

    std::string cubinData;
    if (variantOp.isExternal()) {
      //=== External path: load pre-compiled cubin or tilebc ===//
      if (!variantOp.getObjects().has_value()) {
        return variantOp.emitOpError()
               << "no objects defined for external variant";
      }
      if (variantOp.getObjects()->getValue().size() != 1) {
        return variantOp.emitOpError()
               << "only one object reference is supported for "
                  "external cuda_tile variants";
      }

      auto objectAttr = cast<IREE::HAL::ExecutableObjectAttr>(
          variantOp.getObjects()->getValue().front());
      auto data = objectAttr.loadData();
      if (!data) {
        return variantOp.emitOpError()
               << "object file could not be loaded: " << objectAttr;
      }

      // Detect format: tilebc starts with magic 0x7F 'T' 'i' 'l' 'e' 'I' 'R'.
      StringRef dataRef = data.value();
      bool isTilebc = dataRef.size() >= 7 && dataRef[0] == 0x7F &&
                      dataRef[1] == 'T' && dataRef[2] == 'i' &&
                      dataRef[3] == 'l' && dataRef[4] == 'e' &&
                      dataRef[5] == 'I' && dataRef[6] == 'R';

      if (isTilebc) {
        // Compile tilebc -> cubin via tileiras.
        std::string message;
        FailureOr<std::string> tileirasCompiler =
            findTileirasCompiler(options, &message);
        if (failed(tileirasCompiler)) {
          return variantOp.emitError()
                 << "failed to find tileiras: " << message;
        }
        FailureOr<std::string> maybeCubin = compileWithTileiras(
            tileirasCompiler.value(), options.smArch, options.tileirasParams,
            dataRef, &message);
        if (failed(maybeCubin)) {
          return variantOp.emitError()
                 << "tileiras compilation failed: " << message;
        }
        cubinData = std::move(maybeCubin.value());
      } else {
        // Assume pre-compiled cubin.
        cubinData.assign(dataRef.begin(), dataRef.end());
      }
    } else if (options.enableCodegen) {
      //=== Codegen path: cubin was produced by CudaTileCodegenPass ===//
      // The translation pipeline (CudaTileCodegenPass) already ran:
      //   annotation → cuda_tile ops → tilebc → tileiras → cubin
      // and stored the cubin as a string attribute on the variant.
      auto cubinAttr =
          variantOp->getAttrOfType<StringAttr>("cuda_tile.cubin_data");
      if (!cubinAttr) {
        return variantOp.emitError()
               << "cuda_tile codegen enabled but no cubin data found; "
                  "CudaTileCodegenPass may not have run";
      }
      cubinData.assign(cubinAttr.getValue().data(),
                       cubinAttr.getValue().data() +
                           cubinAttr.getValue().size());
      // Clean up the cubin attribute (grid_dims read later in FlatBuffer build).
      variantOp->removeAttr("cuda_tile.cubin_data");
    } else {
      //=== Codegen disabled and no external objects ===//
      return variantOp.emitError()
             << "cuda_tile codegen path not enabled; use external "
                "objects via #hal.executable.object<...> or pass "
                "--iree-cuda-tile-enable-codegen=true";
    }

    if (!serOptions.dumpBinariesPath.empty()) {
      dumpDataToPath(serOptions.dumpBinariesPath, serOptions.dumpBaseName,
                     variantOp.getName(), ".cubin", cubinData);
    }

    //=== Build CTL1 FlatBuffer (cuda_tile native format) ===//
    FlatbufferBuilder builder;
    iree_hal_cuda_tile_ExecutableDef_start_as_root(builder);

    // Source files for debug info.
    auto sourceFilesRef = createSourceFilesVec(
        serOptions.debugLevel, variantOp.getSourcesAttr(), builder);

    // SM architecture string.
    auto smArchRef = builder.createString(options.smArch);

    // Module containing the cubin binary.
    SmallVector<iree_hal_cuda_tile_ModuleDef_ref_t> moduleRefs;
    {
      auto cubinImageRef = flatbuffers_uint8_vec_create(
          builder, reinterpret_cast<const uint8_t *>(cubinData.data()),
          cubinData.size());
      moduleRefs.push_back(
          iree_hal_cuda_tile_ModuleDef_create(builder, cubinImageRef));
    }
    auto modulesRef = builder.createOffsetVecDestructive(moduleRefs);

    // Per-export debug information.
    auto exportDebugInfos =
        createExportDefs(serOptions.debugLevel, exportOps, builder);

    // Build export definitions.
    SmallVector<iree_hal_cuda_tile_ExportDef_ref_t> exportRefs;
    exportRefs.resize(exportOps.size(), 0);
    for (auto exportOp : exportOps) {
      auto ordinalAttr = exportOp.getOrdinalAttr();
      if (!ordinalAttr) {
        return mlir::emitError(exportOp.getLoc())
               << "could not compile cuda_tile binary: export op is "
                  "missing ordinal";
      }
      int64_t ordinal = ordinalAttr.getInt();

      // The cuda_tile kernel entry is always named "main" (set by the
      // emitter's beginEntry). Use "main" here so cuModuleGetFunction finds it.
      auto kernelNameRef = builder.createString("main");

      // Use grid dims from codegen pass if available, else default {1,1,1}.
      iree_hal_cuda_tile_GridDims_t gridDims = {1, 1, 1};
      if (auto gridAttr =
              variantOp->getAttrOfType<DenseI64ArrayAttr>("cuda_tile.grid_dims")) {
        auto gd = gridAttr.asArrayRef();
        if (gd.size() >= 1) gridDims.x = static_cast<uint32_t>(gd[0]);
        if (gd.size() >= 2) gridDims.y = static_cast<uint32_t>(gd[1]);
        if (gd.size() >= 3) gridDims.z = static_cast<uint32_t>(gd[2]);
      }

      auto layoutAttr = exportOp.getLayoutAttr();
      uint32_t constantCount =
          static_cast<uint32_t>(layoutAttr.getConstants());
      uint32_t bindingCount =
          static_cast<uint32_t>(layoutAttr.getBindings().size());

      SmallVector<iree_hal_cuda_tile_BindingBits_enum_t> bindingFlags;
      for (auto bindingAttr : layoutAttr.getBindings()) {
        iree_hal_cuda_tile_BindingBits_enum_t flags = 0;
        if (allEnumBitsSet(bindingAttr.getFlags(),
                           IREE::HAL::DescriptorFlags::ReadOnly)) {
          flags |= iree_hal_cuda_tile_BindingBits_READ_ONLY;
        }
        if (allEnumBitsSet(bindingAttr.getFlags(),
                           IREE::HAL::DescriptorFlags::Indirect)) {
          flags |= iree_hal_cuda_tile_BindingBits_INDIRECT;
        }
        bindingFlags.push_back(flags);
      }
      auto bindingFlagsRef = iree_hal_cuda_tile_BindingBits_vec_create(
          builder, bindingFlags.data(), bindingFlags.size());

      iree_hal_cuda_tile_ExportDef_start(builder);
      iree_hal_cuda_tile_ExportDef_kernel_name_add(builder, kernelNameRef);
      iree_hal_cuda_tile_ExportDef_grid_dims_add(builder, &gridDims);
      iree_hal_cuda_tile_ExportDef_binding_count_add(builder, bindingCount);
      iree_hal_cuda_tile_ExportDef_constant_count_add(builder, constantCount);
      iree_hal_cuda_tile_ExportDef_binding_flags_add(builder, bindingFlagsRef);
      iree_hal_cuda_tile_ExportDef_debug_info_add(builder,
                                                  exportDebugInfos[ordinal]);
      exportRefs[ordinal] = iree_hal_cuda_tile_ExportDef_end(builder);
    }
    auto exportsRef = builder.createOffsetVecDestructive(exportRefs);

    iree_hal_cuda_tile_ExecutableDef_sm_arch_add(builder, smArchRef);
    iree_hal_cuda_tile_ExecutableDef_exports_add(builder, exportsRef);
    iree_hal_cuda_tile_ExecutableDef_modules_add(builder, modulesRef);
    iree_hal_cuda_tile_ExecutableDef_source_files_add(builder, sourceFilesRef);
    iree_hal_cuda_tile_ExecutableDef_end_as_root(builder);

    // Create the binary op in the executable.
    auto binaryOp = IREE::HAL::ExecutableBinaryOp::create(
        executableBuilder, variantOp.getLoc(), variantOp.getSymName(),
        variantOp.getTarget().getFormat(),
        builder.getHeaderPrefixedBufferAttr(
            executableBuilder.getContext(),
            /*magic=*/iree_hal_cuda_tile_ExecutableDef_file_identifier,
            /*version=*/0));
    binaryOp.setMimeTypeAttr(
        executableBuilder.getStringAttr("application/x-flatbuffers"));

    // Clean up temporary attributes.
    variantOp->removeAttr("cuda_tile.grid_dims");

    return success();
  }

private:
  const CudaTileOptions &options;
};

//===----------------------------------------------------------------------===//
// CudaTileSession — Plugin Registration
//===----------------------------------------------------------------------===//

struct CudaTileSession
    : public PluginSession<CudaTileSession, CudaTileOptions,
                           PluginActivationPolicy::DefaultActivated> {
  static void registerPasses() { CudaTile::registerCudaTilePasses(); }

  void populateHALTargetDevices(IREE::HAL::TargetDeviceList &targets) {
    targets.add("cuda_tile", [&]() {
      return std::make_shared<CudaTileTargetDevice>(options);
    });
  }
  void populateHALTargetBackends(IREE::HAL::TargetBackendList &targets) {
    targets.add("cuda_tile", [&]() {
      return std::make_shared<CudaTileTargetBackend>(options);
    });
  }

  /// Run annotation passes during preprocessing — before global optimization
  /// and before linalg generalization. Named ops (linalg.matmul, linalg.reduce,
  /// linalg.transpose) are still intact at this point.
  void extendPreprocessingPassPipeline(OpPassManager &passManager) override {
    if (!options.enableCodegen)
      return;

    // Decompose linalg.softmax BEFORE dispatch formation.
    // Without this, IREE's SoftmaxFusionOpInterfaceAdapter crashes in
    // FormDispatchRegions (upstream bug). The decomposed form
    // (max-reduce, sub, exp, sum-reduce, div) is numerically stable.
    passManager.addNestedPass<func::FuncOp>(
        createDecomposeSoftmaxPass(/*useFusion=*/false));

    // Convert convolution ops to im2col + matmul.
    passManager.addNestedPass<func::FuncOp>(
        IREE::LinalgExt::createConvertConvToIm2ColOpPass());

    CudaTile::CudaTileTransformOptions transformOptions;
    transformOptions.tileM = options.tileM;
    transformOptions.tileN = options.tileN;
    transformOptions.tileK = options.tileK;

    passManager.addNestedPass<func::FuncOp>(
        CudaTile::createConvertDataMovementToCudaTilePass(transformOptions));
    passManager.addNestedPass<func::FuncOp>(
        CudaTile::createConvertElementwiseToCudaTilePass(transformOptions));
    passManager.addNestedPass<func::FuncOp>(
        CudaTile::createConvertReductionsToCudaTilePass(transformOptions));
    passManager.addNestedPass<func::FuncOp>(
        CudaTile::createConvertContractionsToCudaTilePass(transformOptions));
    passManager.addNestedPass<func::FuncOp>(
        CudaTile::createConvertSCFToCudaTilePass(transformOptions));
  }
};

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL

extern "C" bool iree_register_compiler_plugin_hal_target_cuda_tile(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<
      mlir::iree_compiler::IREE::HAL::CudaTileSession>(
      "hal_target_cuda_tile");
  return true;
}
