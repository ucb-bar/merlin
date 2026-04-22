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
  SmallVector<int64_t> strides(shape.size(), 1);
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * shape[i + 1];
  return strides;
}

/// Read a DenseI64ArrayAttr from an op, or return empty.
static SmallVector<int64_t> getI64ArrayAttr(Operation *op, StringRef name) {
  if (auto attr = op->getAttrOfType<DenseI64ArrayAttr>(name))
    return SmallVector<int64_t>(attr.asArrayRef().begin(),
                                attr.asArrayRef().end());
  return {};
}

/// Compute tile shape for the trailing dims of a shape.
/// Round up to next power of 2 (or keep if already power of 2).
static int64_t nextPow2(int64_t v) {
  if (v <= 1) return 1;
  v--;
  v |= v >> 1; v |= v >> 2; v |= v >> 4;
  v |= v >> 8; v |= v >> 16; v |= v >> 32;
  return v + 1;
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

/// Extract combiner op name from a reduction body.
/// Duplicated from ConvertReductionsToCudaTile.cpp for codegen-time use.
static StringRef matchReduceCombinerLocal(Region &body) {
  if (body.empty() || body.front().empty())
    return "";
  Block &block = body.front();
  auto ops = block.without_terminator();
  auto it = ops.begin();
  if (it == ops.end())
    return "";
  Operation *combinerOp = &*it;
  ++it;
  if (it != ops.end())
    return "";
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

/// Map arith/math ops to cuda_tile op names (codegen-time body walking).
/// Duplicated from ConvertElementwiseToCudaTile.cpp.
static StringRef mapArithToCudaTileLocal(Operation *op) {
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
  if (isa<arith::SelectOp>(op)) return "select";
  return "";
}
static StringRef mapMathToCudaTileLocal(Operation *op) {
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
    StringRef name = mapArithToCudaTileLocal(&bodyOp);
    if (name.empty())
      name = mapMathToCudaTileLocal(&bodyOp);
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

static bool isIdentitySlice(ArrayRef<int64_t> offsets, ArrayRef<int64_t> strides) {
  return llvm::all_of(offsets, [](int64_t v) { return v == 0; }) &&
         llvm::all_of(strides, [](int64_t v) { return v == 1; });
}

//===----------------------------------------------------------------------===//
// CudaTileOpEmitter — builds cuda_tile dialect ops in-process
//===----------------------------------------------------------------------===//

/// Builds a cuda_tile::ModuleOp with ops via OpBuilder, then serializes to
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
    OperationState state(loc, cuda_tile::ModuleOp::getOperationName());
    state.addAttribute(SymbolTable::getSymbolAttrName(),
                       StringAttr::get(ctx, name));
    state.addRegion()->emplaceBlock();
    Operation *op = Operation::create(state);
    moduleOp = cast<cuda_tile::ModuleOp>(op);
    b.setInsertionPointToEnd(&moduleOp.getBody().front());
  }

  /// Create an entry function with numArgs pointer arguments.
  void beginEntry(StringRef name, int numArgs, Type elemType) {
    auto ptrType = cuda_tile::PointerType::get(ctx, elemType);
    auto tilePtrType = cuda_tile::TileType::get(ctx, {}, ptrType);
    SmallVector<Type> argTypes(numArgs, tilePtrType);
    auto funcType = FunctionType::get(ctx, argTypes, {});

    entryOp = b.create<cuda_tile::EntryOp>(
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
      b.create<cuda_tile::ReturnOp>(loc);
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
    auto type = cuda_tile::TileType::get(ctx, {}, b.getI32Type());
    auto attr = DenseElementsAttr::get(cast<ShapedType>(type),
                                       b.getI32IntegerAttr(val));
    return b.create<cuda_tile::ConstantOp>(loc, type,
                                           cast<DenseTypedElementsAttr>(attr))
        .getResult();
  }

  /// Create a splat tile constant: `constant dense<val> : tile<shape x elem>`.
  Value constSplat(ArrayRef<int64_t> shape, Type elemType, double val) {
    auto type = cuda_tile::TileType::get(ctx, shape, elemType);
    Attribute elemVal;
    if (isa<FloatType>(elemType))
      elemVal = FloatAttr::get(elemType, val);
    else
      elemVal = IntegerAttr::get(elemType, static_cast<int64_t>(val));
    auto attr = DenseElementsAttr::get(cast<ShapedType>(type), elemVal);
    return b.create<cuda_tile::ConstantOp>(loc, type,
                                           cast<DenseTypedElementsAttr>(attr))
        .getResult();
  }

  //===-- Data movement ---------------------------------------------------===//

  /// Create make_tensor_view with static shape and strides.
  Value makeTensorView(Value basePtr, ArrayRef<int64_t> shape,
                       ArrayRef<int64_t> strides, Type elemType) {
    auto tvType =
        cuda_tile::TensorViewType::get(ctx, elemType, shape, strides);
    return b
        .create<cuda_tile::MakeTensorViewOp>(loc, tvType, basePtr,
                                             /*dynamicShape=*/ValueRange{},
                                             /*dynamicStrides=*/ValueRange{})
        .getResult();
  }

  /// Create make_partition_view.
  Value makePartitionView(Value tensorView, ArrayRef<int64_t> tileShape) {
    auto tvType = cast<cuda_tile::TensorViewType>(tensorView.getType());
    SmallVector<int32_t> tileShapeI32(tileShape.begin(), tileShape.end());
    auto tileShapeAttr = DenseI32ArrayAttr::get(ctx, tileShapeI32);
    // dim_map: identity map — tile dim i maps to tensor_view dim i.
    // For a 2D tensor_view with 2D tile: [0, 1].
    SmallVector<int32_t> dimMap(tileShape.size());
    std::iota(dimMap.begin(), dimMap.end(), 0);
    auto pvType = cuda_tile::PartitionViewType::get(
        ctx, tileShapeAttr, tvType, dimMap,
        /*paddingValue=*/cuda_tile::PaddingValueAttr());
    return b.create<cuda_tile::MakePartitionViewOp>(loc, pvType, tensorView)
        .getResult();
  }

  /// Create get_tile_block_id → {x, y, z}.
  std::tuple<Value, Value, Value> getTileBlockId() {
    auto i32Tile = cuda_tile::TileType::get(ctx, {}, b.getI32Type());
    auto op =
        b.create<cuda_tile::GetTileBlockIdOp>(loc, i32Tile, i32Tile, i32Tile);
    return {op.getBlockIdX(), op.getBlockIdY(), op.getBlockIdZ()};
  }

  /// Create load_view_tko weak → {tile, token}.
  std::pair<Value, Value> loadViewTko(Value view, ValueRange indices,
                                      ArrayRef<int64_t> tileShape,
                                      Type elemType) {
    auto tileType = cuda_tile::TileType::get(ctx, tileShape, elemType);
    auto tokenType = cuda_tile::TokenType::get(ctx);
    auto op = b.create<cuda_tile::LoadViewTkoOp>(
        loc, tileType, tokenType,
        cuda_tile::MemoryOrderingSemantics::WEAK,
        /*memory_scope=*/cuda_tile::MemoryScopeAttr(), view, indices,
        /*token=*/Value(),
        /*optimization_hints=*/cuda_tile::OptimizationHintsAttr());
    return {op.getTile(), op.getResultToken()};
  }

  /// Create store_view_tko weak → token.
  Value storeViewTko(Value tile, Value view, ValueRange indices) {
    auto tokenType = cuda_tile::TokenType::get(ctx);
    auto op = b.create<cuda_tile::StoreViewTkoOp>(
        loc, tokenType,
        cuda_tile::MemoryOrderingSemantics::WEAK,
        /*memory_scope=*/cuda_tile::MemoryScopeAttr(), tile, view, indices,
        /*token=*/Value(),
        /*optimization_hints=*/cuda_tile::OptimizationHintsAttr());
    return op.getResultToken();
  }

  //===-- Arithmetic / Contractions ---------------------------------------===//

  /// Create mmaf (float matrix multiply-accumulate).
  Value mmaf(Value lhs, Value rhs, Value acc) {
    return b.create<cuda_tile::MmaFOp>(loc, lhs, rhs, acc).getResult();
  }

  /// Create permute op for tile dimension reordering.
  Value permute(Value source, ArrayRef<int32_t> permutation, Type resultType) {
    auto permAttr = DenseI32ArrayAttr::get(ctx, permutation);
    return b.create<cuda_tile::PermuteOp>(loc, resultType, source, permAttr)
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
    auto rnd = cuda_tile::RoundingModeAttr::get(
        ctx, cuda_tile::RoundingMode::NEAREST_EVEN);
    auto ovf = cuda_tile::IntegerOverflowAttr::get(
        ctx, cuda_tile::IntegerOverflow::NONE);

    auto kind = llvm::StringSwitch<EWOp>(opName)
        .Case("addf", EWOp::AddF).Case("subf", EWOp::SubF)
        .Case("mulf", EWOp::MulF).Case("divf", EWOp::DivF)
        .Case("maxf", EWOp::MaxF).Case("minf", EWOp::MinF)
        .Case("pow",  EWOp::Pow)
        .Case("addi", EWOp::AddI).Case("subi", EWOp::SubI)
        .Case("muli", EWOp::MulI)
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
    case EWOp::AddF: return b.create<cuda_tile::AddFOp>(loc, a, c, rnd).getResult();
    case EWOp::SubF: return b.create<cuda_tile::SubFOp>(loc, a, c, rnd).getResult();
    case EWOp::MulF: return b.create<cuda_tile::MulFOp>(loc, a, c, rnd).getResult();
    case EWOp::DivF: return b.create<cuda_tile::DivFOp>(loc, a, c, rnd).getResult();
    // Binary float without rounding
    case EWOp::MaxF: return b.create<cuda_tile::MaxFOp>(loc, a, c).getResult();
    case EWOp::MinF: return b.create<cuda_tile::MinFOp>(loc, a, c).getResult();
    case EWOp::Pow:  return b.create<cuda_tile::PowOp>(loc, a, c).getResult();
    // Binary integer
    case EWOp::AddI: return b.create<cuda_tile::AddIOp>(loc, a, c, ovf).getResult();
    case EWOp::SubI: return b.create<cuda_tile::SubIOp>(loc, a, c, ovf).getResult();
    case EWOp::MulI: return b.create<cuda_tile::MulIOp>(loc, a, c, ovf).getResult();
    case EWOp::MaxI: return b.create<cuda_tile::MaxIOp>(loc, a.getType(), ValueRange{a, c}, cuda_tile::Signedness::Signed).getResult();
    case EWOp::MinI: return b.create<cuda_tile::MinIOp>(loc, a.getType(), ValueRange{a, c}, cuda_tile::Signedness::Signed).getResult();
    // Ternary
    case EWOp::Select: return b.create<cuda_tile::SelectOp>(loc, a, c, d).getResult();
    case EWOp::Fma:    return b.create<cuda_tile::FmaOp>(loc, a, c, d, rnd).getResult();
    // Unary float (simple)
    case EWOp::NegF:  return b.create<cuda_tile::NegFOp>(loc, a).getResult();
    case EWOp::AbsF:  return b.create<cuda_tile::AbsFOp>(loc, a).getResult();
    case EWOp::Exp:   return b.create<cuda_tile::ExpOp>(loc, a).getResult();
    case EWOp::Exp2:  return b.create<cuda_tile::Exp2Op>(loc, a).getResult();
    case EWOp::Log:   return b.create<cuda_tile::LogOp>(loc, a).getResult();
    case EWOp::Log2:  return b.create<cuda_tile::Log2Op>(loc, a).getResult();
    case EWOp::Sin:   return b.create<cuda_tile::SinOp>(loc, a).getResult();
    case EWOp::Cos:   return b.create<cuda_tile::CosOp>(loc, a).getResult();
    case EWOp::Tanh:  return b.create<cuda_tile::TanHOp>(loc, a).getResult();
    case EWOp::Ceil:  return b.create<cuda_tile::CeilOp>(loc, a).getResult();
    case EWOp::Floor: return b.create<cuda_tile::FloorOp>(loc, a).getResult();
    // Unary float with rounding/attrs
    case EWOp::Sqrt:  return b.create<cuda_tile::SqrtOp>(loc, a, rnd).getResult();
    case EWOp::Rsqrt: return b.create<cuda_tile::RsqrtOp>(loc, a).getResult();
    // Unary integer
    case EWOp::NegI:  return b.create<cuda_tile::NegIOp>(loc, a).getResult();
    case EWOp::AbsI:  return b.create<cuda_tile::AbsIOp>(loc, a).getResult();
    // Fallback
    case EWOp::Unknown: return a;
    }
    // clang-format on
    llvm_unreachable("unhandled EWOp");
  }

  /// Emit a cuda_tile comparison (cmpf) from an arith::CmpFPredicate.
  Value emitCmpF(arith::CmpFPredicate pred, Value lhs, Value rhs) {
    using CP = cuda_tile::ComparisonPredicate;
    using CO = cuda_tile::ComparisonOrdering;
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
    auto predAttr = cuda_tile::ComparisonPredicateAttr::get(ctx, ctPred);
    auto orderAttr = cuda_tile::ComparisonOrderingAttr::get(ctx, ctOrder);
    return b.create<cuda_tile::CmpFOp>(loc, predAttr, orderAttr, lhs, rhs)
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
        cuda_tile::TileType::get(ctx, resultShape, elemType);

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

    auto reduceOp = b.create<cuda_tile::ReduceOp>(
        loc, TypeRange{resultTileType}, ValueRange{input}, dimAttr,
        identities);

    // Build the combiner body region.
    Block *body = b.createBlock(&reduceOp.getBody());
    // Region args: [current_elem, prev_accum] for each operand.
    auto scalarTileType = cuda_tile::TileType::get(ctx, {}, elemType);
    body->addArgument(scalarTileType, loc); // current
    body->addArgument(scalarTileType, loc); // accumulator

    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToEnd(body);

    Value cur = body->getArgument(0);
    Value acc = body->getArgument(1);
    Value result = emitElementwise(combiner, {cur, acc});
    b.create<cuda_tile::YieldOp>(loc, ValueRange{result});

    return reduceOp.getResult(0);
  }

  //===-- Shape manipulation -----------------------------------------------===//

  /// Reshape a tile to a new shape (same number of elements).
  /// E.g., tile<4xf32> → tile<4x1xf32>
  Value reshape(Value source, ArrayRef<int64_t> newShape, Type elemType) {
    auto resultType = cuda_tile::TileType::get(ctx, newShape, elemType);
    return b.create<cuda_tile::ReshapeOp>(loc, resultType, source).getResult();
  }

  /// Broadcast a tile: expand 1-dims to match the new shape.
  /// E.g., tile<4x1xf32> → tile<4x8xf32>
  Value broadcastTile(Value source, ArrayRef<int64_t> newShape, Type elemType) {
    auto resultType = cuda_tile::TileType::get(ctx, newShape, elemType);
    return b.create<cuda_tile::BroadcastOp>(loc, resultType, source)
        .getResult();
  }

  //===-- Pointer arithmetic -----------------------------------------------===//

  /// Offset a scalar pointer by a constant number of elements.
  /// result = base + offset * sizeof(element)
  Value offsetPtr(Value basePtr, int64_t offset) {
    auto offsetVal = constI32(offset);
    return b.create<cuda_tile::OffsetOp>(loc, basePtr.getType(), basePtr,
                                         offsetVal)
        .getResult();
  }

  //===-- Control flow ----------------------------------------------------===//

  /// Create a for loop. Returns the ForOp. Caller must build the body
  /// and call endFor() afterwards.
  cuda_tile::ForOp beginFor(Value lb, Value ub, Value step,
                            ValueRange initArgs) {
    auto forOp = b.create<cuda_tile::ForOp>(loc, lb, ub, step, initArgs);
    // ForOp with initArgs doesn't create a terminator automatically.
    // Add a placeholder ContinueOp that endFor() will replace.
    Block *body = forOp.getBody();
    if (!body->mightHaveTerminator()) {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToEnd(body);
      // Yield the iter args back as-is (placeholder).
      b.create<cuda_tile::ContinueOp>(loc, forOp.getRegionIterValues());
    }
    // Set insertion point before the terminator for body ops.
    b.setInsertionPoint(body->getTerminator());
    return forOp;
  }

  /// Finalize the for loop body: replace the implicit ContinueOp with one
  /// that yields the given values, then restore insertion point after the loop.
  void endFor(cuda_tile::ForOp forOp, ValueRange yieldValues) {
    // Replace the implicit (empty) ContinueOp with one carrying yield values.
    auto *terminator = forOp.getBody()->getTerminator();
    b.setInsertionPoint(terminator);
    b.create<cuda_tile::ContinueOp>(loc, yieldValues);
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
    return cuda_tile::writeBytecode(os, moduleOp);
  }

  /// Get the module for inspection/debugging.
  cuda_tile::ModuleOp getModule() { return moduleOp; }
  OpBuilder &builder() { return b; }
  MLIRContext *getContext() { return ctx; }
  Location getLoc() { return loc; }

private:
  MLIRContext *ctx;
  Location loc;
  OpBuilder b;
  cuda_tile::ModuleOp moduleOp = nullptr;
  cuda_tile::EntryOp entryOp = nullptr;
  SmallVector<Value> entryArgs;
};

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
      opInputs.push_back(e.constSplat(tileShape, elemType, *value));
      continue;
    }
    return false;
  }
  return true;
}

static Value emitElementwiseGenericBody(CudaTileOpEmitter &e,
                                        linalg::GenericOp genericOp,
                                        ArrayRef<Value> inputTiles,
                                        ArrayRef<int64_t> tileShape,
                                        Type elemType, Value fallback) {
  Block &body = genericOp.getRegion().front();
  DenseMap<Value, Value> bodyMap;

  int64_t numDpsInputs = genericOp.getNumDpsInputs();
  for (int64_t i = 0; i < numDpsInputs && i < (int64_t)inputTiles.size(); ++i)
    bodyMap[body.getArgument(i)] = inputTiles[i];

  Value current = fallback;
  for (Operation &op : body.without_terminator()) {
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

    StringRef name = mapArithToCudaTileLocal(&op);
    if (name.empty())
      name = mapMathToCudaTileLocal(&op);
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
    auto signAttr = cuda_tile::SignednessAttr::get(
        e.builder().getContext(), cuda_tile::Signedness::Unsigned);
    auto rndAttr = cuda_tile::RoundingModeAttr::get(
        e.builder().getContext(), cuda_tile::RoundingMode::ZERO);
    auto loc = e.builder().getUnknownLoc();
    Value remaining = bidZ;
    for (int64_t i = rank - 3; i >= 0; --i) {
      int64_t dimTiles = (shape[i] + tileShape[i] - 1) / tileShape[i];
      auto dimTilesVal = e.constI32(dimTiles);
      indices[i] =
          e.builder()
              .create<cuda_tile::RemIOp>(loc, remaining, dimTilesVal, signAttr)
              .getResult();
      remaining =
          e.builder()
              .create<cuda_tile::DivIOp>(loc, remaining, dimTilesVal, signAttr,
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
                      Type elemType) {
  KernelBoilerplate bp;

  e.beginModule(kernelName);
  e.beginEntry("main", numArgs, elemType);

  for (int64_t i = 0; i < numArgs; ++i) {
    auto tv = e.makeTensorView(e.getArg(i), shape, strides, elemType);
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
  auto srcTileType = cuda_tile::TileType::get(ctx, srcTileShape, elemType);
  auto dstTileType = cuda_tile::TileType::get(ctx, dstTileShape, elemType);
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
  SmallVector<int64_t> inputTileShape;
  for (size_t i = 0; i < srcShape.size(); ++i) {
    if ((int64_t)i == reduceDim) {
      inputTileShape.push_back(srcShape[i]); // full reduce dim
    } else if (i == srcShape.size() - 1) {
      inputTileShape.push_back(std::min(tileN, srcShape[i]));
    } else if (i == srcShape.size() - 2) {
      inputTileShape.push_back(std::min(tileM, srcShape[i]));
    } else {
      inputTileShape.push_back(srcShape[i]);
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

  e.beginModule(kernelName);
  e.beginEntry("main", 2, elemType); // input + output bindings

  auto srcStrides = computeRowMajorStrides(srcShape);
  auto srcView = e.makeTensorView(e.getArg(0), srcShape, srcStrides, elemType);
  auto srcPart = e.makePartitionView(srcView, inputTileShape);

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

  auto signAttr = cuda_tile::SignednessAttr::get(
      ctx, cuda_tile::Signedness::Unsigned);
  auto rndAttr = cuda_tile::RoundingModeAttr::get(
      ctx, cuda_tile::RoundingMode::ZERO);
  auto batchDivisor = e.constI32(ohTiles);
  auto batchId =
      e.builder().create<cuda_tile::DivIOp>(e.getLoc(), bz, batchDivisor,
                                            signAttr, rndAttr)
          .getResult();
  auto ohBlockId =
      e.builder().create<cuda_tile::RemIOp>(e.getLoc(), bz, batchDivisor,
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
      auto pA = e.makePartitionView(vA, tileA4);

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
      auto pB = e.makePartitionView(vB, tB);

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

  auto signAttr = cuda_tile::SignednessAttr::get(
      ctx, cuda_tile::Signedness::Unsigned);
  auto rndAttr = cuda_tile::RoundingModeAttr::get(
      ctx, cuda_tile::RoundingMode::ZERO);
  auto batchDivisor = e.constI32(ohTiles);
  auto batchId =
      e.builder()
          .create<cuda_tile::DivIOp>(e.getLoc(), bz, batchDivisor, signAttr,
                                     rndAttr)
          .getResult();
  auto ohBlockId =
      e.builder()
          .create<cuda_tile::RemIOp>(e.getLoc(), bz, batchDivisor, signAttr)
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
  auto pA = e.makePartitionView(vA, tileA);

  SmallVector<int64_t> shB = {bShape[0], bShape[1]};
  SmallVector<int64_t> stB = {bShape[1], 1};
  SmallVector<int64_t> tileB = {tK, tN};
  auto vB = e.makeTensorView(e.getArg(bindB), shB, stB, elemType);
  auto pB = e.makePartitionView(vB, tileB);

  auto [bx, by, bz] = e.getTileBlockId();
  Value batchId = bz;
  Value ohBlockId = by;
  if (hasBatch) {
    auto signAttr = cuda_tile::SignednessAttr::get(
        ctx, cuda_tile::Signedness::Unsigned);
    auto rndAttr = cuda_tile::RoundingModeAttr::get(
        ctx, cuda_tile::RoundingMode::ZERO);
    Value ohTiles = e.constI32((OH + tOH - 1) / tOH);
    batchId =
        e.builder()
            .create<cuda_tile::DivIOp>(e.getLoc(), bz, ohTiles, signAttr,
                                       rndAttr)
            .getResult();
    ohBlockId =
        e.builder()
            .create<cuda_tile::RemIOp>(e.getLoc(), bz, ohTiles, signAttr)
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
                    ArrayRef<int64_t> srcShape, ArrayRef<int64_t> dstShape,
                    Type elemType, int64_t tileM, int64_t tileN) {
  LLVM_DEBUG(llvm::dbgs() << "[cuda_tile]   entering reduce path\n");
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

  int64_t numBindings = 0;
  innerModule->walk([&](Operation *op) {
    if (op->getName().getStringRef() == "hal.interface.binding.subspan")
      numBindings++;
  });
  if (numBindings == 0)
    numBindings = 3;
  emissionPlan.numBindings = numBindings;
  emissionPlan.actualNumBindings = numBindings;

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
    bindC = numBindings - 1;

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
                 << " bindings=" << plan.bindingShapes.size() << "\n";
  });

  int64_t tileM = plan.schedule.tileM ? plan.schedule.tileM : options.tileM;
  int64_t tileN = plan.schedule.tileN ? plan.schedule.tileN : options.tileN;
  int64_t tileK = plan.schedule.tileK ? plan.schedule.tileK : options.tileK;

  auto &bindingShapes = plan.bindingShapes;
  int genericOpCount = plan.genericOpCount;

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

        int64_t numBindings = bindingShapes.size();
        if (numBindings < 2)
          numBindings = 2;

        CudaTileOpEmitter e(ctx);
        e.beginModule(kernelName);
        e.beginEntry("main", numBindings, elemType);

        // Create per-binding views (handles broadcast shapes).
        SmallVector<Value> partViews;
        SmallVector<SmallVector<int64_t>> bindTileShapes;
        for (int64_t i = 0; i < numBindings; ++i) {
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
          auto tv =
              e.makeTensorView(e.getArg(i), bShape, bStrides, elemType);
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
                    cuda_tile::TileType::get(ctx, cTile, elemType);
                auto tensorCstType =
                    RankedTensorType::get(cTile, elemType);
                auto attr = DenseElementsAttr::get(
                    tensorCstType,
                    ArrayRef<float>(padded.data(), padded.size()));
                Value tile =
                    e.builder()
                        .create<cuda_tile::ConstantOp>(
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
                matchReduceCombinerLocal(genOp.getRegion());
            SmallVector<StringRef> preOps; // ops applied before reduce
            if (combiner.empty()) {
              // Multi-op body: extract pre-reduce and combiner ops.
              // Walk body ops: everything before the accumulator op is
              // a pre-reduce elementwise; the last 2-input op is the combiner.
              Block &block = genOp.getRegion().front();
              for (auto &op : block.without_terminator()) {
                StringRef name = mapArithToCudaTileLocal(&op);
                if (name.empty())
                  name = mapMathToCudaTileLocal(&op);
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

              StringRef name = mapArithToCudaTileLocal(&op);
              if (name.empty())
                name = mapMathToCudaTileLocal(&op);
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
      semanticKind == CudaTileSemanticKind::Reduction)
    return emitReductionKernel(ctx, kernelName, plan, primaryOp, srcShape,
                               dstShape, elemType, tileM, tileN);

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

    auto vA = e.makeTensorView(e.getArg(bindA), shA2, stA, elemType);

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
      auto vB = e.makeTensorView(e.getArg(bindB), shB2, stB, elemType);
      pBv = e.makePartitionView(vB, tB);
    }

    auto vC = e.makeTensorView(e.getArg(bindC), shC2, stC, elemType);
    SmallVector<int64_t> tA = {aTM, aTK}, tC = {aTM, aTN};
    auto pA = e.makePartitionView(vA, tA);
    auto pC = e.makePartitionView(vC, tC);

    auto [bx, by, bz] = e.getTileBlockId();
    auto accInit = e.constSplat(tC, elemType, 0.0);

    int64_t nK = (K + aTK - 1) / aTK;
    auto lb = e.constI32(0);
    auto ub = e.constI32(nK);
    auto step = e.constI32(1);

    auto forOp = e.beginFor(lb, ub, step, ValueRange{accInit});
    Value iv = forOp.getInductionVar();
    Value iterAcc = forOp.getRegionIterValues()[0];

    auto [tAd, tokA] = e.loadViewTko(pA, {by, iv}, tA, elemType);
    Value tBd;
    if (weightConstant) {
      // B is an embedded constant. Create the full constant tile.
      // For the K-loop, we need the slice [iv*aTK : (iv+1)*aTK, :].
      // When nK==1, the full constant IS the tile.
      // TODO: support nK>1 by slicing the constant per K iteration.
      auto cstOp = weightConstant.getDefiningOp<arith::ConstantOp>();
      auto attrVal = cstOp.getValue();

      // Extract float data from DenseElementsAttr or DenseResourceElementsAttr.
      SmallVector<float> rawData;
      if (auto dense = dyn_cast<DenseElementsAttr>(attrVal)) {
        for (auto val : dense.getValues<float>())
          rawData.push_back(val);
      } else if (auto resAttr =
                     dyn_cast<DenseResourceElementsAttr>(attrVal)) {
        auto blob = resAttr.getRawHandle().getBlob();
        if (blob) {
          auto data = blob->getData();
          auto *floats = reinterpret_cast<const float *>(data.data());
          int64_t numElem = data.size() / sizeof(float);
          rawData.assign(floats, floats + numElem);
        }
      }
      if (rawData.empty())
        return failure();

      // Physical shape is the original weight shape. Tile shape must be
      // power-of-2 padded. Pad the data with zeros for extra rows/cols.
      int64_t physRows = bTransposed ? N : K;
      int64_t physCols = bTransposed ? K : N;
      int64_t tileRows = bTransposed ? aTN : aTK;
      int64_t tileCols = bTransposed ? aTK : aTN;

      SmallVector<float> paddedData(tileRows * tileCols, 0.0f);
      for (int64_t r = 0; r < physRows && r < tileRows; ++r)
        for (int64_t c = 0; c < physCols && c < tileCols; ++c) {
          int64_t srcIdx = r * physCols + c;
          if (srcIdx < (int64_t)rawData.size())
            paddedData[r * tileCols + c] = rawData[srcIdx];
        }

      SmallVector<int64_t> tileBShape = {tileRows, tileCols};
      auto tileBType =
          cuda_tile::TileType::get(ctx, tileBShape, elemType);
      auto tensorType = RankedTensorType::get(tileBShape, elemType);
      auto paddedAttr = DenseElementsAttr::get(
          tensorType,
          ArrayRef<float>(paddedData.data(), paddedData.size()));
      tBd = e.builder()
                .create<cuda_tile::ConstantOp>(
                    e.builder().getUnknownLoc(), tileBType,
                    cast<DenseTypedElementsAttr>(
                        paddedAttr.reshape(cast<ShapedType>(tileBType))))
                .getResult();
      if (bTransposed) {
        // Permute from [aTN, aTK] to [aTK, aTN] for mmaf.
        auto permType =
            cuda_tile::TileType::get(ctx, {aTK, aTN}, elemType);
        tBd = e.permute(tBd, {1, 0}, permType);
      }
    } else if (bTransposed) {
      // Load B tile as [aTN, aTK] then permute to [aTK, aTN].
      auto [tBraw, tokB] = e.loadViewTko(pBv, {bx, iv}, tB, elemType);
      SmallVector<int64_t> permTB = {aTK, aTN};
      auto permType = cuda_tile::TileType::get(ctx, permTB, elemType);
      tBd = e.permute(tBraw, {1, 0}, permType);
    } else {
      auto [tBraw, tokB] = e.loadViewTko(pBv, {iv, bx}, tB, elemType);
      tBd = tBraw;
    }
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
      SmallVector<linalg::GenericOp> postOps;
      innerModule->walk([&](linalg::GenericOp genOp) {
        if (genOp.getOperation() == primaryOp)
          return; // skip the matmul itself
        if (genOp.getNumReductionLoops() == 0 &&
            genOp.getNumParallelLoops() > 0)
          postOps.push_back(genOp);
      });

      for (auto postOp : postOps) {
        // Walk the body's SSA graph to apply the elementwise ops.
        Block &body = postOp.getRegion().front();
        DenseMap<Value, Value> bodyMap;

        // Map block args to tiles. The first arg is typically the matmul
        // result (connected via SSA), subsequent args are additional inputs
        // (bias, etc.) from later bindings.
        int64_t argIdx = 0;
        for (int64_t a = 0; a < postOp.getNumDpsInputs(); ++a) {
          Value input = postOp.getDpsInputs()[a];
          // Check if this input comes from the matmul result.
          bool isMatmulResult = false;
          if (input.getDefiningOp() == primaryOp)
            isMatmulResult = true;
          // Also check linalg.fill → matmul chain.
          for (auto result : primaryOp->getResults()) {
            if (input == result)
              isMatmulResult = true;
          }

          if (isMatmulResult) {
            bodyMap[body.getArgument(a)] = finalAcc;
          } else {
            // Try binding first, then constant.
            bool found = false;

            // Check bindings (trace through dispatch.tensor.load).
            for (int64_t bi = 0;
                 bi < (int64_t)bindingShapes.size() && !found; ++bi) {
              bool match = (bindingShapes[bi].memref == input);
              if (!match) {
                auto it = valueToBind.find(input);
                if (it != valueToBind.end() && it->second == bi)
                  match = true;
              }
              if (match) {
                auto &bInfo = bindingShapes[bi];
                int64_t bRank = bInfo.shape.size();
                SmallVector<int64_t> bTile;
                for (int64_t d = 0; d < bRank; ++d)
                  bTile.push_back(nextPow2(std::min(
                      d == bRank - 1 ? tileN : tileM, bInfo.shape[d])));
                auto bStrides = computeRowMajorStrides(bInfo.shape);
                auto tv = e.makeTensorView(e.getArg(bi), bInfo.shape,
                                           bStrides, elemType);
                auto pv = e.makePartitionView(tv, bTile);
                SmallVector<Value> bIndices;
                if (bRank == 2) bIndices = {by, bx};
                else if (bRank == 1) bIndices = {bx};
                auto [tile, tok] =
                    e.loadViewTko(pv, bIndices, bTile, elemType);
                if (bRank < 2) {
                  SmallVector<int64_t> reshapeShape = {1, bTile[0]};
                  SmallVector<int64_t> broadcastShape = {aTM, aTN};
                  tile = e.reshape(tile, reshapeShape, elemType);
                  if (reshapeShape != broadcastShape)
                    tile = e.broadcastTile(tile, broadcastShape, elemType);
                }
                bodyMap[body.getArgument(a)] = tile;
                found = true;
              }
            }

            // Check if input is a constant (bias embedded in dispatch).
            if (!found) {
              if (auto cstOp =
                      input.getDefiningOp<arith::ConstantOp>()) {
                SmallVector<float> cstData;
                auto cstAttr = cstOp.getValue();
                if (auto dense = dyn_cast<DenseElementsAttr>(cstAttr)) {
                  for (auto v : dense.getValues<float>())
                    cstData.push_back(v);
                } else if (auto resAttr =
                               dyn_cast<DenseResourceElementsAttr>(
                                   cstAttr)) {
                  auto blob = resAttr.getRawHandle().getBlob();
                  if (blob) {
                    auto data = blob->getData();
                    auto *f =
                        reinterpret_cast<const float *>(data.data());
                    cstData.assign(f, f + data.size() / sizeof(float));
                  }
                }
                if (!cstData.empty()) {
                  auto cstType = dyn_cast<ShapedType>(input.getType());
                  auto cstShape = cstType ? cstType.getShape()
                                          : ArrayRef<int64_t>{};
                  int64_t cRank = cstShape.size();
                  // Pad to power-of-2 tile shape.
                  SmallVector<int64_t> cTile;
                  for (int64_t d = 0; d < cRank; ++d)
                    cTile.push_back(nextPow2(cstShape[d]));
                  int64_t cElems = 1;
                  for (auto d : cTile) cElems *= d;
                  // Pad data with zeros (generic N-dimensional).
                  SmallVector<float> padded(cElems, 0.0f);
                  int64_t totalSrcElems = 1;
                  for (int64_t d = 0; d < cRank; ++d)
                    totalSrcElems *= cstShape[d];
                  for (int64_t flat = 0;
                       flat < totalSrcElems &&
                       flat < (int64_t)cstData.size();
                       ++flat) {
                    int64_t rem = flat;
                    SmallVector<int64_t> coords(cRank);
                    for (int64_t d = cRank - 1; d >= 0; --d) {
                      coords[d] = rem % cstShape[d];
                      rem /= cstShape[d];
                    }
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
                      cuda_tile::TileType::get(ctx, cTile, elemType);
                  auto tensorCstType =
                      RankedTensorType::get(cTile, elemType);
                  auto attr = DenseElementsAttr::get(
                      tensorCstType,
                      ArrayRef<float>(padded.data(), padded.size()));
                  Value tile =
                      e.builder()
                          .create<cuda_tile::ConstantOp>(
                              e.builder().getUnknownLoc(), tileCstType,
                              cast<DenseTypedElementsAttr>(
                                  attr.reshape(
                                      cast<ShapedType>(tileCstType))))
                          .getResult();
                  // Broadcast if needed (1D bias → 2D).
                  if (cRank < 2) {
                    SmallVector<int64_t> rshp = {1, cTile[0]};
                    SmallVector<int64_t> bshp = {aTM, aTN};
                    tile = e.reshape(tile, rshp, elemType);
                    if (rshp != bshp)
                      tile = e.broadcastTile(tile, bshp, elemType);
                  }
                  bodyMap[body.getArgument(a)] = tile;
                  found = true;
                }
              }
            }
          }
          argIdx++;
        }

        // Walk body ops and apply them.
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
                opInputs.push_back(e.constSplat(tC, elemType, val));
              } else {
                allResolved = false;
              }
            }
            if (allResolved && opInputs.size() == 2) {
              Value result =
                  e.emitCmpF(cmpOp.getPredicate(), opInputs[0], opInputs[1]);
              bodyMap[op.getResult(0)] = result;
            }
            continue;
          }

          StringRef name = mapArithToCudaTileLocal(&op);
          if (name.empty())
            name = mapMathToCudaTileLocal(&op);
          if (name.empty())
            continue;

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
              opInputs.push_back(e.constSplat(tC, elemType, val));
            } else {
              allResolved = false;
            }
          }
          if (allResolved && !opInputs.empty()) {
            Value result = e.emitElementwise(name, opInputs);
            if (op.getNumResults() > 0)
              bodyMap[op.getResult(0)] = result;
            finalAcc = result;
          }
        }
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

    int64_t numBindings = bindingShapes.size();
    if (numBindings == 0)
      numBindings = primaryOp->getNumOperands() + primaryOp->getNumResults();
    if (numBindings < 2)
      numBindings = 2;

    // Check if any input binding has a different shape (broadcast).
    bool hasBroadcast = false;
    for (int64_t i = 0; i < (int64_t)bindingShapes.size() - 1; ++i) {
      if (!bindingShapes[i].shape.empty() &&
          bindingShapes[i].shape != SmallVector<int64_t>(shape))
        hasBroadcast = true;
    }

    if (!hasBroadcast) {
      // No broadcast: use standard boilerplate.
      auto bp = emitKernelBoilerplate(e, kernelName, numBindings, shape,
                                      strides, tileShape, elemType);

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
                                               tileShape, elemType, current);
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
    for (int64_t i = 0; i < numBindings; ++i) {
      auto bShape =
          (i < (int64_t)bindingShapes.size() && !bindingShapes[i].shape.empty())
              ? SmallVector<int64_t>(bindingShapes[i].shape)
              : SmallVector<int64_t>(shape);
      auto bStrides = computeRowMajorStrides(bShape);
      auto bTile = computeTileShape(bShape, tileM, tileN);
      auto tv = e.makeTensorView(e.getArg(i), bShape, bStrides, elemType);
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

    Value current = loadInputTile(0);

    if (loweringStrategy == CudaTileLoweringStrategy::Elementwise) {
      auto genericOp = dyn_cast<linalg::GenericOp>(primaryOp);
      if (genericOp) {
        SmallVector<Value> inputTiles;
        int64_t inputCount =
            std::min<int64_t>(genericOp.getNumDpsInputs(),
                              std::max<int64_t>(numBindings - 1, 0));
        for (int64_t i = 0; i < inputCount; ++i)
          inputTiles.push_back(loadInputTile(i));
        current = emitElementwiseGenericBody(e, genericOp, inputTiles,
                                             tileShape, elemType, current);
      } else {
        SmallVector<StringRef> ops = getElementwiseOpNames(primaryOp);
        Value tile2;
        if (numBindings >= 3)
          tile2 = loadInputTile(1);
        for (auto opName : ops) {
          if (tile2 && opName == ops.front() && numBindings >= 3)
            current = e.emitElementwise(opName, {current, tile2});
          else
            current = e.emitElementwise(opName, {current});
        }
      }
    }

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
    registry.insert<cuda_tile::CudaTileDialect>();
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
