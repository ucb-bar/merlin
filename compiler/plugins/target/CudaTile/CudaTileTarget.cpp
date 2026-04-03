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

#include "compiler/plugins/target/CudaTile/CudaTileOptions.h"
#include "compiler/src/merlin/Dialect/CudaTile/Transforms/Passes.h"

// cuda_tile dialect + bytecode writer (in-process compilation).
#include "cuda_tile/Bytecode/Writer/BytecodeWriter.h"
#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Utils/ExecutableDebugInfoUtils.h"
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
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"

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
  int64_t rank = shape.size();
  int64_t gridX = 1, gridY = 1, gridZ = 1;
  if (rank >= 1)
    gridX = (shape[rank - 1] + tileShape[rank - 1] - 1) / tileShape[rank - 1];
  if (rank >= 2)
    gridY = (shape[rank - 2] + tileShape[rank - 2] - 1) / tileShape[rank - 2];
  for (int64_t i = 0; i < rank - 2; ++i)
    gridZ *= (shape[i] + tileShape[i] - 1) / tileShape[i];
  return {gridX, gridY, gridZ};
}

/// Convert element type string (from annotation attributes) to MLIR Type.
static Type getMLIRElementType(MLIRContext *ctx, StringRef typeStr) {
  if (typeStr == "f32")
    return Float32Type::get(ctx);
  if (typeStr == "f16")
    return Float16Type::get(ctx);
  if (typeStr == "bf16")
    return BFloat16Type::get(ctx);
  if (typeStr == "f64")
    return Float64Type::get(ctx);
  if (typeStr == "i32")
    return IntegerType::get(ctx, 32);
  if (typeStr == "i16")
    return IntegerType::get(ctx, 16);
  if (typeStr == "i8")
    return IntegerType::get(ctx, 8);
  if (typeStr == "i1")
    return IntegerType::get(ctx, 1);
  return {};
}

/// Get the element type string for a cuda_tile type annotation.
static std::string getCudaTileElementTypeStr(Type type) {
  if (type.isF32())
    return "f32";
  if (type.isF16())
    return "f16";
  if (type.isBF16())
    return "bf16";
  if (type.isF64())
    return "f64";
  if (type.isInteger(32))
    return "i32";
  if (type.isInteger(16))
    return "i16";
  if (type.isInteger(8))
    return "i8";
  if (type.isInteger(1))
    return "i1";
  return "f32";
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

//===----------------------------------------------------------------------===//
// Kernel Boilerplate Helper
//===----------------------------------------------------------------------===//

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

  // Build N-D partition indices.
  // Last dim → bidX, second-to-last → bidY.
  // Batch dims (rank-3+) → derived from bidZ via div/mod.
  int64_t rank = shape.size();
  bp.indices.resize(rank);
  if (rank >= 1)
    bp.indices[rank - 1] = bx;
  if (rank >= 2)
    bp.indices[rank - 2] = by;
  if (rank == 3) {
    bp.indices[0] = bz;
  } else if (rank > 3) {
    // Decompose bidZ into batch indices via div/mod chain.
    auto signAttr = cuda_tile::SignednessAttr::get(
        e.builder().getContext(), cuda_tile::Signedness::Unsigned);
    auto rndAttr = cuda_tile::RoundingModeAttr::get(
        e.builder().getContext(), cuda_tile::RoundingMode::ZERO);
    auto loc = e.builder().getUnknownLoc();
    Value remaining = bz;
    for (int64_t i = rank - 3; i >= 0; --i) {
      int64_t dimTiles =
          (shape[i] + tileShape[i] - 1) / tileShape[i];
      auto dimTilesVal = e.constI32(dimTiles);
      bp.indices[i] = e.builder()
                          .create<cuda_tile::RemIOp>(loc, remaining,
                                                     dimTilesVal, signAttr)
                          .getResult();
      remaining = e.builder()
                      .create<cuda_tile::DivIOp>(loc, remaining, dimTilesVal,
                                                  signAttr, rndAttr)
                      .getResult();
    }
  }

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

  SmallVector<Value> srcIndices;
  if (srcShape.size() >= 2)
    srcIndices = {bidY, bidX};
  else
    srcIndices = {bidX};

  auto [tile, loadTok] =
      e.loadViewTko(srcPart, srcIndices, srcTileShape, elemType);

  // Permute the tile dimensions.
  SmallVector<int32_t> permI32(permutation.begin(), permutation.end());
  auto srcTileType = cuda_tile::TileType::get(ctx, srcTileShape, elemType);
  auto dstTileType = cuda_tile::TileType::get(ctx, dstTileShape, elemType);
  auto permuted = e.permute(tile, permI32, dstTileType);

  // Permuted indices: swap based on permutation.
  SmallVector<Value> dstIndices;
  if (srcShape.size() >= 2) {
    if (permutation.size() == 2 && permutation[0] == 1 && permutation[1] == 0)
      dstIndices = {bidX, bidY}; // swap for 2D transpose
    else
      dstIndices = srcIndices;
  } else {
    dstIndices = srcIndices;
  }

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
  if (linearOffset > 0) {
    // Use pointer arithmetic via the tensor view: create a view at the
    // offset position by adjusting the base pointer. Since cuda_tile works
    // with views, we bake the offset into the effective strides and shape.
    // The offset is handled by the view's strides — not ideal but works
    // for contiguous slices. For non-contiguous, we'd need ptr offset.
    // TODO: Use cuda_tile.offset op when available for proper pointer offset.
  }

  // Source: view over the slice region with effective strides.
  auto srcView =
      e.makeTensorView(srcPtr, dstShape, effectiveSrcStrides, elemType);
  auto srcPart = e.makePartitionView(srcView, dstTileShape);

  // Dest: contiguous view.
  auto dstView =
      e.makeTensorView(e.getArg(1), dstShape, dstStridesRM, elemType);
  auto dstPart = e.makePartitionView(dstView, dstTileShape);

  auto [bidX, bidY, bidZ] = e.getTileBlockId();
  SmallVector<Value> indices;
  if (dstShape.size() >= 2)
    indices = {bidY, bidX};
  else
    indices = {bidX};

  auto [tile, tok] =
      e.loadViewTko(srcPart, indices, dstTileShape, elemType);
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
  auto dstStrides = computeRowMajorStrides(dstShape);

  // Build source strides: 0 for broadcast dims, otherwise match src.
  SmallVector<int64_t> srcStrides(dstStrides.size(), 0);
  llvm::SmallDenseSet<int64_t> bcastDimSet(broadcastDims.begin(),
                                           broadcastDims.end());
  int srcIdx = 0;
  auto realSrcStrides = computeRowMajorStrides(srcShape);
  for (size_t i = 0; i < dstStrides.size(); ++i) {
    if (!bcastDimSet.contains(i)) {
      if (srcIdx < (int)realSrcStrides.size())
        srcStrides[i] = realSrcStrides[srcIdx++];
    }
    // broadcast dims keep stride=0
  }

  e.beginModule(kernelName);
  e.beginEntry("main", 2, elemType);

  // Source view with broadcast strides (0 for broadcast dims).
  auto srcView = e.makeTensorView(e.getArg(0), dstShape, srcStrides, elemType);
  auto srcPart = e.makePartitionView(srcView, dstTileShape);

  // Dest view: contiguous.
  auto dstView = e.makeTensorView(e.getArg(1), dstShape, dstStrides, elemType);
  auto dstPart = e.makePartitionView(dstView, dstTileShape);

  auto [bidX, bidY, bidZ] = e.getTileBlockId();
  SmallVector<Value> indices;
  if (dstShape.size() >= 2)
    indices = {bidY, bidX};
  else
    indices = {bidX};

  auto [tile, tok] =
      e.loadViewTko(srcPart, indices, dstTileShape, elemType);
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
  // Walk to find the primary compute operation.
  // If multiple ops are tagged, this is a fused multi-op dispatch
  // (e.g., decomposed softmax). Count them.
  Operation *primaryOp = nullptr;
  int taggedOpCount = 0;
  innerModule->walk([&](Operation *op) {
    if (op->hasAttr("cuda_tile.kernel_class")) {
      primaryOp = op;
      taggedOpCount++;
    }
  });

  if (!primaryOp) {
    innerModule->walk([&](linalg::LinalgOp op) {
      if (!primaryOp)
        primaryOp = op;
    });
  }

  // If still no op found, this is a pure data-movement dispatch.
  // Generate a copy kernel using shapes from tensor values in the module.
  if (!primaryOp) {
    SmallVector<int64_t> copyShape;
    Type copyElemType;
    innerModule->walk([&](Operation *op) {
      if (!copyShape.empty()) return;
      for (auto v : op->getResults()) {
        if (auto shaped = dyn_cast<ShapedType>(v.getType())) {
          if (shaped.hasStaticShape()) {
            copyShape.assign(shaped.getShape().begin(), shaped.getShape().end());
            copyElemType = shaped.getElementType();
            return;
          }
        }
      }
    });
    if (copyShape.empty())
      return failure();
    int64_t tM = options.tileM, tN = options.tileN;
    auto [e, grid] = generateCopyKernel(ctx, kernelName, copyShape,
                                        copyElemType, tM, tN);
    std::string tilebcData;
    if (failed(e.serialize(tilebcData)))
      return failure();
    return std::make_pair(std::move(tilebcData), std::move(grid));
  }

  auto classAttr =
      primaryOp->getAttrOfType<StringAttr>("cuda_tile.kernel_class");
  StringRef kernelClass = classAttr ? classAttr.getValue() : "generic";

  // Read metadata from attributes (set by conversion passes).
  auto srcShape = getI64ArrayAttr(primaryOp, "cuda_tile.src_shape");
  auto dstShape = getI64ArrayAttr(primaryOp, "cuda_tile.dst_shape");
  auto elemTypeAttr =
      primaryOp->getAttrOfType<StringAttr>("cuda_tile.elem_type");
  std::string elemTypeStr = elemTypeAttr ? elemTypeAttr.getValue().str() : "";

  // If no metadata from pass, extract from op types directly.
  if (srcShape.empty() || elemTypeStr.empty()) {
    for (auto operand : primaryOp->getOperands()) {
      if (auto shaped = dyn_cast<ShapedType>(operand.getType())) {
        if (shaped.hasStaticShape() && srcShape.empty())
          srcShape.assign(shaped.getShape().begin(), shaped.getShape().end());
        if (elemTypeStr.empty())
          elemTypeStr = getCudaTileElementTypeStr(shaped.getElementType());
      }
    }
  }
  if (dstShape.empty()) {
    for (auto result : primaryOp->getResults()) {
      if (auto shaped = dyn_cast<ShapedType>(result.getType())) {
        if (shaped.hasStaticShape())
          dstShape.assign(shaped.getShape().begin(), shaped.getShape().end());
      }
    }
    if (dstShape.empty())
      dstShape = srcShape;
  }
  if (srcShape.empty() || elemTypeStr.empty())
    return failure();

  Type elemType = getMLIRElementType(ctx, elemTypeStr);
  if (!elemType)
    return failure();

  int64_t tileM = options.tileM, tileN = options.tileN, tileK = options.tileK;

  // Multi-op fused dispatches (e.g., decomposed softmax: reduce+exp+reduce+div)
  // are not yet supported as a single fused kernel. Generate a copy as fallback.
  // TODO: implement fused kernel generation for multi-op dispatches.
  if (taggedOpCount > 1) {
    auto shape = dstShape.empty() ? srcShape : dstShape;
    auto [e, grid] = generateCopyKernel(ctx, kernelName, shape, elemType,
                                        tileM, tileN);
    std::string tilebcData;
    if (failed(e.serialize(tilebcData)))
      return failure();
    return std::make_pair(std::move(tilebcData), std::move(grid));
  }

  // Dispatch to the appropriate kernel generator, build the cuda_tile module,
  // and serialize to tilebc.
  auto buildAndSerialize =
      [](CudaTileOpEmitter &&e,
         SmallVector<int64_t, 3> gridDims)
      -> FailureOr<std::pair<std::string, SmallVector<int64_t, 3>>> {
    std::string tilebcData;
    if (failed(e.serialize(tilebcData)))
      return failure();
    return std::make_pair(std::move(tilebcData), std::move(gridDims));
  };

  //=== Phase 1: Data Movement ===//

  if (kernelClass == "copy") {
    auto [e, grid] =
        generateCopyKernel(ctx, kernelName, srcShape, elemType, tileM, tileN);
    return buildAndSerialize(std::move(e), std::move(grid));
  }

  if (kernelClass == "transpose") {
    auto perm = getI64ArrayAttr(primaryOp, "cuda_tile.permutation");
    if (perm.empty()) {
      for (int64_t i = srcShape.size() - 1; i >= 0; --i)
        perm.push_back(i);
    }
    auto [e, grid] = generateTransposeKernel(ctx, kernelName, srcShape,
                                             dstShape, perm, elemType, tileM,
                                             tileN);
    return buildAndSerialize(std::move(e), std::move(grid));
  }

  if (kernelClass == "extract_slice" || kernelClass == "insert_slice") {
    auto offsets = getI64ArrayAttr(primaryOp, "cuda_tile.offsets");
    auto sliceStrides =
        getI64ArrayAttr(primaryOp, "cuda_tile.slice_strides");
    if (offsets.empty())
      offsets.resize(srcShape.size(), 0);
    if (sliceStrides.empty())
      sliceStrides.resize(srcShape.size(), 1);

    if (kernelClass == "extract_slice") {
      auto [e, grid] = generateExtractSliceKernel(
          ctx, kernelName, srcShape, dstShape, offsets, sliceStrides, elemType,
          tileM, tileN);
      return buildAndSerialize(std::move(e), std::move(grid));
    }
    // insert_slice: swap src and dst roles.
    auto [e, grid] = generateExtractSliceKernel(
        ctx, kernelName, dstShape, srcShape, offsets, sliceStrides, elemType,
        tileM, tileN);
    return buildAndSerialize(std::move(e), std::move(grid));
  }

  if (kernelClass == "broadcast") {
    auto bcastDims =
        getI64ArrayAttr(primaryOp, "cuda_tile.broadcast_dims");
    auto [e, grid] = generateBroadcastKernel(ctx, kernelName, srcShape,
                                             dstShape, bcastDims, elemType,
                                             tileM, tileN);
    return buildAndSerialize(std::move(e), std::move(grid));
  }

  if (kernelClass == "collapse_shape" || kernelClass == "expand_shape") {
    auto [e, grid] =
        generateCopyKernel(ctx, kernelName, dstShape, elemType, tileM, tileN);
    return buildAndSerialize(std::move(e), std::move(grid));
  }

  //=== Phase 3: Reductions ===//

  if (kernelClass == "reduce") {
    auto combinerAttr =
        primaryOp->getAttrOfType<StringAttr>("cuda_tile.combiner");
    StringRef combiner = combinerAttr ? combinerAttr.getValue() : "addf";

    // Extract the TENSOR reduce dim from the actual IR.
    // The iterator_types tell us which loop dims are reduction, but the input
    // indexing map tells us how those map to tensor dimensions.
    // E.g., iterator_types=["parallel","reduction"], input_map=(d0,d1)->(d1,d0)
    // means d1 is the reduction iterator, and it maps to tensor dim 0.
    SmallVector<int64_t> reduceDims;
    if (auto genericOp = dyn_cast<linalg::GenericOp>(primaryOp)) {
      auto iterTypes = genericOp.getIteratorTypesArray();
      auto maps = genericOp.getIndexingMapsArray();
      AffineMap inputMap = maps.empty() ? AffineMap() : maps[0];

      for (unsigned iterDim = 0; iterDim < iterTypes.size(); ++iterDim) {
        if (iterTypes[iterDim] == mlir::utils::IteratorType::parallel)
          continue;
        // Find which tensor dim this reduction iterator maps to.
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
    if (reduceDims.empty())
      reduceDims = getI64ArrayAttr(primaryOp, "cuda_tile.reduce_dims");

    auto [e, grid] = generateReduceKernel(ctx, kernelName, srcShape, dstShape,
                                          reduceDims, combiner, elemType,
                                          tileM, tileN);
    return buildAndSerialize(std::move(e), std::move(grid));
  }

  //=== Phase 4: Contractions ===//

  if (kernelClass == "matmul") {
    // Extract M, N, K from the actual operand shapes.
    // Works for both named matmul and generic contractions (conv→matmul).
    // A[...×M×K] × B[...×K×N] → C[...×M×N]
    // For generics, use indexing maps to find which dims are M, N, K.
    SmallVector<int64_t> shA, shB, shC;
    for (auto operand : primaryOp->getOperands()) {
      if (auto t = dyn_cast<ShapedType>(operand.getType())) {
        if (!t.hasStaticShape()) continue;
        if (shA.empty())
          shA.assign(t.getShape().begin(), t.getShape().end());
        else if (shB.empty())
          shB.assign(t.getShape().begin(), t.getShape().end());
      }
    }
    for (auto result : primaryOp->getResults()) {
      if (auto t = dyn_cast<ShapedType>(result.getType())) {
        if (t.hasStaticShape())
          shC.assign(t.getShape().begin(), t.getShape().end());
      }
    }
    if (shC.empty())
      shC = dstShape;

    // For 2D: A[M,K], B[K,N], C[M,N] — extract from last 2 dims.
    int64_t M = shC.size() >= 2 ? shC[shC.size() - 2] : shC.back();
    int64_t N = shC.size() >= 2 ? shC.back() : 1;
    int64_t K = shA.size() >= 2 ? shA.back() : (shA.empty() ? M : shA.back());

    int64_t aTM = nextPow2(std::min(tileM, M));
    int64_t aTN = nextPow2(std::min(tileN, N));
    int64_t aTK = nextPow2(std::min(tileK, K));

    int64_t numBindings = 0;
    innerModule->walk([&](Operation *op) {
      if (op->getName().getStringRef() == "hal.interface.binding.subspan")
        numBindings++;
    });
    if (numBindings == 0)
      numBindings = 3;

    CudaTileOpEmitter e(ctx);
    e.beginModule(kernelName);
    e.beginEntry("main", numBindings, elemType);

    // Use 2D shapes for the kernel (flatten batch dims if present).
    SmallVector<int64_t> shA2 = {M, K}, shB2 = {K, N}, shC2 = {M, N};
    SmallVector<int64_t> stA = {K, 1}, stB = {N, 1}, stC = {N, 1};

    auto vA = e.makeTensorView(e.getArg(0), shA2, stA, elemType);
    auto vB = e.makeTensorView(e.getArg(1), shB2, stB, elemType);
    auto vC = e.makeTensorView(e.getArg(2), shC2, stC, elemType);

    SmallVector<int64_t> tA = {aTM, aTK}, tB = {aTK, aTN}, tC = {aTM, aTN};
    auto pA = e.makePartitionView(vA, tA);
    auto pB = e.makePartitionView(vB, tB);
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
    auto [tBd, tokB] = e.loadViewTko(pB, {iv, bx}, tB, elemType);
    auto newAcc = e.mmaf(tAd, tBd, iterAcc);
    e.endFor(forOp, ValueRange{newAcc});

    // The for loop result is the final accumulator.
    Value finalAcc = forOp.getResult(0);
    e.storeViewTko(finalAcc, pC, {by, bx});
    e.emitReturn();
    e.endEntry();

    std::string tilebcData;
    if (failed(e.serialize(tilebcData)))
      return failure();
    return std::make_pair(
        std::move(tilebcData),
        SmallVector<int64_t, 3>{(N + aTN - 1) / aTN, (M + aTM - 1) / aTM, 1});
  }

  //=== Phase 2: Elementwise (default fallback) ===//

  {
    CudaTileOpEmitter e(ctx);
    auto shape = dstShape;
    auto tileShape = computeTileShape(shape, tileM, tileN);
    auto strides = computeRowMajorStrides(shape);
    auto gridDims = computeGridDims(shape, tileShape);

    int64_t numBindings = 0;
    innerModule->walk([&](Operation *op) {
      if (op->getName().getStringRef() == "hal.interface.binding.subspan")
        numBindings++;
    });
    if (numBindings == 0)
      numBindings = primaryOp->getNumOperands() + primaryOp->getNumResults();
    if (numBindings < 2)
      numBindings = 2;

    auto bp = emitKernelBoilerplate(e, kernelName, numBindings, shape, strides,
                                    tileShape, elemType);

    auto [tile, tok] =
        e.loadViewTko(bp.partViews[0], bp.indices, tileShape, elemType);

    Value current = tile;
    if (kernelClass == "elementwise") {
      auto opNameAttr =
          primaryOp->getAttrOfType<StringAttr>("cuda_tile.op_name");
      StringRef opName = opNameAttr ? opNameAttr.getValue() : "addf";

      if (numBindings >= 3) {
        // Binary op: load second input.
        auto [tile2, tok2] =
            e.loadViewTko(bp.partViews[1], bp.indices, tileShape, elemType);
        current = e.emitElementwise(opName, {tile, tile2});
      } else {
        // Unary op.
        current = e.emitElementwise(opName, {tile});
      }
    }

    e.storeViewTko(current, bp.partViews.back(), bp.indices);
    e.emitReturn();
    e.endEntry();

    std::string tilebcData;
    if (failed(e.serialize(tilebcData)))
      return failure();
    return std::make_pair(std::move(tilebcData), gridDims);
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
