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

#include "compiler/plugins/target/CudaTile/CudaTileOptions.h"

#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Utils/ExecutableDebugInfoUtils.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/compiler/Utils/StringUtils.h"
#include "iree/compiler/Utils/ToolUtils.h"
#include "iree/schemas/cuda_tile_executable_def_builder.h"
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
    // cuda_tile bypasses LLVM NVPTX — no LLVM/NVVM/GPU dialects needed.
    // When the cuda_tile MLIR dialect is linked in, register it here:
    // registry.insert<cuda_tile::CudaTileDialect>();
  }

  void buildConfigurationPassPipeline(
      IREE::HAL::ExecutableTargetAttr targetAttr,
      OpPassManager &passManager) override {
    // TODO(Phase 4): cuda_tile-specific codegen configuration passes.
    // For now, external pre-compiled objects bypass this pipeline.
  }

  void buildTranslationPassPipeline(
      IREE::HAL::ExecutableTargetAttr targetAttr,
      OpPassManager &passManager) override {
    // TODO(Phase 4): LinalgToCudaTileText + BytecodeWriter translation.
    // For now, external pre-compiled objects bypass this pipeline.
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
    } else {
      //=== Codegen path (not yet implemented) ===//
      return variantOp.emitError()
             << "cuda_tile codegen path not yet implemented; use external "
                "objects via #hal.executable.object<...>";
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

      auto kernelNameRef =
          builder.createString(sanitizeSymbolName(exportOp.getName()));

      // Grid dims default to {1,1,1} — runtime overrides via workgroup_count.
      iree_hal_cuda_tile_GridDims_t gridDims = {1, 1, 1};

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
