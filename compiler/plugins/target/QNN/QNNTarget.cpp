// QNN HAL target backend for the Merlin compiler.
//
// Registers an iree-compile target backend that produces
// `#hal.executable.target<"qnn", "qnn-context-binary", {...}>` and
// serializes each `hal.executable.variant` into an `hal.executable.binary`
// whose payload is a pre-built `.qnn-ctx` blob.
//
// IMPORTANT: this does NOT do MLIR -> QNN graph lowering. That requires
// a full op-by-op translation pass (linalg/tensor -> QNN op IR). Instead,
// this backend operates in *passthrough* mode:
//
//   - For each `hal.executable.variant` targeting "qnn", look up its
//     serialized .qnn-ctx via a manifest (--iree-hal-qnn-manifest=<json>).
//     The manifest is a JSON object mapping export symbol -> file path:
//        { "dronet$async_dispatch_5":
//        "build/qnn_chunks/dispatch_5.qnn-gpu.qnn-ctx",
//          "dronet$async_dispatch_8":
//          "build/qnn_chunks/dispatch_8.qnn-htp.qnn-ctx",
//          ... }
//   - The matching .qnn-ctx is read off disk and embedded as the
//     executable binary.
//   - When no entry exists for an export, the backend emits a 4-byte
//     placeholder ("QNNX") with a warning — this lets the rest of the
//     compile pipeline complete so users can see the per-dispatch
//     target binding in phase dumps even before all chunks are built.
//
// The runtime QNN HAL driver
// (`runtime/src/iree/hal/drivers/qnn/qnn_executable.c`) reads
// `application/octet-stream` blobs of format "qnn-context-binary" and
// hands them to QnnContext_createFromBinary, completing the loop.
//
// Future work: replace the lookup-based serializer with an MLIR -> QNN
// graph IR translator. Until then, the on-disk .qnn-ctx files are
// produced separately by `tools/compile_qnn.py`, which wraps Qualcomm's
// qairt-converter / qnn-context-binary-generator pipeline.

#include <fstream>
#include <sstream>

#include "compiler/plugins/target/QNN/Codegen/SerializeGraph.h"
#include "compiler/src/merlin/Dialect/QNN/IR/QNNDialect.h"
#include "compiler/src/merlin/Dialect/QNN/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetDevice.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace {

//===----------------------------------------------------------------------===//
// QNN backend identifiers
//===----------------------------------------------------------------------===//
// The "qnn" backend produces a single .qnn-ctx variant; the actual GPU vs HTP
// backend split is encoded in the executable.target config dictionary as
// `qnn_backend = "gpu" | "htp"`. The runtime QNN HAL driver inspects this when
// dispatching.
constexpr llvm::StringLiteral kBackendName = "qnn";
constexpr llvm::StringLiteral kFormatName = "qnn-context-binary";
constexpr llvm::StringLiteral kPlaceholderSentinel = "QNNX";

//===----------------------------------------------------------------------===//
// Options
//===----------------------------------------------------------------------===//

struct QNNTargetOptions {
	std::string qnnBackend = "gpu"; // gpu | hta | htp
	std::string manifestPath; // --iree-hal-qnn-manifest=<json>
	bool allowPlaceholder = false; // --iree-hal-qnn-allow-placeholder

	void bindOptions(OptionsBinder &binder) {
		static llvm::cl::OptionCategory category("QNN HAL Target");
		binder.opt<std::string>("iree-hal-qnn-backend", qnnBackend,
			llvm::cl::cat(category),
			llvm::cl::desc(
				"QNN backend variant for new device targets: gpu | hta | htp"));
		binder.opt<std::string>("iree-hal-qnn-manifest", manifestPath,
			llvm::cl::cat(category),
			llvm::cl::desc(
				"Path to JSON manifest mapping executable export symbol -> "
				"prebuilt .qnn-ctx file."));
		binder.opt<bool>("iree-hal-qnn-allow-placeholder", allowPlaceholder,
			llvm::cl::cat(category),
			llvm::cl::desc(
				"When set, emit a 4-byte placeholder for export symbols "
				"missing from the manifest instead of failing the compile. "
				"Off by default — the resulting VMFB will fail at runtime "
				"when QnnContext_createFromBinary tries to load it. Useful "
				"only for inspecting phase IR before all chunks are ready."));
	}
};

static QNN::Codegen::Backend getBackendFromVariant(
	IREE::HAL::ExecutableVariantOp variantOp) {
	if (auto cfg = variantOp.getTarget().getConfiguration()) {
		if (auto be = cfg.getAs<StringAttr>("qnn_backend")) {
			if (be.getValue() == "gpu")
				return QNN::Codegen::Backend::Gpu;
			if (be.getValue() == "hta")
				return QNN::Codegen::Backend::Hta;
			if (be.getValue() == "htp")
				return QNN::Codegen::Backend::Htp;
			if (be.getValue() == "cpu")
				return QNN::Codegen::Backend::Cpu;
		}
	}
	return QNN::Codegen::Backend::Gpu;
}

//===----------------------------------------------------------------------===//
// Manifest loading
//===----------------------------------------------------------------------===//

// Loads {symbol -> path} from a JSON object on disk. Errors are logged and
// the resulting map is left empty; callers fall back to the placeholder
// payload for any unmatched symbols.
static llvm::StringMap<std::string> loadManifest(StringRef path) {
	llvm::StringMap<std::string> out;
	if (path.empty())
		return out;

	auto fileOrErr = llvm::MemoryBuffer::getFile(path);
	if (!fileOrErr) {
		llvm::errs() << "[qnn-target] WARNING: cannot read manifest at " << path
					 << ": " << fileOrErr.getError().message() << "\n";
		return out;
	}
	auto valueOrErr = llvm::json::parse((*fileOrErr)->getBuffer());
	if (!valueOrErr) {
		llvm::errs() << "[qnn-target] WARNING: failed parsing manifest JSON: "
					 << llvm::toString(valueOrErr.takeError()) << "\n";
		return out;
	}
	auto *obj = valueOrErr->getAsObject();
	if (!obj) {
		llvm::errs()
			<< "[qnn-target] WARNING: manifest root is not a JSON object\n";
		return out;
	}
	for (const auto &kv : *obj) {
		if (auto path = kv.second.getAsString()) {
			out[kv.first.str()] = path->str();
		}
	}
	return out;
}

static std::optional<std::string> readBinaryFile(StringRef path) {
	auto fileOrErr = llvm::MemoryBuffer::getFile(path);
	if (!fileOrErr)
		return std::nullopt;
	return std::string((*fileOrErr)->getBuffer());
}

//===----------------------------------------------------------------------===//
// EraseQNNVariantBody pass
//===----------------------------------------------------------------------===//
// Walks an hal.executable.variant whose inner builtin.module holds the
// pre-codegen linalg/tensor ops and replaces that module with an empty
// one. The export op (which lives on the variant directly) is preserved.
// Without this pass downstream HAL->VM lowering trips on linalg ops it
// can't translate inside an opaque-target variant.

class EraseQNNVariantBodyPass
	: public PassWrapper<EraseQNNVariantBodyPass,
		  OperationPass<IREE::HAL::ExecutableVariantOp>> {
  public:
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EraseQNNVariantBodyPass)

	StringRef getArgument() const override {
		return "iree-hal-qnn-erase-variant-body";
	}
	StringRef getDescription() const override {
		return "Replace inner builtin.module bodies of QNN executable variants "
			   "with empty modules so opaque-binary serialization can run.";
	}

	void runOnOperation() override {
		IREE::HAL::ExecutableVariantOp variantOp = getOperation();
		if (variantOp.getTarget().getBackend() != kBackendName)
			return;

		// 1. Erase the inner builtin.module body. Linalg/tensor ops in there
		// would crash HAL->VM lowering since QNN does no MLIR codegen.
		SmallVector<ModuleOp> innerModules;
		for (Operation &op : variantOp.getBlock()) {
			if (auto m = dyn_cast<ModuleOp>(op))
				innerModules.push_back(m);
		}
		// Before erasing, capture the binary qnn-graph if the body has any
		// `qnn.*` ops (the in-compiler codegen path from Phase 2). Stash
		// as a `merlin.qnn_graph` attribute on the variant so
		// serializeExecutable can pick it up after erase.
		for (ModuleOp m : innerModules) {
			bool hasQnnOps = false;
			m.walk([&](Operation *op) {
				if (op->getDialect() &&
					op->getDialect()->getNamespace() == "qnn") {
					hasQnnOps = true;
				}
			});
			if (hasQnnOps) {
				std::vector<int8_t> graphBytes;
				if (failed(QNN::Codegen::serializeGraph(
						m, getBackendFromVariant(variantOp), graphBytes))) {
					signalPassFailure();
					return;
				}
				auto bytesAttr = DenseIntElementsAttr::get(
					VectorType::get({static_cast<int64_t>(graphBytes.size())},
						IntegerType::get(&getContext(), 8)),
					std::move(graphBytes));
				variantOp->setAttr("merlin.qnn_graph", bytesAttr);
			} else {
				// Phase-0 diagnostic: snapshot the *source* op kinds that were
				// inside this variant before we erase it. If the manifest
				// path then can't find a .qnn-ctx blob and allowPlaceholder
				// is off (the new default), serializeExecutable surfaces
				// these op kinds in the error so the user knows what
				// ConvertLinalgToQNN pattern needs to be added.
				SmallVector<StringRef> opKinds;
				DenseSet<StringRef> seen;
				m.walk([&](Operation *op) {
					if (auto *d = op->getDialect()) {
						auto ns = d->getNamespace();
						// Capture compute-relevant op kinds. Exclude
						// universally-present plumbing (arith.constant,
						// func.return, hal.interface.binding.subspan, the
						// builtin dialect, scf control-flow) so the diagnostic
						// names what the conversion needs to handle, not the
						// boilerplate around it.
						if (ns == "arith" || ns == "func" || ns == "hal" ||
							ns == "builtin" || ns == "scf") {
							return;
						}
						StringRef name = op->getName().getStringRef();
						if (seen.insert(name).second) {
							opKinds.push_back(name);
						}
					}
				});
				if (!opKinds.empty()) {
					SmallVector<Attribute> kindAttrs;
					for (auto k : opKinds) {
						kindAttrs.push_back(StringAttr::get(&getContext(), k));
					}
					variantOp->setAttr("merlin.qnn_unmatched_ops",
						ArrayAttr::get(&getContext(), kindAttrs));
				}
			}
		}
		for (ModuleOp m : innerModules) {
			m.getBody()->clear();
		}

		// 2. Replace the export op's `count` region.
		// The body produced by dispatch creation contains
		// `iree_tensor_ext.dispatch.workgroup_count_from_slice()`, which is
		// expected to be resolved by the codegen pipeline. With no codegen
		// for QNN we resolve it ourselves to a constant (1, 1, 1) workgroup
		// — runtime QNN dispatches the whole graph as a single call, so the
		// workgroup count is informational only and `(1, 1, 1)` is safe.
		for (auto exportOp :
			variantOp.getBlock().getOps<IREE::HAL::ExecutableExportOp>()) {
			Region &countRegion = exportOp.getWorkgroupCount();
			if (countRegion.empty())
				continue;
			Block &block = countRegion.front();
			block.clear();
			OpBuilder b(&getContext());
			b.setInsertionPointToStart(&block);
			Value one = b.create<arith::ConstantIndexOp>(exportOp.getLoc(), 1);
			b.create<IREE::HAL::ReturnOp>(
				exportOp.getLoc(), ValueRange{one, one, one});
		}
	}
};

static std::unique_ptr<Pass> createEraseQNNVariantBodyPass() {
	return std::make_unique<EraseQNNVariantBodyPass>();
}

//===----------------------------------------------------------------------===//
// QNN target backend
//===----------------------------------------------------------------------===//

static IREE::HAL::ExecutableTargetAttr getQNNExecutableTarget(
	MLIRContext *context, StringRef qnnBackend) {
	Builder b(context);
	SmallVector<NamedAttribute> configItems;
	configItems.emplace_back(
		b.getStringAttr("qnn_backend"), b.getStringAttr(qnnBackend));
	// Hint to the dispatch creator that this target consumes opaque
	// pre-compiled binaries — codegen passes should be no-ops for it.
	configItems.emplace_back(
		b.getStringAttr("opaque_binary"), b.getBoolAttr(true));
	return b.getAttr<IREE::HAL::ExecutableTargetAttr>(
		b.getStringAttr(kBackendName), b.getStringAttr(kFormatName),
		b.getDictionaryAttr(configItems));
}

class QNNTargetBackend final : public TargetBackend {
  public:
	QNNTargetBackend(const QNNTargetOptions &options) : options(options) {}

	std::string getLegacyDefaultDeviceID() const final {
		return std::string(kBackendName);
	}

	void getDefaultExecutableTargets(MLIRContext *context, StringRef deviceID,
		DictionaryAttr deviceConfigAttr,
		SmallVectorImpl<IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs)
		const final {
		// Per-device override via the target-device's config dict
		// (e.g. --iree-hal-target-device=qnn[qnn_backend=hta]). When
		// absent, fall back to the global --iree-hal-qnn-backend option.
		// This lets one compile invocation produce multiple QNN variants
		// with different backends (HTA + GPU + HTP) so a single VMFB can
		// route per-dispatch via affinity to different Qualcomm devices.
		StringRef chosenBackend = options.qnnBackend;
		if (deviceConfigAttr) {
			if (auto be = deviceConfigAttr.getAs<StringAttr>("qnn_backend")) {
				chosenBackend = be.getValue();
			}
		}
		executableTargetAttrs.push_back(
			getQNNExecutableTarget(context, chosenBackend));
	}

	TargetBackend::SupportedTypes getSupportedTypes(
		MLIRContext *context) const final {
		// QNN consumes already-compiled binaries; the MLIR-side type set is
		// generous so any chunk that survives flow/stream legalization can be
		// stamped with a QNN target.
		SupportedTypes s;
		Builder b(context);
		s.addScalarType(b.getIntegerType(8));
		s.addScalarType(b.getIntegerType(16));
		s.addScalarType(b.getIntegerType(32));
		s.addScalarType(b.getIntegerType(64));
		s.addScalarType(b.getF32Type());
		s.addScalarType(b.getF16Type());
		s.addElementType(b.getIntegerType(8));
		s.addElementType(b.getIntegerType(16));
		s.addElementType(b.getIntegerType(32));
		s.addElementType(b.getF32Type());
		s.addElementType(b.getF16Type());
		return s;
	}

	void getDependentDialects(DialectRegistry &registry) const final {
		// No special dialects — the body of QNN executables is opaque to us.
	}

	// Codegen pipelines are no-ops: we don't lower anything; serialization
	// just embeds a pre-built binary.
	void buildConfigurationPassPipeline(
		IREE::HAL::ExecutableTargetAttr targetAttr,
		OpPassManager &passManager) final {}

	void buildTranslationPassPipeline(
		IREE::HAL::ExecutableTargetAttr targetAttr,
		OpPassManager &passManager) final {
		// In-compiler codegen runs HERE (per-variant), not at
		// extendPostGlobalOptimizationPassPipeline. Reason: the conversion
		// produces qnn.* ops that ConvertToHALPass marks "explicitly
		// illegal" if they live at function-body scope. Running at
		// per-variant time keeps qnn.* ops naturally inside
		// hal.executable.variant.builtin.module (which is what
		// serializeExecutable + EraseQNNVariantBodyPass expect anyway).
		// The post-fusion / post-fold IR shape is preserved because
		// dispatch creation lowered each fused linalg.generic into a
		// dispatch whose interior is the same generic.
		passManager.addPass(QNN::createLegalizeLayoutToNHWCPass());
		// Fold ONNX-QDQ-export dequant-requant-roundtrip cycles within
		// each linalg.generic body before the converter sees them. Without
		// this, yolov8n int8 conv-tail bodies have 4-stage SiLU shapes
		// that no single pattern can match.
		passManager.addPass(QNN::createFoldBodyQDQRoundtripPass());
		passManager.addPass(QNN::createConvertLinalgToQNNPass());
		passManager.addPass(createEraseQNNVariantBodyPass());
	}

	void buildLinkingPassPipeline(OpPassManager &passManager) final {}

	LogicalResult serializeExecutable(const SerializationOptions &serOptions,
		IREE::HAL::ExecutableVariantOp variantOp,
		OpBuilder &executableBuilder) final {
		// In-compiler codegen path (Phase 4): when the EraseQNNVariantBodyPass
		// captured a `merlin.qnn_graph` attribute on the variant, we ship that
		// binary description directly as the executable's "qnn-graph" payload.
		// The runtime side (qnn_graph_builder.c) parses it and JIT-builds the
		// QNN graph at load time. No on-board build needed for GPU/HTP; HTA
		// flows through the same binary description but the runtime branches
		// to qnn-context-binary-generator on board.
		if (auto qnnGraphAttr = variantOp->getAttrOfType<DenseIntElementsAttr>(
				"merlin.qnn_graph")) {
			// Suffix the format string with the QNN backend ("hta"/"gpu"/"htp")
			// so the runtime executable cache's can_prepare_format filter
			// rejects variants belonging to other backends. Without this, all
			// QNN variants share format "qnn-graph" and the compiler-emitted
			// __init initializer eagerly tries to load every variant on every
			// QNN-bound device — which fails when e.g. HTA tries to load an
			// fp16 variant intended for the Adreno GPU.
			std::string formatStr = "qnn-graph";
			if (auto cfg = variantOp.getTarget().getConfiguration()) {
				if (auto be = cfg.getAs<StringAttr>("qnn_backend")) {
					formatStr = ("qnn-graph-" + be.getValue()).str();
				}
			}
			auto binaryOp = IREE::HAL::ExecutableBinaryOp::create(
				executableBuilder, variantOp.getLoc(), variantOp.getSymName(),
				/*format=*/executableBuilder.getStringAttr(formatStr),
				qnnGraphAttr);
			binaryOp.setMimeTypeAttr(
				executableBuilder.getStringAttr("application/octet-stream"));
			llvm::errs() << "[qnn-target] embedded "
						 << qnnGraphAttr.getNumElements()
						 << " bytes of in-compiler " << formatStr
						 << " for variant '" << variantOp.getName() << "'\n";
			return success();
		}

		// Legacy manifest-keyed path (kept for fixtures that still use the
		// pre-built `.qnn-ctx` blobs from `kernels/core/precompile.py`).
		auto manifest = loadManifest(options.manifestPath);

		// One entry per executable.export. Concatenate their .qnn-ctx blobs
		// (currently we expect only a single export per QNN variant; the
		// manifest format reflects that).
		SmallVector<int8_t> binaryData;
		bool emittedAny = false;
		bool usedPlaceholder = false;
		std::string usedSourcePath;

		for (auto exportOp :
			variantOp.getBlock().getOps<ExecutableExportOp>()) {
			auto symName = exportOp.getName().str();
			auto it = manifest.find(symName);
			std::optional<std::string> contents;
			if (it != manifest.end()) {
				contents = readBinaryFile(it->second);
				if (!contents) {
					return variantOp.emitOpError()
						<< "[qnn-target] manifest entry for '" << symName
						<< "' points at unreadable file: " << it->second;
				}
				usedSourcePath = it->second;
			} else {
				if (!options.allowPlaceholder) {
					// Phase 0 fail-fast: surface the source op kinds that the
					// in-compiler conversion (ConvertLinalgToQNN) didn't
					// match, so the user knows exactly which pattern to add.
					std::string opKindList;
					bool slowMemcpyOnly = false;
					if (auto attr = variantOp->getAttrOfType<ArrayAttr>(
							"merlin.qnn_unmatched_ops")) {
						SmallVector<StringRef> kinds;
						for (auto a : attr) {
							if (auto s = dyn_cast<StringAttr>(a)) {
								kinds.push_back(s.getValue());
								if (!opKindList.empty())
									opKindList += ", ";
								opKindList += s.str();
							}
						}
						// slow_memcpy / pad-via-store: body is JUST load/store
						// + maybe util.assume.int. Pure memory rearrangements.
						// Allow placeholder so the rest of the model compiles.
						if (!kinds.empty()) {
							slowMemcpyOnly =
								llvm::all_of(kinds, [](StringRef k) {
									return k ==
										"iree_tensor_ext.dispatch.tensor."
										"load" ||
										k ==
										"iree_tensor_ext.dispatch.tensor."
										"store" ||
										k == "util.assume.int";
								});
						}
					}
					if (slowMemcpyOnly) {
						llvm::errs() << "[qnn-target] NOTE: '" << symName
									 << "' is a passthrough memcpy; shipping "
										"placeholder.\n";
						contents = std::string(kPlaceholderSentinel);
						usedPlaceholder = true;
					} else {
						auto err = variantOp.emitOpError()
							<< "[qnn-target] export '" << symName
							<< "' has no QNN graph: ConvertLinalgToQNN did not "
							<< "match the source dispatch's body. ";
						if (!opKindList.empty()) {
							err << "Source op kinds in this dispatch: { "
								<< opKindList << " }. ";
						}
						err << "Add a pattern in "
							   "compiler/src/merlin/Dialect/QNN/"
							   "Transforms/ConvertLinalgToQNN.cpp covering the "
							   "above op(s), or supply a prebuilt .qnn-ctx via "
							   "--iree-hal-qnn-manifest=<path>. Pass "
							   "--iree-hal-qnn-allow-placeholder=true ONLY for "
							   "phase-inspection-only compiles (resulting VMFB "
							   "fails at runtime).";
						return err;
					}
				}
				llvm::errs() << "[qnn-target] WARNING: no manifest entry for '"
							 << symName
							 << "' — embedding placeholder. Resulting VMFB "
								"will fail at runtime.\n";
				contents = std::string(kPlaceholderSentinel);
				usedPlaceholder = true;
			}

			binaryData.reserve(binaryData.size() + contents->size());
			for (char c : *contents)
				binaryData.push_back(static_cast<int8_t>(c));
			emittedAny = true;
		}

		if (!emittedAny) {
			return variantOp.emitOpError()
				<< "QNN variant has no exports; nothing to serialize";
		}

		if (!serOptions.dumpBinariesPath.empty()) {
			SmallVector<char> charData(binaryData.size());
			for (size_t i = 0; i < binaryData.size(); ++i)
				charData[i] = static_cast<char>(binaryData[i]);
			dumpDataToPath<char>(serOptions.dumpBinariesPath,
				serOptions.dumpBaseName, variantOp.getName(), ".qnn-ctx",
				charData);
		}

		auto bufferAttr = DenseIntElementsAttr::get(
			VectorType::get({static_cast<int64_t>(binaryData.size())},
				IntegerType::get(executableBuilder.getContext(), 8)),
			std::move(binaryData));

		auto binaryOp = IREE::HAL::ExecutableBinaryOp::create(executableBuilder,
			variantOp.getLoc(), variantOp.getSymName(),
			variantOp.getTarget().getFormat(), bufferAttr);
		binaryOp.setMimeTypeAttr(
			executableBuilder.getStringAttr("application/octet-stream"));

		if (usedPlaceholder) {
			llvm::errs()
				<< "[qnn-target] NOTE: variant '" << variantOp.getName()
				<< "' was serialized with a placeholder; the resulting VMFB "
				   "is structurally complete but will fail at runtime when "
				   "QnnContext_createFromBinary tries to load it.\n";
		} else if (!usedSourcePath.empty()) {
			llvm::errs() << "[qnn-target] embedded " << binaryData.size()
						 << " bytes from " << usedSourcePath << " into '"
						 << variantOp.getName() << "'\n";
		}
		return success();
	}

  private:
	const QNNTargetOptions &options;
};

//===----------------------------------------------------------------------===//
// QNN target device
//===----------------------------------------------------------------------===//
// Produces #hal.device.target<"qnn", [...]> when the user passes
// --iree-hal-target-device=qnn.

class QNNTargetDevice final : public TargetDevice {
  public:
	explicit QNNTargetDevice(const QNNTargetOptions &options)
		: options(options) {}

	IREE::HAL::DeviceTargetAttr getDefaultDeviceTarget(MLIRContext *context,
		const TargetRegistry &targetRegistry) const final {
		Builder b(context);
		SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargets = {
			getQNNExecutableTarget(context, options.qnnBackend),
		};
		auto deviceConfig = b.getDictionaryAttr({});
		return b.getAttr<IREE::HAL::DeviceTargetAttr>(
			b.getStringAttr("qnn"), deviceConfig, executableTargets);
	}

  private:
	const QNNTargetOptions &options;
};

//===----------------------------------------------------------------------===//
// Plugin session
//===----------------------------------------------------------------------===//

struct QNNSession final
	: PluginSession<QNNSession, QNNTargetOptions,
		  // Explicit (was DefaultActivated): the plugin's
		  // ConvertLinalgToQNN pattern unconditionally rewrites every
		  // linalg.matmul to qnn.matmul, which collides with other
		  // accelerator plugins (e.g. gemmini) that need to recover the
		  // same op for their own codegen path. Activate only when
		  // explicitly requested via `--iree-plugin=hal_target_qnn`.
		  PluginActivationPolicy::Explicit> {
	static void registerPasses() {
		// Wire the merlin-convert-linalg-to-qnn and
		// merlin-qnn-legalize-layout-to-nhwc passes so iree-opt and the
		// implicit pass-pipeline parser can find them by name.
		mlir::iree_compiler::QNN::registerQNNPasses();
	}

	// Y-Ph7h: ensure that materialized conv-weight `arith.constant`s flow
	// INTO their consuming dispatch bodies. IREE's dispatch-creation step
	// clones a producer into a dispatch region only when its byte-length
	// is <= `iree-flow-inline-constants-max-byte-length` (default 256 B).
	// yolov8n conv weights run hundreds of KB to a few MB, so without
	// bumping this threshold the InlineConstantUtilGlobalsPass below would
	// merely move the constant from a `util.global` into func scope where
	// it would still become a dispatch binding (APP_WRITE) instead of an
	// inline `arith.constant` (STATIC). HTA's Conv2d validator demands
	// STATIC weights — anything else fails `graphAddNode rc=6000`.
	//
	// 64 MiB covers the largest layer in any of our target int8 models
	// (yolov8n max weight ≈ 2.3 MB; yolov8m ≈ 8 MB; mobilenet-v3 ≈ 1 MB).
	// Setting this from `globalInitialize` means it's in effect for every
	// downstream compile in the process; activating the QNN plugin means
	// opting in to the larger threshold.
	static void globalInitialize() {
		constexpr int kMaxInlineBytes = 64 * 1024 * 1024;
		auto &opts = llvm::cl::getRegisteredOptions();
		auto it = opts.find("iree-flow-inline-constants-max-byte-length");
		if (it == opts.end())
			return;
		auto *opt = static_cast<llvm::cl::opt<int> *>(it->second);
		if (opt->getValue() < kMaxInlineBytes)
			opt->setValue(kMaxInlineBytes);
	}

	void onRegisterDialects(DialectRegistry &registry) override {
		// Register the merlin QNN dialect so iree-compile can parse
		// `qnn.*` IR (Conv2d, ElementWiseNeuron, etc.) and the codegen
		// path can lower it to a serialized graph description / ctxbin.
		// See compiler/src/merlin/Dialect/QNN/IR/QNNDialect.{td,cpp}.
		registry.insert<mlir::iree_compiler::QNN::QNNDialect>();
	}

	// Convert linalg → qnn AFTER IREE's global optimization has run
	// fusion + constant folding + layout selection. Patterns anchor on
	// `linalg.generic` and inspect indexing-maps + body to identify
	// conv / pool / etc shapes — same approach IREE's own codegen
	// pipeline uses to detect what to tile. Running here means dispatch
	// creation will wrap each qnn.* op into its own hal.executable, and
	// only those hal.executables get the qnn target.
	void extendPreprocessingPassPipeline(OpPassManager &passManager) override {
		// Run LegalizeLayoutToNHWC at PREPROCESSING phase too — this
		// catches `linalg.conv_2d_nchw_fchw{,_q}` named ops BEFORE
		// global-opt generalizes them. yolov8n int8 has 64 named
		// quantized convs at this phase; running the named-op rewrite
		// here produces NHWC + boundary transposes that the subsequent
		// canonicalization can cancel through bias-broadcast and other
		// elementwise ops. The post-global-opt invocation below is a
		// fallback for any conv that survives in generalized form.
		//
		// NB: pass is OperationPass<> — add at module level, NOT nested in
		// func.func. Nesting in func.func causes the pass to never fire
		// for this OperationPass<> kind because the manifest registers it
		// at op-any scope, not at func.func scope.
		passManager.addPass(QNN::createLegalizeLayoutToNHWCPass());
	}
	void extendPostGlobalOptimizationPassPipeline(
		OpPassManager &passManager) override {
		// Step −1: inline constant-initialized `util.global` tensors back into
		// `arith.constant` at every load site. IREE's HoistIntoGlobals (run
		// during global-opt) parks every conv weight + bias as a private
		// global with `inlining_policy = #util.inline.never`; without this
		// pass each one stays a dispatch binding (APP_WRITE) and HTA rejects
		// the Conv2d at `graphAddNode rc=6000`. Combined with the
		// inline-constants-max-byte-length bump in `globalInitialize`,
		// dispatch creation then clones the resulting arith.constants into
		// the dispatch body where SerializeGraph's `extractConstantBytes`
		// emits them as STATIC tensors.
		passManager.addPass(QNN::createInlineConstantUtilGlobalsPass());

		// Step 0a: RewriteToNHWCBindings — propagate NHWC layout through
		// cross-dispatch activations so HTA dispatches don't need internal
		// qnn.transpose bridges. Step 0b: LowerNHWCCastsToTransposes —
		// materialize the unrealized_conversion_cast bridges into real
		// linalg.transpose ops so downstream codegen accepts the IR.
		passManager.addPass(QNN::createRewriteToNHWCBindingsPass());
		passManager.addPass(QNN::createLowerNHWCCastsToTransposesPass());

		// Step 1: stamp QDQ scale/zp onto the dequant/quant generics +
		// their producers as `merlin.qnn.{input,output}_{scale,zero_point}`
		// attrs. ConvertLinalgToQNN reads these attrs at lowering time so
		// SerializeGraph emits qk=1 with real per-tensor quant params on
		// every QNN op input/output. Universal across pt2e/TFLite/ONNX-
		// QDQ-decomposed inputs.
		passManager.addPass(QNN::createRewriteQDQToQuantUniformPass());
		// Step 2: NHWC layout legalization (fallback). Most convs are
		// already NHWC after the preprocessing-phase invocation above;
		// this catches generalized forms that escaped the named-op
		// rewrite.
		passManager.addNestedPass<func::FuncOp>(
			QNN::createLegalizeLayoutToNHWCPass());
	}

	void populateHALTargetDevices(IREE::HAL::TargetDeviceList &targets) {
		// #hal.device.target<"qnn", ...
		targets.add("qnn",
			[=]() { return std::make_shared<QNNTargetDevice>(options); });
	}

	void populateHALTargetBackends(IREE::HAL::TargetBackendList &targets) {
		// #hal.executable.target<"qnn", "qnn-context-binary", ...
		targets.add("qnn",
			[=]() { return std::make_shared<QNNTargetBackend>(options); });
	}
};

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL

extern "C" bool iree_register_compiler_plugin_hal_target_qnn(
	mlir::iree_compiler::PluginRegistrar *registrar) {
	registrar->registerPlugin<mlir::iree_compiler::IREE::HAL::QNNSession>(
		"hal_target_qnn");
	return true;
}

IREE_DEFINE_COMPILER_OPTION_FLAGS(
	mlir::iree_compiler::IREE::HAL::QNNTargetOptions);
