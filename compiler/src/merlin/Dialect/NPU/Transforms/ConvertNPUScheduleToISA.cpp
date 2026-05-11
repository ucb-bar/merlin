#include "compiler/src/merlin/Dialect/NPU/Transforms/Passes.h"

#include "compiler/src/merlin/Dialect/NPU/IR/NPUISADialect.h"
#include "compiler/src/merlin/Dialect/NPU/IR/NPUScheduleDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#include <memory>
#include <string>
#include <utility>

namespace mlir::iree_compiler::NPU {
namespace {

static IntegerAttr i64(PatternRewriter &rewriter, int64_t v) {
	return rewriter.getI64IntegerAttr(v);
}

struct NativeArg {
	std::string name;
	int64_t value = 0;
};

struct NativeInstruction {
	std::string mnemonic;
	SmallVector<NativeArg> args;
};

// A patch point records a scalar register holding a DRAM address (for one of
// the kernel's inputs, outputs, or constants) and the chain of instruction
// indices that set it up. The compiler rewrites these per invocation so each
// invocation writes to a unique DRAM region.
struct NativePatchPoint {
	std::string role; // dram_in_N, dram_out_N, transfer_size
	int64_t registerIdx = 0;
	int64_t originalValue = 0;
	SmallVector<int64_t> instructionIndices; // addi/lui ops to rewrite
};

struct NativeKernel {
	std::string symbolPrefix;
	SmallVector<NativeInstruction> instructions;
	SmallVector<NativePatchPoint> patchPoints;
};

struct NativeKernelManifest {
	llvm::StringMap<NativeKernel> kernels;
	SmallVector<std::pair<std::string, std::string>> symbolPrefixes;

	const NativeKernel *lookupKernel(StringRef name) const {
		auto it = kernels.find(name);
		if (it == kernels.end()) {
			return nullptr;
		}
		return &it->second;
	}

	const NativeKernel *lookupSymbol(StringRef symbol) const {
		for (const auto &entry : symbolPrefixes) {
			if (symbol.starts_with(entry.first)) {
				return lookupKernel(entry.second);
			}
		}
		// Compatibility with the older ukernel symbol emitted before the
		// manifest used the shorter npu_uk_attention prefix.
		if (symbol.starts_with("npu_uk_gemma_attention_")) {
			return lookupKernel("attention");
		}
		return nullptr;
	}
};

static LogicalResult loadNativeKernelManifest(
	StringRef path, NativeKernelManifest &manifest, Location loc) {
	auto bufferOrError = llvm::MemoryBuffer::getFile(path);
	if (!bufferOrError) {
		emitError(loc) << "failed to read NPU kernel manifest '" << path
					   << "': " << bufferOrError.getError().message();
		return failure();
	}

	auto parsed = llvm::json::parse(bufferOrError.get()->getBuffer());
	if (!parsed) {
		emitError(loc) << "failed to parse NPU kernel manifest '" << path
					   << "': " << llvm::toString(parsed.takeError());
		return failure();
	}

	auto *root = parsed->getAsObject();
	if (!root) {
		emitError(loc) << "NPU kernel manifest root must be a JSON object";
		return failure();
	}
	auto *kernels = root->getObject("kernels");
	if (!kernels) {
		emitError(loc) << "NPU kernel manifest is missing a 'kernels' object";
		return failure();
	}

	for (const auto &kernelEntry : *kernels) {
		std::string kernelName = kernelEntry.first.str();
		auto *kernelObj = kernelEntry.second.getAsObject();
		if (!kernelObj) {
			emitError(loc) << "kernel entry '" << kernelName
						   << "' must be an object";
			return failure();
		}

		NativeKernel kernel;
		if (auto prefix = kernelObj->getString("symbol_prefix")) {
			kernel.symbolPrefix = prefix->str();
		}

		auto *instructions = kernelObj->getArray("instructions");
		if (!instructions) {
			emitError(loc) << "kernel entry '" << kernelName
						   << "' is missing an 'instructions' array";
			return failure();
		}

		for (const llvm::json::Value &instructionValue : *instructions) {
			auto *instructionObj = instructionValue.getAsObject();
			if (!instructionObj) {
				emitError(loc) << "kernel entry '" << kernelName
							   << "' contains a non-object instruction";
				return failure();
			}
			auto mnemonic = instructionObj->getString("mnemonic");
			if (!mnemonic) {
				emitError(loc) << "kernel entry '" << kernelName
							   << "' has an instruction without a mnemonic";
				return failure();
			}

			NativeInstruction instruction;
			instruction.mnemonic = mnemonic->str();
			if (auto *args = instructionObj->getObject("args")) {
				for (const auto &arg : *args) {
					std::string argName = arg.first.str();
					auto integer = arg.second.getAsInteger();
					if (!integer) {
						emitError(loc)
							<< "kernel entry '" << kernelName
							<< "' instruction '" << instruction.mnemonic
							<< "' has non-integer arg '" << argName << "'";
						return failure();
					}
					instruction.args.push_back(
						NativeArg{std::move(argName), *integer});
				}
			}
			kernel.instructions.push_back(std::move(instruction));
		}

		// patch_points is optional — kernels without it still lower, they just
		// get the kernel's original hardcoded addresses on every invocation.
		// Run annotate_kernel_patch_points.py to populate them.
		if (auto *patchPointsArr = kernelObj->getArray("patch_points")) {
			for (const llvm::json::Value &ppv : *patchPointsArr) {
				auto *ppObj = ppv.getAsObject();
				if (!ppObj)
					continue;
				NativePatchPoint pp;
				if (auto role = ppObj->getString("role"))
					pp.role = role->str();
				if (auto reg = ppObj->getInteger("register"))
					pp.registerIdx = *reg;
				if (auto orig = ppObj->getInteger("original_value"))
					pp.originalValue = *orig;
				if (auto *instArr = ppObj->getArray("instructions")) {
					for (const llvm::json::Value &iv : *instArr) {
						if (auto idx = iv.getAsInteger())
							pp.instructionIndices.push_back(*idx);
					}
				}
				kernel.patchPoints.push_back(std::move(pp));
			}
		}

		if (!kernel.symbolPrefix.empty()) {
			manifest.symbolPrefixes.push_back(
				{kernel.symbolPrefix, kernelName});
		}
		manifest.kernels.try_emplace(kernelName, std::move(kernel));
	}

	return success();
}

// Per-invocation DRAM region layout. Each ukernel_launch gets a unique slice
// of DRAM for its inputs/outputs so back-to-back invocations don't stomp
// each other. The baseline strategy is `dramStride`-byte slabs indexed by a
// monotonic invocation counter: role address = baseAddr + invocation*stride.
// For a first cut this is deterministic but oversized; a proper liveness
// analysis would pack tightly.
// Per-invocation DRAM layout. First-cut strategy is tight: stride per
// invocation is 0x100 (256 B), input region starts at 0, output region at
// 0x400. This keeps all addresses within a single lui+addi pair, which most
// kernels can represent via their annotated patch-point chains.
//
// Bigger strides require every kernel to emit lui+addi pairs for every DRAM
// register — we can bump this once the manifest's chains are all length ≥ 2
// (or once we teach the rewriter to insert instructions, shifting indices).
struct NativeKernelInvocation {
	int64_t invocationIndex = 0;
	int64_t dramStride = 0x100;
	int64_t inputBase = 0x0;
	int64_t outputBase = 0x400;
};

// Patches an imm field of a (lui/addi) chain so the chain computes
// `target`. Works when:
//   - The chain has ≥ 2 instructions: repurposes the last two as lui+addi.
//   - The chain has 1 instruction and target fits in 12-bit signed: writes
//     a single `addi rd=X, rs1=0, imm=target`.
// Otherwise leaves the chain unchanged (returns false); caller may warn.
static bool rewritePatchChain(SmallVectorImpl<NativeInstruction> &insts,
	const NativePatchPoint &pp, int64_t target) {
	if (pp.instructionIndices.empty())
		return false;
	bool fitsImm12 = (target >= -2048 && target <= 2047);
	int64_t lastIdx = pp.instructionIndices.back();
	if (lastIdx < 0 || lastIdx >= (int64_t)insts.size())
		return false;

	// Case 1: 1-instruction chain + target fits — direct rewrite.
	if (pp.instructionIndices.size() == 1 && fitsImm12) {
		insts[lastIdx].mnemonic = "addi";
		insts[lastIdx].args.clear();
		insts[lastIdx].args.push_back({"rd", pp.registerIdx});
		insts[lastIdx].args.push_back({"rs1", 0});
		insts[lastIdx].args.push_back({"rs2", 0});
		insts[lastIdx].args.push_back({"imm", target});
		return true;
	}

	// Case 2: ≥2-instruction chain — nop earlier entries, place lui+addi at
	// the tail. Handles both small and large targets uniformly.
	if (pp.instructionIndices.size() >= 2) {
		// Nop all but the last two entries.
		for (int64_t i = 0; i + 2 < (int64_t)pp.instructionIndices.size();
			 ++i) {
			int64_t nopIdx = pp.instructionIndices[i];
			if (nopIdx < 0 || nopIdx >= (int64_t)insts.size())
				continue;
			insts[nopIdx].mnemonic = "addi";
			insts[nopIdx].args.clear();
			insts[nopIdx].args.push_back({"rd", 0});
			insts[nopIdx].args.push_back({"rs1", 0});
			insts[nopIdx].args.push_back({"rs2", 0});
			insts[nopIdx].args.push_back({"imm", 0});
		}
		int64_t luiIdx =
			pp.instructionIndices[pp.instructionIndices.size() - 2];
		int64_t addiIdx = pp.instructionIndices.back();
		int64_t upper = (target + 0x800) >> 12;
		int64_t lower = target - (upper << 12);
		insts[luiIdx].mnemonic = "lui";
		insts[luiIdx].args.clear();
		insts[luiIdx].args.push_back({"rd", pp.registerIdx});
		insts[luiIdx].args.push_back({"rs1", 0});
		insts[luiIdx].args.push_back({"rs2", 0});
		insts[luiIdx].args.push_back({"imm", upper});
		insts[addiIdx].mnemonic = "addi";
		insts[addiIdx].args.clear();
		insts[addiIdx].args.push_back({"rd", pp.registerIdx});
		insts[addiIdx].args.push_back({"rs1", pp.registerIdx});
		insts[addiIdx].args.push_back({"rs2", 0});
		insts[addiIdx].args.push_back({"imm", lower});
		return true;
	}

	// Case 3: 1-instruction chain but target is out of 12-bit range. Cannot
	// represent without inserting an instruction (which would shift all
	// subsequent indices). Leave unchanged; caller decides whether to warn.
	return false;
}

// Clone `kernel.instructions` and apply patch-point rewrites for this
// invocation. Each input/output role is assigned a distinct DRAM region
// derived from (invocation.inputBase, invocation.outputBase,
// invocation.dramStride).
static SmallVector<NativeInstruction> patchedInstructionsForInvocation(
	const NativeKernel &kernel, const NativeKernelInvocation &inv) {
	SmallVector<NativeInstruction> insts(
		kernel.instructions.begin(), kernel.instructions.end());
	int numIn = 0, numOut = 0;
	auto startsWith = [](const std::string &s, const char *p) {
		return s.rfind(p, 0) == 0;
	};
	for (const NativePatchPoint &pp : kernel.patchPoints) {
		int64_t target = pp.originalValue; // default: leave unchanged
		if (startsWith(pp.role, "dram_in_")) {
			target = inv.inputBase + inv.invocationIndex * inv.dramStride +
				numIn * (inv.dramStride / 4);
			++numIn;
		} else if (startsWith(pp.role, "dram_out_")) {
			target = inv.outputBase + inv.invocationIndex * inv.dramStride +
				numOut * (inv.dramStride / 4);
			++numOut;
		} else {
			continue; // transfer_size — not patched
		}
		rewritePatchChain(insts, pp, target);
	}
	return insts;
}

static void emitNativeKernel(Location loc, PatternRewriter &rewriter,
	const NativeKernel &kernel, const NativeKernelInvocation &inv) {
	// If the kernel has no patch_points annotations, fall back to emitting
	// the original hardcoded instructions (backward compatible with older
	// manifests). Otherwise rewrite addresses per invocation.
	SmallVector<NativeInstruction> insts = kernel.patchPoints.empty()
		? SmallVector<NativeInstruction>(
			  kernel.instructions.begin(), kernel.instructions.end())
		: patchedInstructionsForInvocation(kernel, inv);
	for (const NativeInstruction &instruction : insts) {
		SmallVector<NamedAttribute> args;
		for (const NativeArg &arg : instruction.args) {
			args.push_back(rewriter.getNamedAttr(
				arg.name, rewriter.getI64IntegerAttr(arg.value)));
		}
		rewriter.create<NPUISA::NativeInstOp>(loc,
			rewriter.getStringAttr(instruction.mnemonic),
			rewriter.getDictionaryAttr(args));
	}
}

static Value createPlaceholderResult(Location loc, Type resultType,
	ValueRange inputs, PatternRewriter &rewriter) {
	auto tensorType = dyn_cast<RankedTensorType>(resultType);
	if (!tensorType) {
		return {};
	}

	SmallVector<Value> dynamicDims;
	for (auto [index, dim] : llvm::enumerate(tensorType.getShape())) {
		if (!ShapedType::isDynamic(dim)) {
			continue;
		}
		Value dynamicDim;
		for (Value input : inputs) {
			auto inputType = dyn_cast<RankedTensorType>(input.getType());
			if (inputType &&
				inputType.getRank() > static_cast<int64_t>(index)) {
				dynamicDim = rewriter.create<tensor::DimOp>(loc, input, index);
				break;
			}
		}
		if (!dynamicDim) {
			dynamicDim = rewriter.create<arith::ConstantIndexOp>(loc, 1);
		}
		dynamicDims.push_back(dynamicDim);
	}

	return rewriter.create<tensor::EmptyOp>(
		loc, tensorType.getShape(), tensorType.getElementType(), dynamicDims);
}

static LogicalResult replaceWithNativeKernel(Operation *op,
	StringRef kernelName, Type resultType, ValueRange inputs,
	const NativeKernelManifest *manifest, bool strictCoverage,
	PatternRewriter &rewriter, int64_t *invocationCounter) {
	if (!manifest) {
		return failure();
	}
	const NativeKernel *kernel = manifest->lookupKernel(kernelName);
	if (!kernel) {
		if (strictCoverage) {
			return rewriter.notifyMatchFailure(
				op, "missing native kernel manifest entry");
		}
		return failure();
	}

	NativeKernelInvocation inv;
	if (invocationCounter) {
		inv.invocationIndex = (*invocationCounter)++;
	}
	emitNativeKernel(op->getLoc(), rewriter, *kernel, inv);
	Value placeholder =
		createPlaceholderResult(op->getLoc(), resultType, inputs, rewriter);
	if (!placeholder) {
		return rewriter.notifyMatchFailure(
			op, "native kernel lowering requires a ranked tensor result");
	}
	rewriter.replaceOp(op, placeholder);
	return success();
}

static void emitMatmulWeightLoad(
	Location loc, PatternRewriter &rewriter, bool matmulUseMxu1Weights) {
	if (matmulUseMxu1Weights) {
		rewriter.create<NPUISA::DmaLoadMxu1Op>(loc, i64(rewriter, 1),
			i64(rewriter, 2048), i64(rewriter, 512), i64(rewriter, 0));
	} else {
		rewriter.create<NPUISA::DmaLoadMxu0Op>(loc, i64(rewriter, 1),
			i64(rewriter, 2048), i64(rewriter, 512), i64(rewriter, 0));
	}
}

static Value emitMatmulSkeleton(Location loc, Value lhs, Value rhs,
	Type resultType, PatternRewriter &rewriter, bool matmulUseMxu1Weights) {
	// Deterministic ISA skeleton for a single matmul tile / ukernel launch.
	rewriter.create<NPUISA::DmaLoadOp>(loc, i64(rewriter, 2), i64(rewriter, 0),
		i64(rewriter, 2048), i64(rewriter, 0));
	rewriter.create<NPUISA::DmaWaitOp>(loc, i64(rewriter, 0));
	emitMatmulWeightLoad(loc, rewriter, matmulUseMxu1Weights);
	rewriter.create<NPUISA::DmaWaitOp>(loc, i64(rewriter, 0));
	auto matmul = rewriter.create<NPUISA::MatmulMxu0Op>(loc, resultType, lhs,
		rhs, i64(rewriter, 0), i64(rewriter, 2), i64(rewriter, 1));
	return matmul.getResult();
}

static Value emitSoftmaxVectorChain(
	Location loc, Value input, PatternRewriter &rewriter) {
	auto scaled = rewriter.create<NPUISA::VMulOp>(loc, input.getType(), input,
		input, i64(rewriter, 4), i64(rewriter, 3), i64(rewriter, 2));
	auto exp = rewriter.create<NPUISA::VExpOp>(loc, input.getType(),
		scaled.getResult(), i64(rewriter, 5), i64(rewriter, 4));
	auto sum = rewriter.create<NPUISA::VReduceSumOp>(loc, input.getType(),
		exp.getResult(), i64(rewriter, 6), i64(rewriter, 5));
	auto inv = rewriter.create<NPUISA::VRcpOp>(loc, input.getType(),
		sum.getResult(), i64(rewriter, 7), i64(rewriter, 6));
	auto soft = rewriter.create<NPUISA::VMulOp>(loc, input.getType(),
		exp.getResult(), inv.getResult(), i64(rewriter, 8), i64(rewriter, 5),
		i64(rewriter, 7));
	return soft.getResult();
}

struct LowerScheduleMatmulToISAPattern
	: public OpRewritePattern<NPUSchedule::MatmulTileOp> {
	LowerScheduleMatmulToISAPattern(MLIRContext *context,
		const NPULoweringOptions &options,
		std::shared_ptr<const NativeKernelManifest> nativeManifest,
		std::shared_ptr<int64_t> invocationCounter)
		: OpRewritePattern<NPUSchedule::MatmulTileOp>(context),
		  options(options), nativeManifest(std::move(nativeManifest)),
		  invocationCounter(std::move(invocationCounter)) {}

	LogicalResult matchAndRewrite(NPUSchedule::MatmulTileOp op,
		PatternRewriter &rewriter) const override {
		if (options.nativeKernelLowering) {
			return replaceWithNativeKernel(op.getOperation(), "matmul",
				op.getResult().getType(), ValueRange{op.getLhs(), op.getRhs()},
				nativeManifest.get(), options.strictNativeKernelCoverage,
				rewriter, invocationCounter.get());
		}

		Value result = emitMatmulSkeleton(op.getLoc(), op.getLhs(), op.getRhs(),
			op.getResult().getType(), rewriter, options.matmulUseMxu1Weights);
		rewriter.replaceOp(op, result);
		return success();
	}

	NPULoweringOptions options;
	std::shared_ptr<const NativeKernelManifest> nativeManifest;
	std::shared_ptr<int64_t> invocationCounter;
};

struct LowerScheduleSoftmaxToISAPattern
	: public OpRewritePattern<NPUSchedule::SoftmaxFragmentOp> {
	LowerScheduleSoftmaxToISAPattern(MLIRContext *context,
		const NPULoweringOptions &options,
		std::shared_ptr<const NativeKernelManifest> nativeManifest,
		std::shared_ptr<int64_t> invocationCounter)
		: OpRewritePattern<NPUSchedule::SoftmaxFragmentOp>(context),
		  options(options), nativeManifest(std::move(nativeManifest)),
		  invocationCounter(std::move(invocationCounter)) {}

	LogicalResult matchAndRewrite(NPUSchedule::SoftmaxFragmentOp op,
		PatternRewriter &rewriter) const override {
		if (options.nativeKernelLowering) {
			return replaceWithNativeKernel(op.getOperation(), "softmax",
				op.getResult().getType(), ValueRange{op.getInput()},
				nativeManifest.get(), options.strictNativeKernelCoverage,
				rewriter, invocationCounter.get());
		}

		Value result =
			emitSoftmaxVectorChain(op.getLoc(), op.getInput(), rewriter);
		rewriter.replaceOp(op, result);
		return success();
	}

	NPULoweringOptions options;
	std::shared_ptr<const NativeKernelManifest> nativeManifest;
	std::shared_ptr<int64_t> invocationCounter;
};

struct LowerScheduleUKernelToISAPattern
	: public OpRewritePattern<NPUSchedule::UKernelLaunchOp> {
	LowerScheduleUKernelToISAPattern(MLIRContext *context,
		const NPULoweringOptions &options,
		std::shared_ptr<const NativeKernelManifest> nativeManifest,
		std::shared_ptr<int64_t> invocationCounter)
		: OpRewritePattern<NPUSchedule::UKernelLaunchOp>(context),
		  options(options), nativeManifest(std::move(nativeManifest)),
		  invocationCounter(std::move(invocationCounter)) {}

	LogicalResult matchAndRewrite(NPUSchedule::UKernelLaunchOp op,
		PatternRewriter &rewriter) const override {
		llvm::StringRef symbol = op.getSymbol();
		auto inputs = op.getInputs();

		if (options.nativeKernelLowering) {
			const NativeKernel *kernel =
				nativeManifest ? nativeManifest->lookupSymbol(symbol) : nullptr;
			if (kernel) {
				NativeKernelInvocation inv;
				if (invocationCounter) {
					inv.invocationIndex = (*invocationCounter)++;
				}
				emitNativeKernel(op.getLoc(), rewriter, *kernel, inv);
				Value placeholder = createPlaceholderResult(
					op.getLoc(), op.getResult().getType(), inputs, rewriter);
				if (!placeholder) {
					return rewriter.notifyMatchFailure(op,
						"native kernel lowering requires a ranked tensor "
						"result");
				}
				rewriter.replaceOp(op, placeholder);
				return success();
			}
			if (options.strictNativeKernelCoverage) {
				return rewriter.notifyMatchFailure(
					op, "missing native kernel for ukernel symbol");
			}
		}

		if (symbol.starts_with("npu_uk_matmul_")) {
			if (inputs.size() < 2) {
				return rewriter.notifyMatchFailure(
					op, "matmul ukernel expects >=2 tensor inputs");
			}
			Value result = emitMatmulSkeleton(op.getLoc(), inputs[0], inputs[1],
				op.getResult().getType(), rewriter,
				options.matmulUseMxu1Weights);
			rewriter.replaceOp(op, result);
			return success();
		}

		if (symbol.starts_with("npu_uk_gemma_mlp_")) {
			if (inputs.size() < 2) {
				return rewriter.notifyMatchFailure(
					op, "gemma_mlp ukernel expects >=2 tensor inputs");
			}
			// Mirrors model_npu/configs/programs/gemma_mlp.py shape.
			rewriter.create<NPUISA::DmaLoadMxu1Op>(op.getLoc(),
				i64(rewriter, 0), i64(rewriter, 0x0000), i64(rewriter, 512),
				i64(rewriter, 0));
			rewriter.create<NPUISA::DmaLoadMxu1Op>(op.getLoc(),
				i64(rewriter, 1), i64(rewriter, 0x0200), i64(rewriter, 512),
				i64(rewriter, 1));
			rewriter.create<NPUISA::DmaWaitOp>(op.getLoc(), i64(rewriter, 0));
			rewriter.create<NPUISA::DmaWaitOp>(op.getLoc(), i64(rewriter, 1));
			rewriter.create<NPUISA::DmaLoadOp>(op.getLoc(), i64(rewriter, 0),
				i64(rewriter, 0x2000), i64(rewriter, 2048), i64(rewriter, 2));
			rewriter.create<NPUISA::DmaWaitOp>(op.getLoc(), i64(rewriter, 2));

			auto gate = rewriter.create<NPUISA::MatmulMxu0Op>(op.getLoc(),
				op.getResult().getType(), inputs[0], inputs[1],
				i64(rewriter, 1), i64(rewriter, 0), i64(rewriter, 0));
			auto up = rewriter.create<NPUISA::MatmulMxu0Op>(op.getLoc(),
				op.getResult().getType(), inputs[0], inputs[1],
				i64(rewriter, 2), i64(rewriter, 0), i64(rewriter, 1));
			auto fused = rewriter.create<NPUISA::VMulOp>(op.getLoc(),
				op.getResult().getType(), gate.getResult(), up.getResult(),
				i64(rewriter, 6), i64(rewriter, 1), i64(rewriter, 2));

			rewriter.create<NPUISA::DmaStoreOp>(op.getLoc(), fused.getResult(),
				i64(rewriter, 6), i64(rewriter, 0x3000), i64(rewriter, 2048),
				i64(rewriter, 0));
			rewriter.create<NPUISA::DmaWaitOp>(op.getLoc(), i64(rewriter, 0));

			rewriter.replaceOp(op, fused.getResult());
			return success();
		}

		if (symbol.starts_with("npu_uk_gemma_attention_")) {
			if (inputs.size() < 2) {
				return rewriter.notifyMatchFailure(
					op, "gemma_attention ukernel expects >=2 tensor inputs");
			}
			// Mirrors model_npu/configs/programs/gemma_attention.py shape.
			rewriter.create<NPUISA::DmaLoadMxu1Op>(op.getLoc(),
				i64(rewriter, 0), i64(rewriter, 0x2000), i64(rewriter, 256),
				i64(rewriter, 0));
			rewriter.create<NPUISA::DmaLoadMxu1Op>(op.getLoc(),
				i64(rewriter, 1), i64(rewriter, 0x3000), i64(rewriter, 256),
				i64(rewriter, 1));
			rewriter.create<NPUISA::DmaWaitOp>(op.getLoc(), i64(rewriter, 0));
			rewriter.create<NPUISA::DmaWaitOp>(op.getLoc(), i64(rewriter, 1));
			rewriter.create<NPUISA::DmaLoadOp>(op.getLoc(), i64(rewriter, 0),
				i64(rewriter, 0x0000), i64(rewriter, 1024), i64(rewriter, 2));
			rewriter.create<NPUISA::DmaLoadOp>(op.getLoc(), i64(rewriter, 2),
				i64(rewriter, 0x4000), i64(rewriter, 2048), i64(rewriter, 0));
			rewriter.create<NPUISA::DmaWaitOp>(op.getLoc(), i64(rewriter, 2));
			rewriter.create<NPUISA::DmaWaitOp>(op.getLoc(), i64(rewriter, 0));

			auto scores = rewriter.create<NPUISA::MatmulMxu0Op>(op.getLoc(),
				op.getResult().getType(), inputs[0], inputs[1],
				i64(rewriter, 3), i64(rewriter, 0), i64(rewriter, 0));
			auto softmax = emitSoftmaxVectorChain(
				op.getLoc(), scores.getResult(), rewriter);
			auto output = rewriter.create<NPUISA::MatmulMxu0Op>(op.getLoc(),
				op.getResult().getType(), softmax, inputs[1], i64(rewriter, 9),
				i64(rewriter, 8), i64(rewriter, 1));

			rewriter.create<NPUISA::DmaStoreOp>(op.getLoc(), output.getResult(),
				i64(rewriter, 9), i64(rewriter, 0x5000), i64(rewriter, 2048),
				i64(rewriter, 1));
			rewriter.create<NPUISA::DmaWaitOp>(op.getLoc(), i64(rewriter, 1));

			rewriter.replaceOp(op, output.getResult());
			return success();
		}

		if (!options.allowUnknownUkernelFallback) {
			return rewriter.notifyMatchFailure(op,
				"unknown ukernel symbol family with "
				"fallback disabled");
		}

		if (inputs.size() < 2) {
			return rewriter.notifyMatchFailure(
				op, "unknown ukernel symbol requires at least 2 tensor inputs");
		}
		Value result = emitMatmulSkeleton(op.getLoc(), inputs[0], inputs[1],
			op.getResult().getType(), rewriter, options.matmulUseMxu1Weights);
		rewriter.replaceOp(op, result);
		return success();
	}

	NPULoweringOptions options;
	std::shared_ptr<const NativeKernelManifest> nativeManifest;
	std::shared_ptr<int64_t> invocationCounter;
};

struct ConvertNPUScheduleToISAPass
	: public PassWrapper<ConvertNPUScheduleToISAPass, OperationPass<ModuleOp>> {
	MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertNPUScheduleToISAPass)

	explicit ConvertNPUScheduleToISAPass(const NPULoweringOptions &options)
		: options(options) {}
	ConvertNPUScheduleToISAPass() = default;

	StringRef getArgument() const final {
		return "convert-npu-schedule-to-isa";
	}
	StringRef getDescription() const final {
		return "Lower npu_schedule matmul/ukernel ops to npu_isa ops, or to "
			   "native manifest instructions when enabled";
	}

	void getDependentDialects(DialectRegistry &registry) const override {
		registry.insert<NPUSchedule::NPUScheduleDialect>();
		registry.insert<NPUISA::NPUISADialect>();
		registry.insert<arith::ArithDialect>();
		registry.insert<tensor::TensorDialect>();
	}

	void runOnOperation() override {
		std::shared_ptr<NativeKernelManifest> nativeManifest;
		if (options.nativeKernelLowering) {
			if (options.kernelManifestPath.empty()) {
				emitError(getOperation().getLoc())
					<< "native NPU kernel lowering requires "
					   "--iree-npu-kernel-manifest";
				signalPassFailure();
				return;
			}
			nativeManifest = std::make_shared<NativeKernelManifest>();
			if (failed(loadNativeKernelManifest(options.kernelManifestPath,
					*nativeManifest, getOperation().getLoc()))) {
				signalPassFailure();
				return;
			}
		}

		// Shared per-module invocation counter for native-kernel address
		// patching. Each emitted kernel instance gets a distinct index so
		// the manifest's patch_points can be rewritten to write to disjoint
		// DRAM regions.
		auto invocationCounter = std::make_shared<int64_t>(0);

		RewritePatternSet patterns(&getContext());
		patterns.add<LowerScheduleMatmulToISAPattern>(
			&getContext(), options, nativeManifest, invocationCounter);
		patterns.add<LowerScheduleSoftmaxToISAPattern>(
			&getContext(), options, nativeManifest, invocationCounter);
		patterns.add<LowerScheduleUKernelToISAPattern>(
			&getContext(), options, nativeManifest, invocationCounter);

		if (failed(
				applyPatternsGreedily(getOperation(), std::move(patterns)))) {
			signalPassFailure();
			return;
		}

		if (options.nativeKernelLowering &&
			options.strictNativeKernelCoverage) {
			bool hasRemainingScheduleOps = false;
			getOperation().walk([&](Operation *op) {
				if (isa<NPUSchedule::MatmulTileOp,
						NPUSchedule::SoftmaxFragmentOp,
						NPUSchedule::UKernelLaunchOp>(op)) {
					op->emitError()
						<< "remaining npu_schedule op after strict native "
						   "kernel lowering";
					hasRemainingScheduleOps = true;
				}
			});
			if (hasRemainingScheduleOps) {
				signalPassFailure();
			}
		}
	}

	NPULoweringOptions options;
};

} // namespace

std::unique_ptr<Pass> createConvertNPUScheduleToISAPass(
	const NPULoweringOptions &options) {
	return std::make_unique<ConvertNPUScheduleToISAPass>(options);
}

std::unique_ptr<Pass> createConvertNPUScheduleToISAPass() {
	return std::make_unique<ConvertNPUScheduleToISAPass>();
}

void registerConvertNPUScheduleToISAPass() {
	PassRegistration<ConvertNPUScheduleToISAPass>();
}

} // namespace mlir::iree_compiler::NPU
