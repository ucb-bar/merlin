#include "compiler/plugins/target/QNN/Codegen/SerializeGraph.h"

#include "compiler/src/merlin/Dialect/QNN/IR/QNNDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/EndianStream.h"

namespace mlir::iree_compiler::QNN::Codegen {

namespace {

// QNN op-kind enum stable across versions of the wire format. Mirrors the
// ordering of qnn.* ops we emit. Runtime decoder uses the same enum.
enum class OpKind : uint32_t {
	Conv2d = 1,
	DepthwiseConv2d = 2,
	FullyConnected = 3,
	MatMul = 4,
	ElementWiseNeuron = 5,
	ElementWiseBinary = 6,
	PoolMax2d = 7,
	PoolAvg2d = 8,
	Concat = 9,
	Reshape = 10,
	Transpose = 11,
	Quantize = 12,
	Dequantize = 13,
	// Phase 4b additions:
	Pad = 14,
	Softmax = 15,
	Reduce = 16,
};

class Writer {
  public:
	explicit Writer(std::vector<int8_t> &out) : out_(out) {}

	void u8(uint8_t v) {
		out_.push_back(static_cast<int8_t>(v));
	}
	void u32(uint32_t v) {
		out_.push_back(static_cast<int8_t>(v & 0xFF));
		out_.push_back(static_cast<int8_t>((v >> 8) & 0xFF));
		out_.push_back(static_cast<int8_t>((v >> 16) & 0xFF));
		out_.push_back(static_cast<int8_t>((v >> 24) & 0xFF));
	}
	void u16(uint16_t v) {
		out_.push_back(static_cast<int8_t>(v & 0xFF));
		out_.push_back(static_cast<int8_t>((v >> 8) & 0xFF));
	}
	void f32(float v) {
		uint32_t bits;
		std::memcpy(&bits, &v, sizeof(bits));
		u32(bits);
	}
	void blob(const int8_t *p, size_t n) {
		out_.insert(out_.end(), p, p + n);
	}

  private:
	std::vector<int8_t> &out_;
};

// Tensor table — maps SSA Value to a stable u32 tensor id; tracks which
// tensors must be emitted into the binary header.
struct TensorTable {
	llvm::DenseMap<Value, uint32_t> ids;
	SmallVector<Value, 32> ordered;

	uint32_t intern(Value v) {
		auto it = ids.find(v);
		if (it != ids.end())
			return it->second;
		uint32_t id = static_cast<uint32_t>(ordered.size());
		ids[v] = id;
		ordered.push_back(v);
		return id;
	}
};

// Per-tensor quant params override harvested from qnn op attrs (e.g. when
// LowerConv2dQGeneric stamped `merlin.qnn_input_scale` etc. on a
// qnn.conv2d). Indexed by the same Value that's the qnn op's input/
// weight/output. writeTensor consults this map first; falls back to
// quant.uniform on element type; falls back to placeholder.
struct QuantOverrides {
	llvm::DenseMap<Value, std::pair<float, int32_t>> byValue;

	void set(Value v, float scale, int32_t zp) {
		if (v)
			byValue[v] = {scale, zp};
	}
	std::optional<std::pair<float, int32_t>> get(Value v) const {
		auto it = byValue.find(v);
		if (it == byValue.end())
			return std::nullopt;
		return it->second;
	}
};

static std::optional<float> getF32Attr(Operation *op, StringRef name) {
	if (auto attr = op->getAttrOfType<FloatAttr>(name))
		return static_cast<float>(attr.getValueAsDouble());
	return std::nullopt;
}

// Walk a qnn.* op and emit its node record. Each op kind has a distinct
// param-blob layout; we reuse the binary param-name index from the wire
// format spec.
LogicalResult writeOp(Operation *op, TensorTable &tt, Writer &w) {
	if (auto c = dyn_cast<Conv2dOp>(op)) {
		w.u32(static_cast<uint32_t>(OpKind::Conv2d));
		// Params: stride [2 i32] | pad_amount [4 i32] | dilation [2 i32] |
		//         group [1 i32] = 9 i32 = 36 bytes.
		w.u32(/*num_params=*/4);
		auto writeI32Array = [&](uint32_t name, ArrayAttr arr) {
			w.u32(name);
			w.u32(static_cast<uint32_t>(arr.size() * 4));
			for (Attribute a : arr) {
				auto ia = cast<IntegerAttr>(a);
				w.u32(static_cast<uint32_t>(ia.getInt()));
			}
		};
		writeI32Array(0 /*stride*/, c.getStride());
		writeI32Array(1 /*pad_amount*/, c.getPadAmount());
		writeI32Array(2 /*dilation*/, c.getDilation());
		w.u32(3 /*group*/);
		w.u32(4);
		w.u32(static_cast<uint32_t>(c.getGroup()));

		SmallVector<uint32_t, 3> ins{
			tt.intern(c.getInput()), tt.intern(c.getWeight())};
		if (c.getBias())
			ins.push_back(tt.intern(c.getBias()));
		w.u32(static_cast<uint32_t>(ins.size()));
		for (uint32_t id : ins)
			w.u32(id);
		w.u32(/*num_outputs=*/1);
		w.u32(tt.intern(c.getOutput()));
		return success();
	}

	// Common emit for ops with no parameters and a single input/output
	// (Reshape, Quantize, Dequantize). Caller provides the OpKind code.
	auto writeUnary = [&](OpKind k, Value in, Value out) -> LogicalResult {
		w.u32(static_cast<uint32_t>(k));
		w.u32(/*num_params=*/0);
		w.u32(/*num_inputs=*/1);
		w.u32(tt.intern(in));
		w.u32(/*num_outputs=*/1);
		w.u32(tt.intern(out));
		return success();
	};

	if (auto d = dyn_cast<DepthwiseConv2dOp>(op)) {
		w.u32(static_cast<uint32_t>(OpKind::DepthwiseConv2d));
		w.u32(/*num_params=*/3);
		auto writeI32Array = [&](uint32_t name, ArrayAttr arr) {
			w.u32(name);
			w.u32(static_cast<uint32_t>(arr.size() * 4));
			for (Attribute a : arr) {
				auto ia = cast<IntegerAttr>(a);
				w.u32(static_cast<uint32_t>(ia.getInt()));
			}
		};
		writeI32Array(0 /*stride*/, d.getStride());
		writeI32Array(1 /*pad_amount*/, d.getPadAmount());
		writeI32Array(2 /*dilation*/, d.getDilation());
		SmallVector<uint32_t, 3> ins{
			tt.intern(d.getInput()), tt.intern(d.getWeight())};
		if (d.getBias())
			ins.push_back(tt.intern(d.getBias()));
		w.u32(static_cast<uint32_t>(ins.size()));
		for (uint32_t id : ins)
			w.u32(id);
		w.u32(1);
		w.u32(tt.intern(d.getOutput()));
		return success();
	}
	if (auto p = dyn_cast<PoolMax2dOp>(op)) {
		w.u32(static_cast<uint32_t>(OpKind::PoolMax2d));
		w.u32(/*num_params=*/3);
		auto writeI32Array = [&](uint32_t name, ArrayAttr arr) {
			w.u32(name);
			w.u32(static_cast<uint32_t>(arr.size() * 4));
			for (Attribute a : arr) {
				w.u32(static_cast<uint32_t>(cast<IntegerAttr>(a).getInt()));
			}
		};
		writeI32Array(0 /*filter_size*/, p.getFilterSize());
		writeI32Array(1 /*stride*/, p.getStride());
		writeI32Array(2 /*pad_amount*/, p.getPadAmount());
		w.u32(1);
		w.u32(tt.intern(p.getInput()));
		w.u32(1);
		w.u32(tt.intern(p.getOutput()));
		return success();
	}
	if (auto p = dyn_cast<PoolAvg2dOp>(op)) {
		w.u32(static_cast<uint32_t>(OpKind::PoolAvg2d));
		w.u32(/*num_params=*/3);
		auto writeI32Array = [&](uint32_t name, ArrayAttr arr) {
			w.u32(name);
			w.u32(static_cast<uint32_t>(arr.size() * 4));
			for (Attribute a : arr) {
				w.u32(static_cast<uint32_t>(cast<IntegerAttr>(a).getInt()));
			}
		};
		writeI32Array(0 /*filter_size*/, p.getFilterSize());
		writeI32Array(1 /*stride*/, p.getStride());
		writeI32Array(2 /*pad_amount*/, p.getPadAmount());
		w.u32(1);
		w.u32(tt.intern(p.getInput()));
		w.u32(1);
		w.u32(tt.intern(p.getOutput()));
		return success();
	}
	if (auto fc = dyn_cast<FullyConnectedOp>(op)) {
		w.u32(static_cast<uint32_t>(OpKind::FullyConnected));
		w.u32(/*num_params=*/0);
		SmallVector<uint32_t, 3> ins{
			tt.intern(fc.getInput()), tt.intern(fc.getWeight())};
		if (fc.getBias())
			ins.push_back(tt.intern(fc.getBias()));
		w.u32(static_cast<uint32_t>(ins.size()));
		for (uint32_t id : ins)
			w.u32(id);
		w.u32(1);
		w.u32(tt.intern(fc.getOutput()));
		return success();
	}
	if (auto mm = dyn_cast<MatMulOp>(op)) {
		w.u32(static_cast<uint32_t>(OpKind::MatMul));
		w.u32(/*num_params=*/2);
		w.u32(0 /*transpose_lhs*/);
		w.u32(4);
		w.u32(static_cast<uint32_t>(mm.getTransposeLhs() ? 1 : 0));
		w.u32(1 /*transpose_rhs*/);
		w.u32(4);
		w.u32(static_cast<uint32_t>(mm.getTransposeRhs() ? 1 : 0));
		w.u32(2);
		w.u32(tt.intern(mm.getLhs()));
		w.u32(tt.intern(mm.getRhs()));
		w.u32(1);
		w.u32(tt.intern(mm.getOutput()));
		return success();
	}
	if (auto n = dyn_cast<ElementWiseNeuronOp>(op)) {
		w.u32(static_cast<uint32_t>(OpKind::ElementWiseNeuron));
		w.u32(/*num_params=*/1);
		w.u32(0 /*op_kind*/);
		w.u32(4);
		w.u32(static_cast<uint32_t>(n.getOpKind()));
		w.u32(1);
		w.u32(tt.intern(n.getInput()));
		w.u32(1);
		w.u32(tt.intern(n.getOutput()));
		return success();
	}
	if (auto b = dyn_cast<ElementWiseBinaryOp>(op)) {
		w.u32(static_cast<uint32_t>(OpKind::ElementWiseBinary));
		w.u32(/*num_params=*/1);
		w.u32(0 /*op_kind*/);
		w.u32(4);
		w.u32(static_cast<uint32_t>(b.getOpKind()));
		w.u32(2);
		w.u32(tt.intern(b.getLhs()));
		w.u32(tt.intern(b.getRhs()));
		w.u32(1);
		w.u32(tt.intern(b.getOutput()));
		return success();
	}
	if (auto c = dyn_cast<ConcatOp>(op)) {
		w.u32(static_cast<uint32_t>(OpKind::Concat));
		w.u32(/*num_params=*/1);
		w.u32(0 /*axis*/);
		w.u32(4);
		w.u32(static_cast<uint32_t>(c.getAxis()));
		auto inputs = c.getInputs();
		w.u32(static_cast<uint32_t>(inputs.size()));
		for (Value v : inputs)
			w.u32(tt.intern(v));
		w.u32(1);
		w.u32(tt.intern(c.getOutput()));
		return success();
	}
	if (auto t = dyn_cast<TransposeOp>(op)) {
		w.u32(static_cast<uint32_t>(OpKind::Transpose));
		w.u32(/*num_params=*/1);
		w.u32(0 /*perm*/);
		auto perm = t.getPerm();
		w.u32(static_cast<uint32_t>(perm.size() * 4));
		for (Attribute a : perm) {
			w.u32(static_cast<uint32_t>(cast<IntegerAttr>(a).getInt()));
		}
		w.u32(1);
		w.u32(tt.intern(t.getInput()));
		w.u32(1);
		w.u32(tt.intern(t.getOutput()));
		return success();
	}
	if (auto r = dyn_cast<ReshapeOp>(op))
		return writeUnary(OpKind::Reshape, r.getInput(), r.getOutput());
	if (auto q = dyn_cast<QuantizeOp>(op))
		return writeUnary(OpKind::Quantize, q.getInput(), q.getOutput());
	if (auto d = dyn_cast<DequantizeOp>(op))
		return writeUnary(OpKind::Dequantize, d.getInput(), d.getOutput());
	if (auto p = dyn_cast<PadOp>(op)) {
		w.u32(static_cast<uint32_t>(OpKind::Pad));
		w.u32(/*num_params=*/3);
		w.u32(/*name=*/0);
		auto padArr = p.getPadAmount();
		w.u32(static_cast<uint32_t>(padArr.size() * 4));
		for (Attribute a : padArr) {
			w.u32(static_cast<uint32_t>(cast<IntegerAttr>(a).getInt()));
		}
		w.u32(/*name=*/1);
		w.u32(4);
		w.u32(static_cast<uint32_t>(p.getScheme()));
		w.u32(/*name=*/2);
		w.u32(4);
		w.f32(p.getPadConstant().convertToFloat());
		w.u32(1);
		w.u32(tt.intern(p.getInput()));
		w.u32(1);
		w.u32(tt.intern(p.getOutput()));
		return success();
	}
	if (auto s = dyn_cast<SoftmaxOp>(op)) {
		w.u32(static_cast<uint32_t>(OpKind::Softmax));
		w.u32(/*num_params=*/2);
		w.u32(/*name=*/0);
		w.u32(4);
		w.u32(static_cast<uint32_t>(s.getAxis()));
		w.u32(/*name=*/1);
		w.u32(4);
		w.f32(s.getBeta().convertToFloat());
		w.u32(1);
		w.u32(tt.intern(s.getInput()));
		w.u32(1);
		w.u32(tt.intern(s.getOutput()));
		return success();
	}
	if (auto rd = dyn_cast<ReduceOp>(op)) {
		w.u32(static_cast<uint32_t>(OpKind::Reduce));
		w.u32(/*num_params=*/3);
		w.u32(/*name=*/0);
		auto axes = rd.getAxes();
		w.u32(static_cast<uint32_t>(axes.size() * 4));
		for (Attribute a : axes) {
			w.u32(static_cast<uint32_t>(cast<IntegerAttr>(a).getInt()));
		}
		w.u32(/*name=*/1);
		w.u32(4);
		w.u32(static_cast<uint32_t>(rd.getOpKind()));
		w.u32(/*name=*/2);
		w.u32(4);
		w.u32(rd.getKeepDims() ? 1u : 0u);
		w.u32(1);
		w.u32(tt.intern(rd.getInput()));
		w.u32(1);
		w.u32(tt.intern(rd.getOutput()));
		return success();
	}

	return op->emitOpError()
		<< "qnn-codegen: unsupported op kind: " << op->getName().getStringRef();
}

static bool isI8Storage(Type elem) {
	if (auto q = dyn_cast<quant::UniformQuantizedType>(elem))
		elem = q.getStorageType();
	else if (auto qpc = dyn_cast<quant::UniformQuantizedPerAxisType>(elem))
		elem = qpc.getStorageType();
	return elem.isInteger(8);
}

// Merlin/yolov8 carries int8 tensors as signless/signed i8 with zp=0. QNN
// GPU/HTA validators are stricter about fixed-point tensor conventions and
// accept the same physical bytes more reliably as UFIXED_POINT_8 with the
// quant offset shifted by +128:
//   scale * (uint8_byte - (zp + 128)) == scale * (int8_value - zp)
// This preserves dispatch ABI bytes while using QNN's expected dtype family.
static bool useUnsignedI8WithShift(Type elem, Backend backend) {
	if (!isI8Storage(elem))
		return false;
	if (backend != Backend::Gpu && backend != Backend::Hta &&
		backend != Backend::Htp)
		return false;
	if (auto q = dyn_cast<quant::UniformQuantizedType>(elem))
		return q.isSigned();
	if (auto qpc = dyn_cast<quant::UniformQuantizedPerAxisType>(elem))
		return qpc.isSigned();
	// Plain signless i8 in the yolov8 pipeline is signed storage.
	return true;
}

static int32_t qnnQuantOffset(Type elem, Backend backend, int32_t zp) {
	return useUnsignedI8WithShift(elem, backend) ? zp + 128 : zp;
}

// Map MLIR element type to a QNN_DATATYPE_* enum value (mirrors QnnTypes.h).
// For quant.uniform-wrapped types we look at the storage type underneath and
// coordinate signed-int8 handling with qnnQuantOffset above.
uint32_t qnnDtype(Type elem, Backend backend) {
	bool isSigned = false;
	bool isQuant = false;
	if (auto q = dyn_cast<quant::UniformQuantizedType>(elem)) {
		isQuant = true;
		isSigned = q.isSigned();
		elem = q.getStorageType();
	} else if (auto qpc = dyn_cast<quant::UniformQuantizedPerAxisType>(elem)) {
		isQuant = true;
		isSigned = qpc.isSigned();
		elem = qpc.getStorageType();
	}
	if (elem.isInteger(8) && useUnsignedI8WithShift(elem, backend))
		return /*UFIXED_POINT_8*/ 0x0408;
	if (elem.isInteger(8))
		return (!isQuant || isSigned) ? /*SFIXED_POINT_8*/ 0x0308
									  : /*UFIXED_POINT_8*/ 0x0408;
	if (elem.isInteger(16))
		return isSigned ? /*SFIXED_POINT_16*/ 0x0316
						: /*UFIXED_POINT_16*/ 0x0416;
	if (elem.isInteger(32)) {
		// For quantized i32 (e.g., Conv2d bias as !quant.uniform<i32:f32, s>),
		// HTA's op-package validator requires SFIXED_POINT_32 (0x0332). Plain
		// i32 without quant params stays as INT_32 (0x0032).
		return isQuant ? /*SFIXED_POINT_32*/ 0x0332 : /*INT_32*/ 0x0032;
	}
	if (elem.isF16())
		return /*FLOAT_16*/ 0x0216;
	if (elem.isF32())
		return /*FLOAT_32*/ 0x0232;
	return 0x0000; // unknown; runtime will reject.
}

// Wire-format quant_kind values (must match qnn_graph_builder.c reader):
//   0 = undefined (no quant params)
//   1 = per-tensor scale/offset (followed by f32 scale + i32 offset)
//   2 = per-axis scale/offset (followed by axis u32, count u32, then
//       count*(f32 scale + i32 offset)).

// If `v` is defined by an `arith.constant` of a dense tensor type,
// returns the raw bytes of that constant (so we can embed them as
// STATIC tensor data in the wire format). Otherwise returns nullopt.
static std::optional<std::pair<const char *, size_t>> extractConstantBytes(
	Value v) {
	// RewriteQDQToQuantUniform wraps weights/biases with
	// unrealized_conversion_cast to attach a quant.uniform element type without
	// rewriting the source tensor. The defining op of the casted value is the
	// cast, NOT the arith.constant — walk through it so STATIC-payload
	// extraction works for quant-typed conv weights and biases. Required for
	// HTA Conv2d which demands STATIC tensors for weight + bias.
	while (auto cast = v.getDefiningOp<UnrealizedConversionCastOp>()) {
		if (cast.getInputs().size() != 1)
			break;
		v = cast.getInputs()[0];
	}
	auto cstOp = v.getDefiningOp<arith::ConstantOp>();
	if (!cstOp)
		return std::nullopt;
	auto dense = dyn_cast<DenseElementsAttr>(cstOp.getValue());
	if (!dense)
		return std::nullopt;
	// For splat dense<X>, getRawData() returns just one element. For
	// non-splat, it's the full row-major buffer.
	ArrayRef<char> raw = dense.getRawData();
	return std::make_pair(raw.data(), raw.size());
}

void writeTensor(Value v, Writer &w, uint32_t id,
	const QuantOverrides &overrides, Backend backend) {
	auto t = cast<RankedTensorType>(v.getType());
	Type elem = t.getElementType();
	w.u32(id);
	w.u32(qnnDtype(elem, backend));
	w.u32(static_cast<uint32_t>(t.getRank()));
	for (int64_t d : t.getShape()) {
		w.u32(static_cast<uint32_t>(d));
	}
	// Quant params priority:
	//   1. Per-op override map (set from qnn op attrs like
	//      `merlin.qnn_input_scale` stamped by LowerConv2dQGeneric).
	//   2. quant.uniform on the tensor element type.
	//   3. Placeholder qk=1 (scale=1, zp=0) for integer dtypes — needed
	//      because QnnGpu/QnnHta reject UFIXED_POINT_* tensors without
	//      quant params at finalize.
	if (auto override_ = overrides.get(v)) {
		w.u32(/*quant_kind=*/1);
		w.f32(override_->first);
		w.u32(static_cast<uint32_t>(
			qnnQuantOffset(elem, backend, override_->second)));
	} else if (auto qpc = dyn_cast<quant::UniformQuantizedPerAxisType>(elem)) {
		// Per-channel (per-axis) quantization (Phase 3): payload is
		//   u32 axis, u32 num_channels, then num_channels * (f32 scale, i32
		//   offset).
		auto scales = qpc.getScales();
		auto zps = qpc.getZeroPoints();
		w.u32(/*quant_kind=*/2);
		w.u32(static_cast<uint32_t>(qpc.getQuantizedDimension()));
		w.u32(static_cast<uint32_t>(scales.size()));
		for (size_t i = 0; i < scales.size(); ++i) {
			w.f32(static_cast<float>(scales[i]));
			int64_t zp = (i < zps.size()) ? zps[i] : 0;
			w.u32(static_cast<uint32_t>(
				qnnQuantOffset(elem, backend, static_cast<int32_t>(zp))));
		}
	} else if (auto q = dyn_cast<quant::UniformQuantizedType>(elem)) {
		w.u32(/*quant_kind=*/1);
		w.f32(static_cast<float>(q.getScale()));
		w.u32(static_cast<uint32_t>(
			qnnQuantOffset(elem, backend, q.getZeroPoint())));
	} else {
		bool isQuantInt =
			elem.isInteger(8) || elem.isInteger(16) || elem.isInteger(32);
		if (isQuantInt) {
			w.u32(/*quant_kind=*/1);
			w.f32(1.0f);
			w.u32(static_cast<uint32_t>(qnnQuantOffset(elem, backend, 0)));
		} else {
			w.u32(/*quant_kind=*/0);
		}
	}
	// Detect inline constants — they need STATIC storage with embedded bytes.
	// For splat constants, we expand to the full element count.
	auto bytesOpt = extractConstantBytes(v);
	if (bytesOpt) {
		int64_t numElements = 1;
		for (int64_t d : t.getShape())
			numElements *= d;
		auto stType = elem;
		if (auto q = dyn_cast<quant::UniformQuantizedType>(stType))
			stType = q.getStorageType();
		int64_t bytesPerElement = stType.getIntOrFloatBitWidth() / 8;
		if (bytesPerElement <= 0)
			bytesPerElement = 1;
		int64_t total = numElements * bytesPerElement;
		w.u32(/*storage_kind=*/2); // STATIC (caller materializes data)
		w.u32(static_cast<uint32_t>(total));
		auto [data, size] = *bytesOpt;
		bool shiftI8Storage =
			bytesPerElement == 1 && useUnsignedI8WithShift(elem, backend);
		if ((int64_t)size == total) {
			// Non-splat: pass through, except for HTA/GPU unsigned fixed-point
			// i8 tensors where the storage convention is uint8 = int8 + 128.
			if (shiftI8Storage) {
				for (size_t i = 0; i < size; ++i)
					w.u8(static_cast<uint8_t>(data[i]) + 128u);
			} else {
				w.blob(reinterpret_cast<const int8_t *>(data), size);
			}
		} else {
			// Splat: expand the single element to `numElements` copies.
			for (int64_t i = 0; i < numElements; ++i) {
				if (shiftI8Storage) {
					w.u8(static_cast<uint8_t>(data[0]) + 128u);
				} else {
					w.blob(reinterpret_cast<const int8_t *>(data),
						bytesPerElement);
				}
			}
		}
	} else {
		w.u32(/*storage_kind=*/3); // native (in-graph IO)
		w.u32(/*data_size=*/0);
	}
}

} // namespace

LogicalResult serializeGraph(
	ModuleOp module, Backend backend, std::vector<int8_t> &out) {
	Writer w(out);
	TensorTable tt;

	// Two-pass: walk ops to populate tensor table + count, then emit header
	// + tensors + nodes.
	SmallVector<Operation *, 64> qnnOps;
	module.walk([&](Operation *op) {
		if (op->getDialect() && op->getDialect()->getNamespace() == "qnn") {
			qnnOps.push_back(op);
		}
	});
	if (qnnOps.empty()) {
		return module.emitOpError()
			<< "qnn-codegen: module has no `qnn.*` ops to serialize";
	}

	Operation *residualLinalg = nullptr;
	module.walk([&](Operation *op) {
		if (residualLinalg)
			return;
		if (!isa<linalg::LinalgOp>(op))
			return;
		// Skip dead DPS linalg ops (linalg.fill, linalg.transpose, linalg.copy)
		// that the convert pass couldn't erase. Greedy DCE doesn't always
		// reclaim them because they have a destination-style "write" effect
		// even when their result is unused. They're pure no-ops once the
		// pattern that consumed their output got rewritten.
		if (op->use_empty() &&
			(isa<linalg::FillOp>(op) || isa<linalg::TransposeOp>(op) ||
				isa<linalg::CopyOp>(op)))
			return;
		residualLinalg = op;
	});
	if (residualLinalg) {
		return residualLinalg->emitOpError()
			<< "qnn-codegen: refusing to serialize partial QNN graph with "
			   "residual linalg op; keep this dispatch on CPU or add a QNN "
			   "lowering pattern for the remaining op";
	}

	for (Operation *op : qnnOps) {
		if (backend == Backend::Gpu && isa<QuantizeOp>(op)) {
			return op->emitOpError()
				<< "qnn-codegen: QNN GPU backend does not provide Quantize; "
				   "keep this dispatch on CPU or lower a larger subgraph that "
				   "does not require a GPU Quantize boundary";
		}
		if (backend == Backend::Hta &&
			(isa<QuantizeOp>(op) || isa<DequantizeOp>(op))) {
			return op->emitOpError()
				<< "qnn-codegen: QNN HTA requires all-int fixed-point graphs; "
				   "standalone Quantize/Dequantize boundaries are unsupported";
		}
		if (backend == Backend::Hta && isa<TransposeOp>(op)) {
			return op->emitOpError()
				<< "qnn-codegen: QNN HTA backend does not provide Transpose; "
				   "run layout/boundary transposes on CPU or split the graph "
				   "so "
				   "HTA receives channel-last tensors directly";
		}
		if (backend == Backend::Hta && isa<MatMulOp>(op)) {
			return op->emitOpError()
				<< "qnn-codegen: QNN HTA backend does not provide MatMul in "
				   "this "
				   "SDK path; keep matmul dispatches on CPU/GPU";
		}
		if (backend == Backend::Hta) {
			// HTA's op-package only supports UFIXED_POINT_8 for
			// ElementWiseNeuron/ ElementWiseBinary. f32 / f16 inputs (e.g., the
			// QDQ-Sigmoid-QDQ pure- fp32 quantize roundtrip in yolov8n's
			// detection head) are rejected at `graphAddNode rc=6000`. Surface
			// the dtype mismatch at compile time instead so the dispatch routes
			// to CPU/GPU.
			auto checkFpReject = [&](Type elem) -> LogicalResult {
				if (elem.isF32() || elem.isF16()) {
					return op->emitOpError()
						<< "qnn-codegen: QNN HTA accepts only quantized "
						   "(UFIXED_POINT_8) tensors for ElementWise ops; got "
						   "f32/"
						   "f16 — keep this dispatch on CPU/GPU";
				}
				return success();
			};
			if (auto n = dyn_cast<ElementWiseNeuronOp>(op)) {
				Type elem = cast<RankedTensorType>(n.getInput().getType())
								.getElementType();
				if (failed(checkFpReject(elem)))
					return failure();
			}
			if (auto b = dyn_cast<ElementWiseBinaryOp>(op)) {
				Type elem = cast<RankedTensorType>(b.getLhs().getType())
								.getElementType();
				if (failed(checkFpReject(elem)))
					return failure();
			}
		}
		if (backend == Backend::Gpu) {
			if (auto c = dyn_cast<Conv2dOp>(op)) {
				Type elem = cast<RankedTensorType>(c.getInput().getType())
								.getElementType();
				if (!elem.isF16() && !elem.isF32()) {
					return op->emitOpError() << "qnn-codegen: QNN GPU Conv2d "
												"supports fp16/fp32 here; "
												"int8 yolov8 conv dispatches "
												"must route to HTA or CPU";
				}
			}
			if (auto m = dyn_cast<MatMulOp>(op)) {
				Type lhsElem = cast<RankedTensorType>(m.getLhs().getType())
								   .getElementType();
				Type rhsElem = cast<RankedTensorType>(m.getRhs().getType())
								   .getElementType();
				if (!lhsElem.isF16() && !lhsElem.isF32() && !rhsElem.isF16() &&
					!rhsElem.isF32()) {
					return op->emitOpError()
						<< "qnn-codegen: QNN GPU MatMul kernel creation fails "
						   "for "
						   "int8 yolov8 matmul dispatches on this SDK/backend; "
						   "keep "
						   "them on CPU or lower a supported fp16/fp32 matmul";
				}
			}
		}
	}

	// Pre-intern operands + results to size the tensor table.
	for (Operation *op : qnnOps) {
		for (Value v : op->getOperands())
			tt.intern(v);
		for (Value v : op->getResults())
			tt.intern(v);
	}

	// Harvest per-tensor quant-param overrides from qnn op attrs.
	// LowerConv2dQGeneric / Lower*QGeneric stamp scale/zp on the qnn op
	// when discovered from upstream dequant generics OR matched rescale
	// chains. Reading them here lets writeTensor emit qk=1 with real
	// per-tensor scales instead of placeholders.
	QuantOverrides overrides;
	auto setIfAttr = [&](Operation *op, StringRef scaleName, StringRef zpName,
						 Value tensor) {
		auto sa = op->getAttrOfType<FloatAttr>(scaleName);
		if (!sa)
			return;
		int32_t zp = 0;
		if (auto za = op->getAttrOfType<IntegerAttr>(zpName))
			zp = static_cast<int32_t>(za.getInt());
		overrides.set(tensor, static_cast<float>(sa.getValueAsDouble()), zp);
	};
	for (Operation *op : qnnOps) {
		if (auto c = dyn_cast<Conv2dOp>(op)) {
			setIfAttr(op, "merlin.qnn_input_scale",
				"merlin.qnn_input_zero_point", c.getInput());
			setIfAttr(op, "merlin.qnn_weight_scale",
				"merlin.qnn_weight_zero_point", c.getWeight());
			setIfAttr(op, "merlin.qnn_output_scale",
				"merlin.qnn_output_zero_point", c.getOutput());
			if (c.getBias()) {
				if (auto accScale =
						getF32Attr(op, "merlin.qnn_accumulator_scale")) {
					overrides.set(c.getBias(), *accScale, /*zp=*/0);
				} else if (auto inScale =
							   getF32Attr(op, "merlin.qnn_input_scale")) {
					if (auto weightScale =
							getF32Attr(op, "merlin.qnn_weight_scale"))
						overrides.set(
							c.getBias(), (*inScale) * (*weightScale), /*zp=*/0);
				}
			}
		} else if (auto m = dyn_cast<MatMulOp>(op)) {
			setIfAttr(op, "merlin.qnn_input_scale",
				"merlin.qnn_input_zero_point", m.getLhs());
			setIfAttr(op, "merlin.qnn_weight_scale",
				"merlin.qnn_weight_zero_point", m.getRhs());
			setIfAttr(op, "merlin.qnn_output_scale",
				"merlin.qnn_output_zero_point", m.getOutput());
		}
		// Future: same for DepthwiseConv2dOp, FullyConnectedOp.
	}

	// Header.
	w.u32(0x6E4E5151); // magic "QQNn"
	w.u16(1); // version
	w.u16(static_cast<uint16_t>(backend));
	w.u32(static_cast<uint32_t>(tt.ordered.size()));
	w.u32(static_cast<uint32_t>(qnnOps.size()));
	w.u32(0); // input_count -- TODO: derive from func args once we have
			  // the function-level wrapper around qnn ops
	w.u32(0); // output_count -- TODO: derive from func returns

	// Tensors.
	for (size_t i = 0; i < tt.ordered.size(); ++i) {
		writeTensor(
			tt.ordered[i], w, static_cast<uint32_t>(i), overrides, backend);
	}

	// Nodes.
	for (Operation *op : qnnOps) {
		if (failed(writeOp(op, tt, w)))
			return failure();
	}
	return success();
}

} // namespace mlir::iree_compiler::QNN::Codegen
