// QNN kernel: int8 NHWC Conv2D (no fused activation). Validates QNN's
// quantization-aware Conv2D path on GPU — the missing primitive for any
// subsequent real-yolov8 work. Quantization params are baked into the
// tensor descriptors via Qnn_QuantizeParams_t.scaleOffsetEncoding; the
// QNN runtime handles the i8 → i32 acc → i8 requant internally.
//
// Shape: input [1, 8, 8, 3] i8, weight [3, 3, 3, 4] i8 HWCF, bias [4] i32,
//        output [1, 6, 6, 4] i8 (VALID 3x3, stride 1).
//
// Q-params (representative, not derived from any specific model):
//   input  scale=0.05   offset=0
//   weight scale=0.025  offset=0
//   output scale=0.10   offset=0
//   bias scale = input_scale * weight_scale = 0.00125 (no offset; bias is
//        already in the i32 acc domain).
//
// Static weights: all i8 value +1. Static bias: all 0. With these the
// reference output for input quantized representation `q` is:
//   acc[c]  = sum over 3x3x3 of (q_in - 0) * (1 - 0)            // i32
//   y[c]    = round( acc[c] * (input_scale * weight_scale) / output_scale )
//          = round( acc[c] * (0.05 * 0.025) / 0.10 )
//          = round( acc[c] * 0.0125 )
//   y_i8[c] = clamp(y[c], -128, 127)

#include "QnnKernelHelpers.hpp"
#include "QnnModel.hpp"
#include "QnnOpDef.h"

#include <cstdint>

#define DO_GRAPH_NODE_VALIDATIONS 1

using namespace qnn_wrapper_api;

namespace {

constexpr uint32_t kKh = 3, kKw = 3, kInC = 3, kOutC = 4;

// HTA Conv2d wants UFIXED_POINT_8 for weights (unsigned 0..255 with offset);
// to represent the value `+1` after subtracting the zero-point, we store
// 128 + 1 = 129 with a zero_point of 128 in q-params. Bias remains signed
// i32 (the implicit pre-requant accumulator).
uint8_t g_weight[kKh * kKw * kInC * kOutC];
int32_t g_bias[kOutC];

void initialize_weights() {
	// weight q=2 (real=0.025*2=0.05) so a single 27-tap conv produces a
	// value within [0, 255]*output_scale on uint8 output without
	// saturating.
	for (uint32_t i = 0; i < kKh * kKw * kInC * kOutC; ++i)
		g_weight[i] = 2;
	for (uint32_t c = 0; c < kOutC; ++c)
		g_bias[c] = 0;
}

uint32_t g_input_dims[4] = {1, 8, 8, 3};
uint32_t g_weight_dims[4] = {kKh, kKw, kInC, kOutC};
uint32_t g_bias_dims[1] = {kOutC};
uint32_t g_output_dims[4] = {1, 6, 6, 4};

uint32_t g_pad_amount[4] = {0, 0, 0, 0};
uint32_t g_pad_dims[2] = {2, 2};
uint32_t g_stride[2] = {1, 1};
uint32_t g_stride_dims[1] = {2};
uint32_t g_dilation[2] = {1, 1};
uint32_t g_dilation_dims[1] = {2};

Qnn_QuantizeParams_t qparams_with(float scale, int32_t offset) {
	return Qnn_QuantizeParams_t{
		QNN_DEFINITION_DEFINED,
		QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
		{.scaleOffsetEncoding = {scale, offset}},
	};
}

} // namespace

extern "C" {

QNN_API
ModelError_t QnnModel_composeGraphs(Qnn_BackendHandle_t backendHandle,
	QNN_INTERFACE_VER_TYPE interface, Qnn_ContextHandle_t contextHandle,
	const GraphConfigInfo_t **graphsConfigInfo,
	const uint32_t numGraphsConfigInfo, GraphInfoPtr_t **graphsInfo,
	uint32_t *numGraphsInfo, bool /*debug*/, QnnLog_Callback_t /*lc*/,
	QnnLog_Level_t /*ll*/) {
	ModelError_t err = MODEL_NO_ERROR;
	initialize_weights();

	QnnModel model;
	const QnnGraph_Config_t **gc = nullptr;
	VALIDATE(getQnnGraphConfigFromInfo("conv2d_int8_smoke", graphsConfigInfo,
				 numGraphsConfigInfo, gc),
		err);
	VALIDATE(model.initialize(backendHandle, interface, contextHandle,
				 "conv2d_int8_smoke", false, DO_GRAPH_NODE_VALIDATIONS, gc),
		err);

	// QNN convention: real_value = scale * (q - offset). HTA appears to
	// reject offset=+128 (saturates output); use asymmetric uint8 with
	// offset=0 instead — the all-positive range works on both libQnnCpu
	// and libQnnHta. With offset=0 and uint8 q in [0, 255], real values
	// span [0, 255*scale]. We bump the input/weight q values upward so
	// the analytical output stays well-bounded:
	//   input q=64 → real = 0.05 * 64 = 3.2
	//   weight q=2 → real = 0.025 * 2 = 0.05
	//   acc = 27 * 3.2 * 0.05 = 4.32 → q_out = 4.32 / 0.10 = 43
	Qnn_QuantizeParams_t qp_in = qparams_with(0.05f, 0);
	Qnn_QuantizeParams_t qp_w = qparams_with(0.025f, 0);
	Qnn_QuantizeParams_t qp_out = qparams_with(0.10f, 0);

	Qnn_Tensor_t input{};
	input.version = QNN_TENSOR_VERSION_1;
	input.v1 = {.id = 0,
		.name = "input",
		.type = QNN_TENSOR_TYPE_APP_WRITE,
		.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
		.dataType = QNN_DATATYPE_UFIXED_POINT_8,
		.quantizeParams = qp_in,
		.rank = 4,
		.dimensions = g_input_dims,
		.memType = QNN_TENSORMEMTYPE_RAW,
		.clientBuf = {nullptr, 0}};
	VALIDATE(model.addTensor("input", &input), err);

	Qnn_Tensor_t weight{};
	weight.version = QNN_TENSOR_VERSION_1;
	weight.v1 = {.id = 0,
		.name = "weight",
		.type = QNN_TENSOR_TYPE_STATIC,
		.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
		.dataType = QNN_DATATYPE_UFIXED_POINT_8,
		.quantizeParams = qp_w,
		.rank = 4,
		.dimensions = g_weight_dims,
		.memType = QNN_TENSORMEMTYPE_RAW,
		.clientBuf = {g_weight, sizeof(g_weight)}};
	VALIDATE(model.addTensor("weight", &weight), err);

	Qnn_Tensor_t bias{};
	bias.version = QNN_TENSOR_VERSION_1;
	Qnn_QuantizeParams_t qp_bias = qparams_with(0.05f * 0.025f, 0);
	bias.v1 = {.id = 0,
		.name = "bias",
		.type = QNN_TENSOR_TYPE_STATIC,
		.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
		.dataType = QNN_DATATYPE_SFIXED_POINT_32,
		.quantizeParams = qp_bias,
		.rank = 1,
		.dimensions = g_bias_dims,
		.memType = QNN_TENSORMEMTYPE_RAW,
		.clientBuf = {g_bias, sizeof(g_bias)}};
	VALIDATE(model.addTensor("bias", &bias), err);

	auto makeStaticParam = [&](const char *name, uint32_t rank, uint32_t *dims,
							   void *data, uint32_t bytes) {
		Qnn_Param_t p{};
		p.paramType = QNN_PARAMTYPE_TENSOR;
		p.name = name;
		p.tensorParam.version = QNN_TENSOR_VERSION_1;
		p.tensorParam.v1 = {.id = 0,
			.name = name,
			.type = QNN_TENSOR_TYPE_STATIC,
			.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
			.dataType = QNN_DATATYPE_UINT_32,
			.quantizeParams = {QNN_DEFINITION_UNDEFINED,
				QNN_QUANTIZATION_ENCODING_UNDEFINED,
				{.scaleOffsetEncoding = {0.0f, 0}}},
			.rank = rank,
			.dimensions = dims,
			.memType = QNN_TENSORMEMTYPE_RAW,
			.clientBuf = {data, bytes}};
		return p;
	};

	Qnn_Param_t conv_params[4];
	conv_params[0] = makeStaticParam(QNN_OP_CONV_2D_PARAM_DILATION, 1,
		g_dilation_dims, g_dilation, sizeof(g_dilation));
	conv_params[1] = makeStaticParam(QNN_OP_CONV_2D_PARAM_PAD_AMOUNT, 2,
		g_pad_dims, g_pad_amount, sizeof(g_pad_amount));
	conv_params[2] = makeStaticParam(QNN_OP_CONV_2D_PARAM_STRIDE, 1,
		g_stride_dims, g_stride, sizeof(g_stride));
	conv_params[3].paramType = QNN_PARAMTYPE_SCALAR;
	conv_params[3].name = QNN_OP_CONV_2D_PARAM_GROUP;
	conv_params[3].scalarParam = {
		.dataType = QNN_DATATYPE_UINT_32, .uint32Value = 1};

	Qnn_Tensor_t output{};
	output.version = QNN_TENSOR_VERSION_1;
	output.v1 = {.id = 0,
		.name = "output",
		.type = QNN_TENSOR_TYPE_APP_READ,
		.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
		.dataType = QNN_DATATYPE_UFIXED_POINT_8,
		.quantizeParams = qp_out,
		.rank = 4,
		.dimensions = g_output_dims,
		.memType = QNN_TENSORMEMTYPE_RAW,
		.clientBuf = {nullptr, 0}};

	const char *conv_inputs[] = {"input", "weight", "bias"};
	VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, "conv_op", "qti.aisw",
				 QNN_OP_CONV_2D, conv_params, 4, conv_inputs, 3, &output, 1),
		err);

	QnnModel *m[] = {&model};
	VALIDATE(getGraphInfoFromModels(*m, 1, graphsInfo), err);
	*numGraphsInfo = 1;
	return err;
}

QNN_API
ModelError_t QnnModel_freeGraphsInfo(
	GraphInfoPtr_t **graphsInfo, uint32_t numGraphsInfo) {
	return freeGraphsInfo(graphsInfo, numGraphsInfo);
}

} // extern "C"
