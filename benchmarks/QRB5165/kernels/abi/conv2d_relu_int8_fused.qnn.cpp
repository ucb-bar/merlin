// QNN kernel: fused Conv2D + ReLU on Hexagon HTA, uint8 quantized.
//
// Two ops in ONE QnnModel: Conv2d → ElementWiseNeuron(Relu). HTA's
// finalize-time optimizer (`fold_relu_activation_into_conv`) collapses
// them into a single HVX kernel that does the conv + clamp(0, ...) +
// requantize as one streaming pass — no intermediate uint8 write +
// re-read between the two ops.
//
// Compared to the unfused alternative (separate Conv2D ctxbin + separate
// Relu ctxbin chained through IREE), this saves:
//   - 1 FastRPC roundtrip (~1.5 ms on QRB5165)
//   - 1 intermediate buffer alloc + memcpy
//   - 1 dequant→relu→requant traversal (subsumed into conv's output requant)
//
// Shape: input [1, 8, 8, 3] uint8 NHWC, weight [3, 3, 3, 4] uint8 HWCF,
//        bias [4] sfixed32, output [1, 6, 6, 4] uint8.
// q-params: input scale=0.05 offset=0; weight scale=0.025 offset=0;
//           output scale=0.10 offset=0; bias scale=input*weight=0.00125.
//
// Same q-params and weight values as `conv2d_int8_smoke.qnn.cpp` so the
// outputs are directly comparable: unfused chain produces the same q
// values, fused does too — the win is purely in execution time.

#include "QnnKernelHelpers.hpp"
#include "QnnModel.hpp"
#include "QnnOpDef.h"

#include <cstdint>

#define DO_GRAPH_NODE_VALIDATIONS 1

using namespace qnn_wrapper_api;

namespace {
constexpr uint32_t kKh = 3, kKw = 3, kInC = 3, kOutC = 4;

uint8_t g_weight[kKh * kKw * kInC * kOutC];
int32_t g_bias[kOutC];

void initialize_weights() {
	for (uint32_t i = 0; i < kKh * kKw * kInC * kOutC; ++i)
		g_weight[i] = 2;
	for (uint32_t c = 0; c < kOutC; ++c)
		g_bias[c] = 0;
}

uint32_t g_input_dims[4] = {1, 8, 8, 3};
uint32_t g_weight_dims[4] = {kKh, kKw, kInC, kOutC};
uint32_t g_bias_dims[1] = {kOutC};
uint32_t g_conv_out_dims[4] = {1, 6, 6, 4};
uint32_t g_output_dims[4] = {1, 6, 6, 4};

uint32_t g_pad_amount[4] = {0, 0, 0, 0};
uint32_t g_pad_dims[2] = {2, 2};
uint32_t g_stride[2] = {1, 1};
uint32_t g_stride_dims[1] = {2};
uint32_t g_dilation[2] = {1, 1};
uint32_t g_dilation_dims[1] = {2};

Qnn_QuantizeParams_t qparams(float scale, int32_t offset) {
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
	VALIDATE(getQnnGraphConfigFromInfo("conv2d_relu_int8_fused",
				 graphsConfigInfo, numGraphsConfigInfo, gc),
		err);
	VALIDATE(
		model.initialize(backendHandle, interface, contextHandle,
			"conv2d_relu_int8_fused", false, DO_GRAPH_NODE_VALIDATIONS, gc),
		err);

	Qnn_QuantizeParams_t qp_in = qparams(0.05f, 0);
	Qnn_QuantizeParams_t qp_w = qparams(0.025f, 0);
	Qnn_QuantizeParams_t qp_act =
		qparams(0.10f, 0); // shared between conv_out and relu_out
	Qnn_QuantizeParams_t qp_bias = qparams(0.05f * 0.025f, 0);

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
			.quantizeParams = merlin_qnn::fp32QuantizeParams(),
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

	// Intermediate tensor between Conv2d and ElementWiseNeuron — same
	// q-params as the relu output (they share scale/offset, which lets
	// HTA's fold_relu_activation_into_conv collapse them into one HVX op
	// that does conv + clamp + requant in a single pass).
	Qnn_Tensor_t conv_out{};
	conv_out.version = QNN_TENSOR_VERSION_1;
	conv_out.v1 = {.id = 0,
		.name = "conv_out",
		.type = QNN_TENSOR_TYPE_NATIVE,
		.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
		.dataType = QNN_DATATYPE_UFIXED_POINT_8,
		.quantizeParams = qp_act,
		.rank = 4,
		.dimensions = g_conv_out_dims,
		.memType = QNN_TENSORMEMTYPE_RAW,
		.clientBuf = {nullptr, 0}};

	const char *conv_inputs[] = {"input", "weight", "bias"};
	VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, "conv_op", "qti.aisw",
				 QNN_OP_CONV_2D, conv_params, 4, conv_inputs, 3, &conv_out, 1),
		err);

	// ElementWiseNeuron with operation=Relu. The single scalar "operation"
	// parameter selects which activation; HTA's optimizer detects the
	// Conv2d→ElementWiseNeuron(Relu) sequence and folds them.
	Qnn_Param_t neuron_params[1];
	neuron_params[0].paramType = QNN_PARAMTYPE_SCALAR;
	neuron_params[0].name = QNN_OP_ELEMENT_WISE_NEURON_PARAM_OPERATION;
	neuron_params[0].scalarParam = {
		.dataType = QNN_DATATYPE_UINT_32,
		.uint32Value = QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU,
	};

	Qnn_Tensor_t output{};
	output.version = QNN_TENSOR_VERSION_1;
	output.v1 = {.id = 0,
		.name = "output",
		.type = QNN_TENSOR_TYPE_APP_READ,
		.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
		.dataType = QNN_DATATYPE_UFIXED_POINT_8,
		.quantizeParams = qp_act,
		.rank = 4,
		.dimensions = g_output_dims,
		.memType = QNN_TENSORMEMTYPE_RAW,
		.clientBuf = {nullptr, 0}};

	const char *relu_inputs[] = {"conv_out"};
	VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, "relu_op", "qti.aisw",
				 QNN_OP_ELEMENT_WISE_NEURON, neuron_params, 1, relu_inputs, 1,
				 &output, 1),
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
