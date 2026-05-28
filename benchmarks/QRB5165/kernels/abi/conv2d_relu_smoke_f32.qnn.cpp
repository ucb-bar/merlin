// QNN kernel: fp32 Conv2D + ReLU multi-op smoke. Hand-authored against
// `qnn_wrapper_api::QnnModel` to validate that the no-ONNX authoring style
// scales beyond single-op (PR-A's Add) to a real fused two-op graph that
// exercises Conv2D's full param surface (stride / pad_amount / dilation /
// group / weight + bias static tensors) plus a chained Relu.
//
// Shape: NHWC (QNN's native layout)
//   input  : [1, 8, 8, 3]   APP_WRITE f32, runtime-supplied
//   weight : [3, 3, 3, 4]   STATIC f32, baked-in (filters of 1.0)
//   bias   : [4]            STATIC f32, baked-in (-1.0)
//   conv   : [1, 6, 6, 4]   NATIVE (intermediate)
//   output : [1, 6, 6, 4]   APP_READ f32 = max(0, conv)
//
// For deterministic verification: weight = all-ones, bias = -1.0. The output
// equals max(0, sum_over_kh_kw_ic(input_NHWC) - 1) for each output channel.
// The board-side test fills input with linspace and compares to a host-side
// numpy reference computed identically.

#include "QnnKernelHelpers.hpp"
#include "QnnModel.hpp"

#define DO_GRAPH_NODE_VALIDATIONS 1

using namespace qnn_wrapper_api;
using merlin_qnn::addOp;
using merlin_qnn::fp32QuantizeParams;
using merlin_qnn::makeTensor;
using merlin_qnn::TensorSpec;

namespace {

// Static weights and biases. Defined at namespace scope so the Qnn_Tensor_t
// clientBuf pointers remain valid for the lifetime of the model.
constexpr uint32_t kKh = 3, kKw = 3, kInC = 3, kOutC = 4;

float g_weight[kKh * kKw * kInC * kOutC];
float g_bias[kOutC];

void initialize_weights() {
	for (uint32_t i = 0; i < kKh * kKw * kInC * kOutC; ++i) {
		g_weight[i] = 1.0f;
	}
	for (uint32_t c = 0; c < kOutC; ++c) {
		g_bias[c] = -1.0f;
	}
}

uint32_t g_pad_amount[2 * 2] = {
	0, 0, 0, 0}; // [[h_before, h_after], [w_before, w_after]]
uint32_t g_pad_dims[2] = {2, 2};
uint32_t g_stride[2] = {1, 1};
uint32_t g_stride_dims[1] = {2};
uint32_t g_dilation[2] = {1, 1};
uint32_t g_dilation_dims[1] = {2};

uint32_t g_weight_dims[4] = {kKh, kKw, kInC, kOutC};
uint32_t g_bias_dims[1] = {kOutC};
uint32_t g_input_dims[4] = {1, 8, 8, 3};
uint32_t g_conv_dims[4] = {1, 6, 6, 4};
uint32_t g_output_dims[4] = {1, 6, 6, 4};

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
	VALIDATE(getQnnGraphConfigFromInfo("conv2d_relu_smoke_f32",
				 graphsConfigInfo, numGraphsConfigInfo, gc),
		err);
	VALIDATE(model.initialize(backendHandle, interface, contextHandle,
				 "conv2d_relu_smoke_f32", false, DO_GRAPH_NODE_VALIDATIONS, gc),
		err);

	Qnn_QuantizeParams_t qp = fp32QuantizeParams();

	// ----- Tensors -----

	Qnn_Tensor_t input{};
	input.version = QNN_TENSOR_VERSION_1;
	input.v1 = {.id = 0,
		.name = "input",
		.type = QNN_TENSOR_TYPE_APP_WRITE,
		.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
		.dataType = QNN_DATATYPE_FLOAT_32,
		.quantizeParams = qp,
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
		.dataType = QNN_DATATYPE_FLOAT_32,
		.quantizeParams = qp,
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
		.dataType = QNN_DATATYPE_FLOAT_32,
		.quantizeParams = qp,
		.rank = 1,
		.dimensions = g_bias_dims,
		.memType = QNN_TENSORMEMTYPE_RAW,
		.clientBuf = {g_bias, sizeof(g_bias)}};
	VALIDATE(model.addTensor("bias", &bias), err);

	// ----- Conv2D node -----

	Qnn_Tensor_t conv_out{};
	conv_out.version = QNN_TENSOR_VERSION_1;
	conv_out.v1 = {.id = 0,
		.name = "conv_out",
		.type = QNN_TENSOR_TYPE_NATIVE,
		.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
		.dataType = QNN_DATATYPE_FLOAT_32,
		.quantizeParams = qp,
		.rank = 4,
		.dimensions = g_conv_dims,
		.memType = QNN_TENSORMEMTYPE_RAW,
		.clientBuf = {nullptr, 0}};

	auto makeStaticTensorParam = [&](const char *name, uint32_t rank,
									 uint32_t *dims, void *data,
									 uint32_t bytes) {
		Qnn_Param_t p{};
		p.paramType = QNN_PARAMTYPE_TENSOR;
		p.name = name;
		p.tensorParam.version = QNN_TENSOR_VERSION_1;
		p.tensorParam.v1 = {.id = 0,
			.name = name,
			.type = QNN_TENSOR_TYPE_STATIC,
			.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
			.dataType = QNN_DATATYPE_UINT_32,
			.quantizeParams = qp,
			.rank = rank,
			.dimensions = dims,
			.memType = QNN_TENSORMEMTYPE_RAW,
			.clientBuf = {data, bytes}};
		return p;
	};

	Qnn_Param_t conv_params[4];
	conv_params[0] = makeStaticTensorParam(QNN_OP_CONV_2D_PARAM_DILATION, 1,
		g_dilation_dims, g_dilation, sizeof(g_dilation));
	conv_params[1] = makeStaticTensorParam(QNN_OP_CONV_2D_PARAM_PAD_AMOUNT, 2,
		g_pad_dims, g_pad_amount, sizeof(g_pad_amount));
	conv_params[2] = makeStaticTensorParam(QNN_OP_CONV_2D_PARAM_STRIDE, 1,
		g_stride_dims, g_stride, sizeof(g_stride));
	conv_params[3].paramType = QNN_PARAMTYPE_SCALAR;
	conv_params[3].name = QNN_OP_CONV_2D_PARAM_GROUP;
	conv_params[3].scalarParam = {
		.dataType = QNN_DATATYPE_UINT_32, .uint32Value = 1};

	const char *conv_inputs[] = {"input", "weight", "bias"};
	VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, "conv_op", "qti.aisw",
				 QNN_OP_CONV_2D, conv_params, 4, conv_inputs, 3, &conv_out, 1),
		err);

	// ----- Relu node (consumes conv_out, produces graph output) -----

	Qnn_Tensor_t output{};
	output.version = QNN_TENSOR_VERSION_1;
	output.v1 = {.id = 0,
		.name = "output",
		.type = QNN_TENSOR_TYPE_APP_READ,
		.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
		.dataType = QNN_DATATYPE_FLOAT_32,
		.quantizeParams = qp,
		.rank = 4,
		.dimensions = g_output_dims,
		.memType = QNN_TENSORMEMTYPE_RAW,
		.clientBuf = {nullptr, 0}};

	VALIDATE(addOp(model, "relu_op", QNN_OP_RELU, {"conv_out"}, output), err);

	QnnModel *models[] = {&model};
	VALIDATE(getGraphInfoFromModels(*models, 1, graphsInfo), err);
	*numGraphsInfo = 1;
	return err;
}

QNN_API
ModelError_t QnnModel_freeGraphsInfo(
	GraphInfoPtr_t **graphsInfo, uint32_t numGraphsInfo) {
	return freeGraphsInfo(graphsInfo, numGraphsInfo);
}

} // extern "C"
