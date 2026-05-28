// QNN kernel: YOLOv8 stem Conv2D in fp16 NHWC, targeting Adreno GPU on
// QAIRT 2.45. Adreno's QNN op package on this SDK supports Conv2d only
// in FLOAT_32 / FLOAT_16 (per QNN/OpDef/GpuOpDefSupplement.html), so any
// quantized-conv island routed to GPU must enter as Dequantize→fp16-conv
// at the boundary.
//
// Shape: input [1, 320, 320, 3] fp16 NHWC, weight [3, 3, 3, 16] fp16
// HWIO, bias [16] fp16, output [1, 160, 160, 16] fp16. Pad H/W = 1, 1.
// Stride 2, dilation 1, group 1. Same numerical conv as the int8/HTA
// `yolov8_stem_nhwc_int8` kernel; numerical equivalence between the two
// is gated downstream (quant-step + ULP tolerance).

#include "QnnKernelHelpers.hpp"
#include "QnnModel.hpp"
#include "QnnOpDef.h"

#include <cstdint>
#include <cstring>

#define DO_GRAPH_NODE_VALIDATIONS 1

using namespace qnn_wrapper_api;

namespace {
constexpr uint32_t kKh = 3, kKw = 3, kInC = 3, kOutC = 16;
constexpr uint32_t kInH = 320, kInW = 320;
constexpr uint32_t kOutH = 160, kOutW = 160;

// fp16 storage. We don't need real values to validate the GPU path —
// the static buffers must merely be the right size and dtype so the
// validator + finalizer accept them. A downstream numerical-equivalence
// test loads real yolov8 weights into the same layout.
uint16_t g_weight_fp16[kKh * kKw * kInC * kOutC] = {0};
uint16_t g_bias_fp16[kOutC] = {0};

uint32_t g_input_dims[4] = {1, kInH, kInW, kInC};
uint32_t g_weight_dims[4] = {kKh, kKw, kInC, kOutC};
uint32_t g_bias_dims[1] = {kOutC};
uint32_t g_output_dims[4] = {1, kOutH, kOutW, kOutC};

// Conv2D params: H/W padding 1 each side, stride 2, dilation 1, group 1.
uint32_t g_pad_amount[4] = {1, 1, 1, 1};
uint32_t g_pad_dims[2] = {2, 2};
uint32_t g_stride[2] = {2, 2};
uint32_t g_stride_dims[1] = {2};
uint32_t g_dilation[2] = {1, 1};
uint32_t g_dilation_dims[1] = {2};
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

	QnnModel model;
	const QnnGraph_Config_t **gc = nullptr;
	VALIDATE(getQnnGraphConfigFromInfo("yolov8_stem_fp16_gpu", graphsConfigInfo,
				 numGraphsConfigInfo, gc),
		err);
	VALIDATE(model.initialize(backendHandle, interface, contextHandle,
				 "yolov8_stem_fp16_gpu", false, DO_GRAPH_NODE_VALIDATIONS, gc),
		err);

	// fp16 tensors carry no quantizeParams — encoding is UNDEFINED.
	Qnn_QuantizeParams_t qp_undef = {QNN_DEFINITION_UNDEFINED,
		QNN_QUANTIZATION_ENCODING_UNDEFINED,
		{.scaleOffsetEncoding = {0.0f, 0}}};

	Qnn_Tensor_t input{};
	input.version = QNN_TENSOR_VERSION_1;
	input.v1 = {.id = 0,
		.name = "input",
		.type = QNN_TENSOR_TYPE_APP_WRITE,
		.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
		.dataType = QNN_DATATYPE_FLOAT_16,
		.quantizeParams = qp_undef,
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
		.dataType = QNN_DATATYPE_FLOAT_16,
		.quantizeParams = qp_undef,
		.rank = 4,
		.dimensions = g_weight_dims,
		.memType = QNN_TENSORMEMTYPE_RAW,
		.clientBuf = {g_weight_fp16, sizeof(g_weight_fp16)}};
	VALIDATE(model.addTensor("weight", &weight), err);

	Qnn_Tensor_t bias{};
	bias.version = QNN_TENSOR_VERSION_1;
	bias.v1 = {.id = 0,
		.name = "bias",
		.type = QNN_TENSOR_TYPE_STATIC,
		.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
		.dataType = QNN_DATATYPE_FLOAT_16,
		.quantizeParams = qp_undef,
		.rank = 1,
		.dimensions = g_bias_dims,
		.memType = QNN_TENSORMEMTYPE_RAW,
		.clientBuf = {g_bias_fp16, sizeof(g_bias_fp16)}};
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
			.quantizeParams = qp_undef,
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
		.dataType = QNN_DATATYPE_FLOAT_16,
		.quantizeParams = qp_undef,
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
