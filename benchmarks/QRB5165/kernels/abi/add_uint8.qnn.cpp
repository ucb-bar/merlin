// QNN kernel: uint8 elementwise add over a 1x16 tensor.
//
// Hand-authored against `qnn_wrapper_api::QnnModel`, targeted at the
// Hexagon HTA (NPU). UFIXED_POINT_8 with offset=0 (asymmetric, all-positive
// range) — same convention as conv2d_int8_smoke after the HTA q-param fix
// (#112). HTA's ElementWiseAdd accepts UFIXED_POINT_8 directly.

#include "QnnKernelHelpers.hpp"
#include "QnnModel.hpp"
#include "QnnOpDef.h"

#include <cstdint>

#define DO_GRAPH_NODE_VALIDATIONS 1

using namespace qnn_wrapper_api;

namespace {
uint32_t g_a_dims[2] = {1, 16};
uint32_t g_b_dims[2] = {1, 16};
uint32_t g_out_dims[2] = {1, 16};

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
	QnnModel model;
	const QnnGraph_Config_t **gc = nullptr;
	VALIDATE(getQnnGraphConfigFromInfo(
				 "add_uint8", graphsConfigInfo, numGraphsConfigInfo, gc),
		err);
	VALIDATE(model.initialize(backendHandle, interface, contextHandle,
				 "add_uint8", false, DO_GRAPH_NODE_VALIDATIONS, gc),
		err);

	// Asymmetric uint8 q-params (offset=0) — works on both libQnnCpu and
	// libQnnHta, matching the convention from #112.
	Qnn_QuantizeParams_t qp_in = qparams(0.05f, 0); // real range [0, 12.75]
	Qnn_QuantizeParams_t qp_out = qparams(0.10f, 0); // real range [0, 25.5]

	Qnn_Tensor_t a{};
	a.version = QNN_TENSOR_VERSION_1;
	a.v1 = {.id = 0,
		.name = "a",
		.type = QNN_TENSOR_TYPE_APP_WRITE,
		.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
		.dataType = QNN_DATATYPE_UFIXED_POINT_8,
		.quantizeParams = qp_in,
		.rank = 2,
		.dimensions = g_a_dims,
		.memType = QNN_TENSORMEMTYPE_RAW,
		.clientBuf = {nullptr, 0}};
	VALIDATE(model.addTensor("a", &a), err);

	Qnn_Tensor_t b{};
	b.version = QNN_TENSOR_VERSION_1;
	b.v1 = {.id = 0,
		.name = "b",
		.type = QNN_TENSOR_TYPE_APP_WRITE,
		.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
		.dataType = QNN_DATATYPE_UFIXED_POINT_8,
		.quantizeParams = qp_in,
		.rank = 2,
		.dimensions = g_b_dims,
		.memType = QNN_TENSORMEMTYPE_RAW,
		.clientBuf = {nullptr, 0}};
	VALIDATE(model.addTensor("b", &b), err);

	Qnn_Tensor_t out{};
	out.version = QNN_TENSOR_VERSION_1;
	out.v1 = {.id = 0,
		.name = "output",
		.type = QNN_TENSOR_TYPE_APP_READ,
		.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
		.dataType = QNN_DATATYPE_UFIXED_POINT_8,
		.quantizeParams = qp_out,
		.rank = 2,
		.dimensions = g_out_dims,
		.memType = QNN_TENSORMEMTYPE_RAW,
		.clientBuf = {nullptr, 0}};

	const char *inputs[] = {"a", "b"};
	VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, "add_op", "qti.aisw",
				 "ElementWiseAdd", nullptr, 0, inputs, 2, &out, 1),
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
