// QNN kernel: standalone uint8 ElementWiseNeuron(Relu). For
// comparison vs the fused conv+relu kernel — paired via IREE chain
// to measure the 2-dispatch overhead this pattern saves.
#include "QnnKernelHelpers.hpp"
#include "QnnModel.hpp"
#include "QnnOpDef.h"
#include <cstdint>
#define DO_GRAPH_NODE_VALIDATIONS 1
using namespace qnn_wrapper_api;
namespace {
uint32_t g_in_dims[4] = {1, 6, 6, 4};
uint32_t g_out_dims[4] = {1, 6, 6, 4};
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
	uint32_t *numGraphsInfo, bool, QnnLog_Callback_t, QnnLog_Level_t) {
	ModelError_t err = MODEL_NO_ERROR;
	QnnModel model;
	const QnnGraph_Config_t **gc = nullptr;
	VALIDATE(getQnnGraphConfigFromInfo(
				 "relu_int8_smoke", graphsConfigInfo, numGraphsConfigInfo, gc),
		err);
	VALIDATE(model.initialize(backendHandle, interface, contextHandle,
				 "relu_int8_smoke", false, DO_GRAPH_NODE_VALIDATIONS, gc),
		err);
	Qnn_QuantizeParams_t qp = qparams(0.10f, 0);
	Qnn_Tensor_t input{};
	input.version = QNN_TENSOR_VERSION_1;
	input.v1 = {.id = 0,
		.name = "input",
		.type = QNN_TENSOR_TYPE_APP_WRITE,
		.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
		.dataType = QNN_DATATYPE_UFIXED_POINT_8,
		.quantizeParams = qp,
		.rank = 4,
		.dimensions = g_in_dims,
		.memType = QNN_TENSORMEMTYPE_RAW,
		.clientBuf = {nullptr, 0}};
	VALIDATE(model.addTensor("input", &input), err);
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
		.quantizeParams = qp,
		.rank = 4,
		.dimensions = g_out_dims,
		.memType = QNN_TENSORMEMTYPE_RAW,
		.clientBuf = {nullptr, 0}};
	const char *inputs[] = {"input"};
	VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, "relu_op", "qti.aisw",
				 QNN_OP_ELEMENT_WISE_NEURON, neuron_params, 1, inputs, 1,
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
}
