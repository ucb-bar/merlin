// QNN kernel: fp32 elementwise Sigmoid over shape [1,16].
//
// Authoring style: same as add_f32.qnn.cpp; we use Qualcomm's
// `qnn_wrapper_api::QnnModel` together with `merlin_qnn::makeTensor` /
// `merlin_qnn::addOp` helpers (under `tools/kernels/qnn/`) to keep the
// per-kernel boilerplate small. Single-input unary op variant.

#include "QnnKernelHelpers.hpp"
#include "QnnModel.hpp"

#define DO_GRAPH_NODE_VALIDATIONS 1

using namespace qnn_wrapper_api;
using merlin_qnn::addOp;
using merlin_qnn::fp32QuantizeParams;
using merlin_qnn::makeTensor;
using merlin_qnn::TensorSpec;

extern "C" {

QNN_API
ModelError_t QnnModel_composeGraphs(Qnn_BackendHandle_t backendHandle,
	QNN_INTERFACE_VER_TYPE interface, Qnn_ContextHandle_t contextHandle,
	const GraphConfigInfo_t **graphsConfigInfo,
	const uint32_t numGraphsConfigInfo, GraphInfoPtr_t **graphsInfo,
	uint32_t *numGraphsInfo, bool /*debug*/, QnnLog_Callback_t /*logCallback*/,
	QnnLog_Level_t /*maxLogLevel*/) {
	ModelError_t err = MODEL_NO_ERROR;

	QnnModel model;
	const QnnGraph_Config_t **graphConfigs = nullptr;
	VALIDATE(getQnnGraphConfigFromInfo("sigmoid_f32", graphsConfigInfo,
				 numGraphsConfigInfo, graphConfigs),
		err);
	VALIDATE(
		model.initialize(backendHandle, interface, contextHandle, "sigmoid_f32",
			/*debug=*/false, DO_GRAPH_NODE_VALIDATIONS, graphConfigs),
		err);

	TensorSpec inputSpec{"input", QNN_TENSOR_TYPE_APP_WRITE,
		QNN_DATATYPE_FLOAT_32, {1, 16}, fp32QuantizeParams()};
	TensorSpec outputSpec{"output", QNN_TENSOR_TYPE_APP_READ,
		QNN_DATATYPE_FLOAT_32, {1, 16}, fp32QuantizeParams()};
	Qnn_Tensor_t input = makeTensor(inputSpec);
	Qnn_Tensor_t output = makeTensor(outputSpec);
	VALIDATE(model.addTensor("input", &input), err);
	VALIDATE(
		addOp(model, "sigmoid_op", QNN_OP_SIGMOID, {"input"}, output), err);

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
