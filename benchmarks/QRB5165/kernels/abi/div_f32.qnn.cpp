// QNN kernel: fp32 elementwise Divide over shape [1,16].
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
	uint32_t *numGraphsInfo, bool /*debug*/, QnnLog_Callback_t /*lc*/,
	QnnLog_Level_t /*ll*/) {
	ModelError_t err = MODEL_NO_ERROR;
	QnnModel model;
	const QnnGraph_Config_t **gc = nullptr;
	VALIDATE(getQnnGraphConfigFromInfo(
				 "div_f32", graphsConfigInfo, numGraphsConfigInfo, gc),
		err);
	VALIDATE(model.initialize(backendHandle, interface, contextHandle,
				 "div_f32", false, DO_GRAPH_NODE_VALIDATIONS, gc),
		err);

	TensorSpec a{"a", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32, {1, 16},
		fp32QuantizeParams()};
	TensorSpec b{"b", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_32, {1, 16},
		fp32QuantizeParams()};
	TensorSpec out{"output", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_32,
		{1, 16}, fp32QuantizeParams()};
	Qnn_Tensor_t ta = makeTensor(a), tb = makeTensor(b), to = makeTensor(out);
	VALIDATE(model.addTensor("a", &ta), err);
	VALIDATE(model.addTensor("b", &tb), err);
	VALIDATE(addOp(model, "div_op", QNN_OP_ELEMENT_WISE_DIVIDE, {"a", "b"}, to),
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
