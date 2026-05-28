// QNN kernel: fp32 elementwise add over shape [1,16].
//
// Authoring style mirrors Qualcomm's own `qnn_model_*.cpp` examples
// shipped under `share/QNN/converter/jni/`: we use the `qnn_wrapper_api::
// QnnModel` helper to declare tensors and append a single ElementWiseAdd
// node. The tool flow is:
//
//   1. g++ -shared -fPIC ... add_f32.qnn.cpp QnnModel.cpp QnnWrapperUtils.cpp
//      QnnModelPal.cpp -> libqnn_add_f32.so
//   2. qnn-context-binary-generator --model libqnn_add_f32.so
//      --backend libQnn{Gpu,Hta,Cpu}.so --binary_file add_f32 -> add_f32.bin
//      (rename to .qnn-ctx; the IREE QNN passthrough plugin reads bytes
//      verbatim).
//
// First-cut kernel — fp32 over a fixed [1,16] shape — to validate the
// authoring + build + serialization pipeline end-to-end. Subsequent
// kernels (int8, dynamic shapes, larger ranks) follow this template.

#include "QnnModel.hpp"
#include "QnnOpDef.h"

#define DO_GRAPH_NODE_VALIDATIONS 1

using namespace qnn_wrapper_api;

extern "C" {

QNN_API
ModelError_t QnnModel_composeGraphs(Qnn_BackendHandle_t backendHandle,
	QNN_INTERFACE_VER_TYPE interface, Qnn_ContextHandle_t contextHandle,
	const GraphConfigInfo_t **graphsConfigInfo,
	const uint32_t numGraphsConfigInfo, GraphInfoPtr_t **graphsInfo,
	uint32_t *numGraphsInfo, bool debug, QnnLog_Callback_t logCallback,
	QnnLog_Level_t maxLogLevel) {
	ModelError_t err = MODEL_NO_ERROR;

	QnnModel addModel;
	const QnnGraph_Config_t **graphConfigs = nullptr;
	VALIDATE(getQnnGraphConfigFromInfo("add_f32", graphsConfigInfo,
				 numGraphsConfigInfo, graphConfigs),
		err);
	VALIDATE(addModel.initialize(backendHandle, interface, contextHandle,
				 "add_f32", debug, DO_GRAPH_NODE_VALIDATIONS, graphConfigs),
		err);

	static uint32_t shape[] = {1, 16};
	Qnn_QuantizeParams_t qparams = {QNN_DEFINITION_UNDEFINED,
		QNN_QUANTIZATION_ENCODING_UNDEFINED,
		{.scaleOffsetEncoding = {0.0f, 0}}};

	auto makeAppTensor = [&](const char *name) -> Qnn_Tensor_t {
		return (Qnn_Tensor_t){.version = QNN_TENSOR_VERSION_1,
			.v1 = {.id = 0,
				.name = name,
				.type = QNN_TENSOR_TYPE_APP_WRITE,
				.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
				.dataType = QNN_DATATYPE_FLOAT_32,
				.quantizeParams = qparams,
				.rank = 2,
				.dimensions = shape,
				.memType = QNN_TENSORMEMTYPE_RAW,
				.clientBuf = {nullptr, 0}}};
	};

	Qnn_Tensor_t inputA = makeAppTensor("input_a");
	Qnn_Tensor_t inputB = makeAppTensor("input_b");
	VALIDATE(addModel.addTensor("input_a", &inputA), err);
	VALIDATE(addModel.addTensor("input_b", &inputB), err);

	Qnn_Tensor_t output = (Qnn_Tensor_t){.version = QNN_TENSOR_VERSION_1,
		.v1 = {.id = 0,
			.name = "output",
			.type = QNN_TENSOR_TYPE_APP_READ,
			.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
			.dataType = QNN_DATATYPE_FLOAT_32,
			.quantizeParams = qparams,
			.rank = 2,
			.dimensions = shape,
			.memType = QNN_TENSORMEMTYPE_RAW,
			.clientBuf = {nullptr, 0}}};

	const char *inputNames[] = {"input_a", "input_b"};
	VALIDATE(addModel.addNode(QNN_OPCONFIG_VERSION_1,
				 "add_op", // node name
				 "qti.aisw", // package
				 QNN_OP_ELEMENT_WISE_ADD,
				 /*params=*/nullptr, /*numOfParams=*/0, inputNames,
				 /*numOfInputs=*/2, &output, /*numOfOutputs=*/1),
		err);

	QnnModel *models[] = {&addModel};
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
