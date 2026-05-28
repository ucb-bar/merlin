// QNN graph serialization — the "qnn-graph" executable format.
//
// Walks a function full of qnn.* ops and emits a flat binary description
// that the runtime hands to Qualcomm's QNN API to build a graph at runtime
// (Adreno GPU, Hexagon HTP) or to qnn-context-binary-generator at
// compile-time-on-board (Hexagon HTA — closed compiler).
//
// Wire format (versioned, little-endian):
//
//   struct Header {
//     u32 magic = 0x6E4E5151;  // "QQNn"
//     u16 version = 1;
//     u16 backend;             // 0=cpu 1=gpu 2=htp 4=hta
//     u32 num_tensors;
//     u32 num_nodes;
//     u32 input_count;
//     u32 output_count;
//   };
//
//   for each tensor (num_tensors):
//     u32 id;
//     u32 dtype;               // QNN_DATATYPE_* enum
//     u32 rank;
//     u32 dims[rank];
//     u32 quant_kind;          // 0=undef 1=scale_offset 2=axis_scale_offset
//     // scale_offset: f32 scale, i32 offset
//     // axis_scale_offset: u32 axis, u32 num_channels, [(f32, i32)] *
//     num_channels u32 storage_kind;        // 0=app_write 1=app_read 2=static
//     3=native u32 data_size;           // for static: bytes that follow u8
//     data[data_size];
//
//   for each node (num_nodes):
//     u32 op_kind;             // QnnGraph op enum
//     u32 num_param_blobs;
//     for each param:
//       u32 name_id;           // index into a fixed param-name table
//       u32 blob_size;
//       u8 blob[blob_size];    // (params are little-endian-packed)
//     u32 num_inputs;
//     u32 input_tensor_ids[num_inputs];
//     u32 num_outputs;
//     u32 output_tensor_ids[num_outputs];
//
// The runtime walks this format once per graph load, calls QnnTensor_create
// + QnnGraph_addNode + QnnGraph_finalize, and caches the resulting
// QnnGraph_handle for subsequent dispatches.

#ifndef IREE_MERLIN_COMPILER_PLUGINS_TARGET_QNN_CODEGEN_SERIALIZEGRAPH_H_
#define IREE_MERLIN_COMPILER_PLUGINS_TARGET_QNN_CODEGEN_SERIALIZEGRAPH_H_

#include <cstdint>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::iree_compiler::QNN::Codegen {

enum class Backend : uint16_t {
	Cpu = 0,
	Gpu = 1,
	Htp = 2,
	Hta = 4,
};

// Walk every `qnn.*` op in `funcOp` and emit a binary graph description into
// `out`. On success the buffer is a self-contained qnn-graph blob.
mlir::LogicalResult serializeGraph(
	mlir::ModuleOp module, Backend backend, std::vector<int8_t> &out);

} // namespace mlir::iree_compiler::QNN::Codegen

#endif
