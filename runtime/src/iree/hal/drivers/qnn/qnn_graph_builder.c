// QNN graph builder runtime — see qnn_graph_builder.h for format details.
//
// Walks the binary qnn-graph blob produced by the compiler's
// SerializeGraph.cpp and constructs a finalized QnnGraph_handle_t at
// load time via the QNN API. Per-op materializers convert each node
// record into a Qnn_OpConfig_t with the right packageName / typeName /
// params and add it via graphAddNode. Tensors are pre-registered with
// tensorCreateGraphTensor before any op references them.

#include "qnn_graph_builder.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "iree/base/api.h"

#include "QnnCommon.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnInterface.h"
#include "QnnOpDef.h"
#include "QnnTensor.h"
#include "QnnTypes.h"

const char IREE_HAL_QNN_GRAPH_FORMAT[] = "qnn-graph";

static bool qnn_trace_enabled(void) {
	const char *v = getenv("IREE_HAL_QNN_TRACE");
	return v && v[0] && strcmp(v, "0") != 0;
}

// Wire-format constants must match SerializeGraph.cpp's emitter.
static const uint32_t kMagic = 0x6E4E5151u; // "QQNn"
static const uint16_t kVersion = 1;

// OpKind enum mirrors SerializeGraph.cpp's OpKind. Keep in sync.
enum {
	kOpConv2d = 1,
	kOpDepthwiseConv2d = 2,
	kOpFullyConnected = 3,
	kOpMatMul = 4,
	kOpElementWiseNeuron = 5,
	kOpElementWiseBinary = 6,
	kOpPoolMax2d = 7,
	kOpPoolAvg2d = 8,
	kOpConcat = 9,
	kOpReshape = 10,
	kOpTranspose = 11,
	kOpQuantize = 12,
	kOpDequantize = 13,
	kOpPad = 14,
	kOpSoftmax = 15,
	kOpReduce = 16,
};

//===----------------------------------------------------------------------===//
// Little-endian readers + cursor (the wire format is fixed-LE per spec).
//===----------------------------------------------------------------------===//

typedef struct {
	const uint8_t *base;
	const uint8_t *p;
	const uint8_t *end;
} reader_t;

static iree_status_t r_init(reader_t *r, const uint8_t *data, size_t size) {
	r->base = data;
	r->p = data;
	r->end = data + size;
	return iree_ok_status();
}
static iree_status_t r_need(reader_t *r, size_t n) {
	if ((size_t)(r->end - r->p) < n) {
		return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
			"qnn-graph blob truncated (need %zu bytes at offset %zu)", n,
			(size_t)(r->p - r->base));
	}
	return iree_ok_status();
}
static iree_status_t r_u32(reader_t *r, uint32_t *out) {
	IREE_RETURN_IF_ERROR(r_need(r, 4));
	*out = ((uint32_t)r->p[0]) | ((uint32_t)r->p[1] << 8) |
		((uint32_t)r->p[2] << 16) | ((uint32_t)r->p[3] << 24);
	r->p += 4;
	return iree_ok_status();
}
static iree_status_t r_u16(reader_t *r, uint16_t *out) {
	IREE_RETURN_IF_ERROR(r_need(r, 2));
	*out = ((uint16_t)r->p[0]) | ((uint16_t)r->p[1] << 8);
	r->p += 2;
	return iree_ok_status();
}
static iree_status_t r_blob(reader_t *r, size_t n, const uint8_t **out_p) {
	IREE_RETURN_IF_ERROR(r_need(r, n));
	*out_p = r->p;
	r->p += n;
	return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// QNN dtype mapping. Mirrors SerializeGraph.cpp::qnnDtype.
//===----------------------------------------------------------------------===//

static Qnn_DataType_t map_dtype(uint32_t code) {
	switch (code) {
		case 0x0308:
			// SFIXED_POINT_8 — signed int8 storage [-128,127]. yolov8 + most
			// pt2e int8 quant flows produce this with zp=0.
			return QNN_DATATYPE_SFIXED_POINT_8;
		case 0x0408:
			// UFIXED_POINT_8 (unsigned 0-255). HTA prefers this with zp=128
			// for symmetric int8 representations; some frontends emit it
			// directly.
			return QNN_DATATYPE_UFIXED_POINT_8;
		case 0x0316:
			return QNN_DATATYPE_SFIXED_POINT_16;
		case 0x0416:
			return QNN_DATATYPE_UFIXED_POINT_16;
		case 0x0032:
			// Legacy / fallback: raw INT_32 code from older VMFBs. HTA accepts
			// SFIXED_POINT_32 (with default scale=1, off=0) but not raw INT_32
			// for graph data tensors. We promote to SFIXED_POINT_32 here.
			return QNN_DATATYPE_SFIXED_POINT_32;
		case 0x0332:
			// SerializeGraph emits this for !quant.uniform<i32:f32, s>-wrapped
			// tensors (Conv2d bias). HTA's op-package validator requires this
			// exact dtype code for bias; the legacy 0x0032 above was a
			// workaround that some HTA op variants reject.
			return QNN_DATATYPE_SFIXED_POINT_32;
		case 0x0216:
			return QNN_DATATYPE_FLOAT_16;
		case 0x0232:
			return QNN_DATATYPE_FLOAT_32;
		default:
			return QNN_DATATYPE_UNDEFINED;
	}
}

// Returns true for any QNN_DATATYPE_*FIXED_POINT_* that needs quantize
// params populated.
static bool dtype_needs_quant_params(Qnn_DataType_t dt) {
	switch (dt) {
		case QNN_DATATYPE_SFIXED_POINT_8:
		case QNN_DATATYPE_SFIXED_POINT_16:
		case QNN_DATATYPE_SFIXED_POINT_32:
		case QNN_DATATYPE_UFIXED_POINT_8:
		case QNN_DATATYPE_UFIXED_POINT_16:
		case QNN_DATATYPE_UFIXED_POINT_32:
			return true;
		default:
			return false;
	}
}

//===----------------------------------------------------------------------===//
// Decoded tables.
//===----------------------------------------------------------------------===//

typedef struct {
	uint32_t id;
	Qnn_DataType_t dtype;
	uint32_t rank;
	uint32_t *dims; // owned, length = rank
	bool is_input; // marked after we know inputs/outputs ordering
	bool is_output;
	bool is_static; // STATIC tensor with embedded data (storage_kind=2)
	bool has_quant;
	float quant_scale;
	int32_t quant_offset;
	// Per-axis (Phase 3) quant: when has_per_axis_quant is true, the
	// tensor uses Qnn_AxisScaleOffset_t encoding. axis is the channel
	// dim; per_axis_count is the channel count; the arrays below have
	// length per_axis_count.
	bool has_per_axis_quant;
	uint32_t per_axis_axis;
	uint32_t per_axis_count;
	float *per_axis_scales; // owned, length per_axis_count
	int32_t *per_axis_offsets; // owned, length per_axis_count
	Qnn_ScaleOffset_t
		*per_axis_pairs; // owned, length per_axis_count (for QNN API)
	void *static_data; // owned aligned copy when is_static; NULL otherwise
	uint32_t static_size;
	Qnn_Tensor_t tensor; // populated by build step
	char name[32]; // synthesized "t<id>"
} qnn_tensor_decl_t;

typedef struct {
	uint32_t op_kind;
	uint32_t num_params;
	// Each param: { uint32_t name; uint32_t size; uint8_t* data (length=size)
	// }. We keep raw param bytes around for the lifetime of the build because
	// QNN's op-config references them by pointer.
	struct {
		uint32_t name;
		uint32_t size;
		const uint8_t *data;
	} *params;
	uint32_t num_inputs;
	uint32_t *input_ids;
	uint32_t num_outputs;
	uint32_t *output_ids;
	char op_name[24]; // synthesized "n<index>"
} qnn_node_decl_t;

typedef struct {
	iree_allocator_t alloc;

	// Decoded tables.
	uint32_t num_tensors;
	qnn_tensor_decl_t *tensors;
	uint32_t num_nodes;
	qnn_node_decl_t *nodes;
	uint32_t input_count;
	uint32_t output_count;

	// Backed param-tensor storage. QNN's tensor-typed params want a real
	// Qnn_Tensor_t pointing at heap data. We materialize each
	// (stride/pad/dilation/...) as a static tensor and keep handles here
	// so they outlive the addNode call.
	Qnn_Tensor_t *param_tensors;
	iree_host_size_t param_tensor_count;
	iree_host_size_t param_tensor_capacity;
	uint32_t *param_tensor_dims_storage; // bumped allocator
	iree_host_size_t param_tensor_dims_used;
	iree_host_size_t param_tensor_dims_capacity;

	// String storage for tensor + op names (lifetime = builder).
	char *string_pool;
	iree_host_size_t string_pool_used;
	iree_host_size_t string_pool_capacity;
} qnn_decoded_t;

static void decoded_destroy(qnn_decoded_t *d) {
	if (!d)
		return;
	if (d->tensors) {
		for (uint32_t i = 0; i < d->num_tensors; ++i) {
			iree_allocator_free(d->alloc, d->tensors[i].dims);
			iree_allocator_free(d->alloc, d->tensors[i].static_data);
			iree_allocator_free(d->alloc, d->tensors[i].per_axis_scales);
			iree_allocator_free(d->alloc, d->tensors[i].per_axis_offsets);
			iree_allocator_free(d->alloc, d->tensors[i].per_axis_pairs);
		}
		iree_allocator_free(d->alloc, d->tensors);
	}
	if (d->nodes) {
		for (uint32_t i = 0; i < d->num_nodes; ++i) {
			iree_allocator_free(d->alloc, d->nodes[i].params);
			iree_allocator_free(d->alloc, d->nodes[i].input_ids);
			iree_allocator_free(d->alloc, d->nodes[i].output_ids);
		}
		iree_allocator_free(d->alloc, d->nodes);
	}
	if (d->param_tensors) {
		// Free the aligned data buffers we copied for each STATIC param.
		for (iree_host_size_t i = 0; i < d->param_tensor_count; ++i) {
			iree_allocator_free(
				d->alloc, d->param_tensors[i].v1.clientBuf.data);
		}
	}
	iree_allocator_free(d->alloc, d->param_tensors);
	iree_allocator_free(d->alloc, d->param_tensor_dims_storage);
	iree_allocator_free(d->alloc, d->string_pool);
}

//===----------------------------------------------------------------------===//
// Decode pass: walk the binary, populate the tables. No QNN calls yet.
//===----------------------------------------------------------------------===//

static iree_status_t decode_blob(const uint8_t *data, size_t size,
	iree_allocator_t alloc, qnn_decoded_t *out) {
	memset(out, 0, sizeof(*out));
	out->alloc = alloc;

	reader_t r;
	IREE_RETURN_IF_ERROR(r_init(&r, data, size));

	// Header.
	uint32_t magic;
	uint16_t version, backend;
	IREE_RETURN_IF_ERROR(r_u32(&r, &magic));
	if (magic != kMagic) {
		return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
			"qnn-graph blob magic mismatch (got 0x%08x)", magic);
	}
	IREE_RETURN_IF_ERROR(r_u16(&r, &version));
	IREE_RETURN_IF_ERROR(r_u16(&r, &backend));
	if (version != kVersion) {
		return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
			"qnn-graph blob version %u not supported (this runtime expects %u)",
			version, kVersion);
	}
	IREE_RETURN_IF_ERROR(r_u32(&r, &out->num_tensors));
	IREE_RETURN_IF_ERROR(r_u32(&r, &out->num_nodes));
	IREE_RETURN_IF_ERROR(r_u32(&r, &out->input_count));
	IREE_RETURN_IF_ERROR(r_u32(&r, &out->output_count));

	// Tensors.
	IREE_RETURN_IF_ERROR(iree_allocator_malloc(alloc,
		out->num_tensors * sizeof(qnn_tensor_decl_t), (void **)&out->tensors));
	memset(out->tensors, 0, out->num_tensors * sizeof(qnn_tensor_decl_t));
	for (uint32_t i = 0; i < out->num_tensors; ++i) {
		qnn_tensor_decl_t *t = &out->tensors[i];
		uint32_t dtype_code, rank, quant_kind, storage_kind, data_size;
		IREE_RETURN_IF_ERROR(r_u32(&r, &t->id));
		IREE_RETURN_IF_ERROR(r_u32(&r, &dtype_code));
		IREE_RETURN_IF_ERROR(r_u32(&r, &rank));
		t->dtype = map_dtype(dtype_code);
		t->rank = rank;
		if (rank > 0) {
			IREE_RETURN_IF_ERROR(iree_allocator_malloc(
				alloc, rank * sizeof(uint32_t), (void **)&t->dims));
			for (uint32_t d = 0; d < rank; ++d) {
				IREE_RETURN_IF_ERROR(r_u32(&r, &t->dims[d]));
			}
		}
		IREE_RETURN_IF_ERROR(r_u32(&r, &quant_kind));
		if (quant_kind == 1) {
			// Per-tensor scale (f32) + offset (i32).
			uint32_t scale_bits, off_bits;
			IREE_RETURN_IF_ERROR(r_u32(&r, &scale_bits));
			IREE_RETURN_IF_ERROR(r_u32(&r, &off_bits));
			memcpy(&t->quant_scale, &scale_bits, sizeof(float));
			t->quant_offset = (int32_t)off_bits;
			t->has_quant = true;
		} else if (quant_kind == 2) {
			// Per-axis (Phase 3): axis u32, count u32, then
			// count*(scale+offset).
			uint32_t axis, count;
			IREE_RETURN_IF_ERROR(r_u32(&r, &axis));
			IREE_RETURN_IF_ERROR(r_u32(&r, &count));
			t->has_per_axis_quant = true;
			t->per_axis_axis = axis;
			t->per_axis_count = count;
			if (count > 0) {
				IREE_RETURN_IF_ERROR(iree_allocator_malloc(alloc,
					count * sizeof(float), (void **)&t->per_axis_scales));
				IREE_RETURN_IF_ERROR(iree_allocator_malloc(alloc,
					count * sizeof(int32_t), (void **)&t->per_axis_offsets));
				IREE_RETURN_IF_ERROR(iree_allocator_malloc(alloc,
					count * sizeof(Qnn_ScaleOffset_t),
					(void **)&t->per_axis_pairs));
				for (uint32_t k = 0; k < count; ++k) {
					uint32_t scale_bits, off_bits;
					IREE_RETURN_IF_ERROR(r_u32(&r, &scale_bits));
					IREE_RETURN_IF_ERROR(r_u32(&r, &off_bits));
					memcpy(&t->per_axis_scales[k], &scale_bits, sizeof(float));
					t->per_axis_offsets[k] = (int32_t)off_bits;
					t->per_axis_pairs[k].scale = t->per_axis_scales[k];
					t->per_axis_pairs[k].offset = t->per_axis_offsets[k];
				}
			}
		}
		IREE_RETURN_IF_ERROR(r_u32(&r, &storage_kind));
		IREE_RETURN_IF_ERROR(r_u32(&r, &data_size));
		if (data_size > 0) {
			const uint8_t *data_p;
			IREE_RETURN_IF_ERROR(r_blob(&r, data_size, &data_p));
			if (storage_kind == 2) {
				// STATIC tensor: copy data into an aligned heap buffer that
				// outlives the wire blob. HTA's lazy reads need 4-byte
				// alignment; the wire blob is only u32-aligned globally and
				// packed tensor data may end up at sub-aligned offsets.
				IREE_RETURN_IF_ERROR(
					iree_allocator_malloc(alloc, data_size, &t->static_data));
				memcpy(t->static_data, data_p, data_size);
				t->static_size = data_size;
				t->is_static = true;
			}
		}
		snprintf(t->name, sizeof(t->name), "t%u", t->id);
	}

	// Nodes.
	IREE_RETURN_IF_ERROR(iree_allocator_malloc(
		alloc, out->num_nodes * sizeof(qnn_node_decl_t), (void **)&out->nodes));
	memset(out->nodes, 0, out->num_nodes * sizeof(qnn_node_decl_t));
	for (uint32_t i = 0; i < out->num_nodes; ++i) {
		qnn_node_decl_t *n = &out->nodes[i];
		IREE_RETURN_IF_ERROR(r_u32(&r, &n->op_kind));
		IREE_RETURN_IF_ERROR(r_u32(&r, &n->num_params));
		if (n->num_params > 0) {
			IREE_RETURN_IF_ERROR(iree_allocator_malloc(alloc,
				n->num_params * sizeof(*n->params), (void **)&n->params));
		}
		for (uint32_t p = 0; p < n->num_params; ++p) {
			IREE_RETURN_IF_ERROR(r_u32(&r, &n->params[p].name));
			IREE_RETURN_IF_ERROR(r_u32(&r, &n->params[p].size));
			IREE_RETURN_IF_ERROR(
				r_blob(&r, n->params[p].size, &n->params[p].data));
		}
		IREE_RETURN_IF_ERROR(r_u32(&r, &n->num_inputs));
		if (n->num_inputs > 0) {
			IREE_RETURN_IF_ERROR(iree_allocator_malloc(alloc,
				n->num_inputs * sizeof(uint32_t), (void **)&n->input_ids));
			for (uint32_t k = 0; k < n->num_inputs; ++k) {
				IREE_RETURN_IF_ERROR(r_u32(&r, &n->input_ids[k]));
			}
		}
		IREE_RETURN_IF_ERROR(r_u32(&r, &n->num_outputs));
		if (n->num_outputs > 0) {
			IREE_RETURN_IF_ERROR(iree_allocator_malloc(alloc,
				n->num_outputs * sizeof(uint32_t), (void **)&n->output_ids));
			for (uint32_t k = 0; k < n->num_outputs; ++k) {
				IREE_RETURN_IF_ERROR(r_u32(&r, &n->output_ids[k]));
			}
		}
		snprintf(n->op_name, sizeof(n->op_name), "n%u", i);
	}

	return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Tensor materialization. The first `input_count` tensors are graph
// inputs (APP_WRITE), the next `output_count` are graph outputs
// (APP_READ); the rest are NATIVE intermediate tensors.
//===----------------------------------------------------------------------===//

static iree_status_t build_tensors(
	qnn_decoded_t *d, void *graph_handle, const QnnInterface_t *iface) {
	// Compiler currently emits input_count=output_count=0 in the header
	// (TODO: derive from function signature). Compute them from node
	// connectivity instead: a tensor that's never produced by any node
	// is a graph input; one that's never consumed is a graph output.
	bool *is_produced = NULL;
	bool *is_consumed = NULL;
	iree_status_t st = iree_allocator_malloc(
		d->alloc, d->num_tensors * sizeof(bool), (void **)&is_produced);
	if (!iree_status_is_ok(st))
		return st;
	st = iree_allocator_malloc(
		d->alloc, d->num_tensors * sizeof(bool), (void **)&is_consumed);
	if (!iree_status_is_ok(st)) {
		iree_allocator_free(d->alloc, is_produced);
		return st;
	}
	memset(is_produced, 0, d->num_tensors * sizeof(bool));
	memset(is_consumed, 0, d->num_tensors * sizeof(bool));
	for (uint32_t ni = 0; ni < d->num_nodes; ++ni) {
		qnn_node_decl_t *n = &d->nodes[ni];
		for (uint32_t k = 0; k < n->num_inputs; ++k) {
			for (uint32_t t = 0; t < d->num_tensors; ++t) {
				if (d->tensors[t].id == n->input_ids[k]) {
					is_consumed[t] = true;
					break;
				}
			}
		}
		for (uint32_t k = 0; k < n->num_outputs; ++k) {
			for (uint32_t t = 0; t < d->num_tensors; ++t) {
				if (d->tensors[t].id == n->output_ids[k]) {
					is_produced[t] = true;
					break;
				}
			}
		}
	}

	for (uint32_t i = 0; i < d->num_tensors; ++i) {
		qnn_tensor_decl_t *t = &d->tensors[i];
		Qnn_TensorType_t type;
		if (t->is_static) {
			// STATIC constants get priority over input/output classification.
			type = QNN_TENSOR_TYPE_STATIC;
		} else if (!is_produced[i] && is_consumed[i]) {
			// Source tensor with no producer + at least one consumer ⇒ input.
			type = QNN_TENSOR_TYPE_APP_WRITE;
			t->is_input = true;
		} else if (is_produced[i] && !is_consumed[i]) {
			type = QNN_TENSOR_TYPE_APP_READ;
			t->is_output = true;
		} else if (!is_produced[i] && !is_consumed[i]) {
			// Dangling — make NATIVE for safety.
			type = QNN_TENSOR_TYPE_NATIVE;
		} else {
			type = QNN_TENSOR_TYPE_NATIVE;
		}
		Qnn_Tensor_t qt = QNN_TENSOR_INIT;
		qt.version = QNN_TENSOR_VERSION_1;
		qt.v1.id = 0; // QNN assigns
		qt.v1.name = t->name;
		qt.v1.type = type;
		qt.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
		qt.v1.dataType = t->dtype;
		qt.v1.quantizeParams = (Qnn_QuantizeParams_t)QNN_QUANTIZE_PARAMS_INIT;
		if (dtype_needs_quant_params(t->dtype)) {
			qt.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
			if (t->has_per_axis_quant) {
				// Phase 3: per-channel/axis quantization
				// (Qnn_AxisScaleOffset_t). Required for most yolov8 conv
				// weights (per-output-channel).
				qt.v1.quantizeParams.quantizationEncoding =
					QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
				qt.v1.quantizeParams.axisScaleOffsetEncoding =
					(Qnn_AxisScaleOffset_t){.axis = (int32_t)t->per_axis_axis,
						.numScaleOffsets = t->per_axis_count,
						.scaleOffset = t->per_axis_pairs};
			} else {
				// Per-tensor scale/offset from wire format if present, else
				// benign default so QNN accepts the tensor.
				qt.v1.quantizeParams.quantizationEncoding =
					QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
				qt.v1.quantizeParams.scaleOffsetEncoding = (Qnn_ScaleOffset_t){
					.scale = t->has_quant ? t->quant_scale : 1.0f,
					.offset = t->has_quant ? t->quant_offset : 0};
			}
		}
		qt.v1.rank = t->rank;
		qt.v1.dimensions = t->dims;
		qt.v1.memType = QNN_TENSORMEMTYPE_RAW;
		if (t->is_static && t->static_data) {
			qt.v1.clientBuf = (Qnn_ClientBuffer_t){
				.data = t->static_data, .dataSize = t->static_size};
		} else {
			qt.v1.clientBuf = (Qnn_ClientBuffer_t){.data = NULL, .dataSize = 0};
		}
		if (qnn_trace_enabled()) {
			fprintf(stderr,
				"[qnn-graph] tensor id=%u name=%s type=%d dtype=%d rank=%u "
				"dims=",
				t->id, t->name, (int)type, (int)t->dtype, t->rank);
			for (uint32_t di = 0; di < t->rank; ++di) {
				fprintf(stderr, "%s%u", di ? "x" : "", t->dims[di]);
			}
			if (t->has_per_axis_quant) {
				fprintf(stderr, " q=axis(axis=%u,count=%u)", t->per_axis_axis,
					t->per_axis_count);
			} else if (t->has_quant) {
				fprintf(stderr, " q=(scale=%g,zp=%d)", t->quant_scale,
					t->quant_offset);
			} else {
				fprintf(stderr, " q=<default>");
			}
			fprintf(stderr, " static=%u static_size=%u\n",
				t->is_static ? 1u : 0u, t->static_size);
		}
		Qnn_ErrorHandle_t rc =
			iface->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(
				(Qnn_GraphHandle_t)graph_handle, &qt);
		if (rc != QNN_SUCCESS) {
			iree_allocator_free(d->alloc, is_produced);
			iree_allocator_free(d->alloc, is_consumed);
			return iree_make_status(IREE_STATUS_INTERNAL,
				"tensorCreateGraphTensor rc=%lld for tensor #%u (id=%u, "
				"name=%s, "
				"type=%d)",
				(long long)rc, i, t->id, t->name, (int)type);
		}
		t->tensor = qt;
	}
	iree_allocator_free(d->alloc, is_produced);
	iree_allocator_free(d->alloc, is_consumed);
	return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Param-tensor synthesis. Many QNN op params are typed as static tensors
// (e.g. Conv2d's stride is a u32[2] tensor). We pull the raw param bytes
// out of the blob, allocate a fresh Qnn_Tensor_t (STATIC) referencing
// them, and the resulting tensor goes into the Qnn_Param_t.tensorParam
// slot of the op config.
//===----------------------------------------------------------------------===//

static iree_status_t make_static_tensor_param(qnn_decoded_t *d,
	const char *name, Qnn_DataType_t dtype, uint32_t rank, const uint32_t *dims,
	const void *data, uint32_t data_size, Qnn_Param_t *out_param) {
	// Reserve a Qnn_Tensor_t slot.
	if (d->param_tensor_count == d->param_tensor_capacity) {
		iree_host_size_t new_cap =
			d->param_tensor_capacity ? d->param_tensor_capacity * 2 : 32;
		Qnn_Tensor_t *nt = NULL;
		IREE_RETURN_IF_ERROR(iree_allocator_malloc(
			d->alloc, new_cap * sizeof(Qnn_Tensor_t), (void **)&nt));
		if (d->param_tensor_count > 0) {
			memcpy(nt, d->param_tensors,
				d->param_tensor_count * sizeof(Qnn_Tensor_t));
		}
		iree_allocator_free(d->alloc, d->param_tensors);
		d->param_tensors = nt;
		d->param_tensor_capacity = new_cap;
	}
	// Reserve dims storage.
	if (d->param_tensor_dims_used + rank > d->param_tensor_dims_capacity) {
		iree_host_size_t new_cap = d->param_tensor_dims_capacity
			? d->param_tensor_dims_capacity * 2
			: 256;
		while (new_cap < d->param_tensor_dims_used + rank)
			new_cap *= 2;
		uint32_t *nd = NULL;
		IREE_RETURN_IF_ERROR(iree_allocator_malloc(
			d->alloc, new_cap * sizeof(uint32_t), (void **)&nd));
		if (d->param_tensor_dims_used > 0) {
			memcpy(nd, d->param_tensor_dims_storage,
				d->param_tensor_dims_used * sizeof(uint32_t));
		}
		iree_allocator_free(d->alloc, d->param_tensor_dims_storage);
		d->param_tensor_dims_storage = nd;
		d->param_tensor_dims_capacity = new_cap;
	}
	uint32_t *dims_slot =
		d->param_tensor_dims_storage + d->param_tensor_dims_used;
	for (uint32_t i = 0; i < rank; ++i)
		dims_slot[i] = dims[i];
	d->param_tensor_dims_used += rank;

	// Copy param data into an aligned heap buffer. The wire blob is
	// packed and may be read at non-natural alignment by HTA's
	// do_append_const_node which uses memcpy with strict aarch64
	// alignment expectations. Owning the buffer also guarantees
	// lifetime past the executable_data blob.
	void *data_owned = NULL;
	iree_status_t st = iree_allocator_malloc(d->alloc, data_size, &data_owned);
	if (!iree_status_is_ok(st))
		return st;
	memcpy(data_owned, data, data_size);

	Qnn_Tensor_t *qt = &d->param_tensors[d->param_tensor_count++];
	*qt = (Qnn_Tensor_t)QNN_TENSOR_INIT;
	qt->version = QNN_TENSOR_VERSION_1;
	qt->v1.id = 0;
	qt->v1.name = name;
	qt->v1.type = QNN_TENSOR_TYPE_STATIC;
	qt->v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
	qt->v1.dataType = dtype;
	qt->v1.quantizeParams = (Qnn_QuantizeParams_t)QNN_QUANTIZE_PARAMS_INIT;
	qt->v1.rank = rank;
	qt->v1.dimensions = dims_slot;
	qt->v1.memType = QNN_TENSORMEMTYPE_RAW;
	qt->v1.clientBuf =
		(Qnn_ClientBuffer_t){.data = data_owned, .dataSize = data_size};

	*out_param = (Qnn_Param_t)QNN_PARAM_INIT;
	out_param->paramType = QNN_PARAMTYPE_TENSOR;
	out_param->name = name;
	out_param->tensorParam = *qt;
	return iree_ok_status();
}

// Pre-register all stashed param tensors with the graph. HTA validation
// rejects tensorParam references whose underlying tensor wasn't first
// announced via tensorCreateGraphTensor.
static iree_status_t register_param_tensors(
	qnn_decoded_t *d, void *graph_handle, const QnnInterface_t *iface) {
	for (iree_host_size_t i = 0; i < d->param_tensor_count; ++i) {
		Qnn_ErrorHandle_t rc =
			iface->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(
				(Qnn_GraphHandle_t)graph_handle, &d->param_tensors[i]);
		if (rc != QNN_SUCCESS) {
			return iree_make_status(IREE_STATUS_INTERNAL,
				"tensorCreateGraphTensor (param) rc=%lld for param tensor #%zu",
				(long long)rc, (size_t)i);
		}
	}
	return iree_ok_status();
}

static Qnn_Param_t make_scalar_u32_param(const char *name, uint32_t value) {
	Qnn_Param_t p = QNN_PARAM_INIT;
	p.paramType = QNN_PARAMTYPE_SCALAR;
	p.name = name;
	p.scalarParam.dataType = QNN_DATATYPE_UINT_32;
	p.scalarParam.uint32Value = value;
	return p;
}

static Qnn_Param_t make_scalar_bool_param(const char *name, bool value) {
	Qnn_Param_t p = QNN_PARAM_INIT;
	p.paramType = QNN_PARAMTYPE_SCALAR;
	p.name = name;
	p.scalarParam.dataType = QNN_DATATYPE_BOOL_8;
	p.scalarParam.bool8Value = value ? 1 : 0;
	return p;
}

//===----------------------------------------------------------------------===//
// Per-op materializers. Each fills out a Qnn_OpConfig_t with the proper
// Qualcomm typeName + params, then calls graphAddNode.
//===----------------------------------------------------------------------===//

// Map our serialized i32 stride/pad/dilation arrays (which are in
// little-endian raw bytes from the blob) to QNN's u32 tensor params.
// Returns the param count actually populated and a pointer to a
// Qnn_Param_t array that lives in the static `params_out` storage.
static iree_status_t build_conv2d_params(qnn_decoded_t *d,
	const qnn_node_decl_t *n, Qnn_Param_t *params_out, uint32_t *out_count) {
	// Expected param layout (matches SerializeGraph.cpp):
	//   name=0 stride         size=8  (2 i32)
	//   name=1 pad_amount     size=16 (4 i32)
	//   name=2 dilation       size=8  (2 i32)
	//   name=3 group          size=4  (1 i32)
	uint32_t cnt = 0;
	for (uint32_t i = 0; i < n->num_params; ++i) {
		switch (n->params[i].name) {
			case 0: { // stride
				uint32_t dims[1] = {2};
				IREE_RETURN_IF_ERROR(make_static_tensor_param(d, "stride",
					QNN_DATATYPE_UINT_32, 1, dims, n->params[i].data,
					n->params[i].size, &params_out[cnt++]));
				break;
			}
			case 1: { // pad_amount [2,2]
				uint32_t dims[2] = {2, 2};
				IREE_RETURN_IF_ERROR(make_static_tensor_param(d, "pad_amount",
					QNN_DATATYPE_UINT_32, 2, dims, n->params[i].data,
					n->params[i].size, &params_out[cnt++]));
				break;
			}
			case 2: { // dilation
				uint32_t dims[1] = {2};
				IREE_RETURN_IF_ERROR(make_static_tensor_param(d, "dilation",
					QNN_DATATYPE_UINT_32, 1, dims, n->params[i].data,
					n->params[i].size, &params_out[cnt++]));
				break;
			}
			case 3: { // group
				uint32_t group = 1;
				if (n->params[i].size >= 4) {
					memcpy(&group, n->params[i].data, 4);
				}
				params_out[cnt++] = make_scalar_u32_param("group", group);
				break;
			}
			default:
				// Unknown param — skip gracefully.
				break;
		}
	}
	*out_count = cnt;
	return iree_ok_status();
}

// PoolMax2d / PoolAvg2d share params: filter_size, stride, pad_amount.
static iree_status_t build_pool_params(qnn_decoded_t *d,
	const qnn_node_decl_t *n, Qnn_Param_t *params_out, uint32_t *out_count) {
	uint32_t cnt = 0;
	for (uint32_t i = 0; i < n->num_params; ++i) {
		switch (n->params[i].name) {
			case 0: { // filter_size
				uint32_t dims[1] = {2};
				IREE_RETURN_IF_ERROR(make_static_tensor_param(d, "filter_size",
					QNN_DATATYPE_UINT_32, 1, dims, n->params[i].data,
					n->params[i].size, &params_out[cnt++]));
				break;
			}
			case 1: { // stride
				uint32_t dims[1] = {2};
				IREE_RETURN_IF_ERROR(make_static_tensor_param(d, "stride",
					QNN_DATATYPE_UINT_32, 1, dims, n->params[i].data,
					n->params[i].size, &params_out[cnt++]));
				break;
			}
			case 2: { // pad_amount
				uint32_t dims[2] = {2, 2};
				IREE_RETURN_IF_ERROR(make_static_tensor_param(d, "pad_amount",
					QNN_DATATYPE_UINT_32, 2, dims, n->params[i].data,
					n->params[i].size, &params_out[cnt++]));
				break;
			}
			default:
				break;
		}
	}
	*out_count = cnt;
	return iree_ok_status();
}

// Map our ElementWiseBinary op_kind to QNN's typeName. Returns NULL on
// unknown kinds — callers must surface this as an error rather than
// silently substituting "ElementWiseAdd" (the prior behavior masked
// serialization bugs).
static const char *binary_typename(uint32_t op_kind) {
	switch (op_kind) {
		case 0:
			return "ElementWiseAdd";
		case 1:
			return "ElementWiseSubtract";
		case 2:
			return "ElementWiseMultiply";
		case 3:
			return "ElementWiseDivide";
		default:
			return NULL;
	}
}

// Map our ElementWiseNeuron op_kind to QNN's "operation" enum value.
// QnnOpDef.h: 0=Relu, 1=Relu6, 2=Sigmoid, 3=Tanh.
static uint32_t neuron_operation(uint32_t op_kind) {
	return op_kind; // same enum
}

// Build Qnn_OpConfig_t for a single node. Resolves typeName + params,
// then collects input/output Qnn_Tensor_t handles from the decoded
// table by id. The op-config and its params/inputs/outputs arrays must
// outlive the addNode call; we pass stack-allocated arrays sized to the
// maxima below and copy on demand.
#define MAX_PARAMS 8
#define MAX_INPUTS 8
#define MAX_OUTPUTS 4

static iree_status_t add_one_node(qnn_decoded_t *d, qnn_node_decl_t *n,
	void *graph_handle, const QnnInterface_t *iface) {
	Qnn_Param_t params[MAX_PARAMS];
	Qnn_Tensor_t inputs[MAX_INPUTS];
	Qnn_Tensor_t outputs[MAX_OUTPUTS];
	uint32_t num_params = 0;
	const char *type_name = NULL;
	const iree_host_size_t param_tensor_pretail = d->param_tensor_count;

	// Resolve params + typeName per op kind.
	switch (n->op_kind) {
		case kOpConv2d:
			type_name = "Conv2d";
			IREE_RETURN_IF_ERROR(
				build_conv2d_params(d, n, params, &num_params));
			break;
		case kOpDepthwiseConv2d:
			type_name = "DepthwiseConv2d";
			IREE_RETURN_IF_ERROR(
				build_conv2d_params(d, n, params, &num_params));
			break;
		case kOpPoolMax2d:
			type_name = "PoolMax2d";
			IREE_RETURN_IF_ERROR(build_pool_params(d, n, params, &num_params));
			break;
		case kOpPoolAvg2d:
			type_name = "PoolAvg2d";
			IREE_RETURN_IF_ERROR(build_pool_params(d, n, params, &num_params));
			break;
		case kOpFullyConnected:
			type_name = "FullyConnected";
			// QNN's FullyConnected has an optional `keep_dims` scalar bool;
			// emit defaulted false. We didn't serialize it, so skip.
			break;
		case kOpMatMul: {
			type_name = "MatMul";
			// Two scalar-bool params: transpose_in0, transpose_in1.
			for (uint32_t i = 0; i < n->num_params; ++i) {
				if (n->params[i].size < 4)
					continue;
				uint32_t v;
				memcpy(&v, n->params[i].data, 4);
				if (n->params[i].name == 0) {
					params[num_params++] =
						make_scalar_bool_param("transpose_in0", v != 0);
				} else if (n->params[i].name == 1) {
					params[num_params++] =
						make_scalar_bool_param("transpose_in1", v != 0);
				}
			}
			break;
		}
		case kOpElementWiseNeuron: {
			type_name = "ElementWiseNeuron";
			for (uint32_t i = 0; i < n->num_params; ++i) {
				if (n->params[i].name == 0 && n->params[i].size >= 4) {
					uint32_t v;
					memcpy(&v, n->params[i].data, 4);
					params[num_params++] =
						make_scalar_u32_param("operation", neuron_operation(v));
				}
			}
			break;
		}
		case kOpElementWiseBinary: {
			uint32_t binary_kind = 0;
			bool got_kind = false;
			for (uint32_t i = 0; i < n->num_params; ++i) {
				if (n->params[i].name == 0 && n->params[i].size >= 4) {
					memcpy(&binary_kind, n->params[i].data, 4);
					got_kind = true;
				}
			}
			if (!got_kind) {
				return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
					"qnn-graph: ElementWiseBinary node missing op_kind param "
					"(expected param name=0, u32 0..3)");
			}
			type_name = binary_typename(binary_kind);
			if (!type_name) {
				return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
					"qnn-graph: unknown ElementWiseBinary op_kind=%u (expected "
					"0=Add, 1=Subtract, 2=Multiply, 3=Divide)",
					binary_kind);
			}
			break;
		}
		case kOpConcat: {
			type_name = "Concat";
			for (uint32_t i = 0; i < n->num_params; ++i) {
				if (n->params[i].name == 0 && n->params[i].size >= 4) {
					uint32_t axis;
					memcpy(&axis, n->params[i].data, 4);
					params[num_params++] = make_scalar_u32_param("axis", axis);
				}
			}
			break;
		}
		case kOpTranspose: {
			type_name = "Transpose";
			for (uint32_t i = 0; i < n->num_params; ++i) {
				if (n->params[i].name == 0) {
					uint32_t rank = n->params[i].size / 4;
					uint32_t dims[1] = {rank};
					IREE_RETURN_IF_ERROR(make_static_tensor_param(d, "perm",
						QNN_DATATYPE_UINT_32, 1, dims, n->params[i].data,
						n->params[i].size, &params[num_params++]));
				}
			}
			break;
		}
		case kOpReshape:
			type_name = "Reshape";
			break;
		case kOpQuantize:
			type_name = "Quantize";
			break;
		case kOpDequantize:
			type_name = "Dequantize";
			break;
		case kOpPad: {
			type_name = "Pad";
			// params: pad_amount (i32 array, name=0), scheme (i32 scalar,
			// name=1), pad_constant (f32 scalar, name=2).
			for (uint32_t i = 0; i < n->num_params; ++i) {
				if (n->params[i].name == 0) {
					// pad_amount as a static 2D tensor [rank,2].
					uint32_t total = n->params[i].size / 4;
					uint32_t rank = total / 2;
					uint32_t dims[2] = {rank, 2};
					IREE_RETURN_IF_ERROR(
						make_static_tensor_param(d, "pad_amount",
							QNN_DATATYPE_UINT_32, 2, dims, n->params[i].data,
							n->params[i].size, &params[num_params++]));
				} else if (n->params[i].name == 1 && n->params[i].size >= 4) {
					uint32_t v;
					memcpy(&v, n->params[i].data, 4);
					params[num_params++] = make_scalar_u32_param("scheme", v);
				} else if (n->params[i].name == 2 && n->params[i].size >= 4) {
					float v;
					memcpy(&v, n->params[i].data, 4);
					Qnn_Param_t p = {.paramType = QNN_PARAMTYPE_SCALAR,
						.name = "pad_constant_value"};
					p.scalarParam.dataType = QNN_DATATYPE_FLOAT_32;
					p.scalarParam.floatValue = v;
					params[num_params++] = p;
				}
			}
			break;
		}
		case kOpSoftmax: {
			type_name = "Softmax";
			for (uint32_t i = 0; i < n->num_params; ++i) {
				if (n->params[i].name == 0 && n->params[i].size >= 4) {
					uint32_t v;
					memcpy(&v, n->params[i].data, 4);
					params[num_params++] = make_scalar_u32_param("axis", v);
				} else if (n->params[i].name == 1 && n->params[i].size >= 4) {
					float v;
					memcpy(&v, n->params[i].data, 4);
					Qnn_Param_t p = {
						.paramType = QNN_PARAMTYPE_SCALAR, .name = "beta"};
					p.scalarParam.dataType = QNN_DATATYPE_FLOAT_32;
					p.scalarParam.floatValue = v;
					params[num_params++] = p;
				}
			}
			break;
		}
		case kOpReduce: {
			// Map our reduce kind to QNN's typeName (ReduceSum/Mean/Max).
			uint32_t reduce_kind = 0;
			const uint8_t *axes_data = NULL;
			uint32_t axes_size = 0;
			uint32_t keep_dims = 0;
			for (uint32_t i = 0; i < n->num_params; ++i) {
				if (n->params[i].name == 0) {
					axes_data = n->params[i].data;
					axes_size = n->params[i].size;
				} else if (n->params[i].name == 1 && n->params[i].size >= 4) {
					memcpy(&reduce_kind, n->params[i].data, 4);
				} else if (n->params[i].name == 2 && n->params[i].size >= 4) {
					memcpy(&keep_dims, n->params[i].data, 4);
				}
			}
			switch (reduce_kind) {
				case 0:
					type_name = "ReduceSum";
					break;
				case 1:
					type_name = "ReduceMean";
					break;
				case 2:
					type_name = "ReduceMax";
					break;
				default:
					return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
						"qnn-graph: unknown Reduce op_kind=%u "
						"(0=Sum,1=Mean,2=Max)",
						reduce_kind);
			}
			if (axes_data && axes_size > 0) {
				uint32_t count = axes_size / 4;
				uint32_t dims[1] = {count};
				IREE_RETURN_IF_ERROR(
					make_static_tensor_param(d, "axes", QNN_DATATYPE_UINT_32, 1,
						dims, axes_data, axes_size, &params[num_params++]));
			}
			params[num_params++] =
				make_scalar_bool_param("keep_dims", keep_dims != 0);
			break;
		}
		default:
			return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
				"qnn-graph builder: unknown op_kind %u", n->op_kind);
	}

	// Resolve input/output tensors by id.
	if (n->num_inputs > MAX_INPUTS || n->num_outputs > MAX_OUTPUTS) {
		return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
			"qnn-graph node has too many operands "
			"(%u inputs, %u outputs)",
			n->num_inputs, n->num_outputs);
	}
	for (uint32_t k = 0; k < n->num_inputs; ++k) {
		bool found = false;
		for (uint32_t t = 0; t < d->num_tensors; ++t) {
			if (d->tensors[t].id == n->input_ids[k]) {
				inputs[k] = d->tensors[t].tensor;
				found = true;
				break;
			}
		}
		if (!found) {
			return iree_make_status(IREE_STATUS_NOT_FOUND,
				"qnn-graph node references unknown tensor "
				"id %u (input #%u of op %s)",
				n->input_ids[k], k, type_name);
		}
	}
	for (uint32_t k = 0; k < n->num_outputs; ++k) {
		bool found = false;
		for (uint32_t t = 0; t < d->num_tensors; ++t) {
			if (d->tensors[t].id == n->output_ids[k]) {
				outputs[k] = d->tensors[t].tensor;
				found = true;
				break;
			}
		}
		if (!found) {
			return iree_make_status(IREE_STATUS_NOT_FOUND,
				"qnn-graph node references unknown tensor "
				"id %u (output #%u of op %s)",
				n->output_ids[k], k, type_name);
		}
	}

	// Register any newly-allocated static param tensors with the graph
	// BEFORE referencing them in graphAddNode. HTA validation rejects
	// tensorParam refs whose underlying tensor wasn't first announced.
	for (iree_host_size_t i = param_tensor_pretail; i < d->param_tensor_count;
		 ++i) {
		Qnn_ErrorHandle_t prc =
			iface->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(
				(Qnn_GraphHandle_t)graph_handle, &d->param_tensors[i]);
		if (prc != QNN_SUCCESS) {
			return iree_make_status(IREE_STATUS_INTERNAL,
				"tensorCreateGraphTensor (param) rc=%lld for "
				"param tensor #%zu of op %s",
				(long long)prc, (size_t)i, type_name);
		}
		// Refresh the params[] slot's tensorParam to pick up the newly
		// assigned id from QNN — graphAddNode references tensors by id
		// internally.
		for (uint32_t pi = 0; pi < num_params; ++pi) {
			if (params[pi].paramType == QNN_PARAMTYPE_TENSOR &&
				params[pi].tensorParam.v1.name == d->param_tensors[i].v1.name) {
				params[pi].tensorParam = d->param_tensors[i];
			}
		}
	}

	Qnn_OpConfig_t cfg = QNN_OPCONFIG_INIT;
	cfg.version = QNN_OPCONFIG_VERSION_1;
	cfg.v1.name = n->op_name;
	cfg.v1.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
	cfg.v1.typeName = type_name;
	cfg.v1.numOfParams = num_params;
	cfg.v1.params = num_params > 0 ? params : NULL;
	cfg.v1.numOfInputs = n->num_inputs;
	cfg.v1.inputTensors = n->num_inputs > 0 ? inputs : NULL;
	cfg.v1.numOfOutputs = n->num_outputs;
	cfg.v1.outputTensors = n->num_outputs > 0 ? outputs : NULL;

	Qnn_ErrorHandle_t rc = iface->QNN_INTERFACE_VER_NAME.graphAddNode(
		(Qnn_GraphHandle_t)graph_handle, cfg);
	if (rc != QNN_SUCCESS) {
		return iree_make_status(IREE_STATUS_INTERNAL,
			"graphAddNode rc=%lld for op %s (kind=%u)", (long long)rc,
			type_name, n->op_kind);
	}
	return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Public entry point.
//===----------------------------------------------------------------------===//

// Build a heap-allocated array of `Qnn_Tensor_t` for graph IO. Each
// tensor has its own heap-allocated `name` (strdup) and `dimensions`
// (cloned uint32_t buffer). Caller frees via the matching free helper.
static iree_status_t clone_io_protos(qnn_decoded_t *d, bool want_inputs,
	iree_host_size_t *out_count, Qnn_Tensor_t **out_arr) {
	iree_host_size_t cnt = 0;
	for (uint32_t i = 0; i < d->num_tensors; ++i) {
		if ((want_inputs && d->tensors[i].is_input) ||
			(!want_inputs && d->tensors[i].is_output)) {
			cnt++;
		}
	}
	*out_count = cnt;
	if (cnt == 0) {
		*out_arr = NULL;
		return iree_ok_status();
	}
	Qnn_Tensor_t *arr = NULL;
	IREE_RETURN_IF_ERROR(iree_allocator_malloc(
		iree_allocator_system(), cnt * sizeof(Qnn_Tensor_t), (void **)&arr));
	iree_host_size_t k = 0;
	for (uint32_t i = 0; i < d->num_tensors; ++i) {
		if ((want_inputs && d->tensors[i].is_input) ||
			(!want_inputs && d->tensors[i].is_output)) {
			Qnn_Tensor_t *dst = &arr[k++];
			*dst = d->tensors[i].tensor; // shallow copy
			// Deep-copy the name string.
			size_t name_len = strlen(d->tensors[i].name) + 1;
			char *name_storage = NULL;
			IREE_RETURN_IF_ERROR(iree_allocator_malloc(
				iree_allocator_system(), name_len, (void **)&name_storage));
			memcpy(name_storage, d->tensors[i].name, name_len);
			dst->v1.name = name_storage;
			// Deep-copy the dimensions array.
			if (d->tensors[i].rank > 0) {
				uint32_t *dims = NULL;
				IREE_RETURN_IF_ERROR(
					iree_allocator_malloc(iree_allocator_system(),
						d->tensors[i].rank * sizeof(uint32_t), (void **)&dims));
				memcpy(dims, d->tensors[i].dims,
					d->tensors[i].rank * sizeof(uint32_t));
				dst->v1.dimensions = dims;
			}
		}
	}
	*out_arr = arr;
	return iree_ok_status();
}

void iree_hal_qnn_graph_builder_free_io(void *inputs,
	iree_host_size_t input_count, void *outputs,
	iree_host_size_t output_count) {
	Qnn_Tensor_t *in = (Qnn_Tensor_t *)inputs;
	Qnn_Tensor_t *out = (Qnn_Tensor_t *)outputs;
	iree_allocator_t alloc = iree_allocator_system();
	for (iree_host_size_t i = 0; i < input_count; ++i) {
		iree_allocator_free(alloc, (void *)in[i].v1.name);
		iree_allocator_free(alloc, (void *)in[i].v1.dimensions);
	}
	iree_allocator_free(alloc, in);
	for (iree_host_size_t i = 0; i < output_count; ++i) {
		iree_allocator_free(alloc, (void *)out[i].v1.name);
		iree_allocator_free(alloc, (void *)out[i].v1.dimensions);
	}
	iree_allocator_free(alloc, out);
}

iree_status_t iree_hal_qnn_graph_builder_create(
	iree_hal_qnn_interface_t qnn_interface, void *backend_handle,
	void *device_handle, void *context_handle, const uint8_t *data,
	size_t data_size, void **out_graph_handle,
	iree_host_size_t *out_input_count, iree_host_size_t *out_output_count,
	void **out_inputs, void **out_outputs) {
	if (data_size < 24) {
		return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
			"qnn-graph blob too small (%zu bytes, header is 24)", data_size);
	}
	const QnnInterface_t *iface = (const QnnInterface_t *)qnn_interface;
	*out_graph_handle = NULL;
	*out_input_count = 0;
	*out_output_count = 0;
	if (out_inputs)
		*out_inputs = NULL;
	if (out_outputs)
		*out_outputs = NULL;

	// 1. Decode the blob into in-memory tables (no QNN calls yet).
	qnn_decoded_t decoded;
	iree_status_t status =
		decode_blob(data, data_size, iree_allocator_system(), &decoded);
	if (!iree_status_is_ok(status)) {
		decoded_destroy(&decoded);
		return status;
	}

	// 2. Create the graph.
	Qnn_GraphHandle_t graph = NULL;
	Qnn_ErrorHandle_t rc = iface->QNN_INTERFACE_VER_NAME.graphCreate(
		(Qnn_ContextHandle_t)context_handle, "merlin_qnn_graph",
		/*configs=*/NULL, &graph);
	if (rc != QNN_SUCCESS || graph == NULL) {
		decoded_destroy(&decoded);
		return iree_make_status(
			IREE_STATUS_INTERNAL, "graphCreate failed rc=%lld", (long long)rc);
	}

	// 3. Materialize all tensors.
	status = build_tensors(&decoded, (void *)graph, iface);
	if (!iree_status_is_ok(status)) {
		decoded_destroy(&decoded);
		return status;
	}

	// 4. Add nodes one by one.
	for (uint32_t i = 0; i < decoded.num_nodes; ++i) {
		status =
			add_one_node(&decoded, &decoded.nodes[i], (void *)graph, iface);
		if (!iree_status_is_ok(status)) {
			decoded_destroy(&decoded);
			return status;
		}
	}

	// 5. Finalize.
	if (qnn_trace_enabled()) {
		fprintf(stderr,
			"[qnn-graph] before graphFinalize tensors=%u nodes=%u\n",
			decoded.num_tensors, decoded.num_nodes);
	}
	rc = iface->QNN_INTERFACE_VER_NAME.graphFinalize(graph, /*profile=*/NULL,
		/*signal=*/NULL);
	if (qnn_trace_enabled()) {
		fprintf(
			stderr, "[qnn-graph] after graphFinalize rc=%lld\n", (long long)rc);
	}
	if (rc != QNN_SUCCESS) {
		decoded_destroy(&decoded);
		return iree_make_status(IREE_STATUS_INTERNAL,
			"graphFinalize failed rc=%lld", (long long)rc);
	}

	// 6. Clone the IO tensor protos so the executable can wire bindings.
	iree_host_size_t input_count = 0, output_count = 0;
	Qnn_Tensor_t *inputs_arr = NULL;
	Qnn_Tensor_t *outputs_arr = NULL;
	if (out_inputs) {
		if (qnn_trace_enabled()) {
			fprintf(stderr, "[qnn-graph] clone input protos\n");
		}
		status = clone_io_protos(
			&decoded, /*want_inputs=*/true, &input_count, &inputs_arr);
		if (!iree_status_is_ok(status)) {
			decoded_destroy(&decoded);
			return status;
		}
	}
	if (out_outputs) {
		if (qnn_trace_enabled()) {
			fprintf(stderr, "[qnn-graph] clone output protos\n");
		}
		status = clone_io_protos(
			&decoded, /*want_inputs=*/false, &output_count, &outputs_arr);
		if (!iree_status_is_ok(status)) {
			iree_hal_qnn_graph_builder_free_io(
				inputs_arr, input_count, NULL, 0);
			decoded_destroy(&decoded);
			return status;
		}
	}

	*out_graph_handle = (void *)graph;
	*out_input_count =
		input_count > 0 ? input_count : (iree_host_size_t)decoded.input_count;
	*out_output_count = output_count > 0
		? output_count
		: (iree_host_size_t)decoded.output_count;
	if (qnn_trace_enabled()) {
		fprintf(stderr, "[qnn-graph] graph ready inputs=%zu outputs=%zu\n",
			(size_t)*out_input_count, (size_t)*out_output_count);
	}
	if (out_inputs)
		*out_inputs = inputs_arr;
	if (out_outputs)
		*out_outputs = outputs_arr;
	decoded_destroy(&decoded);
	return iree_ok_status();
}
