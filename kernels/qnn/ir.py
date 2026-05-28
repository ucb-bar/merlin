"""Tiny intermediate representation for QNN graphs, used by the MLIR→QNN
emitter (`qnn_emit.py`). Sits between the MLIR parser and the C++ emitter
so adding new ops (depthwise conv, pooling, concat, reshape, fused
activations, …) is a matter of adding NodeDesc kinds, not extending an
ad-hoc string-splicer.

Design point: the emitter target is the same `.qnn.cpp` source format
hand-authored kernels use (see `benchmarks/QRB5165/kernels/abi/*.qnn.cpp`).
That keeps the boundary between hand-authored and emitter-produced kernels
at the source-file layer — both flow through `kernels/qnn/build.py`'s
board-native build path identically.

Type system:

  TensorDesc — one tensor in the graph. `role` distinguishes runtime inputs
              (APP_WRITE) from static weights/biases (STATIC, with bytes
              baked in via clientBuf), intermediate tensors (NATIVE), and
              graph outputs (APP_READ).

  NodeDesc — one QNN op invocation. `params` carries QNN_PARAMTYPE_TENSOR
            and QNN_PARAMTYPE_SCALAR attributes (e.g., conv2d's stride /
            pad_amount / dilation / group).

  QnnGraphDesc — a complete graph. `emit_qnn_cpp` produces the .qnn.cpp
                source string that defines `QnnModel_composeGraphs`.

For now the emitter targets fp32 only and supports Conv2d + Relu +
ElementWiseAdd (bias add). Each new op extends `NodeDesc`'s emit logic.
"""

from __future__ import annotations

import dataclasses
import struct

_DTYPE_TO_QNN = {
    "float32": "QNN_DATATYPE_FLOAT_32",
    "float16": "QNN_DATATYPE_FLOAT_16",
    "uint32": "QNN_DATATYPE_UINT_32",
    "int32": "QNN_DATATYPE_INT_32",
    "int8": "QNN_DATATYPE_SFIXED_POINT_8",
    "uint8": "QNN_DATATYPE_UFIXED_POINT_8",
    # Quantized 32-bit integer used for biases on HTA / DSP. Same memory
    # layout as int32 but carries a per-tensor scale (typically
    # input_scale * weight_scale) so the backend knows how to dequantize.
    "sfixed_point_32": "QNN_DATATYPE_SFIXED_POINT_32",
}

_ROLE_TO_QNN = {
    "input": "QNN_TENSOR_TYPE_APP_WRITE",
    "output": "QNN_TENSOR_TYPE_APP_READ",
    "static": "QNN_TENSOR_TYPE_STATIC",
    "native": "QNN_TENSOR_TYPE_NATIVE",
}


@dataclasses.dataclass(frozen=True)
class QuantParams:
    """Per-tensor scale-offset quantization params, matching QNN's
    `Qnn_QuantizeParams_t.scaleOffsetEncoding`. The dequantization formula
    is `real = scale * (q - offset)`.

    For unsigned int8 tensors, offset=128 maps the unsigned range [0, 255]
    to a signed-equivalent representation. For signed int8, offset is
    typically 0.
    """

    scale: float
    offset: int = 0

    def is_undefined(self) -> bool:
        return self.scale == 0.0 and self.offset == 0


@dataclasses.dataclass(frozen=True)
class TensorDesc:
    name: str
    shape: tuple[int, ...]
    dtype: str
    role: str
    # For static tensors only: raw bytes of the constant payload, in
    # row-major / C order matching `shape`.
    static_data: bytes | None = None
    # When set, declares per-tensor scale/offset quantization on the QNN
    # tensor descriptor. Required for any QNN_DATATYPE_*FIXED_POINT_* dtype;
    # ignored for fp32/fp16/int32 tensors (use None for those).
    quant: QuantParams | None = None

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPE_TO_QNN:
            raise ValueError(f"unknown dtype '{self.dtype}'")
        if self.role not in _ROLE_TO_QNN:
            raise ValueError(f"unknown role '{self.role}'")
        if self.role == "static" and self.static_data is None:
            raise ValueError(f"static tensor '{self.name}' missing bytes")
        if self.role != "static" and self.static_data is not None:
            raise ValueError(f"non-static tensor '{self.name}' has bytes attached")
        # Quant params are required for fixed-point dtypes per QNN's API.
        is_fixed_point = "fixed_point" in self.dtype.lower() or self.dtype in (
            "uint8",
            "int8",
            "uint16",
            "int16",
            "sfixed_point_32",
        )
        # For sfixed_point_32 bias the per-tensor scale is implied (input *
        # weight) so quant is sometimes None at the source level — leave
        # validation lenient.
        if not is_fixed_point and self.quant is not None:
            raise ValueError(
                f"tensor '{self.name}' has quant params but dtype '{self.dtype}'" " is not a fixed-point type"
            )


@dataclasses.dataclass(frozen=True)
class TensorParam:
    """A QNN_PARAMTYPE_TENSOR-style parameter (a small static tensor used as
    op metadata, e.g. conv stride / pad_amount / dilation)."""

    name: str
    shape: tuple[int, ...]
    dtype: str  # typically "uint32"
    values: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class ScalarParam:
    """A QNN_PARAMTYPE_SCALAR parameter (e.g. conv group=1).

    `value` may be an integer / float (numeric literal in the emitted
    C++) or a string (emitted verbatim — used for QNN op-defs whose
    canonical form is a macro symbol like
    `QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU`, defined in `QnnOpDef.h`).
    """

    name: str
    dtype: str
    value: int | float | str


@dataclasses.dataclass(frozen=True)
class NodeDesc:
    name: str
    op_package: str
    op_type: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    tensor_params: tuple[TensorParam, ...] = ()
    scalar_params: tuple[ScalarParam, ...] = ()


@dataclasses.dataclass(frozen=True)
class QnnGraphDesc:
    name: str
    tensors: tuple[TensorDesc, ...]
    nodes: tuple[NodeDesc, ...]


# ---------------------------------------------------------------------------
# C++ emitter
# ---------------------------------------------------------------------------


_HEADER = """\
// Auto-generated QNN kernel for graph "{graph_name}".
// Emitted by kernels/qnn/emit.py from MLIR source. Do not edit by hand.
//
// This file follows the same structure hand-authored kernels under
// `benchmarks/QRB5165/kernels/abi/*.qnn.cpp` use, so both flow through
// `kernels/qnn/build.py`'s board-native build path identically.

#include "QnnKernelHelpers.hpp"
#include "QnnModel.hpp"
#include "QnnOpDef.h"

#include <cstdint>

#define DO_GRAPH_NODE_VALIDATIONS 1

using namespace qnn_wrapper_api;
using merlin_qnn::TensorSpec;
using merlin_qnn::makeTensor;
using merlin_qnn::addOp;
using merlin_qnn::fp32QuantizeParams;

namespace {{
"""


def _bytes_to_initializer(b: bytes) -> str:
    """Render `bytes` as a comma-separated `uint8_t[]` initializer split
    across lines so the emitted C++ stays under typical column limits."""
    parts = []
    line: list[str] = []
    for byte in b:
        line.append(f"0x{byte:02x}")
        if len(line) >= 12:
            parts.append("    " + ", ".join(line) + ",")
            line = []
    if line:
        parts.append("    " + ", ".join(line))
    return "\n".join(parts)


def _emit_static_tensor_storage(t: TensorDesc) -> str:
    """Emit a `uint8_t` byte array + a `uint32_t[]` dimensions array for a
    STATIC tensor. We store the payload as raw bytes regardless of dtype to
    keep the emitter dtype-agnostic; the QNN tensor's `dataType` field
    selects how the runtime interprets it."""
    assert t.static_data is not None
    dims = ", ".join(str(d) for d in t.shape)
    storage = _bytes_to_initializer(t.static_data)
    return (
        f"uint32_t g_{t.name}_dims[{len(t.shape)}] = {{ {dims} }};\n"
        f"uint8_t g_{t.name}_data[{len(t.static_data)}] = {{\n{storage}\n}};\n"
    )


def _emit_runtime_tensor_dims(t: TensorDesc) -> str:
    dims = ", ".join(str(d) for d in t.shape)
    return f"uint32_t g_{t.name}_dims[{len(t.shape)}] = {{ {dims} }};\n"


def _emit_tensor_param_storage(node: NodeDesc, p: TensorParam) -> str:
    """Static byte storage for a QNN_PARAMTYPE_TENSOR attribute (uint32)."""
    if p.dtype != "uint32":
        raise NotImplementedError(f"tensor param '{p.name}' dtype '{p.dtype}' not yet supported")
    qnn_param_name = f"g_{node.name}_{p.name}"
    dims = ", ".join(str(d) for d in p.shape)
    values = ", ".join(str(v) for v in p.values)
    return (
        f"uint32_t {qnn_param_name}_dims[{len(p.shape)}] = {{ {dims} }};\n"
        f"uint32_t {qnn_param_name}_values[] = {{ {values} }};\n"
    )


def _emit_qparams_expr(t: TensorDesc) -> str:
    """Render the C++ initializer for `Qnn_Tensor_t.v1.quantizeParams`
    based on the tensor's quant settings."""
    if t.quant is None:
        return "fp32QuantizeParams()"
    # Format the scale at 7 significant digits — the f32 round-trip
    # precision. This collapses f64-vs-f32 representation differences that
    # would otherwise diverge between the regex emitter (text-parsed
    # scale, exact-decimal Python float) and the bindings emitter
    # (f32-stored scale read via FloatAttr.value, which surfaces the
    # actual f32 representation as a float64). Both round to the same
    # f32 at compile time so .qnn-ctx output is identical either way.
    return f"merlin_qnn::int8QuantizeParams(" f"{t.quant.scale:.7g}f, {t.quant.offset})"


def _emit_addtensor_call(t: TensorDesc) -> str:
    """Emit a `model.addTensor` invocation for tensor `t` inside the
    composeGraphs body."""
    qnn_dtype = _DTYPE_TO_QNN[t.dtype]
    qnn_role = _ROLE_TO_QNN[t.role]
    qparams = _emit_qparams_expr(t)
    if t.role == "static":
        client_buf = f", .clientBuf = {{ g_{t.name}_data, " f"sizeof(g_{t.name}_data) }}"
    else:
        client_buf = ", .clientBuf = {nullptr, 0}"

    return (
        f"    {{\n"
        f"        Qnn_Tensor_t t{{}};\n"
        f"        t.version = QNN_TENSOR_VERSION_1;\n"
        f'        t.v1 = {{ .id = 0, .name = "{t.name}",'
        f" .type = {qnn_role},\n"
        f"                  .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,\n"
        f"                  .dataType = {qnn_dtype},\n"
        f"                  .quantizeParams = {qparams},\n"
        f"                  .rank = {len(t.shape)},\n"
        f"                  .dimensions = g_{t.name}_dims,\n"
        f"                  .memType = QNN_TENSORMEMTYPE_RAW{client_buf} }};\n"
        f'        VALIDATE(model.addTensor("{t.name}", &t), err);\n'
        f"    }}\n"
    )


def _emit_addnode_call(node: NodeDesc, output_tensor: TensorDesc) -> str:
    """Emit a single `model.addNode` invocation for a node with one output
    tensor. The output tensor is constructed inline (QNN's `addNode` creates
    output tensors as a side effect)."""
    inputs_lit = ", ".join(f'"{n}"' for n in node.inputs)
    qnn_role = _ROLE_TO_QNN[output_tensor.role]
    qnn_dtype = _DTYPE_TO_QNN[output_tensor.dtype]
    out_var = f"out_{node.name}"

    # Build params array (tensor params + scalar params) in QNN_OPCONFIG order.
    param_inits = []
    for tp in node.tensor_params:
        qnn_dtype_p = _DTYPE_TO_QNN[tp.dtype]
        nbytes = len(tp.values) * 4  # uint32 = 4 bytes
        param_inits.append(
            f"        {{\n"
            f"            .paramType = QNN_PARAMTYPE_TENSOR,\n"
            f'            .name = "{tp.name}",\n'
            f"            .tensorParam = (Qnn_Tensor_t){{\n"
            f"                .version = QNN_TENSOR_VERSION_1,\n"
            f'                .v1 = {{ .id = 0, .name = "{node.name}_{tp.name}",'
            f" .type = QNN_TENSOR_TYPE_STATIC,\n"
            f"                          .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,\n"
            f"                          .dataType = {qnn_dtype_p},\n"
            f"                          .quantizeParams = fp32QuantizeParams(),\n"
            f"                          .rank = {len(tp.shape)},\n"
            f"                          .dimensions = g_{node.name}_{tp.name}_dims,\n"
            f"                          .memType = QNN_TENSORMEMTYPE_RAW,\n"
            f"                          .clientBuf = {{ g_{node.name}_{tp.name}_values, {nbytes} }} }} }} \n"
            f"        }},\n"
        )
    for sp in node.scalar_params:
        qnn_dtype_s = _DTYPE_TO_QNN[sp.dtype]
        # For uint32 scalar use uint32Value; emit a small switch.
        if sp.dtype == "uint32":
            field = f".uint32Value = {sp.value}"
        elif sp.dtype == "int32":
            field = f".int32Value = {sp.value}"
        elif sp.dtype == "float32":
            field = f".floatValue = {sp.value}f"
        else:
            raise NotImplementedError(f"scalar param dtype '{sp.dtype}' not yet supported")
        param_inits.append(
            f"        {{ .paramType = QNN_PARAMTYPE_SCALAR,\n"
            f'          .name = "{sp.name}",\n'
            f"          .scalarParam = {{ .dataType = {qnn_dtype_s}, {field} }} }},\n"
        )
    n_params = len(node.tensor_params) + len(node.scalar_params)
    if param_inits:
        params_array = f"    Qnn_Param_t {node.name}_params[] = {{\n" + "".join(param_inits) + "    };\n"
        params_arg = f"{node.name}_params"
    else:
        params_array = ""
        params_arg = "nullptr"

    inputs_array = f"    const char* {node.name}_inputs[] = {{ {inputs_lit} }};\n"
    out_qparams = _emit_qparams_expr(output_tensor)
    output_construction = (
        f"    Qnn_Tensor_t {out_var}{{}};\n"
        f"    {out_var}.version = QNN_TENSOR_VERSION_1;\n"
        f'    {out_var}.v1 = {{ .id = 0, .name = "{output_tensor.name}",'
        f" .type = {qnn_role},\n"
        f"                       .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,\n"
        f"                       .dataType = {qnn_dtype},\n"
        f"                       .quantizeParams = {out_qparams},\n"
        f"                       .rank = {len(output_tensor.shape)},\n"
        f"                       .dimensions = g_{output_tensor.name}_dims,\n"
        f"                       .memType = QNN_TENSORMEMTYPE_RAW,\n"
        f"                       .clientBuf = {{nullptr, 0}} }};\n"
    )
    addnode_call = (
        f"    VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1,\n"
        f'        "{node.name}", "{node.op_package}", "{node.op_type}",\n'
        f"        {params_arg}, {n_params},\n"
        f"        {node.name}_inputs, {len(node.inputs)},\n"
        f"        &{out_var}, 1), err);\n"
    )
    return (
        f"    // ---- Node: {node.name} ({node.op_type}) ----\n"
        + params_array
        + inputs_array
        + output_construction
        + addnode_call
    )


def emit_qnn_cpp(graph: QnnGraphDesc) -> str:
    """Render a `.qnn.cpp` source file that defines `QnnModel_composeGraphs`
    for `graph`. Output is a complete translation unit that compiles via
    g++ together with Qualcomm's QnnModel.cpp + QnnWrapperUtils.cpp +
    QnnModelPal.cpp + our QnnKernelHelpers.hpp.
    """
    parts: list[str] = [_HEADER.format(graph_name=graph.name)]

    # Static-tensor storage at namespace scope (lives for the program's
    # lifetime; QNN reads `clientBuf` lazily when finalizing the graph).
    for t in graph.tensors:
        if t.role == "static":
            parts.append(_emit_static_tensor_storage(t))
        else:
            parts.append(_emit_runtime_tensor_dims(t))
    for n in graph.nodes:
        for tp in n.tensor_params:
            parts.append(_emit_tensor_param_storage(n, tp))

    parts.append("}  // namespace\n\n")

    parts.append(
        f'extern "C" {{\n\n'
        f"QNN_API\n"
        f"ModelError_t QnnModel_composeGraphs("
        f"Qnn_BackendHandle_t backendHandle,\n"
        f"    QNN_INTERFACE_VER_TYPE interface,\n"
        f"    Qnn_ContextHandle_t contextHandle,\n"
        f"    const GraphConfigInfo_t** graphsConfigInfo,\n"
        f"    const uint32_t numGraphsConfigInfo,\n"
        f"    GraphInfoPtr_t** graphsInfo, uint32_t* numGraphsInfo,\n"
        f"    bool /*debug*/, QnnLog_Callback_t /*lc*/,\n"
        f"    QnnLog_Level_t /*ll*/) {{\n"
        f"    ModelError_t err = MODEL_NO_ERROR;\n"
        f"    QnnModel model;\n"
        f"    const QnnGraph_Config_t** gc = nullptr;\n"
        f'    VALIDATE(getQnnGraphConfigFromInfo("{graph.name}",\n'
        f"        graphsConfigInfo, numGraphsConfigInfo, gc), err);\n"
        f"    VALIDATE(model.initialize(backendHandle, interface,\n"
        f'        contextHandle, "{graph.name}", false,\n'
        f"        DO_GRAPH_NODE_VALIDATIONS, gc), err);\n\n"
    )

    # Add tensors first (inputs + statics; intermediates and outputs are
    # created as a side effect of addNode).
    for t in graph.tensors:
        if t.role in ("input", "static"):
            parts.append(_emit_addtensor_call(t))

    # Map name -> TensorDesc for output lookups.
    by_name = {t.name: t for t in graph.tensors}

    # Add nodes; each node's output tensor is constructed inline in the
    # call. We require each NodeDesc to have exactly one output for now.
    for n in graph.nodes:
        if len(n.outputs) != 1:
            raise NotImplementedError(
                f"node '{n.name}' has {len(n.outputs)} outputs; emitter "
                f"currently supports exactly one output per node"
            )
        out_t = by_name.get(n.outputs[0])
        if out_t is None:
            raise ValueError(f"node '{n.name}' output '{n.outputs[0]}' not in tensor list")
        parts.append(_emit_addnode_call(n, out_t))

    parts.append(
        "    QnnModel* m[] = { &model };\n"
        "    VALIDATE(getGraphInfoFromModels(*m, 1, graphsInfo), err);\n"
        "    *numGraphsInfo = 1;\n"
        "    return err;\n"
        "}\n\n"
        "QNN_API\n"
        "ModelError_t QnnModel_freeGraphsInfo(GraphInfoPtr_t** graphsInfo,\n"
        "    uint32_t numGraphsInfo) {\n"
        "    return freeGraphsInfo(graphsInfo, numGraphsInfo);\n"
        "}\n\n"
        '}  // extern "C"\n'
    )

    return "".join(parts)


# ---------------------------------------------------------------------------
# Helpers for building common ops as NodeDesc
# ---------------------------------------------------------------------------


def conv2d_node(
    name: str,
    input_tensor: str,
    weight_tensor: str,
    bias_tensor: str | None,
    output_tensor: str,
    *,
    strides: tuple[int, int] = (1, 1),
    pad_before_after_hw: tuple[tuple[int, int], tuple[int, int]] = (
        (0, 0),
        (0, 0),
    ),
    dilation: tuple[int, int] = (1, 1),
    group: int = 1,
) -> NodeDesc:
    inputs = (input_tensor, weight_tensor)
    if bias_tensor is not None:
        inputs = inputs + (bias_tensor,)
    return NodeDesc(
        name=name,
        op_package="qti.aisw",
        op_type="Conv2d",
        inputs=inputs,
        outputs=(output_tensor,),
        tensor_params=(
            TensorParam(name="dilation", shape=(2,), dtype="uint32", values=dilation),
            TensorParam(
                name="pad_amount",
                shape=(2, 2),
                dtype="uint32",
                values=tuple(v for hw in pad_before_after_hw for v in hw),
            ),
            TensorParam(name="stride", shape=(2,), dtype="uint32", values=strides),
        ),
        scalar_params=(ScalarParam(name="group", dtype="uint32", value=group),),
    )


def relu_node(name: str, input_tensor: str, output_tensor: str) -> NodeDesc:
    return NodeDesc(
        name=name,
        op_package="qti.aisw",
        op_type="Relu",
        inputs=(input_tensor,),
        outputs=(output_tensor,),
    )


def unary_op_node(
    name: str,
    op_type: str,
    input_tensor: str,
    output_tensor: str,
) -> NodeDesc:
    """Generic single-input single-output op (Sigmoid, Relu, Tanh,
    HardSwish, Relu6, etc.). The QNN op_type string identifies which one;
    these all share the same shape (1 input, 1 output, no params)."""
    return NodeDesc(
        name=name,
        op_package="qti.aisw",
        op_type=op_type,
        inputs=(input_tensor,),
        outputs=(output_tensor,),
    )


def binary_op_node(
    name: str,
    op_type: str,
    lhs: str,
    rhs: str,
    output_tensor: str,
) -> NodeDesc:
    """Generic two-input single-output elementwise op (ElementWiseAdd,
    ElementWiseSubtract, ElementWiseMultiply, ElementWiseDivide). The QNN
    op_type string identifies which one; shape and param surface are
    identical across the family."""
    return NodeDesc(
        name=name,
        op_package="qti.aisw",
        op_type=op_type,
        inputs=(lhs, rhs),
        outputs=(output_tensor,),
    )


def depthwise_conv2d_node(
    name: str,
    input_tensor: str,
    weight_tensor: str,
    bias_tensor: str | None,
    output_tensor: str,
    *,
    strides: tuple[int, int] = (1, 1),
    pad_before_after_hw: tuple[tuple[int, int], tuple[int, int]] = (
        (0, 0),
        (0, 0),
    ),
    dilation: tuple[int, int] = (1, 1),
) -> NodeDesc:
    """QNN DepthWiseConv2d. Same param surface as Conv2d but the weight
    shape is HWCM (kh, kw, channels, channel_multiplier=1). The linalg
    form `linalg.depthwise_conv_2d_nhwc_hwc` uses HWC; lowering must
    reshape weights to HWCM with M=1.
    """
    inputs = (input_tensor, weight_tensor)
    if bias_tensor is not None:
        inputs = inputs + (bias_tensor,)
    return NodeDesc(
        name=name,
        op_package="qti.aisw",
        op_type="DepthWiseConv2d",
        inputs=inputs,
        outputs=(output_tensor,),
        tensor_params=(
            TensorParam(name="dilation", shape=(2,), dtype="uint32", values=dilation),
            TensorParam(
                name="pad_amount",
                shape=(2, 2),
                dtype="uint32",
                values=tuple(v for hw in pad_before_after_hw for v in hw),
            ),
            TensorParam(name="stride", shape=(2,), dtype="uint32", values=strides),
        ),
        scalar_params=(),
    )


def reshape_node(
    name: str,
    input_tensor: str,
    output_tensor: str,
) -> NodeDesc:
    """QNN Reshape — single input, single output. The output tensor's shape
    declares the target shape; QNN derives the reshape from the rank/dim
    difference. No params (the target shape is implicit in the output
    tensor's `dimensions` field)."""
    return NodeDesc(
        name=name,
        op_package="qti.aisw",
        op_type="Reshape",
        inputs=(input_tensor,),
        outputs=(output_tensor,),
    )


def concat_node(
    name: str,
    input_tensors: tuple[str, ...],
    output_tensor: str,
    *,
    axis: int,
) -> NodeDesc:
    """QNN Concat — variable-input single-output. Concatenates `input_tensors`
    along `axis`. All input tensors must have the same shape except along
    that axis, where their sizes sum to the output's axis size."""
    if len(input_tensors) < 2:
        raise ValueError("Concat requires at least 2 input tensors")
    return NodeDesc(
        name=name,
        op_package="qti.aisw",
        op_type="Concat",
        inputs=input_tensors,
        outputs=(output_tensor,),
        tensor_params=(),
        scalar_params=(ScalarParam(name="axis", dtype="uint32", value=axis),),
    )


def pool_max_2d_node(
    name: str,
    input_tensor: str,
    output_tensor: str,
    *,
    filter_size: tuple[int, int],
    strides: tuple[int, int] = (1, 1),
    pad_before_after_hw: tuple[tuple[int, int], tuple[int, int]] = (
        (0, 0),
        (0, 0),
    ),
    rounding_mode: int = 0,  # 0 = floor, 1 = ceil
) -> NodeDesc:
    """QNN PoolMax2d. Single input, single output. `filter_size` is the
    spatial pooling window (h, w). `strides` and `pad_amount` follow the
    same shape as Conv2d. `rounding_mode` selects floor (0) or ceil (1)
    for output spatial-dim computation."""
    return NodeDesc(
        name=name,
        op_package="qti.aisw",
        op_type="PoolMax2d",
        inputs=(input_tensor,),
        outputs=(output_tensor,),
        tensor_params=(
            TensorParam(name="filter_size", shape=(2,), dtype="uint32", values=filter_size),
            TensorParam(
                name="pad_amount",
                shape=(2, 2),
                dtype="uint32",
                values=tuple(v for hw in pad_before_after_hw for v in hw),
            ),
            TensorParam(name="stride", shape=(2,), dtype="uint32", values=strides),
        ),
        scalar_params=(ScalarParam(name="rounding_mode", dtype="uint32", value=rounding_mode),),
    )


def transpose_node(
    name: str,
    input_tensor: str,
    output_tensor: str,
    *,
    perm: tuple[int, ...],
) -> NodeDesc:
    """QNN Transpose — permutes tensor dimensions. `perm` is the dim
    permutation (e.g. `(0, 2, 3, 1)` for NCHW→NHWC).

    Used by recognizers that bridge IREE's NCHW layout to QNN's NHWC
    convention for Conv2d / pool / etc. The output tensor's declared
    `dimensions` must match the post-permutation shape.
    """
    return NodeDesc(
        name=name,
        op_package="qti.aisw",
        op_type="Transpose",
        inputs=(input_tensor,),
        outputs=(output_tensor,),
        tensor_params=(
            TensorParam(
                name="perm",
                shape=(len(perm),),
                dtype="uint32",
                values=tuple(perm),
            ),
        ),
    )


def dequantize_node(
    name: str,
    input_tensor: str,
    output_tensor: str,
) -> NodeDesc:
    """QNN Dequantize — converts a fixed-point input tensor to a float
    output using the input's per-tensor q-params (`real = (q - offset) *
    scale`). Single input, single output, no params; the q-params come
    from the input tensor descriptor."""
    return NodeDesc(
        name=name,
        op_package="qti.aisw",
        op_type="Dequantize",
        inputs=(input_tensor,),
        outputs=(output_tensor,),
    )


# QnnOpDef.h macro names for the ElementWiseNeuron `operation` enum. We
# emit these verbatim into the generated `.qnn.cpp` so the resulting
# source binds against the SDK's stable values rather than the literal
# integers (which match the macros today but are SDK-version-coupled).
ELEMENT_WISE_NEURON_OPERATIONS = {
    "Relu": "QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU",
    "Relu6": "QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU_MIN_MAX",
    "Tanh": "QNN_OP_ELEMENT_WISE_NEURON_OPERATION_TANH",
    "Sigmoid": "QNN_OP_ELEMENT_WISE_NEURON_OPERATION_SIGMOID",
    "Elu": "QNN_OP_ELEMENT_WISE_NEURON_OPERATION_ELU",
    "HardSwish": "QNN_OP_ELEMENT_WISE_NEURON_OPERATION_HARD_SWISH",
    "Gelu": "QNN_OP_ELEMENT_WISE_NEURON_OPERATION_GELU",
}


def element_wise_neuron_node(
    name: str,
    input_tensor: str,
    output_tensor: str,
    *,
    operation: str,
) -> NodeDesc:
    """QNN ElementWiseNeuron — single-input single-output activation
    op selected by the `operation` scalar enum.

    `operation` is one of the keys in `ELEMENT_WISE_NEURON_OPERATIONS`
    (e.g. `"Relu"`); the emitter renders the corresponding QNN macro
    symbol verbatim so the generated `.qnn.cpp` compiles against the
    SDK's stable enum values.

    Pairing this op directly after a `Conv2d` with shared output
    q-params lets HTA's `fold_relu_activation_into_conv` finalize-time
    optimizer collapse the two ops into one HVX kernel — the structure
    matches `benchmarks/QRB5165/kernels/abi/conv2d_relu_int8_fused.qnn.cpp`.
    """
    macro = ELEMENT_WISE_NEURON_OPERATIONS.get(operation)
    if macro is None:
        raise ValueError(
            f"unknown ElementWiseNeuron operation '{operation}'; "
            f"choose from {sorted(ELEMENT_WISE_NEURON_OPERATIONS)}"
        )
    return NodeDesc(
        name=name,
        op_package="qti.aisw",
        op_type="ElementWiseNeuron",
        inputs=(input_tensor,),
        outputs=(output_tensor,),
        scalar_params=(ScalarParam(name="operation", dtype="uint32", value=macro),),
    )


def f32_to_bytes(values: list[float] | tuple[float, ...]) -> bytes:
    """Render a list of floats as little-endian fp32 bytes (matches QNN's
    expected raw layout)."""
    return struct.pack(f"<{len(values)}f", *values)


def f16_to_bytes(values: list[float] | tuple[float, ...]) -> bytes:
    """Render a list of floats as little-endian IEEE-754 binary16 (fp16)
    bytes. struct's 'e' format is half-precision (2 bytes per value)."""
    return struct.pack(f"<{len(values)}e", *values)
