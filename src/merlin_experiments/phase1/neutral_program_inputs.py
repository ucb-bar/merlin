"""Host-only mathematical input projection for the raw-RTL EL1 condition.

This is a source representation adapter, not a compiler or target lowering. It
erases residency/materialization commands and publishes a mathematical expression
DAG. The author receives only the resulting data; this implementation stays with
the judge. Existing generic linalg/tensor/arith source programs retain exact bytes.
The same encoder must be used for author views and candidate runtime inputs.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import yaml

SCHEMA = "numerical_program.v1"
FORMAT = "neutral_math_v1"


class UnsupportedProgram(ValueError):
    """The mathematical adapter cannot represent the complete source program."""


@dataclass(frozen=True)
class ProgramInput:
    format: str
    filename: str
    payload: bytes
    source_sha256: str


def _checked_attrs(attrs, allowed, required=()):
    extra, missing = set(attrs) - set(allowed), set(required) - set(attrs)
    if extra or missing:
        raise UnsupportedProgram(f"unsupported or missing mathematical attributes: extra={sorted(extra)}, missing={sorted(missing)}")


def _sequence(value, length, name, *, zero=False):
    if (not isinstance(value, list) or len(value) != length
            or any(type(v) is not int or v < (0 if zero else 1) for v in value)):
        raise UnsupportedProgram("invalid mathematical extent: " + name)
    return value


def _factor(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UnsupportedProgram("scale must be numeric")
    try:
        number = struct.unpack("<f", struct.pack("<f", float(value)))[0]
    except (OverflowError, struct.error) as error:
        raise UnsupportedProgram("scale is not representable as finite float32") from error
    if not math.isfinite(number):
        raise UnsupportedProgram("scale is not representable as finite float32")
    return value


class _Graph:
    def __init__(self):
        self.inputs = []
        self.expressions = []
        self.outputs = []
        self.values = {}

    def ref(self, name):
        if name not in self.values:
            raise UnsupportedProgram("unknown mathematical value: " + str(name))
        return self.values[name]

    def bind(self, name, value):
        if not isinstance(name, str) or not name or name in self.values:
            raise UnsupportedProgram("duplicate/invalid source value")
        self.values[name] = value

    def emit(self, operator, args, shape, dtype, parameters=None):
        value = {"id": "v" + str(len(self.expressions)), "operator": operator,
                 "operands": [x["id"] for x in args], "shape": list(shape), "dtype": dtype}
        if parameters:
            value["parameters"] = parameters
        self.expressions.append(value)
        return value

    def publish(self, name, value, shape, dtype):
        if value["shape"] != shape or value["dtype"] != dtype:
            raise UnsupportedProgram("mathematical result differs from declared output type")
        if any(row["name"] == name for row in self.outputs):
            raise UnsupportedProgram("duplicate named output")
        self.outputs.append({"name": name, "value": value["id"], "shape": list(shape), "dtype": dtype})

    def cast(self, value, dtype):
        if dtype not in {"i8", "i32"}:
            raise UnsupportedProgram("unsupported integer result dtype: " + str(dtype))
        if value["dtype"] == dtype:
            return value
        return self.emit("cast", [value], value["shape"], dtype,
                         {"conversion": "saturate" if dtype == "i8" else "sign_extend"})

    def product(self, lhs, rhs):
        a, b = lhs["shape"], rhs["shape"]
        if len(a) < 2 or len(b) < 2 or a[-1] != b[-2]:
            raise UnsupportedProgram("inconsistent matrix product extents")
        if len(b) == 2:
            shape = [*a[:-2], a[-2], b[-1]]
        elif a[:-2] == b[:-2]:
            shape = [*a[:-2], a[-2], b[-1]]
        else:
            raise UnsupportedProgram("unsupported matrix batch broadcasting")
        return self.emit("matmul", [lhs, rhs], shape, "i32")

    def finish(self, value, attrs):
        stages = attrs.get("epilogue", [])
        if not isinstance(stages, list) or any(not isinstance(s, str) for s in stages):
            raise UnsupportedProgram("mathematical post-operations must be an ordered list")
        for stage in stages:
            if stage in {"bias", "bias_add"}:
                if "bias" not in attrs:
                    raise UnsupportedProgram("bias input is undeclared")
                bias = self.ref(attrs["bias"])
                if bias["shape"] != [value["shape"][-1]]:
                    raise UnsupportedProgram("bias is not a vector over the last output dimension")
                value = self.emit("add_bias", [value, bias], value["shape"], value["dtype"])
            elif stage == "acc_scale":
                if "acc_scale_by_column" in attrs:
                    factors = attrs["acc_scale_by_column"]
                    if "acc_scale" in attrs or not isinstance(factors, list) or len(factors) != value["shape"][-1]:
                        raise UnsupportedProgram("column scale factors disagree with output extent")
                    parameters = {"factors": [_factor(x) for x in factors], "axis": -1}
                else:
                    parameters = {"factor": _factor(attrs.get("acc_scale"))}
                value = self.emit("scale_round", [value], value["shape"], value["dtype"],
                                  {**parameters, "arithmetic_dtype": "f32", "rounding": "ties_to_even"})
            elif stage == "requant":
                if type(attrs.get("requant_shift")) is not int:
                    raise UnsupportedProgram("rounding shift is missing")
                value = self.emit("round_shift_right", [value], value["shape"], value["dtype"],
                                  {"shift": attrs["requant_shift"], "rounding": "half_up", "nonpositive_shift": "identity"})
            elif stage == "relu":
                value = self.emit("relu", [value], value["shape"], value["dtype"])
            elif stage == "maxpool":
                h, w = _sequence(attrs.get("pool_in_dims"), 2, "pool_in_dims")
                ph, pw = _sequence(attrs.get("pool_size"), 2, "pool_size")
                sh, sw = _sequence(attrs.get("pool_stride"), 2, "pool_stride")
                pads = _sequence(attrs.get("pool_padding", [0, 0, 0, 0]), 4, "pool_padding", zero=True)
                pad_value = attrs.get("pool_pad_value")
                if (pad_value is not None and type(pad_value) is not int) or (any(pads) and pad_value is None):
                    raise UnsupportedProgram("nonzero pool padding requires an explicit integer value")
                if len(value["shape"]) != 2 or value["shape"][0] % (h * w):
                    raise UnsupportedProgram("pool input has no consistent spatial geometry")
                batch = value["shape"][0] // (h * w)
                oh, ow = (h + pads[0] + pads[2] - ph) // sh + 1, (w + pads[1] + pads[3] - pw) // sw + 1
                if min(oh, ow) < 1:
                    raise UnsupportedProgram("empty pool result")
                value = self.emit("max_pool", [value], [batch * oh * ow, value["shape"][1]], value["dtype"],
                                  {"spatial_shape": [h, w], "window": [ph, pw], "stride": [sh, sw],
                                   "padding": pads, "padding_value": pad_value})
            else:
                raise UnsupportedProgram("unsupported mathematical post-operation: " + stage)
        if "output_dtype" not in attrs:
            raise UnsupportedProgram("result dtype must be explicit")
        return self.cast(value, attrs["output_dtype"])


_POST = {"epilogue", "output_dtype", "bias", "acc_scale", "acc_scale_by_column", "requant_shift",
         "pool_in_dims", "pool_size", "pool_stride", "pool_padding", "pool_pad_value"}


def _source_line_tokens(line):
    from xdsl.utils.lexer import Input
    from xdsl.utils.mlir_lexer import MLIRLexer, MLIRTokenKind

    lexer = MLIRLexer(Input(line, "<interface line>"))
    result = []
    while (token := lexer.lex()).kind != MLIRTokenKind.EOF:
        result.append(token.text)
    return result


def _module_header(tokens):
    if tokens == ["module", "{"]:
        return True
    return (tokens[:3] == ["module", "attributes", "{"] and tokens[-2:] == ["}", "{"]
            and not any(token in ("{", "}") for token in tokens[3:-2]))


def _module_attribute(tokens):
    if tokens[-1:] == [","]:
        tokens = tokens[:-1]
    return (len(tokens) == 3 and tokens[0] in
            ("merlin_iface.version", "merlin_iface.target", "merlin_iface.abi_version")
            and tokens[1] == "=" and tokens[2].startswith('"') and tokens[2].endswith('"'))


def _mathematical_document(text):
    # The canonical host parser already refuses unsupported/missing grammar.
    # Its residency command model is consumed privately and never serialized.
    from merlin.targetgen.contract import interface_emit as I

    def tensor_type(tail):
        declared = I._last_type(tail)
        if "," in declared:
            raise UnsupportedProgram("encoded tensor layouts require an explicit mathematical definition")
        shape, dtype = I._shape_dtype(declared)
        if not shape or any(type(extent) is not int or extent <= 0 for extent in shape):
            raise UnsupportedProgram("mathematical tensor extents must be static and positive")
        return shape, dtype

    try:
        I.parse_interface_mlir(text)
        records = []
        header = False
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("//"):
                continue
            mnemonics = I.op_mnemonics(line)
            if mnemonics:
                if len(mnemonics) != 1:
                    raise UnsupportedProgram("multiple source operations on one line")
                records.append(I._op_line(line))
            elif _module_header(_source_line_tokens(line)) or line == "}":
                continue
            elif line == "module attributes {":
                header = True
            elif header and line == "} {":
                header = False
            elif header and _module_attribute(_source_line_tokens(line)):
                continue
            else:
                raise UnsupportedProgram("unconsumed non-mathematical source statement")
        if any(record is None for record in records):
            raise UnsupportedProgram("source parser could not decompose every operation")
        if len(records) != len(I.op_mnemonics(text)):
            raise UnsupportedProgram("source parser did not consume every operation")
    except ValueError as error:
        raise UnsupportedProgram("unsupported complete source program") from error
    graph = _Graph()
    for record in records:
        op = record["mnemonic"]
        attrs = I._parse_attr_block(record["attrs_body"])
        operands, result = record["operands"], record["result"]
        if op == "tensor":
            _checked_attrs(attrs, {"name", "role"})
            shape, dtype = tensor_type(record["tail"])
            name = attrs.get("name", result)
            if dtype not in {"i8", "i32"} or attrs.get("role", "input") not in {"input", "weight", "bias"}:
                raise UnsupportedProgram("unsupported mathematical input declaration")
            if any(row["name"] == name for row in graph.inputs):
                raise UnsupportedProgram("duplicate mathematical input name")
            value = {"id": "i" + str(len(graph.inputs)), "shape": shape, "dtype": dtype}
            graph.inputs.append({"id": value["id"], "name": name, "shape": shape, "dtype": dtype})
            graph.bind(result, value)
            if name != result:
                graph.bind(name, value)
        elif op == "resident_pack":
            _checked_attrs(attrs, {"layout"})
            if len(operands) != 1:
                raise UnsupportedProgram("invalid source operand count")
            graph.bind(result, graph.ref(operands[0]))  # erase residency entirely
        elif op == "evict":
            if attrs or len(operands) != 1:
                raise UnsupportedProgram("invalid source lifetime operation")
            graph.ref(operands[0])  # validate reference, publish no lifetime operation
        elif op == "matmul":
            _checked_attrs(attrs, set())
            if len(operands) != 2:
                raise UnsupportedProgram("invalid matrix product operand count")
            if "!merlin_iface.acc<i32>" not in record["tail"]:
                raise UnsupportedProgram("unsupported matrix accumulation type")
            lhs, rhs = [graph.ref(x) for x in operands]
            if len(lhs["shape"]) != 2 or len(rhs["shape"]) != 2:
                raise UnsupportedProgram("plain matrix products require two rank-2 operands")
            graph.bind(result, graph.product(lhs, rhs))
        elif op in {"commit", "conv2d", "movement", "residual_add", "matmul_batched", "bias_add"}:
            name = attrs.pop("name", result)
            shape, dtype = tensor_type(record["tail"])
            if op == "commit":
                _checked_attrs(attrs, _POST, {"output_dtype"})
                if len(operands) != 1:
                    raise UnsupportedProgram("invalid mathematical result operand count")
                value = graph.finish(graph.ref(operands[0]), attrs)
            elif op == "conv2d":
                _checked_attrs(attrs, _POST | {"kernel", "stride", "padding", "dilation", "layout"}, {"kernel", "layout", "output_dtype"})
                if len(operands) != 2:
                    raise UnsupportedProgram("invalid convolution operand count")
                activation, weight = [graph.ref(x) for x in operands]
                kh, kw, ci, co = _sequence(attrs["kernel"], 4, "kernel")
                stride = _sequence(attrs.get("stride", [1, 1]), 2, "stride")
                padding = _sequence(attrs.get("padding", [0, 0, 0, 0]), 4, "padding", zero=True)
                dilation = _sequence(attrs.get("dilation", [1, 1]), 2, "dilation")
                if attrs["layout"] != "nhwc" or len(activation["shape"]) != 4 or weight["shape"] != [kh * kw * ci, co]:
                    raise UnsupportedProgram("unsupported mathematical convolution layout")
                n, h, w, channels = activation["shape"]
                if channels != ci:
                    raise UnsupportedProgram("convolution channel mismatch")
                oh = (h + padding[0] + padding[2] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
                ow = (w + padding[1] + padding[3] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
                if min(oh, ow) < 1:
                    raise UnsupportedProgram("empty convolution result")
                value = graph.emit("conv2d", [activation, weight], [n * oh * ow, co], "i32",
                                   {"kernel": [kh, kw], "stride": stride, "padding": padding, "dilation": dilation,
                                    "input_axes": "NHWC", "weight_axes": "(KH*KW*CI,CO)", "output_axes": "(N*OH*OW,CO)"})
                value = graph.finish(value, attrs)
            elif op == "movement":
                _checked_attrs(attrs, {"semantic", "output_dtype"}, {"output_dtype"})
                if len(operands) != 1 or attrs.get("semantic", "mvin_mvout") != "mvin_mvout":
                    raise UnsupportedProgram("unsupported identity/cast program")
                value = graph.cast(graph.ref(operands[0]), attrs["output_dtype"])
            elif op == "matmul_batched":
                _checked_attrs(attrs, _POST | {"batch", "semantic"}, {"output_dtype"})
                if len(operands) != 2:
                    raise UnsupportedProgram("invalid batched matrix product")
                lhs, rhs = [graph.ref(x) for x in operands]
                if len(lhs["shape"]) < 3 or lhs["shape"][:-2] != rhs["shape"][:-2]:
                    raise UnsupportedProgram("matrix batch prefixes must match without broadcasting")
                value = graph.finish(graph.product(lhs, rhs), attrs)
            elif op == "bias_add":
                _checked_attrs(attrs, _POST, {"output_dtype"})
                if len(operands) != 2:
                    raise UnsupportedProgram("invalid bias addition")
                value, bias = [graph.ref(x) for x in operands]
                if bias["shape"] != [value["shape"][-1]]:
                    raise UnsupportedProgram("bias vector extent mismatch")
                value = graph.emit("add_bias", [value, bias], value["shape"], value["dtype"])
                value = graph.finish(value, attrs)
            else:
                _checked_attrs(attrs, {"lhs_scale", "rhs_scale", "bound_lsb", "epilogue", "output_dtype", "source_scales"},
                               {"lhs_scale", "rhs_scale", "bound_lsb", "output_dtype"})
                if len(operands) != 2:
                    raise UnsupportedProgram("invalid scaled addition")
                lhs, rhs = [graph.ref(x) for x in operands]
                if lhs["shape"] != rhs["shape"] or type(attrs["bound_lsb"]) is not int or attrs["bound_lsb"] < 0:
                    raise UnsupportedProgram("inconsistent scaled-add input/tolerance")
                value = graph.emit("scaled_add", [lhs, rhs], lhs["shape"], "i32",
                                   {"lhs_factor": _factor(attrs["lhs_scale"]), "rhs_factor": _factor(attrs["rhs_scale"]),
                                    "arithmetic_dtype": "f32", "rounding": "ties_to_even_once_after_sum",
                                    "allowed_error_lsb": attrs["bound_lsb"]})
                value = graph.finish(value, attrs)
            graph.bind(result, value)
            if name != result:
                graph.bind(name, value)
            graph.publish(name, value, shape, dtype)
        else:
            raise UnsupportedProgram("unsupported mathematical operation: " + op)
    if not graph.outputs:
        raise UnsupportedProgram("source declares no mathematical outputs")
    return {"schema": SCHEMA, "inputs": graph.inputs, "expressions": graph.expressions, "outputs": graph.outputs}


def encode_source(source: bytes) -> bytes:
    """Canonical author/runtime bytes; refuse rather than leak an unsupported dialect."""
    if not isinstance(source, bytes):
        raise TypeError("source must be bytes")
    text = source.decode("utf-8")
    if "merlin_iface" not in text:
        from merlin.common.mlir_query import op_name
        from merlin.frontends.linalg_mlir import parse_mlir_text

        try:
            module = parse_mlir_text(text)
            names = [op_name(op) for op in module.walk()]
        except Exception as error:
            raise UnsupportedProgram("source is not an admitted generic mathematical MLIR program") from error
        if "func.func" not in names or not any(name.startswith(("linalg.", "tensor.", "arith.")) for name in names):
            raise UnsupportedProgram("source is not an admitted generic mathematical MLIR program")
        return source
    document = _mathematical_document(text)
    result = (json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if any(token in result for token in (b"merlin_iface", b"RES_PACK", b"COMMIT", b"EVICT", b"packed_rhs", b"resident")):
        raise UnsupportedProgram("non-mathematical source detail survived the projection")
    return result


def project_input(capsule, source: bytes) -> ProgramInput:
    """Pure projection; capsule metadata is never copied into the program payload."""
    payload = encode_source(source)
    generic = payload == source and "merlin_iface" not in source.decode("utf-8")
    return ProgramInput("generic_math_mlir_v1" if generic else FORMAT,
                        "capsule.math.mlir" if generic else "capsule.program.json", payload,
                        hashlib.sha256(source).hexdigest())


def file_transform_for(source_root: Path):
    """Callback for author_input_views.project_tree, before readonly publication."""
    root = Path(source_root).resolve()

    def view_name(path, raw):
        return path.with_suffix(".program.json") if b"merlin_iface" in raw else path

    def transform(relative: str, raw: bytes):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise UnsupportedProgram("author input view path escapes its root")
        if path.name == "capsule.yaml":
            capsule = yaml.safe_load(raw)
            declared = capsule.get("interface_mlir") if isinstance(capsule, dict) else None
            if not isinstance(declared, str):
                raise UnsupportedProgram("capsule lacks a declared complete mathematical source")
            source_path = Path(declared)
            if source_path.is_absolute() or ".." in source_path.parts:
                raise UnsupportedProgram("capsule source path escapes its directory")
            source = (root / path.parent / source_path).resolve()
            if not source.is_relative_to(root / path.parent):
                raise UnsupportedProgram("capsule source alias escapes its directory")
            source_raw = source.read_bytes()
            projected = project_input(capsule, source_raw)
            view_manifest = copy.deepcopy(capsule)
            view_manifest.pop("interface_mlir", None)
            view_manifest.pop("linalg_mlir", None)
            view_manifest["program_input"] = view_name(source_path, source_raw).as_posix()
            view_manifest["program_format"] = projected.format
            return relative, yaml.safe_dump(view_manifest, sort_keys=False).encode()
        if path.suffix == ".mlir":
            return view_name(path, raw).as_posix(), encode_source(raw)
        return relative, raw

    return transform


def contract_text() -> str:
    """Public mathematical format specification; contains no target implementation advice."""
    return """# Mathematical program inputs

Inputs use either canonical JSON with `schema: numerical_program.v1`, or an
unchanged generic mathematical MLIR program (`func`, `linalg`, `tensor`, `arith`).
The capsule manifest's `program_input` names the file and `program_format` names
`neutral_math_v1` or `generic_math_mlir_v1`. JSON is a mathematical expression DAG:
`inputs` bind an `id` to a tensor `name`, `shape`, and integer `dtype`; ordered
`expressions` bind new ids through an `operator`, earlier `operands`, result shape,
dtype and optional parameters; `outputs` bind observable tensor names to value ids.
All dimensions are static and positive. Tensor storage is row-major. Every named
output is observable, including a value also consumed by a later expression.

Source-plan identity uses the exact delivered input file bytes, not reserialized
JSON. For numerical_program.v1, source_op_indices are zero-based positions in
expressions; source_op_count is len(expressions). Each expression has one result,
so source_values.result_index is 0. Include explicit casts and epilogues; inputs
and outputs records are not operations. entry_bindings follows inputs array order;
output_bindings follows outputs array order and resolves each output's value id
to its source result. JSON expressions have no region IDs; do not invent them.
For generic mathematical MLIR, number every direct operation except func.return
in the unique defined single-block function, including constants and initializers;
exclude nested scalar-body operations. Arguments follow block order, results
follow operation result order, and returns follow func.return operand order.
The staged runtime protocol and mixed_program_plan schema specify the remaining
source/task and materialized-buffer bindings. A whole_program buffer with any
role=intermediate tensor requires params.global_program_plan even for a single
mathematical operation. readwrite does not supply a source seed: mutable scratch
must have a source-owned writer before its first read.

`matmul` contracts the last axis of its left operand with the penultimate axis of
its right operand. Matching leading batch axes select independent products. Its
result is i32. `conv2d` is cross-correlation with NHWC inputs, zero outside the
input image, flattened `(KH*KW*CI,CO)` weights, and flattened `(N*OH*OW,CO)` output.
Kernel, stride, dilation and padding are mathematical parameters; four-sided
padding always means `[top,left,bottom,right]`. Spatial output sizes are floor
divisions: `(input + before + after - dilation*(kernel-1) - 1)//stride + 1`.

`add_bias` adds one integer bias per last-axis column. `scale_round` converts the
integer and the scalar factor (or last-axis `factors`) to IEEE float32, multiplies
in float32, then rounds to an integer with ties to even. `scaled_add` converts
both integers and factors to float32, multiplies each pair in float32, sums in
float32, then rounds once with ties to even. Its `allowed_error_lsb` is the
declared numerical tolerance, not another arithmetic step. `round_shift_right`
computes `(x + 2**(shift-1)) >> shift` for positive shifts and identity otherwise.
`relu` takes max(x,0). `cast` to i8 saturates to [-128,127]; widening to i32
sign-extends. These expressions occur in the declared order: narrowing is not
implicitly fused into a preceding operation.

`max_pool` views `[batch*H*W,C]` as NHWC, takes window maxima per channel, and
returns `[batch*OH*OW,C]`. Its spatial shape, window, stride and four-sided padding
are explicit. A nonzero pad requires a declared `padding_value`; null means no
padded cells are sampled. No storage placement, tile sizes, command sequence,
resource limits, target-derived facts, or implementation schedules are supplied
by this format. The candidate chooses its own implementation.
"""
