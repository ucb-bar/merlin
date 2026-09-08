"""Structural xDSL front end for both frozen interface grammars."""
from dataclasses import dataclass, field
from typing import Any

from xdsl.context import Context
from xdsl.dialects.arith import Arith
from xdsl.dialects.builtin import Builtin, StringAttr, TensorType
from xdsl.dialects.cf import Cf
from xdsl.dialects.func import Func
from xdsl.dialects.linalg import Linalg
from xdsl.dialects.math import Math
from xdsl.dialects.scf import Scf
from xdsl.dialects.tensor import Tensor
from xdsl.ir import Dialect
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, result_def
from xdsl.parser import Parser

from .fp8 import register_fp8_types


@irdl_op_definition
class IfaceTensorOp(IRDLOperation):
    name = "merlin_iface.tensor"
    result = result_def()
    assembly_format = "attr-dict `:` type($result)"


@irdl_op_definition
class IfaceUnaryOp(IRDLOperation):
    name = "merlin_iface.movement"
    src = operand_def()
    result = result_def()
    assembly_format = "$src attr-dict `:` `(` type($src) `)` `->` type($result)"


def _binary_op(op_name):
    @irdl_op_definition
    class Binary(IRDLOperation):
        name = op_name
        lhs = operand_def()
        rhs = operand_def()
        result = result_def()
        assembly_format = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)"
    return Binary


@irdl_op_definition
class IfacePackOp(IRDLOperation):
    name = "merlin_iface.resident_pack"
    src = operand_def()
    result = result_def()
    assembly_format = "$src attr-dict `:` `(` type($src) `)` `->` type($result)"


@irdl_op_definition
class IfaceMatmulOp(IRDLOperation):
    name = "merlin_iface.matmul"
    lhs = operand_def()
    rhs = operand_def()
    result = result_def()
    assembly_format = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)"


@irdl_op_definition
class IfaceCommitOp(IRDLOperation):
    name = "merlin_iface.commit"
    src = operand_def()
    result = result_def()
    assembly_format = "$src attr-dict `:` `(` type($src) `)` `->` type($result)"


@irdl_op_definition
class IfaceEvictOp(IRDLOperation):
    name = "merlin_iface.evict"
    src = operand_def()
    assembly_format = "$src attr-dict `:` `(` type($src) `)` `->` `(` `)`"


_BINARY_NAMES = (
    "rmsnorm", "attention_qk", "attention_pv", "matmul_batched", "bias_add",
)
_BINARY_CLASSES = [_binary_op("merlin_iface." + name) for name in _BINARY_NAMES]


IFACE_DIALECT = Dialect(
    "merlin_iface",
    [IfaceTensorOp, IfaceUnaryOp, IfacePackOp, IfaceMatmulOp, IfaceCommitOp, IfaceEvictOp, *_BINARY_CLASSES],
    [],
)


def _shape_dtype(typ):
    if not isinstance(typ, TensorType):
        return [], str(typ)
    shape = [int(v) for v in typ.get_shape()]
    raw = str(typ.element_type)
    dtype = {"f8E4M3FN": "fp8_e4m3", "f8E5M2": "fp8_e5m2"}.get(raw, raw)
    return shape, dtype


def _str_attr(op, key, default=""):
    value = op.attributes.get(key)
    return value.data if isinstance(value, StringAttr) else default


@dataclass
class TensorSpec:
    name: str
    shape: list[int]
    dtype: str
    role: str
    value: Any = None


@dataclass
class Workload:
    module: Any
    grammar: str
    tensors: list[TensorSpec] = field(default_factory=list)
    outputs: list[TensorSpec] = field(default_factory=list)
    ops: list[dict[str, Any]] = field(default_factory=list)


def _context():
    register_fp8_types()
    ctx = Context(allow_unregistered=True)
    for dialect in (Builtin, Func, Arith, Linalg, Tensor, Scf, Math, Cf, IFACE_DIALECT):
        ctx.load_dialect(dialect)
    return ctx


def parse_verified(text: str) -> Workload:
    module = Parser(_context(), text).parse_module()
    module.verify()
    iface_tensors = [op for op in module.walk() if op.name == "merlin_iface.tensor"]
    if iface_tensors:
        return _read_iface(module, iface_tensors)
    return _read_linalg(module)


def _read_iface(module, tensor_ops):
    workload = Workload(module, "merlin_iface")
    names = {}
    for op in tensor_ops:
        shape, dtype = _shape_dtype(op.results[0].type)
        spec = TensorSpec(_str_attr(op, "name"), shape, dtype, _str_attr(op, "role", "input"), op.results[0])
        workload.tensors.append(spec)
        names[op.results[0]] = spec.name
    handles = {}
    accs = {}
    for op in module.walk():
        if op.name == "merlin_iface.resident_pack":
            src = names[op.operands[0]]
            handle = src + "_resident"
            handles[op.results[0]] = handle
            workload.ops.append({"op": "res_pack", "src": src, "dst": handle, "attrs": _attrs(op)})
        elif op.name == "merlin_iface.matmul":
            lhs = names[op.operands[0]]
            rhs = handles.get(op.operands[1], names.get(op.operands[1], ""))
            acc = "acc" + str(len(accs))
            accs[op.results[0]] = acc
            workload.ops.append({"op": "matmul", "lhs": lhs, "rhs": rhs, "dst": acc, "attrs": _attrs(op)})
        elif op.name == "merlin_iface.commit":
            shape, dtype = _shape_dtype(op.results[0].type)
            out = TensorSpec(_str_attr(op, "name", "Y0"), shape, dtype, "output", op.results[0])
            workload.outputs.append(out)
            workload.ops.append({"op": "commit", "src": accs[op.operands[0]], "dst": out.name, "attrs": _attrs(op)})
        elif op.name == "merlin_iface.evict":
            workload.ops.append({"op": "evict", "handle": handles[op.operands[0]], "attrs": {}})
        elif op.name.startswith("merlin_iface.") and op.name not in ("merlin_iface.tensor",):
            mnemonic = op.name.split(".")[-1]
            if mnemonic in _BINARY_NAMES or mnemonic == "movement":
                operands = [names.get(v, "") for v in op.operands]
                shape, dtype = _shape_dtype(op.results[0].type)
                out = TensorSpec(_str_attr(op, "name", "Y0"), shape, dtype, "output", op.results[0])
                workload.outputs.append(out)
                workload.ops.append({"op": mnemonic, "inputs": operands, "dst": out.name, "attrs": _attrs(op)})
    return workload


def _attrs(op):
    out = {}
    for key, value in op.attributes.items():
        if isinstance(value, StringAttr):
            out[key] = value.data
        elif hasattr(value, "value") and hasattr(value.value, "data"):
            out[key] = value.value.data
        elif hasattr(value, "data"):
            out[key] = [getattr(v, "data", str(v)) for v in value.data]
    return out


def _read_linalg(module):
    workload = Workload(module, "linalg-on-tensors")
    funcs = [op for op in module.walk() if op.name == "func.func"]
    if not funcs:
        raise ValueError("interface has neither merlin_iface tensors nor func.func")
    fn = funcs[0]
    block = fn.body.blocks[0]
    args = list(block.args)
    result_types = list(fn.function_type.outputs)
    payload = []
    for op in block.ops:
        prov = _str_attr(op, "prov.op")
        if prov and prov != "fill":
            payload.append({"kind": op.name, "semantic": prov, "operands": list(op.operands), "results": list(op.results)})
    semantic = _classify_linalg(payload, module)
    names = _argument_names(semantic, len(args))
    if semantic == "add" and len(args) > 1:
        rhs_shape, _ = _shape_dtype(args[1].type)
        if len(rhs_shape) == 1:
            names = ["X", "B", *names[2:]]
    for index, arg in enumerate(args):
        shape, dtype = _shape_dtype(arg.type)
        name = names[index] if index < len(names) else "I" + str(index)
        # Linalg function arguments are host inputs.  Do not invent a special
        # bias role from the spelling "B": Atlas' loader preloads input/weight
        # roles, while explicit merlin_iface tensors retain their declared role.
        role = "weight" if name.startswith(("W", "G")) else "input"
        workload.tensors.append(TensorSpec(name, shape, dtype, role, arg))
    for index, typ in enumerate(result_types):
        shape, dtype = _shape_dtype(typ)
        workload.outputs.append(TensorSpec("Y" + str(index), shape, dtype, "output"))
    attrs = {}
    if semantic == "attention_full":
        attrs["causal"] = any(row["semantic"] in ("compare", "select")
                              for row in payload)
    elif semantic == "layernorm":
        constants = []
        for op in module.walk():
            if op.name != "arith.constant" or "value" not in op.properties:
                continue
            value = op.properties["value"]
            if hasattr(value, "value") and hasattr(value.value, "data"):
                number = float(value.value.data)
                if 0.0 < number < 1.0e-2:
                    constants.append(number)
        attrs["eps"] = min(constants) if constants else 1.0e-5
    workload.ops = [{"op": semantic, "payload": payload, "inputs": names,
                     "dst": "Y0", "attrs": attrs}]
    return workload


def _classify_linalg(payload, module):
    module_attrs = next(iter(module.walk())).attributes
    if "prov.weights_file" in module_attrs:
        return "model"
    semantics = [row["semantic"] for row in payload]
    unique = set(semantics)
    if unique == {"add"}:
        return "add"
    matmuls = sum(value in ("matmul", "batch_matmul") for value in semantics)
    if matmuls == 2 and ("softmax" in unique or
                         {"reduce_max", "exp", "reduce_sum"}.issubset(unique)):
        return "attention_full"
    if "gelu" in unique:
        return "gelu"
    if "softmax" in unique or {"reduce_max", "exp", "reduce_sum"}.issubset(unique):
        return "softmax"
    if unique == {"reduce_sum"}:
        return "reduce_sum"
    if matmuls == 2 and "sigmoid" in unique:
        return "geglu"
    if matmuls == 2:
        return "k_chain"
    if matmuls == 1 and "add" in unique:
        return "fused_matmul_bias"
    if "batch_matmul" in unique:
        return "gemv_batched"
    if "sigmoid" in unique and "mul" in unique:
        return "silu"
    if "layer_norm" in unique or ("rsqrt" in unique and "reduce_mean" in unique):
        return "layernorm"
    if "depthwise_conv2d" in unique or "convolution_im2col_matmul" in unique:
        return "depthwise_conv2d"
    if "cos" in unique and "sin" in unique:
        return "rope"
    return semantics[-1] if semantics else "unknown"


def _argument_names(op, count):
    known = {
        "gelu": ["X"], "silu": ["X"], "softmax": ["X"], "reduce_sum": ["X"], "rope": ["X"],
        "add": ["A", "B"], "fused_matmul_bias": ["X", "W", "B"], "k_chain": ["A0", "W", "W2"],
        "gemv_batched": ["A0", "V"], "geglu": ["X", "WG", "WU"],
        "attention_full": ["Q", "K", "V"],
        "layernorm": ["X", "W", "B"], "depthwise_conv2d": ["X", "W"],
    }
    if op == "model":
        return ["I" + str(i) for i in range(count)]
    return known.get(op, ["I" + str(i) for i in range(count)])
