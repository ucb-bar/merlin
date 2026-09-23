"""xDSL-only structural ingestion and verified interface program extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xdsl.context import Context
from xdsl.dialects import get_all_dialects
from xdsl.dialects.builtin import ArrayAttr, FloatAttr, IntegerAttr, StringAttr, TensorType
from xdsl.parser import Parser

from xdsl_dialects.gemmini import GEMMINI_DIALECT
from xdsl_dialects.merlin_iface import MERLIN_IFACE_DIALECT


@dataclass
class TensorSpec:
    name: str
    shape: list[int]
    dtype: str
    role: str


@dataclass
class InterfaceProgram:
    module: Any
    target: str = "gemmini"
    abi_version: str = "0.1"
    tensors: dict[str, TensorSpec] = field(default_factory=dict)
    commands: list[dict[str, Any]] = field(default_factory=list)
    value_names: dict[Any, str] = field(default_factory=dict)
    is_contract_module: bool = False


def make_context() -> Context:
    ctx = Context(allow_unregistered=True)
    for name, factory in get_all_dialects().items():
        try:
            ctx.register_dialect(name, factory)
        except Exception:
            pass
    ctx.load_dialect(MERLIN_IFACE_DIALECT)
    ctx.load_dialect(GEMMINI_DIALECT)
    return ctx


def parse_verified(path: str | Path) -> tuple[Context, Any]:
    text = Path(path).read_text(encoding="utf-8")
    ctx = make_context()
    module = Parser(ctx, text, str(path)).parse_module()
    module.verify()
    return ctx, module


def _attr_value(attr: Any) -> Any:
    if isinstance(attr, StringAttr):
        return attr.data
    if isinstance(attr, IntegerAttr):
        return int(attr.value.data)
    if isinstance(attr, FloatAttr):
        return float(attr.value.data)
    if isinstance(attr, ArrayAttr):
        return [_attr_value(item) for item in attr]
    data = getattr(attr, "data", None)
    if isinstance(data, (str, int, float, bool)):
        return data
    return str(attr)


def _tensor_spec(name: str, typ: TensorType, role: str) -> TensorSpec:
    return TensorSpec(name, [int(d) for d in typ.get_shape()], str(typ.element_type), role)


def _value_name(program: InterfaceProgram, value: Any) -> str:
    if value in program.value_names:
        return program.value_names[value]
    hint = getattr(value, "name_hint", None)
    if hint is None:
        hint = f"v{len(program.value_names)}"
    name = str(hint)
    program.value_names[value] = name
    return name


def extract_program(module: Any) -> InterfaceProgram:
    attrs = module.attributes
    target = getattr(attrs.get("merlin_iface.target"), "data", "gemmini")
    abi = getattr(attrs.get("merlin_iface.abi_version"), "data", "0.1")
    version = getattr(attrs.get("merlin_iface.version"), "data", None)
    program = InterfaceProgram(module=module, target=target, abi_version=abi)
    contract_ops = [op for op in module.walk() if op.name.startswith("merlin_iface.")]
    if not contract_ops:
        return program
    program.is_contract_module = True
    if version != "0.1" or abi != "0.1" or target != "gemmini":
        raise ValueError(f"unsupported interface header version={version!r} abi={abi!r} target={target!r}")

    for op in contract_ops:
        name = op.name
        attrs_py = {key: _attr_value(value) for key, value in op.attributes.items()}
        if name == "merlin_iface.tensor":
            logical = attrs_py["name"]
            spec = _tensor_spec(logical, op.results[0].type, attrs_py["role"])
            program.tensors[logical] = spec
            program.value_names[op.results[0]] = logical
        elif name == "merlin_iface.resident_pack":
            src = _value_name(program, op.operands[0])
            dst = _value_name(program, op.results[0])
            program.commands.append({"opcode": "RES_PACK", "operands": {"src": src, "dst": dst},
                                     "attributes": {"layout": attrs_py["layout"]}})
        elif name == "merlin_iface.matmul":
            lhs = _value_name(program, op.operands[0])
            rhs = _value_name(program, op.operands[1])
            dst = _value_name(program, op.results[0])
            program.commands.append({"opcode": "MATMUL_RESIDENT",
                                     "operands": {"lhs": lhs, "rhs": rhs, "dst": dst}})
        elif name == "merlin_iface.commit":
            src = _value_name(program, op.operands[0])
            dst = attrs_py.pop("name")
            spec = _tensor_spec(dst, op.results[0].type, "output")
            program.tensors[dst] = spec
            program.value_names[op.results[0]] = dst
            program.commands.append({"opcode": "COMMIT", "operands": {"src": src, "dst": dst},
                                     "attributes": attrs_py})
        elif name == "merlin_iface.evict":
            program.commands.append({"opcode": "EVICT",
                                     "operands": {"handle": _value_name(program, op.operands[0])}})
        elif name == "merlin_iface.movement":
            src = _value_name(program, op.operands[0])
            dst = attrs_py.pop("name")
            spec = _tensor_spec(dst, op.results[0].type, "output")
            program.tensors[dst] = spec
            program.value_names[op.results[0]] = dst
            attrs_py.setdefault("semantic", "mvin_mvout")
            attrs_py.setdefault("output_dtype", spec.dtype)
            program.commands.append({"opcode": "MOVEMENT", "operands": {"src": src, "dst": dst},
                                     "attributes": attrs_py})
        elif name == "merlin_iface.conv2d":
            ifm = _value_name(program, op.operands[0])
            weight = _value_name(program, op.operands[1])
            dst = attrs_py.pop("name")
            spec = _tensor_spec(dst, op.results[0].type, "output")
            program.tensors[dst] = spec
            program.value_names[op.results[0]] = dst
            program.commands.append({"opcode": "CONV2D",
                                     "operands": {"ifm": ifm, "weight": weight, "dst": dst},
                                     "attributes": attrs_py})
        elif name == "merlin_iface.attention_qk":
            q = _value_name(program, op.operands[0])
            k = _value_name(program, op.operands[1])
            dst = attrs_py.pop("name")
            spec = _tensor_spec(dst, op.results[0].type, "output")
            program.tensors[dst] = spec
            program.value_names[op.results[0]] = dst
            attrs_py.setdefault("epilogue", [])
            program.commands.append({"opcode": "ATTENTION_QK",
                                     "operands": {"q": q, "k": k, "dst": dst},
                                     "attributes": attrs_py})
    return program


def command_buffer(program: InterfaceProgram) -> dict[str, Any]:
    if not program.is_contract_module:
        return {"abi_version": "0.1", "target": "gemmini", "commands": [],
                "declined": {"reason": "upstream linalg model regions are not yet routed to the target pipeline",
                             "op": "model"}}
    tensors = {name: {"shape": list(spec.shape), "dtype": spec.dtype, "role": spec.role}
               for name, spec in program.tensors.items()}
    # The command-buffer ABI directly models movement, attention, and NHWC
    # convolution.  Preserve those interface operations instead of exposing
    # compiler-internal transpose/im2col tensors as harness operands.
    commands = list(program.commands)
    cb = {
        "abi_version": program.abi_version,
        "target": program.target,
        "tensors": tensors,
        "commands": commands,
        "resources": {"mesh": [16, 16], "scratchpad_bytes": 262144, "accumulator_bytes": 65536},
    }
    return cb
