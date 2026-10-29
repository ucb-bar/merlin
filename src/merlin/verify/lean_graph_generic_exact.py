"""Actual pinned scalar producer and independent complete typed SSA projection.

No witness counts or optimizer replay plans authorize this correspondence.
Every module/function/region/call operand is parsed from both actual buffers.
Unknown CFG/region semantics are refused, never erased as an empty boundary.
"""

from __future__ import annotations

import hashlib
import json
import struct
from enum import Enum
from pathlib import Path

from .lean_graph_generic_affine import affine_payload

REVISION = "4499eebb959c0f92cefd92cc6d80646ba66474ed"


def file_ref(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def exact(value):
    from xdsl.ir import Attribute, Data, ParametrizedAttribute

    kind = type(value)
    if value is None:
        return ["none"]
    if kind in (bool, int, str):
        return [kind.__name__, value]
    if kind is float:
        return ["float64-bits", struct.pack(">d", value).hex()]
    if kind is bytes:
        return ["bytes", value.hex()]
    if isinstance(value, Enum):
        return [kind.__module__ + "." + kind.__qualname__, value.name, exact(value.value)]
    if kind in (tuple, list):
        return [kind.__name__, [exact(item) for item in value]]
    if kind in (set, frozenset):
        return [kind.__name__, sorted([exact(item) for item in value], key=canonical)]
    if kind is dict:
        return ["dictionary", [[key, exact(item)] for key, item in sorted(value.items())]]
    if isinstance(value, ParametrizedAttribute):
        return [
            "attribute",
            kind.__module__ + "." + kind.__qualname__,
            value.name,
            [exact(item) for item in value.parameters],
        ]
    if isinstance(value, Data):
        return ["data-attribute", kind.__module__ + "." + kind.__qualname__, value.name, exact(value.data)]
    if (affine := affine_payload(value)) is not None:
        return ["affine-data", exact(affine)]
    if isinstance(value, Attribute):
        raise ValueError("unsupported complete opaque attribute representation: " + kind.__qualname__)
    raise ValueError("unsupported complete parsed representation: " + kind.__qualname__)


def metadata(operation):
    return {
        "class": type(operation).__module__ + "." + type(operation).__qualname__,
        "name": operation.name,
        "attributes": exact(operation.attributes),
        "properties": exact(operation.properties),
        "operand_types": [exact(value.type) for value in operation.operands],
        "result_types": [exact(value.type) for value in operation.results],
    }


def selected_primitive(operation):
    """Independent closed-whitelist reconstruction from actual operations."""
    from xdsl.dialects import arith, builtin
    from xdsl.traits import get_effects

    kinds = {
        arith.AddiOp: "add",
        arith.SubiOp: "sub",
        arith.MuliOp: "mul",
        arith.ExtSIOp: "sext",
        arith.ExtUIOp: "zext",
        arith.TruncIOp: "trunc",
        arith.ConstantOp: "constant",
    }
    name = kinds.get(type(operation))
    if name is None or operation.regions or operation.successors or operation.attributes or len(operation.results) != 1:
        return None
    if get_effects(operation) != set():
        return None
    types = [value.type for value in (*operation.operands, *operation.results)]
    if any(
        type(t) is not builtin.IntegerType
        or t.signedness.data != builtin.Signedness.SIGNLESS
        or not 1 <= t.width.data <= 256
        for t in types
    ):
        return None
    allowed = {"value"} if name == "constant" else {"overflowFlags"} if name in ("add", "sub", "mul") else set()
    if set(operation.properties) - allowed:
        return None
    if name in ("add", "sub", "mul") and operation.overflow_flags != arith.IntegerOverflowAttr("none"):
        return None
    if name == "constant":
        if not isinstance(operation.value, builtin.IntegerAttr):
            return None
        return {"kind": name, "width": types[-1].width.data, "value": operation.value.value.data}
    if name in ("add", "sub", "mul"):
        return {"kind": name, "width": types[-1].width.data}
    return {"kind": name, "source": types[0].width.data, "target": types[-1].width.data}
