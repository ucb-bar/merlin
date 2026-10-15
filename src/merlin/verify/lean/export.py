"""Export MLIR to the ``merlin.lean.ir.v1`` JSON the Lean checkers decode, and round-trip it.

Design D9 (``docs/design/merlin_lean_verification.md``). This module walks an xDSL module, read
through Merlin's own parser (``frontends.linalg_mlir.parse_mlir_text``), so it sees exactly what
Merlin's passes see. The round trip closes the loop against MLIR's own parser: the JSON is decoded
and printed by Lean (``merlin-lean print``), and ``mlir-opt --mlir-print-op-generic`` must print the
original file and the Lean-printed file to identical bytes. A difference means the verified
reading and the compiled reading of the program disagree; it is reported, never ignored.

Types and attributes are emitted structurally from xDSL's objects. Dialect enum attributes are split
from their printed form ``#dialect.name<value>`` with ``str.partition`` (no regex). Anything without
a structural encoding is emitted as ``opaque`` with its printed text, never dropped.
"""

from __future__ import annotations

import hashlib
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "merlin.lean.ir.v1"
PRODUCER = "merlin.verify.lean.export(xdsl)"


class ExportError(ValueError):
    """The module cannot be represented faithfully in ``merlin.lean.ir.v1``."""


# --- types --------------------------------------------------------------------------------------


def _shape(dims) -> list[int | None]:
    return [None if d < 0 else int(d) for d in dims]


def type_json(t) -> dict[str, Any]:
    from xdsl.dialects import builtin as b

    if isinstance(t, b.IntegerType):
        kind = {b.Signedness.SIGNLESS: "int", b.Signedness.SIGNED: "sint", b.Signedness.UNSIGNED: "uint"}
        return {"kind": kind[t.signedness.data], "width": t.width.data}
    if isinstance(t, b.IndexType):
        return {"kind": "index"}
    for cls, fmt in ((b.Float16Type, "f16"), (b.BFloat16Type, "bf16"), (b.Float32Type, "f32"), (b.Float64Type, "f64")):
        if isinstance(t, cls):
            return {"kind": "float", "format": fmt}
    if isinstance(t, b.TensorType):
        enc = t.encoding
        return {
            "kind": "tensor",
            "shape": _shape(t.get_shape()),
            "element": type_json(t.element_type),
            "encoding": None if isinstance(enc, b.NoneAttr) else attr_json(enc),
        }
    if isinstance(t, b.FunctionType):
        return {
            "kind": "function",
            "inputs": [type_json(x) for x in t.inputs.data],
            "results": [type_json(x) for x in t.outputs.data],
        }
    if isinstance(t, b.NoneType):
        return {"kind": "none"}
    return {"kind": "opaque", "text": str(t)}


# --- attributes ---------------------------------------------------------------------------------


def _float_bits(value: float, fmt: str) -> int:
    if fmt == "f64":
        return struct.unpack("<Q", struct.pack("<d", value))[0]
    if fmt == "f32":
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        if struct.unpack("<f", struct.pack("<I", bits))[0] != value and value == value:
            raise ExportError(f"float {value!r} is not exactly representable as f32")
        return bits
    if fmt == "f16":
        return struct.unpack("<H", struct.pack("<e", value))[0]
    if fmt == "bf16":
        bits32 = struct.unpack("<I", struct.pack("<f", value))[0]
        if bits32 & 0xFFFF:
            raise ExportError(f"float {value!r} is not exactly representable as bf16")
        return bits32 >> 16
    raise ExportError(f"unsupported float format {fmt}")


def _elem_bytes(value, elt: dict[str, Any]) -> bytes:
    kind = elt["kind"]
    if kind in ("int", "sint", "uint"):
        width = elt["width"]
        nbytes = 1 if width == 1 else (width + 7) // 8
        return (int(value) % (1 << (8 * nbytes))).to_bytes(nbytes, "little")
    if kind == "index":
        return (int(value) % (1 << 64)).to_bytes(8, "little")
    if kind == "float":
        fmt = elt["format"]
        width = {"f16": 2, "bf16": 2, "f32": 4, "f64": 8}[fmt]
        return _float_bits(float(value), fmt).to_bytes(width, "little")
    raise ExportError(f"dense element type {elt} has no byte encoding")


def affine_json(e) -> dict[str, Any]:
    from xdsl.ir import affine as af

    if isinstance(e, af.AffineDimExpr):
        return {"kind": "dim", "pos": e.position}
    if isinstance(e, af.AffineSymExpr):
        return {"kind": "symbol", "pos": e.position}
    if isinstance(e, af.AffineConstantExpr):
        return {"kind": "constant", "value": str(e.value)}
    if isinstance(e, af.AffineBinaryOpExpr):
        kind = {
            af.AffineBinaryOpKind.Add: "add",
            af.AffineBinaryOpKind.Mul: "mul",
            af.AffineBinaryOpKind.Mod: "mod",
            af.AffineBinaryOpKind.FloorDiv: "floordiv",
            af.AffineBinaryOpKind.CeilDiv: "ceildiv",
        }[e.kind]
        return {"kind": kind, "lhs": affine_json(e.lhs), "rhs": affine_json(e.rhs)}
    raise ExportError(f"unsupported affine expression {e}")


def _enum_or_opaque(a) -> dict[str, Any]:
    text = str(a)
    if text.startswith("#") and text.endswith(">") and "<" in text:
        head, _, rest = text[1:].partition("<")
        dialect, dot, name = head.partition(".")
        if dot and dialect and name and " " not in head:
            return {"kind": "enum", "dialect": dialect, "name": name, "value": rest[:-1]}
    return {"kind": "opaque", "text": text}


def attr_json(a) -> dict[str, Any]:
    from xdsl.dialects import builtin as b

    if isinstance(a, b.UnitAttr):
        return {"kind": "unit"}
    if isinstance(a, b.IntegerAttr):
        return {"kind": "int", "value": str(int(a.value.data)), "type": type_json(a.type)}
    if isinstance(a, b.FloatAttr):
        ty = type_json(a.type)
        if ty["kind"] != "float":
            raise ExportError(f"FloatAttr of non-float type {a.type}")
        return {"kind": "float", "bits": format(_float_bits(a.value.data, ty["format"]), "x"), "type": ty}
    if isinstance(a, b.StringAttr):
        return {"kind": "string", "value": a.data}
    if isinstance(a, b.ArrayAttr):
        return {"kind": "array", "elements": [attr_json(x) for x in a.data]}
    if isinstance(a, b.DenseArrayBase):
        elt = type_json(a.elt_type)
        if elt["kind"] not in ("int", "sint", "uint"):
            return {"kind": "opaque", "text": str(a)}
        return {"kind": "dense_array", "element": elt, "values": [str(int(v)) for v in a.get_values()]}
    if isinstance(a, b.DenseIntOrFPElementsAttr):
        ty = type_json(a.get_type())
        elt = ty.get("element")
        if ty["kind"] != "tensor" or elt is None:
            return {"kind": "opaque", "text": str(a)}
        values = a.get_values()
        if values and isinstance(values[0], tuple):  # complex
            return {"kind": "opaque", "text": str(a)}
        splat = bool(a.is_splat())
        stored = values[:1] if splat else values
        data = b"".join(_elem_bytes(v, elt) for v in stored)
        return {"kind": "dense", "type": ty, "splat": splat, "hex": data.hex()}
    if isinstance(a, b.AffineMapAttr):
        m = a.data
        return {
            "kind": "affine_map",
            "dims": m.num_dims,
            "symbols": m.num_symbols,
            "results": [affine_json(e) for e in m.results],
        }
    if isinstance(a, b.DictionaryAttr):
        return {"kind": "dict", "entries": {k: attr_json(v) for k, v in sorted(a.data.items())}}
    if isinstance(a, b.SymbolRefAttr):
        return {
            "kind": "symbol_ref",
            "root": a.root_reference.data,
            "nested": [r.data for r in a.nested_references.data],
        }
    if isinstance(a, b.TypeAttribute):
        return {"kind": "type", "value": type_json(a)}
    return _enum_or_opaque(a)


# --- ops and modules ----------------------------------------------------------------------------


def _number_values(module) -> dict[Any, int]:
    """Module-unique ids in program order: block arguments, then op results."""
    ids: dict[Any, int] = {}

    def block(bl) -> None:
        for arg in bl.args:
            ids[arg] = len(ids)
        for op in bl.ops:
            for r in op.results:
                ids[r] = len(ids)
            for reg in op.regions:
                for b2 in reg.blocks:
                    block(b2)

    for reg in module.regions:
        for bl in reg.blocks:
            block(bl)
    return ids


def _op_json(op, ids: dict[Any, int]) -> dict[str, Any]:
    def value(v) -> dict[str, Any]:
        return {"id": ids[v], "type": type_json(v.type)}

    def operand_id(v) -> int:
        if v not in ids:
            raise ExportError(f"{op.name}: operand defined outside the module")
        return ids[v]

    from xdsl.dialects.builtin import UnregisteredOp

    name = op.name
    attributes = dict(op.attributes)
    if isinstance(op, UnregisteredOp):
        # xDSL keeps an unregistered op as `builtin.unregistered` plus an `op_name__` attribute.
        name = op.op_name.data
        attributes.pop("op_name__", None)
    # The semantics models single-block regions only; refuse branches rather than export a
    # representation the Lean printer cannot reproduce (the round trip then fails and the boundary
    # abstains).
    if op.successors or any(len(reg.blocks) > 1 for reg in op.regions):
        raise ValueError(f"{name}: successors and multi-block regions are outside the exported fragment")
    return {
        "name": name,
        "results": [value(r) for r in op.results],
        "operands": [operand_id(v) for v in op.operands],
        "attrs": {k: attr_json(v) for k, v in sorted(attributes.items())},
        "props": {k: attr_json(v) for k, v in sorted(op.properties.items())},
        "regions": [
            {"blocks": [{"args": [value(a) for a in bl.args], "ops": [_op_json(o, ids) for o in bl.ops]}
                        for bl in reg.blocks]}
            for reg in op.regions
        ],
        "successors": [],
    }


def export_module(module, *, source: bytes) -> dict[str, Any]:
    """``merlin.lean.ir.v1`` document for an xDSL ``builtin.module`` read from ``source`` bytes."""
    ids = _number_values(module)
    body = [o for reg in module.regions for bl in reg.blocks for o in bl.ops]
    return {
        "schema": SCHEMA,
        "source": {"sha256": hashlib.sha256(source).hexdigest(), "bytes": len(source)},
        "producer": {"tool": PRODUCER, "version": "1"},
        "module": {
            "ops": [_op_json(o, ids) for o in body],
            "attrs": {k: attr_json(v) for k, v in sorted(module.attributes.items())},
            "props": {k: attr_json(v) for k, v in sorted(module.properties.items())},
        },
    }


def export_text(text: str) -> dict[str, Any]:
    from merlin.frontends.linalg_mlir import parse_mlir_text

    return export_module(parse_mlir_text(text), source=text.encode())


# --- round trip ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class RoundTrip:
    status: str  # "identical" | "different" | "export_failed" | "tool_failed"
    detail: str = ""


MLIR_EXPORT = Path(__file__).with_name("mlir_export.py")


def export_file(mlir_path: Path, json_path: Path, *, backend: str = "mlir", m2m_python: Path | None = None) -> None:
    """Export ``mlir_path`` to ``json_path``. ``backend="mlir"`` (authoritative, design D9) parses with
    the MLIR the pipeline compiles with (``torch_mlir`` in the model2MLIR environment);
    ``backend="xdsl"`` reads through Merlin's xDSL parser. Both sides of a boundary must use one backend."""
    import json

    if backend == "xdsl":
        json_path.write_text(json.dumps(export_text(mlir_path.read_text())))
        return
    if m2m_python is None:
        raise ExportError("the mlir backend needs the model2MLIR interpreter (m2m_python)")
    proc = subprocess.run([str(m2m_python), str(MLIR_EXPORT), str(mlir_path), str(json_path)],
                          capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise ExportError(proc.stderr.strip()[-800:])


def canonical_generic(path: Path, *, backend: str, mlir_opt: Path | None = None,
                      m2m_python: Path | None = None) -> subprocess.CompletedProcess:
    if backend == "mlir":
        return subprocess.run([str(m2m_python), str(MLIR_EXPORT), "--canonical", str(path)],
                              capture_output=True, text=True, check=False)
    return subprocess.run(
        [str(mlir_opt), "--mlir-print-op-generic", "--allow-unregistered-dialect", str(path)],
        capture_output=True, text=True, check=False,
    )


def round_trip(mlir_path: Path, workdir: Path, *, lean_exe: Path, mlir_opt: Path | None = None,
               backend: str = "xdsl", m2m_python: Path | None = None) -> RoundTrip:
    """Export ``mlir_path``, print it back through Lean, and compare canonical generic prints made by
    the same MLIR (``mlir_opt`` for the xdsl backend; torch_mlir for the mlir backend)."""
    workdir.mkdir(parents=True, exist_ok=True)
    jpath = workdir / "export.json"
    try:
        export_file(mlir_path, jpath, backend=backend, m2m_python=m2m_python)
    except Exception as exc:  # the reason is the result
        return RoundTrip("export_failed", f"{type(exc).__name__}: {exc}")
    lpath = workdir / "lean.mlir"
    proc = subprocess.run([str(lean_exe), "print", str(jpath), str(lpath)], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return RoundTrip("tool_failed", f"merlin-lean print: {proc.stderr.strip()[:500]}")
    kw = {"backend": backend, "mlir_opt": mlir_opt, "m2m_python": m2m_python}
    a = canonical_generic(mlir_path, **kw)
    if a.returncode != 0:
        return RoundTrip("tool_failed", f"canonical print of original: {a.stderr.strip()[-500:]}")
    b = canonical_generic(lpath, **kw)
    if b.returncode != 0:
        return RoundTrip("different", f"MLIR rejects the Lean print: {b.stderr.strip()[-500:]}")
    if a.stdout == b.stdout:
        return RoundTrip("identical")
    for i, (x, y) in enumerate(zip(a.stdout.splitlines(), b.stdout.splitlines())):
        if x != y:
            return RoundTrip("different", f"line {i + 1}:\n  original: {x[:300]}\n  lean:     {y[:300]}")
    return RoundTrip("different", "outputs differ in length")


# --- what a function-scoped verdict does not cover ------------------------------------------------


def outside_function_changes(before: dict[str, Any], after: dict[str, Any], fn: str) -> list[str]:
    """Differences between two exported modules that lie OUTSIDE function ``fn``'s body.

    A C1 verdict is a claim about the meaning of ``fn``'s body only. Anything listed here (module
    attributes, the function op's own attributes, other top-level ops) is outside that claim and
    must be accounted for by another checker (e.g. the erasure check) or reported as unverified.
    """
    out: list[str] = []
    mb, ma = before["module"], after["module"]
    for key in ("attrs", "props"):
        if mb.get(key, {}) != ma.get(key, {}):
            keys = sorted(set(mb.get(key, {})) ^ set(ma.get(key, {})) | {
                k for k in set(mb.get(key, {})) & set(ma.get(key, {})) if mb[key][k] != ma[key][k]})
            out.append(f"module {key}: {', '.join(keys)}")

    def sym(op: dict[str, Any]) -> str | None:
        name = op["props"].get("sym_name") or op["attrs"].get("sym_name")
        return name.get("value") if isinstance(name, dict) else None

    tb = {(o["name"], sym(o)): o for o in mb["ops"]}
    ta = {(o["name"], sym(o)): o for o in ma["ops"]}
    for k in sorted(set(tb) ^ set(ta), key=str):
        out.append(f"top-level op {'removed' if k in tb else 'added'}: {k[0]} @{k[1]}")
    for k in sorted(set(tb) & set(ta), key=str):
        ob, oa = tb[k], ta[k]
        for key in ("attrs", "props"):
            diff = sorted(x for x in set(ob[key]) | set(oa[key]) if ob[key].get(x) != oa[key].get(x))
            if diff:
                out.append(f"{k[0]} @{k[1]} {key}: {', '.join(diff)}")
        if k[1] != fn and ob["regions"] != oa["regions"]:
            out.append(f"{k[0]} @{k[1]}: body changed (not checked by this verdict)")
    return out
