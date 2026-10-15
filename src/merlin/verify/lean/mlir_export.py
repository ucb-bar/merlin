"""Authoritative ``merlin.lean.ir.v1`` exporter: MLIR's own parser via ``torch_mlir.ir`` (design D9).

The paper path lowers with the MLIR bundled in ``torch_mlir`` (design §2.1), so this parser's
reading is the compiled reading. This file is self-contained (no ``merlin`` imports): it runs in the
model2MLIR environment as ``<m2m-python> mlir_export.py <in.mlir> <out.json>``.

The bindings do not separate inherent from discardable attributes; every attribute is emitted under
``attrs`` and ``props`` is empty. That is semantically equivalent (op semantics look attributes up in
both) and the round trip prints them back as one dictionary, which MLIR re-sorts into properties.
Both sides of a boundary must be exported by the same exporter.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys

from torch_mlir import ir

DYNAMIC = -9223372036854775808


def _shape(t) -> list:
    return [None if d == DYNAMIC or d < 0 else int(d) for d in t.shape]


def type_json(t) -> dict:
    if isinstance(t, ir.IntegerType):
        it = ir.IntegerType(t)
        kind = "int" if it.is_signless else ("sint" if it.is_signed else "uint")
        return {"kind": kind, "width": it.width}
    if isinstance(t, ir.IndexType):
        return {"kind": "index"}
    for cls, fmt in ((ir.F16Type, "f16"), (ir.BF16Type, "bf16"), (ir.F32Type, "f32"), (ir.F64Type, "f64")):
        if isinstance(t, cls):
            return {"kind": "float", "format": fmt}
    if isinstance(t, ir.RankedTensorType):
        rt = ir.RankedTensorType(t)
        enc = rt.encoding
        return {"kind": "tensor", "shape": _shape(rt), "element": type_json(rt.element_type),
                "encoding": None if enc is None else attr_json(enc)}
    if isinstance(t, ir.FunctionType):
        ft = ir.FunctionType(t)
        return {"kind": "function", "inputs": [type_json(x) for x in ft.inputs],
                "results": [type_json(x) for x in ft.results]}
    if isinstance(t, ir.NoneType):
        return {"kind": "none"}
    return {"kind": "opaque", "text": str(t)}


def _float_bits(value: float, fmt: str) -> int:
    if fmt == "f64":
        return struct.unpack("<Q", struct.pack("<d", value))[0]
    bits32 = struct.unpack("<I", struct.pack("<f", value))[0]
    if fmt == "f32":
        return bits32
    if fmt == "f16":
        return struct.unpack("<H", struct.pack("<e", value))[0]
    if fmt == "bf16":
        if bits32 & 0xFFFF:
            raise ValueError(f"{value!r} is not a bf16 value")
        return bits32 >> 16
    raise ValueError(fmt)


def _elem_bytes(value, elt: dict) -> bytes:
    if elt["kind"] in ("int", "sint", "uint"):
        n = 1 if elt["width"] == 1 else (elt["width"] + 7) // 8
        return (int(value) % (1 << (8 * n))).to_bytes(n, "little")
    if elt["kind"] == "index":
        return (int(value) % (1 << 64)).to_bytes(8, "little")
    if elt["kind"] == "float":
        width = {"f16": 2, "bf16": 2, "f32": 4, "f64": 8}[elt["format"]]
        return _float_bits(float(value), elt["format"]).to_bytes(width, "little")
    raise ValueError(f"no byte encoding for {elt}")


def affine_json(e) -> dict:
    for cls, kind in ((ir.AffineAddExpr, "add"), (ir.AffineMulExpr, "mul"), (ir.AffineModExpr, "mod"),
                      (ir.AffineFloorDivExpr, "floordiv"), (ir.AffineCeilDivExpr, "ceildiv")):
        if isinstance(e, cls):
            b = cls(e)
            return {"kind": kind, "lhs": affine_json(b.lhs), "rhs": affine_json(b.rhs)}
    if isinstance(e, ir.AffineDimExpr):
        return {"kind": "dim", "pos": ir.AffineDimExpr(e).position}
    if isinstance(e, ir.AffineSymbolExpr):
        return {"kind": "symbol", "pos": ir.AffineSymbolExpr(e).position}
    if isinstance(e, ir.AffineConstantExpr):
        return {"kind": "constant", "value": str(ir.AffineConstantExpr(e).value)}
    raise ValueError(f"unsupported affine expression {e}")


def _enum_or_opaque(a) -> dict:
    text = str(a)
    if text.startswith("#") and text.endswith(">") and "<" in text:
        head, _, rest = text[1:].partition("<")
        dialect, dot, name = head.partition(".")
        if dot and dialect and name and " " not in head:
            return {"kind": "enum", "dialect": dialect, "name": name, "value": rest[:-1]}
    return {"kind": "opaque", "text": text}


def _dense_elements(a) -> dict:
    d = ir.DenseElementsAttr(a)
    ty = type_json(d.type)
    elt = ty.get("element")
    if ty["kind"] != "tensor" or elt is None or elt["kind"] not in ("int", "sint", "uint", "index", "float"):
        return {"kind": "opaque", "text": str(a)}
    splat = bool(d.is_splat)
    if isinstance(a, ir.DenseIntElementsAttr):
        da = ir.DenseIntElementsAttr(a)
        vals = [da.get_splat_value().value] if splat else [da[i] for i in range(len(da))]
        if splat and elt["kind"] in ("int", "sint", "uint") and elt["width"] == 1:
            vals = [int(bool(v)) for v in vals]
    else:
        da = ir.DenseFPElementsAttr(a)
        vals = [ir.FloatAttr(da.get_splat_value()).value] if splat else [da[i] for i in range(len(da))]
    return {"kind": "dense", "type": ty, "splat": splat, "hex": b"".join(_elem_bytes(v, elt) for v in vals).hex()}


def attr_json(a) -> dict:
    if isinstance(a, ir.UnitAttr):
        return {"kind": "unit"}
    if isinstance(a, ir.BoolAttr):
        return {"kind": "int", "value": str(int(ir.BoolAttr(a).value)), "type": {"kind": "int", "width": 1}}
    if isinstance(a, ir.IntegerAttr):
        ia = ir.IntegerAttr(a)
        return {"kind": "int", "value": str(ia.value), "type": type_json(ia.type)}
    if isinstance(a, ir.FloatAttr):
        fa = ir.FloatAttr(a)
        ty = type_json(fa.type)
        return {"kind": "float", "bits": format(_float_bits(fa.value, ty["format"]), "x"), "type": ty}
    if isinstance(a, ir.StringAttr):
        return {"kind": "string", "value": ir.StringAttr(a).value}
    if isinstance(a, ir.ArrayAttr):
        return {"kind": "array", "elements": [attr_json(x) for x in ir.ArrayAttr(a)]}
    for cls, elt in ((ir.DenseI8ArrayAttr, 8), (ir.DenseI16ArrayAttr, 16), (ir.DenseI32ArrayAttr, 32),
                     (ir.DenseI64ArrayAttr, 64), (ir.DenseBoolArrayAttr, 1)):
        if isinstance(a, cls):
            return {"kind": "dense_array", "element": {"kind": "int", "width": elt},
                    "values": [str(int(v)) for v in cls(a)]}
    if isinstance(a, ir.DenseElementsAttr):
        return _dense_elements(a)
    if isinstance(a, ir.AffineMapAttr):
        m = ir.AffineMapAttr(a).value
        return {"kind": "affine_map", "dims": m.n_dims, "symbols": m.n_symbols,
                "results": [affine_json(e) for e in m.results]}
    if isinstance(a, ir.DictAttr):
        da = ir.DictAttr(a)
        return {"kind": "dict", "entries": {n.name: attr_json(n.attr) for n in sorted(da, key=lambda x: x.name)}}
    if isinstance(a, ir.FlatSymbolRefAttr):
        return {"kind": "symbol_ref", "root": ir.FlatSymbolRefAttr(a).value, "nested": []}
    if isinstance(a, ir.TypeAttr):
        return {"kind": "type", "value": type_json(ir.TypeAttr(a).value)}
    return _enum_or_opaque(a)


def _matching(text: str, start: int, open_c: str, close_c: str) -> int:
    """Index of the delimiter closing ``text[start]`` (== ``open_c``), skipping string literals."""
    depth, i, n = 0, start, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            i += 1
            while i < n and text[i] != '"':
                i += 2 if text[i] == "\\" else 1
        elif c == open_c:
            depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("unbalanced delimiters in generic op print")


def _unregistered_props(o) -> dict:
    """Properties of an UNREGISTERED op. The bindings expose only its discardable attributes, so the
    property dictionary is taken from the op's own generic print (balanced, string-aware scan) and
    parsed by MLIR's attribute parser. Failing to recover it fails the export: never a silent drop."""
    asm = o.get_asm(print_generic_op_form=True)
    head = f'"{o.name}"('
    at = asm.find(head)
    if at < 0:
        raise ValueError(f"cannot locate {o.name} in its own print")
    close = _matching(asm, at + len(head) - 1, "(", ")")
    rest = asm[close + 1:].lstrip()
    if not rest.startswith("<{"):
        return {}
    body_start = asm.index("<{", close)
    end = _matching(asm, body_start + 1, "{", "}")
    d = ir.Attribute.parse(asm[body_start + 1:end + 1])
    if not isinstance(d, ir.DictAttr):
        raise ValueError(f"{o.name}: properties are not a dictionary")
    return {n.name: attr_json(n.attr) for n in sorted(d, key=lambda x: x.name)}


def export(text: str) -> dict:
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = ir.Module.parse(text)
        ids: dict = {}

        def number(block) -> None:
            for arg in block.arguments:
                ids[arg] = len(ids)
            for op in block.operations:
                o = op.operation
                for r in o.results:
                    ids[r] = len(ids)
                for reg in o.regions:
                    for b in reg.blocks:
                        number(b)

        number(module.body)

        def op_json(op) -> dict:
            o = op.operation
            # The semantics models single-block regions only; refuse branches rather than export a
            # representation that drops them (the round trip then fails and the boundary abstains).
            if len(o.successors) or any(len(reg.blocks) > 1 for reg in o.regions):
                raise ValueError(f"{o.name}: successors and multi-block regions are outside the exported fragment")
            props = {} if ctx.is_registered_operation(o.name) else _unregistered_props(o)
            return {
                "name": o.name,
                "results": [{"id": ids[r], "type": type_json(r.type)} for r in o.results],
                "operands": [ids[v] for v in o.operands],
                "attrs": {n: attr_json(o.attributes[n]) for n in sorted(o.attributes[i].name for i in range(len(o.attributes)))},
                "props": props,
                "regions": [{"blocks": [{"args": [{"id": ids[a], "type": type_json(a.type)} for a in b.arguments],
                                         "ops": [op_json(x) for x in b.operations]} for b in reg.blocks]}
                            for reg in o.regions],
                "successors": [],
            }

        mod_op = module.operation
        return {
            "schema": "merlin.lean.ir.v1",
            "source": {"sha256": hashlib.sha256(text.encode()).hexdigest(), "bytes": len(text.encode())},
            "producer": {"tool": "merlin.verify.lean.mlir_export(torch_mlir)", "version": "1"},
            "module": {
                "ops": [op_json(x) for x in module.body.operations],
                "attrs": {n: attr_json(mod_op.attributes[n]) for n in sorted(mod_op.attributes[i].name for i in range(len(mod_op.attributes)))},
                "props": {},
            },
        }


def canonical(text: str) -> str:
    """Generic-form print by this same MLIR (the round trip's comparison form)."""
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = ir.Module.parse(text)
        return module.operation.get_asm(print_generic_op_form=True)


if __name__ == "__main__":
    if sys.argv[1] == "--canonical":
        with open(sys.argv[2]) as f:
            sys.stdout.write(canonical(f.read()))
    else:
        src, dst = sys.argv[1], sys.argv[2]
        with open(src) as f:
            doc = export(f.read())
        with open(dst, "w") as f:
            json.dump(doc, f)
