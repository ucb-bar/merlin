"""Complete Core AST terms from authoritative buffers, without semantic hashes.

This literal serializer is a named host trust component. Source and adapter /
selected declaration exports remain separate inputs; eligibility is kernel checked.
"""

from __future__ import annotations

import json


def natural(value):
    if type(value) is not int or value < 0:
        raise ValueError("strict natural-number field required")
    return str(value)


def signed(value):
    if type(value) is str and str(int(value)) == value:
        return f"({value}:Int)"
    if type(value) is int:
        return f"({value}:Int)"
    raise ValueError("canonical signed integer required")


def text(value):
    if type(value) is not str or not value.isascii():
        raise ValueError("ASCII metadata required in this profile")
    return json.dumps(value)


def array(values, encode):
    if type(values) is not list:
        raise ValueError("list field required")
    return "[" + ",".join(encode(value) for value in values) + "]"


def optional(value, encode):
    return "none" if value is None else "(some " + encode(value) + ")"


def type_term(value):
    kind = value["kind"]
    if kind in {"int", "sint", "uint"}:
        return f"(.{kind} {natural(value['width'])})"
    if kind in {"index", "none"}:
        return "." + kind
    if kind == "float" and value["format"] in {"f16", "bf16", "f32", "f64"}:
        return f"(.float .{value['format']})"
    if kind == "tensor":
        return (
            f"(.tensor {array(value['shape'], lambda n: optional(n, natural))} "
            f"{type_term(value['element'])} {optional(value['encoding'], attribute)})"
        )
    if kind == "function":
        return f"(.function {array(value['inputs'], type_term)} {array(value['results'], type_term)})"
    if kind == "opaque":
        return f"(.opaque {text(value['text'])})"
    raise ValueError("unsupported AST type; no success evidence")


def affine(value):
    kind = value["kind"]
    if kind in {"dim", "symbol"}:
        return f"(.{'dim' if kind == 'dim' else 'sym'} {natural(value['pos'])})"
    if kind == "constant":
        return f"(.const {signed(value['value'])})"
    if kind in {"add", "mul", "mod", "floordiv", "ceildiv"}:
        return f"(.{kind} {affine(value['lhs'])} {affine(value['rhs'])})"
    raise ValueError("unsupported affine syntax")


def dictionary(values):
    if type(values) is not dict:
        raise ValueError("complete metadata dictionary required")
    return "[" + ",".join(f"({text(k)},{attribute(v)})" for k, v in sorted(values.items())) + "]"


def attribute(value):
    kind = value["kind"]
    if kind == "unit":
        return ".unit"
    if kind == "bool" and type(value["value"]) is bool:
        return ".bool " + str(value["value"]).lower()
    if kind == "int":
        return f"(.int {signed(value['value'])} {type_term(value['type'])})"
    if kind == "float":
        return f"(.float {int(value['bits'], 16)} {type_term(value['type'])})"
    if kind == "string":
        return f"(.str {text(value['value'])})"
    if kind == "type":
        return f"(.type {type_term(value['value'])})"
    if kind == "array":
        return f"(.array {array(value['elements'], attribute)})"
    if kind == "dense_array":
        return f"(.denseArray {type_term(value['element'])} {array(value['values'], signed)})"
    if kind == "dense" and type(value["splat"]) is bool:
        return (
            f"(.dense {type_term(value['type'])} {str(value['splat']).lower()} "
            f"{array(list(bytes.fromhex(value['hex'])), natural)})"
        )
    if kind == "affine_map":
        return f"(.affineMap ⟨{natural(value['dims'])},{natural(value['symbols'])},{array(value['results'], affine)}⟩)"
    if kind == "enum":
        return f"(.enum {text(value['dialect'])} {text(value['name'])} {text(value['value'])})"
    if kind == "dict":
        return f"(.dict {dictionary(value['entries'])})"
    if kind == "symbol_ref":
        return f"(.symbolRef {text(value['root'])} {array(value['nested'], text)})"
    if kind == "opaque":
        return f"(.opaque {text(value['text'])})"
    raise ValueError("unsupported attribute syntax")


def variable(value):
    return f"⟨{natural(value['id'])},{type_term(value['type'])}⟩"


def block(value):
    if set(value) != {"args", "ops"}:
        raise ValueError("complete block required")
    return f"(.mk {array(value['args'], variable)} {array(value['ops'], operation)})"


def region(value):
    if set(value) != {"blocks"}:
        raise ValueError("complete region required")
    return f"(.mk {array(value['blocks'], block)})"


def operation(value):
    if set(value) != {"name", "results", "operands", "attrs", "props", "regions", "successors"}:
        raise ValueError("complete seven-field operation required")
    return (
        f"(.mk {text(value['name'])} {array(value['results'], variable)} {array(value['operands'], natural)} "
        f"{dictionary(value['attrs'])} {dictionary(value['props'])} "
        f"{array(value['regions'], region)} {array(value['successors'], natural)})"
    )


def module(value):
    if value["schema"] != "merlin.lean.ir.v1" or set(value["module"]) != {"ops", "attrs", "props"}:
        raise ValueError("complete authoritative module required")
    body = value["module"]
    return f"⟨{array(body['ops'], operation)},{dictionary(body['attrs'])},{dictionary(body['props'])}⟩"
