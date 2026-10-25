"""Lossless supported Core AST literal serialization from authoritative exports.

This serialization and JSON reader are explicit trusted host components. Lean
checks complete AST/model correspondence and execution; no eligibility Booleans
or host numerical assertions enter its proof.
"""

from __future__ import annotations

import json


def string(value):
    if type(value) is not str or not value.isascii():
        raise ValueError("ASCII source literal required in this profile")
    return json.dumps(value)


def nat(value):
    if type(value) is not int or value < 0:
        raise ValueError("strict natural number required")
    return str(value)


def integer(value):
    if type(value) is int:
        return f"({value} : Int)"
    if type(value) is not str or str(int(value)) != value:
        raise ValueError("canonical signed integer required")
    return f"({value} : Int)"


def sequence(values, render):
    if type(values) is not list:
        raise ValueError("array required")
    return "[" + ",".join(render(value) for value in values) + "]"


def optional(value, render):
    return "none" if value is None else f"(some {render(value)})"


def ty(value):
    kind = value["kind"]
    if kind in ("int", "sint", "uint"):
        return f"(.{kind} {nat(value['width'])})"
    if kind in ("index", "none"):
        return "." + kind
    if kind == "float":
        if value["format"] not in ("f16", "bf16", "f32", "f64"):
            raise ValueError("unknown AST float tag")
        return f"(.float .{value['format']})"
    if kind == "tensor":
        dims = sequence(value["shape"], lambda dim: optional(dim, nat))
        return f"(.tensor {dims} {ty(value['element'])} {optional(value['encoding'], attr)})"
    if kind == "function":
        return f"(.function {sequence(value['inputs'], ty)} {sequence(value['results'], ty)})"
    if kind == "opaque":
        return f"(.opaque {string(value['text'])})"
    raise ValueError("AST type outside literal serializer; no success receipt")


def affine(value):
    kind = value["kind"]
    if kind in ("dim", "symbol"):
        tag = "dim" if kind == "dim" else "sym"
        return f"(.{tag} {nat(value['pos'])})"
    if kind == "constant":
        return f"(.const {integer(value['value'])})"
    if kind in ("add", "mul", "mod", "floordiv", "ceildiv"):
        return f"(.{kind} {affine(value['lhs'])} {affine(value['rhs'])})"
    raise ValueError("unknown affine AST")


def attributes(values):
    if type(values) is not dict:
        raise ValueError("attribute object required")
    return "[" + ",".join(f"({string(key)},{attr(value)})" for key, value in sorted(values.items())) + "]"


def attr(value):
    kind = value["kind"]
    if kind == "unit":
        return ".unit"
    if kind == "bool":
        if type(value["value"]) is not bool:
            raise ValueError("strict Boolean required")
        return ".bool " + str(value["value"]).lower()
    if kind == "int":
        return f"(.int {integer(value['value'])} {ty(value['type'])})"
    if kind == "float":
        return f"(.float {int(value['bits'], 16)} {ty(value['type'])})"
    if kind == "string":
        return f"(.str {string(value['value'])})"
    if kind == "type":
        return f"(.type {ty(value['value'])})"
    if kind == "array":
        return f"(.array {sequence(value['elements'], attr)})"
    if kind == "dense_array":
        return f"(.denseArray {ty(value['element'])} {sequence(value['values'], integer)})"
    if kind == "dense":
        raw = list(bytes.fromhex(value["hex"]))
        if type(value["splat"]) is not bool:
            raise ValueError("strict splat Boolean required")
        return f"(.dense {ty(value['type'])} {str(value['splat']).lower()} {sequence(raw, nat)})"
    if kind == "affine_map":
        return f"(.affineMap ⟨{nat(value['dims'])},{nat(value['symbols'])},{sequence(value['results'], affine)}⟩)"
    if kind == "enum":
        return f"(.enum {string(value['dialect'])} {string(value['name'])} {string(value['value'])})"
    if kind == "dict":
        return f"(.dict {attributes(value['entries'])})"
    if kind == "symbol_ref":
        return f"(.symbolRef {string(value['root'])} {sequence(value['nested'], string)})"
    if kind == "opaque":
        return f"(.opaque {string(value['text'])})"
    raise ValueError("unknown attribute AST")


def variable(value):
    return f"⟨{nat(value['id'])},{ty(value['type'])}⟩"


def region(value):
    return (
        "(.mk "
        + sequence(
            value["blocks"],
            lambda block: f"(.mk {sequence(block['args'], variable)} {sequence(block['ops'], operation)})",
        )
        + ")"
    )


def operation(value):
    expected = {"name", "results", "operands", "attrs", "props", "regions", "successors"}
    if type(value) is not dict or set(value) != expected:
        raise ValueError("complete exact exported operation required")
    return (
        f"(.mk {string(value['name'])} {sequence(value['results'], variable)} "
        f"{sequence(value['operands'], nat)} {attributes(value['attrs'])} {attributes(value['props'])} "
        f"{sequence(value['regions'], region)} {sequence(value['successors'], nat)})"
    )


def module(document):
    if document.get("schema") != "merlin.lean.ir.v1":
        raise ValueError("authoritative export schema required")
    value = document["module"]
    if set(value) != {"ops", "attrs", "props"}:
        raise ValueError("complete exported module required")
    return f"⟨{sequence(value['ops'], operation)},{attributes(value['attrs'])},{attributes(value['props'])}⟩"
