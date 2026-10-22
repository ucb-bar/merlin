"""Positive lifting fixtures with distinct named values, nested regions, calls and full outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from merlin.verify.lean.differential import Tools, ValueType, execute_mlir
from merlin.verify.lean.qinner import actual


def fixture(kind="f32", shape=(4,), *, nested=False, legacy=False):
    dims = "".join(f"{n}x" for n in shape)
    ty = f"tensor<{dims}{kind}>"
    left = "w" if legacy else "0"
    right = "s" if legacy else "0"
    # Legacy aliases choose operand 1 for scale, so the second cast is a two-operand generic
    # only in the separate legacy fixture below; regular fixture uses numeric operand zero.
    if legacy:
        return f"""builtin.module {{
 func.func @forward(%x: {ty}) -> ({ty}, {ty}, {ty}) {{
  %a = tensor.empty() : {ty}
  %b = tensor.empty() : {ty}
  %u = "test.pair"(%a, %b)
    {{prov.quant_inner_{left} = "left", prov.quant_inner_{right} = "right"}} : ({ty}, {ty}) -> {ty}
  func.return %a, %b, %x : {ty}, {ty}, {ty}
 }}
}}"""
    prefix = f"""builtin.module {{
 func.func @helper(%v: {ty}) -> {ty} {{ func.return %v : {ty} }}
 func.func @forward(%x: {ty}, %flag: i1) -> ({ty}, {ty}, {ty}) {{"""
    if nested:
        body = f"""
  %v = scf.if %flag -> ({ty}) {{
   %a = tensor.empty() : {ty}
   %ac = "tensor.cast"(%a) {{prov.quant_inner_0 = "left"}} : ({ty}) -> {ty}
   scf.yield %ac : {ty}
  }} else {{
   %b = tensor.empty() : {ty}
   %bc = "tensor.cast"(%b) {{prov.quant_inner_0 = "right"}} : ({ty}) -> {ty}
   scf.yield %bc : {ty}
  }}
  %r = func.call @helper(%v) : ({ty}) -> {ty}
  func.return %r, %x, %v : {ty}, {ty}, {ty}"""
    else:
        body = f"""
  %a = tensor.empty() : {ty}
  %b = tensor.empty() : {ty}
  %ac = "tensor.cast"(%a) {{prov.quant_inner_0 = "left"}} : ({ty}) -> {ty}
  %bc = "tensor.cast"(%b) {{prov.quant_inner_0 = "right"}} : ({ty}) -> {ty}
  %r = func.call @helper(%ac) : ({ty}) -> {ty}
  func.return %bc, %r, %x : {ty}, {ty}, {ty}"""
    return prefix + body + "\n }\n}\n"


def arrays(kind="f32", shape=(4,)):
    dtype = {
        "f32": np.float32,
        "f64": np.float64,
        "f16": np.float16,
        "i8": np.int8,
        "i16": np.int16,
        "i32": np.int32,
        "i64": np.int64,
        "i1": np.bool_,
    }[kind]
    count = math.prod(shape)
    return {
        "qinner::left": (np.arange(count) + 1).astype(dtype).reshape(shape),
        "qinner::right": (np.arange(count) + 21).astype(dtype).reshape(shape),
    }


def reference(source, kind, shape):
    """Independent valid source-binding elaboration: concrete initialized dense constants."""
    ty = f"tensor<{''.join(f'{n}x' for n in shape)}{kind}>"
    for name, value in (("a", 7), ("b", 29)):
        scalar = f"{value}.0" if kind.startswith("f") else str(value)
        source = source.replace(f"%{name} = tensor.empty() : {ty}", f"%{name} = arith.constant dense<{scalar}> : {ty}")
    return source


def native_compare(compiler_python, destination, *, kind="f32", shape=(4,), nested=False):
    destination.mkdir(parents=True, exist_ok=True)
    source = fixture(kind, shape, nested=nested)
    generated = actual(source)
    (destination / "after.mlir").write_text(generated["lifted"])
    (destination / "reference.mlir").write_text(reference(source, kind, shape))
    ty = ValueType(kind, shape)
    count = math.prod(shape)
    dtype = {"f32": np.float32, "f64": np.float64, "i8": np.int8}[kind]

    def bits(value):
        return int.from_bytes(np.array(value, dtype=dtype).tobytes(), "little")

    inputs = []
    for flag in (0, 1):
        for value in (3, 11):
            inputs.append([[bits(value)] * count, [flag]])
    extended = [sample + [[bits(7)] * count, [bits(29)] * count] for sample in inputs]
    tools = Tools(compiler_python, Path("unused"), Path("unused"))
    observations = {}
    for label, samples, types in (
        ("after", extended, [ty, ValueType("i1"), ty, ty]),
        ("reference", inputs, [ty, ValueType("i1")]),
    ):
        observations[label] = execute_mlir(
            destination / f"{label}.mlir",
            samples,
            destination / label,
            tools,
            inputs=types,
            outputs=[ty, ty, ty],
            function="forward",
        )
    report = {
        "kind": kind,
        "shape": shape,
        "nested": nested,
        "samples": len(inputs),
        "inputs": inputs,
        "extended_inputs": extended,
        "results": observations["after"]["results"],
        "reference_results": observations["reference"]["results"],
        "all_output_bits_match": observations["after"]["results"] == observations["reference"]["results"],
        "source_is_explicit_runtime_binding_reference": True,
        "approved": False,
    }
    (destination / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["all_output_bits_match"], report
    return report
