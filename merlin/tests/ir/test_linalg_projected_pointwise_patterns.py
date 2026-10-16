"""Neutral, opt-in static singleton-projection pointwise source checks."""

from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256

import pytest

from merlin.common import mlir_query as mq
from merlin.frontends.linalg_patterns import (
    InvalidLinalgPattern,
    recognize_static_pointwise,
    recognize_static_projected_pointwise,
    screen_static_projected_pointwise_source,
    validate_serialized_static_projected_pointwise_pattern,
    validate_static_projected_pointwise_source_body,
)


def _program(
    *,
    shape: str = "2x3",
    inputs: tuple[tuple[str, str, str], ...] = (("2x1", "i64", "(d0, 0)"), ("2x3", "i64", "(d0, d1)")),
    result: str = "i64",
    operation: str = "arith.addi",
    properties: str = "",
    scalar_args: str = "%a0, %a1",
    result_map: str | None = None,
    iterators: str | None = None,
    extra_body: str = "",
) -> str:
    rank = len(shape.split("x"))
    dims = ", ".join(f"d{i}" for i in range(rank))
    output = f"tensor<{shape}x{result}>"
    inputs_text = [f"tensor<{extent + 'x' if extent else ''}{dtype}>" for extent, dtype, _ in inputs]
    signatures = [f"%x{i}: {typ}" for i, typ in enumerate(inputs_text)]
    signatures.append(f"%init: {output}")
    maps = [f"affine_map<({dims}) -> {mapping}>" for _, _, mapping in inputs]
    maps.append(f"affine_map<({dims}) -> {result_map or f'({dims})'}>")
    body_args = [f"%a{i}: {dtype}" for i, (_, dtype, _) in enumerate(inputs)]
    body_args.append(f"%old: {result}")
    operand_types = ", ".join(dtype for _, dtype, _ in inputs)
    iterator_values = iterators or ", ".join("#linalg.iterator_type<parallel>" for _ in range(rank))
    return f'''builtin.module {{
  func.func @neutral({", ".join(signatures)}) -> {output} {{
    %out = "linalg.generic"({", ".join(f"%x{i}" for i in range(len(inputs)))}, %init)
      <{{indexing_maps = [{", ".join(maps)}], iterator_types = [{iterator_values}],
        operandSegmentSizes = array<i32: {len(inputs)}, 1>}}> ({{
      ^bb0({", ".join(body_args)}):
        %v = "{operation}"({scalar_args}){properties} : ({operand_types}) -> {result}
        {extra_body}
        "linalg.yield"(%v) : ({result}) -> ()
      }}) : ({", ".join((*inputs_text, output))}) -> {output}
    func.return %out : {output}
  }}
}}'''


def _generic(text: str):
    return next(mq.walk(mq.parse(text), "linalg.generic"))


@pytest.mark.parametrize(
    ("text", "operation", "predicate", "shapes"),
    [
        (_program(), "arith.addi", None, ((2, 1), (2, 3))),
        (
            _program(
                inputs=(("2x3", "i1", "(d0, d1)"), ("", "f32", "()"), ("", "f32", "()")),
                result="f32", operation="arith.select", scalar_args="%a0, %a1, %a2",
            ),
            "arith.select", None, ((2, 3), (), ()),
        ),
        (
            _program(
                shape="2x3x4",
                inputs=(("2x1x4", "f32", "(d0, 0, d2)"), ("2x3x1", "f32", "(d0, d1, 0)")),
                result="i1",
                operation="arith.cmpf",
                properties=" <{predicate = 5 : i64, fastmath = #arith.fastmath<none>}>",
            ),
            "arith.cmpf",
            "ole",
            ((2, 1, 4), (2, 3, 1)),
        ),
        (
            _program(inputs=(("", "i1", "()"), ("2x3", "i1", "(d0, d1)")), result="i1", operation="arith.andi"),
            "arith.andi",
            None,
            ((), (2, 3)),
        ),
    ],
)
def test_closed_projected_scalar_forms_keep_old_identity_checker_strict(text, operation, predicate, shapes):
    op = _generic(text)
    with pytest.raises(InvalidLinalgPattern, match="shapes differ"):
        recognize_static_pointwise(op)
    pattern = recognize_static_projected_pointwise(op)
    assert pattern.operation == operation
    assert pattern.predicate == predicate
    assert pattern.input_shapes == shapes
    assert pattern.shape in ((2, 3), (2, 3, 4))
    validate_serialized_static_projected_pointwise_pattern(asdict(pattern))
    validate_serialized_static_projected_pointwise_pattern(json.loads(json.dumps(asdict(pattern))))


def test_projected_declaration_and_serialized_maps_are_closed():
    declaration = {"schema": "merlin.static_projected_pointwise_body.v1", "operation": "arith.addi"}
    assert validate_static_projected_pointwise_source_body(declaration) == declaration
    for changed in (
        {**declaration, "predicate": "sle"},
        {**declaration, "allow_dynamic": True},
        {**declaration, "schema": "merlin.static_pointwise_source_body.v1"},
    ):
        with pytest.raises(InvalidLinalgPattern):
            validate_static_projected_pointwise_source_body(changed)
    pattern = asdict(recognize_static_projected_pointwise(_generic(_program())))
    for changed in (
        {"input_maps": ["(d0, d1) -> (d1, 0)", pattern["input_maps"][1]]},
        {"input_maps": ["(d0, d1) -> (d0, 1)", pattern["input_maps"][1]]},
        {"input_maps": ["(d0, d1) -> (d0 + 1, 0)", pattern["input_maps"][1]]},
        {"input_shapes": [[2, 2], [2, 3]]},
        {"shape": [0, 3]},
        {"ordered_types": ["i1", "i64", "i64", "i64"]},
        {"predicate": "sle"},
    ):
        with pytest.raises(InvalidLinalgPattern):
            validate_serialized_static_projected_pointwise_pattern({**pattern, **changed})


@pytest.mark.parametrize(
    "changed",
    [
        lambda s: s.replace("(d0, 0)", "(d1, 0)"),
        lambda s: s.replace("(d0, 0)", "(d0, 1)"),
        lambda s: s.replace("(d0, 0)", "(d0 + 1, 0)"),
        lambda s: s.replace("affine_map<(d0, d1) -> (d0, d1)>]", "affine_map<(d0, d1) -> (d1, d0)>]"),
        lambda s: s.replace("#linalg.iterator_type<parallel>", "#linalg.iterator_type<reduction>", 1),
        lambda s: s.replace("%a0, %a1", "%a0, %old"),
        lambda s: s.replace('"arith.addi"', '"arith.addf"'),
        lambda s: s.replace(
            '"arith.addi"(%a0, %a1)', '"arith.addi"(%a0, %a1) <{overflowFlags = #arith.overflow<nsw>}>'
        ),
        lambda s: s.replace(
            '"linalg.yield"(%v)', '%other = "arith.addi"(%a0, %a1) : (i64, i64) -> i64\n        "linalg.yield"(%v)'
        ),
    ],
)
def test_projected_near_misses_refuse(changed):
    with pytest.raises((InvalidLinalgPattern, ValueError)):
        recognize_static_projected_pointwise(_generic(changed(_program())))


def test_projected_source_wrapper_binds_complete_normalized_module(tmp_path):
    path = tmp_path / "neutral.mlir"
    path.write_text(_program())
    ordinal = next(i for i, op in enumerate(mq.walk(mq.parse(_program()))) if op.name == "linalg.generic")
    proof = screen_static_projected_pointwise_source(path, (ordinal,))
    assert proof.raw_sha256 == sha256(path.read_bytes()).hexdigest()
    assert proof.ordinals[0][1] == recognize_static_projected_pointwise(_generic(_program()))
    path.write_text(_program(scalar_args="%a0, %old"))
    with pytest.raises(InvalidLinalgPattern):
        screen_static_projected_pointwise_source(path, (ordinal,))
