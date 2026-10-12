"""Neutral source-only checks for the closed f32 bucketize count reduction."""

from __future__ import annotations

import json
from hashlib import sha256

import pytest

from merlin.common import mlir_query as mq
from merlin.frontends.bucketize_source import (
    STATIC_BUCKETIZE_SOURCE_BODY_SCHEMA,
    bucketize_pattern_fits_index,
    recognize_bucketize_source,
    recognize_static_bucketize_body,
    screen_bucketize_source,
    validate_static_bucketize_source_body,
)
from merlin.frontends.linalg_patterns import InvalidLinalgPattern


def _program(*, right: bool = False, input_shape: str = "2x3", boundary_count: int = 4) -> str:
    shape = input_shape
    dims = tuple(f"d{i}" for i in range(len(shape.split("x"))))
    loop = ", ".join((*dims, f"d{len(dims)}"))
    projected = ", ".join(dims)
    axis = f"d{len(dims)}"
    iterator_types = ", ".join((*("#linalg.iterator_type<parallel>" for _ in dims), "#linalg.iterator_type<reduction>"))
    pred = 12 if right else 11
    return f"""builtin.module {{
  func.func @forward(%input: tensor<{shape}xf32>, %boundaries: tensor<{boundary_count}xf32>)
      -> tensor<{shape}xi64> {{
    %zero = "arith.constant"() <{{value = 0 : i64}}> : () -> i64
    %one = "arith.constant"() <{{value = 1 : i64}}> : () -> i64
    %init = "tensor.splat"(%zero) : (i64) -> tensor<{shape}xi64>
    %out = "linalg.generic"(%input, %boundaries, %init)
      <{{indexing_maps = [
        affine_map<({loop}) -> ({projected})>,
        affine_map<({loop}) -> ({axis})>,
        affine_map<({loop}) -> ({projected})>],
        iterator_types = [{iterator_types}],
        operandSegmentSizes = array<i32: 2, 1>}}> ({{
      ^bb0(%x: f32, %b: f32, %acc: i64):
        %less = "arith.cmpf"(%b, %x)
          <{{predicate = {pred} : i64, fastmath = #arith.fastmath<none>}}> : (f32, f32) -> i1
        %increment = "arith.select"(%less, %one, %zero) : (i1, i64, i64) -> i64
        %count = "arith.addi"(%acc, %increment)
          <{{overflowFlags = #arith.overflow<none>}}> : (i64, i64) -> i64
        "linalg.yield"(%count) : (i64) -> ()
      }}) {{prov.source_node_ids = ["g:prepared:root:n1"], prov.aten = "aten.bucketize.Tensor"}}
        : (tensor<{shape}xf32>, tensor<{boundary_count}xf32>, tensor<{shape}xi64>) -> tensor<{shape}xi64>
    func.return %out : tensor<{shape}xi64>
  }}
}}"""


def _generic(text: str):
    return next(mq.walk(mq.parse(text), "linalg.generic"))


def test_host_bucketize_body_is_explicit_literal_and_conditionally_typed():
    declaration = {"schema": STATIC_BUCKETIZE_SOURCE_BODY_SCHEMA, "operation": "f32_literal_boundary_count_i64"}
    assert validate_static_bucketize_source_body(declaration) == declaration
    for wrong in ({**declaration, "extra": True}, {**declaration, "operation": "count"}, {"schema": "other"}):
        with pytest.raises(InvalidLinalgPattern):
            validate_static_bucketize_source_body(wrong)
    for right in (False, True):
        text = _program(right=right).replace(
            "    %zero =",
            '    %boundaries = "arith.constant"() <{value = dense<[0.0, 0.25, 0.25, 1.0]> : tensor<4xf32>}> : () -> tensor<4xf32>\n    %zero =',
        )
        # Replace only the boundary argument; the same SSA value is now literal.
        text = text.replace("%boundaries: tensor<4xf32>", "%unused: tensor<4xf32>")
        pattern = recognize_static_bucketize_body(_generic(text))
        assert pattern.operation == declaration["operation"]
        assert pattern.ordered_types == ("f32", "f32", "i64", "i64")
        assert pattern.right is right
        assert pattern.count_range == (0, 4)
        assert len(pattern.literal_sha256) == 64
        assert bucketize_pattern_fits_index(recognize_bucketize_source(_generic(text), right=right), 64)
        assert not bucketize_pattern_fits_index(recognize_bucketize_source(_generic(text), right=right), 3)
    with pytest.raises(InvalidLinalgPattern, match="boundary ordering"):
        recognize_static_bucketize_body(_generic(_program()))


def _trace(text: str, *, right: bool) -> dict:
    operations = tuple(mq.walk(mq.parse(text)))
    ordinal = next(i for i, op in enumerate(operations) if mq.op_name(op) == "linalg.generic")
    node_id = "g:prepared:root:n1"
    original_id = "g:original:root:n1"
    quantized_id = "g:quantized:root:n1"
    unresolved_ids = ["g:staged:root:n1", "g:quantization_input:root:n1"]
    return {
        "schema": "m2m.frontend_trace.v1",
        "status": "complete",
        "mlir": {
            "sha256": sha256(text.encode()).hexdigest(),
            "bytes": len(text.encode()),
            "operations": [
                {
                    "ordinal": i,
                    "operation": mq.op_name(op),
                    "source_node_ids": [node_id] if i == ordinal else [],
                    "operand_types": [str(value.type) for value in op.operands],
                    "result_types": [str(value.type) for value in op.results],
                }
                for i, op in enumerate(operations)
            ],
            "source_correspondence": [{"node_id": node_id, "status": "lowered", "mlir_ordinals": [ordinal]}],
        },
        "graphs": {
            "prepared": {
                "nodes": [
                    {
                        "id": node_id,
                        "target": "aten.bucketize.Tensor",
                        "kwargs": {"right": right},
                        "origin_node_ids": [original_id, *unresolved_ids, quantized_id],
                    }
                ]
            },
            "original": {"nodes": [{"id": original_id, "target": "aten.bucketize.Tensor", "kwargs": {"right": right}}]},
            "quantized": {
                "nodes": [
                    {
                        "id": quantized_id,
                        "target": "aten.bucketize.Tensor",
                        "kwargs": {"right": right},
                        "origin_node_ids": [original_id, *unresolved_ids],
                    }
                ]
            },
        },
        "transformations": [
            {
                "from_stage": "original",
                "to_stage": "quantized",
                "status": "complete",
                "relations": [{"kind": "introduced", "source_ids": [original_id], "destination_ids": [quantized_id]}],
            },
            {
                "from_stage": "quantized",
                "to_stage": "prepared",
                "status": "complete",
                "relations": [{"kind": "introduced", "source_ids": [quantized_id], "destination_ids": [node_id]}],
            },
        ],
    }


@pytest.mark.parametrize("right", [False, True])
def test_closed_count_is_typed_bounded_and_trace_bound(tmp_path, right):
    text = _program(right=right)
    model = tmp_path / "model.mlir"
    trace_path = tmp_path / "frontend-trace.json"
    model.write_text(text)
    trace_path.write_text(json.dumps(_trace(text, right=right)))
    ordinal = next(i for i, op in enumerate(mq.walk(mq.parse(text))) if mq.op_name(op) == "linalg.generic")
    proof = screen_bucketize_source(model, trace_path, (ordinal,))
    assert proof.raw_sha256 == sha256(text.encode()).hexdigest()
    assert len(proof.normalized_sha256) == 64
    assert proof.ordinals[0][1].input_shape == (2, 3)
    assert proof.ordinals[0][1].count_range == (0, 4)
    assert proof.ordinals[0][1].comparison == ("ule" if right else "ult")
    assert proof.ordinals[0][1].right is right
    binding = proof.trace_bindings[0][1]
    assert binding.original_node_id == "g:original:root:n1"
    assert binding.quantized_node_id == "g:quantized:root:n1"
    assert binding.unresolved_ancestry_ids == ("g:staged:root:n1", "g:quantization_input:root:n1")
    assert binding.ancestry_status == "original_quantized_prepared_flag_match_only"


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("predicate = 11", "predicate = 4"),  # ordered predicate loses NaN behavior
        ("predicate = 11", "predicate = 12"),  # trace right=False differs
        ("%b, %x", "%x, %b"),
        ("%less, %one, %zero", "%less, %zero, %one"),
        ("%acc, %increment", "%increment, %acc"),
        ("%count) : (i64)", "%acc) : (i64)"),
        ("value = 0 : i64", "value = 2 : i64"),
        ("value = 1 : i64", "value = 2 : i64"),
        ("#arith.fastmath<none>", "#arith.fastmath<nnan>"),
        ("#arith.overflow<none>", "#arith.overflow<nsw>"),
        ("#linalg.iterator_type<reduction>", "#linalg.iterator_type<parallel>"),
        ("-> (d2)", "-> (d1)"),
        ("-> (d0, d1)", "-> (d1, d0)"),
        ("tensor<4xf32>", "tensor<4xi64>"),
        ('%init = "tensor.splat"(%zero)', '%init = "tensor.splat"(%one)'),
    ],
)
def test_scalar_seed_map_and_type_near_misses_refuse(before, after):
    text = _program()
    assert before in text
    with pytest.raises(InvalidLinalgPattern):
        recognize_bucketize_source(_generic(text.replace(before, after)), right=False)


def test_trace_right_ancestry_and_source_bytes_must_match(tmp_path):
    text = _program(right=True)
    model = tmp_path / "model.mlir"
    trace_path = tmp_path / "frontend-trace.json"
    model.write_text(text)
    ordinal = next(i for i, op in enumerate(mq.walk(mq.parse(text))) if mq.op_name(op) == "linalg.generic")
    trace = _trace(text, right=True)
    for mutate in (
        lambda row: row["graphs"]["prepared"]["nodes"][0]["kwargs"].update(right=False),
        lambda row: row["graphs"]["prepared"]["nodes"][0]["kwargs"].update(right=1),
        lambda row: row["graphs"]["original"]["nodes"][0]["kwargs"].update(right=False),
        lambda row: row["graphs"]["quantized"]["nodes"][0]["kwargs"].update(right=False),
        lambda row: row["graphs"]["prepared"]["nodes"][0].update(origin_node_ids=[]),
        lambda row: row["graphs"]["prepared"]["nodes"][0]["origin_node_ids"].append("g:original:root:missing"),
        lambda row: row["graphs"]["prepared"]["nodes"][0]["origin_node_ids"].append("g:original:root:n1"),
        lambda row: row["graphs"]["prepared"]["nodes"][0]["origin_node_ids"].append([]),
        lambda row: row["graphs"]["prepared"]["nodes"][0]["origin_node_ids"].append("g:quantized:root:missing"),
        lambda row: row["graphs"]["quantized"]["nodes"][0]["origin_node_ids"].append("g:unwitnessed:root:n2"),
        lambda row: row["transformations"][0]["relations"].clear(),
        lambda row: row["mlir"]["source_correspondence"][0].update(mlir_ordinals=[]),
        lambda row: row["mlir"]["source_correspondence"][0].update(mlir_ordinals=[ordinal, ordinal]),
        lambda row: row["mlir"]["source_correspondence"][0].update(mlir_ordinals=[ordinal, True]),
        lambda row: row["mlir"]["source_correspondence"][0].update(mlir_ordinals=[ordinal, []]),
        lambda row: row["mlir"]["source_correspondence"][0].update(mlir_ordinals=[ordinal, 10000]),
        lambda row: row["mlir"]["source_correspondence"][0].update(mlir_ordinals=[ordinal, ordinal - 1]),
        lambda row: row["mlir"]["operations"][ordinal].update(source_node_ids=["unknown"]),
        lambda row: row["mlir"]["operations"][ordinal].update(operand_types=[]),
        lambda row: row["mlir"].update(sha256="0" * 64),
        lambda row: row.update(status="partial"),
    ):
        wrong = json.loads(json.dumps(trace))
        mutate(wrong)
        trace_path.write_text(json.dumps(wrong))
        with pytest.raises(InvalidLinalgPattern):
            screen_bucketize_source(model, trace_path, (ordinal,))
    trace_path.write_text(json.dumps(trace))
    for ordinals in ((ordinal, ordinal), (True,), (10000,), (None,)):
        with pytest.raises(InvalidLinalgPattern):
            screen_bucketize_source(model, trace_path, ordinals)
    model.write_text(text + "\n")
    with pytest.raises(InvalidLinalgPattern, match="source bytes"):
        screen_bucketize_source(model, trace_path, (ordinal,))


def test_unrostered_intervening_id_is_retained_but_not_proved(tmp_path):
    text = _program(right=True)
    model = tmp_path / "model.mlir"
    trace_path = tmp_path / "frontend-trace.json"
    model.write_text(text)
    trace = _trace(text, right=True)
    old = "g:staged:root:n1"
    replacement = "g:unrostered:root:n7"
    for stage in ("prepared", "quantized"):
        origins = trace["graphs"][stage]["nodes"][0]["origin_node_ids"]
        origins[origins.index(old)] = replacement
    trace_path.write_text(json.dumps(trace))
    ordinal = next(i for i, op in enumerate(mq.walk(mq.parse(text))) if mq.op_name(op) == "linalg.generic")
    binding = screen_bucketize_source(model, trace_path, (ordinal,)).trace_bindings[0][1]
    assert binding.unresolved_ancestry_ids == (replacement, "g:quantization_input:root:n1")
    assert binding.ancestry_status == "original_quantized_prepared_flag_match_only"


def test_preserved_bucketize_identity_relations_keep_exact_ancestry_guards(tmp_path):
    text = _program(right=True)
    model = tmp_path / 'model.mlir'
    trace_path = tmp_path / 'frontend-trace.json'
    model.write_text(text)
    ordinal = next(i for i, op in enumerate(mq.walk(mq.parse(text))) if mq.op_name(op) == 'linalg.generic')
    trace = _trace(text, right=True)
    for stage in trace['transformations']:
        stage['relations'][0]['kind'] = 'identity'
    trace_path.write_text(json.dumps(trace))
    proof = screen_bucketize_source(model, trace_path, (ordinal,))
    assert proof.trace_bindings[0][1].original_node_id == 'g:original:root:n1'
    trace['transformations'][1]['relations'][0]['destination_ids'] = ['g:prepared:root:missing']
    trace_path.write_text(json.dumps(trace))
    with pytest.raises(InvalidLinalgPattern, match='exact selected transformation relation'):
        screen_bucketize_source(model, trace_path, (ordinal,))


def test_bf16_bucketize_requires_one_lossless_f32_extension():
    text = _program(right=True).replace('tensor<2x3xf32>', 'tensor<2x3xbf16>').replace('%x: f32', '%x: bf16')
    text = text.replace('        %less =', '        %wide = "arith.extf"(%x) : (bf16) -> f32\n        %less =').replace('(%b, %x)', '(%b, %wide)')
    pattern = recognize_bucketize_source(_generic(text), right=True)
    assert pattern.input_dtype == 'bf16'
    with pytest.raises(InvalidLinalgPattern):
        recognize_bucketize_source(_generic(text.replace('%wide =', '%extra = "arith.extf"(%x) : (bf16) -> f32\n        %wide =')), right=True)
