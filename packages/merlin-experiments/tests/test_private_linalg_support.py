"""Versioned source-body evidence is exact before and after linking."""

# ruff: noqa: E501 - emitted MLIR lines retain the producer's operation form.

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256

import pytest
from merlin_experiments.phase1.feedback import private_linalg_support as support

from merlin.common import mlir_query as mq
from merlin.compile.model_execution_inputs import strict_tree_sha256
from merlin.frontends.capture_normalization import normalize_capture_mlir
from merlin.frontends.linalg_composite_math import recognize_static_composite_math_body
from merlin.targetgen.host_capabilities import admit_host_operation
from merlin.targetgen.host_linkage_contract import SCHEMA as LINKAGE_SCHEMA
from merlin.targetgen.host_linkage_contract import SUPPLIER


def _source(
    operation: str,
    *,
    false_constant: bool = False,
    scalar_first: bool = False,
    literal_value: str | None = None,
):
    """Parse a harmless typed source fixture, independent of checkout-only tests."""
    forms = {
        "arith.addf": (
            "tensor<2x3xf32>",
            "tensor<2x3xf32>",
            "f32",
            "f32",
            '%v = "arith.addf"(%a, %b) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32',
        ),
        "arith.addi": (
            "tensor<2x1xi64>",
            "tensor<2x3xi64>",
            "i64",
            "i64",
            '%v = "arith.addi"(%a, %b) : (i64, i64) -> i64',
        ),
        "arith.cmpf": (
            "tensor<2x1xf32>",
            "tensor<2x3xf32>",
            "f32",
            "i1",
            '%v = "arith.cmpf"(%a, %b) <{predicate = 5 : i64, fastmath = #arith.fastmath<none>}> : (f32, f32) -> i1',
        ),
        "math.sin": (
            "tensor<2x3xf32>",
            None,
            "f32",
            "f32",
            '%v = "math.sin"(%a) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32',
        ),
        "math.cos": (
            "tensor<2x3xf32>",
            None,
            "f32",
            "f32",
            '%v = "math.cos"(%a) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32',
        ),
        "math.exp": (
            "tensor<2x3xf32>",
            None,
            "f32",
            "f32",
            '%v = "math.exp"(%a) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32',
        ),
        "math.roundeven": (
            "tensor<2x3xf32>",
            None,
            "f32",
            "f32",
            '%v = "math.roundeven"(%a) <{fastmath = #arith.fastmath<none>}> : (f32) -> f32',
        ),
        "clamp_upper_literal": (
            "tensor<2x3xf32>",
            None,
            "f32",
            "f32",
            '%bound = "arith.constant"() <{value = 2.250000e+00 : f32}> : () -> f32\n'
            '        %v = "arith.minimumf"(%a, %bound) '
            "<{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32",
        ),
        "literal_base_pow_f32": (
            "tensor<2x3xf32>",
            None,
            "f32",
            "f32",
            '%base = "arith.constant"() <{value = 3.250000e+00 : f32}> : () -> f32\n'
            '        %v = "math.powf"(%base, %a) '
            "<{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32",
        ),
        "i1_not": (
            "tensor<2x3xi1>",
            None,
            "i1",
            "i1",
            '%one = "arith.constant"() <{value = true}> : () -> i1\n'
            '        %v = "arith.xori"(%a, %one) : (i1, i1) -> i1',
        ),
        "f32_nonzero_to_i1": (
            "tensor<2x3xf32>",
            None,
            "f32",
            "i1",
            '%zero = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32\n'
            '        %v = "arith.cmpf"(%a, %zero) '
            "<{predicate = 13 : i64, fastmath = #arith.fastmath<none>}> : (f32, f32) -> i1",
        ),
        "i1_mul_singleton_projected": (
            "tensor<2x1xi1>",
            "tensor<2x3xi1>",
            "i1",
            "i1",
            '%v = "arith.muli"(%a, %b) <{overflowFlags = #arith.overflow<none>}> : (i1, i1) -> i1',
        ),
    }
    first, second, input_type, output_type, body = forms[operation]
    if scalar_first:
        assert operation in {"i1_mul_singleton_projected", "arith.addi"}
        first = f"tensor<{input_type}>"
    if false_constant:
        body = body.replace("value = true", "value = false")
    if literal_value is not None:
        body = body.replace("3.250000e+00" if operation == "literal_base_pow_f32" else "2.250000e+00", literal_value)
    output = f"tensor<2x3x{output_type}>"
    inputs = f"%x: {first}, %y: {second}" if second else f"%x: {first}"
    operands = "%x, %y, %init" if second else "%x, %init"
    args = (
        f"%a: {input_type}, %b: {input_type}, %old: {output_type}"
        if second
        else f"%a: {input_type}, %old: {output_type}"
    )
    types = f"{first}, {second}, {output}" if second else f"{first}, {output}"
    first_map = (
        "affine_map<(d0, d1) -> ()>"
        if scalar_first
        else "affine_map<(d0, d1) -> (d0, 0)>"
        if operation in {"i1_mul_singleton_projected", "arith.addi", "arith.cmpf"}
        else "affine_map<(d0, d1) -> (d0, d1)>"
    )
    identity = "affine_map<(d0, d1) -> (d0, d1)>"
    maps = f"{first_map}, {identity}, {identity}" if second else f"{identity}, {identity}"
    count = 2 if second else 1
    text = f"""builtin.module {{
      func.func @f({inputs}, %init: {output}) -> {output} {{
        %out = "linalg.generic"({operands}) <{{indexing_maps = [{maps}],
          iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>],
          operandSegmentSizes = array<i32: {count}, 1>}}> ({{
          ^bb0({args}):
            {body}
            "linalg.yield"(%v) : ({output_type}) -> ()
        }}) : ({types}) -> {output}
        func.return %out : {output}
      }}
    }}"""
    return text


def _op(operation: str, **kwargs):
    return next(mq.walk(mq.parse(_source(operation, **kwargs)), "linalg.generic"))


def _case(schema: str, operation: str, *, scalar_first: bool = False):
    types = {
        "arith.addf": ["f32", "f32", "f32"],
        "arith.addi": ["i64", "i64", "i64"],
        "arith.cmpf": ["f32", "f32", "i1"],
        "math.sin": ["f32", "f32"],
        "math.cos": ["f32", "f32"],
        "math.exp": ["f32", "f32"],
        "math.roundeven": ["f32", "f32"],
        "clamp_upper_literal": ["f32", "f32"],
        "literal_base_pow_f32": ["f32", "f32"],
        "i1_not": ["i1", "i1"],
        "f32_nonzero_to_i1": ["f32", "i1"],
        "i1_mul_singleton_projected": ["i1", "i1", "i1"],
    }[operation]
    result = (
        "i64"
        if operation == "arith.addi"
        else "f32"
        if operation
        in {
            "arith.addf",
            "math.sin",
            "math.cos",
            "math.exp",
            "math.roundeven",
            "clamp_upper_literal",
            "literal_base_pow_f32",
        }
        else "i1"
    )
    parsed = (_op(operation, scalar_first=scalar_first), _op(operation, scalar_first=scalar_first))
    selected = {
        "host": {
            "package_sha256": "a" * 64,
            "capability_spec_sha256": "b" * 64,
            "dtype_strategy": "int8_w8a8",
            "capability_spec": {
                "schema": "merlin.host_capabilities.v1",
                "status": "reviewed",
                "compiler": {"package_sha256": "a" * 64, "dtype_strategy": "int8_w8a8"},
                "operations": [
                    {
                        "id": "neutral_source_body",
                        "ops": ["linalg.generic"],
                        "placement": "host",
                        "signature": {
                            "ordered_operand_dtypes": types,
                            "ordered_result_dtypes": [result],
                            "ranks": [2],
                        },
                        "source_body": {
                            "schema": schema,
                            "operation": operation,
                            **({"predicate": "ole"} if operation == "arith.cmpf" else {}),
                        },
                        **(
                            {
                                "linkage_contract": {
                                    "schema": LINKAGE_SCHEMA,
                                    "supplier": SUPPLIER,
                                    "archive_sha256": "c" * 64,
                                    "symbols": ["powf"],
                                }
                            }
                            if operation == "literal_base_pow_f32"
                            else {}
                        ),
                    }
                ],
                "evidence": {"scope": "synthetic declaration only"},
            },
        }
    }
    observed = {
        "family": "elementwise_map",
        "ordered_operand_dtypes": types,
        "ordered_result_dtypes": [result],
        "rank": 2,
    }
    row = {"mlir_operation": "linalg.generic", "count": len(parsed), "ordinals": list(range(len(parsed)))}
    joined = {i: row for i in range(len(parsed))}
    admission = admit_host_operation(selected, row, observed, source_operations=parsed)
    assert admission["status"] == "admitted"
    digest = sha256(repr((schema, operation)).encode()).hexdigest()
    return parsed, row, joined, digest, admission


def _linked(parsed, row, joined, digest, admission):
    selected, actual = _index_lowering(32)
    proof = support.begin(digest, digest, len(parsed), selected)
    support.record(proof, row, admission, parsed, joined)
    source = {
        "source_sha256": digest,
        "normalized_source_sha256": digest,
        "n_source_operations": len(parsed),
        "selected_index_observation": selected,
        "linalg_host_support": proof,
    }
    entry = {
        "source_sha256": digest,
        "capture_tree_sha256": "c" * 64,
        "elf_sha256": "e" * 64,
        "index_lowering": actual,
    }
    assert not support.linked_complete(source, entry, "d" * 64)
    support.link(
        source,
        actual,
        {"candidate_tree_sha256": "d" * 64, "capture_tree_sha256": "c" * 64, "elf_sha256": "e" * 64},
    )
    assert support.linked_complete(source, entry, "d" * 64)
    return source, entry


def _index_lowering(bits: int):
    selected = {
        "schema": "merlin.selected-index-lowering.v1",
        "compiler_requested": "synthetic-clang",
        "compiler_resolved": "/synthetic/clang",
        "compiler_sha256": "a" * 64,
        "cross_flags": ["--target=synthetic"],
        "data_layout": f"e-p:{bits}:{bits}",
        "index_bits": bits,
    }
    passes = (
        "convert-index-to-llvm",
        "convert-arith-to-llvm",
        "finalize-memref-to-llvm",
        "convert-func-to-llvm",
        "convert-cf-to-llvm",
    )
    return selected, {**selected, "effective_pipeline": ",".join(f"{name}{{index-bitwidth={bits}}}" for name in passes)}


@pytest.mark.parametrize(
    ("schema", "operation"),
    [
        ("merlin.static_pointwise_source_body.v1", "arith.addf"),
        ("merlin.static_f32_math_source_body.v1", "math.sin"),
        ("merlin.static_f32_math_source_body.v1", "math.cos"),
        ("merlin.static_f32_math_source_body.v1", "math.exp"),
        ("merlin.static_f32_math_source_body.v1", "math.roundeven"),
        ("merlin.static_boolean_body.v1", "i1_not"),
        ("merlin.static_boolean_body.v1", "f32_nonzero_to_i1"),
        ("merlin.static_boolean_body.v1", "i1_mul_singleton_projected"),
    ],
)
def test_every_occurrence_recomputed_and_linked(schema, operation):
    parsed, row, joined, digest, admission = _case(schema, operation)
    source, entry = _linked(parsed, row, joined, digest, admission)
    occurrences = source["linalg_host_support"]["occurrences"]
    assert [item["ordinal"] for item in occurrences] == row["ordinals"]
    assert {item["schema"] for item in occurrences} == {schema}
    if operation in {"math.exp", "math.roundeven"}:
        assert all(item["linkage_requirement"] is None for item in occurrences)
    assert support.linked_complete(json.loads(json.dumps(source)), json.loads(json.dumps(entry)), "d" * 64)
    assert not support.linked_complete(source, entry, "f" * 64)
    source["linalg_host_support"]["occurrences"] = occurrences[:-1]
    assert not support.linked_complete(source, entry, "d" * 64)


def test_pointwise_empty_shapes_require_a_typed_sequence():
    source, entry = _linked(*_case("merlin.static_pointwise_source_body.v1", "arith.addf"))
    original = deepcopy(source["linalg_host_support"]["occurrences"])
    for malformed in (None, {}, "", False):
        source["linalg_host_support"]["occurrences"] = [{**original[0], "input_shapes": malformed}, original[1]]
        assert not support.linked_complete(source, entry, "d" * 64)


def test_second_mutated_occurrence_and_inventory_owner_refuse():
    parsed, row, joined, digest, admission = _case("merlin.static_boolean_body.v1", "i1_not")
    bad = (parsed[0], _op("i1_not", false_constant=True))
    with pytest.raises(ValueError, match="parsed source"):
        support.record(support.begin(digest, digest, len(parsed)), row, admission, bad, joined)
    with pytest.raises(ValueError, match="parsed source"):
        support.record(support.begin(digest, digest, len(parsed)), row, admission, parsed, {**joined, 1: dict(row)})
    with pytest.raises(ValueError, match="parsed source"):
        support.record(
            support.begin(digest, digest, len(parsed)), {**row, "ordinals": [0, 0]}, admission, parsed, joined
        )


def test_schema_operation_and_linked_identity_cannot_be_relabelled():
    case = _case("merlin.static_boolean_body.v1", "i1_not")
    source, entry = _linked(*case)
    original = deepcopy(source["linalg_host_support"]["occurrences"])
    for change in (
        {"schema": "merlin.static_pointwise_source_body.v1"},
        {"operation": "arith.xori"},
        {"operation": "i1_mul_singleton_projected"},
        {"predicate": "eq"},
        {"ordered_types": ["i1", "i64", "i64"]},
        {"input_maps": []},
        {"input_maps": ["nonsense"]},
        {"input_maps": ["(d0, d1) -> (d1, d0)"]},
        {"shape": [0, 3]},
        {"input_shapes": [[0, 3]]},
        {"capability_spec_sha256": "wrong"},
    ):
        source["linalg_host_support"]["occurrences"] = [{**original[0], **change}, original[1]]
        assert not support.linked_complete(source, entry, "d" * 64), change
    source["linalg_host_support"]["occurrences"] = original
    entry["elf_sha256"] = "f" * 64
    assert not support.linked_complete(source, entry, "d" * 64)


def test_serialized_singleton_projection_cannot_change_after_source_proof():
    source, entry = _linked(*_case("merlin.static_boolean_body.v1", "i1_mul_singleton_projected"))
    original = deepcopy(source["linalg_host_support"]["occurrences"])
    for change in (
        {"input_maps": ["(d0, d1) -> (d1, 0)", original[0]["input_maps"][1]]},
        {"input_maps": ["(d0, d1) -> (d0, 1)", original[0]["input_maps"][1]]},
        {"input_maps": ["(d0, d1) -> (d0 + 1, 0)", original[0]["input_maps"][1]]},
        {"input_shapes": [[2, 0], [2, 3]]},
        {"input_shapes": [[3, 1], [2, 3]]},
        {"shape": [0, 3]},
    ):
        source["linalg_host_support"]["occurrences"] = [{**original[0], **change}, original[1]]
        assert not support.linked_complete(source, entry, "d" * 64), change


def test_rank_zero_scalar_tensor_projection_is_source_bound_and_linked():
    source, entry = _linked(*_case("merlin.static_boolean_body.v1", "i1_mul_singleton_projected", scalar_first=True))
    occurrence = source["linalg_host_support"]["occurrences"][0]
    assert occurrence["input_shapes"][0] == ()
    assert occurrence["input_maps"][0] == "(d0, d1) -> ()"
    assert support.linked_complete(json.loads(json.dumps(source)), entry, "d" * 64)
    source["linalg_host_support"]["occurrences"][0]["input_maps"] = ["(d0, d1) -> (d0)", occurrence["input_maps"][1]]
    assert not support.linked_complete(source, entry, "d" * 64)


@pytest.mark.parametrize("operation", ["arith.addi", "arith.cmpf"])
def test_projected_pointwise_source_body_requires_exact_map_roster_and_link(operation):
    schema = "merlin.static_projected_pointwise_body.v1"
    source, entry = _linked(*_case(schema, operation))
    occurrence = source["linalg_host_support"]["occurrences"][0]
    assert occurrence["input_shapes"] == ((2, 1), (2, 3))
    assert occurrence["input_maps"][0] == "(d0, d1) -> (d0, 0)"
    assert support.linked_complete(json.loads(json.dumps(source)), entry, "d" * 64)
    original = deepcopy(source["linalg_host_support"]["occurrences"])
    for change in (
        {"input_shapes": [[2, 2], [2, 3]]},
        {"input_maps": ["(d0, d1) -> (d1, 0)", original[0]["input_maps"][1]]},
        {"input_maps": []},
        {"schema": "merlin.static_pointwise_source_body.v1"},
        {"ordered_types": ["i1", *original[0]["ordered_types"][1:]]},
    ):
        source["linalg_host_support"]["occurrences"] = [{**original[0], **change}, original[1]]
        assert not support.linked_complete(source, entry, "d" * 64), change
    source["linalg_host_support"]["occurrences"] = original
    assert not support.linked_complete(source, {**entry, "elf_sha256": "f" * 64}, "d" * 64)


def test_projected_rank_zero_scalar_input_and_selected_byte_bounds():
    from merlin_experiments.phase1.feedback.private_program_checks import ProgramCheckFailure

    source, entry = _linked(*_case("merlin.static_projected_pointwise_body.v1", "arith.addi", scalar_first=True))
    occurrence = source["linalg_host_support"]["occurrences"][0]
    assert occurrence["input_shapes"][0] == ()
    assert occurrence["input_maps"][0] == "(d0, d1) -> ()"
    selected, actual = _index_lowering(6)
    proof = support.begin("a" * 64, "a" * 64, 2, selected)
    parsed, row, joined, _, admission = _case("merlin.static_projected_pointwise_body.v1", "arith.addi")
    support.record(proof, row, admission, parsed, joined)
    with pytest.raises(ProgramCheckFailure, match="signed index address span") as refusal:
        support.link(
            {"selected_index_observation": selected, "linalg_host_support": proof},
            actual,
            {
                "candidate_tree_sha256": "d" * 64,
                "capture_tree_sha256": "c" * 64,
                "elf_sha256": "e" * 64,
            },
        )
    assert refusal.value.check == "linalg_span"


@pytest.mark.parametrize(
    "shape,index_bits,expected",
    [
        ([1, 32, 32, 64], 64, True),
        ([7], 6, True),
        ([8], 6, False),
        ([256], 11, False),
        ([0], 64, False),
        ([-1], 64, False),
        ([True], 64, False),
    ],
)
def test_i32_to_float_source_obeys_actual_four_byte_signed_index_span(shape, index_bits, expected):
    occurrence = {
        "schema": support.MIXED_SCHEMA,
        "shape": shape,
        "ordered_types": ["i32", "f32", "f32"],
        "input_shapes": [shape],
    }
    assert support._all_shapes_fit([occurrence], index_bits) is expected
    occurrence["ordered_types"][0] = "i24"
    assert not support._all_shapes_fit([occurrence], index_bits)


def test_math_witness_refuses_changed_body_roster_schema_and_linked_identity():
    parsed, row, joined, digest, admission = _case("merlin.static_f32_math_source_body.v1", "math.sin")
    with pytest.raises(ValueError, match="exact source occurrence roster"):
        support.record(
            support.begin(digest, digest, len(parsed)), row, {**admission, "reviewed": False}, parsed, joined
        )
    with pytest.raises(ValueError, match="parsed source"):
        support.record(support.begin(digest, digest, len(parsed)), row, admission, (parsed[0], _op("math.cos")), joined)
    with pytest.raises(ValueError, match="parsed source"):
        support.record(support.begin(digest, digest, len(parsed)), row, admission, parsed, {**joined, 1: dict(row)})
    source, entry = _linked(parsed, row, joined, digest, admission)
    original = deepcopy(source["linalg_host_support"]["occurrences"])
    for change in (
        {"schema": "merlin.static_pointwise_source_body.v1"},
        {"operation": "math.powf"},
        {"predicate": "oeq"},
        {"ordered_types": ["f64", "f64", "f64"]},
        {"input_maps": ["(d0, d1) -> (d0, d1)"]},
        {"input_shapes": [[2, 3]]},
        {"capability_spec_sha256": "not-a-digest"},
    ):
        source["linalg_host_support"]["occurrences"] = [{**original[0], **change}, original[1]]
        assert not support.linked_complete(source, entry, "d" * 64), change
    source["linalg_host_support"]["occurrences"] = original
    assert not support.linked_complete({**source, "source_sha256": "f" * 64}, entry, "d" * 64)
    assert not support.linked_complete({**source, "normalized_source_sha256": "f" * 64}, entry, "d" * 64)
    assert not support.linked_complete({**source, "n_source_operations": 3}, entry, "d" * 64)
    assert not support.linked_complete(source, {**entry, "capture_tree_sha256": "f" * 64}, "d" * 64)
    assert not support.linked_complete(source, {**entry, "elf_sha256": "f" * 64}, "d" * 64)
    assert not support.linked_complete(source, entry, "f" * 64)


@pytest.mark.parametrize("operation", ("clamp_upper_literal", "literal_base_pow_f32"))
def test_composite_math_witness_requires_selected_capture_and_reparses_after_link(tmp_path, monkeypatch, operation):
    schema = "merlin.static_composite_math_body.v1"
    parsed, row, joined, digest, admission = _case(schema, operation)
    assert admission["status"] == "admitted"  # Synthetic reviewed source placement only.
    with pytest.raises(ValueError, match="selected captured-stage tree"):
        _linked(parsed, row, joined, digest, admission)
    capture = tmp_path / "capture"
    capture.mkdir()
    model = capture / "model.mlir"
    model.write_text(_source(operation))
    receipt = capture / "capture_receipt.json"
    receipt.write_text('{"scope":"neutral fixture"}')
    raw = sha256(model.read_bytes()).hexdigest()
    normalized, _ = normalize_capture_mlir(model.read_text())
    normalized_digest = sha256(normalized.encode()).hexdigest()
    all_ops = tuple(mq.walk(mq.parse(normalized)))
    ordinal = next(i for i, op in enumerate(all_ops) if op.name == "linalg.generic")
    source_row = {"mlir_operation": "linalg.generic", "count": 1, "ordinals": [ordinal]}
    actual_pattern = recognize_static_composite_math_body(all_ops[ordinal])
    admission["source_body_proof"]["patterns"] = [asdict(actual_pattern)]
    selected, actual = _index_lowering(32)
    proof = support.begin(raw, normalized_digest, len(all_ops), selected)
    support.record(proof, source_row, admission, all_ops, {ordinal: source_row})
    tree = strict_tree_sha256(capture)["sha256"]
    source = {
        "source_sha256": raw,
        "normalized_source_sha256": normalized_digest,
        "capture_receipt_sha256": sha256(receipt.read_bytes()).hexdigest(),
        "n_source_operations": len(all_ops),
        "selected_index_observation": selected,
        "linalg_host_support": proof,
    }
    entry = {"source_sha256": raw, "capture_tree_sha256": tree, "elf_sha256": "e" * 64, "index_lowering": actual}
    linked_build = {"candidate_tree_sha256": "d" * 64, "capture_tree_sha256": tree, "elf_sha256": "e" * 64}
    with pytest.raises(ValueError, match="selected captured-stage tree"):
        support.link(source, actual, linked_build)
    support.link(source, actual, linked_build, capture_path=capture)
    occurrences = source["linalg_host_support"]["occurrences"]
    assert all(item["schema"] == schema and item["literal_bits"] for item in occurrences)
    assert all(item["lowering_intrinsic_obligations"] is not None for item in occurrences)
    assert support.linked_complete(json.loads(json.dumps(source)), entry, "d" * 64)
    original = deepcopy(occurrences)
    for mutation in (
        {"literal_bits": []},
        {"literal_bits": ["not-bits"]},
        {"lowering_intrinsic_obligations": ["math.sin"]},
        {"lowering_intrinsic_obligations": None},
        {"schema": "merlin.static_f32_math_source_body.v1"},
        {"linkage_requirement": None if operation == "literal_base_pow_f32" else {"symbols": ["powf"]}},
    ):
        source["linalg_host_support"]["occurrences"] = [{**original[0], **mutation}]
        assert not support.linked_complete(source, entry, "d" * 64), mutation
    source["linalg_host_support"]["occurrences"] = original
    source["linalg_host_support"]["occurrences"] = []
    source["linalg_host_support"]["count"] = 0
    assert not support.linked_complete(source, entry, "d" * 64)
    source["linalg_host_support"]["occurrences"] = original
    source["linalg_host_support"]["count"] = 1
    source["linalg_host_support"]["occurrences"][0]["literal_bits"] = ["0x3f800000"]
    assert not support.linked_complete(source, entry, "d" * 64)
    source["linalg_host_support"]["occurrences"] = original
    model.write_text(_source(operation, literal_value="2.750000e+00"))
    assert not support.linked_complete(source, entry, "d" * 64)
    model.write_text(_source(operation))
    other = tmp_path / "other"
    other.mkdir()
    (other / "model.mlir").write_text(_source(operation, literal_value="2.750000e+00"))
    (other / "capture_receipt.json").write_bytes(receipt.read_bytes())
    source["linalg_host_support"]["source_capture_path"] = str(other)
    assert not support.linked_complete(source, entry, "d" * 64)
    source["linalg_host_support"]["source_capture_path"] = str(capture)
    recognize = support.recognize_static_composite_math_body

    def changed_during_reparse(op):
        pattern = recognize(op)
        model.write_text(_source(operation, literal_value="2.750000e+00"))
        return pattern

    monkeypatch.setattr(support, "recognize_static_composite_math_body", changed_during_reparse)
    assert not support.linked_complete(source, entry, "d" * 64)


def test_composite_math_record_refuses_second_changed_source_occurrence():
    parsed, row, joined, digest, admission = _case("merlin.static_composite_math_body.v1", "clamp_upper_literal")
    changed = _op("clamp_upper_literal", literal_value="2.750000e+00")
    # The changed constant is still a valid form, but no longer matches the pinned per-ordinal proof.
    # A separate parsed instance with changed literal must not inherit the earlier occurrence proof.
    with pytest.raises(ValueError, match="parsed source"):
        support.record(support.begin(digest, digest, len(parsed)), row, admission, (parsed[0], changed), joined)


def test_missing_and_empty_witness_are_distinct_and_mandatory():
    digest = "a" * 64
    selected, actual = _index_lowering(32)
    source = {
        "source_sha256": digest,
        "normalized_source_sha256": digest,
        "n_source_operations": 2,
        "selected_index_observation": selected,
    }
    entry = {
        "source_sha256": digest,
        "capture_tree_sha256": "b" * 64,
        "elf_sha256": "c" * 64,
        "index_lowering": actual,
    }
    assert not support.linked_complete(source, entry, "d" * 64)
    source["linalg_host_support"] = support.begin(digest, digest, 2, selected)
    support.link(
        source,
        actual,
        {"candidate_tree_sha256": "d" * 64, "capture_tree_sha256": "b" * 64, "elf_sha256": "c" * 64},
    )
    assert support.linked_complete(source, entry, "d" * 64)
    source["linalg_host_support"]["count"] = 1
    assert not support.linked_complete(source, entry, "d" * 64)


def test_linked_source_refuses_missing_selected_index_width():
    source, entry = _linked(*_case("merlin.static_pointwise_source_body.v1", "arith.addf"))
    source["linalg_host_support"].pop("selected_index_observation")
    assert not support.linked_complete(source, entry, "d" * 64)


def test_linked_source_refuses_forged_or_missing_actual_width():
    source, entry = _linked(*_case("merlin.static_pointwise_source_body.v1", "arith.addf"))
    for changed in (
        None,
        {**entry["index_lowering"], "index_bits": 64},
        {**entry["index_lowering"], "effective_pipeline": ""},
    ):
        assert not support.linked_complete(source, {**entry, "index_lowering": changed}, "d" * 64)
    source["linalg_host_support"]["actual_index_observation"] = {**entry["index_lowering"], "index_bits": 64}
    assert not support.linked_complete(source, entry, "d" * 64)


def test_linked_source_refuses_unrecognized_index_observation_and_boolean_dtype():
    source, entry = _linked(*_case("merlin.static_boolean_body.v1", "f32_nonzero_to_i1"))
    original = deepcopy(source)
    source["linalg_host_support"]["occurrences"][0]["ordered_types"] = ["unknown", "i1", "i1"]
    assert not support.linked_complete(source, entry, "d" * 64)
    source = original
    source["selected_index_observation"]["schema"] = "unknown.index.schema"
    source["linalg_host_support"]["selected_index_observation"]["schema"] = "unknown.index.schema"
    assert not support.linked_complete(source, entry, "d" * 64)
    source = original
    selected, actual = _index_lowering(129)
    source["selected_index_observation"] = selected
    source["linalg_host_support"]["selected_index_observation"] = selected
    source["linalg_host_support"]["actual_index_observation"] = actual
    assert not support.linked_complete(source, {**entry, "index_lowering": actual}, "d" * 64)


def test_link_refuses_width_drift_before_mutating_pending_witness():
    parsed, row, joined, digest, admission = _case("merlin.static_pointwise_source_body.v1", "arith.addf")
    selected, actual = _index_lowering(32)
    witness = support.begin(digest, digest, len(parsed), selected)
    support.record(witness, row, admission, parsed, joined)
    source = {"selected_index_observation": selected, "linalg_host_support": witness}
    linked = {"candidate_tree_sha256": "d" * 64, "capture_tree_sha256": "c" * 64, "elf_sha256": "e" * 64}
    with pytest.raises(ValueError, match="selected index-lowering"):
        support.link(source, {**actual, "index_bits": 64}, linked)
    assert witness["status"] == support.PENDING
    assert "linked_build" not in witness


def test_linked_source_refuses_static_byte_span_beyond_signed_index():
    source, entry = _linked(*_case("merlin.static_pointwise_source_body.v1", "arith.addf"))
    occurrence = source["linalg_host_support"]["occurrences"][0]
    for shape in ([1 << 31, 3], [1 << 29]):
        occurrence["shape"] = shape
        assert not support.linked_complete(source, entry, "d" * 64)


def test_boolean_input_bytes_and_scalar_projection_are_bounded():
    source, entry = _linked(*_case("merlin.static_boolean_body.v1", "f32_nonzero_to_i1"))
    occurrence = source["linalg_host_support"]["occurrences"][0]
    occurrence["shape"] = [2, 1 << 29]
    occurrence["input_shapes"] = [[2, 1 << 29]]
    assert not support.linked_complete(source, entry, "d" * 64)
    scalar, scalar_entry = _linked(
        *_case("merlin.static_boolean_body.v1", "i1_mul_singleton_projected", scalar_first=True)
    )
    assert scalar["linalg_host_support"]["occurrences"][0]["input_shapes"][0] == ()
    assert support.linked_complete(scalar, scalar_entry, "d" * 64)


_IDENTITY = (
    "indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], "
    "iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], "
    "operandSegmentSizes = array<i32: 1, 1>"
)


def _int8_quantize_source(low: int = -127) -> str:
    fm = "<{fastmath = #arith.fastmath<none>}>"
    return f"""builtin.module {{
  func.func @forward(%x: tensor<2x3xf32>) -> tensor<2x3xi8> {{
    %e0 = "tensor.empty"() : () -> tensor<2x3xf32>
    %rounded = "linalg.generic"(%x, %e0) <{{{_IDENTITY}}}> ({{
      ^bb0(%a: f32, %b: f32):
        %r = "math.roundeven"(%a) {fm} : (f32) -> f32
        "linalg.yield"(%r) : (f32) -> ()
    }}) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    %e1 = "tensor.empty"() : () -> tensor<2x3xf32>
    %clamped = "linalg.generic"(%rounded, %e1) <{{{_IDENTITY}}}> ({{
      ^bb0(%input: f32, %acc: f32):
        %low = "arith.constant"() <{{value = {low} : i64}}> : () -> i64
        %lowf = "arith.sitofp"(%low) : (i64) -> f32
        %below = "arith.cmpf"(%input, %lowf) <{{predicate = 4 : i64, fastmath = #arith.fastmath<none>}}> : (f32, f32) -> i1
        %raised = "arith.select"(%below, %lowf, %input) : (i1, f32, f32) -> f32
        %high = "arith.constant"() <{{value = 127 : i64}}> : () -> i64
        %highf = "arith.sitofp"(%high) : (i64) -> f32
        %above = "arith.cmpf"(%raised, %highf) <{{predicate = 2 : i64, fastmath = #arith.fastmath<none>}}> : (f32, f32) -> i1
        %value = "arith.select"(%above, %highf, %raised) : (i1, f32, f32) -> f32
        "linalg.yield"(%value) : (f32) -> ()
    }}) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    %e2 = "tensor.empty"() : () -> tensor<2x3xi8>
    %q = "linalg.generic"(%clamped, %e2) <{{{_IDENTITY}}}> ({{
      ^bb0(%v: f32, %w: i8):
        %c = "arith.fptosi"(%v) : (f32) -> i8
        "linalg.yield"(%c) : (i8) -> ()
    }}) : (tensor<2x3xf32>, tensor<2x3xi8>) -> tensor<2x3xi8>
    func.return %q : tensor<2x3xi8>
  }}
}}"""


def test_int8_cast_producer_domain_is_reproved_on_reparse_and_linked(tmp_path):
    capture = tmp_path / "capture"
    capture.mkdir()
    model = capture / "model.mlir"
    model.write_text(_int8_quantize_source())
    receipt = capture / "capture_receipt.json"
    receipt.write_text('{"scope":"neutral fixture"}')
    raw = sha256(model.read_bytes()).hexdigest()
    normalized, _ = normalize_capture_mlir(model.read_text())
    normalized_digest = sha256(normalized.encode()).hexdigest()
    all_ops = tuple(mq.walk(mq.parse(normalized)))
    ordinal = [i for i, op in enumerate(all_ops) if op.name == "linalg.generic"][-1]
    row = {"mlir_operation": "linalg.generic", "count": 1, "ordinals": [ordinal]}
    selected_host = {
        "host": {
            "package_sha256": "a" * 64,
            "capability_spec_sha256": "b" * 64,
            "dtype_strategy": "int8_w8a8",
            "capability_spec": {
                "schema": "merlin.host_capabilities.v1",
                "status": "reviewed",
                "compiler": {"package_sha256": "a" * 64, "dtype_strategy": "int8_w8a8"},
                "operations": [
                    {
                        "id": "neutral_int8_cast",
                        "ops": ["linalg.generic"],
                        "placement": "host",
                        "signature": {"ordered_operand_dtypes": ["f32", "i8"], "ordered_result_dtypes": ["i8"]},
                        "source_body": {
                            "schema": "merlin.static_composite_math_body.v1",
                            "operation": "fptosi_i8_of_rounded_int8_grid_clamp",
                        },
                    }
                ],
                "evidence": {"scope": "neutral placement declaration only"},
            },
        }
    }
    signature = {
        "family": "elementwise_map",
        "ordered_operand_dtypes": ["f32", "i8"],
        "ordered_result_dtypes": ["i8"],
        "rank": 2,
    }
    admission = admit_host_operation(selected_host, row, signature, source_operations=(all_ops[ordinal],))
    assert admission["status"] == "admitted"
    selected, actual = _index_lowering(32)
    proof = support.begin(raw, normalized_digest, len(all_ops), selected)
    support.record(proof, row, admission, all_ops, {ordinal: row})
    tree = strict_tree_sha256(capture)["sha256"]
    source = {
        "source_sha256": raw,
        "normalized_source_sha256": normalized_digest,
        "capture_receipt_sha256": sha256(receipt.read_bytes()).hexdigest(),
        "n_source_operations": len(all_ops),
        "selected_index_observation": selected,
        "linalg_host_support": proof,
    }
    entry = {"source_sha256": raw, "capture_tree_sha256": tree, "elf_sha256": "e" * 64, "index_lowering": actual}
    linked_build = {"candidate_tree_sha256": "d" * 64, "capture_tree_sha256": tree, "elf_sha256": "e" * 64}
    support.link(source, actual, linked_build, capture_path=capture)
    occurrence = source["linalg_host_support"]["occurrences"][0]
    assert occurrence["operation"] == "fptosi_i8_of_rounded_int8_grid_clamp"
    assert support.linked_complete(json.loads(json.dumps(source)), entry, "d" * 64)
    # A producer outside the int8 grid no longer proves the cast on the post-link reparse.
    model.write_text(_int8_quantize_source(low=-129))
    assert not support.linked_complete(source, entry, "d" * 64)
