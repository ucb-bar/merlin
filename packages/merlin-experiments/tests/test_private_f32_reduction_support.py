"""Neutral private joins for reviewed, closed f32 host reductions and their dispatch."""

# ruff: noqa: E501 - emitted MLIR lines retain the producer's operation form.

from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256

import pytest
from merlin_experiments.phase1.feedback import private_f32_reduction_support as support
from merlin_experiments.phase1.feedback import private_host_source_dispatch as dispatch
from merlin_experiments.phase1.feedback import private_integer_reduction_support as integer
from merlin_experiments.phase1.feedback import private_linalg_support as linalg

from merlin.common import mlir_query as mq
from merlin.frontends.capture_normalization import normalize_capture_mlir
from merlin.targetgen.application_inventory import operation_structure
from merlin.targetgen.host_capabilities import admit_host_operation

_SUM = """builtin.module {
  func.func @forward(%arg0: tensor<2x3xf32>) -> tensor<2xf32> {
    %zero = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
    %init = "tensor.splat"(%zero) : (f32) -> tensor<2xf32>
    %sum = "linalg.reduce"(%arg0, %init) <{dimensions = array<i64: 1>}> ({
    ^bb1(%elem: f32, %acc: f32):
      %added = "arith.addf"(%elem, %acc) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
      "linalg.yield"(%added) : (f32) -> ()
    }) {prov.aten = "aten.sum.dim_IntList"} : (tensor<2x3xf32>, tensor<2xf32>) -> tensor<2xf32>
    func.return %sum : tensor<2xf32>
  }
}"""
_INTEGER_SUM = """builtin.module {
  func.func @forward(%arg0: tensor<2x3xi64>) -> tensor<2xi64> {
    %zero = "arith.constant"() <{value = 0 : i64}> : () -> i64
    %init = "tensor.splat"(%zero) : (i64) -> tensor<2xi64>
    %sum = "linalg.reduce"(%arg0, %init) <{dimensions = array<i64: 1>}> ({
    ^bb1(%elem: i64, %acc: i64):
      %added = "arith.addi"(%elem, %acc) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      "linalg.yield"(%added) : (i64) -> ()
    }) {prov.aten = "aten.sum.dim_IntList"} : (tensor<2x3xi64>, tensor<2xi64>) -> tensor<2xi64>
    func.return %sum : tensor<2xi64>
  }
}"""


def _selected() -> dict:
    return {
        "schema": "merlin.selected-index-lowering.v1",
        "compiler_requested": "neutral-clang",
        "compiler_resolved": "/neutral/clang",
        "compiler_sha256": "a" * 64,
        "cross_flags": ["--target=neutral"],
        "data_layout": "e-p:64:64",
        "index_bits": 64,
        "scope": "neutral test premise",
    }


def _actual(selected: dict) -> dict:
    return {
        **selected,
        "effective_pipeline": ",".join(
            f"{name}{{index-bitwidth=64}}"
            for name in (
                "convert-index-to-llvm",
                "convert-arith-to-llvm",
                "finalize-memref-to-llvm",
                "convert-func-to-llvm",
                "convert-cf-to-llvm",
            )
        ),
    }


def _prepared(tmp_path, raw: str = _SUM):
    path = tmp_path / "neutral.mlir"
    path.write_text(raw)
    normalized, receipt = normalize_capture_mlir(raw)
    parsed = tuple(mq.walk(mq.parse(normalized)))
    ordinals = [i for i, op in enumerate(parsed) if mq.op_name(op) == "linalg.reduce"]
    assert len(ordinals) == 1
    structure = operation_structure(parsed[ordinals[0]])
    row = {
        "mlir_operation": "linalg.reduce",
        "frontend_op": "aten.sum.dim_IntList",
        "ordinals": ordinals,
        "count": 1,
        "semantic_family": "reduction",
        **structure,
    }
    selected = _selected()
    raw_sha = sha256(raw.encode()).hexdigest()
    source = {
        "source_sha256": raw_sha,
        "normalized_source_sha256": receipt["output_sha256"],
        "n_source_operations": len(parsed),
        "selected_index_observation": selected,
        support.FIELD: support.begin(raw_sha, receipt["output_sha256"], len(parsed), selected),
    }
    return path, parsed, row, source


def _host(row, *, declared_ops=("aten.sum.dim_IntList",), status="reviewed", dtypes=None):
    observed = {
        "family": "reduction",
        "ordered_operand_dtypes": [item["dtype"] for item in row["ordered_operand_types"]],
        "ordered_result_dtypes": [item["dtype"] for item in row["ordered_result_types"]],
        "rank": len(row["ordered_result_types"][0]["shape"]),
    }
    declaration = {
        "id": "neutral_f32_sum",
        "status": status,
        "ops": list(declared_ops),
        "families": ["reduction"],
        "placement": "host",
        "signature": {
            "family": "reduction",
            "ordered_operand_dtypes": dtypes or observed["ordered_operand_dtypes"],
            "ordered_result_dtypes": observed["ordered_result_dtypes"],
        },
    }
    selected = {
        "neutral": {
            "package_sha256": "a" * 64,
            "capability_spec_sha256": "b" * 64,
            "dtype_strategy": "int8_w8a8",
            "capability_spec": {
                "schema": "merlin.host_capabilities.v1",
                "status": "reviewed",
                "compiler": {"package_sha256": "a" * 64, "dtype_strategy": "int8_w8a8"},
                "operations": [declaration],
                "evidence": {"scope": "neutral structural test, not numerical qualification"},
            },
        }
    }
    return admit_host_operation(selected, row, observed)


def _dispatch(source, parsed, row, decision, *, with_f32=True):
    witnesses = {
        "linalg": linalg.begin(source["source_sha256"], source["normalized_source_sha256"], len(parsed), _selected()),
        "integer": integer.begin(source["source_sha256"], source["normalized_source_sha256"], len(parsed), _selected()),
        "f32": support.begin(source["source_sha256"], source["normalized_source_sha256"], len(parsed), _selected()),
    }
    dispatch.record(
        witnesses["linalg"],
        witnesses["integer"],
        {"expected_ordinals": []},
        row,
        decision,
        parsed,
        {row["ordinals"][0]: row},
        None,
        f32_reductions=witnesses["f32"] if with_f32 else None,
    )
    return witnesses


def _linked(source):
    triple = {"candidate_tree_sha256": "c" * 64, "capture_tree_sha256": "d" * 64, "elf_sha256": "e" * 64}
    actual = _actual(source["selected_index_observation"])
    support.link(source, actual, triple)
    return {**triple, "source_sha256": source["source_sha256"], "index_lowering": actual}


def test_declared_reviewed_f32_sum_goes_to_the_f32_witness_and_links(tmp_path):
    path, parsed, row, source = _prepared(tmp_path)
    decision = _host(row)
    assert decision["status"] == "admitted" and decision["reviewed"] is True
    witnesses = _dispatch(source, parsed, row, decision)
    assert witnesses["integer"]["count"] == 0
    assert witnesses["f32"]["count"] == 1
    occurrence = witnesses["f32"]["occurrences"][0]
    assert occurrence["declaration"] == "neutral_f32_sum"
    assert occurrence["pattern"] == {
        "operation": "f32_sum",
        "input_shape": [2, 3],
        "output_shape": [2],
        "axis": [1],
        "accumulator_type": "f32",
        "init_bits": "0x00000000",
        "index_bits_premise": 64,
    }
    source[support.FIELD] = witnesses["f32"]
    support.verify_source(source[support.FIELD], path)
    entry = _linked(source)
    assert support.linked_complete(source, entry, "c" * 64)
    assert support.linked_complete(json.loads(json.dumps(source)), json.loads(json.dumps(entry)), "c" * 64)
    assert not support.linked_complete(source, entry, "f" * 64)
    for field in ("capture_tree_sha256", "elf_sha256", "index_lowering"):
        damaged = deepcopy(entry)
        damaged.pop(field)
        assert not support.linked_complete(source, damaged, "c" * 64)
    for key, value in (
        ("axis", [0]),
        ("init_bits", "0x80000000"),
        ("accumulator_type", "f64"),
        ("index_bits_premise", 32),
    ):
        damaged = deepcopy(source)
        damaged[support.FIELD]["occurrences"][0]["pattern"][key] = value
        damaged[support.FIELD]["occurrences_sha256"] = support._digest(damaged[support.FIELD]["occurrences"])
        assert not support.linked_complete(damaged, entry, "c" * 64)


def test_f32_sum_is_admitted_only_when_declared_and_reviewed(tmp_path):
    _, parsed, row, source = _prepared(tmp_path)
    assert _host(row, declared_ops=("aten.mean.dim",))["status"] != "admitted"
    assert _host(row, dtypes=["i64", "i64"])["status"] != "admitted"
    unreviewed = _host(row, status="unreviewed")
    assert unreviewed["status"] != "admitted"
    decision = _host(row)
    with pytest.raises(ValueError, match="no mandatory private witness"):
        _dispatch(source, parsed, row, decision, with_f32=False)
    for edit, message in (
        (lambda bad: bad.update(reviewed=False), "reviewed host admission"),
        (lambda bad: bad["profiles"][0]["decisions"][0].update(op="aten.mean.dim"), "reviewed declaration"),
        (lambda bad: bad["profiles"][0]["decisions"][0].update(review_status="unreviewed"), "reviewed declaration"),
        (
            lambda bad: bad["profiles"][0]["decisions"].append(deepcopy(bad["profiles"][0]["decisions"][0])),
            "reviewed declaration",
        ),
        (lambda bad: bad["profiles"][0].pop("decisions"), "reviewed declaration"),
        (lambda bad: bad["profiles"][0].update(capability_spec_sha256="wrong"), "unique reviewed host profile"),
        (lambda bad: bad.update(source_body_proof={"schema": "forged"}), "reviewed host admission"),
    ):
        bad = deepcopy(decision)
        edit(bad)
        with pytest.raises(ValueError, match=message):
            support.record(source[support.FIELD], row, bad, parsed, {row["ordinals"][0]: row})
    support.record(source[support.FIELD], row, decision, parsed, {row["ordinals"][0]: row})
    with pytest.raises(ValueError, match="repeated"):
        support.record(source[support.FIELD], row, decision, parsed, {row["ordinals"][0]: row})


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('"arith.addf"(%elem, %acc)', '"arith.mulf"(%elem, %acc)'),
        ('"arith.addf"(%elem, %acc)', '"arith.maximumf"(%elem, %acc)'),
        ('"arith.addf"(%elem, %acc)', '"arith.addf"(%acc, %elem)'),
        ("fastmath = #arith.fastmath<none>", "fastmath = #arith.fastmath<reassoc>"),
        ("value = 0.000000e+00 : f32", "value = 1.000000e+00 : f32"),
        ("value = 0.000000e+00 : f32", "value = -0.000000e+00 : f32"),
    ],
)
def test_mismatched_combiner_or_identity_is_refused(tmp_path, old, new):
    assert _SUM.count(old) == 1
    _, parsed, row, source = _prepared(tmp_path, _SUM.replace(old, new))
    with pytest.raises(ValueError, match="not a closed f32 sum"):
        support.record(source[support.FIELD], row, _host(row), parsed, {row["ordinals"][0]: row})


def test_mismatched_dims_or_dtype_is_refused(tmp_path):
    # A reduce over no dimension is not a sum reduction.
    no_dims = _SUM.replace("dimensions = array<i64: 1>", "dimensions = array<i64>").replace(
        "tensor<2xf32>", "tensor<2x3xf32>"
    )
    _, parsed, row, source = _prepared(tmp_path, no_dims)
    with pytest.raises(ValueError, match="not a closed f32 sum"):
        support.record(source[support.FIELD], row, _host(row), parsed, {row["ordinals"][0]: row})
    # The recorded reduced axis is bound: the same geometry over another axis is refused.
    square = _SUM.replace("tensor<2x3xf32>", "tensor<2x2xf32>")
    path, parsed, row, source = _prepared(tmp_path, square)
    support.record(source[support.FIELD], row, _host(row), parsed, {row["ordinals"][0]: row})
    assert source[support.FIELD]["occurrences"][0]["pattern"]["axis"] == [1]
    path.write_text(square.replace("dimensions = array<i64: 1>", "dimensions = array<i64: 0>"))
    with pytest.raises(ValueError, match="source bytes or recomputed ordinal bodies changed"):
        support.verify_source(source[support.FIELD], path)
    # An f64 (or other non-f32) sum is a different reduction: the f32 witness refuses it, and
    # dispatch leaves it to the integer witness, which refuses it exactly as before.
    f64 = _SUM.replace("f32", "f64")
    _, parsed, row, source = _prepared(tmp_path, f64)
    assert not support.selects(row, parsed)
    with pytest.raises(ValueError, match="differs from parsed source"):
        support.record(source[support.FIELD], row, _host(row), parsed, {row["ordinals"][0]: row})
    with pytest.raises(ValueError, match="integer reduction host support"):
        _dispatch(source, parsed, row, _host(row))


def test_integer_sum_dispatch_and_refusal_are_unchanged(tmp_path):
    _, parsed, row, source = _prepared(tmp_path, _INTEGER_SUM)
    assert not support.selects(row, parsed)
    witnesses = _dispatch(source, parsed, row, _host(row))
    assert witnesses["f32"]["count"] == 0
    assert witnesses["integer"]["count"] == 1
    assert witnesses["integer"]["occurrences"][0]["kind"] == "sum"
    # The integer witness itself still refuses an f32 sum, exactly as before.
    _, parsed, row, source = _prepared(tmp_path, _SUM)
    witness = integer.begin(source["source_sha256"], source["normalized_source_sha256"], len(parsed), _selected())
    with pytest.raises(
        ValueError, match="integer reduction host support: parsed source body is not a closed reduction"
    ):
        integer.record(witness, row, _host(row), parsed, {row["ordinals"][0]: row})


def test_changed_source_and_missing_width_are_refused(tmp_path):
    path, parsed, row, source = _prepared(tmp_path)
    support.record(source[support.FIELD], row, _host(row), parsed, {row["ordinals"][0]: row})
    path.write_text(path.read_text().replace("value = 0.000000e+00 : f32", "value = 1.000000e+00 : f32"))
    with pytest.raises(ValueError):
        support.verify_source(source[support.FIELD], path)
    with pytest.raises(ValueError, match="source premise"):
        support.link(source, _actual(_selected()), {"candidate_tree_sha256": "c" * 64})
    _, parsed, row, source = _prepared(tmp_path)
    empty = support.begin(source["source_sha256"], source["normalized_source_sha256"], len(parsed), None)
    with pytest.raises(ValueError, match="selected producer-owned index width"):
        support.record(empty, row, _host(row), parsed, {row["ordinals"][0]: row})
