"""Neutral source-to-linked-build checks for prepared f32 maximum roots."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256

import pytest
from merlin_experiments.phase1.feedback import private_f32_maximum_support as support
from merlin_experiments.phase1.feedback import private_host_source_dispatch as dispatch

from merlin.common import mlir_query as mq
from merlin.compile.model_execution_inputs import strict_tree_sha256
from merlin.frontends.capture_normalization import normalize_capture_mlir
from merlin.frontends.linalg_f32_maximum_patterns import (
    STATIC_F32_MAXIMUM_SOURCE_BODY_SCHEMA,
    recognize_static_f32_maximum,
    serialized_f32_maximum_pattern,
)
from merlin.targetgen.application_inventory import operation_structure
from merlin.targetgen.host_capabilities import admit_host_operation


def _program() -> str:
    return """builtin.module {
  func.func @neutral(%x: tensor<2x3x7xf32>) -> tensor<2x3xf32> {
    %negative = arith.constant 0xff800000 : f32
    %init = tensor.splat %negative : tensor<2x3xf32>
    %result = "linalg.reduce"(%x, %init) <{dimensions = array<i64: 2>}> ({
      ^bb0(%value: f32, %acc: f32):
        %maximum = "arith.maximumf"(%value, %acc) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "linalg.yield"(%maximum) : (f32) -> ()
    }) {prov.aten = "aten.amax.default"} : (tensor<2x3x7xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    func.return %result : tensor<2x3xf32>
  }
}"""


def _two_roots_program() -> str:
    text = _program()
    reduction = text.split('    %result = "linalg.reduce"', 1)[1].split("    func.return %result", 1)[0]
    second = '    %second = "linalg.reduce"' + reduction.replace("%maximum", "%maximum2")
    return text.replace("    func.return %result", second + "    func.return %result")


def _selected() -> dict:
    return {
        "schema": "merlin.selected-index-lowering.v1",
        "compiler_requested": "neutral-clang",
        "compiler_resolved": "/neutral/clang",
        "compiler_sha256": "a" * 64,
        "cross_flags": ["--target=neutral"],
        "data_layout": "e-p:64:64",
        "index_bits": 64,
        "scope": "neutral explicit premise",
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


def _prepared(tmp_path, text: str | None = None):
    capture = tmp_path / "capture"
    capture.mkdir()
    text = _program() if text is None else text
    model = capture / "model.mlir"
    model.write_text(text, encoding="utf-8")
    receipt = capture / "capture_receipt.json"
    receipt.write_text('{"schema":"neutral"}', encoding="utf-8")
    normalized, _ = normalize_capture_mlir(text)
    parsed = tuple(mq.walk(mq.parse(normalized)))
    ordinal = next(index for index, op in enumerate(parsed) if mq.op_name(op) == "linalg.reduce")
    selected = _selected()
    pattern = serialized_f32_maximum_pattern(recognize_static_f32_maximum(parsed[ordinal], index_bits=64))
    row = {"frontend_op": "aten.amax.default", "mlir_operation": "linalg.reduce", "ordinals": [ordinal], "count": 1}
    body = {
        "schema": STATIC_F32_MAXIMUM_SOURCE_BODY_SCHEMA,
        "declaration": "neutral-reviewed-maximum",
        "operation": "arith.maximumf",
        "selected_index_observation": deepcopy(selected),
        "patterns": [pattern],
        "profile": "neutral-host",
        "capability_spec_sha256": "b" * 64,
    }
    inner = {key: value for key, value in body.items() if key not in {"profile", "capability_spec_sha256"}}
    decision = {
        "status": "admitted",
        "reviewed": True,
        "source_body_proof": body,
        "profiles": [
            {
                "status": "admitted",
                "reviewed": True,
                "profile": "neutral-host",
                "capability_spec_sha256": "b" * 64,
                "decisions": [{"status": "admitted", "declaration": body["declaration"], "source_body_proof": inner}],
            }
        ],
    }
    raw = sha256(text.encode()).hexdigest()
    normalized_sha = sha256(normalized.encode()).hexdigest()
    receipt_sha = sha256(receipt.read_bytes()).hexdigest()
    witness = support.begin(raw, normalized_sha, receipt_sha, parsed, selected)
    return capture, parsed, ordinal, row, decision, selected, witness


def _linked(tmp_path):
    capture, parsed, ordinal, row, decision, selected, witness = _prepared(tmp_path)
    support.record(witness, row, decision, parsed, {ordinal: row})
    support.verify_source(witness, capture / "model.mlir")
    linked = {
        "candidate_tree_sha256": "c" * 64,
        "capture_tree_sha256": strict_tree_sha256(capture)["sha256"],
        "elf_sha256": "d" * 64,
    }
    source = {
        "source_sha256": witness["raw_source_sha256"],
        "normalized_source_sha256": witness["normalized_source_sha256"],
        "capture_receipt_sha256": witness["capture_receipt_sha256"],
        "n_source_operations": len(parsed),
        "selected_index_observation": selected,
        support.FIELD: witness,
    }
    actual = _actual(selected)
    support.link(source, actual, linked, capture_path=capture)
    entry = {
        "source_sha256": source["source_sha256"],
        "capture_tree_sha256": linked["capture_tree_sha256"],
        "elf_sha256": linked["elf_sha256"],
        "index_lowering": actual,
    }
    return source, entry, linked, capture


def test_reviewed_maximum_requires_exact_source_profile_and_linked_capture(tmp_path):
    source, entry, linked, _capture = _linked(tmp_path)
    assert source[support.FIELD]["count"] == 1
    assert support.linked_complete(source, entry, linked["candidate_tree_sha256"])


def test_prepared_parse_never_reuses_a_maximum_decision_or_capture_receipt(tmp_path):
    from merlin_experiments.phase1.feedback import private_linked_source_preparation as preparation

    source, entry, linked, capture = _linked(tmp_path)
    candidate = linked["candidate_tree_sha256"]
    with preparation.validation_scope():
        assert support.linked_complete(source, entry, candidate)
        changed = deepcopy(source)
        changed[support.FIELD]["occurrences"][0]["pattern"]["axis"] = 0
        changed[support.FIELD]["occurrences_sha256"] = support._digest(changed[support.FIELD]["occurrences"])
        assert not support.linked_complete(changed, entry, candidate)
        assert support.linked_complete(source, entry, candidate)
        (capture / "capture_receipt.json").write_text('{"schema":"changed"}')
        assert not support.linked_complete(source, entry, candidate)


def test_maximum_cannot_disappear_into_an_empty_host_roster(tmp_path):
    capture, _parsed, ordinal, _row, _decision, _selected_index, witness = _prepared(tmp_path)
    assert witness["expected_ordinals"] == [ordinal]
    with pytest.raises(ValueError, match="lacks reviewed admission"):
        support.verify_source(witness, capture / "model.mlir")


def test_reparsed_source_refuses_forged_subset_of_two_maximum_roots(tmp_path):
    capture, parsed, first, row, decision, selected, witness = _prepared(tmp_path, _two_roots_program())
    roots = [index for index, op in enumerate(parsed) if mq.op_name(op) == "linalg.reduce"]
    assert len(roots) == 2
    support.record(witness, row, decision, parsed, {first: row})
    witness["expected_ordinals"] = [first]
    witness["occurrences_sha256"] = support._digest(witness["occurrences"])  # noqa: PLC2701 -- forged roster
    with pytest.raises(ValueError, match="recomputed maximum ordinals"):
        support.verify_source(witness, capture / "model.mlir")

    # A forged linked record must also fail when all hashes and the captured
    # source itself consistently describe both actual roots.
    witness["status"] = support.LINKED
    witness["source_verified"] = True
    witness["source_capture_path"] = str(capture)
    linked = {
        "candidate_tree_sha256": "c" * 64,
        "capture_tree_sha256": strict_tree_sha256(capture)["sha256"],
        "elf_sha256": "d" * 64,
    }
    witness["linked_build"] = linked
    source = {
        "source_sha256": witness["raw_source_sha256"],
        "normalized_source_sha256": witness["normalized_source_sha256"],
        "capture_receipt_sha256": witness["capture_receipt_sha256"],
        "n_source_operations": len(parsed),
        "selected_index_observation": selected,
        support.FIELD: witness,
    }
    entry = {
        "source_sha256": witness["raw_source_sha256"],
        "capture_tree_sha256": linked["capture_tree_sha256"],
        "elf_sha256": linked["elf_sha256"],
        "index_lowering": _actual(selected),
    }
    assert not support.linked_complete(source, entry, linked["candidate_tree_sha256"])


def test_selected_index_width_requires_canonical_layout_and_compiler_identity(tmp_path):
    _capture, parsed, ordinal, row, decision, _selected_index, witness = _prepared(tmp_path)
    for mutation in ({"compiler_sha256": "bad"}, {"data_layout": "e-p:32:32"}):
        altered = deepcopy(witness)
        altered["selected_index_observation"].update(mutation)
        with pytest.raises(ValueError, match="selected producer-owned index width"):
            support.record(altered, row, decision, parsed, {ordinal: row})


def test_dispatch_refuses_new_schema_without_mandatory_private_witness(tmp_path):
    _capture, parsed, ordinal, row, decision, _selected_index, _witness = _prepared(tmp_path)
    with pytest.raises(ValueError, match="mandatory private witness"):
        dispatch.record({}, {}, {}, row, decision, parsed, {ordinal: row}, None)


def test_real_core_screen_proof_reaches_independent_private_witness(tmp_path):
    capture, parsed, ordinal, row, _decision, selected, witness = _prepared(tmp_path)
    structure = operation_structure(parsed[ordinal])
    row.update(structure)
    profile = {
        "neutral": {
            "package_sha256": "a" * 64,
            "capability_spec_sha256": "b" * 64,
            "dtype_strategy": "int8_w8a8",
            "capability_spec": {
                "schema": "merlin.host_capabilities.v1",
                "status": "reviewed",
                "compiler": {"package_sha256": "a" * 64, "dtype_strategy": "int8_w8a8"},
                "operations": [
                    {
                        "id": "neutral-reviewed-maximum",
                        "ops": ["aten.amax.default"],
                        "placement": "host",
                        "signature": {
                            "family": "reduction",
                            "ordered_operand_dtypes": ["f32", "f32"],
                            "ordered_result_dtypes": ["f32"],
                        },
                        "source_body": {
                            "schema": STATIC_F32_MAXIMUM_SOURCE_BODY_SCHEMA,
                            "operation": "arith.maximumf",
                        },
                        "numerical_contract": {"status": "unreviewed"},
                    }
                ],
                "evidence": {"scope": "neutral source screen only"},
            },
        }
    }
    decision = admit_host_operation(
        profile,
        row,
        {
            "family": "reduction",
            "ordered_operand_dtypes": ["f32", "f32"],
            "ordered_result_dtypes": ["f32"],
            "rank": 2,
        },
        source_operations=(parsed[ordinal],),
        source_context={"selected_index_observation": selected},
    )
    assert decision["status"] == "admitted"
    dispatch.record({}, {}, {}, row, decision, parsed, {ordinal: row}, None, maximum=witness)
    assert witness["count"] == 1
    support.verify_source(witness, capture / "model.mlir")


@pytest.mark.parametrize("mutation", ["unreviewed", "missing_body", "wrong_pattern", "wrong_inner", "wrong_width"])
def test_maximum_refuses_missing_or_changed_reviewed_source_proof(tmp_path, mutation):
    _capture, parsed, ordinal, row, decision, selected, witness = _prepared(tmp_path)
    if mutation == "unreviewed":
        decision["reviewed"] = False
    elif mutation == "missing_body":
        decision["source_body_proof"] = None
    elif mutation == "wrong_pattern":
        decision["source_body_proof"]["patterns"][0]["axis"] = 1
    elif mutation == "wrong_inner":
        decision["profiles"][0]["decisions"][0]["source_body_proof"]["declaration"] = "other"
    else:
        decision["source_body_proof"]["selected_index_observation"]["index_bits"] = 32
    with pytest.raises(ValueError):
        support.record(witness, row, decision, parsed, {ordinal: row})
    assert selected["index_bits"] == 64


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "old_schema",
        "changed_elf",
        "changed_source",
        "changed_receipt",
        "bool_axis",
        "float_ordinal",
        "omitted_root",
    ],
)
def test_maximum_completion_refuses_missing_old_or_tampered_evidence(tmp_path, mutation):
    source, entry, linked, capture = _linked(tmp_path)
    if mutation == "missing":
        source.pop(support.FIELD)
    elif mutation == "old_schema":
        source[support.FIELD]["status"] = support.PENDING
    elif mutation == "changed_elf":
        entry["elf_sha256"] = "e" * 64
    elif mutation == "changed_source":
        (capture / "model.mlir").write_text(_program().replace("arith.maximumf", "arith.minimumf"))
    elif mutation == "changed_receipt":
        (capture / "capture_receipt.json").write_text('{"schema":"changed"}')
    elif mutation == "omitted_root":
        source[support.FIELD]["expected_ordinals"] = []
    elif mutation == "float_ordinal":
        source[support.FIELD]["expected_ordinals"] = [float(source[support.FIELD]["expected_ordinals"][0])]
    else:
        source[support.FIELD]["occurrences"][0]["pattern"]["axis"] = True
        source[support.FIELD]["occurrences_sha256"] = support._digest(  # noqa: PLC2701 -- adversarial serialized record
            source[support.FIELD]["occurrences"]
        )
    assert not support.linked_complete(source, entry, linked["candidate_tree_sha256"])


def test_maximum_source_recheck_refuses_changed_body_before_link(tmp_path):
    capture, parsed, ordinal, row, decision, _selected_index, witness = _prepared(tmp_path)
    support.record(witness, row, decision, parsed, {ordinal: row})
    (capture / "model.mlir").write_text(_program().replace("#arith.fastmath<none>", "#arith.fastmath<fast>"))
    with pytest.raises(ValueError):
        support.verify_source(witness, capture / "model.mlir")
