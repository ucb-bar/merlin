"""Bind bounded W8A8 claims to exact contracts, actual programs and interpreted proof artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from test_lean_w8a8_bounded import contract, embedding_source
from test_lean_w8a8_bounded_calls import called_embedding
from test_lean_w8a8_bounded_chain import preprocessed
from test_lean_w8a8_receipts import stage_at

from merlin.common.paths import repo_root
from merlin.verify.lean import bounded_numerics, bounded_policy, interpreted

pytestmark = pytest.mark.usefixtures("lean_oracle")


def invoke(stages, bounds, destination, tools, *, index_bits=64, scope="all", **kwargs):
    contracts = destination.parent / f"{destination.name}-contracts.json"
    contracts.write_text(json.dumps(bounds))
    receipt = bounded_policy.check_w8a8_bounded_chain_draft(
        stages,
        destination,
        input_contracts=contracts,
        m2m_python=tools.m2m_python,
        scope=scope,
        index_bits=index_bits,
        **kwargs,
    )
    return receipt, contracts


def evidence(receipt):
    import jsonschema

    assert receipt["verdict"] == "abstracted", receipt
    jsonschema.validate(
        receipt, json.loads((repo_root() / "merlin/contract/schemas/lean_receipt.schema.json").read_text())
    )
    assert receipt["claim_type"] == "policy"
    assert receipt["checker"] == f"{bounded_numerics.MODULE}.check"
    assert receipt["theorem"] == f"{bounded_numerics.MODULE}.check_sound"
    report = json.loads(Path(receipt["cert"]["path"]).read_text())
    assert receipt["lean"]["checker_sha256"] == report["runtime_sha256"]
    assert (
        hashlib.sha256(json.dumps(report["runtime"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        == report["runtime_sha256"]
    )
    assert report["runtime"]["kind"] == "lean_interpreter_with_local_proof_artifacts"
    assert report["input_contracts"]["snapshot"]["sha256"] in receipt["domain"]
    assert set(receipt["axioms"]) <= {"propext", "Classical.choice", "Quot.sound"}
    numerical = report["conditional_numerical"]
    for key in ("policy_reviewed", "primitive_model_laws_verified", "whole_pass_verified"):
        assert numerical[key] is False
    assert numerical["input_domain"] == bounded_numerics.INPUT_DOMAIN
    return report


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("kind", ["embedding", "called", "projection"])
@pytest.mark.parametrize("preprocess", [False, True])
def test_actual_bounded_boundary_and_suffix_have_truthful_receipts(
    tmp_path, lean_tools, monkeypatch, index_bits, kind, preprocess
):
    if kind == "called":
        before, bounds = called_embedding(3)
    elif kind == "projection":
        from lean_w8a8_projection_cases import CASES

        before = CASES["two_reductions"].source()
        bounds = [{"function": "forward", "intervals": [None, None]}]
    else:
        before, bounds = embedding_source(), contract(7)
    first = stage_at(tmp_path, before)
    if preprocess:
        from merlin.frontends.linalg_mlir import parse_mlir_text
        from merlin.verify.lean import stages as recording

        preprocessed(parse_mlir_text((first / "after.mlir").read_text()), tmp_path, monkeypatch)
        rest = recording.completed(tmp_path / "preprocessing")
    else:
        rest = []
    receipt, _ = invoke([first, *rest], bounds, tmp_path / "bounded", lean_tools, index_bits=index_bits)
    report = evidence(receipt)
    assert report["conditional_numerical"]["conditional_initialized_policy_refinement"]
    assert len(report["boundary_links"]) == len(rest) + 1
    assert report["input_contracts"]["entries"] == {entry["function"]: entry["intervals"] for entry in bounds}
    assert report["conditional_numerical"]["functions"][0]["input_intervals"] == bounds[0]["intervals"]


@pytest.mark.parametrize("change", ["unknown", "empty", "arity", "out_of_bounds"])
def test_unproved_domains_remain_refused_in_receipts(tmp_path, lean_tools, change):
    first = stage_at(tmp_path, embedding_source())
    bounds = contract(7)
    if change == "unknown":
        bounds[0]["intervals"][0] = None
    elif change == "empty":
        bounds[0]["intervals"][0]["lower"] = 7
    elif change == "arity":
        bounds[0]["intervals"].pop()
    else:
        bounds[0]["intervals"][0]["upper"] = 8
    receipt, _ = invoke([first], bounds, tmp_path / "bounded", lean_tools)
    numerical = evidence(receipt)["conditional_numerical"]
    assert not numerical["conditional_initialized_policy_refinement"]
    assert not numerical["boundary_qualification_passed"]
    assert "refused" in receipt["reason"]


def test_changed_output_is_not_hidden_by_pending_review(tmp_path, lean_tools):
    first = stage_at(tmp_path, embedding_source())
    (first / "after.mlir").write_bytes((first / "before.mlir").read_bytes())
    receipt, _ = invoke([first], contract(7), tmp_path / "bounded", lean_tools)
    assert not evidence(receipt)["conditional_numerical"]["conditional_initialized_policy_refinement"]


@pytest.mark.parametrize("change", ["duplicate", "missing", "extra", "negative", "float", "bool", "unknown_key"])
def test_malformed_contracts_invalidate_the_attempt(tmp_path, lean_tools, change):
    first = stage_at(tmp_path, embedding_source())
    bounds = contract(7)
    if change == "duplicate":
        bounds += contract(7)
    elif change == "missing":
        bounds = []
    elif change == "extra":
        bounds += [{"function": "unrelated", "intervals": []}]
    elif change == "unknown_key":
        bounds[0]["intervals"][0]["trusted"] = True
    else:
        bounds[0]["intervals"][0]["lower"] = {"negative": -1, "float": 0.5, "bool": False}[change]
    receipt, _ = invoke([first], bounds, tmp_path / "bounded", lean_tools)
    assert receipt["schema"] == "merlin.lean.attempt.v1" and receipt["verdict"] == "unmeasured", receipt
    assert "cert" not in receipt


@pytest.mark.parametrize("integer_left", [False, True])
def test_captured_quantization_keeps_its_source_interpretation(tmp_path, lean_tools, integer_left):
    from test_lean_w8a8_dynamic_policy import reused_dynamic_source

    first = stage_at(tmp_path, reused_dynamic_source(integer_left))
    bounds = [{"function": "forward", "intervals": [None, None]}]
    receipt, _ = invoke([first], bounds, tmp_path / "bounded", lean_tools, scope="captured")
    report = evidence(receipt)
    assert report["conditional_numerical"]["conditional_initialized_policy_refinement"]
    assert report["conditional_numerical"]["policy_scope"] == "captured"


def test_duplicate_json_object_keys_are_not_a_domain_declaration(tmp_path, lean_tools):
    first = stage_at(tmp_path, embedding_source())
    contracts = tmp_path / "contracts.json"
    contracts.write_text('[{"function":"forward","intervals":[{"lower":0,"lower":1,"upper":7},null,null]}]')
    receipt = bounded_policy.check_w8a8_bounded_chain_draft(
        [first], tmp_path / "bounded", input_contracts=contracts, m2m_python=lean_tools.m2m_python
    )
    assert receipt["verdict"] == "unmeasured" and "duplicate input contract object key" in receipt["reason"]


@pytest.mark.parametrize("change", ["contracts", "source", "export", "entry"])
def test_live_byte_changes_cannot_reuse_a_successful_result(tmp_path, lean_tools, monkeypatch, change):
    first = stage_at(tmp_path, embedding_source())
    original_capture = interpreted.capture
    original_run = interpreted.Runtime.run
    copied_entry = tmp_path / "Entry.lean"
    copied_entry.write_bytes((repo_root() / "merlin/lean/W8A8BoundedMain.lean").read_bytes())

    def capture(_entry, **kwargs):
        return original_capture(copied_entry, **kwargs)

    def altered(self, arguments, **kwargs):
        result = original_run(self, arguments, **kwargs)
        path = {
            "contracts": tmp_path / "bounded-contracts.json",
            "source": first / "before.mlir",
            "export": Path(arguments[0]),
            "entry": copied_entry,
        }[change]
        path.chmod(0o644)
        path.write_text(path.read_text() + "\n")
        return result

    monkeypatch.setattr(interpreted, "capture", capture)
    monkeypatch.setattr(interpreted.Runtime, "run", altered)
    receipt, _ = invoke([first], contract(7), tmp_path / "bounded", lean_tools)
    assert receipt["schema"] == "merlin.lean.attempt.v1" and receipt["verdict"] == "unmeasured", receipt
    assert "changed" in receipt["reason"] and "cert" not in receipt
