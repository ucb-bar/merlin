"""Replay a real W8A8 boundary followed by actual Merlin preprocessing boundaries."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from test_lean_w8a8_module import source
from test_lean_w8a8_receipts import stage_at

from merlin.common.paths import repo_root
from merlin.llvmlower.passes_xdsl import preprocess_text
from merlin.verify.lean import policy
from merlin.verify.lean import stages as recording


def report(receipt):
    return json.loads(Path(receipt["cert"]["path"]).read_text())


def identity_suffix(root, previous, *, name="identity"):
    stage = root / name
    stage.mkdir()
    for key in ("before", "after"):
        (stage / f"{key}.mlir").write_bytes((previous / "after.mlir").read_bytes())
    (stage / "meta.json").write_text(json.dumps({"boundary": name}))
    return stage


@pytest.mark.parametrize("preprocess", [False, True])
def test_policy_with_real_preprocessing_suffix(tmp_path, lean_tools, monkeypatch, preprocess):
    import jsonschema

    first = stage_at(tmp_path, source())
    suffix = []
    if preprocess:
        monkeypatch.setenv("MERLIN_LEAN_STAGES", str(tmp_path / "preprocessing"))
        preprocess_text((first / "after.mlir").read_text())
        suffix = recording.completed(tmp_path / "preprocessing")
        assert len(suffix) == 4
    receipt = policy.check_w8a8_chain_draft(
        [first, *suffix], tmp_path / "result", m2m_python=lean_tools.m2m_python, scope="all"
    )
    jsonschema.validate(
        receipt, json.loads((repo_root() / "merlin/contract/schemas/lean_receipt.schema.json").read_text())
    )
    assert receipt["checker"] == policy.CHAIN_CHECKER
    assert receipt["theorem"] == policy.CHAIN_THEOREM
    assert receipt["verdict"] == "abstracted", receipt
    checked = report(receipt)
    assert checked["conditional_match"] and checked["suffix_match"] and checked["qualification_passed"], checked
    assert checked["policy_reviewed"] is False and checked["whole_pass_verified"] is False
    assert len(checked["suffix_boundaries"]) == len(suffix)
    assert len(checked["boundary_links"]) == len(suffix) + 1
    assert len(checked["program_exports"]) == len(suffix) + 2
    assert receipt["before"]["path"] == str((first / "before.mlir").resolve())
    assert receipt["after"]["path"] == str(((suffix[-1] if suffix else first) / "after.mlir").resolve())


def test_typed_suffix_preserves_each_functions_original_domain(tmp_path, lean_tools):
    from test_lean_splat_inputs import _source

    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.splat_inputs import scalarize_splat_inputs
    from merlin.verify.lean.check import check_stage

    before = source()
    helper = parse_mlir_text(_source(element="f32"))
    before.body.block.add_ops(op.clone() for op in helper.body.block.ops)
    before.verify()
    first = stage_at(tmp_path, before)
    suffix = identity_suffix(tmp_path, first)
    compacted = parse_mlir_text((suffix / "after.mlir").read_text())
    assert scalarize_splat_inputs(compacted) == 1
    compacted.verify()
    (suffix / "after.mlir").write_text(str(compacted))
    # This boundary requires a typed splat proof; a passing universal C1 path cannot mask it.
    universal = check_stage(suffix, m2m_python=lean_tools.m2m_python)
    assert universal["verdict"] == "abstracted"
    receipt = policy.check_w8a8_chain_draft(
        [first, suffix], tmp_path / "result", m2m_python=lean_tools.m2m_python, scope="all"
    )
    checked = report(receipt)
    assert checked["functions"] == ["forward", "subject"]
    assert checked["conditional_match"] and checked["suffix_match"] and checked["qualification_passed"], checked
    assert receipt["verdict"] == "abstracted" and checked["policy_reviewed"] is False


def test_valid_suffix_mutation_refused(tmp_path, lean_tools):
    from test_lean_w8a8_body import exported
    from xdsl.dialects.builtin import FloatAttr, f32

    from merlin.frontends.linalg_mlir import parse_mlir_text

    first = stage_at(tmp_path, source())
    suffix = identity_suffix(tmp_path, first)
    changed = parse_mlir_text((suffix / "after.mlir").read_text())
    scale = next(
        op
        for op in changed.walk()
        if op.name == "linalg.generic" and any(body.name == "arith.select" for body in op.body.block.ops)
    )
    list(scale.body.block.ops)[2].properties["value"] = FloatAttr(1e-4, f32)
    verification = tmp_path / "compiler-verified"
    verification.mkdir()
    exported(changed, verification, lean_tools)
    (suffix / "after.mlir").write_text(str(changed))
    receipt = policy.check_w8a8_chain_draft(
        [first, suffix], tmp_path / "result", m2m_python=lean_tools.m2m_python, scope="all"
    )
    checked = report(receipt)
    assert checked["conditional_match"]
    assert not checked["suffix_match"] and not checked["qualification_passed"]
    assert "suffix boundary 0" in receipt["reason"]


def test_chain_refuses_disconnected_actual_boundaries(tmp_path, lean_tools):
    first = stage_at(tmp_path, source())
    suffix = identity_suffix(tmp_path, first)
    with (suffix / "before.mlir").open("a") as stream:
        stream.write("\n// different recorded boundary\n")
    receipt = policy.check_w8a8_chain_draft(
        [first, suffix], tmp_path / "result", m2m_python=lean_tools.m2m_python, scope="all"
    )
    assert receipt["verdict"] == "unmeasured"
    assert "byte-identical policy/suffix chain" in receipt["reason"]


def test_live_intermediate_change_during_check_refused(tmp_path, lean_tools, monkeypatch):
    first = stage_at(tmp_path, source())
    middle = identity_suffix(tmp_path, first, name="middle")
    final = identity_suffix(tmp_path, middle, name="final")
    run = subprocess.run

    def mutate(command, **kwargs):
        result = run(command, **kwargs)
        if command[1] == "check-w8a8-chain-draft":
            with (middle / "after.mlir").open("a") as stream:
                stream.write("\n// changed after snapshot\n")
        return result

    monkeypatch.setattr(policy.subprocess, "run", mutate)
    receipt = policy.check_w8a8_chain_draft(
        [first, middle, final], tmp_path / "result", m2m_python=lean_tools.m2m_python, scope="all"
    )
    assert receipt["verdict"] == "unmeasured"
    assert "input or bound report changed" in receipt["reason"]


def test_bound_policy_report_is_checked_for_freshness(tmp_path, lean_tools, monkeypatch):
    first = stage_at(tmp_path, source())
    finish = policy._finish

    def corrupt_report(destination, receipt):
        with Path(receipt["cert"]["path"]).open("a") as stream:
            stream.write("\n")
        return finish(destination, receipt)

    monkeypatch.setattr(policy, "_finish", corrupt_report)
    receipt = policy.check_w8a8_chain_draft([first], tmp_path / "result", m2m_python=lean_tools.m2m_python, scope="all")
    assert receipt["verdict"] == "unmeasured"
    assert "cert file changed" in receipt["reason"]


def test_forged_input_receipt_does_not_replace_a_policy_check(tmp_path, lean_tools):
    first = stage_at(tmp_path, source())
    (first / "after.mlir").write_bytes((first / "before.mlir").read_bytes())
    (first / "receipt.json").write_text(json.dumps({"verdict": "verified"}))
    receipt = policy.check_w8a8_chain_draft([first], tmp_path / "result", m2m_python=lean_tools.m2m_python, scope="all")
    assert receipt["verdict"] == "abstracted"
    assert not report(receipt)["conditional_match"]
    assert not report(receipt)["qualification_passed"]


def test_empty_chain_refuses_without_tools(tmp_path):
    receipt = policy.check_w8a8_chain_draft([], tmp_path / "result", m2m_python=Path("missing"))
    assert receipt["verdict"] == "unmeasured"
    assert "requires a policy boundary" in receipt["reason"]
