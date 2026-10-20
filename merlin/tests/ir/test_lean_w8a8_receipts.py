"""Byte-bound W8A8 draft integration must not bypass review or ordinary-chain obligations."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from test_lean_w8a8_module import calibrated_source, source

from merlin.common.paths import repo_root
from merlin.llvmlower.passes_quant_int import lower_contraction_int8
from merlin.verify.lean.check import check_support, compose_chain
from merlin.verify.lean.policy import check_w8a8_draft


def stage_at(tmp_path, before):
    stage = tmp_path / "stage"
    stage.mkdir()
    after = before.clone()
    assert lower_contraction_int8(after) == 1
    (stage / "before.mlir").write_text(str(before))
    (stage / "after.mlir").write_text(str(after))
    (stage / "meta.json").write_text(json.dumps({"boundary": "prepare-quantize-int8"}))
    return stage


@pytest.mark.parametrize("calibrated", [False, True])
def test_draft_receipt_keeps_matching_and_review_separate(tmp_path, lean_tools, calibrated):
    import jsonschema

    stage = stage_at(tmp_path, calibrated_source() if calibrated else source())
    receipt = check_w8a8_draft(stage, m2m_python=lean_tools.m2m_python, scope="captured" if calibrated else "all")
    schema = repo_root() / "merlin/contract/schemas/lean_receipt.schema.json"
    jsonschema.validate(receipt, json.loads(schema.read_text()))
    assert receipt["verdict"] == "abstracted", receipt
    assert receipt["claim_type"] == "policy"
    assert receipt["exporter"]["round_trip"] == "identical"
    report = json.loads(Path(receipt["cert"]["path"]).read_text())
    assert report["conditional_match"]
    assert report["qualification_passed"] is not calibrated
    assert report["policy_reviewed"] is False
    assert "independent specification review" in receipt["reason"]


def test_bad_output_is_not_hidden_by_the_pending_review(tmp_path, lean_tools):
    stage = stage_at(tmp_path, source())
    (stage / "after.mlir").write_text((stage / "before.mlir").read_text())
    receipt = check_w8a8_draft(stage, m2m_python=lean_tools.m2m_python, scope="all")
    assert receipt["verdict"] == "abstracted"
    report = json.loads(Path(receipt["cert"]["path"]).read_text())
    assert not report["conditional_match"]
    assert not report["qualification_passed"]


def test_forged_policy_verdict_cannot_be_composed_as_ordinary_refinement(tmp_path, lean_tools):
    stage = stage_at(tmp_path, source())
    receipt = check_w8a8_draft(stage, m2m_python=lean_tools.m2m_python, scope="all")
    support = tmp_path / "support"
    support_receipt = check_support(stage / "before.mlir", "forward", support, m2m_python=lean_tools.m2m_python)
    assert support_receipt["verdict"] == "verified", support_receipt
    receipt["verdict"] = "verified"
    (stage / "receipt.json").write_text(json.dumps(receipt))
    chain = compose_chain(
        [stage], support / "receipt.json", "forward", tmp_path / "chain.json", m2m_python=lean_tools.m2m_python
    )
    assert chain["verdict"] != "verified"
    assert chain["refinement_verdict"] == "abstracted"


def test_driver_cannot_promote_an_unreviewed_draft(tmp_path, lean_tools, monkeypatch):
    import merlin.verify.lean.policy as policy

    stage = stage_at(tmp_path, source())
    original_run = subprocess.run

    def forged(command, **kwargs):
        result = original_run(command, **kwargs)
        if command[1] == "check-w8a8-draft":
            report = json.loads(result.stdout)
            report["verdict"] = "verified"
            return subprocess.CompletedProcess(command, 0, json.dumps(report), "")
        return result

    monkeypatch.setattr(policy.subprocess, "run", forged)
    receipt = check_w8a8_draft(stage, m2m_python=lean_tools.m2m_python, scope="all")
    assert receipt["verdict"] == "unmeasured"
    assert receipt["schema"] == "merlin.lean.attempt.v1"
    assert "unreviewed policy cannot produce a verified receipt" in receipt["reason"]
