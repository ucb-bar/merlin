"""Actual-boundary contract substitution and lifetime attacks against the new wrapper."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from lean_w8a8_ordered_cases import source
from test_lean_w8a8_ordered import prepare

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.verify.lean import bounded_numerics, ordered_evidence
from merlin.verify.lean import ordered_bound_policy as bound
from merlin.verify.lean.check import _file_ref

pytestmark = pytest.mark.usefixtures("lean_oracle")


def prepared(work, tools, *, integer=False, width=64):
    before = source()
    if integer:
        text = str(before)
        marker = ") -> tensor<2x4xf32> {"
        assert text.count(marker) == 1
        before = parse_mlir_text(text.replace(marker, ", %tag: tensor<1xi64>" + marker))
    result = prepare(work, tools, before=before, index_bits=width)
    if integer:
        result[1].write_text(
            json.dumps([{"function": "forward", "intervals": [None, None, {"lower": 0, "upper": 10}]}])
        )
        caps = json.loads(result[2].read_text())
        caps[0]["exponent_caps"].append(None)
        result[2].write_text(json.dumps(caps))
    return result


def invoke(config, destination, tools):
    stage, integers, exponents, width = config
    receipt = bound.check_w8a8_ordered_bound_draft(
        stage,
        destination,
        input_contracts=integers,
        exponent_contracts=exponents,
        m2m_python=tools.m2m_python,
        scope="all",
        index_bits=width,
    )
    assert receipt["boundary"] == "w8a8-ordered-contract-bound-candidate", receipt
    report = json.loads(Path(receipt["cert"]["path"]).read_text())
    assert report["requested_prefix_checks_passed"]
    assert not report["evidence"]["whole_pass_verified"]
    return receipt, report


def forged_receipt(work, receipt, wrapper, *, kind, replacement, copies):
    """Rehash every changed outer link, retaining the actual checked snapshots unchanged."""
    work.mkdir()
    wrapper = copy.deepcopy(wrapper)
    base_receipt = json.loads(Path(wrapper["base_receipt"]["path"]).read_text())
    base_report = json.loads(Path(wrapper["base_report"]["path"]).read_text())
    if copies in {"base", "both"}:
        base_report[kind]["original"] = _file_ref(replacement)
    if copies in {"wrapper", "both"}:
        wrapper["contract_bindings"][kind]["original"] = _file_ref(replacement)
    report_path, base_path, outer_path = work / "base-report.json", work / "base-receipt.json", work / "wrapper.json"
    report_path.write_text(json.dumps(base_report))
    base_receipt["cert"] = _file_ref(report_path)
    base_path.write_text(json.dumps(base_receipt))
    wrapper["base_receipt"], wrapper["base_report"] = _file_ref(base_path), _file_ref(report_path)
    outer_path.write_text(json.dumps(wrapper))
    forged = {**receipt, "cert": _file_ref(outer_path)}
    path = work / "receipt.json"
    path.write_text(json.dumps(forged))
    return path


@pytest.mark.parametrize("kind", ["input_contracts", "exponent_contracts"])
@pytest.mark.parametrize("copies", ["base", "wrapper", "both"])
def test_valid_original_replacement_is_refused_even_after_all_outer_rehashes(
    tmp_path, lean_tools, monkeypatch, kind, copies
):
    config = prepared(tmp_path, lean_tools, integer=True)
    receipt, report = invoke(config, tmp_path / "receipt", lean_tools)
    original_receipt = _file_ref(tmp_path / "receipt/receipt.json")
    replacement = tmp_path / "different-original.json"
    if kind == "exponent_contracts":
        raw = json.loads(config[2].read_text())
        raw[0]["exponent_caps"][:2] = [254, 254]
        replacement.write_text(json.dumps(raw))
        # The substitution is a valid request, not a schema/environment failure.
        base = json.loads(Path(report["base_report"]["path"]).read_text())
        from merlin.verify.lean import rotary_policy

        before = json.loads(Path(base["program_exports"][0]["path"]).read_text())
        ordered_evidence.contracts(raw, rotary_policy._roster(before))
    else:
        raw = json.loads(config[1].read_text())
        raw[0]["intervals"][2]["upper"] = 20
        replacement.write_text(json.dumps(raw))
        bounded_numerics.decode(replacement.read_text())
    forged = forged_receipt(tmp_path / "forged", receipt, report, kind=kind, replacement=replacement, copies=copies)

    def must_not_run(*args, **kwargs):
        raise AssertionError("forged association reached the numerical checker")

    monkeypatch.setattr(bound, "check_w8a8_ordered_bound_draft", must_not_run)
    with pytest.raises(ValueError, match="byte identities|contract records differ"):
        bound.replay(forged, tmp_path / "replay", m2m_python=lean_tools.m2m_python)
    assert not (tmp_path / "replay/receipt.json").exists()
    assert _file_ref(tmp_path / "receipt/receipt.json") == original_receipt


@pytest.mark.parametrize("kind", ["input_contracts", "exponent_contracts"])
def test_original_change_after_fresh_check_prevents_replay_publication(tmp_path, lean_tools, monkeypatch, kind):
    config = prepared(tmp_path, lean_tools, integer=True)
    invoke(config, tmp_path / "receipt", lean_tools)
    changed_path = config[1] if kind == "input_contracts" else config[2]
    original = changed_path.read_bytes()
    implementation = bound.check_w8a8_ordered_bound_draft
    checks = []

    def change_after_check(*args, **kwargs):
        result = implementation(*args, **kwargs)
        assert result["boundary"] == "w8a8-ordered-contract-bound-candidate"
        checks.append(result)
        changed_path.write_bytes(original + b"\n")
        return result

    monkeypatch.setattr(bound, "check_w8a8_ordered_bound_draft", change_after_check)
    try:
        with pytest.raises(ValueError, match="file identity changed"):
            bound.replay(tmp_path / "receipt/receipt.json", tmp_path / "replay", m2m_python=lean_tools.m2m_python)
        assert len(checks) == 1
        assert not (tmp_path / "replay/receipt.json").exists()
    finally:
        changed_path.write_bytes(original)


@pytest.mark.parametrize("width", [32, 64])
def test_genuine_actual_pass_contract_bindings_replay(tmp_path, lean_tools, width):
    config = prepared(tmp_path, lean_tools, width=width)
    _, report = invoke(config, tmp_path / "receipt", lean_tools)
    replayed = bound.replay(tmp_path / "receipt/receipt.json", tmp_path / "replay", m2m_python=lean_tools.m2m_python)
    fresh = json.loads(Path(replayed["cert"]["path"]).read_text())
    assert fresh["evidence"] == report["evidence"]
    assert fresh["requested_prefix_checks_passed"]
