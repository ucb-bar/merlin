"""Counts transport can complete while timing stays UNKNOWN; synthetic fixtures."""

import hashlib
import json

import pytest
from merlin_experiments.phase2 import _public_native_observation as OBS
from merlin_experiments.phase2 import public_native_exploration as N
from merlin_experiments.phase2.contracts import StageGateError
from test_ordinary_native_archive_screen import ordinary_inputs as ordinary_inputs
from test_ordinary_native_archive_screen import source as source
from test_public_native_exploration import selection


def test_explicit_count_selection_is_owned_versioned_and_cannot_change(source):
    owner = N.prepare_public_native_exploration(selection(source), count_protocol=N.UNQUALIFIED_COUNT_PROTOCOL)
    assert owner.record()["schema"] == "merlin.public_native_exploration.v3"
    assert owner.record()["promotion_authority"] == "NONE"
    object.__setattr__(owner, "count_protocol", None)
    with pytest.raises(StageGateError, match="original issued owners"):
        owner.verify()


@pytest.mark.parametrize("protocol", [True, {}, "implicit", ""])
def test_unknown_count_protocol_cannot_issue_owner(source, protocol):
    with pytest.raises(StageGateError, match="exact explicit"):
        N.prepare_public_native_exploration(selection(source), count_protocol=protocol)


def test_original_counts_retained_without_claiming_uncovered_prediction(source, tmp_path, monkeypatch):
    elf = tmp_path / "original.elf"
    elf.write_bytes(b"neutral nonexecutable ELF stand-in; no target jobs")
    report = tmp_path / "counts.json"
    report.write_text(json.dumps({"fixture": "not qualified timing"}))
    calls = []
    monkeypatch.setattr(N.DOMAIN, "current", lambda *_a: {"status": "UNKNOWN", "reason": "capacity domain unqualified"})
    monkeypatch.setattr(N.NATIVE, "inference_case", lambda **_kw: pytest.fail("unqualified source must not predict"))

    def counts(actual, **kwargs):
        assert actual == elf and kwargs["numerical_pass"] is True
        calls.append(actual)
        return {"status": "observed", "report": {"path": str(report),
                "sha256": hashlib.sha256(report.read_bytes()).hexdigest()}}

    monkeypatch.setattr(N.COUNTS, "observe", counts)
    owner = N.prepare_public_native_exploration(selection(source), count_protocol=N.UNQUALIFIED_COUNT_PROTOCOL)
    value = {"functional_status": "PASS", "compiled": {"elf_sha256": hashlib.sha256(elf.read_bytes()).hexdigest()}}
    member = type("Member", (), {"source_sha256": "a" * 64})()
    result = OBS.observe(owner, None, member, tmp_path, value, sandbox=None, remaining_s=10,
                         include_unqualified_counts=True)
    assert calls == [elf]
    assert result["status"] == "UNKNOWN" and result["prediction"] is None
    assert result["promotion_authority"] == "NONE" and result["measured_cycles"] is None
    assert len(result["receipt_sha256"]) == 64
    assert "Fresh counts observed" in result["reason"]
    OBS.validate_projection(result, owner=owner)


def test_even_opted_in_counts_require_original_complete_functional_pass(source, tmp_path, monkeypatch):
    monkeypatch.setattr(N.COUNTS, "observe", lambda *_a, **_kw: pytest.fail("failed numerics cannot count"))
    owner = N.prepare_public_native_exploration(selection(source), count_protocol=N.UNQUALIFIED_COUNT_PROTOCOL)
    result = OBS.observe(owner, None, None, tmp_path, {"functional_status": "FAIL"}, sandbox=None,
                         remaining_s=10, include_unqualified_counts=True)
    assert result["status"] == "UNKNOWN" and result["receipt_sha256"] is None
