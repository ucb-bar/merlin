"""Synthetic count transport only; no target, timing or numerical qualification."""

import hashlib
import json
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import _public_native_observation as OBS
from merlin_experiments.phase2 import public_native_exploration as N
from merlin_experiments.phase2.contracts import StageGateError
from test_ordinary_native_archive_screen import ordinary_inputs as ordinary_inputs
from test_ordinary_native_archive_screen import source as source
from test_public_native_exploration import selection


def _observation(source, tmp_path, monkeypatch, *, refusal, mutation=None):
    elf = tmp_path / "original.elf"
    elf.write_bytes(b"neutral executable stand-in; never run")
    report = tmp_path / "counts.json"
    report.write_text(json.dumps({"fixture": "original neutral count report"}))
    report_bytes = report.read_bytes()
    receipt = {
        "status": "observed",
        "report": {"path": str(report), "sha256": hashlib.sha256(report_bytes).hexdigest()},
    }
    calls = []
    original_activity = OBS.REVIEW.domain_reason

    def counts(actual, **kwargs):
        assert actual == elf and kwargs["numerical_pass"] is True
        calls.append(actual)
        return receipt

    def domain(*_args):
        return (
            {"status": "UNKNOWN", "reason": "unsupported source"}
            if refusal == "source"
            else {"status": "source_domain_supported", "family": "neutral"}
        )

    def activity(actual, ranges):
        # Preparing the original owner must still review its archive normally.
        if actual.get("fixture") != "original neutral count report":
            return original_activity(actual, ranges)
        if mutation == "receipt":
            path = tmp_path / "public_native_counts_receipt.json"
            path.write_bytes(path.read_bytes() + b" ")
        elif mutation == "owner":
            object.__setattr__(owner, "catalog_json", b"{}")
        elif mutation == "source":
            source["observations_path"].write_text("changed selected original observations")
        return "raw activity outside TRAIN" if refusal == "activity" else None

    def inference(**_kwargs):
        assert refusal == "prediction", "a refused source or activity must not predict"
        return {}

    monkeypatch.setattr(OBS.DOMAIN, "current", domain)
    monkeypatch.setattr(OBS.COUNTS, "observe", counts)
    monkeypatch.setattr(OBS.REVIEW, "domain_reason", activity)
    monkeypatch.setattr(OBS.NATIVE, "inference_case", inference)
    monkeypatch.setattr(
        OBS.NATIVE,
        "compare",
        lambda *_a, **_kw: {
            "cases": [{"status": "UNKNOWN", "public_feedback_eligible": False, "reason": "event scope unresolved"}]
        },
    )
    owner = N.prepare_public_native_exploration(selection(source), count_protocol=N.UNQUALIFIED_COUNT_PROTOCOL)
    value = {"functional_status": "PASS", "compiled": {"elf_sha256": hashlib.sha256(elf.read_bytes()).hexdigest()}}
    policy = SimpleNamespace(
        search_corpus=SimpleNamespace(manifest_sha256="b" * 64),
        _checked_source_diagnostic=object(),
        _compiler_pass_selection=object(),
    )

    def observe():
        return OBS.observe(
            owner,
            policy,
            SimpleNamespace(source_sha256="a" * 64),
            tmp_path,
            value,
            sandbox=None,
            remaining_s=10,
            include_unqualified_counts=True,
        )

    return observe, owner, receipt, report, report_bytes, calls


@pytest.mark.parametrize("refusal", ["source", "activity", "prediction"])
def test_every_post_count_abstention_retains_original_receipt(source, tmp_path, monkeypatch, refusal):
    observe, owner, receipt, report, before, calls = _observation(source, tmp_path, monkeypatch, refusal=refusal)
    result = observe()
    path = tmp_path / "public_native_counts_receipt.json"
    assert calls and report.read_bytes() == before
    assert json.loads(path.read_bytes()) == receipt
    assert result["receipt_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["status"] == "UNKNOWN" and result["prediction"] is None
    assert result["measured_cycles"] is None and result["promotion_authority"] == "NONE"
    OBS.validate_projection(result, owner=owner)


@pytest.mark.parametrize("mutation", ["receipt", "owner", "source"])
def test_post_count_refusal_rechecks_receipt_and_original_owner(source, tmp_path, monkeypatch, mutation):
    observe, *_ = _observation(source, tmp_path, monkeypatch, refusal="activity", mutation=mutation)
    with pytest.raises(StageGateError, match="count receipt changed|issued owners|evidence changed"):
        observe()


@pytest.mark.parametrize("receipt", [True, 1, {}, "", "a" * 63, "A" * 64, "g" * 64])
def test_unknown_projection_refuses_malformed_receipt_digest(source, receipt):
    owner = N.prepare_public_native_exploration(selection(source))
    value = OBS.unknown(owner, "unchanged abstention")
    value["receipt_sha256"] = receipt
    with pytest.raises(StageGateError, match="exact count receipt digest"):
        OBS.validate_projection(value, owner=owner)


@pytest.mark.parametrize("receipt", [None, "a" * 64])
def test_unknown_projection_retains_none_or_exact_digest_without_authority(source, receipt):
    owner = N.prepare_public_native_exploration(selection(source))
    value = OBS.unknown(owner, "unchanged abstention")
    value["receipt_sha256"] = receipt
    assert OBS.validate_projection(value, owner=owner) == value
    assert value["prediction"] is None and value["promotion_authority"] == "NONE"
