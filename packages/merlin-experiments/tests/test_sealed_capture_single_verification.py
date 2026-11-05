"""One fresh complete attestation per source proof; legacy defaults stay unchanged."""

import copy

import pytest
from merlin_experiments.phase0 import capture_execution_attestation as A
from merlin_experiments.phase0 import connected_integer_qualification as Q
from merlin_experiments.phase0 import sealed_generation as SG


def cap(*, materialized=False, source="pytorch", raw="a"):
    return dict(
        materialized_capture={} if not materialized else dict(receipt_sha256="r"),
        source=source,
        frontend_trace=dict(capture_mlir_sha256=raw),
        capture_execution_attestations=[
            dict(selection=dict(path="selection", sha256="s"), capture=dict(model_sha256="a"))
        ],
    )


@pytest.mark.parametrize("materialized,source", [(False, "pytorch"), (True, "pytorch"), (False, "other")])
def test_required_mode_verifies_actual_complete_attestation_once(monkeypatch, materialized, source):
    capsule = cap(materialized=materialized, source=source)
    calls = []
    monkeypatch.setattr(A, "require_verified_execution", lambda att: calls.append(att))
    assert SG.verified_capture_failure(capsule, require_attestation=True) is None
    assert calls == capsule["capture_execution_attestations"]


@pytest.mark.parametrize("capsule", [dict(materialized_capture=dict(x=1)), dict(source="other")])
def test_legacy_early_returns_remain_unchanged_and_stricter_mode_refuses_missing(capsule):
    assert SG.verified_capture_failure(capsule) is None
    assert "no sealed-runner attestation" in SG.verified_capture_failure(capsule, require_attestation=True)


@pytest.mark.parametrize("materialized", [False, True])
def test_required_mode_refuses_unverified_or_wrong_source_membership(monkeypatch, materialized):
    capsule = cap(materialized=materialized)

    def refuse(_):
        raise A.AttestationNotVerified("original complete runtime changed")

    monkeypatch.setattr(A, "require_verified_execution", refuse)
    assert SG.verified_capture_failure(capsule, require_attestation=True) == "original complete runtime changed"
    monkeypatch.setattr(A, "require_verified_execution", lambda _: None)
    capsule["frontend_trace"]["capture_mlir_sha256"] = "b"
    assert "not the attested capture" in SG.verified_capture_failure(capsule, require_attestation=True)


def test_complete_original_session_roster_is_verified_then_member_selected(monkeypatch):
    capsule = cap(materialized=True)
    capsule["capture_execution_attestations"][0]["capture"] = dict(
        kind="session", programs=[dict(model_sha256="a"), dict(model_sha256="b")]
    )
    calls = []
    monkeypatch.setattr(A, "require_verified_execution", lambda att: calls.append(copy.deepcopy(att)))
    assert SG.verified_capture_failure(capsule, require_attestation=True) is None
    assert len(calls) == 1 and len(calls[0]["capture"]["programs"]) == 2
    capsule["frontend_trace"]["capture_mlir_sha256"] = "c"
    assert "not the attested capture" in SG.verified_capture_failure(capsule, require_attestation=True)


@pytest.mark.parametrize("materialized,source", [(False, "pytorch"), (True, "pytorch"), (False, "other")])
def test_source_proof_reopens_exactly_once_and_next_call_refuses_original_drift(
    monkeypatch, tmp_path, materialized, source
):
    capsule = cap(materialized=materialized, source=source)
    calls = []

    def verify(att):
        calls.append(att)
        if len(calls) > 1:
            raise A.AttestationNotVerified("original live seal drift")

    monkeypatch.setattr(A, "require_verified_execution", verify)

    class StopAfterSeal(Exception):
        pass

    def stop(*_, **__):
        raise StopAfterSeal

    monkeypatch.setattr(Q.capture_selection, "load", stop)
    with pytest.raises(StopAfterSeal):
        Q.source_proof(capsule, tmp_path)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="original live seal drift"):
        Q.source_proof(capsule, tmp_path)
    assert len(calls) == 2


def test_source_proof_never_selects_a_partial_attestation_roster(monkeypatch, tmp_path):
    capsule = cap()
    capsule["capture_execution_attestations"] *= 2
    monkeypatch.setattr(A, "require_verified_execution", lambda _: pytest.fail("partial roster entered verification"))
    with pytest.raises(ValueError, match="requires one complete sealed source"):
        Q.source_proof(capsule, tmp_path)
