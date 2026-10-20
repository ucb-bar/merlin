"""A copied correctness trajectory cannot become independent FP quality."""

import numpy as np
import pytest

from merlin.targetgen import bundle_gate as G


def _contract(tmp_path, observation):
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    result = {"version": 1, "steps": 2, "paper_ready": False}
    for channel, kind in (("correctness", "eager_same_precision"), ("quality", "eager_fp32")):
        path = tmp_path / f"{channel}.npz"
        np.savez(path, values=values)
        result[channel] = {
            "scope": "trajectory",
            "golden": path.name,
            "key": "values",
            "output_index": 0,
            "reference": kind,
            "reference_sha256": G.reference_digest(path, "values"),
        }
    if observation is not None:
        result["quality"]["observation"] = observation
    return result


OBSERVED = {
    "schema": "m2m.session_quality_observation.v1",
    "status": "observed",
    "origin": "provided_quality_reference",
}


def _gate(contract, tmp_path, channel="quality"):
    return G.gate_from_session_contract(
        contract, model="fixture", datapath="w8a8", bundle_dir=tmp_path, atol=0, rtol=0, channel=channel
    )


def test_independent_identical_values_on_synthetic_input_are_valid_quality(tmp_path):
    contract = _contract(tmp_path, OBSERVED)
    assert contract["correctness"]["reference_sha256"] == contract["quality"]["reference_sha256"]
    gate = _gate(contract, tmp_path)
    assert gate.channel == "quality" and gate.reference_kind == "eager_fp32"
    assert gate.steps == 2 and gate.output_elements == 3


@pytest.mark.parametrize(
    "observation",
    [
        None,
        {},
        {**OBSERVED, "status": "unavailable", "origin": "correctness_fallback"},
        {**OBSERVED, "origin": "correctness_fallback"},
        {**OBSERVED, "schema": "unknown"},
        {**OBSERVED, "qualified": True},
        True,
        "observed",
    ],
)
def test_unknown_fallback_or_malformed_quality_provenance_refuses(tmp_path, observation):
    contract = _contract(tmp_path, observation)
    contract["paper_ready"] = True  # This cannot issue reference provenance.
    with pytest.raises(G.GateError, match="independent FP quality is unavailable or unknown"):
        _gate(contract, tmp_path)
    assert _gate(contract, tmp_path, "correctness").channel == "correctness"


def test_explicitly_unavailable_quality_does_not_invent_a_tolerance_floor(tmp_path):
    contract = _contract(tmp_path, {**OBSERVED, "status": "unavailable", "origin": "correctness_fallback"})
    np.savez(tmp_path / "quality.npz", values=np.full((2, 3), 100, dtype=np.float32))
    gate = _gate(contract, tmp_path, "correctness")
    assert "own two references" not in gate.scope_note


def test_observed_quality_still_verifies_actual_array_digest(tmp_path):
    contract = _contract(tmp_path, OBSERVED)
    np.savez(tmp_path / "quality.npz", values=np.full((2, 3), 100, dtype=np.float32))
    with pytest.raises(G.GateError, match="digests to"):
        _gate(contract, tmp_path)


def test_an_unknown_unavailable_marker_cannot_bypass_legacy_correctness_floor(tmp_path):
    contract = _contract(tmp_path, {"schema": "unknown", "status": "unavailable"})
    np.savez(tmp_path / "quality.npz", values=np.full((2, 3), 100, dtype=np.float32))
    with pytest.raises(G.GateError, match="tighter than this capture"):
        _gate(contract, tmp_path, "correctness")
