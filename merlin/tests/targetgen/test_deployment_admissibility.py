"""Target-neutral deployment profile, egress, and wrapper-order qualification."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from merlin.common.schemas import validate_or_raise
from merlin.perf.deployment_admissibility import (
    DeploymentAdmissibilityError,
    assess_deployment_admissibility,
    deployment_profile_sha256,
    require_deployment_admissible,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _inputs(tmp_path: Path):
    artifacts = {}
    for role in ("contract", "config", "runtime_header", "bitstream"):
        path = tmp_path / f"{role}.bin"
        path.write_text(f"synthetic vector engine {role}\n", encoding="utf-8")
        artifacts[role] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    profile = {
        "schema": "deployment_profile_v1",
        "artifacts": artifacts,
        "supported_physical_egress": [{"encoding": "signed_integer", "width_bits": 8}],
    }
    profile_sha = deployment_profile_sha256(profile)
    identity = {"compiler": _sha("compiler"), "emitted_program": _sha("program")}
    egress = {
        "schema": "physical_egress_evidence_v1",
        "profile_sha256": profile_sha,
        "emission_identity": identity,
        "derivation_status": "verified",
        "coverage_status": "complete",
        "egresses": [{
            "name": "result",
            "status": "verified",
            "emitted_representation": {"encoding": "signed_integer", "width_bits": 8},
            "physical_readout": {"encoding": "signed_integer", "width_bits": 8},
        }],
    }
    wrapper_path = tmp_path / "wrapper.ir"
    wrapper_path.write_text("synthetic non-accelerator-specific wrapper\n", encoding="utf-8")
    wrapper = {
        "schema": "wrapper_event_evidence_v1",
        "profile_sha256": profile_sha,
        "emission_identity": identity,
        "wrapper_artifact": {
            "path": wrapper_path.name,
            "sha256": hashlib.sha256(wrapper_path.read_bytes()).hexdigest(),
        },
        "derivation_status": "verified",
        "events": ["warm", "completion", "reset", "start", "measured", "completion",
                   "end", "validation"],
    }
    return profile, profile_sha, identity, egress, wrapper


def _assess(tmp_path: Path, profile, profile_sha, identity, egress, wrapper):
    return assess_deployment_admissibility(
        profile, expected_profile_sha256=profile_sha,
        expected_emission_identity=identity, egress_evidence=egress,
        wrapper_evidence=wrapper, profile_root=tmp_path, wrapper_root=tmp_path)


def _codes(result):
    return [row["code"] for row in result["ranked_actionable_diagnostics"]]


def test_synthetic_non_target_specific_profile_is_admitted(tmp_path):
    profile, profile_sha, identity, egress, wrapper = _inputs(tmp_path)
    validate_or_raise(profile, "deployment_profile")

    result = _assess(tmp_path, profile, profile_sha, identity, egress, wrapper)

    assert result["status"] == "admitted"
    assert result["admitted"] is True
    assert result["ranked_actionable_diagnostics"] == []
    assert all(row["status"] == "verified"
               for row in result["profile"]["artifact_bindings"].values())
    require_deployment_admissible(result, expected_profile_sha256=profile_sha)


def test_unsupported_physical_egress_is_refused(tmp_path):
    profile, profile_sha, identity, egress, wrapper = _inputs(tmp_path)
    egress["egresses"][0]["emitted_representation"]["width_bits"] = 32
    egress["egresses"][0]["physical_readout"]["width_bits"] = 32

    result = _assess(tmp_path, profile, profile_sha, identity, egress, wrapper)

    assert result["status"] == "refused"
    assert "unsupported_physical_egress" in _codes(result)
    with pytest.raises(DeploymentAdmissibilityError, match="unsupported_physical_egress"):
        require_deployment_admissible(result, expected_profile_sha256=profile_sha)


def test_evidence_identity_mismatch_is_refused(tmp_path):
    profile, profile_sha, identity, egress, wrapper = _inputs(tmp_path)
    wrapper["emission_identity"] = {**identity, "emitted_program": _sha("other-program")}

    result = _assess(tmp_path, profile, profile_sha, identity, egress, wrapper)

    assert result["status"] == "refused"
    assert "emission_identity_mismatch" in _codes(result)


def test_artifact_identity_mismatch_is_refused(tmp_path):
    profile, profile_sha, identity, egress, wrapper = _inputs(tmp_path)
    (tmp_path / "config.bin").write_text("changed config\n", encoding="utf-8")

    result = _assess(tmp_path, profile, profile_sha, identity, egress, wrapper)

    assert result["status"] == "refused"
    assert _codes(result)[0] == "artifact_identity_mismatch"


def test_wrapper_end_before_measured_completion_is_refused(tmp_path):
    profile, profile_sha, identity, egress, wrapper = _inputs(tmp_path)
    wrapper["events"] = ["warm", "completion", "reset", "start", "measured", "end",
                         "completion", "validation"]

    result = _assess(tmp_path, profile, profile_sha, identity, egress, wrapper)

    assert result["status"] == "refused"
    assert "measured_end_before_completion" in _codes(result)


def test_wrapper_warm_without_completion_is_refused(tmp_path):
    profile, profile_sha, identity, egress, wrapper = _inputs(tmp_path)
    wrapper["events"] = ["warm", "reset", "start", "measured", "completion", "end",
                         "validation"]

    result = _assess(tmp_path, profile, profile_sha, identity, egress, wrapper)

    assert result["status"] == "refused"
    assert "warm_without_completion" in _codes(result)


def test_missing_wrapper_proof_remains_unknown(tmp_path):
    profile, profile_sha, identity, egress, _ = _inputs(tmp_path)

    result = _assess(tmp_path, profile, profile_sha, identity, egress, None)

    assert result["status"] == "UNKNOWN"
    assert result["admitted"] is False
    assert "wrapper_evidence_missing" in _codes(result)


def test_incomplete_egress_derivation_cannot_omit_an_unsupported_result(tmp_path):
    profile, profile_sha, identity, egress, wrapper = _inputs(tmp_path)
    egress.pop("coverage_status")

    result = _assess(tmp_path, profile, profile_sha, identity, egress, wrapper)

    assert result["status"] == "UNKNOWN"
    assert "egress_coverage_incomplete" in _codes(result)


def test_validation_inside_compute_window_is_refused(tmp_path):
    profile, profile_sha, identity, egress, wrapper = _inputs(tmp_path)
    wrapper["events"] = ["warm", "completion", "reset", "start", "measured", "validation",
                         "completion", "end"]

    result = _assess(tmp_path, profile, profile_sha, identity, egress, wrapper)

    assert result["status"] == "refused"
    assert "validation_inside_measurement_window" in _codes(result)


def test_profile_identity_is_exact_and_stale_result_cannot_promote(tmp_path):
    profile, profile_sha, identity, egress, wrapper = _inputs(tmp_path)
    result = _assess(tmp_path, profile, profile_sha, identity, egress, wrapper)
    stale = copy.deepcopy(result)
    stale["profile"]["sha256"] = _sha("different profile")

    with pytest.raises(DeploymentAdmissibilityError, match="stale profile binding"):
        require_deployment_admissible(stale, expected_profile_sha256=profile_sha)
