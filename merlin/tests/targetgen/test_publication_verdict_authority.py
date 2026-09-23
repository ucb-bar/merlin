"""Publication cannot turn missing or contradictory evidence into authority."""

import pytest

from merlin.common.yaml import load_yaml, write_yaml
from merlin.targetgen import publish as P


@pytest.mark.parametrize("family", ["vector_schedule", "mlir_oot_target_backend"])
@pytest.mark.parametrize(
    "case",
    [
        "failed",
        "missing_oracle",
        "missing_target",
        "wrong_target",
        "missing_run",
        "unknown_kind",
        "oracle_failed",
        "untyped_tier",
    ],
)
def test_new_verdict_cannot_borrow_old_certified_status(tmp_path, family, case):
    artifacts = tmp_path / "artifacts"
    manifest = artifacts / "targets/fixture/package/manifest.yaml"
    manifest.parent.mkdir(parents=True)
    write_yaml(manifest, {"package_id": "package", "family": family, "status": "rtl_certified"})
    original = manifest.read_bytes()
    result = {
        "target": "fixture",
        "run_id": "test-run",
        "status": "pass",
        "oracle": {"kind": "synthetic", "result": "pass", "derived_from_rtl": False, "cycle_accurate": False},
    }
    if case == "failed":
        result["status"] = "fail"
        result["oracle"]["result"] = "fail"
    elif case == "missing_oracle":
        del result["oracle"]
    elif case == "missing_target":
        del result["target"]
    elif case == "wrong_target":
        result["target"] = "other"
    elif case == "missing_run":
        del result["run_id"]
    elif case == "unknown_kind":
        result["oracle"]["kind"] = "UNKNOWN"
    elif case == "oracle_failed":
        result["oracle"]["result"] = "fail"
    else:
        result["oracle"]["cycle_accurate"] = "false"
    evidence = tmp_path / "results.yaml"
    write_yaml(evidence, result)
    if case == "wrong_target":
        with pytest.raises(P.PublishError, match="target"):
            P.record_certification("fixture", "package", [evidence], artifacts_root=artifacts)
        assert manifest.read_bytes() == original
        return
    recorded = P.record_certification("fixture", "package", [evidence], artifacts_root=artifacts)
    assert recorded["certification"] == ("fail" if case == "failed" else "unverified")
    assert load_yaml(manifest)["status"] == "rtl_certified", "retain historical source status"
    selected = P.select_champion("fixture", package_id="package", artifacts_root=artifacts)
    assert not P._check_gate(selected)[0]
    with pytest.raises(P.PublishError, match="gate refused"):
        P.promote("fixture", "package", artifacts_root=artifacts)
