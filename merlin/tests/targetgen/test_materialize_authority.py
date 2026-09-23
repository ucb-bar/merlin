"""Installation is not certification, even when the source claims authority.

Exercise real materialization, selection and the promotion gate with synthetic
local inputs. No certification, successful promotion, build or publication runs.
"""

import json

import pytest

from merlin.common.yaml import write_yaml
from merlin.targetgen import publish as PB


@pytest.mark.parametrize("family", ["vector_schedule", "mlir_oot_target_backend"])
@pytest.mark.parametrize("status", sorted(PB.CERTIFIED_STATUSES))
@pytest.mark.parametrize("with_score", [False, True])
def test_materialization_does_not_inherit_authority(tmp_path, monkeypatch, family, status, with_score):
    target = "fixture_target"
    source = tmp_path / "source"
    source.mkdir()
    original_publication = {
        "champion": True,
        "certification": "pass",
        "certified_by": "historical_evaluator",
        "certified_by_run": "old-run",
        "certification_tier": {"cycle_accurate": True},
        "certified_rungs": [{"status": "pass"}],
        "fingerprint": "old-fingerprint",
        "measured": {"speedup": 2},
        "role": "baseline",
    }
    write_yaml(
        source / "manifest.yaml",
        {
            "target": target,
            "package_id": "source-package",
            "family": family,
            "status": status,
            "publication": original_publication,
        },
    )
    (source / "compiler.py").write_text("# synthetic package bytes\n")
    original = {p.name: p.read_bytes() for p in source.iterdir()}
    score_path = None
    if with_score:
        score_path = tmp_path / "score.json"
        score_path.write_text(
            json.dumps(
                {
                    "integrity_status": "clean",
                    "gradeable": True,
                    "n_passed": 1,
                    "n_capsules": 1,
                    "per_capsule": [{"capsule": "fixture", "status": "pass", "tiers": {"L2": "pass"}}],
                }
            )
        )
    artifacts = tmp_path / "artifacts"
    existing = artifacts / "targets" / target / "existing" / "manifest.yaml"
    existing.parent.mkdir(parents=True)
    write_yaml(existing, {"package_id": "existing", "family": family, "status": status})
    existing_bytes = existing.read_bytes()
    destination = PB.materialize_package(
        target,
        source,
        package_id="installed",
        certified_by_run="installation-run",
        score_path=score_path,
        artifacts_root=artifacts,
    )
    selected = PB.select_champion(target, package_id="installed", artifacts_root=artifacts)
    assert not PB._check_gate(selected)[0]
    manifest_before = (destination / "manifest.yaml").read_bytes()

    # If the gate regresses, stop before any promotion write rather than run one.
    def forbidden_write(*args, **kwargs):
        pytest.fail("uncertified installation reached a promotion write")

    monkeypatch.setattr(PB.package_records, "write_record", forbidden_write)
    with pytest.raises(PB.PublishError, match="promote gate refused"):
        PB.promote(target, "installed", artifacts_root=artifacts)
    assert (destination / "manifest.yaml").read_bytes() == manifest_before
    assert (destination / "manifest.yaml").read_bytes() == original["manifest.yaml"]
    installed = PB.package_records.read_record(destination)
    assert installed["publication"] == {"champion": False, "certification": "unverified"}
    assert installed["promotion"]["installed_from_run"] == "installation-run"
    assert installed["promotion"]["evidence"]["payload_binding"] == "unverified"
    assert installed["promotion"]["source_status"] == status
    assert installed["promotion"]["source_publication"] == original_publication
    assert {p.name: p.read_bytes() for p in source.iterdir()} == original
    assert (destination / "compiler.py").read_bytes() == original["compiler.py"]
    assert existing.read_bytes() == existing_bytes
    assert PB.select_champion(target, artifacts_root=artifacts).package_id == "existing"
