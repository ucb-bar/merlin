"""Grade-time binding between a materialized public cohort and its target descriptor."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from merlin.targetgen import capsule_grade as CG
from merlin.targetgen import corpora
from merlin.targetgen import target_experiment


def test_grade_rejects_cohort_materialized_from_another_descriptor(monkeypatch, tmp_path):
    """A valid-looking stale admission record must fail before any capsule is executed."""
    capsules = tmp_path / "capsules"
    capsules.mkdir()
    record = {
        "version": 1,
        "policy": "all_discovered",
        "n_source_capsules": 1,
        "n_admitted_capsules": 1,
        "n_capability_excluded": 0,
        "n_resource_excluded": 0,
        "excluded_name_set_sha256": CG._name_set_sha256([]),
        "admitted_name_set_sha256": CG._name_set_sha256(["A0"]),
        "descriptor_sha256": "a" * 64,
    }
    (capsules / ".cohort_admission.json").write_text(
        json.dumps(record), encoding="utf-8")
    descriptor = tmp_path / "target_experiment.yaml"
    descriptor.write_text("target: gemmini\n", encoding="utf-8")

    monkeypatch.setattr(CG, "source_experiment_env", lambda _target: [])
    monkeypatch.setattr(CG, "load_package", lambda *_args, **_kwargs:
                        SimpleNamespace(integrity_exempt=False))
    monkeypatch.setattr(CG, "integrity_scan", lambda _package: None)
    monkeypatch.setattr(CG, "build_package", lambda _package: None)
    monkeypatch.setattr(CG.CR, "discover_capsules", lambda *_args, **_kwargs: [
        {"name": "A0", "kind": "isa", "label": "public"},
    ])
    suite_called = False

    def run_suite(*_args, **_kwargs):
        nonlocal suite_called
        suite_called = True
        return []

    monkeypatch.setattr(CG.CR, "run_suite", run_suite)
    monkeypatch.setattr(corpora, "descriptor_path", lambda _target: descriptor)
    monkeypatch.setattr(
        target_experiment, "load_target_experiment",
        lambda path: SimpleNamespace(
            path=path, target="gemmini", descriptor_sha256="b" * 64))

    with pytest.raises(ValueError, match="does not match the target descriptor used for grade"):
        CG.grade(
            tmp_path / "package", capsules_root=capsules,
            runs_root=tmp_path / "runs", target="gemmini", max_workers=1)

    assert suite_called is False

