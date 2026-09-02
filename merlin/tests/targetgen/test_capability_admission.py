"""A sealed hidden pool may exclude only target-proven impossible operand dtypes."""
from __future__ import annotations

from merlin.targetgen import capsule_grade as CG


def test_arm4_hidden_capability_admission_matches_sealed_cardinalities():
    """Prove the held-out 11→10 boundary without publishing the excluded capsule's name."""
    from merlin.common.paths import repo_root
    from merlin.targetgen import capsule_runner as CR
    from merlin.targetgen.target_experiment import load_target_experiment

    root = repo_root()
    te = load_target_experiment(
        root / "merlin/experiments/capsule_bench/targets/gemmini/target_experiment.yaml")
    hidden = CR.discover_capsules(
        te.hidden_roots(), labels={"hidden"}, contract=str(root / "merlin/contract"))
    hidden_ops = [cap for cap in hidden if cap.get("kind") != "model"]
    _eligible, excluded = CR._split_ineligible(hidden_ops, te.target)
    excluded_count = len(excluded)
    admitted_count = len(hidden) - excluded_count

    assert te.hidden_expected_source_capsules == len(hidden) == 15
    assert te.hidden_expected_admitted_capsules == admitted_count == 14
    assert excluded_count == 1


def test_capability_admission_filters_before_the_suite_and_seals_counts(monkeypatch):
    source = [
        {"name": "H0", "kind": "isa"},
        {"name": "GH0", "kind": "model_slice"},
        {"name": "M0", "kind": "model"},
    ]
    seen = []
    results = [
        {"capsule": "H0", "kind": "isa", "label": "hidden", "status": "fail", "tiers": {}},
        {"capsule": "M0", "kind": "model", "label": "hidden", "status": "gated", "tiers": {}},
    ]
    monkeypatch.setattr(CG, "load_package", lambda *_a, **_k:
                        type("P", (), {"integrity_exempt": False})())
    monkeypatch.setattr(CG, "integrity_scan", lambda *_a, **_k: None)
    monkeypatch.setattr(CG, "build_package", lambda *_a, **_k: None)
    monkeypatch.setattr(CG, "source_experiment_env", lambda *_a, **_k: None)
    monkeypatch.setattr(CG.CR, "discover_capsules", lambda *_a, **_k: list(source))
    monkeypatch.setattr(CG.CR, "_split_ineligible", lambda caps, _target: (
        [c for c in caps if c["name"] != "GH0"],
        [{"capsule": "GH0", "status": "not_graded"}],
    ))

    def run_suite(caps, *_args, **_kwargs):
        seen.extend(c["name"] for c in caps)
        return list(results)

    monkeypatch.setattr(CG.CR, "run_suite", run_suite)
    score = CG.grade("pkg", capsules_root=["root"], runs_root="runs", target="gemmini",
                     max_workers=1, capability_admission=True)

    assert seen == ["H0", "M0"]
    assert score["n_not_graded_ineligible"] == 0
    assert score["cohort_admission"] == {
        "version": 1,
        "policy": "frozen_target_capability_operand_dtype",
        "n_source_capsules": 3,
        "n_admitted_capsules": 2,
        "n_capability_excluded": 1,
        "n_resource_excluded": 0,
        "excluded_name_set_sha256": score["cohort_admission"]["excluded_name_set_sha256"],
        "admitted_name_set_sha256": score["cohort_admission"]["admitted_name_set_sha256"],
    }
    assert len(score["cohort_admission"]["excluded_name_set_sha256"]) == 64


def test_capability_admission_fails_open_when_hardware_cannot_prove_exclusion(monkeypatch):
    source = [{"name": "H0", "kind": "isa"}]
    monkeypatch.setattr(CG, "load_package", lambda *_a, **_k:
                        type("P", (), {"integrity_exempt": False})())
    monkeypatch.setattr(CG, "integrity_scan", lambda *_a, **_k: None)
    monkeypatch.setattr(CG, "build_package", lambda *_a, **_k: None)
    monkeypatch.setattr(CG, "source_experiment_env", lambda *_a, **_k: None)
    monkeypatch.setattr(CG.CR, "discover_capsules", lambda *_a, **_k: list(source))
    monkeypatch.setattr(CG.CR, "_split_ineligible", lambda caps, _target: (list(caps), []))
    monkeypatch.setattr(CG.CR, "run_suite", lambda caps, *_a, **_k: [
        {"capsule": caps[0]["name"], "kind": "isa", "label": "hidden", "status": "fail",
         "tiers": {}}
    ])
    score = CG.grade("pkg", capsules_root=["root"], runs_root="runs", target="gemmini",
                     max_workers=1, capability_admission=True)
    assert score["cohort_admission"]["n_source_capsules"] == 1
    assert score["cohort_admission"]["n_admitted_capsules"] == 1
    assert score["cohort_admission"]["n_capability_excluded"] == 0
