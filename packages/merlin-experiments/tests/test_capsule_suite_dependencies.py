"""A screen may cite only a sibling already completed in this selected grade."""

import json
import threading

import pytest

from merlin.targetgen import capsule_runner as runner
from merlin.targetgen import tier_policy


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("anchor_passes", [True, False])
def test_suite_finishes_selected_sibling_before_screen(tmp_path, monkeypatch, workers, anchor_passes):
    capsules = [
        {"name": "a_extension", "extends": "z_anchor", "max_oracle_tier": "L2"},
        {"name": "b_independent"},
        {"name": "z_anchor"},
    ]
    monkeypatch.setattr(runner, "load_package", lambda *a, **k: object())
    monkeypatch.setattr(runner, "integrity_scan", lambda *a, **k: None)
    monkeypatch.setattr(runner, "build_package", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_split_ineligible", lambda caps, target: (caps, []))
    # The extension sorts first even after covering-set priority. Ordering it
    # ahead of its evidence source was the real sequential/parallel failure.
    monkeypatch.setattr(tier_policy, "covering_set", lambda caps: ["a_extension"])
    monkeypatch.setattr(tier_policy, "priced_tiers", lambda *a: [])
    monkeypatch.delenv("MERLIN_PASS_LOG", raising=False)
    finished = threading.Event()

    def run(cap, package, *, runs_root, **kwargs):
        if cap["name"] == "a_extension":
            assert finished.is_set(), "extension started before sibling finished"
            evidence = tier_policy.verify_extends("fixture", cap, "L2", declared_tiers=["L2", "L3"], roots=[runs_root])
            assert evidence.verified is anchor_passes
            if anchor_passes:
                assert evidence.tier == "L3"
        result = {
            "capsule": cap["name"],
            "status": "fail" if cap["name"] == "z_anchor" and not anchor_passes else "pass",
            "tiers": {"L3": {"status": "pass", "cycle_accurate": True}},
        }
        path = tmp_path / "runs" / cap["name"]
        path.mkdir(parents=True)
        (path / "capsule_result.json").write_text(json.dumps(result))
        if cap["name"] == "z_anchor":
            finished.set()
        return result

    monkeypatch.setattr(runner, "run_capsule", run)
    result = runner._run_suite(
        capsules, tmp_path / "package", runs_root=tmp_path / "runs", target="fixture", max_workers=workers
    )
    assert {row["capsule"] for row in result} == {cap["name"] for cap in capsules}
    assert len(result) == len(capsules)
    assert sum(row["status"] == "fail" for row in result) == (not anchor_passes)


def test_dependency_waves_preserve_every_member_and_unknown_siblings():
    capsules = [
        {"name": "first", "extends": "anchor"},
        {"name": "unknown", "extends": "not_selected"},
        {"name": "anchor"},
        {"name": "last", "extends": "first"},
    ]
    assert [[c["name"] for c in wave] for wave in runner._capsule_dependency_waves(capsules)] == [
        ["unknown", "anchor"],
        ["first"],
        ["last"],
    ]
    with pytest.raises(ValueError, match="cyclic"):
        runner._capsule_dependency_waves([{"name": "a", "extends": "b"}, {"name": "b", "extends": "a"}])
    with pytest.raises(ValueError, match="unique"):
        runner._capsule_dependency_waves([{"name": "a"}, {"name": "a"}])


@pytest.mark.parametrize('screen,l2_pass,attempted', [(True, True, True), (True, False, False), (False, True, False)])
def test_capstone_gate_uses_selected_screen_without_certifying_withheld_tier(
    tmp_path, monkeypatch, screen, l2_pass, attempted,
):
    caps = [
        {'name': 'op', 'kind': 'layer', 'required_oracle_tiers': ['L0', 'L1', 'L2', 'L3']},
        {'name': 'model', 'kind': 'model', 'gate': {'after_op_pass_fraction': 0.8}},
    ]
    monkeypatch.setattr(runner, 'load_package', lambda *a, **k: object())
    monkeypatch.setattr(runner, 'integrity_scan', lambda *a, **k: None)
    monkeypatch.setattr(runner, 'build_package', lambda *a, **k: None)
    monkeypatch.setattr(runner, '_split_ineligible', lambda caps, target: (caps, []))
    monkeypatch.setattr(runner, '_gate_counts', lambda *a: True)
    monkeypatch.setattr(tier_policy, 'covering_set', lambda caps: [])
    monkeypatch.setattr(tier_policy, 'priced_tiers', lambda *a: [])
    monkeypatch.delenv('MERLIN_PASS_LOG', raising=False)
    called = []

    def run(cap, *args, **kwargs):
        called.append(cap['name'])
        if cap['name'] == 'model':
            return {'capsule': 'model', 'status': 'incomplete'}
        return {
            'capsule': 'op', 'status': 'incomplete',
            'failure': {'plane': 'oracle_unavailable', 'category': 'NOT_RUN_IS_NOT_PASS',
                        'tier': 'L3', 'tier_status': 'unavailable'},
            'numeric': {'status': 'pass' if l2_pass else 'fail'},
            'tiers': {'L0': {'status': 'skipped', 'not_applicable': True},
                      'L1': {'status': 'skipped', 'not_applicable': True},
                      'L2': {'status': 'pass' if l2_pass else 'fail'}, 'L3': {'status': 'unavailable'}},
        }

    monkeypatch.setattr(runner, 'run_capsule', run)
    adapters = {'L2': lambda: None}
    if not screen:
        adapters['L3'] = lambda: None
    results = runner._run_suite(caps, tmp_path / 'package', runs_root=tmp_path / 'runs', oracle_adapters=adapters)
    assert ('model' in called) is attempted
    assert results[0]['status'] == 'incomplete'  # opening the screen never grants certification
