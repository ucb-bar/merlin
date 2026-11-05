"""Existing-case readiness ordering; no new target execution or shared-failure guesses."""
import threading
from types import SimpleNamespace

import pytest

from merlin.targetgen import capsule_runner as C, capsule_scheduling as S


def cap(name, n):
    return {'name': name, 'kind': 'layer', 'label': 'public', 'inputs': [{'shape': [n]}]}


def test_small_probe_gates_only_its_route_and_is_not_executed_twice(monkeypatch):
    monkeypatch.delenv('MERLIN_CAPSULE_ORDER', raising=False)
    small, other = threading.Event(), threading.Event()
    calls = []
    caps = [cap('large', 100), cap('small', 1), cap('other', 2)]
    readiness = S.BoundaryReadiness({'large': 'routeA', 'small': 'routeA', 'other': 'routeB'})
    def run(c, workers):
        calls.append(c['name'])
        if c['name'] == 'small':
            small.set()
            assert other.wait(5)
        elif c['name'] == 'other':
            other.set()
            assert small.wait(5)
        else:
            assert readiness.completed_routes >= {'routeA'}
        return {'capsule': c['name'], 'status': 'pass', 'tiers': {'L2': {'status': 'pass'}}, 'evidence': c['name']}
    rows = S.run_pipelined(caps, run, max_workers=3, boundary_readiness=readiness)
    assert calls.count('small') == 1 and set(calls[:2]) == {'small', 'other'}
    assert len(rows) == 3
    probe = next(r for r in rows if r['capsule'] == 'small')
    assert probe['evidence'] == 'small' and probe['tiers']['L2']['status'] == 'pass'
    assert probe['boundary_readiness']['scope'] == 'case'
    assert probe['boundary_readiness']['cross_case_pass_credit'] is False


@pytest.mark.parametrize('category', ['UNSUPPORTED_OPERATION', 'FUNCTIONAL_MISMATCH', 'STRUCTURAL_INVARIANT_VIOLATION'])
def test_case_failure_cannot_suppress_other_members(category):
    seen = []
    def run(c, _workers):
        seen.append(c['name'])
        return {'capsule': c['name'], 'status': 'fail', 'failure': {'plane': 'target_to_command_buffer', 'category': category}}
    rows = S.run_pipelined([cap('large', 100), cap('small', 1)], run, max_workers=2,
        boundary_readiness=S.BoundaryReadiness({'large': 'route', 'small': 'route'}))
    assert seen == ['small', 'large'] and len(rows) == 2
    assert rows[1]['boundary_readiness']['route_global_failure_proven'] is False
    assert all(r['status'] == 'fail' for r in rows)  # Both actual per-case calls failed.


def test_slow_probe_releases_fanout_without_cancelling_or_claiming_pass():
    second = threading.Event()
    def run(c, _workers):
        if c['name'] == 'small':
            assert second.wait(5)
        else:
            second.set()
        return {'capsule': c['name'], 'status': 'fail'}
    readiness = S.BoundaryReadiness({'large': 'route', 'small': 'route'}, wait_s=0.02)
    rows = S.run_pipelined([cap('large', 100), cap('small', 1)], run, max_workers=2,
        boundary_readiness=readiness)
    assert len(rows) == 2 and all(r['status'] == 'fail' for r in rows)


def test_native_selection_binds_snapshot_manifest_profile_and_only_public_l2(monkeypatch):
    from merlin_experiments.phase1 import joint_control
    from merlin_experiments.phase1.feedback import compiler_profile_runtime as R
    monkeypatch.setattr(joint_control, 'selected', lambda: {})
    executor = SimpleNamespace(source_sha256='compilerA', owner=SimpleNamespace(
        snapshot_digest='snapshotA', profile={'sha256': 'profileA'}))
    monkeypatch.setattr(R, 'active_original_execution', lambda: executor)
    package = SimpleNamespace(manifest={'target': 'selected', 'commands': {'emit_target_bundle': {'argv': ['compiler']}}})
    caps = [cap('public', 1), {**cap('private', 1), 'label': 'hidden'},
            {**cap('model', 1), 'kind': 'model'}]
    def select():
        return C._candidate_boundary_readiness(caps, package, oracle_adapters={'L2': object()})
    initial = select()
    assert set(initial.routes) == {'public'}
    assert C._candidate_boundary_readiness(caps, package, oracle_adapters={'L3': object()}) is None
    assert C._candidate_boundary_readiness(caps, package, oracle_adapters={}) is None
    keys = {initial.routes['public']}
    for obj, field, value in [(executor, 'source_sha256', 'compilerB'),
                             (executor.owner, 'snapshot_digest', 'snapshotB'),
                             (executor.owner, 'profile', {'sha256': 'profileB'}),
                             (package, 'manifest', {'commands': {'emit_target_bundle': {'argv': ['changed']}}})]:
        setattr(obj, field, value)
        keys.add(select().routes['public'])
    assert len(keys) == 5
    assert not select().completed_routes  # Never shares prior snapshot pass state.


def test_production_suite_uses_existing_l2_case_result_once(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import joint_control
    from merlin_experiments.phase1.feedback import compiler_profile_runtime as R
    monkeypatch.setattr(joint_control, 'selected', lambda: {})
    owner = SimpleNamespace(source_sha256='immutable', owner=SimpleNamespace(
        snapshot_digest='frozen-inputs', profile={'sha256': 'selected-profile'}))
    monkeypatch.setattr(R, 'active_original_execution', lambda: owner)
    monkeypatch.setenv('MERLIN_OPERATOR_GRADE_PROCESSES', '0')
    monkeypatch.setattr(C, 'load_package', lambda *a, **k: SimpleNamespace(manifest={}))
    monkeypatch.setattr(C, 'integrity_scan', lambda p: None)
    monkeypatch.setattr(C, 'build_package', lambda p: None)
    monkeypatch.setattr(C, '_split_ineligible', lambda caps, target: (caps, []))
    monkeypatch.setattr(C._tier_policy, 'covering_set', lambda caps: [])
    monkeypatch.setattr(C, '_CALIBRATION_CAP', 0)
    seen = []
    def native(c, *args, **kwargs):
        assert set(kwargs['oracle_adapters']) == {'L2'}
        seen.append(c['name'])
        return {'capsule': c['name'], 'status': 'pass', 'tiers': {'L2': {'status': 'pass'}}}
    monkeypatch.setattr(C, 'run_capsule', native)
    rows = C._run_suite([cap('large', 100), cap('small', 1)], tmp_path,
        runs_root=tmp_path/'runs', max_workers=2, oracle_adapters={'L2': object()})
    assert seen == ['small', 'large']
    assert len(rows) == 2 and rows[1]['boundary_readiness']['result_reused']
