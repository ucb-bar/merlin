"""Separate operator provisioning never changes public admission or old budgets."""

import hashlib
import json
import os
import resource
import time
from contextlib import contextmanager

import pytest
from merlin_experiments.phase0 import operator_identity_resources as R


def selected(tmp_path, monkeypatch, mutation=None):
    root = tmp_path/'private'
    root.mkdir(mode=0o700)
    now = time.time_ns()
    data = dict(schema=R.SCHEMA, inventory_sha256='a'*64, start_unix_ns=now-10**9,
        deadline_unix_ns=now+899*10**9, cpu_ids=[6, 7], cpu_slot_root=str(tmp_path/'slots'),
        limits=dict(window_bytes=4<<20, aggregate_compact_bytes=192<<20, source_scratch_bytes=32<<20,
            rss_bytes=1<<30, source_cpu_seconds=120, source_wall_seconds=180, total_cpu_seconds=600,
            total_wall_seconds=900, source_bytes=5*(1<<29), total_source_bytes=6<<30))
    if mutation:
        mutation(data)
    path = root/'policy.json'
    path.write_text(json.dumps(data))
    path.chmod(0o600)
    monkeypatch.setenv('MERLIN_COMPUTE_CPUS', '6,7')
    monkeypatch.setenv('MERLIN_COMPUTE_CPU_SLOTS', str(tmp_path/'slots'))
    monkeypatch.setattr(resource, 'getrlimit', lambda key: ((600, 601) if key == resource.RLIMIT_CPU
                                                         else (32<<20, 32<<20)))
    return R.select(path, hashlib.sha256(path.read_bytes()).hexdigest(), progress=root/'progress.json'), data


def test_explicit_inventory_limits_and_shared_namespace_are_exact(tmp_path, monkeypatch):
    owner, data = selected(tmp_path, monkeypatch)
    owner.validate('a'*64, (4<<20, 192<<20, 5*(1<<29), 6<<30))
    for digest, limits in [('b'*64, (4<<20, 192<<20, 5*(1<<29), 6<<30)),
                           ('a'*64, (4<<20, 64<<20, 5*(1<<29), 6<<30))]:
        with pytest.raises(R.ResourceRefusal):
            owner.validate(digest, limits)
    assert owner.limits == data['limits']
    with pytest.raises(R.ResourceRefusal):
        R.OperatorIdentityResources(owner.path, owner.sha256, owner.progress, _token=None)
    assert json.loads(owner.progress.read_bytes())['status'] == 'selected'
    assert owner.progress.stat().st_mode & 0o077 == 0


@pytest.mark.parametrize('mutation', [
    lambda d: d.update(deadline_unix_ns=1),
    lambda d: d['limits'].update(source_cpu_seconds=True),
    lambda d: d.update(inventory_sha256='not a digest'),
    lambda d: d.update(cpu_ids=[6, 6]),
    lambda d: d.update(cpu_ids=[7]),
    lambda d: d.update(extra='silently ignored'),
])
def test_malformed_or_mismatched_provisioning_refuses(tmp_path, monkeypatch, mutation):
    with pytest.raises(R.ResourceRefusal):
        selected(tmp_path, monkeypatch, mutation)


def test_each_stage_releases_lease_and_records_original_cost_even_on_refusal(tmp_path, monkeypatch):
    owner, _ = selected(tmp_path, monkeypatch)
    events = []

    @contextmanager
    def lease(**kwargs):
        events.append(('acquire', kwargs['timeout_s']))
        try:
            yield
        finally:
            events.append(('release', None))

    monkeypatch.setattr(R, 'shared_compute_pool', lease)
    monkeypatch.setattr(R, 'current_thread_owns_cpu_lease', lambda: False)
    monkeypatch.setattr(resource, 'setrlimit', lambda *args: None)
    with owner.stage('source_identity', 0):
        assert json.loads(owner.progress.read_bytes())['status'] == 'active'
    with pytest.raises(ValueError, match='unsupported'):
        with owner.stage('source_identity', 1):
            raise ValueError('unsupported semantics')
    assert [event[0] for event in events] == ['acquire', 'release', 'acquire', 'release']
    assert [row['status'] for row in owner.rows] == ['complete', 'refused']
    assert all(row['cpu_seconds'] >= 0 and row['wall_seconds'] >= 0 for row in owner.rows)
    with pytest.raises(R.ResourceRefusal, match='did not complete'):
        owner.receipt()
    monkeypatch.setattr(R, 'current_thread_owns_cpu_lease', lambda: True)
    with pytest.raises(R.ResourceRefusal, match='unleased'):
        with owner.stage('source_identity', 2):
            pytest.fail('must not reuse a parent-held lease')


def test_original_policy_drift_cannot_complete_an_existing_owner(tmp_path, monkeypatch):
    owner, _ = selected(tmp_path, monkeypatch)
    owner.path.write_text('{}')
    with pytest.raises(R.ResourceRefusal, match='changed'):
        owner.receipt()
    assert os.stat(owner.progress).st_mode & 0o077 == 0


def test_original_issuer_uses_per_source_provisioning_and_never_leaves_partial_selection(tmp_path, monkeypatch):
    import test_operator_capture_exclusion as C
    from merlin_experiments.phase0 import operator_exclusion as E
    from merlin_experiments.phase0 import operator_exclusion_intake as I
    from merlin_experiments.phase0 import operator_streaming_exclusion_issuer as S
    from merlin_experiments.phase0 import operator_windowed_ir as W

    owner, _ = selected(tmp_path, monkeypatch)
    source = tmp_path/'original.mlir'
    source.write_bytes(C.source())
    roster = {str(source): dict(sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                               read_path=str(source), alias=None)}
    monkeypatch.setattr(I, 'resolve_streaming_v2', lambda *_: roster)
    monkeypatch.setattr(R, 'current_thread_owns_cpu_lease', lambda: False)
    monkeypatch.setattr(resource, 'setrlimit', lambda *args: None)

    @contextmanager
    def lease(**_kwargs):
        yield

    monkeypatch.setattr(R, 'shared_compute_pool', lease)
    path = owner.path.parent/'selection.json'
    kwargs = dict(forbidden_roots=[tmp_path/'public'], source_mode=C.C.SCHEMA, input_mode=W.SCHEMA,
        max_ir_bytes=4<<20, max_total_ir_bytes=192<<20, max_source_bytes=5*(1<<29), max_total_source_bytes=6<<30,
        operator_resources=owner)
    live, receipt = S.issue(tmp_path/'pointer', 'a'*64, path, **kwargs)
    assert type(live) is E.OperatorPrivateExclusion
    assert receipt['operator_resource_policy']['policy_sha256'] == owner.sha256
    assert [row['stage'] for row in owner.rows] == [
        'original_inventory', 'source_identity', 'final_original_inventory_and_source_bytes']
    assert json.loads(path.read_bytes())['schema'] == E.SCHEMA_V6
    monkeypatch.setattr(owner, 'receipt', lambda: (_ for _ in ()).throw(R.ResourceRefusal('cost refusal')))
    failed = owner.path.parent/'refused-selection.json'
    with pytest.raises(R.ResourceRefusal, match='cost refusal'):
        S.issue(tmp_path/'pointer', 'a'*64, failed, **kwargs)
    assert not failed.exists()
