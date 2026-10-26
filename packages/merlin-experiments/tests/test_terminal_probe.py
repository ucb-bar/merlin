"""Readonly display telemetry: fixtures contain no agents or target execution."""
import hashlib
import json
import os

from merlin_experiments.tracking.terminal_probe import Sampler


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def stat(pid, ticks, last_cpu=0, start=10):
    fields = ['0'] * 40
    fields[0], fields[1], fields[11], fields[19], fields[36] = 'R', '1', str(ticks), str(start), str(last_cpu)
    return f'{pid} (fixture worker) ' + ' '.join(fields)


def fixture(tmp_path):
    task, proc, cgroup = (tmp_path / name for name in ('task', 'proc', 'cgroup'))
    run, pool = task / 'run', task / 'pool'
    run.mkdir(parents=True)
    pool.mkdir()
    argv = ['python', '-m', 'merlin_experiments.phase1']
    selection = {'run_id': 'fixture', 'run_dir': str(run), 'taskset_cpus': [0, 1],
                 'environment': {'MERLIN_COMPUTE_CPU_SLOTS': str(pool)}}
    path = task / 'selection.json'
    raw = json.dumps(selection)
    write(path, raw)
    owner = {'pid': 100, 'started_at_unix': 1, 'argv': argv,
             'selection': {'path': str(path), 'sha256': hashlib.sha256(raw.encode()).hexdigest()}}
    write(task / '.tmp/fixture/process/owner.json', json.dumps(owner))
    write(proc / '100/stat', stat(100, 10))
    write(proc / '100/cmdline', '\0'.join(argv) + '\0')
    write(proc / '100/cgroup', '0::/fixture\n')
    write(cgroup / 'fixture/cgroup.procs', '100\n200\n')
    write(proc / '100/task/100/stat', stat(100, 10))
    write(proc / '100/task/100/status', 'Cpus_allowed_list:\t0-1\n')
    write(proc / '200/cmdline', 'spike\0/private/target-capsule-bench/SECRET/generated/program.elf\0')
    write(proc / '200/task/201/stat', stat(201, 20))
    write(proc / '200/task/201/status', 'Cpus_allowed_list:\t0\n')
    write(proc / 'stat', 'cpu0 100 0 0 100 0 0 0 0\ncpu1 0 0 0 200 0 0 0 0\n')
    write(proc / 'locks', '')
    write(pool / 'cpu-0.lock', '')
    return task, proc, cgroup, run, pool


def test_core_delta_is_not_last_cpu_thread_delta_and_private_redaction(tmp_path, monkeypatch):
    task, proc, cgroup, run, pool = fixture(tmp_path)
    monkeypatch.setattr('merlin_experiments.tracking.terminal_probe.time.monotonic', lambda: 10.)
    sampler = Sampler(task, proc_root=proc, cgroup_root=cgroup)
    first = sampler.sample()
    assert first['cores'][0]['measured_utilization_pct'] is None
    assert not [j for j in first['jobs'] if j['kind'] == 'cpu_lease']  # File existence is not a held lease.
    write(proc / 'stat', 'cpu0 100 0 0 200 0 0 0 0\ncpu1 100 0 0 200 0 0 0 0\n')
    write(proc / '200/task/201/stat', stat(201, 120))  # last_cpu remains 0, but core0 measured idle.
    monkeypatch.setattr('merlin_experiments.tracking.terminal_probe.time.monotonic', lambda: 11.)
    sample = sampler.sample()
    assert [c['measured_utilization_pct'] for c in sample['cores']] == [0., 100.]
    assert next(t for t in sample['threads'] if t['tid'] == 201)['cpu_time_delta_s'] > 0
    assert 'SECRET' not in json.dumps(sample)
    assert sample['campaign']['selection_pin'] == {'present': True}
    private = Sampler(task, True, proc_root=proc, cgroup_root=cgroup).sample()
    assert next(t for t in private['threads'] if t['tid'] == 201)['capsule'] == 'SECRET'
    assert next(t for t in private['threads'] if t['tid'] == 100)['capsule'] is None


def test_inode_lock_and_setup_records_are_distinct_from_cpu_usage(tmp_path):
    task, proc, cgroup, run, pool = fixture(tmp_path)
    st = (pool / 'cpu-0.lock').stat()
    write(proc / 'locks', f'1: FLOCK ADVISORY WRITE 100 {os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino} 0 EOF\n')
    stage = run / '_qa_work/profile_package_execution/setup-timing/a.json'
    write(stage, json.dumps({'schema': 'merlin.compiler_setup_timing.v1', 'pid': 100, 'source_sha256': 'PRIVATE',
                            'stages': [{'stage': 'policy', 'status': 'running', 'started_ns': 1}]}))
    sampler = Sampler(task, proc_root=proc, cgroup_root=cgroup)
    sample = sampler.sample()
    assert sample['cores'][0]['lease_owner_pids'] == [100]
    assert sample['cores'][0]['measured_utilization_pct'] is None
    job = next(j for j in sample['jobs'] if j['kind'] == 'setup_stage')
    assert job['tid'] is None and job['capsule'] is None and job['source_sha256'] is None
    write(stage, json.dumps({'schema': 'merlin.compiler_setup_timing.v1', 'stages': [{'status': 'complete'}]}))
    assert not [j for j in sampler.sample()['jobs'] if j['kind'] == 'setup_stage']


def test_changed_pid_identity_does_not_read_unrelated_cgroup(tmp_path):
    task, proc, cgroup, run, pool = fixture(tmp_path)
    write(proc / '100/cmdline', 'unrelated\0')
    sample = Sampler(task, proc_root=proc, cgroup_root=cgroup).sample()
    assert sample['status'] == 'owner_not_running'
    assert sample['threads'] == [] and sample['campaign']['cgroup'] is None


def test_foreign_selection_or_missing_records_refuse_without_writes(tmp_path):
    task, proc, cgroup, run, pool = fixture(tmp_path)
    write(task / 'selection.json', '{}')
    sample = Sampler(task, proc_root=proc, cgroup_root=cgroup).sample()
    assert sample['status'] == 'selection_unavailable'
    assert sample['read_only'] is True


def test_generic_declared_engine_and_structural_waiter_fields(tmp_path):
    from merlin_experiments.tracking.terminal_probe import classify
    assert classify(['/selected/emulator'], {'environment': {'MERLIN_GSIM_EMU_TARGET_X': '/selected/emulator'}})[0] == 'L3'
    assert classify(['/selected/emulator'], {'environment': {'MERLIN_TARGET_X_GSIM_EMU': '/selected/emulator'}})[0] == 'L3'
    task, proc, cgroup, run, pool = fixture(tmp_path)
    write(pool / 'waiters/2-1234-100-101-deadbeef.wait', '{"cpus": [0, 1]}')
    write(pool / 'waiters/2-1234-100-101-nothex.wait', '{}')
    write(run / 'environment.yaml', 'paths:\n  - /a/agent-workspaces/phase1/target/arm/id/workspace\n')
    sample = Sampler(task, proc_root=proc, cgroup_root=cgroup).sample()
    waiters = [j for j in sample['jobs'] if j['kind'] == 'cpu_waiter']
    assert len(waiters) == 1 and waiters[0]['tid'] == 101
    assert waiters[0]['state'] == 'ticket_liveness_unknown'
    assert sample['campaign']['workspace'] == '/a/agent-workspaces/phase1/target/arm/id/workspace'


def test_optional_setup_identity_associates_only_current_named_thread(tmp_path):
    import time
    task, proc, cgroup, run, pool = fixture(tmp_path)
    with (proc / 'stat').open('a') as f:
        f.write('btime 100\n')
    stage_path = run / '_qa_work/profile_package_execution/setup-timing/current.json'
    doc = {'schema': 'merlin.compiler_setup_timing.v1', 'pid': 100, 'native_tid': 999,
           'capsule': 'SECRET', 'tier': 'L2', 'source_sha256': 'a' * 64,
           'stages': [{'stage': 'invocation_snapshot.enter', 'native_tid': 100,
                       'status': 'running', 'started_ns': time.time_ns() - 1000000000}]}
    write(stage_path, json.dumps(doc))
    public = Sampler(task, proc_root=proc, cgroup_root=cgroup).sample()
    row = next(t for t in public['threads'] if t['tid'] == 100)
    assert row['action'] == 'verification' and row['stage'] == 'invocation_snapshot.enter'
    assert row['capsule'] is None and row['source_sha256'] is None
    assert 'SECRET' not in json.dumps(public)
    private = Sampler(task, True, proc_root=proc, cgroup_root=cgroup).sample()
    row = next(t for t in private['threads'] if t['tid'] == 100)
    assert row['capsule'] == 'SECRET' and row['action'] != 'L2'
    assert next(t for t in private['threads'] if t['tid'] == 201)['action'] == 'L2_or_cycle_observer'
    # A reused TID, missing source identity, or missing optional TID cannot attach.
    for change in ({'started_ns': 1}, {'native_tid': 999}):
        altered = json.loads(json.dumps(doc))
        altered['stages'][0].update(change)
        write(stage_path, json.dumps(altered))
        result = Sampler(task, True, proc_root=proc, cgroup_root=cgroup).sample()
        assert next(t for t in result['threads'] if t['tid'] == 100)['action'] == 'native_evaluator'
    doc['source_sha256'] = None
    write(stage_path, json.dumps(doc))
    result = Sampler(task, True, proc_root=proc, cgroup_root=cgroup).sample()
    assert next(t for t in result['threads'] if t['tid'] == 100)['action'] == 'native_evaluator'


def test_service_roles_do_not_claim_target_execution():
    from merlin_experiments.tracking.terminal_probe import classify
    for leaf in ('selfcheck', 'simjob', 'model_check', 'model_check_worker', 'isa_tools', 'cca'):
        action, basis = classify(['python3.12', '-B', '-m', 'merlin_experiments.phase1.brokers.' + leaf], {})
        assert action.startswith('host_') and action not in ('L3', 'L2_or_cycle_observer')
        assert 'stage unknown' in basis
    assert classify(['python3.12', '-m', 'merlin_experiments.phase1.feedback.selfcheck'], {})[0] == 'host_selfcheck_worker'
    assert classify(['python3.12', '/repo/src/merlin/common/read_lease_digest.py'], {})[0] == 'verification_lease_service'
    assert classify(['bwrap', '--', '/bin/spike'], {})[0] == 'host_sandbox_wrapper'
    assert classify(['/bin/codex-code-mode-host'], {})[0] == 'agent_tool_host'
    # Passing a module name as data must not select that service's role.
    assert classify(['python3', 'unrelated.py', 'merlin_experiments.phase1.brokers.simjob'], {})[0] == 'unknown'


def test_host_stage_roots_are_bounded_and_redacted(tmp_path):
    import time
    task, proc, cgroup, run, pool = fixture(tmp_path)
    with (proc / 'stat').open('a') as f:
        f.write('btime 100\n')
    stage_path = run / '_qa_work/fruns_1/host-stage-timing/setup-timing/one.json'
    doc = {'schema': 'merlin.compiler_setup_timing.v1', 'pid': 100, 'native_tid': 100,
           'source_sha256': 'a' * 64, 'capsule': 'SECRET', 'snapshot_id': 'PRIVATE-SNAPSHOT',
           'request_id': 'PRIVATE-REQUEST', 'stages': [{'stage': 'emission', 'status': 'running',
           'started_ns': time.time_ns() - 1000000000}]}
    write(stage_path, json.dumps(doc))
    sampler = Sampler(task, proc_root=proc, cgroup_root=cgroup, max_timing_records=1)
    # One root-discovery item then one file, never a recursive unbounded walk.
    first = sampler.sample()
    assert not [j for j in first['jobs'] if j['kind'] == 'setup_stage']
    sample = sampler.sample()
    job = next(j for j in sample['jobs'] if j['kind'] == 'setup_stage')
    assert job['stage'] == 'emission' and job['thread_association'] != 'unknown'
    assert 'PRIVATE' not in json.dumps(sample) and 'SECRET' not in json.dumps(sample)
    private = Sampler(task, True, proc_root=proc, cgroup_root=cgroup).sample()
    job = next(j for j in private['jobs'] if j['kind'] == 'setup_stage')
    assert job['snapshot_id'] == 'PRIVATE-SNAPSHOT' and job['request_id'] == 'PRIVATE-REQUEST'
    doc['stages'][0]['status'] = 'complete'
    write(stage_path, json.dumps(doc))
    assert not [j for j in sampler.sample()['jobs'] if j['kind'] == 'setup_stage']


def test_sleeping_thread_does_not_paint_idle_core_as_busy_unknown():
    from merlin_experiments.tracking.terminal_views import activity, dominant
    row = {'measured_utilization_pct': 0., 'threads': [
        {'state': 'S', 'action': 'unknown', 'cpu_time_delta_s': 0.}]}
    assert dominant(row) == {} and activity(row) == 'idle'
    row['measured_utilization_pct'] = 30.
    # Busy core without an observed active owner remains honestly unattributed.
    assert dominant(row) == {} and activity(row) == 'unknown'


def test_one_lock_inventory_serves_every_caller_until_it_ages(tmp_path, monkeypatch):
    # A run's read leases put 10^5 entries in /proc/locks and each read() of it re-walks the list:
    # re-reading it for every caller of every refresh stalled the console for a minute per refresh.
    from merlin_experiments.tracking import terminal_probe as P

    proc = tmp_path / 'proc'
    clock = [100.]
    monkeypatch.setattr(P.time, 'monotonic', lambda: clock[0])
    line = '1: FLOCK ADVISORY WRITE {pid} 08:01:77 0 EOF\n'
    write(proc / 'locks', line.format(pid=10))
    key = (os.makedev(8, 1), 77)
    assert P.locks(proc)[key][0]['pid'] == 10
    write(proc / 'locks', line.format(pid=20))
    clock[0] += P.LOCKS_MAX_AGE_S / 2
    assert P.locks(proc, wanted={key})[key][0]['pid'] == 10
    assert P.locks(proc, max_age_s=0)[key][0]['pid'] == 20
    write(proc / 'locks', line.format(pid=30))
    clock[0] += P.LOCKS_MAX_AGE_S + 1
    assert P.locks(proc)[key][0]['pid'] == 30
