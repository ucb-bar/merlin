"""Independent captures overlap without crossing dependent writes or sharing evidence."""

import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor

from merlin_experiments.phase0.parallel_capture import ordered_jobs
from merlin_experiments.phase0.sealed_generation import SealedCaptureSource


def test_parallel_captures_keep_context_order_and_failure_ledger():
    selected = contextvars.ContextVar('selected', default=None)
    selected.set('pinned')
    barrier = threading.Barrier(2, timeout=5)
    completed = []
    entries = [{'name': str(i), 'source': 'pytorch'} for i in range(2)]
    entries += [{'name': 'dependent', 'kind': 'model'}]

    def write(entry):
        assert selected.get() == 'pinned'
        if entry['name'] == 'dependent':
            assert sorted(completed) == ['0', '1']
        else:
            barrier.wait()
            completed.append(entry['name'])
            if entry['name'] == '0':
                raise ValueError('original refusal')
        return entry['name']

    observed = []
    for entry, job in ordered_jobs(entries, write, workers=2):
        try:
            observed.append(job.result())
        except ValueError as exc:
            observed.append(str(exc))
    assert observed == ['original refusal', '1', 'dependent']


def test_capture_attestations_are_local_to_the_writer(tmp_path):
    source = SealedCaptureSource({'venv': str(tmp_path), 'm2m_root': str(tmp_path), 'runs_root': str(tmp_path/'runs')})
    barrier = threading.Barrier(2, timeout=5)

    def capture(identity):
        source.attestations.append({'identity': identity})
        barrier.wait()
        return source.attestations

    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(capture, identity) for identity in (1, 2)]
        assert [job.result() for job in jobs] == [[{'identity': 1}], [{'identity': 2}]]
    assert source.attestations == []


def test_parallel_staging_publishes_complete_distinct_members(tmp_path):
    from merlin_experiments.phase0.staging import write_staged

    barrier = threading.Barrier(4, timeout=5)

    def build(index):
        def write(root):
            member = root / 'ops' / str(index)
            member.mkdir(parents=True)
            (member/'capsule.yaml').write_text(f'name: member{index}\n')
            barrier.wait()
            (member/'golden.yaml').write_text('complete\n')
            return member
        return write_staged(write, tmp_path, member=f'ops/{index}')

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(build, range(4)))
    assert len(set(results)) == 4
    assert all((path/'golden.yaml').read_text() == 'complete\n' for path in results)
    assert not list(tmp_path.glob('.staging-*'))


def _roomy(monkeypatch):
    from merlin_experiments.phase0 import parallel_capture as PC

    monkeypatch.setattr(PC.os, "sched_getaffinity", lambda _pid: set(range(64)), raising=False)
    monkeypatch.setattr(PC, "_available_memory", lambda: 512 * 1024**3)
    monkeypatch.setattr(PC, "_cgroup_limits", lambda: (None, None))
    return PC


def test_worker_limit_default_cap_is_eight(monkeypatch):
    PC = _roomy(monkeypatch)
    monkeypatch.delenv(PC.WORKER_LIMIT_ENV, raising=False)
    assert PC.worker_limit() == 8


def test_worker_limit_env_raises_and_lowers_the_cap(monkeypatch):
    PC = _roomy(monkeypatch)
    monkeypatch.setenv(PC.WORKER_LIMIT_ENV, "32")
    assert PC.worker_limit() == 32
    monkeypatch.setenv(PC.WORKER_LIMIT_ENV, "3")
    assert PC.worker_limit() == 3


def test_worker_limit_env_never_lifts_cpu_or_memory_bounds(monkeypatch):
    PC = _roomy(monkeypatch)
    monkeypatch.setenv(PC.WORKER_LIMIT_ENV, "1000")
    assert PC.worker_limit() == 64
    monkeypatch.setattr(PC, "_available_memory", lambda: 10 * 1024**3)
    assert PC.worker_limit() == 5


def test_worker_limit_env_refuses_a_bad_value(monkeypatch):
    import pytest

    PC = _roomy(monkeypatch)
    for bad in ("0", "-2", "many", ""):
        monkeypatch.setenv(PC.WORKER_LIMIT_ENV, bad)
        with pytest.raises(ValueError, match=PC.WORKER_LIMIT_ENV):
            PC.worker_limit()
