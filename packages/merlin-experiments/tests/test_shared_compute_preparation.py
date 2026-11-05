"""Bounded source preparation releases its compute slot before independent work."""

import fcntl
import hashlib
import os

from merlin.common.cpu_pool import shared_compute_pool
from merlin_experiments.phase1.brokers import selfcheck


def test_source_digest_is_leased_without_changing_digest_or_retaining_core(tmp_path, monkeypatch):
    cpu = min(os.sched_getaffinity(0))
    slots = tmp_path / "compute"
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", str(cpu))
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(slots))
    root = tmp_path / "source"
    root.mkdir()
    source = b"def function(): return 17\n"
    (root / "entry.py").write_bytes(source)
    (root / "build").mkdir()
    (root / "build" / "ignored.py").write_text("disposable")
    expected = hashlib.sha256(b"entry.py\0" + source).hexdigest()
    original_hash = selfcheck.hashlib.sha256
    entered = []

    def hashing(*args, **kwargs):
        with (slots / f"cpu-{cpu}.lock").open("r+") as probe:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                entered.append(True)
            else:
                raise AssertionError("source hashing was not leased")
        return original_hash(*args, **kwargs)

    monkeypatch.setattr(selfcheck.hashlib, "sha256", hashing)
    assert selfcheck._submission_digest(root) == expected
    assert entered == [True]
    with (slots / f"cpu-{cpu}.lock").open("r+") as probe:
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with shared_compute_pool(timeout_s=0):
        pass


def test_initial_grade_scan_and_discovery_release_before_build_and_suite(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pytest
    from merlin.targetgen import capsule_grade as grade

    cpu = min(os.sched_getaffinity(0))
    slots = tmp_path / "compute"
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", str(cpu))
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(slots))
    events = []

    def check(leased, name):
        events.append(name)
        with (slots / f"cpu-{cpu}.lock").open("r+") as probe:
            if leased:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def load(*args, **kwargs):
        check(True, "load")
        return SimpleNamespace(integrity_exempt=False)

    def discover(*args, **kwargs):
        check(True, "discovery")
        return [{"name": "selected", "kind": "op", "label": "public"}]

    class SuiteReached(Exception):
        pass

    def suite(*args, **kwargs):
        check(False, "suite")
        raise SuiteReached()

    monkeypatch.setattr(grade, "source_experiment_env", lambda _: {})
    monkeypatch.setattr(grade, "load_package", load)
    monkeypatch.setattr(grade, "integrity_scan", lambda *a, **kw: check(True, "scan"))
    monkeypatch.setattr(grade, "build_package", lambda *a: check(False, "build"))
    monkeypatch.setattr(grade, "_materialized_cohort_admission", lambda *a: None)
    monkeypatch.setattr(grade.CR, "discover_capsules", discover)
    monkeypatch.setattr(grade.CR, "run_suite", suite)
    with pytest.raises(SuiteReached):
        grade.grade(tmp_path / "package", capsules_root=tmp_path / "capsules", runs_root=tmp_path / "runs",
                    target="fixture", oracle_adapters={}, max_workers=1)
    assert events == ["load", "scan", "build", "discovery", "suite"]
