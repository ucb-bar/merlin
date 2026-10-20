"""Real guarded snapshot reads on the ordinary executor; no target qualification."""

import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback import compiler_profile_runtime as R
from test_compiler_profile_execution import execution as execution  # noqa: F401

from merlin.common import bundle_snapshot as B
from merlin.targetgen import package_runtime as P


def _command(execution, monkeypatch, *, during=None):
    owner, candidate, _work = execution
    incoming, output = candidate.parent / "input.mlir", candidate.parent / "output.json"
    incoming.write_text("module {}")
    package = SimpleNamespace(directory=candidate, integrity_exempt=False, manifest={})
    monkeypatch.setattr(P, "_resolve_argv", lambda _p, _n, inp, out: ["compiler", str(inp), str(out)])
    monkeypatch.setattr(P, "_needs_interpreter", lambda *_: False)
    calls = []

    def invoke(self, policy, package, argv, **kwargs):
        calls.append(B._ACTIVE_VERIFICATION.get())
        if during:
            during()
        Path(argv[-1]).write_text('{"actual":1}')
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(R._Executor, "_invoke", invoke)
    return lambda executor: executor.run_entrypoint(package, "emit", incoming, output), output, calls


def _entry(owner):
    return B.bundle_snapshot_root(owner.workspace)


def _payload(owner):
    return B.snapshot_path(_entry(owner), owner.repo / "input", owner.repo)


def _read_boundaries(monkeypatch):
    actual, boundaries = B.verify_bundle_snapshot, []

    def verify(*args, **kwargs):
        boundaries.append(B._ACTIVE_VERIFICATION.get() is None)
        return actual(*args, **kwargs)

    monkeypatch.setattr(B, "verify_bundle_snapshot", verify)
    return boundaries


def test_guarded_entry_reuses_only_entry_and_final_reads_stay_fresh(execution, monkeypatch):
    monkeypatch.setenv("MERLIN_SNAPSHOT_LEASE_CACHE", "1")
    owner, candidate, work = execution
    command, output, calls = _command(execution, monkeypatch)
    with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
        executor = P.active_package_executor()
        entry = executor.snapshot_entry_for_call()
        assert type(entry) is B.SharedEntry
        boundaries = _read_boundaries(monkeypatch)
        assert command(executor).returncode == 0
        assert boundaries == [False, True]  # Guarded entry, mandatory independent final.
        assert calls[0] is not None and output.exists()
        boundaries.clear()
        _payload(owner).touch()  # Even identical bytes end the guarded entry.
        assert command(executor).returncode == 0
        assert boundaries == [True, True]  # Invalid guard falls back to full reads.
    assert executor._snapshot_entry is None and entry.guard.fd is None
    assert B._ACTIVE_VERIFICATION.get() is None


def test_disabled_cache_keeps_legacy_fresh_reads(execution, monkeypatch):
    monkeypatch.setenv("MERLIN_SNAPSHOT_LEASE_CACHE", "0")
    owner, candidate, work = execution
    command, _, _ = _command(execution, monkeypatch)
    with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
        executor = P.active_package_executor()
        assert executor.snapshot_entry_for_call() is None
        boundaries = _read_boundaries(monkeypatch)
        assert command(executor).returncode == 0
        assert boundaries == [True, True]
    assert executor._snapshot_entry is None


def test_enclosing_emission_uses_original_entry_but_each_command_finishes_fresh(execution, monkeypatch):
    monkeypatch.setenv("MERLIN_SNAPSHOT_LEASE_CACHE", "1")
    owner, candidate, work = execution
    command, _, calls = _command(execution, monkeypatch)
    with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
        executor = P.active_package_executor()
        entry = executor.snapshot_entry_for_call()
        assert B._selected_entry() is entry
        boundaries = _read_boundaries(monkeypatch)
        with B.verified_snapshot_scope(owner.workspace, owner.bundle, repo=owner.repo):
            outer = B._ACTIVE_VERIFICATION.get()
            for _ in range(3):
                assert command(executor).returncode == 0
                assert B._ACTIVE_VERIFICATION.get() is outer
        assert boundaries == [False, False, True, False, True, False, True, True]
        assert len(set(calls)) == 3 and outer not in calls
    assert B._selected_entry() is None and B._ACTIVE_VERIFICATION.get() is None


def test_candidate_source_mutation_still_refuses_with_implicit_entry(execution, monkeypatch):
    monkeypatch.setenv("MERLIN_SNAPSHOT_LEASE_CACHE", "1")
    owner, candidate, work = execution

    def mutate():
        (candidate / "compiler.py").write_text("different actual compiler source")

    command, output, _ = _command(execution, monkeypatch, during=mutate)
    with pytest.raises(ValueError, match="another source revision"):
        with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
            with B.verified_snapshot_scope(owner.workspace, owner.bundle, repo=owner.repo):
                command(P.active_package_executor())
    assert not output.exists()
    assert B._selected_entry() is None and B._ACTIVE_VERIFICATION.get() is None


@pytest.mark.parametrize("changed", ["source", "profile", "workspace", "staging", "pid", "active_owner"])
def test_entry_cannot_cross_original_selection(execution, monkeypatch, changed):
    monkeypatch.setenv("MERLIN_SNAPSHOT_LEASE_CACHE", "1")
    owner, candidate, work = execution
    with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
        executor = P.active_package_executor()
        assert executor.snapshot_entry_for_call() is not None
        with monkeypatch.context() as patch:
            if changed == "source":
                patch.setattr(executor, "source_sha256", "another candidate")
            elif changed == "profile":
                patch.setattr(owner, "profile", {"different": True}, raising=False)
            elif changed == "workspace":
                patch.setattr(owner, "workspace", owner.workspace.parent / "another")
            elif changed == "staging":
                patch.setattr(executor, "work_root", executor.work_root / "another")
            elif changed == "pid":
                patch.setattr(R.os, "getpid", lambda: -1)
            else:
                patch.setattr(R, "active_original_execution", lambda: object())
            with pytest.raises(ValueError, match="original executor/selection/process"):
                executor.snapshot_entry_for_call()


def test_payload_change_during_command_refuses_before_publication_and_closes(execution, monkeypatch):
    monkeypatch.setenv("MERLIN_SNAPSHOT_LEASE_CACHE", "1")
    owner, candidate, work = execution

    def mutate():
        payload = _payload(owner)
        payload.chmod(0o600)
        payload.write_text("changed during the actual call")

    command, output, calls = _command(execution, monkeypatch, during=mutate)
    with pytest.raises(RuntimeError, match="content differs"):
        with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
            executor = P.active_package_executor()
            entry = executor.snapshot_entry_for_call()
            command(executor)
    assert calls and not output.exists()
    assert executor._snapshot_entry is None and entry.guard.fd is None
    assert P.active_package_executor() is None and B._ACTIVE_VERIFICATION.get() is None
    assert list(work.iterdir()) == []


def test_setup_failure_closes_guard_before_executor_publication(execution, monkeypatch):
    monkeypatch.setenv("MERLIN_SNAPSHOT_LEASE_CACHE", "1")
    owner, candidate, work = execution
    actual, entries = B.shared_entry_verification, []

    def opened(*args, **kwargs):
        entry = actual(*args, **kwargs)
        entries.append(entry)
        return entry

    @contextmanager
    def refuse(_service):
        raise RuntimeError("imports refused")
        yield

    monkeypatch.setattr(B, "shared_entry_verification", opened)
    monkeypatch.setattr(R, "original_compiler_imports", refuse)
    with pytest.raises(RuntimeError, match="imports refused"):
        with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
            pytest.fail("failed setup reached caller")
    assert len(entries) == 1 and entries[0].guard.fd is None
    assert P.active_package_executor() is None and list(work.iterdir()) == []


def test_direct_executor_constructor_does_not_allocate_unowned_guard(execution):
    owner, candidate, work = execution
    executor = R._Executor(owner, None, candidate, work)
    assert executor.snapshot_entry_for_call() is None
