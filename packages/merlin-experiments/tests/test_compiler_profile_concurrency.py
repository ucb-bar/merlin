"""Actual original snapshot/admission ownership; process stubs grant no target proof."""

from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1.feedback import compiler_profile_runtime as R
from merlin_experiments.phase2 import campaign as C

from merlin.common import bundle_snapshot as B
from merlin.common import invocation_record as I
from merlin.common.paths import repo_root
from merlin.targetgen import package_runtime as P
from merlin.targetgen.sandbox import bwrap
from merlin.targetgen.tool_profile import COMMON_SUBSTRATE_V3, attach_common_substrate


@pytest.fixture
def original(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_SNAPSHOT_LEASE_CACHE", "1")
    repo = repo_root()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    bundle = {"arm": "raw_baseline", "tools": [], "allowed": [], "denied": [], "host_inputs": []}
    attach_common_substrate(bundle, version=COMMON_SUBSTRATE_V3, python_source_root=repo / "src", host_index_bits=64)
    for row in bundle["allowed"]:
        if row["path"].startswith("merlin/python/"):
            row["path"] = str(repo / "src" / row["path"].removeprefix("merlin/python/"))
    archive = tmp_path / "archive.yaml"
    archive.write_text(yaml.safe_dump(bundle))
    bwrap.materialize_bundle_inputs(workspace, bundle, repo=repo)
    owner = R.CommonTransformService(workspace=workspace, bundle_manifest=archive, repo=repo)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "compiler.py").write_text("# immutable synthetic compiler source\n")
    return owner, candidate, tmp_path / "staging"


def test_real_original_owner_two_jobs_have_thread_local_guard_and_final_reads(original, monkeypatch):
    owner, candidate, work = original
    barrier = threading.Barrier(2)
    seen, marker = [], ContextVar("independent_job", default=None)
    pkg = SimpleNamespace(directory=candidate, integrity_exempt=False, manifest={})
    script = "import pathlib,sys; pathlib.Path(sys.argv[2]).write_text(pathlib.Path(sys.argv[1]).read_text())"
    monkeypatch.setattr(P, "_resolve_argv", lambda _p, _n, inp, out: [sys.executable, "-c", script, str(inp), str(out)])
    monkeypatch.setattr(P, "_needs_interpreter", lambda *_: False)
    monkeypatch.setattr(
        C,
        "package_sandbox_policy",
        lambda target, workspace, package, **kwargs: C.PackageSandboxPolicy(
            ("box",), (), (), workspace, package, target, env_prefix=""
        ),
    )
    from merlin_experiments.phase1.brokers import compiler_call_channel

    @contextmanager
    def transport(*_a, **kw):
        yield SimpleNamespace(env_prefix=lambda prefix: prefix, remaining_timeout=lambda: kw["timeout_s"])

    monkeypatch.setattr(compiler_call_channel, "compiler_call_transport", transport)

    def invoke(self, policy, package, argv, **kwargs):
        seen.append((threading.get_ident(), Path(argv[-1]).parent, B._ACTIVE_VERIFICATION.get(), marker.get()))
        barrier.wait(timeout=10)
        return I.run(
            argv,
            directory=kwargs["evidence"],
            stage="original_profile_package",
            capture_output=True,
            text=True,
            timeout=10,
        )

    monkeypatch.setattr(R._Executor, "_invoke", invoke)
    original_verify, reads = B.verify_bundle_snapshot, []

    def verify(*args, **kwargs):
        if threading.current_thread() is not threading.main_thread():
            reads.append((threading.get_ident(), B._ACTIVE_VERIFICATION.get() is None))
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(B, "verify_bundle_snapshot", verify)
    with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
        executor = P.active_package_executor()
        assert executor.concurrent_entrypoints_v1() is True
        plan = executor.policy_scope.capsules.module_location_plan
        assert plan is not None
        entry = executor.snapshot_entry_for_call()
        assert entry is not None

        def job(index):
            assert B._selected_entry() is None  # A copied parent's thread selector has no authority.
            with executor.concurrent_entrypoint_scope_v1():
                assert B._selected_entry() is entry
                assert R.active_original_execution() is executor
                token = marker.set(index)
                try:
                    source, output = candidate.parent / f"in-{index}", candidate.parent / f"out-{index}"
                    source.write_text(json.dumps({"job": index}))
                    result = executor.run_entrypoint(
                        pkg, "emit", source, output, invocation_directory=candidate.parent / f"evidence-{index}"
                    )
                    assert result.returncode == 0 and json.loads(output.read_text()) == {"job": index}
                finally:
                    marker.reset(token)
            assert B._selected_entry() is None

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(copy_context().run, job, index) for index in range(2)]
            for future in futures:
                future.result(timeout=20)
        assert len({row[0] for row in seen}) == len({row[1] for row in seen}) == 2
        assert len({id(row[2]) for row in seen}) == 2 and {row[3] for row in seen} == {0, 1}
        assert sum(final for _thread, final in reads) >= 2  # Each command finishes independently.
    assert executor.concurrent_entrypoints_v1() is False
    assert executor._concurrent_binding is None and executor._snapshot_entry is None
    assert executor.policy_scope.capsules.module_location_plan is None
    with pytest.raises(RuntimeError, match="process lifetime"):
        with plan.batch(candidate):
            pytest.fail("closed original lifetime admitted declaration reuse")
    assert entry.guard.fd is None and P.active_package_executor() is None
    assert not list(work.iterdir())


@pytest.mark.parametrize("change", ["pid", "source", "profile", "workspace", "staging", "target", "inputs", "active"])
def test_concurrent_capability_cannot_cross_original_identity(original, monkeypatch, change):
    owner, candidate, work = original
    with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
        executor = P.active_package_executor()
        assert executor.concurrent_entrypoints_v1() is True
        with monkeypatch.context() as patch:
            if change == "pid":
                patch.setattr(R.os, "getpid", lambda: -1)
            elif change == "source":
                patch.setattr(executor, "source_sha256", "different")
            elif change == "profile":
                patch.setattr(owner, "profile", {"different": True})
            elif change == "workspace":
                patch.setattr(owner, "workspace", work / "other")
            elif change == "staging":
                patch.setattr(executor, "work_root", work / "other")
            elif change == "target":
                patch.setattr(executor, "target", object())
            elif change == "inputs":
                patch.setattr(executor, "inputs", object())
            else:
                patch.setattr(R, "active_original_execution", lambda: object())
            with pytest.raises(ValueError, match="process owner|lost their original"):
                executor.concurrent_entrypoints_v1()


def test_direct_constructor_and_expired_copied_context_do_not_advertise(original):
    owner, candidate, work = original
    direct = R._Executor(owner, None, candidate, work)
    assert direct.concurrent_entrypoints_v1() is False
    with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
        executor = P.active_package_executor()
        inherited = copy_context()
    assert inherited.run(executor.concurrent_entrypoints_v1) is False
    with pytest.raises(ValueError, match="live original lifetime"):
        with inherited.run(executor.concurrent_entrypoint_scope_v1):
            pytest.fail("expired lifetime admitted worker")


def test_worker_exception_clears_thread_selection_and_original_guard(original):
    owner, candidate, work = original
    with pytest.raises(RuntimeError, match="worker failed"):
        with R.original_compiler_execution(owner, target_experiment=None, candidate=candidate, work_root=work):
            executor = P.active_package_executor()
            entry = executor.snapshot_entry_for_call()
            with executor.concurrent_entrypoint_scope_v1():
                raise RuntimeError("worker failed")
    assert executor.concurrent_entrypoints_v1() is False
    assert B._selected_entry() is None and entry.guard.fd is None
    assert not list(work.iterdir())
