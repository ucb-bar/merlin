"""Real child processes retain one queue grant and the shared CPU lock namespace."""

import json
import os
import selectors
import subprocess
import sys
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import worker_resources as W
from merlin_experiments.phase2.contracts import StageGateError
from merlin_experiments.phase2.evaluation_pool import ComponentEvaluationPool


def test_worker_environment_narrows_all_children_without_reserving_parent_slots(tmp_path):
    cpus = tuple(sorted(os.sched_getaffinity(0)))[:2]
    inherited = {
        "MERLIN_COMPUTE_CPUS": ",".join(map(str, cpus)),
        "MERLIN_COMPUTE_CPU_SLOTS": str(tmp_path / "shared"),
        "MERLIN_PREPARATION_CPUS": ",".join(map(str, cpus)),
        "OPENBLAS_NUM_THREADS": "128",
    }
    selected = W.environment(cpus[:1], temporary=tmp_path / "scratch", inherited=inherited)
    assert selected["MERLIN_COMPUTE_CPU_SLOTS"] == inherited["MERLIN_COMPUTE_CPU_SLOTS"]
    assert all(
        selected[key] == str(cpus[0]) for key in ("MERLIN_COMPUTE_CPUS", "MERLIN_PREPARATION_CPUS", "MERLIN_GRADE_CPUS")
    )
    assert selected["CMAKE_BUILD_PARALLEL_LEVEL"] == selected["OPENBLAS_NUM_THREADS"] == "1"
    assert inherited["OPENBLAS_NUM_THREADS"] == "128"
    assert not (tmp_path / "shared").exists()


def test_actual_subprocesses_and_nested_leases_keep_the_worker_cpu(tmp_path):
    cpus = tuple(sorted(os.sched_getaffinity(0)))[:2]
    env = dict(
        os.environ, MERLIN_COMPUTE_CPUS=",".join(map(str, cpus)), MERLIN_COMPUTE_CPU_SLOTS=str(tmp_path / "slots")
    )
    script = """import json,os,subprocess,sys
from pathlib import Path
from merlin_experiments.phase2.worker_resources import configure
from merlin.common.cpu_pool import shared_compute_pool,worker_pool_command,compute_capacity
configure((int(sys.argv[1]),),temporary=Path(sys.argv[2]))
before=sorted(os.sched_getaffinity(0))
with shared_compute_pool(timeout_s=2):
    during=sorted(os.sched_getaffinity(0))
child=[sys.executable,'-c','import json,os;print(json.dumps(sorted(os.sched_getaffinity(0))))']
plain=json.loads(subprocess.check_output(child,text=True))
nested=json.loads(subprocess.check_output(worker_pool_command(child,dict(os.environ)),text=True))
print(json.dumps([before,during,plain,nested,compute_capacity(),os.environ['MERLIN_COMPUTE_CPU_SLOTS']]))
"""
    result = subprocess.check_output(
        [sys.executable, "-B", "-c", script, str(cpus[-1]), str(tmp_path / "scratch")], env=env, text=True, timeout=10
    )
    assert json.loads(result) == [[cpus[-1]]] * 4 + [1, str(tmp_path / "slots")]


def test_worker_cannot_widen_shared_grant_and_legacy_gets_private_namespace(tmp_path):
    cpus = tuple(sorted(os.sched_getaffinity(0)))[:2]
    plain = W.environment(cpus, temporary=tmp_path / "scratch", inherited={})
    assert plain["MERLIN_COMPUTE_CPU_SLOTS"] == str(tmp_path / "scratch/cpu-slots")
    assert plain["CMAKE_BUILD_PARALLEL_LEVEL"] == str(len(cpus))
    if len(cpus) == 2:
        with pytest.raises(StageGateError, match="exceeds"):
            W.environment(
                cpus,
                temporary=tmp_path / "scratch",
                inherited={"MERLIN_COMPUTE_CPUS": str(cpus[0]), "MERLIN_COMPUTE_CPU_SLOTS": str(tmp_path / "slots")},
            )
    with pytest.raises(ValueError, match="shared compute"):
        W.environment(cpus, temporary=tmp_path / "scratch", inherited={"MERLIN_COMPUTE_CPUS": str(cpus[0])})


def test_shared_lock_metadata_is_an_explicit_writable_resource_owner(tmp_path, monkeypatch):
    slots = tmp_path / "external-slots"
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", str(min(os.sched_getaffinity(0))))
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(slots))
    pool = SimpleNamespace(
        root=tmp_path / "jobs",
        revisions=SimpleNamespace(root=tmp_path / "revisions"),
        policy=SimpleNamespace(candidate=tmp_path / "candidate", receipt_path=tmp_path / "receipts/file.json"),
        inputs=SimpleNamespace(analytical=None, rtl=None),
        _external_boundary=None,
    )
    assert slots in ComponentEvaluationPool._resource_writable_roots(pool, tmp_path / "leases/queue.json")


def test_actual_workers_overlap_on_disjoint_cpus_and_share_contention(tmp_path):
    cpus = tuple(sorted(os.sched_getaffinity(0)))[:2]
    if len(cpus) < 2:
        pytest.skip("two assigned CPUs are needed to test real worker overlap")
    env = dict(
        os.environ, MERLIN_COMPUTE_CPUS=",".join(map(str, cpus)), MERLIN_COMPUTE_CPU_SLOTS=str(tmp_path / "slots")
    )
    script = """import json,os,sys
from pathlib import Path
from merlin_experiments.phase2.worker_resources import configure
from merlin.common.cpu_pool import shared_compute_pool
configure((int(sys.argv[1]),),temporary=Path(sys.argv[2]))
try:
    with shared_compute_pool(timeout_s=0):
        print(json.dumps(sorted(os.sched_getaffinity(0))),flush=True)
        if sys.argv[3]=='hold':
            assert sys.stdin.readline()=='release\\n'
except TimeoutError:
    print('blocked',flush=True)
"""
    holders = []
    try:
        for index, cpu in enumerate(cpus):
            child = subprocess.Popen(
                [sys.executable, "-B", "-c", script, str(cpu), str(tmp_path / f"hold-{index}"), "hold"],
                env=env,
                text=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            holders.append(child)
            with selectors.DefaultSelector() as ready:
                ready.register(child.stdout, selectors.EVENT_READ)
                assert ready.select(timeout=5), "worker did not publish its bounded lease observation"
                assert child.stdout.readline().strip() == json.dumps([cpu])
        # Both workers hold their actual, disjoint CPU lease before either is
        # released. A third process cannot bypass that same lock namespace.
        probe = subprocess.run(
            [sys.executable, "-B", "-c", script, str(cpus[0]), str(tmp_path / "probe"), "probe"],
            env=env,
            text=True,
            capture_output=True,
            timeout=5,
            check=True,
        )
        assert probe.stdout.strip() == "blocked"
        for child in holders:
            child.stdin.write("release\n")
            child.stdin.flush()
        for child in holders:
            assert child.wait(timeout=5) == 0
    finally:
        for child in holders:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=5)
