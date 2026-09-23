"""Real isolated process transport, including explicitly guarded descendants."""

from __future__ import annotations

import importlib.util
import json
import os
import py_compile
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import pytest

from merlin.common.paths import module_source_path, python_import_roots, repo_root

SCRIPTS = Path("merlin/experiments/gemmini_perf_bench/scripts")
VERIFIER_SOURCE = module_source_path("merlin_experiments.source_snapshot")
MAIN_PROGRAM = """from __future__ import annotations
import __main__, pickle, sys
from dataclasses import dataclass, fields
from typing import ClassVar
@dataclass
class Record:
    class_only: ClassVar[int] = 2
    value: int = 3
assert __main__.Record is Record
assert [field.name for field in fields(Record)] == ['value']
assert pickle.loads(pickle.dumps(Record())) == Record()
assert sys.argv[1:] == ['argument']
assert sys.stdout.write_through
print('main-parity')
"""


def _module(name):
    if name == "perf_frozen_python":
        from merlin_experiments import frozen_python

        return frozen_python
    if name == "perf_snapshot":
        from merlin_experiments import source_snapshot

        return source_snapshot
    path = repo_root() / SCRIPTS / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(root, relative, text):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def frozen(tmp_path):
    snap, gateway = _module("perf_snapshot"), _module("perf_frozen_python")
    source, snapshot = tmp_path / "source", tmp_path / "snapshot"
    _write(
        source, "src/merlin/__init__.py", "from pkgutil import extend_path\n__path__=extend_path(__path__, __name__)\n"
    )
    _write(source, "src/merlin/core.py", "VALUE='frozen-core'\n")
    _write(
        source,
        "src/merlin/perf/__init__.py",
        "from pkgutil import extend_path\n__path__=extend_path(__path__, __name__)\n",
    )
    _write(
        source, "src/merlin/perf/execution_policy.py", (repo_root() / "src/merlin/perf/execution_policy.py").read_text()
    )
    _write(
        source,
        "packages/merlin-experiments/src/merlin/perf/analysis_worker.py",
        (repo_root() / "packages/merlin-experiments/src/merlin/perf/analysis_worker.py").read_text(),
    )
    _write(source, str(SCRIPTS / "trusted_stage.py"), "VALUE='actual'\n")
    _write(source, "packages/merlin-experiments/src/merlin_experiments/phase2/__init__.py", "")
    _write(source, "packages/merlin-experiments/src/merlin_experiments/phase2/stage_inputs.py", "")
    _write(source, "packages/merlin-experiments/src/merlin_experiments/phase2/emission_analysis.py", "VALUE='actual'\n")
    _write(source, "packages/merlin-experiments/src/merlin/optional.py", "VALUE='frozen-optional'\n")
    _write(source, "packages/merlin-experiments/src/merlin_experiments/__init__.py", "")
    _write(
        source,
        "packages/merlin-experiments/src/merlin_experiments/frozen_python.py",
        Path(gateway.__file__).read_text(),
    )
    _write(source, str(SCRIPTS / "main_identity.py"), MAIN_PROGRAM)
    _write(
        source,
        str(SCRIPTS / "run_global_perf_experiment.py"),
        "import json\ndef consume_global_candidate(path):\n import merlin.optional\n"
        " if json.loads(path.read_text()).get('reject'): import merlin.missing\n",
    )
    _write(
        source,
        str(SCRIPTS / "entry.py"),
        "import json,sys,os; print(json.dumps({'argv':sys.argv,'cwd':os.getcwd()}))\n",
    )
    snap.create(
        source,
        snapshot,
        output_root=tmp_path / "out",
        source_roots=("src", "packages/merlin-experiments/src", str(SCRIPTS)),
        python_roots=("packages/merlin-experiments/src", "src"),
        legacy_roots=("merlin/experiments/gemmini_perf_bench/scripts",),
    )
    yield snap, gateway, snapshot
    for path in [tmp_path, *tmp_path.rglob("*")]:
        if path.is_dir() and not path.is_symlink():
            path.chmod(0o700)


def _command(gateway, snapshot, native_argv):
    return gateway.python_command(snapshot, native_argv, verifier_source=VERIFIER_SOURCE)


def _run(command, **kwargs):
    return subprocess.run(command, capture_output=True, text=True, timeout=20, **kwargs)


def test_script_argv_cwd_and_optional_before_core(frozen, tmp_path):
    _, gateway, snapshot = frozen
    entry = snapshot / SCRIPTS / "entry.py"
    command = _command(gateway, snapshot, [sys.executable, str(entry), "value with spaces", "--flag"])
    result = _run(command, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"argv": [str(entry), "value with spaces", "--flag"], "cwd": str(tmp_path)}
    result = _run(
        _command(
            gateway,
            snapshot,
            [
                sys.executable,
                "-c",
                "import merlin.optional,merlin.core; print(merlin.optional.VALUE,merlin.core.VALUE)",
            ],
        )
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "frozen-optional frozen-core"


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_lazy_missing_owner_never_falls_back_in_children(frozen, tmp_path, monkeypatch, depth):
    _, gateway, snapshot = frozen
    live = tmp_path / "live"
    _write(live, "merlin/missing.py", "print('LIVE LEAK')\n")
    monkeypatch.syspath_prepend(str(live))
    code = f"import sys; sys.path.insert(0, {str(live)!r}); import merlin.optional; import merlin.missing"
    for _ in range(depth):
        argv = [sys.executable, "-c", code]
        code = (
            "import subprocess,sys; from merlin_experiments.frozen_python import inherited_python_command; "
            f"sys.exit(subprocess.call(inherited_python_command({argv!r})))"
        )
    result = _run(_command(gateway, snapshot, [sys.executable, "-c", code]))
    assert result.returncode != 0
    assert "module unavailable in frozen source receipt: merlin.missing" in result.stderr
    assert "LIVE LEAK" not in result.stdout


@pytest.mark.parametrize("kind", ["source", "seal", "instrumentation"])
def test_transport_rechecks_pinned_authority(frozen, kind):
    snap, gateway, snapshot = frozen
    command = _command(gateway, snapshot, [sys.executable, "-c", "print('should-not-run')"])
    if kind == "source":
        path = snapshot / "src/merlin/core.py"
    elif kind == "seal":
        path, _ = snap.load_seal(snapshot, "snapshot")
    else:
        document = json.loads(command[-1])
        document["context"]["resolver"]["sha256"] = "0" * 64
        command[-1] = json.dumps(document)
        path = None
    if path:
        path.chmod(0o600)
        path.write_text(path.read_text() + "\n")
        path.chmod(0o444)
    result = _run(command)
    assert result.returncode != 0
    assert "should-not-run" not in result.stdout


def test_module_entrypoint_and_nonzero_exit(frozen):
    _, gateway, snapshot = frozen
    result = _run(_command(gateway, snapshot, [sys.executable, "-m", "merlin.core"]))
    assert result.returncode == 0, result.stderr
    result = _run(_command(gateway, snapshot, [sys.executable, "-c", "raise SystemExit(23)"]))
    assert result.returncode == 23
    result = _run(_command(gateway, snapshot, [sys.executable, "-m", "json.tool"]))
    assert result.returncode != 0
    assert "not sealed source" in result.stderr


def test_unfrozen_child_preserves_original_argv(monkeypatch):
    gateway = _module("perf_frozen_python")
    monkeypatch.delenv(gateway.CONTEXT, raising=False)
    argv = [sys.executable, "-c", "print('ordinary')"]
    assert gateway.inherited_python_command(argv) == argv


def test_snapshot_root_does_not_depend_on_inherited_environment(frozen, tmp_path, monkeypatch):
    _, gateway, snapshot = frozen
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path / "live-checkout"))
    result = _run(
        _command(gateway, snapshot, [sys.executable, "-c", "import os; print(os.environ['MERLIN_REPO_ROOT'])"])
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(snapshot)


@pytest.mark.parametrize("inline", [False, True])
def test_real_main_identity_and_unbuffered_output(frozen, inline):
    _, gateway, snapshot = frozen
    target = ["-c", MAIN_PROGRAM] if inline else [str(snapshot / SCRIPTS / "main_identity.py")]
    result = _run(_command(gateway, snapshot, [sys.executable, "-u", *target, "argument"]))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "main-parity"


def test_instrumentation_fifo_refused_without_reading(tmp_path):
    gateway = _module("perf_frozen_python")
    fifo = tmp_path / "instrumentation"
    os.mkfifo(fifo)
    with pytest.raises(RuntimeError, match="ordinary file"):
        gateway._checked({"path": str(fifo), "sha256": "0" * 64})


def test_no_pth_execution_or_wrong_interpreter_site_dirs(frozen, tmp_path, monkeypatch):
    _, gateway, snapshot = frozen
    dependencies = tmp_path / "dependencies"
    marker = tmp_path / "should-not-exist"
    _write(dependencies, "evil.pth", f"import pathlib; pathlib.Path({str(marker)!r}).touch()\n")
    wrong = tmp_path / "lib/python0.1/site-packages"
    wrong.mkdir(parents=True)
    monkeypatch.syspath_prepend(str(wrong))
    monkeypatch.syspath_prepend(str(dependencies))
    command = _command(gateway, snapshot, [sys.executable, "-c", "print('no-site')"])
    assert str(wrong) not in json.loads(command[-1])["dependency_roots"]
    result = _run(command)
    assert result.returncode == 0, result.stderr
    assert not marker.exists()


@pytest.mark.parametrize("legacy_root_call", [False, True])
def test_historical_v1_requires_fresh_snapshot_without_rewriting_archive(tmp_path, monkeypatch, legacy_root_call):
    snap, gateway = _module("perf_snapshot"), _module("perf_frozen_python")
    source, snapshot = tmp_path / "historical-source", tmp_path / "historical-snapshot"
    _write(source, "merlin/python/merlin/__init__.py", "")
    _write(source, str(SCRIPTS / "old_native.py"), "VALUE=1\n")
    seal = snap.create(
        source,
        snapshot,
        output_root=tmp_path / "out",
        source_roots=("merlin/python", str(SCRIPTS)),
        python_roots=("merlin/python",),
        legacy_roots=(str(SCRIPTS),),
    )
    _, receipt = snap.load_seal(snapshot, "snapshot")
    receipt["schema"] = "merlin.performance-source-snapshot.v1"
    for key in ("python_roots", "legacy_roots", "legacy_names", "directories", "internal_aliases"):
        receipt.pop(key)
    snapshot.chmod(0o700)
    seal.unlink()
    original_seal = snap.seal(snapshot, "snapshot", receipt)
    snapshot.chmod(0o500)
    original_bytes = original_seal.read_bytes()
    try:
        if legacy_root_call:
            with pytest.raises(snap.SnapshotError, match="newly frozen"):
                snap.verify(snapshot)
        else:
            with pytest.raises(RuntimeError, match="newly frozen"):
                _command(gateway, snapshot, [sys.executable, "-c", "raise AssertionError('must not execute')"])
        assert original_seal.read_bytes() == original_bytes
        assert snap.load_seal(snapshot, "snapshot")[1] == receipt
    finally:
        for path in [snapshot, *snapshot.rglob("*")]:
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o700)


@pytest.mark.parametrize("missing", [False, True])
def test_chia_remote_task_executes_parent_pinned_transport_without_inherited_context(
    frozen, tmp_path, monkeypatch, missing, managed_native_endpoint
):
    _, gateway, snapshot = frozen
    # Fake only external task assignment; the public task body, receipt validation and
    # subprocess transport are real. No Ray service, agents or target hardware are used.
    chia = types.ModuleType("chia")
    trace = types.ModuleType("chia.trace")
    trace.__file__ = str(_write(tmp_path, "trace.py", "# synthetic external CHIA dependency\n"))
    chia.trace = trace
    decorator = types.ModuleType("chia.base.ChiaFunction")
    decorator.ChiaFunction = lambda **kwargs: lambda function: function
    monkeypatch.setitem(sys.modules, "chia", chia)
    monkeypatch.setitem(sys.modules, "chia.trace", trace)
    monkeypatch.setitem(sys.modules, "chia.base", types.ModuleType("chia.base"))
    monkeypatch.setitem(sys.modules, "chia.base.ChiaFunction", decorator)
    ray = types.ModuleType("ray")
    ray.get_runtime_context = lambda: types.SimpleNamespace(
        get_assigned_resources=lambda: {"codex_slots": 1, "gsim_slots": 1}
    )
    monkeypatch.setitem(sys.modules, "ray", ray)
    wrapper = _module("chia_agentic_perf_experiment")
    program = (
        "import merlin.optional; import os,json,hashlib; from pathlib import Path; "
        "payload=Path(os.environ['MERLIN_CHIA_LAUNCH_RECEIPT']).read_bytes(); "
        "assert hashlib.sha256(payload).hexdigest()==os.environ['MERLIN_CHIA_LAUNCH_RECEIPT_SHA256']; "
        "assert json.loads(payload)['plan_sha256']==os.environ['MERLIN_CHIA_ENVELOPE_PLAN_SHA256']"
    )
    native = [sys.executable, "-c", "import merlin.missing" if missing else program]
    transport = _command(gateway, snapshot, native)
    plan = {
        "wrapper": gateway._pin(Path(wrapper.__file__)),
        "chia_trace": gateway._pin(Path(trace.__file__)),
        "command_artifacts": wrapper.chia_launch.command_artifacts(native),
        "launch_policy": wrapper.chia_launch.policy_identity(),
        "transport_command": transport,
    }
    plan["sha256"] = wrapper.hashlib.sha256(wrapper._canonical(plan)).hexdigest()
    monkeypatch.delenv(gateway.CONTEXT, raising=False)
    from merlin_experiments.execution.chia_native import Session, cleanup, setup

    with Session(managed_native_endpoint) as session:
        invitation = session.reserve()
        setup(invitation)
        try:
            result = wrapper.run_coordinator(native, str(tmp_path), plan, str(tmp_path / "receipts"))
        finally:
            cleanup(invitation)
        lifecycle = session.receipt(invitation)
        assert lifecycle["guardian_reaped"] and lifecycle["cleanup_complete"]
        assert lifecycle["guardian"]["native_started"]
        assert lifecycle["guardian"]["returncode"] == result["returncode"]
    assert (result["returncode"] != 0) is missing
    launch = json.loads(Path(result["launch_receipt"]["path"]).read_text())
    assert launch["command"] == native
    assert launch["plan"]["transport_command"] == transport
    completion = json.loads(Path(result["completion_receipt"]["path"]).read_text())
    assert completion["status"] == ("failed" if missing else "complete")


@pytest.fixture
def managed_native_endpoint():
    """Only a provisioned AF_UNIX supervisor; never starts Ray or a TCP listener."""
    from merlin_experiments.execution import _protocol

    try:
        descriptor = _protocol.self_pidfd()
    except (OSError, RuntimeError) as error:
        pytest.skip(f"Linux pidfds unavailable: {error}")
    os.close(descriptor)
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, python_import_roots()))}
    environment.pop("MERLIN_FROZEN_PYTHON_CONTEXT", None)
    with tempfile.TemporaryDirectory(prefix="mpf-", dir="/tmp") as runtime:
        endpoint = Path(runtime) / "service.sock"
        process = subprocess.Popen(
            [sys.executable, "-m", "merlin_experiments.execution.native_supervisor", "--endpoint", str(endpoint)],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            deadline = time.monotonic() + 10
            while not endpoint.exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    pytest.fail("managed service startup failed: " + process.communicate(timeout=1)[1])
                time.sleep(0.02)
            yield endpoint
        finally:
            if process.poll() is None:
                process.terminate()
            _, error = process.communicate(timeout=12)
            assert process.returncode == 0, error


@pytest.mark.parametrize("reject", [False, True])
def test_actual_retained_verifier_uses_original_sealed_controller(frozen, tmp_path, monkeypatch, reject):
    _, gateway, snapshot = frozen
    monkeypatch.syspath_prepend(str(repo_root() / SCRIPTS))
    controller = __import__("run_global_perf_experiment")
    original = {
        "reject": reject,
        "host_verification_policy": {
            "sources": {
                str(snapshot / SCRIPTS / "run_global_perf_experiment.py"): gateway._pin(
                    snapshot / SCRIPTS / "run_global_perf_experiment.py"
                )["sha256"]
            }
        },
    }
    path = _write(tmp_path, "retained/checkpoints/original.json", json.dumps(original))
    _write(tmp_path, "retained/launch.json", json.dumps({"source_snapshot": str(snapshot)}))
    before = path.read_bytes()
    if reject:
        with pytest.raises(ValueError, match="module unavailable in frozen source receipt: merlin.missing"):
            controller.verify_retained_global_checkpoint(path)
    else:
        assert controller.verify_retained_global_checkpoint(path) == original
    assert path.read_bytes() == before


@pytest.mark.parametrize("drift_after_start", [False, True])
def test_real_analysis_worker_stage_load_is_source_only_and_rechecks_pin(frozen, tmp_path, drift_after_start):
    _, gateway, snapshot = frozen
    stage = snapshot / "packages/merlin-experiments/src/merlin_experiments/phase2/emission_analysis.py"
    # A valid timestamp-and-size bytecode cache would execute the wrong value under
    # SourceFileLoader. It is deliberately outside the authoritative source receipt.
    decoy = _write(tmp_path, "decoy.py", "VALUE='cached'\n")
    os.utime(decoy, ns=(stage.stat().st_atime_ns, stage.stat().st_mtime_ns))
    cached = Path(importlib.util.cache_from_source(str(stage)))
    prepared_cache = tmp_path / "prepared.pyc"
    py_compile.compile(str(decoy), cfile=str(prepared_cache), dfile=str(stage), doraise=True)
    mutation = (
        "stage.chmod(0o600); stage.write_text(\"VALUE='drift!'\\n\"); stage.chmod(0o444); " if drift_after_start else ""
    )
    program = (
        "from pathlib import Path; from merlin.perf.analysis_worker import _load_analysis; "
        f"stage=Path({str(stage)!r}); stage.parent.chmod(0o700); cache=Path({str(cached)!r}); "
        f"cache.parent.mkdir(); cache.write_bytes({prepared_cache.read_bytes()!r}); stage.parent.chmod(0o500); "
        + mutation
        + "print(_load_analysis(stage).VALUE)"
    )
    result = _run(_command(gateway, snapshot, [sys.executable, "-c", program]))
    if drift_after_start:
        assert result.returncode != 0
        assert "frozen source hash mismatch" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "actual"
