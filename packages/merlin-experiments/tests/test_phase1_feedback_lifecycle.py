"""Real local processes/threads; no candidate, oracle, credentials or hardware."""

from __future__ import annotations

import ast
import importlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.context import InvocationContext
from merlin_experiments.phase1.feedback import lifecycle as L

from merlin.common.paths import module_source_path, python_import_roots, repo_root


def _config(root, *, tools=()):
    context = InvocationContext(root, root / "descriptor.yaml", root, "fixture", root, root, root, ())
    return L.BrokerConfig(context, tools, root / "timing.json")


def _until(predicate):
    deadline = time.monotonic() + 5
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert predicate(), "local lifecycle operation did not complete"


def test_real_canonical_tool_broker_request_and_shutdown(tmp_path, monkeypatch):
    # CCA's unknown-command branch is a real redacted protocol response, with no
    # target/toolchain lookup. Only the common oracle brokers are omitted here.
    monkeypatch.setattr(L._TR, "COMMON_BROKERS", ())
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(map(str, python_import_roots())))
    actual_popen, logs = subprocess.Popen, []

    def spawn(argv, **kwargs):
        logs.append(kwargs["stdout"])
        return actual_popen(argv, **kwargs)

    monkeypatch.setattr(L.subprocess, "Popen", spawn)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    brokers = L.start_brokers(workspace, _config(tmp_path, tools=("cca_tools",)))
    assert len(brokers) == 1
    try:
        assert logs and all(log.closed for log in logs)
        channel = workspace / ".cca_channel"
        (channel / "req_probe.json").write_text(json.dumps({"cmd": "unrecognized-probe"}))
        _until(lambda: (channel / "done_probe").exists())
        response = json.loads((channel / "resp_probe.json").read_text())
        assert "error" in response
        assert brokers[0].poll() is None
    finally:
        L.stop_brokers(workspace, brokers)
    assert brokers[0].returncode == 0
    for module, staged_as in L._TR.COMMON_CLIENTS:
        assert (workspace / staged_as).read_bytes() == module_source_path(module).read_bytes()
    assert not (workspace / "merlin_experiments").exists()


def test_real_thread_single_flight_handoff_and_first_grade_retry(tmp_path, monkeypatch):
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
    calls = []
    monkeypatch.setattr(L, "FIRST_GRADE_POLL_S", 0.01)

    def first(ws, run, tick, timeout):
        calls.append((tick, timeout))
        if len(calls) == 1:
            raise ValueError("synthetic half-written submission")
        entered.set()
        assert release.wait(5)
        return {"all_pass": None, "tiers_graded": ["loop"], "tiers_not_run": ["cert"]}

    def interval(*args, **kwargs):
        pytest.fail("interval must not run after stop or under the rounds schedule")

    handle = L.start_background(
        tmp_path / "workspace",
        tmp_path / "run",
        L.GradeCadence(1800, False, 900),
        interval_grades=False,
        grade_callback=interval,
        fast_grade_callback=first,
    )
    waiter = threading.Thread(target=lambda: (L.stop_background(handle), stopped.set()))
    try:
        assert entered.wait(5)
        waiter.start()
        _until(handle[1].is_set)
        assert not stopped.wait(0.05), "teardown abandoned an active grade"
        release.set()
        waiter.join(5)
        assert stopped.is_set() and not handle[0].is_alive()
        assert calls == [(900, 900), (900, 900)]
    finally:
        release.set()
        L.stop_background(handle)
        if waiter.ident is not None:
            waiter.join(5)


def test_import_and_thread_operation_outside_checkout_without_native_owners(tmp_path):
    program = """
import importlib.abc, json, os, pathlib, sys
sys.path[:0] = json.loads(sys.argv[1])
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'_common', 'run_baseline_qa_loop', 'run_agent_experiment'}:
            raise AssertionError('native import: ' + fullname)
sys.meta_path.insert(0, NoNative())
before = dict(os.environ)
from merlin_experiments.phase1.feedback import lifecycle as L
assert dict(os.environ) == before
root = pathlib.Path(sys.argv[2])
def grade(*args, **kwargs):
    return {'all_pass': None}
handle = L.start_background(root, root, L.GradeCadence(60, True, 900),
    interval_grades=False, grade_callback=grade, fast_grade_callback=grade)
L.stop_background(handle)
assert not handle[0].is_alive()
"""
    dependencies = [p for p in sys.path if p and Path(p).is_dir() and "site-packages" in p]
    roots = list(map(str, python_import_roots())) + dependencies
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", program, json.dumps(roots), str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt, SystemExit])
def test_partial_start_failure_unwinds_only_owned_children_and_closes_logs(tmp_path, monkeypatch, failure):
    actual_popen = subprocess.Popen
    children = []
    logs = []
    primary = failure("synthetic second broker launch failure")
    for channel in (".isa_channel", ".cca_channel"):
        (tmp_path / channel).mkdir()
        (tmp_path / channel / "existing-peer-marker").write_text("unchanged")
    borrowed = actual_popen(
        [
            sys.executable,
            "-c",
            "import pathlib,sys,time\nwhile not pathlib.Path(sys.argv[1]).exists(): time.sleep(0.01)\n",
            str(tmp_path / ".isa_channel/STOP"),
        ]
    )
    program = (
        "import pathlib,sys,time\n"
        "stop = pathlib.Path(sys.argv[1]) / '.qa_channel/STOP'\n"
        "while not stop.exists(): time.sleep(0.01)\n"
    )

    def spawn(argv, **kwargs):
        logs.append(kwargs["stdout"])
        if children:
            raise primary
        child = actual_popen([sys.executable, "-c", program, str(tmp_path)], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(L.subprocess, "Popen", spawn)
    try:
        with pytest.raises(failure, match="second broker") as raised:
            L.start_brokers(tmp_path, _config(tmp_path, tools=("isa_tools", "cca_tools")))
        assert raised.value is primary
        assert len(children) == 1 and children[0].returncode == 0
        assert all(log.closed for log in logs) and len(logs) == 2
        assert (tmp_path / ".qa_channel/STOP").read_text() == "stop"
        assert borrowed.poll() is None
        for channel in (".isa_channel", ".cca_channel"):
            assert not (tmp_path / channel / "STOP").exists()
            assert (tmp_path / channel / "existing-peer-marker").read_text() == "unchanged"
    finally:
        for child in [*children, borrowed]:
            child.kill()
            child.wait(timeout=5)


def test_stop_kills_and_reaps_after_wait_timeout(tmp_path):
    (tmp_path / ".qa_channel").mkdir()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    calls = []

    class FailingWait:
        def wait(self, timeout):
            calls.append(("wait", timeout))
            if len(calls) == 1:
                raise subprocess.TimeoutExpired("synthetic-broker", timeout)
            return child.wait(timeout=timeout)

        def kill(self):
            calls.append(("kill",))
            child.kill()

    try:
        L.stop_brokers(tmp_path, [FailingWait()])
        assert calls == [("wait", 15), ("kill",), ("wait", L.BROKER_REAP_SECONDS)]
        assert child.returncode is not None
        assert (tmp_path / ".qa_channel/STOP").read_text() == "stop"
    finally:
        child.kill()
        child.wait(timeout=5)


@pytest.mark.parametrize("primary_type", [None, RuntimeError, KeyboardInterrupt, SystemExit])
def test_cleanup_failures_attempt_every_child_and_preserve_primary(tmp_path, monkeypatch, primary_type):
    for name in (".qa_channel", ".isa_channel", ".cca_channel"):
        (tmp_path / name).mkdir()
    children = [subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"]) for _ in range(2)]
    original_write = Path.write_text
    events = []

    def write(path, *args, **kwargs):
        if path == tmp_path / ".qa_channel/STOP":
            raise PermissionError("synthetic STOP refusal")
        return original_write(path, *args, **kwargs)

    class Owned:
        def __init__(self, child, *, broken):
            self.child, self.pid, self.broken, self.waits = child, child.pid, broken, 0

        def wait(self, timeout):
            self.waits += 1
            events.append((self.pid, "wait", self.waits))
            if self.waits == 1:
                raise subprocess.TimeoutExpired("synthetic-grace", timeout)
            return self.child.wait(timeout=timeout)

        def kill(self):
            events.append((self.pid, "kill"))
            if self.broken:
                raise PermissionError("synthetic kill refusal")
            self.child.kill()

    monkeypatch.setattr(Path, "write_text", write)
    monkeypatch.setattr(L, "BROKER_REAP_SECONDS", 0.05)
    owned = [Owned(children[0], broken=True), Owned(children[1], broken=False)]
    primary = primary_type("synthetic provider failure") if primary_type else None
    try:
        if primary is None:
            with pytest.raises(L.BrokerCleanupError) as caught:
                L.stop_brokers(tmp_path, owned)
            assert caught.value.unreaped_pids == (children[0].pid,)
            detail = str(caught.value)
        else:
            with pytest.raises(primary_type) as caught:
                try:
                    raise primary
                finally:
                    L.stop_brokers(tmp_path, owned, primary_error=sys.exception())
            assert caught.value is primary
            detail = "\n".join(primary.__notes__)
        assert "signal .qa_channel" in detail and "kill PID" in detail and "reap PID" in detail
        assert str(children[0].pid) in detail
        assert (tmp_path / ".isa_channel/STOP").read_text() == "stop"
        assert (tmp_path / ".cca_channel/STOP").read_text() == "stop"
        assert children[1].returncode is not None
        assert events == [
            (child.pid, step, *extra)
            for child in children
            for step, extra in (("wait", [1]), ("kill", []), ("wait", [2]))
        ]
    finally:
        for child in children:
            child.kill()
            child.wait(timeout=5)


def test_partial_start_failure_keeps_primary_when_cleanup_also_fails(tmp_path, monkeypatch):
    actual_popen = subprocess.Popen
    child = None
    logs = []
    primary = OSError("synthetic spawn failure")
    original_write = Path.write_text

    def write(path, *args, **kwargs):
        if path.name == "STOP":
            raise PermissionError("synthetic signal failure")
        return original_write(path, *args, **kwargs)

    def spawn(argv, **kwargs):
        nonlocal child
        logs.append(kwargs["stdout"])
        if child is not None:
            raise primary
        child = actual_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        return child

    monkeypatch.setattr(Path, "write_text", write)
    monkeypatch.setattr(L.subprocess, "Popen", spawn)
    monkeypatch.setattr(L, "BROKER_GRACE_SECONDS", 0.01)
    try:
        with pytest.raises(OSError) as caught:
            L.start_brokers(tmp_path, _config(tmp_path))
        assert caught.value is primary
        assert "signal .qa_channel" in "\n".join(primary.__notes__)
        assert child is not None and child.returncode is not None
        assert all(log.closed for log in logs)
    finally:
        if child is not None:
            child.kill()
            child.wait(timeout=5)


def test_log_close_failure_does_not_replace_launch_exception(tmp_path, monkeypatch):
    actual_open = open
    logs = []
    primary = OSError("synthetic Popen failure")

    class FailingClose:
        def __init__(self, path, mode):
            self.file = actual_open(path, mode)

        def close(self):
            self.file.close()
            raise OSError("synthetic log close failure")

    def log_file(path, mode):
        log = FailingClose(path, mode)
        logs.append(log)
        return log

    def spawn(*args, **kwargs):
        raise primary

    monkeypatch.setattr(L, "open", log_file, raising=False)
    monkeypatch.setattr(L.subprocess, "Popen", spawn)
    with pytest.raises(OSError) as caught:
        L.start_brokers(tmp_path, _config(tmp_path))
    assert caught.value is primary
    assert logs and all(log.file.closed for log in logs)
    assert "log close failed" in "\n".join(primary.__notes__)
    assert not (tmp_path / ".qa_channel/STOP").exists()  # no owned process was started


@pytest.mark.parametrize("caller", ["provider", "smoke"])
@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt, SystemExit, None])
def test_actual_caller_finally_preserves_primary_and_stops_brokers(tmp_path, monkeypatch, caller, failure):
    # Execute the actual caller's try/finally AST, without initializing its native
    # target/provider globals. The external provider body alone is substituted.
    harness = repo_root() / "merlin/experiments/capsule_bench/harness"
    from merlin_experiments.phase1.providers import execution

    path = Path(execution.__file__) if caller == "provider" else harness / "smoke_agent_check.py"
    tree = ast.parse(path.read_text())
    stop_try = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        and any(
            isinstance(call, ast.Call) and ast.unparse(call.func) == "FL.stop_brokers"
            for statement in node.finalbody
            for call in ast.walk(statement)
        )
    )
    if caller == "provider":
        stop_try.body = ast.parse("run_provider()\n").body
    initializer = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "primary_error" for target in node.targets)
        and isinstance(node.value, ast.Constant)
        and node.value.value is None
    )
    body = ast.FunctionDef(
        name="run",
        args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[initializer, stop_try],
        decorator_list=[],
    )
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    (tmp_path / ".qa_channel").mkdir()
    primary = failure("synthetic provider failure") if failure is not None else None

    def run_provider(*args, **kwargs):
        if primary is not None:
            raise primary
        return SimpleNamespace(stdout="")

    if primary is None:
        original_write = Path.write_text

        def failed_stop(path, *args, **kwargs):
            if path.name == "STOP":
                raise PermissionError("synthetic cleanup failure after normal return")
            return original_write(path, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", failed_stop)

    monkeypatch.setattr(L, "BROKER_GRACE_SECONDS", 0.01)
    namespace = {
        "FL": L,
        "sys": sys,
        "ws": tmp_path,
        "broker": [child],
        "cmd": "unused",
        "a": SimpleNamespace(timeout=1),
        "run_provider": run_provider,
        "subprocess": SimpleNamespace(run=run_provider, TimeoutExpired=subprocess.TimeoutExpired),
    }
    try:
        exec(
            compile(ast.fix_missing_locations(ast.Module(body=[body], type_ignores=[])), str(path), "exec"),
            namespace,
        )
        unrelated = ValueError("unrelated handled caller exception")
        try:
            raise unrelated
        except ValueError:
            with pytest.raises(failure or L.BrokerCleanupError) as caught:
                namespace["run"]()
        if primary is not None:
            assert caught.value is primary
        assert not getattr(unrelated, "__notes__", ())
        assert child.returncode is not None
    finally:
        child.kill()
        child.wait(timeout=5)


def test_log_close_failure_inside_unrelated_except_raises_and_reaps(tmp_path, monkeypatch):
    actual_open, actual_popen = open, subprocess.Popen
    children, logs = [], []
    close_error = OSError("synthetic close failure after successful spawn")

    class FailingClose:
        def __init__(self, path, mode):
            self.file = actual_open(path, mode)

        def fileno(self):
            return self.file.fileno()

        def close(self):
            self.file.close()
            raise close_error

    def log_file(path, mode):
        log = FailingClose(path, mode)
        logs.append(log)
        return log

    def spawn(*args, **kwargs):
        child = actual_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(L, "open", log_file, raising=False)
    monkeypatch.setattr(L.subprocess, "Popen", spawn)
    monkeypatch.setattr(L, "BROKER_GRACE_SECONDS", 0.01)
    unrelated = ValueError("unrelated handled caller exception")
    try:
        try:
            raise unrelated
        except ValueError:
            with pytest.raises(OSError) as caught:
                L.start_brokers(tmp_path, _config(tmp_path))
        assert caught.value is close_error
        assert len(children) == 1 and children[0].returncode is not None
        assert all(log.file.closed for log in logs)
        assert not getattr(unrelated, "__notes__", ())
    finally:
        for child in children:
            child.kill()
            child.wait(timeout=5)


def test_lifecycle_is_withheld_under_broad_candidate_grant(tmp_path, monkeypatch):
    from merlin.common import access
    from merlin.targetgen.sandbox import bwrap

    surfaces_module = importlib.import_module("merlin.targetgen.sandbox.answer_surfaces")
    private = tmp_path / "packages/merlin-experiments/src/merlin_experiments/phase1/feedback/lifecycle.py"
    private.parent.mkdir(parents=True)
    private.write_text("# private host lifecycle sentinel\n")
    monkeypatch.setattr(access, "sys", SimpleNamespace(path=[], prefix=str(tmp_path / "python"), modules={}))
    monkeypatch.setattr(surfaces_module, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(surfaces_module, "artifacts_dir", lambda: tmp_path / "out/artifacts")
    monkeypatch.setattr(surfaces_module, "_evicted_oracle_modules", lambda: [])
    monkeypatch.setattr(surfaces_module, "experimenter_memory_dir", lambda: tmp_path / "absent-memory")
    policy = SimpleNamespace(
        target="fixture",
        capsule_corpus=None,
        corpus_siblings=lambda: (),
        hidden_corpus=lambda: None,
        prior_backends=(),
        backend_package=None,
    )
    surfaces = surfaces_module.answer_surfaces(policy)
    unmasked = ["--ro-bind", str(tmp_path), str(tmp_path)]
    gaps = bwrap.coverage_gap(unmasked, surfaces)
    assert any(private.is_relative_to(surface.path) for surface in gaps)
    assert bwrap.coverage_gap(bwrap.apply_answer_masks(unmasked, surfaces), surfaces) == []


@pytest.mark.parametrize("mutation", ["bytes", "member"])
def test_source_receipt_rejects_lifecycle_drift(tmp_path, monkeypatch, mutation):
    from merlin_experiments.phase1 import source_inputs as SI
    from merlin_experiments.spec import SpecError

    package = tmp_path / "phase1"
    shutil.copytree(
        module_source_path("merlin_experiments.phase1").parent, package, ignore=shutil.ignore_patterns("__pycache__")
    )
    original = SI._source

    def copied_source(module):
        if module == "merlin_experiments.phase1":
            return package / "__init__.py"
        if module.startswith("merlin_experiments.phase1."):
            path = package.joinpath(*module.split(".")[2:])
            return path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
        return original(module)

    monkeypatch.setattr(SI, "_source", copied_source)
    inputs = {"repo": tmp_path, "entrypoint": tmp_path / "synthetic_transport.py"}
    record = SI.record(**inputs)
    member = package / "feedback/lifecycle.py"
    assert record["inputs"]["phase1:source:feedback/lifecycle.py"]["path"] == str(member)
    SI.verify(record, **inputs)
    if mutation == "bytes":
        member.write_text(member.read_text() + "# changed lifecycle\n")
    else:
        member.with_name("added_lifecycle.py").write_text("# new lifecycle member\n")
    with pytest.raises(SpecError, match="source identity changed"):
        SI.verify(record, **inputs)
