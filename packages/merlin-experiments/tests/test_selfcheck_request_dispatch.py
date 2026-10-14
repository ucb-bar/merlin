"""Bounded independent request dispatch, without an oracle or compiler fixture."""

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.brokers import selfcheck as B
from merlin_experiments.phase1.feedback import selfcheck as S


def request(ch, rid, **fields):
    path = ch / f"req_{rid}.json"
    path.write_text(json.dumps({"capsules": "all", "workers": 1, "sim": "spike",
        "deadline_unix_ns": time.time_ns() + 30_000_000_000, **fields}))
    return path


@pytest.mark.parametrize("value", ["", "0", "3", "-1", "true", "2.0"])
def test_dispatch_capacity_rejects_unbounded_selection(monkeypatch, value):
    monkeypatch.setenv("MERLIN_SELFCHECK_BROKER_JOBS", value)
    with pytest.raises(ValueError, match="1 or 2"):
        B._dispatch_capacity()


def test_dispatch_capacity_preserves_legacy_default(monkeypatch):
    monkeypatch.delenv("MERLIN_SELFCHECK_BROKER_JOBS", raising=False)
    assert B._dispatch_capacity() == 1
    monkeypatch.setenv("MERLIN_SELFCHECK_BROKER_JOBS", "2")
    assert B._dispatch_capacity() == 2


def test_command_and_lane_bind_same_original_request_bytes(tmp_path):
    ch = tmp_path / ".qa_channel"
    ch.mkdir()
    req = request(ch, "case", offload_census=True, artifact_root="/untrusted",
                  repo="/untrusted", compiler_bundle_manifest="/untrusted")
    original = req.read_bytes()
    req.write_text(json.dumps({"offload_census": False}))
    a = SimpleNamespace(ws=tmp_path, poll=.01, rtl_facts=tmp_path / "facts",
                        compiler_bundle_manifest=tmp_path / "original-profile")
    ctx = SimpleNamespace(descriptor=tmp_path / "descriptor", repo=tmp_path / "original-repo",
                          readback_policy=None)
    assert B._diagnostic_request(req, request_bytes=original)
    cmd = B._handler_command(a, ctx, tmp_path / "public", tmp_path / "contract", None,
                            req, ch / ".stop", request_bytes=original)
    assert cmd[:3] == [sys.executable, "-m", "merlin_experiments.phase1.brokers.selfcheck"]
    assert cmd[cmd.index("--repo") + 1] == str(ctx.repo)
    assert cmd[cmd.index("--compiler-bundle-manifest") + 1] == str(a.compiler_bundle_manifest)
    assert cmd[cmd.index("--request-sha256") + 1] == hashlib.sha256(original).hexdigest()
    assert "/untrusted" not in cmd


@pytest.mark.parametrize("bad", ["foreign_root", "different_request", "symlink"])
def test_artifact_root_refuses_namespace_or_identity_change(tmp_path, monkeypatch, bad):
    monkeypatch.chdir(tmp_path)
    ch = tmp_path / ".qa_channel"
    ch.mkdir()
    root = tmp_path / "selfcheck_out" / "requests" / "case"
    root.parent.mkdir(parents=True)
    if bad == "foreign_root":
        root = tmp_path / "outside"
    elif bad == "different_request":
        root = root.with_name("other")
    else:
        root.symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="original per-request"):
        S._request_artifact_root(str(root), progress=str(ch / "progress_case.json"),
                                output=str(ch / ".resp_case.123.tmp"))


def test_artifact_root_accepts_only_original_matching_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "selfcheck_out" / "requests" / "case"
    assert S._request_artifact_root(str(root), progress=str(tmp_path / ".qa_channel/progress_case.json"),
        output=str(tmp_path / ".qa_channel/.resp_case.123.tmp")) == root
    with pytest.raises(ValueError, match="response identity"):
        S._request_artifact_root(str(root), progress=str(tmp_path / ".qa_channel/progress_case.json"),
            output=str(tmp_path / ".qa_channel/.resp_other.123.tmp"))


def test_original_absolute_deadline_is_not_renewed_after_queue(monkeypatch):
    monkeypatch.setattr(B.time, "time", lambda: 100.)
    monkeypatch.setattr(B.time, "monotonic", lambda: 500.)
    body = {"deadline_unix_ns": 103_000_000_000, "timeout": 1800}
    assert B._child_deadline_monotonic(body) == 503.
    monkeypatch.setattr(B.time, "time", lambda: 102.)
    assert B._child_deadline_monotonic(body) == 501.


def test_requester_death_and_cancel_stop_only_its_request(tmp_path, monkeypatch):
    req = request(tmp_path, "case")
    monkeypatch.setattr(B, "_should_stop", lambda *args: None)
    monkeypatch.setattr(B, "_requester_alive", lambda body: False)
    assert B._abort_reason(tmp_path, os.getppid(), req, time.monotonic() + 10) == "requester exited"
    req.unlink()
    assert "cancelled" in B._abort_reason(tmp_path, os.getppid(), req, time.monotonic() + 10)


def test_changed_dispatched_bytes_refuse_before_worker_launch(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import frozen_facts
    ch = tmp_path / ".qa_channel"
    ch.mkdir()
    req = request(ch, "case", offload_census=True)
    pinned = hashlib.sha256(req.read_bytes()).hexdigest()
    req.write_text(req.read_text().replace("true", "false"))
    monkeypatch.setattr(frozen_facts, "select", lambda *args: None)
    monkeypatch.setattr(B, "resolve_context", lambda *args: SimpleNamespace(repo=tmp_path))
    monkeypatch.setattr(B, "worker_command", lambda *args, **kwargs: pytest.fail("worker must not launch"))
    assert B.main(["--ws", str(tmp_path), "--capsules-root", str(tmp_path),
                   "--one-request", req.name, "--request-sha256", pinned]) == 0
    result = json.loads((ch / "resp_case.json").read_bytes())
    assert "error" in result
    assert "n_passed" not in result


def test_owned_worker_stop_reaps_children_and_releases_locks(tmp_path):
    lock = tmp_path / "held.lock"
    child_info = tmp_path / "child.json"
    child = "import fcntl,time,sys; f=open(sys.argv[1],'w'); fcntl.flock(f,fcntl.LOCK_EX); time.sleep(30)"
    parent = ("import json,subprocess,sys,time; "
              "p=subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]); "
              "open(sys.argv[3],'w').write(json.dumps({'pid':p.pid})); time.sleep(30)")
    process = subprocess.Popen([sys.executable, "-c", parent, child, str(lock), str(child_info)])
    try:
        end = time.monotonic() + 5
        while not child_info.exists():
            assert time.monotonic() < end
            time.sleep(.01)
        pid = json.loads(child_info.read_bytes())["pid"]
        B._stop_owned_worker(process)
        assert process.poll() is not None
        record = B._process_record(pid)
        assert record is None or record[2] == "Z"
        import fcntl
        with lock.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if process.poll() is None:
            B._stop_owned_worker(process)


def test_two_process_lanes_overlap_and_cancel_independently(tmp_path, monkeypatch):
    """Use the actual request handler; substitute only the external grading executable."""
    ws = tmp_path / "workspace"
    ch = ws / ".qa_channel"
    ch.mkdir(parents=True)
    source = ws / "submission/compiler.py"
    source.parent.mkdir()
    source.write_text("compiler source\n")
    worker = tmp_path / "fixture_worker.py"
    worker.write_text('''import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out');p.add_argument('--progress-out');p.add_argument('--artifact-root');a,_=p.parse_known_args()
channel=Path(a.out).parent;rid=Path(a.progress_out).stem[len('progress_'):]
start=time.monotonic();(channel/('started_'+rid)).write_text(json.dumps({'pid':os.getpid(),'start':start,'root':a.artifact_root}))
while not (channel/'release').exists():time.sleep(.01)
Path(a.out).write_text(json.dumps({'all_pass':False,'per_capsule':[],'fixture_pid':os.getpid(),'fixture_started':start,'fixture_finished':time.monotonic(),'artifact_root':a.artifact_root}))
''')
    handler = tmp_path / "fixture_handler.py"
    handler.write_text('''import sys
from pathlib import Path
from types import SimpleNamespace
from merlin_experiments.phase1.brokers import selfcheck as B
from merlin_experiments.phase1 import frozen_facts
frozen_facts.select=lambda *args:None
B.resolve_context=lambda a,*args:SimpleNamespace(
    repo=Path(a.repo),descriptor=Path(a.descriptor),target='synthetic',readback_policy=None)
B.worker_command=lambda *args,**kwargs:[sys.executable,''' + repr(str(worker)) + ''']
B._sim_policy_error=lambda sim:None
raise SystemExit(B.main())
''')
    for key in ("MERLIN_COMPUTE_CPUS", "MERLIN_COMPUTE_CPU_SLOTS", "MERLIN_GRADE_CPUS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("MERLIN_SELFCHECK_BROKER_JOBS", "2")
    diagnostic = request(ch, "diagnostic", offload_census=True)
    time.sleep(.01)
    numerical = request(ch, "numerical")
    later = request(ch, "later_numeric")
    arrival = time.time()
    for offset, path in ((3, diagnostic), (2, numerical), (1, later)):
        os.utime(path, (arrival - offset, arrival - offset))
    original = B._handler_command
    launch_order = []
    def command(*args, **kwargs):
        cmd = original(*args, **kwargs)
        launch_order.append(cmd[cmd.index("--one-request") + 1])
        return [sys.executable, str(handler), *cmd[3:]]
    monkeypatch.setattr(B, "_handler_command", command)
    observed = []
    def monitor():
        try:
            end = time.monotonic() + 15
            while not all((ch / ("started_" + rid)).exists() for rid in ("diagnostic", "numerical")):
                assert time.monotonic() < end
                time.sleep(.02)
            assert not (ch / "started_later_numeric").exists()
            numerical.unlink()  # Existing ordinary cancellation protocol.
            while not (ch / "done_numerical").exists():
                assert time.monotonic() < end
                time.sleep(.02)
            assert not (ch / "done_diagnostic").exists()
            (ch / "release").touch()
            while not (ch / "done_diagnostic").exists():
                assert time.monotonic() < end
                time.sleep(.02)
        except BaseException as error:
            observed.append(error)
        finally:
            (ch / "STOP").touch()
    thread = threading.Thread(target=monitor)
    thread.start()
    a = SimpleNamespace(ws=ws, poll=.01, rtl_facts=None, compiler_bundle_manifest=None)
    context = SimpleNamespace(repo=tmp_path, descriptor=tmp_path / "descriptor", readback_policy=None)
    B._dispatch_requests(a, context, tmp_path, None, None)
    thread.join(timeout=20)
    assert not observed, observed
    assert launch_order[:2] == [numerical.name, diagnostic.name]
    canceled = json.loads((ch / "resp_numerical.json").read_bytes())
    finished = json.loads((ch / "resp_diagnostic.json").read_bytes())
    assert "cancelled" in canceled["error"] and canceled["all_pass"] is False
    assert "error" not in finished and finished["all_pass"] is False
    assert finished["artifact_root"] == str(ws / "selfcheck_out/requests/diagnostic")
    for rid in ("diagnostic", "numerical"):
        info = json.loads((ch / ("started_" + rid)).read_bytes())
        assert B._process_record(info["pid"]) is None
