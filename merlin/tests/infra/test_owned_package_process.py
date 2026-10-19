"""Real harmless package descendants are reaped before success or timeout returns."""

import ctypes
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from merlin.common import invocation_record as I
from merlin.common import owned_process as P
from merlin.targetgen import package_runtime as R


def _subreaper():
    flag = ctypes.c_int()
    assert ctypes.CDLL(None).prctl(37, ctypes.byref(flag), 0, 0, 0) == 0
    return flag.value


@pytest.mark.parametrize("recorded", [False, True])
@pytest.mark.parametrize("leader_exits", [False, True])
def test_package_reaps_session_escaping_descendant_before_return(tmp_path, monkeypatch, recorded, leader_exits):
    # This is the real package path, including its ordinary invocation observer.
    # A child-created session specifically defeats process-group-only cleanup.
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    tool = package_dir / "tool.py"
    pid_file = tmp_path / "descendant.pid"
    source = tmp_path / "source.mlir"
    source.write_text("synthetic input")
    tool.write_text(
        "import os,signal,time,pathlib,sys\n"
        "pid=os.fork()\n"
        "if pid == 0:\n"
        " os.setsid()\n"
        " signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
        f" pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
        " time.sleep(30)\n"
        "else:\n"
        ' print("actual parent output",flush=True)\n' + (" time.sleep(0.15)\n" if leader_exits else " time.sleep(30)\n")
    )
    pkg = R.Package(
        package_dir, {"language": "python", "commands": {"parse": {"argv": [sys.executable, str(tool)]}}}, tool
    )
    monkeypatch.setattr(R, "_admit_artifact_protocol", lambda *_args: None)
    before = _subreaper()
    kwargs = {"invocation_directory": tmp_path / "evidence"} if recorded else {}
    if leader_exits:
        result = R.run_entrypoint(pkg, "parse", source, timeout=2, **kwargs)
        assert result.returncode == 0 and result.stdout == "actual parent output\n"
    else:
        with pytest.raises(subprocess.TimeoutExpired) as raised:
            R.run_entrypoint(pkg, "parse", source, timeout=0.4, **kwargs)
        assert raised.value.output == b"actual parent output\n"
    assert pid_file.is_file()
    assert not Path("/proc", pid_file.read_text()).exists(), "A session-escaping child survived the package call"
    assert _subreaper() == before, "Only the dedicated guardian may become a subreaper"
    if recorded:
        path = next((tmp_path / "evidence").rglob("invocation.json"))
        cleanup = json.loads(path.read_text())["process_tree_cleanup"]
        assert cleanup["complete"] and cleanup["guardian_reaped"] and cleanup["reaped_children"] >= 2
        if leader_exits:
            document = I.verify(path)
            assert document["returncode"] == 0
            assert {Path(pin["path"]).name for pin in document["dependencies"]} >= {
                "owned_process.py",
                "owned_children.py",
            }
        else:
            assert json.loads(path.read_text())["status"] == "interrupted"
            with pytest.raises(ValueError):
                I.verify(path)


def test_owned_command_retains_failure_streams_and_exit_code():
    result = P.run(
        [sys.executable, "-c", 'import sys;print("out");print("err",file=sys.stderr);raise SystemExit(7)'],
        timeout=2,
        capture_output=True,
        text=True,
    )
    assert (result.returncode, result.stdout, result.stderr) == (7, "out\n", "err\n")


def test_owned_command_preserves_executable_startup_error(tmp_path):
    with pytest.raises(FileNotFoundError) as caught:
        P.run([tmp_path / "missing-executable"], timeout=2)
    assert caught.value.filename == str(tmp_path / "missing-executable")


def test_parallel_guardians_do_not_keep_other_jobs_pipe_writers(tmp_path):
    release = tmp_path / "release-slow-command"
    slow_code = f"from pathlib import Path\nimport time\nwhile not Path({str(release)!r}).exists(): time.sleep(0.01)\n"
    with ThreadPoolExecutor(max_workers=2) as pool:
        slow = pool.submit(P.run, [sys.executable, "-c", slow_code], timeout=4)
        fast = pool.submit(P.run, [sys.executable, "-c", 'print("fast")'], timeout=2)
        try:
            assert fast.result(timeout=2).stdout == b"fast\n"
        finally:
            release.write_text("done")
        assert slow.result().returncode == 0
