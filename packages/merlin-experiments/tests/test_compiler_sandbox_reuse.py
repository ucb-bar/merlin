"""Compiler calls reuse one sandbox per worker; each call still starts from a clean sandbox."""

import json
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback import sandbox_pool as SP
from merlin_experiments.phase1.feedback import snapshot_execution as S

from merlin.targetgen import package_runtime as P

TOOL = """import json,os,subprocess,sys
from pathlib import Path
incoming,outgoing,private=map(Path,sys.argv[1:4])
seen={"private_visible":private.exists(),
      "tmp":sorted(os.listdir("/tmp")),"shm":sorted(os.listdir("/dev/shm")),
      "io":sorted(os.listdir(os.environ["TMPDIR"])),
      "processes":sorted(int(n) for n in os.listdir("/proc") if n.isdigit()),
      "pid":os.getpid(),"observed":incoming.read_text()}
Path("/tmp/left-by-call").write_text("state")
Path("/tmp/nested").mkdir(exist_ok=True); Path("/tmp/nested/deep").write_text("state")
Path("/dev/shm/left-by-call").write_text("state")
Path(os.environ["TMPDIR"], "extra").write_text("state")
os.chmod("/tmp", 0o500)
subprocess.Popen(["sleep","300"],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
outgoing.write_text(json.dumps(seen))
"""


@pytest.fixture
def reused(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import compiler_profile_runtime as R
    from merlin_experiments.phase2 import campaign as C
    from merlin_experiments.phase2 import compiler_libraries

    candidate = tmp_path / "candidate"
    candidate.mkdir()
    tool = candidate / "tool.py"
    tool.write_text(TOOL)
    private = tmp_path / "private-answer"
    private.write_text("withheld")

    def policy(target, workspace, package, **kwargs):
        return SimpleNamespace(
            argv=(
                "/usr/bin/bwrap",
                "--die-with-parent",
                "--unshare-pid",
                "--unshare-net",
                "--clearenv",
                "--ro-bind",
                "/usr",
                "/usr",
                "--ro-bind",
                "/lib",
                "/lib",
                "--ro-bind",
                "/lib64",
                "/lib64",
                "--symlink",
                "usr/bin",
                "/bin",
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--tmpfs",
                "/tmp",
                "--dir",
                "/tmp/.xdg",
                "--ro-bind",
                str(package),
                str(package),
                "--bind",
                str(workspace),
                str(workspace),
            ),
            env_prefix="export PATH=/usr/bin:/bin; ",
            workspace=workspace,
        )

    monkeypatch.setattr(C, "package_sandbox_policy", policy)
    monkeypatch.setattr(R, "public_package_policy", lambda policy, **kw: policy)
    monkeypatch.setattr(compiler_libraries, "apply", lambda policy: policy)
    monkeypatch.setenv(SP.ENV, SP.PERSISTENT)
    executor = S.SnapshotExecutor(
        SimpleNamespace(snapshot_digest="bundle", profile={"sha256": "profile"}),
        None,
        candidate,
        tmp_path / "host-work",
    )
    pkg = P.Package(
        candidate,
        {
            "language": "python",
            "commands": {
                "parse": {"argv": ["/usr/bin/python3", "{tool}", "{input_mlir}", "{output_json}", str(private)]}
            },
        },
        tool,
    )
    yield executor, pkg
    if executor.live:
        executor.close()


def test_calls_share_one_sandbox_and_each_starts_clean(reused, tmp_path):
    from merlin.common import invocation_record as I

    executor, pkg = reused
    incoming = tmp_path / "interface.mlir"
    incoming.write_text("public source")
    evidence = tmp_path / "results"
    seen = []
    with P.scoped_package_executor(executor):
        for number in range(4):
            output = tmp_path / f"result{number}.json"
            result = P.run_entrypoint(pkg, "parse", incoming, output, invocation_directory=evidence, timeout=30)
            assert result.returncode == 0, result.stderr
            seen.append(json.loads(output.read_text()))
    assert executor.sandboxes.started == 1
    for observed in seen:
        assert observed["observed"] == "public source"
        assert observed["private_visible"] is False
        # Only the skeleton bwrap created survives; the previous call's files are gone.
        assert observed["tmp"] == [".xdg"] and observed["shm"] == []
        assert "extra" not in observed["io"] and "left-by-call" not in observed["io"]
        # The previous call's background process was killed: only init, server and this call.
        assert len(observed["processes"]) <= 3
    receipts = [I.verify(path) for path in evidence.glob("invocations/*/invocation.json")]
    assert len(receipts) == 4
    assert len({receipt["sandbox_reuse"]["sandbox"] for receipt in receipts}) == 1
    assert sorted(receipt["sandbox_reuse"]["call"] for receipt in receipts) == [0, 1, 2, 3]
    executor.close()
    assert not any((executor.work_root / "sandboxes").iterdir())


def test_build_and_per_call_selection_keep_one_sandbox_per_call(tmp_path, monkeypatch):
    monkeypatch.setenv(SP.ENV, SP.PER_CALL)
    assert not SP.selected()
    monkeypatch.delenv(SP.ENV)
    assert not SP.selected()
    monkeypatch.setenv(SP.ENV, "anything")
    with pytest.raises(ValueError, match="unknown sandbox reuse"):
        SP.selected()


def test_resumed_run_keeps_its_recorded_mechanism(monkeypatch):
    monkeypatch.setenv(SP.ENV, SP.PERSISTENT)
    assert SP.adopt_recorded({"compiler_execution": "snapshot_external_v1"}) == SP.PER_CALL
    assert SP.adopt_recorded({SP.KEY: SP.PERSISTENT}) == SP.PERSISTENT
    assert SP.adopt_recorded(None) == SP.PER_CALL


def test_policy_without_private_pid_namespace_is_not_reused(tmp_path):
    io = tmp_path / "io"
    base = ["bwrap", "--unshare-pid", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--bind", "/x", str(io)]
    assert SP.reset_roots(base, io) == ("/dev/shm", "/tmp")
    assert SP.reset_roots([b for b in base if b != "--unshare-pid"], io) is None
    assert SP.reset_roots(["bwrap", "--unshare-pid", "--dev", "/dev"], io) is None
    assert SP.reset_roots([*base, "--ro-bind", "/y", str(io / "nested")], io) is None
    assert SP.reset_roots([*base, "--unknown-flag", "value"], io) is None
    fresh = tmp_path / "box"
    assert SP.reset_roots([*base, "--bind", str(fresh / "0"), "/scratch"], io, fresh)[-1] == "/scratch"
