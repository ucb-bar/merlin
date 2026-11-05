"""Private descriptor launch controls, including one tiny real Linux namespace.

These establish argument transport and descriptor privacy, never accelerator,
compiler, numerical or deployed runtime qualification.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common import bwrap_args as B
from merlin.common import invocation_record


def _large_options():
    marker = "PRIVATE_MASK_METADATA_" + "x" * 1100
    # Exceeds Linux ARG_MAX while respecting this installed bwrap's separate
    # 9,000-token parser limit. Transport does not override binary capacity.
    return tuple(token for index in range(2400) for token in ("--unsetenv", marker + str(index)))


def test_exact_large_arguments_are_readonly_sealed_and_closed_without_files(tmp_path):
    tokens = ("/usr/bin/bwrap", "--setenv", "KEY", "quote ' newline\n unicode α", *_large_options(), "--", "true")
    encoded = b"\0".join(os.fsencode(token) for token in tokens[1:-2]) + b"\0"
    original = set(tmp_path.iterdir())
    with B.argument_transport(tokens, policy_end=len(tokens) - 2) as transport:
        fd = transport.descriptor
        assert transport.logical_argv == tokens
        assert transport.command == ("/usr/bin/bwrap", "--args", str(fd), "--", "true")
        assert transport.pass_fds == (fd,)
        assert fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY
        assert fcntl.fcntl(fd, B._GET_SEALS) & B._SEALS == B._SEALS
        assert os.pread(fd, len(encoded), 0) == encoded
        with pytest.raises(OSError):
            os.write(fd, b"modified private mask")
        report = transport.record()
        assert report["payload_sha256"] == hashlib.sha256(encoded).hexdigest()
        assert report["payload_bytes"] == len(encoded) and "descriptor" not in report
    with pytest.raises(OSError) as closed:
        os.fstat(fd)
    assert closed.value.errno == errno.EBADF
    assert set(tmp_path.iterdir()) == original


def test_compressed_launch_observation_retains_logical_argv_and_transport_dependency(tmp_path, monkeypatch):
    calls = []
    argv = ("/usr/bin/bwrap", *_large_options(), "--", "true")

    def launch(command, **options):
        assert len(command) == 5 and command[1] == "--args" and command[-2:] == ["--", "true"]
        fd = options["pass_fds"][0]
        assert command[2] == str(fd)
        calls.append(fd)
        return subprocess.CompletedProcess(command, 0, "actual process boundary stub", "")

    monkeypatch.setattr(B.subprocess, "run", launch)
    result = B.run(
        argv, policy_end=len(argv) - 2, directory=tmp_path, stage="transport_control", capture_output=True, text=True
    )
    assert tuple(result.args) == argv and result.returncode == 0
    [path] = tmp_path.rglob("invocation.json")
    record = invocation_record.verify(path)
    assert record["argv"] == list(argv)
    assert record["argument_transport"]["transport"] == "sealed_readonly_fd"
    assert any(row["path"] == B.__file__ for row in record["dependencies"])
    with pytest.raises(OSError):
        os.fstat(calls[0])


@pytest.mark.parametrize("failure", [OSError(errno.EIO, "launch failed"), subprocess.TimeoutExpired("bwrap", 1)])
def test_failed_launch_closes_descriptor_and_records_interruption(tmp_path, monkeypatch, failure):
    descriptors = []

    def launch(command, **options):
        descriptors.extend(options["pass_fds"])
        raise failure

    monkeypatch.setattr(B.subprocess, "run", launch)
    with pytest.raises(type(failure)):
        argv = ("/usr/bin/bwrap", *_large_options(), "--", "true")
        B.run(
            argv,
            policy_end=len(argv) - 2,
            directory=tmp_path,
            stage="failed_transport",
            capture_output=True,
        )
    assert len(descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])
    [path] = tmp_path.rglob("invocation.json")
    document = json.loads(path.read_text())
    assert document["status"] == "interrupted" and document["argument_transport"]["transport"] == "sealed_readonly_fd"


def test_small_launch_and_invalid_or_unbounded_transport_keep_original_rules(monkeypatch):
    with B.argument_transport(("/usr/bin/bwrap", "--", "true")) as transport:
        assert transport.command == transport.logical_argv and not transport.pass_fds
        assert transport.record()["transport"] == "inline"
    for argv in ((), ("/usr/bin/bwrap", "contains\0NUL"), ("/usr/bin/bwrap", 1)):
        with pytest.raises(ValueError):
            with B.argument_transport(argv):
                pytest.fail("invalid original argv admitted")
    monkeypatch.setattr(B, "_MAX_BYTES", 8)
    with pytest.raises(ValueError, match="byte budget"):
        with B.argument_transport(("/usr/bin/bwrap", "longer than eight bytes")):
            pytest.fail("unbounded original argv admitted")


def test_selected_bwrap_parser_capacity_refusal_is_preserved_not_a_pass(tmp_path, monkeypatch):
    argv = ("/usr/bin/bwrap", *_large_options(), "--", "/usr/bin/true")
    calls = []

    def parser_refusal(command, **options):
        # An explicit native boundary stub reports the real selected binary's
        # observed capacity refusal. It does not establish capacity elsewhere.
        calls.append(command[0])
        assert command[1] == "--args" and len(options["pass_fds"]) == 1
        return subprocess.CompletedProcess(command, 1, "", "bwrap: Exceeded maximum number of arguments 9000\n")

    monkeypatch.setattr(B.subprocess, "run", parser_refusal)
    result = B.run(
        argv,
        policy_end=len(argv) - 2,
        directory=tmp_path,
        stage="selected_capacity_refusal",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1 and "9000" in result.stderr and calls == ["/usr/bin/bwrap"]
    [path] = tmp_path.rglob("invocation.json")
    assert json.loads(path.read_text())["status"] == "failed"
    with pytest.raises(ValueError, match="did not complete"):
        invocation_record.verify(path)


def test_large_original_executor_launch_exceeds_argmax_and_hides_fd_before_payload(tmp_path):
    from merlin_experiments.phase1.feedback.compiler_profile_runtime import _Executor

    binary = shutil.which("bwrap")
    if binary is None:
        pytest.skip("Linux bwrap is unavailable")
    prefix = [
        binary,
        "--die-with-parent",
        "--unshare-pid",
        "--unshare-net",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        "/lib",
        "/lib",
        "--ro-bind",
        "/lib64",
        "/lib64",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--dir",
        "/work",
        "--clearenv",
        *_large_options(),
    ]
    code = (
        "import json,os,sys\n"
        "def links(pid):\n"
        " result=[]\n"
        " for name in os.listdir('/proc/'+pid+'/fd'):\n"
        "  try: result.append(os.readlink('/proc/'+pid+'/fd/'+name))\n"
        "  except FileNotFoundError: pass\n"
        " return result\n"
        "print(json.dumps({'self':links('self'),'init':links('1'),'argv':sys.argv[1:]}))\n"
    )
    payload = ["/usr/bin/python3", "-c", code, "ordinary payload"]
    assert sum(len(os.fsencode(token)) + 1 for token in (*prefix, *payload)) > os.sysconf("SC_ARG_MAX")
    # This directly reproduces execve's original aggregate failure without
    # launching a compiler or any workload.
    with pytest.raises(OSError) as original_failure:
        subprocess.run([*prefix, "--", *payload], capture_output=True, text=True, timeout=10)
    assert original_failure.value.errno == errno.E2BIG
    policy = SimpleNamespace(argv=tuple(prefix), env_prefix="", workspace=tmp_path)
    executor = object.__new__(_Executor)  # Exercise transport only; never claims original source admission.
    try:
        actual = executor._invoke(policy, Path("/work"), payload, timeout=15, evidence=tmp_path / "evidence")
    except subprocess.TimeoutExpired:
        pytest.fail("tiny Linux argument transport exceeded its fixed control budget")
    if actual.returncode and "Operation not permitted" in actual.stderr:
        pytest.skip("Linux namespace creation is unavailable")
    assert actual.returncode == 0, actual.stderr
    result = json.loads(actual.stdout)
    assert result["argv"] == ["ordinary payload"]
    assert not any("merlin-private-bwrap-args" in name for group in ("self", "init") for name in result[group])
    [path] = (tmp_path / "evidence").rglob("invocation.json")
    record = invocation_record.verify(path)
    assert len(record["argv"]) > 4000 and record["argument_transport"]["transport"] == "sealed_readonly_fd"
    assert record["argv"][-4:] == payload
    assert all("bwrap-args" not in path.name for path in tmp_path.rglob("*"))
