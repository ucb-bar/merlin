"""Actual host census lifecycle under FD pressure; no target or timing claims."""

import os
import resource
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from test_dynamic_census import census, lines

from merlin.common import invocation_record as I
from merlin.perf import _census_process as P
from merlin.perf._dynamic_stream import CensusUnavailable


@contextmanager
def high_fds():
    if resource.getrlimit(resource.RLIMIT_NOFILE)[0] < 1200:
        pytest.skip("actual high-FD census needs at least 1200 available descriptors")
    descriptors = []
    try:
        while not descriptors or descriptors[-1] < 1050:
            descriptors.append(os.open(os.devnull, os.O_RDONLY | os.O_CLOEXEC))
        yield
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def watch_real_pidfds(monkeypatch):
    actual = P.children.pidfd_open
    opened = []

    def watch(pid):
        descriptor = actual(pid)
        assert descriptor > 1023  # Also executes in the actual guardian.
        opened.append(pid)
        return descriptor

    monkeypatch.setattr(P.children, "pidfd_open", watch)
    return opened


def assert_guardian_reaped(opened):
    guardians = set(opened) - {os.getpid()}
    assert guardians
    for pid in guardians:
        assert not Path("/proc", str(pid)).exists()
        with pytest.raises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)


def test_two_actual_high_fd_captures_keep_streams_complete_and_reap_guardians(tmp_path, monkeypatch):
    opened = watch_real_pidfds(monkeypatch)
    payload = "\n".join(lines()) + "\n"
    with high_fds():
        for index in range(2):
            directory = tmp_path / str(index)
            directory.mkdir()
            reader = census()
            argv = [sys.executable, "-c", f"import sys; sys.stderr.write({payload!r}); print('DONE')"]
            with I.observe(directory, stage="actual_high_fd_host_census", argv=argv, env=dict(os.environ)) as record:
                assert P.capture(argv, environment=dict(os.environ), cwd=directory, limits=reader.limits,
                                 record=record, census=reader) == 0
                assert record.census_started and record.census_cleanup_complete
                record.complete_streams(0)
                assert I.verify(record.path)["returncode"] == 0
                assert reader.finish()["accelerator"]["commands_in_entry"] == 3
                assert (record.directory / "stderr.bin").read_text() == payload
    assert_guardian_reaped(opened)


@pytest.mark.parametrize("failure", ["deadline", "parser_exception"])
def test_actual_high_fd_cancellation_reaps_escaped_descendant_and_preserves_failure(
    tmp_path, monkeypatch, failure
):
    opened = watch_real_pidfds(monkeypatch)
    leader, descendant = tmp_path / "leader.pid", tmp_path / "descendant.pid"
    code = (
        "import os,pathlib,time,sys; "
        f"pathlib.Path({str(leader)!r}).write_text(str(os.getpid())); pid=os.fork(); "
        f"pathlib.Path({str(descendant)!r}).write_text(str(pid)) if pid else os.setsid(); "
        "time.sleep(.03); sys.stderr.write('trigger\\n'); sys.stderr.flush(); time.sleep(30)"
    )
    reader = census()
    if failure == "parser_exception":
        def fail(_line):
            raise RuntimeError("original parser exception")

        monkeypatch.setattr(reader, "line", fail)
    else:
        monkeypatch.setattr(reader, "line", lambda _line: None)
    limits = replace(reader.limits, wall_s=.3 if failure == "deadline" else 3)
    argv = [sys.executable, "-c", code]
    with high_fds(), I.observe(tmp_path, stage="actual_high_fd_census_cancellation", argv=argv,
                              env=dict(os.environ)) as record:
        expected = CensusUnavailable if failure == "deadline" else RuntimeError
        reason = "wall deadline" if failure == "deadline" else "original parser exception"
        with pytest.raises(expected, match=reason):
            P.capture(argv, environment=dict(os.environ), cwd=tmp_path, limits=limits, record=record, census=reader)
        assert record.census_started and record.census_cleanup_complete
    for path in (leader, descendant):
        assert path.is_file() and not Path("/proc", path.read_text()).exists()
    assert_guardian_reaped(opened)
