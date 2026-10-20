"""Bounded streaming capture under the shared original child cleanup owner.

Only the dedicated forked guardian becomes a subreaper. Its private channel is
closed in the executed child; actual ECHILD and guardian exit precede completion.
"""

from __future__ import annotations

import json
import math
import os
import resource
import select
import selectors
import signal
import socket
import time
from types import SimpleNamespace

from merlin.common import owned_children as children
from merlin.runtime import _native_files as F

from ._dynamic_stream import CensusCleanupUnknown, CensusUnavailable


def _guardian(argv, environment, cwd, limits, control, owner_fd, stdout, stderr):
    stopped, process, child_fd, error = False, None, None, None

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    try:
        os.setsid()
        children.become_child_subreaper()
        pid = os.fork()
        if pid == 0:
            try:
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                os.dup2(stdout, 1)
                os.dup2(stderr, 2)
                null = os.open(os.devnull, os.O_RDONLY)
                os.dup2(null, 0)
                os.closerange(3, 1 << 20)
                resource.setrlimit(resource.RLIMIT_AS, (limits.max_ram_bytes, limits.max_ram_bytes))
                cpu = max(1, math.ceil(limits.wall_s))
                resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
                resource.setrlimit(resource.RLIMIT_FSIZE, (limits.max_output_bytes, limits.max_output_bytes))
                os.chdir(cwd)
                os.execve(argv[0], argv, environment)
            finally:
                os._exit(126)
        process = SimpleNamespace(pid=pid, returncode=None)
        os.close(stdout)
        os.close(stderr)
        child_fd = children.pidfd_open(pid)
        deadline = time.monotonic() + limits.wall_s
        while not stopped and time.monotonic() < deadline:
            ready = select.select([control, owner_fd, child_fd], [], [], 0.02)[0]
            if child_fd in ready or owner_fd in ready:
                break
            if control in ready:
                if control.recv(16) != b"cancel":
                    error = "invalid private cancellation"
                break
    except BaseException as caught:
        error = type(caught).__name__
    finally:
        if child_fd is not None:
            os.close(child_fd)
    # Unknown cleanup keeps the owner alive. The enclosing original job lease
    # may not be released while this child remains; a JSON roster is no proof.
    while True:
        cleanup = children.reap_owned_children(process, 4)
        if cleanup["complete"]:
            break
        time.sleep(0.02)
    packet = {"returncode": None if process is None else process.returncode, "cleanup": cleanup, "error": error}
    try:
        control.send(json.dumps(packet, sort_keys=True).encode())
        # As in the original feedback guardian, consume cancellation packets
        # before closing a seqpacket socket so its cleanup packet is retained.
        acknowledge_until = time.monotonic() + 1
        while time.monotonic() < acknowledge_until and not select.select([owner_fd], [], [], 0)[0]:
            if not select.select([control], [], [], 0.02)[0]:
                continue
            message = control.recv(16)
            if message == b"cleanup_received" or not message:
                break
    except OSError:
        pass
    control.close()
    os.close(owner_fd)
    os._exit(0 if error is None else 1)


def _writable_budget(roots, readonly_files, limits):
    seen, size = set(), 0
    for root in roots:
        for path in F.members(root, max_files=limits.max_output_files):
            if path in seen or path in readonly_files:
                continue
            seen.add(path)
            if len(seen) > limits.max_output_files or path.is_symlink():
                raise CensusUnavailable("census scratch exceeds file budget or contains a link")
            if path.is_file():
                size += path.stat().st_size
                if size > limits.max_output_bytes:
                    raise CensusUnavailable("census scratch exceeds sampled byte budget")


def capture(argv, *, environment, cwd, limits, record, census, writable_roots=(), readonly_files=(), require_done=True):
    """Internal fixed producer call; accepts no author callback or receipt."""
    record.census_cleanup_complete = False
    record.census_started = False
    files = {name: (record.directory / (name + ".bin")).open("xb") for name in ("stdout", "stderr")}
    owner_fd = children.pidfd_open(os.getpid())
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    out_read, out_write = os.pipe2(os.O_CLOEXEC)
    err_read, err_write = os.pipe2(os.O_CLOEXEC)
    try:
        guardian = os.fork()
    except BaseException:
        for stream in files.values():
            stream.close()
        parent.close()
        child.close()
        for descriptor in (owner_fd, out_read, out_write, err_read, err_write):
            os.close(descriptor)
        raise
    if guardian == 0:
        parent.close()
        os.close(out_read)
        os.close(err_read)
        _guardian(argv, environment, cwd, limits, child, owner_fd, out_write, err_write)
        os._exit(1)
    record.census_started = True
    child.close()
    for descriptor in (owner_fd, out_write, err_write):
        os.close(descriptor)
    try:
        guardian_fd = children.pidfd_open(guardian)
    except BaseException as error:
        parent.close()
        for descriptor in (out_read, err_read):
            os.close(descriptor)
        for stream in files.values():
            stream.close()
        raise CensusCleanupUnknown("census child owner pidfd is unavailable") from error
    buffers, size, packet, failure, done = {"stdout": b"", "stderr": b""}, 0, None, None, 0
    deadline, cancelled, next_sample = time.monotonic() + limits.wall_s, False, 0
    try:
        with selectors.DefaultSelector() as selector:
            for descriptor, name in ((out_read, "stdout"), (err_read, "stderr")):
                os.set_blocking(descriptor, False)
                selector.register(descriptor, selectors.EVENT_READ, name)
            selector.register(parent, selectors.EVENT_READ, "control")
            while selector.get_map():
                if not cancelled and time.monotonic() >= next_sample:
                    try:
                        _writable_budget(writable_roots, readonly_files, limits)
                    except ValueError as error:
                        failure = str(error)
                    next_sample = time.monotonic() + 0.05
                if not cancelled and time.monotonic() >= deadline:
                    failure = "census wall deadline expired"
                if failure is not None and not cancelled:
                    try:
                        parent.send(b"cancel")
                    except OSError:
                        pass
                    cancelled, deadline = True, time.monotonic() + 6
                if cancelled and time.monotonic() >= deadline:
                    break
                for key, _ in selector.select(0.02):
                    name = key.data
                    if name == "control":
                        raw = parent.recv(4096)
                        if raw:
                            packet = json.loads(raw)
                            parent.send(b"cleanup_received")
                        selector.unregister(parent)
                        continue
                    data = os.read(key.fd, 16384)
                    if not data:
                        if buffers[name] and name == "stderr" and failure is None:
                            failure = "census trace ended in an unterminated record"
                        selector.unregister(key.fd)
                        continue
                    if failure is not None:
                        continue  # Drain killed children without retaining excess data.
                    size += len(data)
                    if size > limits.max_output_bytes:
                        failure = "census raw output exceeds byte budget"
                        continue
                    files[name].write(data)
                    buffers[name] += data
                    while b"\n" in buffers[name]:
                        raw, buffers[name] = buffers[name].split(b"\n", 1)
                        try:
                            if len(raw) > limits.max_line_bytes:
                                raise CensusUnavailable("census line exceeds byte budget")
                            text = raw.decode("utf-8", errors="strict")
                            if name == "stderr":
                                census.line(text)
                            elif text.strip() == "DONE":
                                done += 1
                        except (ValueError, UnicodeError) as caught:
                            failure = str(caught)
                            break
                    if len(buffers[name]) > limits.max_line_bytes:
                        failure = "census line exceeds byte budget"
            closed = bool(packet and packet.get("cleanup", {}).get("complete"))
            if not select.select([guardian_fd], [], [], 6)[0]:
                closed = False
            else:
                _, status = os.waitpid(guardian, 0)
                closed = closed and os.waitstatus_to_exitcode(status) == 0
            if not closed:
                raise CensusCleanupUnknown("census cleanup remains UNKNOWN; original job owns retained descendants")
            record.census_cleanup_complete = True
            if failure is not None:
                raise CensusUnavailable(failure)
            if packet.get("returncode") != 0 or packet.get("error") or (require_done and done != 1):
                raise CensusUnavailable("census process or complete console did not finish successfully")
            return packet["returncode"]
    except BaseException as error:
        if record.census_cleanup_complete:
            raise
        try:
            parent.send(b"cancel")
        except OSError:
            pass
        if select.select([parent], [], [], 6)[0]:
            raw = parent.recv(4096)
            if raw:
                packet = json.loads(raw)
                parent.send(b"cleanup_received")
        if packet and packet.get("cleanup", {}).get("complete") and select.select([guardian_fd], [], [], 6)[0]:
            _, status = os.waitpid(guardian, 0)
            record.census_cleanup_complete = os.waitstatus_to_exitcode(status) == 0
        if not record.census_cleanup_complete:
            raise CensusCleanupUnknown("census cleanup remains UNKNOWN after cancellation") from error
        raise
    finally:
        for stream in files.values():
            stream.close()
        for descriptor in (out_read, err_read, guardian_fd):
            os.close(descriptor)
        parent.close()
