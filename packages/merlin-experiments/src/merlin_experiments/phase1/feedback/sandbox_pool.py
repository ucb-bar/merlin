"""Reuse one compiler sandbox per grading worker instead of starting one per call.

Every bwrap start copies the host mount table and performs its binds under the
kernel's global namespace lock, so hundreds of per-call starts serialize behind
each other. A worker instead keeps one sandbox, built from the same policy argv
a per-call launch would use, and runs each call through a tiny server inside it.

Per call the server kills every other process in the sandbox's PID namespace and
empties its fresh-per-call writable roots (tmpfs masks and ``/dev/shm``) back to
the skeleton bwrap created, never touching a mount point. The call's private IO
directory is staged in and out by the host, so it is empty between calls. A
reset that cannot complete discards the sandbox. The server is non-dumpable, so
a call cannot open its request/reply pipes through ``/proc``. Goldens and grader
state never enter the sandbox; only the policy's own grants do.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import os
import select
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ENV = "MERLIN_SANDBOX_REUSE"
KEY = "sandbox_reuse"
PERSISTENT, PER_CALL = "persistent", "per_call"
CHOICES = (PERSISTENT, PER_CALL)
START_TIMEOUT_S = 900.0
GRACE_S = 60.0

_ARITY = {
    **dict.fromkeys(("--ro-bind", "--bind", "--ro-bind-try", "--bind-try", "--dev-bind", "--dev-bind-try",
                     "--setenv", "--symlink", "--chmod", "--bind-data", "--ro-bind-data", "--file"), 2),
    **dict.fromkeys(("--tmpfs", "--dir", "--proc", "--dev", "--chdir", "--unsetenv", "--remount-ro", "--args",
                     "--perms", "--size", "--uid", "--gid", "--hostname", "--mqueue", "--userns", "--pidns",
                     "--userns2", "--seccomp", "--add-seccomp-fd", "--block-fd", "--sync-fd", "--info-fd",
                     "--json-status-fd", "--exec-label", "--file-label", "--cap-add", "--cap-drop",
                     "--lock-file", "--argv0"), 1),
}
_MOUNTS = {"--ro-bind", "--bind", "--ro-bind-try", "--bind-try", "--dev-bind", "--dev-bind-try", "--tmpfs",
           "--dir", "--proc", "--dev", "--symlink", "--bind-data", "--ro-bind-data", "--file", "--mqueue"}


def selected() -> bool:
    value = os.environ.get(ENV, PER_CALL)
    if value not in CHOICES:
        raise ValueError("unknown sandbox reuse selection")
    return value == PERSISTENT


def adopt_recorded(run_config) -> str:
    """A resumed run keeps the mechanism it started with; a record without one predates reuse."""
    value = (run_config or {}).get(KEY, PER_CALL) if isinstance(run_config, dict) else PER_CALL
    os.environ[ENV] = value if value in CHOICES else PER_CALL
    return os.environ[ENV]


_SWITCHES = {"--unshare-all", "--unshare-pid", "--unshare-net", "--unshare-ipc", "--unshare-uts",
             "--unshare-cgroup", "--unshare-cgroup-try", "--unshare-user", "--unshare-user-try", "--share-net",
             "--die-with-parent", "--new-session", "--as-pid-1", "--clearenv", "--disable-userns",
             "--assert-userns-disabled"}


def reset_roots(argv, io_destination, fresh_sources=None) -> tuple[str, ...] | None:
    """Fresh-per-call writable roots of a policy, or None when it cannot be reused safely.

    Roots are tmpfs destinations, ``/dev/shm`` and binds whose host source lies
    under ``fresh_sources`` (capped tmpfs replacements). Reuse requires a private
    PID namespace with its own /proc, so a reset sees and kills only sandbox
    processes, and nothing mounted below the IO destination, so the host may
    stage that directory freely.
    """
    io = str(io_destination)
    fresh = None if fresh_sources is None else str(fresh_sources) + "/"
    roots, pid_ns, proc, index = [], False, False, 1
    while index < len(argv):
        flag = argv[index]
        arity = 0 if flag in _SWITCHES else _ARITY.get(flag)
        if arity is None or index + arity >= len(argv):
            return None
        values = argv[index + 1 : index + 1 + arity]
        pid_ns = pid_ns or flag in {"--unshare-pid", "--unshare-all"}
        if flag in _MOUNTS:
            destination = values[-1]
            if destination.startswith(io + "/"):
                return None
            if flag == "--tmpfs":
                roots.append(destination)
            elif flag == "--proc":
                proc = destination == "/proc"
            elif flag == "--dev" and destination == "/dev":
                roots.append("/dev/shm")
            elif flag == "--bind" and fresh is not None and values[0].startswith(fresh) and destination != io:
                roots.append(destination)
        index += 1 + arity
    if not (pid_ns and proc):
        return None
    return tuple(dict.fromkeys(roots))


_SERVER = r"""
import codecs, ctypes, json, os, shutil, signal, subprocess, sys, time
ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)
me = os.getpid()
if me > 2:
    sys.exit(97)
roots = json.loads(sys.argv[1])
def mounts():
    found = set()
    with open("/proc/self/mountinfo", "rb") as stream:
        for line in stream:
            found.add(os.fsdecode(codecs.escape_decode(line.split()[4])[0]))
    return found
def skeleton(live):
    kept = {}
    def walk(path):
        kept[path] = os.lstat(path).st_mode
        if path in live and path not in roots:
            return
        if os.path.isdir(path) and not os.path.islink(path):
            for entry in os.scandir(path):
                walk(entry.path)
    for root in roots:
        if os.path.isdir(root):
            walk(root)
    return kept
mounted = mounts()
initial = skeleton(mounted)
def clear():
    for _ in range(200):
        victims = [int(n) for n in os.listdir("/proc") if n.isdigit() and int(n) not in (1, me)]
        if not victims:
            return True
        for pid in victims:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        try:
            while os.waitpid(-1, os.WNOHANG)[0]:
                pass
        except ChildProcessError:
            pass
        time.sleep(0.005)
    return False
def unlocked(function, path, _):
    os.chmod(os.path.dirname(path), 0o700)
    function(path)
def wipe(root, live):
    for entry in os.scandir(root):
        path = entry.path
        if path in live and path not in roots:
            continue
        if path in initial and entry.is_dir(follow_symlinks=False):
            os.chmod(path, 0o700)
            wipe(path, live)
            os.chmod(path, initial[path] & 0o7777)
        elif path not in initial:
            if entry.is_dir(follow_symlinks=False):
                os.chmod(path, 0o700)
                shutil.rmtree(path, onerror=unlocked)
            else:
                os.unlink(path)
def reset():
    if not clear():
        return "processes survived reset"
    live = mounts()
    if live != mounted:
        return "mount table changed"
    try:
        for root in roots:
            if os.path.isdir(root):
                os.chmod(root, 0o700)
                wipe(root, live)
                os.chmod(root, initial[root] & 0o7777)
    except OSError as error:
        return "writable root reset failed: " + type(error).__name__
    if skeleton(live).keys() != initial.keys():
        return "writable root kept call state"
    return None
def read(fd):
    os.lseek(fd, 0, 0)
    chunks = []
    while True:
        chunk = os.read(fd, 1 << 20)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(fd)
    return b"".join(chunks).decode("utf-8", "replace")
reply = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)
reply.write(json.dumps({"ready": me}) + "\n"); reply.flush()
for line in sys.stdin:
    request = json.loads(line)
    out, err = os.memfd_create("stdout"), os.memfd_create("stderr")
    result = {"returncode": None, "timed_out": False}
    try:
        child = subprocess.Popen(request["argv"], cwd=request["cwd"], stdin=subprocess.DEVNULL,
                                 stdout=out, stderr=err, start_new_session=True)
    except OSError as error:
        os.write(err, ("%s: %s\n" % (request["argv"][0], error.strerror)).encode())
        result["returncode"] = 126 if isinstance(error, PermissionError) else 127
    else:
        try:
            result["returncode"] = child.wait(timeout=request["timeout"])
        except subprocess.TimeoutExpired:
            result["timed_out"] = True
            child.kill()
            result["returncode"] = child.wait()
    result["reset_error"] = reset()
    result["stdout"], result["stderr"] = read(out), read(err)
    reply.write(json.dumps(result) + "\n"); reply.flush()
"""


def _clear_directory(path: Path) -> None:
    for entry in list(path.iterdir()):
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry, onerror=lambda f, p, _: (os.chmod(os.path.dirname(p), 0o700), f(p)))
        else:
            entry.unlink()


class Unavailable(RuntimeError):
    """No reusable sandbox could start; the caller launches this call on its own."""


class _Box:
    def __init__(self, pool, key, ident):
        self.pool, self.key, self.ident, self.calls = pool, key, ident, 0
        self.directory = pool.root / ident
        self.io = self.directory / "io"
        self.io.mkdir(parents=True, mode=0o700)
        self.stack = contextlib.ExitStack()
        self.process, self.broken, self._buffer = None, False, b""

    def start(self):
        from merlin.common import bwrap_args

        policy, payload = self.pool.launch(self.io, self.key, self.stack)
        self.roots = reset_roots(policy, self.pool.io_destination, self.directory)
        if self.roots is None:
            raise ValueError("compiler sandbox policy cannot be reused")
        self.prefix = (*policy, *payload)
        command = [*self.prefix, "python3", "-I", "-S", "-c", _SERVER, json.dumps(self.roots)]
        stderr = (self.directory / "server.stderr").open("wb")
        with bwrap_args.argument_transport(command, policy_end=len(policy)) as selected:
            self.transport = selected.record()
            self.process = subprocess.Popen(
                list(selected.command), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                pass_fds=selected.pass_fds, env=dict(os.environ),
            )
        stderr.close()
        ready = self._reply(time.monotonic() + START_TIMEOUT_S)
        if not isinstance(ready, dict) or "ready" not in ready:
            raise RuntimeError("persistent compiler sandbox did not start")
        return self

    def _reply(self, deadline):
        descriptor = self.process.stdout.fileno()
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([descriptor], [], [], remaining)[0]:
                return None
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                return None
            self._buffer += chunk
        line, _, self._buffer = self._buffer.partition(b"\n")
        return json.loads(line)

    def logical_argv(self, argv):
        return [*self.prefix, *argv]

    def record(self):
        return {"schema": "merlin.persistent_sandbox_call.v1", "sandbox": self.ident, "call": self.calls,
                "reset_roots": list(self.roots), "server": "pid-namespace reset per call"}

    def _stage_out(self, scratch):
        for entry in list(self.io.iterdir()):
            target = scratch / entry.name
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            elif target.exists() or target.is_symlink():
                target.unlink()
            try:
                shutil.move(str(entry), str(target))
            except (OSError, shutil.Error):
                self.broken = True
        if any(self.io.iterdir()):
            _clear_directory(self.io)

    def run(self, argv, scratch, *, cwd, timeout):
        scratch = Path(scratch)
        if any(self.io.iterdir()):
            raise RuntimeError("persistent compiler sandbox IO was not emptied")
        for entry in scratch.iterdir():
            target = self.io / entry.name
            if entry.is_dir() and not entry.is_symlink():
                shutil.copytree(entry, target, symlinks=True)
            else:
                shutil.copy2(entry, target, follow_symlinks=False)
        self.calls += 1
        logical = self.logical_argv(argv)
        reply = None
        try:
            request = json.dumps({"argv": list(argv), "cwd": str(cwd), "timeout": timeout}) + "\n"
            self.process.stdin.write(request.encode())
            self.process.stdin.flush()
            reply = self._reply(time.monotonic() + (timeout or START_TIMEOUT_S) + GRACE_S)
        except (OSError, ValueError):
            reply = None
        finally:
            self._stage_out(scratch)
        if reply is None:
            self.broken = True
            if self.process.poll() is None:
                raise subprocess.TimeoutExpired(logical, timeout)
            return subprocess.CompletedProcess(logical, -9, "", "persistent compiler sandbox ended during the call")
        if reply.get("reset_error"):
            self.broken = True
        if reply["timed_out"]:
            raise subprocess.TimeoutExpired(logical, timeout, output=reply["stdout"], stderr=reply["stderr"])
        return subprocess.CompletedProcess(logical, reply["returncode"], reply["stdout"], reply["stderr"])

    def alive(self):
        return not self.broken and self.process is not None and self.process.poll() is None

    def close(self):
        try:
            if self.process is not None:
                if self.process.poll() is None:
                    self.process.kill()
                self.process.wait()
                for stream in (self.process.stdin, self.process.stdout):
                    with contextlib.suppress(OSError):
                        stream.close()
        finally:
            try:
                self.stack.close()
            finally:
                shutil.rmtree(self.directory, ignore_errors=True)


class SandboxPool:
    """Idle sandboxes for one executor; a worker takes one, runs a call, and returns it.

    ``launch(io, key, stack)`` returns the per-call policy argv (with ``io`` bound
    at ``io_destination``) and the payload prefix; contexts it needs for the
    sandbox lifetime go on ``stack``. Starts and closes run on one long-lived
    thread so ``--die-with-parent`` never fires when a worker thread exits.
    """

    def __init__(self, root: Path, io_destination: Path, launch):
        self.root, self.io_destination, self.launch = Path(root), Path(io_destination), launch
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._idle, self._lock, self._closed = [], threading.Lock(), False
        self._owner = ThreadPoolExecutor(1, thread_name_prefix="compiler-sandbox")
        self._ids = itertools.count()
        self.started = 0
        self.disabled = False

    def _start(self, key):
        box = _Box(self, key, f"box-{next(self._ids)}-{uuid.uuid4().hex[:8]}")
        try:
            return box.start()
        except BaseException:
            box.close()
            raise

    @contextlib.contextmanager
    def box(self, key=None):
        with self._lock:
            if self._closed:
                raise ValueError("compiler sandbox pool is closed")
            box = next((b for b in self._idle if b.key is key), None)
            if box is not None:
                self._idle.remove(box)
        if box is None:
            if self.disabled:
                raise Unavailable("compiler sandbox reuse is disabled for this executor")
            try:
                box = self._owner.submit(self._start, key).result()
            except Exception as error:
                self.disabled = True
                print(f"compiler sandbox reuse unavailable ({type(error).__name__}: {error}); "
                      "launching one sandbox per call", file=sys.stderr)
                raise Unavailable(str(error)) from error
            with self._lock:
                self.started += 1
        try:
            yield box
        finally:
            with self._lock:
                if box.alive() and not self._closed:
                    self._idle.append(box)
                    box = None
            if box is not None:
                self._owner.submit(box.close).result()

    def close(self):
        with self._lock:
            self._closed, idle, self._idle = True, self._idle, []
        try:
            for box in idle:
                self._owner.submit(box.close).result()
        finally:
            self._owner.shutdown(wait=True)
