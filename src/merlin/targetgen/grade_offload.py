"""Run the grader's pure-Python calculations in worker processes instead of on grading threads.

A grade runs each capsule on a thread of one process, and the golden, reference and simulate
engines, the numeric comparison and the tree hashing are pure Python. Threads of one process
serialize on the interpreter lock: measured on a 20-CPU run, 14 capsules in flight held one core
between them while the other leased CPUs sat idle, and a 16x16x16 matmul waited 16 minutes behind
the large ones. :func:`call` runs a registered calculation in a worker process instead, on the CPU the
caller's compute lease pinned it to, so slot accounting is unchanged and only the interpreter differs.

What runs out of process is deliberately narrow:

* only functions passed to :func:`register`, matched by identity (a monkeypatched replacement runs in
  process). Each takes and returns plain data and reads nothing but its arguments and files;
* only when the arguments pickle;
* a call that raises in the worker, or whose worker dies, runs again in process, so the caller sees
  exactly the in-process result or exception, never a transport error.
* a call that asks for a CPU slot in the worker runs again in process too: the caller holds its slot
  while it waits, so the worker refuses every slot request instead of queueing behind other blocked
  holders, and the in-process run reuses the caller's lease.

A worker is ``python -m merlin.targetgen.grade_offload`` speaking length-prefixed pickles on two
dedicated pipes (its stdout goes to stderr, so a calculation that prints cannot corrupt a reply). It
never imports the caller's ``__main__``, imports through the caller's ``sys.path``, inherits no
descriptor but its pipes (no CPU-slot lock survives into it), keeps every calculation in process
itself, runs each call in the caller's working directory on the caller's CPUs, and is replaced after
:data:`TASKS_PER_WORKER` calls.
``MERLIN_GRADE_CPU_PROCESSES=0`` keeps every calculation in process.
"""

from __future__ import annotations

import importlib
import os
import pickle
import struct
import subprocess
import sys
import threading
from collections.abc import Callable
from typing import Any

ENV = "MERLIN_GRADE_CPU_PROCESSES"
#: Calls a worker serves before it is replaced (bounds what a long grade can accumulate in one).
TASKS_PER_WORKER = 64
#: Idle workers kept for reuse; more are started on demand, one per concurrent call.
MAX_IDLE = 64
_FDS_ENV = "MERLIN_GRADE_OFFLOAD_FDS"
_HEADER = struct.Struct("!Q")

_REGISTERED: dict[int, tuple[Callable, str, str]] = {}
_LOCK = threading.Lock()
_IDLE: list[_Worker] = []


def register(*functions: Callable) -> None:
    """Allow these functions to run in a worker process; one a worker cannot import stays in process."""
    for function in functions:
        module, qualname = getattr(function, "__module__", None), getattr(function, "__qualname__", "")
        if module and module != "__main__" and qualname and "<" not in qualname:
            _REGISTERED[id(function)] = (function, module, qualname)


def enabled() -> bool:
    return os.environ.get(ENV, "1").strip() != "0" and _FDS_ENV not in os.environ


def _send(fd: int, value: Any) -> None:
    data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    view = memoryview(_HEADER.pack(len(data)) + data)
    while view:
        view = view[os.write(fd, view) :]


def _receive(fd: int) -> Any:
    def exactly(n: int) -> bytes:
        chunks, remaining = [], n
        while remaining:
            chunk = os.read(fd, min(remaining, 1 << 20))
            if not chunk:
                raise EOFError("worker pipe closed")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    (size,) = _HEADER.unpack(exactly(_HEADER.size))
    return pickle.loads(exactly(size))


class _Worker:
    def __init__(self) -> None:
        to_child, self._out = os.pipe()
        self._in, from_child = os.pipe()
        env = dict(os.environ)
        env[_FDS_ENV] = f"{to_child},{from_child}"
        env[ENV] = "0"
        # Resolve imports exactly as the caller does, including paths it added at run time.
        env["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)
        try:
            self._process = subprocess.Popen(
                [sys.executable, "-m", __name__],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=sys.stderr.fileno() if hasattr(sys.stderr, "fileno") else None,
                pass_fds=(to_child, from_child),
                close_fds=True,
            )
        except BaseException:
            for fd in (to_child, from_child, self._out, self._in):
                os.close(fd)
            raise
        os.close(to_child)
        os.close(from_child)
        self.calls = 0

    def run(self, request: tuple) -> tuple[bool, Any]:
        _send(self._out, request)
        self.calls += 1
        return _receive(self._in)

    def close(self) -> None:
        for fd in (self._out, self._in):
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()


def _take() -> _Worker:
    with _LOCK:
        if _IDLE:
            return _IDLE.pop()
    return _Worker()


def _give(worker: _Worker) -> None:
    with _LOCK:
        if worker.calls < TASKS_PER_WORKER and len(_IDLE) < MAX_IDLE:
            _IDLE.append(worker)
            return
    worker.close()


def call(function: Callable, /, *args, **kwargs):
    """``function(*args, **kwargs)``, computed in a worker process when that is safe (see the module)."""
    entry = _REGISTERED.get(id(function))
    if entry is None or entry[0] is not function or not enabled():
        return function(*args, **kwargs)
    try:
        payload = pickle.dumps((args, kwargs), protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:  # noqa: BLE001 -- unpicklable arguments stay in process
        return function(*args, **kwargs)
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = []
    try:
        cwd = os.getcwd()
    except OSError:
        cwd = ""
    try:
        worker = _take()
    except OSError:
        return function(*args, **kwargs)
    try:
        ok, value = worker.run((entry[1], entry[2], payload, cpus, cwd))
    except BaseException:  # noqa: BLE001 -- a dead or unreadable worker: the in-process call decides
        worker.close()
        return function(*args, **kwargs)
    _give(worker)
    if not ok:
        return function(*args, **kwargs)
    return value


def shutdown() -> None:
    """Stop the idle workers (tests; a grader process otherwise keeps them for its lifetime)."""
    with _LOCK:
        workers = list(_IDLE)
        _IDLE.clear()
    for worker in workers:
        worker.close()


def _resolve(module: str, qualname: str) -> Callable:
    value: Any = importlib.import_module(module)
    for part in qualname.split("."):
        value = getattr(value, part)
    return value


def _serve() -> int:
    """Worker loop: one request, one ``(ok, result)`` reply, until the parent closes the pipe."""
    read_fd, write_fd = (int(x) for x in os.environ[_FDS_ENV].split(","))
    from merlin.common.cpu_pool import refuse_slot_waits, slot_wait_refusals

    # The caller holds a CPU slot for this call and is blocked on the reply, so a slot request here
    # could only wait behind other blocked holders. It raises instead; the caller then runs the
    # call in its own thread, reusing its lease.
    refuse_slot_waits("a grade offload worker runs on its caller's CPU slot")
    while True:
        try:
            module, qualname, payload, cpus, cwd = _receive(read_fd)
        except EOFError:
            return 0
        try:
            if cpus:
                os.sched_setaffinity(0, cpus)
            if cwd and os.getcwd() != cwd:
                os.chdir(cwd)
            args, kwargs = pickle.loads(payload)
            refused = slot_wait_refusals()
            value = _resolve(module, qualname)(*args, **kwargs)
            # A refused slot request may have been caught and answered with a degraded result.
            reply = (True, value) if slot_wait_refusals() == refused else (False, None)
        except BaseException:  # noqa: BLE001 -- the caller re-runs it, where the exception is its own
            reply = (False, None)
        try:
            _send(write_fd, reply)
        except Exception:  # noqa: BLE001 -- e.g. an unpicklable result: report failure instead
            _send(write_fd, (False, None))


__all__ = ["ENV", "call", "enabled", "register", "shutdown"]


if __name__ == "__main__":
    raise SystemExit(_serve())
