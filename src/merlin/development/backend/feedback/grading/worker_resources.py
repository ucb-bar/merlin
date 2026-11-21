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
import functools
import itertools
import json
import logging
import os
import select
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from merlin.analysis.programs.command_buffers import is_stored_name

# ---- sandbox_pool ------------------------------------------------------------------------------

ENV = "MERLIN_SANDBOX_REUSE"

KEY = "sandbox_reuse"

PERSISTENT, PER_CALL = "persistent", "per_call"

CHOICES = (PERSISTENT, PER_CALL)

START_TIMEOUT_S = 900.0

GRACE_S = 60.0

_ARITY = {
    **dict.fromkeys(
        (
            "--ro-bind",
            "--bind",
            "--ro-bind-try",
            "--bind-try",
            "--dev-bind",
            "--dev-bind-try",
            "--setenv",
            "--symlink",
            "--chmod",
            "--bind-data",
            "--ro-bind-data",
            "--file",
        ),
        2,
    ),
    **dict.fromkeys(
        (
            "--tmpfs",
            "--dir",
            "--proc",
            "--dev",
            "--chdir",
            "--unsetenv",
            "--remount-ro",
            "--args",
            "--perms",
            "--size",
            "--uid",
            "--gid",
            "--hostname",
            "--mqueue",
            "--userns",
            "--pidns",
            "--userns2",
            "--seccomp",
            "--add-seccomp-fd",
            "--block-fd",
            "--sync-fd",
            "--info-fd",
            "--json-status-fd",
            "--exec-label",
            "--file-label",
            "--cap-add",
            "--cap-drop",
            "--lock-file",
            "--argv0",
        ),
        1,
    ),
}

_MOUNTS = {
    "--ro-bind",
    "--bind",
    "--ro-bind-try",
    "--bind-try",
    "--dev-bind",
    "--dev-bind-try",
    "--tmpfs",
    "--dir",
    "--proc",
    "--dev",
    "--symlink",
    "--bind-data",
    "--ro-bind-data",
    "--file",
    "--mqueue",
}


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


_SWITCHES = {
    "--unshare-all",
    "--unshare-pid",
    "--unshare-net",
    "--unshare-ipc",
    "--unshare-uts",
    "--unshare-cgroup",
    "--unshare-cgroup-try",
    "--unshare-user",
    "--unshare-user-try",
    "--share-net",
    "--die-with-parent",
    "--new-session",
    "--as-pid-1",
    "--clearenv",
    "--disable-userns",
    "--assert-userns-disabled",
}


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
        from merlin.common.process import invocation_records as bwrap_args

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
                list(selected.command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                pass_fds=selected.pass_fds,
                env=dict(os.environ),
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
        return {
            "schema": "merlin.persistent_sandbox_call.v1",
            "sandbox": self.ident,
            "call": self.calls,
            "reset_roots": list(self.roots),
            "server": "pid-namespace reset per call",
        }

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
                print(
                    f"compiler sandbox reuse unavailable ({type(error).__name__}: {error}); "
                    "launching one sandbox per call",
                    file=sys.stderr,
                )
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


# ---- scratch_retention -------------------------------------------------------------------------
# Bound the disk a run's grade and self-check scratch can hold.
#
# Every background grade builds each capsule in ``_qa_work/fruns_<token>/`` and every self-check in its
# own raw runs root, and nothing removed either afterwards. Measured on one 12-hour run: 1-8 GB per
# grade, 18 GB of retained self-checks and 24 GB of scalability staging after 3.3 hours, until the
# volume filled and the grader died mid-write. Most of those bytes are build products -- a copied
# compiler tree per capsule, linked ELFs and objects, C harnesses with the inputs as literals -- that
# no reader opens once the grade is published.
#
# What the readers that outlive a grade actually open, and so what is kept:
#
# * ``capsule_result.json``, byte-identical: the correctness curriculum reuses a pass only when its
#   pinned result still verifies, scheduling history prices capsules from it, and the operator audits,
#   convergence report and calibration-index readers count from it -- together with the expanded command
#   buffer it references (:mod:`merlin.analysis.programs.command_buffers`), which used to be embedded in it.
#   A finished self-check drops that buffer: its only reader was the check itself, which has written its
#   response by then, and every later reader of the result (audits, timing archives) takes the tensor
#   shapes from the inline summary. It was 90% of a retained scalability check -- 1.1 GB of the 1.2 GB a
#   132-case check kept, once per check -- while the bytes it encodes are the inputs of the public suite
#   and the compiler's commands, both reproducible from the pinned source. Measured buffers (an RTL tier
#   ran) stay with the rest of their capsule directory, below;
# * every small file (receipts, reports, timing records, logs up to :data:`KEEP_FILE_BYTES`), and the
#   issued scalability qualification beside them;
# * every capsule directory holding paid RTL evidence (a tier derived from RTL that passed or failed),
#   untouched: the timing archive is ready only when every measured gSIM pass keeps its ELF, harness,
#   command buffer, trace and numeric report.
#
# L3 promotion freezes its own source snapshot before it enqueues, and calibration intake copies its
# result into the run's registry while the grade runs, so neither reads a grade's scratch later. The
# newest :data:`KEEP_NEWEST` grades and the one the agent's verdict was published from stay whole.
#
# Whatever the size rule says, a path a seal, release, run record or derivation cites is kept
# (:mod:`merlin.common.storage.retention`): the run's own records, plus the trees ``MERLIN_CITED_ROOTS``
# and the storage contract add -- study ledgers and papers that quote a grade's files by path. When
# that registry cannot be read nothing is removed, because a full volume is recoverable and a deleted
# citation is not.

SCHEMA = "merlin.scratch_retention.v1"

MARKER = "retention.json"

MIN_FREE_ENV = "MERLIN_GRADE_MIN_FREE_BYTES"

DEFAULT_MIN_FREE_BYTES = 25 << 30

KEEP_NEWEST = 3

KEEP_FILE_BYTES = 1 << 20

KEPT_NAMES = frozenset({"capsule_result.json", "issued.json", "qualification.json"})

BULK_DIRECTORIES = frozenset({"compiler-source", "executed"})

BULK_SUFFIXES = frozenset({".elf", ".o", ".a", ".so", ".bin"})

_RTL_EXECUTED = frozenset({"pass", "fail"})

_RESULT_BYTES_READ = 64 << 20

_TIERS_HEAD_BYTES = 4 << 20

CITED_MAX_AGE_S = 600.0


def say(message: str) -> None:
    """Report one line; a full log volume drops it instead of raising into the caller."""
    try:
        print(message, flush=True)
    except (OSError, ValueError):
        logging.getLogger(__name__).warning("Scratch retention stdout unavailable", exc_info=True)


def min_free_bytes() -> int:
    raw = os.environ.get(MIN_FREE_ENV, "").strip()
    if raw.isdigit():
        return int(raw)
    return DEFAULT_MIN_FREE_BYTES


def free_bytes(path) -> int | None:
    try:
        stat = os.statvfs(path)
    except OSError:
        return None
    return stat.f_bavail * stat.f_frsize


def free_space_ok(path, min_bytes: int | None = None) -> bool:
    """Whether the volume holding ``path`` has at least ``min_bytes`` free (unknown counts as enough)."""
    free = free_bytes(path)
    return free is None or free >= (min_free_bytes() if min_bytes is None else min_bytes)


def _leading_tiers(result: Path):
    """Parse only the ``tiers`` object of a result too large to load; it precedes the embedded artifacts.

    Raises ``ValueError`` when the object is not within the head, which callers treat as evidence."""
    with result.open("rb") as stream:
        head = stream.read(_TIERS_HEAD_BYTES).decode("utf-8", errors="replace")
    key = head.find('"tiers"')
    colon = head.find(":", key) if key >= 0 else -1
    if colon < 0:
        raise ValueError("tier record not found in the result head")
    start = colon + 1
    while start < len(head) and head[start] in " \t\r\n":
        start += 1
    tiers, _end = json.JSONDecoder().raw_decode(head, start)
    return tiers


def _paid_rtl_evidence(capsule_dir: Path) -> bool:
    """A tier derived from RTL actually ran here, or the result cannot be read to say otherwise."""
    result = capsule_dir / "capsule_result.json"
    try:
        if result.is_symlink() or not result.is_file():
            return False
        if result.stat().st_size > _RESULT_BYTES_READ:
            tiers = _leading_tiers(result)
        else:
            document = json.loads(result.read_text(encoding="utf-8"))
            tiers = document.get("tiers") if isinstance(document, dict) else None
    except (OSError, ValueError):
        return True
    if not isinstance(tiers, dict):
        return False
    return any(
        isinstance(record, dict) and record.get("derived_from_rtl") is True and record.get("status") in _RTL_EXECUTED
        for record in tiers.values()
    )


class _Kept:
    """Citations that keep scratch alive; ``None`` registry means unknown, so everything is kept."""

    def __init__(self, root: Path, extra: tuple[Path, ...] = ()):
        from merlin.common.storage import retention as cited_evidence

        self.root = root
        try:
            roots = list(dict.fromkeys([*extra, *cited_evidence.extra_roots()]))
            self.registry = cited_evidence.registry(roots, max_age=CITED_MAX_AGE_S) if roots else None
            self.readable = True
        except Exception as error:  # noqa: BLE001 -- an unreadable registry protects, it never deletes
            say(f"[retention] citation registry unreadable, keeping {root}: {type(error).__name__}: {error}")
            self.registry, self.readable = None, False

    def cited(self, path: Path) -> bool:
        if not self.readable:
            return True
        return bool(self.registry and self.registry.citations(path, ignore_records_under=self.root))


def _remove(path: Path) -> int:
    """Unlink one entry; returns the bytes this name was the last holder of."""
    try:
        stat = path.lstat()
        path.unlink()
    except OSError:
        return 0
    return stat.st_size if stat.st_nlink == 1 else 0


def _remove_tree(path: Path) -> int:
    freed = 0
    for parent, directories, files in os.walk(path, topdown=False):
        for name in files:
            freed += _remove(Path(parent) / name)
        for name in directories:
            entry = Path(parent) / name
            if entry.is_symlink():
                freed += _remove(entry)
            else:
                try:
                    entry.rmdir()
                except OSError:
                    logging.getLogger(__name__).warning(
                        "Scratch directory retained after cleanup failure", exc_info=True
                    )
    try:
        path.rmdir()
    except OSError:
        shutil.rmtree(path, ignore_errors=True)
    return freed


def slim_tree(root, *, citing: tuple[Path, ...] = (), keep_command_buffers: bool = True) -> int:
    """Remove build products under ``root`` and keep its evidence (see the module); returns bytes freed.

    Directory times are restored, so readers that date a check by its directories see the same times.
    Links are removed as names, never followed. ``citing`` adds trees whose records may cite ``root``.
    ``keep_command_buffers=False`` also removes stored expanded command buffers outside capsule
    directories with RTL-tier evidence (a finished self-check; a grade keeps them).
    """
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        return 0
    kept = _Kept(root, citing)
    if not kept.readable:
        return 0
    freed = 0
    times: dict[Path, tuple[int, int]] = {}
    for parent, directories, files in os.walk(root):
        here = Path(parent)
        try:
            stat = here.stat()
            times[here] = (stat.st_atime_ns, stat.st_mtime_ns)
        except OSError:
            logging.getLogger(__name__).warning(
                "Scratch retention could not inspect or restore a directory/file", exc_info=True
            )
            continue
        if (here / "capsule_result.json").is_file() and _paid_rtl_evidence(here):
            directories[:] = []
            continue
        for name in list(directories):
            entry = here / name
            if entry.is_symlink():
                directories.remove(name)
                if not kept.cited(entry):
                    freed += _remove(entry)
            elif name in BULK_DIRECTORIES and not kept.cited(entry):
                directories.remove(name)
                freed += _remove_tree(entry)
        for name in files:
            entry = here / name
            if name in KEPT_NAMES or kept.cited(entry):
                continue
            if is_stored_name(name):
                if not keep_command_buffers:
                    freed += _remove(entry)
                continue
            try:
                size = entry.lstat().st_size
            except OSError:
                logging.getLogger(__name__).warning(
                    "Scratch retention could not inspect or restore a directory/file", exc_info=True
                )
                continue
            if entry.is_symlink() or Path(name).suffix in BULK_SUFFIXES or size > KEEP_FILE_BYTES:
                freed += _remove(entry)
    for directory, (atime, mtime) in times.items():
        try:
            os.utime(directory, ns=(atime, mtime))
        except OSError:
            logging.getLogger(__name__).warning(
                "Scratch retention could not inspect or restore a directory/file", exc_info=True
            )
    return freed


def slim_once(root, *, scope: str, citing: tuple[Path, ...] = (), keep_command_buffers: bool = True) -> int:
    """:func:`slim_tree` a finished scratch root and mark it, so later passes skip it."""
    root = Path(root)
    marker = root / MARKER
    if root.is_symlink() or not root.is_dir() or marker.exists():
        return 0
    if not _Kept(root, citing).readable:
        return 0  # leave it unmarked so a later pass with a readable registry slims it
    started = time.time()
    freed = slim_tree(root, citing=citing, keep_command_buffers=keep_command_buffers)
    try:
        stat = root.stat()
        marker.write_text(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "scope": scope,
                    "slimmed_at_epoch": started,
                    "bytes_freed": freed,
                    "kept": f"capsule_result.json, files up to {KEEP_FILE_BYTES} bytes, issued qualifications, "
                    + ("stored command buffers, " if keep_command_buffers else "")
                    + "and every capsule directory with RTL-tier evidence",
                },
                indent=2,
            )
            + "\n"
        )
        os.utime(root, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    except OSError:
        logging.getLogger(__name__).warning("Scratch retention marker unavailable; next pass may retry", exc_info=True)
    return freed


_SELFCHECK_SCRATCH: ContextVar[list | None] = ContextVar("merlin_selfcheck_scratch", default=None)


def temporary(path) -> None:
    """Remove this self-check-owned directory (an isolated copy, a capsule subset) when the check ends."""
    held = _SELFCHECK_SCRATCH.get()
    if held is not None:
        held.append(("remove", Path(path)))


def raw_runs(path) -> None:
    """Slim this self-check's raw runs root when the check ends; its results and receipts stay."""
    held = _SELFCHECK_SCRATCH.get()
    if held is not None:
        held.append(("slim", Path(path)))


@contextmanager
def selfcheck_scratch():
    """Own the scratch a self-check registers; released after its response is written.

    Nothing reads these after the check returns: the broker reads only the response the check wrote,
    and promotion freezes its own source copy before it enqueues. So the raw runs also lose their
    stored command buffers (the module docstring says why that is safe).
    """
    held: list = []
    token = _SELFCHECK_SCRATCH.set(held)
    try:
        yield held
    finally:
        _SELFCHECK_SCRATCH.reset(token)
        for kind, path in reversed(held):
            try:
                if path.is_symlink() or not path.is_dir():
                    continue
                if kind == "remove":
                    if not _Kept(path).cited(path):
                        _remove_tree(path)
                else:
                    slim_once(path, scope="completed self-check", keep_command_buffers=False)
            except Exception as error:  # noqa: BLE001 -- cleanup never replaces the check's own outcome
                say(f"[retention] self-check scratch kept: {type(error).__name__}: {error}")


def releases_selfcheck_scratch(function):
    """Run ``function`` inside :func:`selfcheck_scratch`."""

    @functools.wraps(function)
    def call(*args, **kwargs):
        with selfcheck_scratch():
            return function(*args, **kwargs)

    return call


def _published_token(workspace) -> str | None:
    if workspace is None:
        return None
    try:
        document = json.loads((Path(workspace) / "qa" / ".snapshot-publication.json").read_text())
    except (OSError, ValueError):
        return None
    token = document.get("token") if isinstance(document, dict) else None
    return str(token) if token is not None else None


def prune_superseded_grades(run_dir, workspace=None, *, keep_newest: int = KEEP_NEWEST) -> int:
    """Slim every finished background grade older than the newest ``keep_newest`` and unpublished.

    Only grades whose owner record says ``complete`` or ``deferred`` are touched; one still admitting or
    draining, or with no owner record, is left whole, and so is one whose advisory timing is still
    being observed (:func:`.rolling_scalability.busy_runs`: it reads that grade's executables). Returns
    bytes freed.
    """
    from merlin.development.backend.feedback.scalability.rolling_checks import busy_runs

    work = Path(run_dir) / "_qa_work"
    owners = work / "background_snapshot_owners"
    records = {}
    try:
        entries = list(owners.iterdir())
    except OSError:
        return 0
    for entry in entries:
        if entry.suffix != ".json" or entry.name.endswith(".next.json") or not entry.stem.isdigit():
            continue
        try:
            document = json.loads(entry.read_text())
        except (OSError, ValueError):
            logging.getLogger(__name__).warning("Grade owner unreadable; retaining its scratch", exc_info=True)
            continue
        if isinstance(document, dict) and str(document.get("token")) == entry.stem:
            records[entry.stem] = document
    newest = sorted(records, key=int, reverse=True)
    protected = set(newest[: max(0, keep_newest)])
    published = _published_token(workspace)
    if published is not None:
        protected.add(published)
    observed = busy_runs(run_dir)
    freed = 0
    for token in newest:
        record = records[token]
        if token in protected or record.get("state") not in ("complete", "deferred"):
            continue
        runs = work / f"fruns_{token}"
        if str(runs.resolve()) in observed or str(runs) in observed:
            continue
        if record.get("runs_dir") not in (None, str(runs)):
            continue  # an owner naming another directory is not this run's scratch
        freed += slim_once(runs, scope="superseded background grade", citing=(Path(run_dir),))
    return freed


def admit_grade(run_dir, workspace=None) -> bool:
    """Whether a new background grade may start: enough free space, after pruning if needed."""
    if free_space_ok(run_dir):
        return True
    try:
        freed = prune_superseded_grades(run_dir, workspace)
    except Exception as error:  # noqa: BLE001 -- retention never stops the grader
        freed = 0
        say(f"[retention] pruning failed: {type(error).__name__}: {error}")
    if free_space_ok(run_dir):
        say(f"[retention] freed {freed >> 20} MiB of superseded grade scratch; admitting the next grade")
        return True
    free = free_bytes(run_dir)
    say(
        f"[retention] only {(free or 0) >> 30} GiB free (floor {min_free_bytes() >> 30} GiB); "
        f"no new background grade until space returns"
    )
    return False


if __name__ == "__main__":  # operator use: python -m ... <run_dir> [workspace]
    _freed = prune_superseded_grades(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    say(f"freed {_freed >> 20} MiB")


__all__ = [
    "DEFAULT_MIN_FREE_BYTES",
    "KEEP_NEWEST",
    "MIN_FREE_ENV",
    "admit_grade",
    "free_bytes",
    "free_space_ok",
    "min_free_bytes",
    "prune_superseded_grades",
    "raw_runs",
    "releases_selfcheck_scratch",
    "say",
    "selfcheck_scratch",
    "slim_once",
    "slim_tree",
    "temporary",
]
