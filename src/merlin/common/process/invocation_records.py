"""Records of invoked processes: exact file/tool/source pins, complete streams, pin replay, ``run_checked``
and bubblewrap argv transport.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import inspect
import itertools
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
import weakref
from collections.abc import Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

# ---- invocation_record -------------------------------------------------------------------------
# Retain exact inputs and products at an actually invoked process boundary.
#
# These observations confer no correctness or stage applicability. Callers own
# which inputs/dependencies are complete and the verifier owns semantic lift.
# Records are written outside invoked packages and never accepted from stdout.

SCHEMA = "merlin.invocation_record.v1"
ARTIFACT_REUSE_SCHEMA = "merlin.artifact_reuse.v1"
_TIMING_LISTENERS = ContextVar("merlin_invocation_timing_listeners", default=())
TIMING_SCHEMA = "merlin.invocation_wall_timing.v1"


@contextlib.contextmanager
def observe_timings():
    """Collect same-context host wall intervals, without interpreting hardware costs."""
    events = []
    token = _TIMING_LISTENERS.set((*_TIMING_LISTENERS.get(), events))
    try:
        yield events
    finally:
        _TIMING_LISTENERS.reset(token)


@contextlib.contextmanager
def timed_event(kind: str, label: str):
    """Observe one original boundary; nested intervals must not be summed blindly."""
    listeners = _TIMING_LISTENERS.get()
    if not listeners:
        yield
        return
    event = {"kind": kind, "label": label, "pid": os.getpid(), "started_ns": time.monotonic_ns(), "status": "returned"}
    try:
        yield
    except BaseException:
        event["status"] = "failed"
        raise
    finally:
        event["finished_ns"] = time.monotonic_ns()
        for events in listeners:
            events.append(dict(event))


def timing_document(events):
    """No cycle, CPU-time, complete-wait or transitive-child coverage follows."""
    return {"schema": TIMING_SCHEMA, "clock": "same-process monotonic_ns", "events": list(events)}


def verify_timing_document(document, *, root_label=None):
    """Validate optional diagnostic intervals; file pins still own artifact custody."""
    if (
        not isinstance(document, dict)
        or set(document) != {"schema", "clock", "events"}
        or document["schema"] != TIMING_SCHEMA
        or document["clock"] != "same-process monotonic_ns"
        or type(document["events"]) is not list
    ):
        raise ValueError("invocation timing has no versioned host-wall observation")
    for event in document["events"]:
        if (
            not isinstance(event, dict)
            or set(event) != {"kind", "label", "pid", "started_ns", "finished_ns", "status"}
            or event["kind"] not in {"phase", "process", "compute_acquire"}
            or not isinstance(event["label"], str)
            or not event["label"]
            or event["status"] not in {"returned", "failed"}
            or any(type(event[k]) is not int for k in ("pid", "started_ns", "finished_ns"))
            or event["pid"] < 1
            or not 0 <= event["started_ns"] <= event["finished_ns"]
        ):
            raise ValueError("invocation timing has a malformed or negative host-wall interval")
    if root_label is not None:
        events = document["events"]
        if not events or events[-1]["kind"] != "phase" or events[-1]["label"] != root_label:
            raise ValueError("invocation timing has no enclosing original phase")
        root = events[-1]
        if any(
            event["pid"] != root["pid"]
            or event["started_ns"] < root["started_ns"]
            or event["finished_ns"] > root["finished_ns"]
            for event in events
        ):
            raise ValueError("invocation timing escapes its same-process phase interval")
    return document


def _pin(path: Path) -> dict:
    path = Path(path).resolve()
    try:
        if not path.is_file():
            raise OSError("not a regular file")
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        after = path.stat()
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode")
        if any(getattr(before, name) != getattr(after, name) for name in fields):
            raise OSError("observation member changed during its read")
        return {"path": str(path), "sha256": digest.hexdigest()}
    except OSError as error:
        # An observation must not replace the process's own missing-tool/input
        # failure. A verifier cannot admit an unavailable pin.
        return {"path": str(path), "sha256": None, "unavailable": type(error).__name__}


def _pins(paths) -> list[dict]:
    return [_pin(path) for path in sorted({Path(path).resolve() for path in paths})]


def environment_identity(environment) -> dict:
    """Bind actual process environment bytes without retaining their values.

    POSIX environment keys and values use the filesystem encoding, as Python's
    process launcher does. The complete sorted mapping is hashed with lengths
    and a domain separator. This is an observation, not a replay environment,
    dependency closure, secret-management mechanism or sandbox qualification.
    """
    members = {}
    for key, value in environment.items():
        encoded_key, encoded_value = os.fsencode(key), os.fsencode(value)
        if (
            not encoded_key
            or b"=" in encoded_key
            or b"\0" in encoded_key
            or b"\0" in encoded_value
            or encoded_key in members
        ):
            raise ValueError("environment has an invalid or ambiguous process mapping")
        members[encoded_key] = encoded_value
    digest = hashlib.sha256(b"merlin.process_environment.v1\0")
    for key, value in sorted(members.items()):
        for part in (key, value):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return {
        "schema": "merlin.process_environment.v1",
        "sha256": digest.hexdigest(),
        "keys": [os.fsdecode(key) for key in sorted(members)],
        "scope": "complete effective mapping; values withheld; transitive dependencies unproved",
    }


class Invocation:
    def __init__(
        self,
        directory: Path,
        *,
        stage: str,
        argv,
        cwd=None,
        env=None,
        inputs=(),
        outputs=(),
        dependencies=(),
    ):
        if not stage or not argv:
            raise ValueError("invocation observation requires a stage and command")
        self.directory = Path(directory).resolve() / "invocations" / uuid.uuid4().hex
        self.directory.mkdir(parents=True, mode=0o700)
        self.outputs = tuple(Path(path).resolve() for path in outputs)
        self.inputs = tuple(Path(path).resolve() for path in inputs)
        self.dependencies = tuple(Path(path).resolve() for path in dependencies)
        selected_env = dict(os.environ if env is None else env)
        command = [str(token) for token in argv]
        work = Path.cwd() if cwd is None else Path(cwd).resolve()
        executable = Path(command[0])
        if not executable.is_absolute():
            if os.path.dirname(command[0]):
                executable = work / executable
            else:
                # Relative search directories belong to the child's cwd.
                search = [str(work / entry) for entry in os.get_exec_path(selected_env)]
                found = shutil.which(command[0], path=os.pathsep.join(search))
                executable = Path(found) if found else work / executable
        self.document = {
            "schema": SCHEMA,
            "stage": stage,
            "argv": command,
            "cwd": str(work),
            "kind": "subprocess",
            "executable": _pin(executable),
            "inputs": _pins(self.inputs),
            "dependencies": _pins(self.dependencies),
            "status": "running",
            "started_ns": time.time_ns(),
            "environment": environment_identity(selected_env),
        }
        self.path = self.directory / "invocation.json"
        self._write()

    def _write(self):
        self.path.write_text(json.dumps(self.document, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    def complete(self, result: subprocess.CompletedProcess):
        for name in ("stdout", "stderr"):
            value = getattr(result, name, None)
            data = value if isinstance(value, bytes) else (value or "").encode("utf-8")
            path = self.directory / (name + ".bin")
            path.write_bytes(data)
        self.complete_streams(result.returncode)

    def complete_streams(self, returncode: int):
        """Seal streams written incrementally by the actual process caller.

        The caller owns bounded capture and real process cleanup. Existing raw
        stream files are pinned without loading a full trace into RAM.
        """
        if type(returncode) is not int:
            raise ValueError("streamed invocation requires its actual process exit status")
        for name in ("stdout", "stderr"):
            self.document[name] = _pin(self.directory / (name + ".bin"))
        self.document.update(
            status="completed" if returncode == 0 else "failed",
            returncode=returncode,
            finished_ns=time.time_ns(),
            outputs=_pins(path for path in self.outputs if path.is_file()),
            inputs_unchanged=_pins(self.inputs) == self.document["inputs"],
            dependencies_unchanged=_pins(self.dependencies) == self.document["dependencies"],
            executable_unchanged=_pin(Path(self.document["executable"]["path"])) == self.document["executable"],
        )
        self._write()

    def failed(self, error: BaseException):
        if self.document["status"] == "running":
            self.document.update(status="interrupted", error=type(error).__name__, finished_ns=time.time_ns())
            self._write()


class CallInvocation(Invocation):
    """An observed Python dispatch, distinct from a subprocess command claim."""

    def __init__(
        self, directory: Path, *, stage: str, function, arguments: dict, inputs=(), outputs=(), dependencies=()
    ):
        try:
            origin = inspect.getsourcefile(function)
        except TypeError:
            origin = None
        # An uninspectable callable remains an unavailable observation; it does
        # not turn optional lineage into a new functional refusal.
        source = origin or "/unavailable-callable-source"
        super().__init__(
            directory,
            stage=stage,
            argv=(source,),
            inputs=inputs,
            outputs=outputs,
            dependencies=(*dependencies, Path(source)),
        )
        name = (
            f"{getattr(function, '__module__', type(function).__module__)}."
            f"{getattr(function, '__qualname__', type(function).__qualname__)}"
        )
        self.document.update(
            kind="python_call", callable=name, callable_source_available=origin is not None, arguments=arguments
        )
        self.document.pop("argv")
        self.document.pop("environment")
        self._write()

    def returned(self, *, stdout=b"", stderr=b""):
        self.complete(subprocess.CompletedProcess([], 0, stdout=stdout, stderr=stderr))
        self.document.pop("returncode")
        self.document["outcome"] = "returned"
        self._write()


@contextlib.contextmanager
def observe(directory: Path, **kwargs):
    """Observe only a caller's actual invocation, including failed boundaries."""
    invocation = Invocation(directory, **kwargs)
    try:
        yield invocation
    except BaseException as error:
        invocation.failed(error)
        raise
    finally:
        if invocation.document["status"] == "running":
            invocation.failed(RuntimeError("invocation did not report completion"))


def run(
    argv,
    *,
    directory: Path,
    stage: str,
    inputs=(),
    outputs=(),
    dependencies=(),
    owned_process_tree: bool = False,
    **kwargs,
):
    # Observe and execute the same immutable snapshot, even if a caller's
    # mutable mapping or ambient environment changes at the observation point.
    kwargs["env"] = dict(os.environ if kwargs.get("env") is None else kwargs["env"])
    runner = subprocess.run
    if owned_process_tree:
        import sys

        from merlin.common.process import ownership

        runner = ownership.run
        dependencies = (*dependencies, Path(ownership.__file__), Path(sys.executable))
    with observe(
        directory,
        stage=stage,
        argv=argv,
        inputs=inputs,
        outputs=outputs,
        dependencies=dependencies,
        cwd=kwargs.get("cwd"),
        env=kwargs.get("env"),
    ) as record:
        try:
            result = runner(argv, **kwargs)
        except BaseException as error:
            if owned_process_tree and (cleanup := getattr(error, "owned_process_cleanup", None)):
                record.document["process_tree_cleanup"] = cleanup
                record._write()
            raise
        if owned_process_tree:
            record.document["process_tree_cleanup"] = result.owned_process_cleanup
        record.complete(result)
        return result


@contextlib.contextmanager
def observe_call(directory: Path, **kwargs):
    record = CallInvocation(directory, **kwargs)
    try:
        yield record
    except BaseException as error:
        record.failed(error)
        raise
    finally:
        if record.document["status"] == "running":
            record.failed(RuntimeError("call did not report its return"))


def verify(path: Path, *, pin_replay=None) -> dict:
    """Reopen exact observation bytes; completeness/applicability remain caller-owned."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    returned = (
        document.get("returncode") == 0
        if document.get("kind") == "subprocess"
        else document.get("outcome") == "returned"
    )
    unchanged = ("inputs_unchanged", "dependencies_unchanged", "executable_unchanged")
    if (
        document.get("schema") != SCHEMA
        or document.get("status") != "completed"
        or not returned
        or any(document.get(name) is not True for name in unchanged)
    ):
        raise ValueError("invocation did not complete with unchanged observed inputs")
    roles = [(document["executable"], True), (document["stdout"], False), (document["stderr"], False)]
    roles.extend((pin, False) for pin in (*document["inputs"], *document["outputs"]))
    roles.extend((pin, True) for pin in document["dependencies"])
    for pin, selected_role in roles:
        actual = None
        if selected_role and pin_replay is not None:
            actual = replayed_pin(pin_replay, Path(pin["path"]))
        if actual is None:
            actual = _pin(Path(pin["path"]))
        if not pin.get("sha256") or actual != pin:
            raise ValueError("invocation input, dependency or product changed")
    return document


def require_environment(path: Path, *, environment, pin_replay=None) -> dict:
    """Reopen an actual successful process and compare its protected selection.

    Older observations without this binding cannot establish environment
    identity. A matching saved record alone confers no live execution authority.
    """
    document = verify(path, pin_replay=pin_replay)
    if document.get("kind") != "subprocess" or document.get("environment") != environment_identity(environment):
        raise ValueError("invocation has no matching effective process environment binding")
    return document


def record_artifact_reuse(
    directory: Path, *, stage: str, selection_sha256: str, original_invocation: Path, inputs=(), outputs=()
) -> Path:
    """Record restored bytes separately from an actual process invocation.

    The live build owner must authenticate the original closed process before
    calling. This byte receipt issues no runtime closure, new invocation,
    compiler timing, repeat or numerical verdict.
    """
    original = verify(original_invocation)
    if original["kind"] != "subprocess" or not stage:
        raise ValueError("artifact reuse requires its original completed process")
    if (
        type(selection_sha256) is not str
        or len(selection_sha256) != 64
        or any(char not in "0123456789abcdef" for char in selection_sha256)
    ):
        raise ValueError("artifact reuse requires its complete selected build identity")
    document = {
        "schema": ARTIFACT_REUSE_SCHEMA,
        "kind": "artifact_reuse",
        "stage": stage,
        "status": "restored",
        "selection_sha256": selection_sha256,
        "original_invocation": _pin(original_invocation),
        "inputs": _pins(inputs),
        "outputs": _pins(outputs),
        "observed_ns": time.time_ns(),
        "new_process": False,
    }
    original_hashes = {pin["sha256"] for pin in original["outputs"]}
    if not document["outputs"] or any(pin["sha256"] not in original_hashes for pin in document["outputs"]):
        raise ValueError("restored artifact has no matching original completed product")
    encoded = json.dumps(document, sort_keys=True, indent=2) + "\n"
    if len(encoded.encode()) > 1024 * 1024:
        raise ValueError("artifact reuse receipt exceeds its byte budget")
    parent = Path(directory).resolve() / "invocations" / uuid.uuid4().hex
    parent.mkdir(parents=True, mode=0o700)
    path = parent / "artifact_reuse.json"
    path.write_text(encoded, encoding="utf-8")
    verify_artifact_reuse(path)
    return path


def verify_artifact_reuse(path: Path) -> dict:
    """Reopen byte provenance; saved receipts never issue a live cache owner."""
    path = Path(path)
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("artifact reuse receipt exceeds its byte budget")
    document = json.loads(path.read_text())
    if (
        document.get("schema") != ARTIFACT_REUSE_SCHEMA
        or document.get("kind") != "artifact_reuse"
        or document.get("status") != "restored"
        or document.get("new_process") is not False
        or any(name in document for name in ("argv", "returncode", "started_ns", "finished_ns", "timing"))
    ):
        raise ValueError("artifact reuse receipt claims an unrelated process")
    for pin in (*document["inputs"], *document["outputs"], document["original_invocation"]):
        if not pin.get("sha256") or _pin(Path(pin["path"])) != pin:
            raise ValueError("artifact reuse input, restored product or original record changed")
    original = verify(Path(document["original_invocation"]["path"]))
    if original["kind"] != "subprocess":
        raise ValueError("artifact reuse has no actual original process")
    original_hashes = {pin["sha256"] for pin in original["outputs"]}
    if not document["outputs"] or any(pin["sha256"] not in original_hashes for pin in document["outputs"]):
        raise ValueError("artifact reuse differs from its original completed products")
    return document


# ---- selected_pin_replay -----------------------------------------------------------------------
# Explicit content reuse inside one bounded thread and process replay call.
#
# Only caller-selected canonical files participate. Actual process observation
# never uses this owner. Complete content reads before and after the call bound
# the reuse; every record role and every unselected file remains caller-owned.
# This is a verification cost control, not source or stage admission.

_ACTIVE = weakref.WeakSet()


def _canonical(path):
    path = Path(path)
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.resolve() != path
        or any(item.is_symlink() for item in (path, *path.parents))
    ):
        raise ValueError("selected replay pins require explicit canonical regular files")
    return path


def _fresh_pin(path):
    path = _canonical(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@dataclass(frozen=True, eq=False)
class SelectedPinReplay:
    """A live call-local owner, issued only by ``replay_selected_pins``."""

    _pins: MappingProxyType
    _thread: threading.Thread
    _process: int

    def get(self, path):
        if self not in _ACTIVE or threading.current_thread() is not self._thread or os.getpid() != self._process:
            raise ValueError("selected pin replay requires its live same-thread same-process call scope")
        key = str(Path(path))
        if key not in self._pins:
            return None
        _canonical(path)
        return dict(self._pins[key])


@contextlib.contextmanager
def replay_selected_pins(paths, *, max_pins):
    """Share explicit pins only until mandatory complete post-call rereads.

    The cardinality limit precedes file expansion and hashing. No filename,
    mtime, previous scope, nested scope or ambient process supplies a selection.
    Callers still own source membership and file-byte limits. Exceptional exits
    also reread every selected file and revoke the owner.
    """
    if type(max_pins) is not int or max_pins < 1:
        raise ValueError("selected replay needs an explicit positive pin-count limit")
    selected = tuple(itertools.islice(paths, max_pins + 1))
    if not selected or len(selected) > max_pins:
        raise ValueError("selected replay exceeds its complete pin-count limit")
    canonical = tuple(_canonical(path) for path in selected)
    if len(set(canonical)) != len(canonical):
        raise ValueError("selected replay pins must have distinct canonical identities")
    pins = {str(path): _fresh_pin(path) for path in canonical}
    owner = SelectedPinReplay(
        MappingProxyType({path: MappingProxyType(pin) for path, pin in pins.items()}),
        threading.current_thread(),
        os.getpid(),
    )
    _ACTIVE.add(owner)
    try:
        yield owner
    finally:
        _ACTIVE.discard(owner)
        failures = []
        for path in canonical:
            try:
                actual = _fresh_pin(path)
            except (OSError, ValueError) as error:
                failures.append(error)
            else:
                if actual != pins[str(path)]:
                    failures.append(ValueError("selected file bytes differ"))
        if failures:
            raise ValueError("selected replay file content changed during its call scope") from failures[0]


def replayed_pin(owner, path):
    if type(owner) is not SelectedPinReplay:
        raise ValueError("pin replay cannot accept saved or substitute owners")
    return owner.get(path)


# ---- proc --------------------------------------------------------------------------------------
# Run a subprocess and turn a failure into the caller's own error: the helper nine modules re-wrote.
#
# Measured 2026-09-14: baselines/{buddy,executorch,exo}, runtime/{kernel_spike,elf_audit},
# runtime/backends/{spike_model,zephyr_model}, llvmlower/codegen and runtime/remote_linux each carried a private
# ``_run`` doing the same thing -- stringify the argv, capture text output, raise the module's own error on
# a non-zero exit with a tail of the output -- with nine different tail lengths and three different answers
# to "what happens on a timeout". The error class, the tail and the timeout behaviour stay each caller's
# choice; the mechanics live here once.
#
# ``wrap_timeout=False`` lets ``subprocess.TimeoutExpired`` propagate unchanged, for callers that relied on
# catching it; ``True`` turns it into the caller's error with the limit that fired.

_SHOWN_CMD_CHARS = 1000


def run_checked(
    cmd: Sequence[Any],
    *,
    error: type[Exception] = RuntimeError,
    timeout: float | None = None,
    timeout_hint: str = "",
    wrap_timeout: bool = True,
    tail: int = 2000,
    **kw: Any,
) -> subprocess.CompletedProcess:
    """Run ``cmd`` (each element ``str()``-ed), capturing text output; raise ``error`` on a non-zero exit.

    The message names the exit status, the command (capped at 1000 characters) and the last ``tail``
    characters of stdout and stderr. ``kw`` passes through to :func:`subprocess.run` (``cwd``, ``env``,
    ``input``...).
    """
    argv = [str(c) for c in cmd]
    shown = " ".join(argv)[:_SHOWN_CMD_CHARS]
    try:
        with timed_event("process", "subprocess.run"):
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, **kw)
    except subprocess.TimeoutExpired as expired:
        if not wrap_timeout:
            raise
        failure = error(f"command timed out after {timeout}s{timeout_hint}: {shown}")
        failure.captured_timeout = expired
        raise failure from None
    if proc.returncode != 0:
        failure = error(
            f"command failed (rc {proc.returncode}): {shown}\n"
            f"STDOUT:{(proc.stdout or '')[-tail:]}\nSTDERR:{(proc.stderr or '')[-tail:]}"
        )
        # Keep the concise caller-facing message, while owned build observers
        # can retain the complete captured diagnostics, including the first
        # compiler assertion that a long stack trace would otherwise hide.
        failure.captured_process = proc
        raise failure
    return proc


# ---- bwrap_args --------------------------------------------------------------------------------
# Exact large bwrap argv transport over a private sealed read-only descriptor.
#
# No argument file enters a candidate workspace. bwrap consumes and closes its
# ``--args`` policy descriptor before executing the separate original payload.
# The caller supplies that exact boundary; command operands are never interpreted
# as policy. Observations retain the
# logical argv and transport byte identity; a saved FD number grants no authority.

_INLINE_BYTES = 64 * 1024
_MAX_BYTES = 32 * 1024 * 1024
_MAX_ARGUMENTS = 262144
# Linux UAPI linux/fcntl.h. Compatible portable CPython builds can omit these
# constants and os.memfd_create; the fallback uses the same enforced kernel API.
_SEALS, _ADD_SEALS, _GET_SEALS = 0x000F, 1033, 1034


def _memfd():
    if sys.platform != "linux":
        raise ValueError("private bwrap argument transport requires Linux sealed descriptors")
    if hasattr(os, "memfd_create"):
        return os.memfd_create("merlin-private-bwrap-args", 0x0001 | 0x0002)
    import ctypes

    function = ctypes.CDLL(None, use_errno=True).memfd_create
    function.argtypes, function.restype = (ctypes.c_char_p, ctypes.c_uint), ctypes.c_int
    descriptor = function(b"merlin-private-bwrap-args", 0x0001 | 0x0002)
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), "private bwrap sealed descriptor creation failed")
    return descriptor


@dataclass(frozen=True)
class ArgumentTransport:
    logical_argv: tuple[str, ...]
    command: tuple[str, ...]
    descriptor: int | None
    payload_sha256: str
    payload_bytes: int
    policy_end: int

    @property
    def pass_fds(self):
        return () if self.descriptor is None else (self.descriptor,)

    def record(self):
        if self.descriptor is not None:
            if (fcntl.fcntl(self.descriptor, fcntl.F_GETFL) & os.O_ACCMODE) != os.O_RDONLY or (
                fcntl.fcntl(self.descriptor, _GET_SEALS) & _SEALS != _SEALS
            ):
                raise ValueError("private bwrap arguments lost sealed read-only ownership")
            if os.fstat(self.descriptor).st_size != self.payload_bytes:
                raise ValueError("private bwrap argument byte inventory changed")
        return {
            "schema": "merlin.bwrap_argument_transport.v1",
            "transport": "sealed_readonly_fd" if self.descriptor is not None else "inline",
            "payload_sha256": self.payload_sha256,
            "payload_bytes": self.payload_bytes,
            "argument_count": len(self.logical_argv) - 1,
            "encoded_argument_count": self.policy_end - 1,
            "scope": "exact logical argv transport; no source, semantic or execution qualification",
        }


@contextlib.contextmanager
def argument_transport(argv, *, policy_end=None):
    """Keep exact token order and bytes, privately bounded on the original host."""
    logical = tuple(argv)
    if (
        not logical
        or len(logical) > _MAX_ARGUMENTS
        or any(type(token) is not str or "\0" in token for token in logical)
    ):
        raise ValueError("bwrap argument transport requires a bounded original string argv without NUL")
    if not logical[0]:
        raise ValueError("bwrap argument transport has no original executable")
    if policy_end is not None and (type(policy_end) is not int or not 1 <= policy_end < len(logical)):
        raise ValueError("bwrap argument transport requires the original policy/payload boundary")
    end = len(logical) if policy_end is None else policy_end
    encoded, size = [], 0
    for token in logical[1:end]:
        raw = os.fsencode(token)
        size += len(raw) + 1
        if size > _MAX_BYTES:
            raise ValueError("private bwrap arguments exceed the original fixed byte budget")
        encoded.append(raw)
    payload = b"\0".join(encoded) + (b"\0" if encoded else b"")
    identity = hashlib.sha256(payload).hexdigest()
    if size <= _INLINE_BYTES:
        yield ArgumentTransport(logical, logical, None, identity, size, end)
        return
    if policy_end is None:
        raise ValueError("large bwrap argument transport requires the explicit original policy/payload boundary")
    if Path(logical[0]).name != "bwrap":
        raise ValueError("large argument descriptor transport requires the original bwrap executable")
    writable = readonly = None
    try:
        writable = _memfd()
        with os.fdopen(os.dup(writable), "wb") as stream:
            stream.write(payload)
        fcntl.fcntl(writable, _ADD_SEALS, _SEALS)
        readonly = os.open(f"/proc/self/fd/{writable}", os.O_RDONLY | os.O_CLOEXEC)
        os.close(writable)
        writable = None
        selected = ArgumentTransport(
            logical, (logical[0], "--args", str(readonly), *logical[end:]), readonly, identity, size, end
        )
        selected.record()
        yield selected
        selected.record()
    finally:
        for descriptor in (readonly, writable):
            if descriptor is not None:
                os.close(descriptor)


def run_bwrap(
    argv, *, policy_end=None, directory=None, stage="bwrap", inputs=(), outputs=(), dependencies=(), **options
):
    """Launch the same command and observe logical argv plus original transport."""
    if options.get("pass_fds") or options.get("close_fds") is False:
        raise ValueError("private bwrap transport cannot inherit unrelated original descriptors")
    options["env"] = dict(os.environ if options.get("env") is None else options["env"])
    with argument_transport(argv, policy_end=policy_end) as selected:
        options["pass_fds"] = selected.pass_fds
        observation = (
            contextlib.nullcontext(None)
            if directory is None
            else observe(
                directory,
                stage=stage,
                argv=selected.logical_argv,
                cwd=options.get("cwd"),
                env=options["env"],
                inputs=inputs,
                outputs=outputs,
                dependencies=(*dependencies, Path(__file__)),
            )
        )
        with observation as record:
            if record is not None:
                record.document["argument_transport"] = selected.record()
            actual = subprocess.run(list(selected.command), **options)
            result = subprocess.CompletedProcess(
                list(selected.logical_argv), actual.returncode, actual.stdout, actual.stderr
            )
            if record is not None:
                record.complete(result)
            return result
