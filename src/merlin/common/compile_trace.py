"""Stop a compile at a named stage, dump its IR, and keep a trace of the whole descent.

Every lowering route already names its stages where it records them. :class:`merlin.common.ir_audit.IrAudit`
keeps exact named-stage IR (``input``, ``contract``, ..., ``llvm-final``);
:func:`merlin.xdsl_dialects.ir_inspection.record_stage` serializes an xDSL module at a stage; and the
native pass manager prints every pass once :func:`merlin.llvmlower.ir_inspection.bind_inspection` binds
its printer (``passes/segment-NNNN``, one directory per pass-manager invocation, with its
``pipeline.txt``). None of it was reachable from a command line, and nothing could stop a compile part
way. This module is the switch: a :class:`Request` names the stages to dump (after, or before), the one
to stop after, and the trace directory, and each of those recording points reports to :func:`observe`
(an IR text), :func:`artifact` (a file a stage wrote: an object, a linked program) or
:func:`harvest_native` (the native printer's dumps).

THE REQUEST TRAVELS IN THE ENVIRONMENT (:data:`ENV`), as ``MERLIN_PASSES`` does, so a stage reached in
a child process -- a package entrypoint, a lowering worker -- is dumped too, into ``ir/p<pid>/``. Only
the process that opened the :func:`session` STOPS. A child that reaches the stop stage records that it
did, and the owner stops at its own next stage boundary (:func:`stop_if_reached`): a child that exited
early would otherwise be read by its caller as a failed group, and the build would carry on.

A STOP IS NOT AN ERROR AND NOT A RESULT. :class:`StopAfterStage` derives from ``BaseException`` (as
``SystemExit`` does) because stages are reached inside code that turns every ``Exception`` into a
refusal -- a per-group lowering records a failed group and continues -- and a stop read that way would
let the build finish a program that is missing a piece and looks complete. The front doors catch it,
say where the IR is, and exit 0 without writing a final artifact.

Which stages exist is declared by each pipeline beside the code that records them (:func:`declare`),
never listed here; a test runs each route and holds the declaration to what it records. The native
passes are not declared at all: they are read off the pass pipeline the lowering builds.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import shutil
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

#: The process-wide request, as JSON. Read where each stage is reached, never frozen at import.
ENV = "MERLIN_COMPILE_TRACE"
SCHEMA = "merlin_compile_trace_v1"
#: Selects every stage (``--dump-ir-after all``).
ALL = "all"
#: A native pass stage is ``mlir:<pass argument>``: the name the pass is registered under.
MLIR_PREFIX = "mlir:"
#: The xDSL pass-invocation log a session turns on when the caller has not
#: (:data:`merlin.xdsl_dialects.lowering.passes.PASS_LOG_ENV`, restated so reading a request imports no xDSL).
PASS_LOG_ENV = "MERLIN_PASS_LOG"

INDEX = "trace.json"
EVENTS = "events.jsonl"
PIPELINE = "pipeline.txt"
PASS_LOG = "pass_log.jsonl"

#: Outcomes a trace index records.
COMPLETED = "completed"
STOPPED = "stopped"
FAILED = "failed"
#: The compile ran to its end without reaching the stage it was asked to stop after.
NOT_REACHED = "stop_stage_not_reached"


class TraceError(ValueError):
    """A request naming no known stage, or a trace directory that cannot hold a new trace."""


class StopAfterStage(BaseException):  # noqa: N818 -- a control-flow signal, like SystemExit, not an error
    """The requested stage was reached and its IR written; nothing after it ran.

    ``files`` are the dumps of that stage (more than one for a native pass that ran once per function).
    """

    def __init__(self, stage: str, files: Sequence[str | Path], directory: str | Path, *, note: str = "") -> None:
        self.stage = stage
        self.files = tuple(str(f) for f in files)
        self.directory = str(directory)
        self.note = note
        super().__init__(f"stopped after stage {stage!r}")

    def message(self, tool: str) -> str:
        lines = [f"[{tool}] stopped after stage {self.stage!r}, as requested; nothing after it was kept."]
        lines += [f"    IR: {f}" for f in self.files] or ["    (the stage wrote no file of its own)"]
        lines.append(f"    trace: {Path(self.directory) / INDEX}")
        if self.note:
            lines.append(f"    note: {self.note}")
        lines.append("    no final artifact was produced; this is not a build result.")
        return "\n".join(lines)


# ------------------------------------------------------------------------------------------- stages

_DECLARED: dict[str, dict[str, Any]] = {}


def declare(pipeline: str, stages: Sequence[str], *, entry: str, summary: str) -> tuple[str, ...]:
    """Record the stages ``pipeline`` reaches, in order, as the code that reaches them names them.

    Called by each pipeline module at import, beside the calls that record those stages; returns the
    stages so the module can use the same tuple. Re-declaring a pipeline replaces it (a reload)."""
    names = tuple(str(s) for s in stages)
    if not names or any(not n or n.startswith(MLIR_PREFIX) or "#" in n for n in names):
        raise TraceError(f"pipeline {pipeline!r} declares an unusable stage name in {names}")
    _DECLARED[str(pipeline)] = {"pipeline": str(pipeline), "entry": entry, "summary": summary, "stages": names}
    return names


def _label(pipeline: str) -> str:
    """A declared pipeline's short name for its entry point (``merlin.llvmlower.lower_model`` -> ``llvm``),
    so a stage an audit reports under its producer is indexed under the pipeline that declared it."""
    for row in _DECLARED.values():
        if row["entry"] == pipeline:
            return row["pipeline"]
    return pipeline


def declared() -> list[dict[str, Any]]:
    """Every declared pipeline, in declaration order (import order of the pipeline modules)."""
    return [dict(v) for v in _DECLARED.values()]


def parse_selector(text: str) -> tuple[str, int | None]:
    """``stage`` or ``stage#N`` (the Nth time the stage is reached, 1-based) -> ``(stage, N or None)``."""
    name, sep, ordinal = str(text).strip().partition("#")
    name = name.strip()
    if not name:
        raise TraceError(f"empty stage name in {text!r}")
    if not sep:
        return name, None
    if not ordinal.isdigit() or int(ordinal) < 1:
        raise TraceError(f"{text!r}: the occurrence after '#' must be a positive integer")
    return name, int(ordinal)


def validate(selectors: Iterable[str], known: Iterable[str], *, allow_all: bool) -> tuple[str, ...]:
    """The selectors, each refused unless it names a known stage, a native pass, or (when allowed) ``all``."""
    known = set(known)
    out = []
    for raw in selectors:
        for item in str(raw).split(","):
            item = item.strip()
            if not item:
                continue
            if item == ALL:
                if not allow_all:
                    raise TraceError("'all' selects every stage; name one stage to stop after")
                out.append(ALL)
                continue
            name, _ = parse_selector(item)
            if not name.startswith(MLIR_PREFIX) and name not in known:
                raise TraceError(f"unknown stage {name!r}; --list-stages prints every stage this compiler names")
            if name == MLIR_PREFIX:
                raise TraceError("a native pass stage names its pass: mlir:<pass>")
            out.append(item)
    return tuple(out)


# ------------------------------------------------------------------------------------------ request


@dataclasses.dataclass(frozen=True)
class Request:
    """What to dump and where to stop. ``owner``/``started`` are set by :func:`session`."""

    directory: str
    dump_after: tuple[str, ...] = ()
    dump_before: tuple[str, ...] = ()
    stop_after: str | None = None
    owner: int = 0
    started: float = 0.0

    def _selects(self, selectors: Sequence[str], name: str, occurrence: int) -> bool:
        for selector in selectors:
            if selector == ALL:
                return True
            base, ordinal = parse_selector(selector)
            if base == name and ordinal in (None, occurrence):
                return True
        return False

    def dumps_after(self, name: str, occurrence: int) -> bool:
        return self._selects(self.dump_after, name, occurrence)

    def dumps_before(self, name: str, occurrence: int) -> bool:
        return self._selects(self.dump_before, name, occurrence)

    def stops_at(self, name: str, occurrence: int) -> bool:
        if self.stop_after is None:
            return False
        base, ordinal = parse_selector(self.stop_after)
        return base == name and (ordinal or 1) == occurrence

    @property
    def path(self) -> Path:
        return Path(self.directory)

    def owns(self) -> bool:
        return os.getpid() == self.owner

    def native_printing(self) -> tuple[bool, bool] | None:
        """``(before, after)`` for the native pass printer, or None when no native pass is selected."""

        def native(selectors: Sequence[str]) -> bool:
            return any(s == ALL or s.startswith(MLIR_PREFIX) for s in selectors)

        before = native(self.dump_before)
        after = native(self.dump_after) or str(self.stop_after or "").startswith(MLIR_PREFIX)
        return (before, after) if before or after else None

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> Request:
        doc = json.loads(raw)
        if not isinstance(doc, dict) or not doc.get("directory"):
            raise TraceError(f"{ENV} holds no trace request")
        return cls(
            directory=str(doc["directory"]),
            dump_after=tuple(doc.get("dump_after") or ()),
            dump_before=tuple(doc.get("dump_before") or ()),
            stop_after=doc.get("stop_after"),
            owner=int(doc.get("owner") or 0),
            started=float(doc.get("started") or 0.0),
        )


def active() -> Request | None:
    """The request :data:`ENV` holds, or None when no trace is open (the common case: one dict read)."""
    raw = os.environ.get(ENV)
    if not raw:
        return None
    return Request.from_json(raw)


# ------------------------------------------------------------------------------------ per process


class _State:
    """One process's view of one trace: its sequence number, occurrence counts and last stage per thread."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.seq = 0
        self.native_calls = 0
        self.occurrences: dict[str, int] = {}
        self.last_time: dict[int, float] = {}
        self.last: dict[int, tuple[str, str, str]] = {}


_STATES: dict[tuple[str, float, int], _State] = {}
_STATES_LOCK = threading.Lock()


def _state(request: Request) -> _State:
    # Per SESSION and process: a directory reused by a later session (or a forked child) starts afresh.
    key = (request.directory, request.started, os.getpid())
    with _STATES_LOCK:
        return _STATES.setdefault(key, _State())


def _ir_root(request: Request) -> Path:
    root = request.path / "ir"
    return root if request.owns() else root / f"p{os.getpid()}"


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def _write(path: Path, text: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = text.encode("utf-8")
    path.write_bytes(payload)
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _append(request: Request, event: dict[str, Any]) -> None:
    event.setdefault("pid", os.getpid())
    event.setdefault("t", round(time.time(), 6))
    line = (json.dumps(event, sort_keys=True, default=str) + "\n").encode("utf-8")
    request.path.mkdir(parents=True, exist_ok=True)
    # One write per line on an O_APPEND descriptor: lines from concurrent processes never interleave.
    fd = os.open(request.path / EVENTS, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def _relative(request: Request, path: Path) -> str:
    try:
        return str(path.relative_to(request.path))
    except ValueError:
        return str(path)


def _tick(request: Request, state: _State, key: str) -> tuple[int, int, float]:
    """(sequence, occurrence of ``key``, seconds since this thread's previous stage)."""
    now = time.time()
    thread = threading.get_ident()
    with state.lock:
        state.seq += 1
        occurrence = state.occurrences[key] = state.occurrences.get(key, 0) + 1
        previous = state.last_time.get(thread, request.started or now)
        state.last_time[thread] = now
        return state.seq, occurrence, round(max(0.0, now - previous), 6)


def observe(
    stage: str,
    *,
    pipeline: str,
    content: str | None = None,
    render: Callable[[], str] | None = None,
    fmt: str = "mlir",
) -> None:
    """``stage`` was reached and ``content`` (or ``render()``) is its IR. A no-op with no open trace.

    ``render`` is called only when the IR is wanted (dumped now, or kept for a later before-dump), so an
    unselected stage costs no serialization. Raises :class:`StopAfterStage` in the owner process when
    this is the stage to stop after, AFTER its IR is written and the event appended."""
    request = active()
    if request is None:
        return
    state = _state(request)
    seq, occurrence, seconds = _tick(request, state, stage)
    stop = request.stops_at(stage, occurrence)
    after = stop or request.dumps_after(stage, occurrence)
    before = request.dumps_before(stage, occurrence)
    text = (
        content
        if content is not None
        else (render() if render is not None and (after or request.dump_before) else None)
    )
    ext = "ll" if fmt == "llvm-ir" else "mlir"
    event: dict[str, Any] = {
        "kind": "ir",
        "stage": stage,
        "pipeline": _label(pipeline),
        "occurrence": occurrence,
        "seq": seq,
        "seconds": seconds,
        "format": fmt,
    }
    files: list[str] = []
    thread = threading.get_ident()
    if before:
        previous = state.last.get(thread)
        if previous is None:
            event["before"] = "no earlier stage was reached in this thread"
        else:
            path = _ir_root(request) / f"{seq:03d}-before-{_safe(stage)}.{'ll' if previous[2] == 'llvm-ir' else 'mlir'}"
            _write(path, previous[1])
            event["before_file"] = _relative(request, path)
            event["before_is"] = previous[0]
    if after and text is not None:
        path = _ir_root(request) / f"{seq:03d}-{_safe(stage)}.{ext}"
        event.update(_write(path, text))
        event["file"] = _relative(request, path)
        files.append(str(path))
    if request.dump_before and text is not None:
        state.last[thread] = (stage, text, fmt)
    if stop:
        event["stop"] = True
    _append(request, event)
    if stop and request.owns():
        raise StopAfterStage(stage, files, request.directory)


def artifact(stage: str, paths: Iterable[str | Path], *, pipeline: str, seconds: float | None = None) -> None:
    """``stage`` wrote ``paths`` (an object, a linked program, a builder stage's products). A no-op with no
    open trace. Selected files are hard-linked (copied across filesystems) into ``products/``."""
    request = active()
    if request is None:
        return
    state = _state(request)
    seq, occurrence, elapsed = _tick(request, state, stage)
    stop = request.stops_at(stage, occurrence)
    keep = stop or request.dumps_after(stage, occurrence)
    sources = [Path(p) for p in paths if Path(p).is_file()]
    event: dict[str, Any] = {
        "kind": "files",
        "stage": stage,
        "pipeline": _label(pipeline),
        "occurrence": occurrence,
        "seq": seq,
        "seconds": round(seconds, 6) if seconds is not None else elapsed,
        "sources": [str(p) for p in sources],
    }
    kept: list[str] = []
    if keep:
        root = (
            request.path
            / "products"
            / (f"{seq:03d}-{_safe(stage)}" if request.owns() else f"p{os.getpid()}-{seq:03d}-{_safe(stage)}")
        )
        for source in sources:
            target = root / _product_name(source, sources)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                try:
                    os.link(source, target)  # the same bytes, no copy: a trace of a big build costs no disk
                except OSError:
                    shutil.copy2(source, target)  # another filesystem
            kept.append(str(target))
        event["files"] = [_relative(request, Path(k)) for k in kept]
    if stop:
        event["stop"] = True
    _append(request, event)
    if stop and request.owns():
        raise StopAfterStage(stage, kept, request.directory)


def snapshot(root: str | Path) -> dict[str, int] | None:
    """``{path: mtime_ns}`` of every file under ``root`` while a trace is open (None when none is), so a
    stage's products can be told from what was there before it (:func:`written_since`)."""
    if active() is None:
        return None
    stamps: dict[str, int] = {}
    root = Path(root)
    for path in root.rglob("*") if root.is_dir() else ():
        with contextlib.suppress(FileNotFoundError):  # a file another worker removed while this walked
            if path.is_file():
                stamps[str(path)] = path.stat().st_mtime_ns
    return stamps


def written_since(root: str | Path, before: dict[str, int] | None) -> list[Path]:
    """The files under ``root`` created or rewritten since ``before`` (a :func:`snapshot`), outside the
    trace directory itself."""
    request = active()
    if before is None or request is None or not Path(root).is_dir():
        return []
    after = snapshot(root) or {}
    trace = request.path.resolve()
    return sorted(
        Path(p) for p, stamp in after.items() if before.get(p) != stamp and not Path(p).resolve().is_relative_to(trace)
    )


def _product_name(source: Path, sources: Sequence[Path]) -> str:
    """The file's path below the deepest directory every source shares, so two ``command_buffer.json`` of
    two groups keep their group directory and never overwrite each other."""
    parents = [p.parent for p in sources]
    common = Path(os.path.commonpath([str(p) for p in parents])) if parents else source.parent
    return str(source.relative_to(common))


# --------------------------------------------------------------------------- native pass printing


def native_directory() -> tuple[Path, bool, bool] | None:
    """A fresh directory for one native pass-manager run's printer, with ``(before, after)``, or None
    when no open trace selects a native pass."""
    request = active()
    if request is None:
        return None
    printing = request.native_printing()
    if printing is None:
        return None
    state = _state(request)
    with state.lock:
        state.native_calls += 1
        call = state.native_calls
    name = f"call-{call:03d}" if request.owns() else f"p{os.getpid()}-call-{call:03d}"
    directory = request.path / "mlir" / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory, printing[0], printing[1]


def _dump_header(path: Path) -> tuple[str, str] | None:
    """``("before"|"after", pass argument)`` from a native dump's first line, read structurally:
    ``// -----// IR Dump After CanonicalizerPass: canonicalize{...} //----- //``."""
    with path.open(encoding="utf-8", errors="replace") as fh:
        first = fh.readline()
    _, found, rest = first.partition("IR Dump ")
    if not found:
        return None
    when, _, rest = rest.partition(" ")
    if when not in ("Before", "After"):
        return None
    _, sep, argument = rest.partition(": ")
    if not sep:
        return None
    name = argument.split(" //", 1)[0].split("{", 1)[0].strip()
    return (when.lower(), name) if name else None


def _counter(path: Path) -> tuple[int, ...]:
    """The printer's counter prefix of a dump (``17_canonicalize`` -> ``(17,)``; nested ``5_1_x`` ->
    ``(5, 1)``). It orders the dumps OF ONE PASS (the second canonicalize's counter is the larger), and
    one nested pass dumped once per function repeats it; it does not order different passes."""
    key = []
    for part in path.stem.split("_"):
        if not part.isdigit():
            break
        key.append(int(part))
    return tuple(key)


def native_dumps(directory: str | Path) -> list[dict[str, Any]]:
    """Every pass the native printer dumped under ``directory``, in the order the pipeline ran them:
    ``{segment, position, when, pass, files}``, one row per pass execution and side.

    The order is the segment's own ``pipeline.txt`` (the pass pipeline that pass manager ran): the Nth
    dump of a pass, by its counter, is that pass's Nth position in the pipeline. A dump of a pass the
    pipeline does not name (a dynamic pipeline) keeps its counter order after the named ones."""
    rows = []
    for segment in sorted((Path(directory) / "passes").glob("segment-*")):
        text = segment / "pipeline.txt"
        order = split_pipeline(text.read_text(encoding="utf-8")) if text.is_file() else []
        found: dict[tuple[str, str], dict[tuple[int, ...], list[Path]]] = {}
        for path in segment.rglob("*.mlir"):
            header = _dump_header(path)
            if header is not None:
                found.setdefault(header, {}).setdefault(_counter(path), []).append(path)
        for (when, name), by_counter in found.items():
            positions = [i for i, step in enumerate(order) if step == name]
            for k, counter in enumerate(sorted(by_counter)):
                position = positions[k] if k < len(positions) else len(order) + k
                files = sorted(str(p) for p in by_counter[counter])
                rows.append({"segment": segment.name, "position": position, "when": when, "pass": name, "files": files})
    return sorted(rows, key=lambda r: (r["segment"], r["position"], r["when"] != "before"))


def harvest_native(directory: str | Path, *, pipeline: str, prune: bool = True) -> None:
    """Index the native printer's dumps under ``directory`` into the open trace, as ``mlir:<pass>`` stages.

    Dumps no selector asks for are deleted when ``prune`` (the directory is the trace's own); a stop
    selected on a native pass raises :class:`StopAfterStage` in the owner once its dumps are indexed --
    the printer ran in a child process, so this is the first point the owner can stop at."""
    request = active()
    if request is None:
        return
    state = _state(request)
    stop_files: list[str] = []
    stop_stage = None
    for row in native_dumps(directory):
        stage = MLIR_PREFIX + row["pass"]
        key = stage if row["when"] == "after" else f"before:{stage}"
        seq, occurrence, _ = _tick(request, state, key)
        stop = row["when"] == "after" and request.stops_at(stage, occurrence)
        wanted = (
            stop or request.dumps_after(stage, occurrence)
            if row["when"] == "after"
            else request.dumps_before(stage, occurrence)
        )
        files = [Path(f) for f in row["files"]]
        if not wanted and prune:
            for path in files:
                path.unlink(missing_ok=True)
            continue
        event = {
            "kind": "native",
            "stage": stage,
            "pipeline": _label(pipeline),
            "occurrence": occurrence,
            "seq": seq,
            "when": row["when"],
            "segment": row["segment"],
            "files": [_relative(request, p) for p in files],
        }
        if stop:
            event["stop"] = True
            stop_stage, stop_files = f"{stage}#{occurrence}", [str(p) for p in files]
        _append(request, event)
    if prune:
        for scope in sorted((Path(directory) / "passes").rglob("*"), reverse=True):
            if scope.is_dir() and not any(scope.iterdir()):
                scope.rmdir()
    if stop_stage is not None and request.owns():
        raise StopAfterStage(stop_stage, stop_files, request.directory)


# ------------------------------------------------------------------------------ owner stop points


def stop_if_reached() -> None:
    """Raise :class:`StopAfterStage` in the owner when a CHILD process already reached the stop stage.

    Called at the owner's own stage boundaries (a whole-model build stage): the child recorded its
    dump and carried on, and the owner stops at the first point it controls."""
    request = active()
    if request is None or request.stop_after is None or not request.owns():
        return
    for event in _events(request):
        if event.get("stop") and int(event.get("pid") or 0) != request.owner:
            files = [str(request.path / f) for f in (event.get("files") or [event.get("file")]) if f]
            raise StopAfterStage(
                str(event.get("stage")),
                files,
                request.directory,
                note=f"reached in process {event.get('pid')}; the build stopped at its next stage boundary",
            )


def _events(request: Request) -> list[dict[str, Any]]:
    path = request.path / EVENTS
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------------------- session


@contextlib.contextmanager
def session(
    request: Request,
    *,
    command: Sequence[str] = (),
    on_open: Callable[[Request], dict[str, Any] | None] | None = None,
) -> Iterator[Request]:
    """Open a trace: make ``request`` the process's (and its children's), and write the index at exit.

    The directory must be new or empty -- two compiles' events in one index would read as one. The
    xDSL pass-invocation log goes to ``pass_log.jsonl`` unless the caller already set one; ``on_open``
    may return extra facts for the index (what it installed). The environment is restored on exit,
    even on a stop or an error, and the index records which of the three it was."""
    root = Path(request.directory).absolute()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise TraceError(f"trace directory {root} is not empty; name a new one")
    root.mkdir(parents=True, exist_ok=True)
    request = dataclasses.replace(request, directory=str(root), owner=os.getpid(), started=time.time())
    saved = {key: os.environ.get(key) for key in (ENV, PASS_LOG_ENV)}
    os.environ[ENV] = request.to_json()
    if not saved[PASS_LOG_ENV]:
        os.environ[PASS_LOG_ENV] = str(root / PASS_LOG)
    extra: dict[str, Any] = {"pass_log": os.environ[PASS_LOG_ENV]}
    outcome, stop = FAILED, None
    try:
        if on_open is not None:
            extra.update(on_open(request) or {})
        yield request
        outcome = COMPLETED if request.stop_after is None else NOT_REACHED
    except StopAfterStage as exc:
        outcome, stop = STOPPED, exc
        raise
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        write_index(request, outcome=outcome, command=command, stop=stop, extra=extra)


def write_index(
    request: Request,
    *,
    outcome: str,
    command: Sequence[str] = (),
    stop: StopAfterStage | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """``trace.json`` (every stage reached, in order, with its files and seconds) and ``pipeline.txt``."""
    events = sorted(_events(request), key=lambda e: (float(e.get("t") or 0.0), int(e.get("seq") or 0)))
    timing: dict[str, float] = {}
    for event in events:
        if event.get("seconds") is not None:
            label = f"{event.get('pipeline')}/{event.get('stage')}"
            timing[label] = round(timing.get(label, 0.0) + float(event["seconds"]), 6)
    segments = []
    for text in sorted((request.path / "mlir").glob("*/passes/segment-*/pipeline.txt")):
        segments.append({"directory": _relative(request, text.parent), "pipeline": text.read_text(encoding="utf-8")})
    index = {
        "schema": SCHEMA,
        "outcome": outcome,
        "command": list(command),
        "request": {
            "dump_after": list(request.dump_after),
            "dump_before": list(request.dump_before),
            "stop_after": request.stop_after,
        },
        "stop": (
            {"stage": stop.stage, "files": [_relative(request, Path(f)) for f in stop.files], "note": stop.note}
            if stop is not None
            else None
        ),
        "wall_seconds": round(time.time() - request.started, 3) if request.started else None,
        "stages": events,
        "timing_seconds": timing,
        "native_segments": segments,
        **(extra or {}),
    }
    (request.path / INDEX).write_text(json.dumps(index, indent=1, default=str) + "\n", encoding="utf-8")
    lines = ["# stages in the order this compile reached them: pid  pipeline  stage  seconds  file"]
    for event in events:
        seconds = event.get("seconds")
        lines.append(
            f"{event.get('pid')}  {event.get('pipeline')}  {event.get('stage')}"
            + (f"#{event['occurrence']}" if int(event.get("occurrence") or 1) > 1 else "")
            + (f"  ({event.get('when')})" if event.get("kind") == "native" else "")
            + (f"  {seconds:.3f}s" if isinstance(seconds, (int, float)) else "")
            + (f"  {event['file']}" if event.get("file") else "")
        )
    for segment in segments:
        lines += ["", f"# native pass-manager segment {segment['directory']}", segment["pipeline"].strip()]
    (request.path / PIPELINE).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return request.path / INDEX


# ------------------------------------------------------------------------------ pass pipelines


def split_pipeline(text: str) -> list[str]:
    """The passes of a textual pass pipeline, in order, nesting flattened and options dropped:
    ``canonicalize,func.func(a,b{x=1}),cse`` -> ``[canonicalize, a, b, cse]``. Read with an explicit
    bracket-depth tokenizer (an option value holds commas and braces), never a pattern."""
    passes: list[str] = []

    def walk(segment: str) -> None:
        depth, start = 0, 0
        for i, ch in enumerate(segment + ","):
            if ch in "({":
                depth += 1
            elif ch in ")}":
                depth -= 1
            elif ch == "," and depth == 0:
                item = segment[start:i].strip()
                start = i + 1
                if not item:
                    continue
                head, paren, inner = item.partition("(")
                if paren and "{" not in head:
                    walk(inner[: inner.rfind(")")])
                else:
                    passes.append(item.split("{", 1)[0].strip())

    walk(text.strip())
    return passes


def with_ordinals(names: Sequence[str]) -> list[str]:
    """``[a, b, a]`` -> ``[a, b, a#2]``: the selector that names each occurrence."""
    seen: dict[str, int] = {}
    out = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name}#{seen[name]}")
    return out
