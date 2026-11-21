"""Write one corpus member in a staging directory and move it into the corpus only when it succeeds.

A writer that fails part-way used to leave its half-written member in the corpus: a ``capsule.yaml``
and a ``golden.yaml`` that MANIFEST.yaml does not list, which the next ``rglob`` picks up as though
it had been built. Writing under a private staging root and renaming each finished member into place
makes a failed (or killed) write leave nothing at the member's corpus path.

The stage is a view of the corpus. Every member already written is linked into it, so a writer
that consults its siblings -- a qualified model reuses a per-group capsule another model already
wrote -- sees and reuses the real one exactly as before, and only directories it newly creates are
promoted. The member being written is not linked: a rerun into an existing corpus writes it afresh
and replaces the previous copy only once the new one is complete.
"""

from __future__ import annotations

import contextvars
import os
import shutil
import tempfile
import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import groupby
from pathlib import Path

from merlin.common import record_formats as strict_json

# ---- staging -----------------------------------------------------------------------------------

STAGING_PREFIX = ".staging-"
_PUBLICATION_LOCK = threading.Lock()


def _corpus_members(root: Path) -> list[Path]:
    out = []
    for category in sorted(root.iterdir()):
        if not category.is_dir() or category.is_symlink() or category.name.startswith(STAGING_PREFIX):
            continue
        out += [m for m in sorted(category.iterdir()) if m.is_dir() and not m.is_symlink()]
    return out


def _link_existing(stage: Path, out_root: Path, own: str | None) -> None:
    for member in _corpus_members(out_root):
        relative = member.relative_to(out_root)
        if relative.as_posix() == own:
            continue
        link = stage / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(member.resolve(), target_is_directory=True)


def promote(stage: Path, out_root: Path) -> dict[Path, Path]:
    """Move every member the write created (not the linked ones) into ``out_root``."""
    moved: dict[Path, Path] = {}
    retired = stage / ".replaced"
    for member in _corpus_members(stage):
        destination = out_root / member.relative_to(stage)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            # A rerun's fresh copy of this member: the old one is set aside, then dropped with the stage.
            aside = retired / member.relative_to(stage)
            aside.parent.mkdir(parents=True, exist_ok=True)
            destination.rename(aside)
        member.rename(destination)
        moved[member] = destination
    return moved


def write_staged(
    write: Callable[[Path], object], out_root: Path | str, *, member: str | None = None, operator_exclusion=None
):
    """Run ``write(stage_root)`` and promote what it created; a failure leaves the corpus untouched.

    ``member`` is the ``<category>/<name>`` being written, which is not linked into the stage.
    Returns what ``write`` returned, with a staged member path mapped to its final corpus path.
    """
    out_root = Path(out_root)
    from merlin.development.testgen.public_sources.operator_exclusion import (
        check_publication,
        require_selection,
    )

    require_selection(operator_exclusion, out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=out_root))
    try:
        with _PUBLICATION_LOCK:
            _link_existing(stage, out_root, member)
        written = write(stage)
        for created in _corpus_members(stage):
            check_publication(operator_exclusion, created)
        with _PUBLICATION_LOCK:
            moved = promote(stage, out_root)
        if isinstance(written, (str, Path)) and written and Path(written).is_relative_to(stage):
            return moved.get(Path(written), out_root / Path(written).relative_to(stage))
        return written
    finally:
        shutil.rmtree(stage, ignore_errors=True)


class PreparedWrite:
    """One unpublished source write, finished once by the ordered generation caller."""

    def __init__(self, stage, out_root, member, operator_exclusion, written, finish):
        self.stage, self.out_root, self.member = stage, out_root, member
        self.operator_exclusion, self.written, self._finish = operator_exclusion, written, finish
        self._context, self._pid, self._closed = contextvars.copy_context(), os.getpid(), False

    def close(self):
        """Discard only this private attempt; completed or failed publications are untouched."""
        if os.getpid() != self._pid:
            raise ValueError("prepared corpus write belongs to another process")
        if not self._closed:
            self._closed = True
            shutil.rmtree(self.stage, ignore_errors=True)

    def finish(self):
        """Refresh siblings, recheck, qualify and publish after all earlier model writes."""
        from merlin.development.testgen.public_sources.operator_exclusion import (
            check_publication,
            require_selection,
        )

        if os.getpid() != self._pid or self._closed:
            raise ValueError("prepared corpus write is closed or belongs to another process")
        try:
            require_selection(self.operator_exclusion, self.out_root)
            with _PUBLICATION_LOCK:
                _link_existing(self.stage, self.out_root, self.member)
            written = self._context.run(self._finish, self.written, self.stage)
            for created in _corpus_members(self.stage):
                check_publication(self.operator_exclusion, created)
            with _PUBLICATION_LOCK:
                moved = promote(self.stage, self.out_root)
            if isinstance(written, (str, Path)) and written and Path(written).is_relative_to(self.stage):
                return moved.get(Path(written), self.out_root / Path(written).relative_to(self.stage))
            return written
        finally:
            self.close()


def prepare_staged(write, finish, out_root, *, member, operator_exclusion=None) -> PreparedWrite:
    """Capture into a private stage without selecting a stale shared-sibling view.

    The source callback may write only its own member. The unchanged finishing
    callback runs later, under this exact observed context, on the ordered caller.
    Nothing becomes a corpus member until finishing and publication both succeed.
    """
    from merlin.development.testgen.public_sources.operator_exclusion import require_selection

    out_root = Path(out_root)
    require_selection(operator_exclusion, out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=out_root))
    try:
        written = write(stage)
        if any(created.relative_to(stage).as_posix() != member for created in _corpus_members(stage)):
            raise ValueError("a prepared source write created a sibling member")
        return PreparedWrite(stage, out_root, member, operator_exclusion, written, finish)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


# ---- parallel_capture --------------------------------------------------------------------------
# Bounded independent capture batches; dependent model writes retain declaration order.


def _available_memory(path: Path = Path("/proc/meminfo")) -> int:
    """Prefer reclaimable host memory; retain conservative free-page fallback."""
    try:
        rows = [
            line.partition(":")[2].split()
            for line in path.read_text().splitlines()
            if line.partition(":")[0] == "MemAvailable"
        ]
        if len(rows) == 1 and len(rows[0]) == 2 and rows[0][0].isdecimal() and rows[0][1] == "kB":
            return int(rows[0][0]) * 1024
    except OSError:
        pass
    try:
        pages, size = os.sysconf("SC_AVPHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
        return pages * size if pages >= 0 and size > 0 else 0
    except (AttributeError, OSError, ValueError):
        return 0


def _unescape_mount(value: str) -> str:
    for escaped, plain in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
        value = value.replace(escaped, plain)
    return value


def _cgroup_limits(proc: Path = Path("/proc")) -> tuple[int | None, int | None]:
    """Read visible unified ancestors, without configuring any resource owner.

    Missing Linux controls retain the affinity/free-memory fallback. Unsupported
    or incomplete visible controls conservatively select only one worker.
    """
    try:
        membership = (proc / "self/cgroup").read_text().splitlines()
    except FileNotFoundError:
        return None, None
    except OSError:
        return 1, 0
    try:
        selected = [line[3:] for line in membership if line.startswith("0::")]
        if not selected:
            return None, None
        if len(selected) != 1:
            raise ValueError("ambiguous cgroup membership")
        member = Path(selected[0])
        if not member.is_absolute() or ".." in member.parts:
            raise ValueError("unsupported cgroup membership")
        mounts = []
        for line in (proc / "self/mountinfo").read_text().splitlines():
            before, separator, after = line.partition(" - ")
            if separator and after.split()[0] == "cgroup2":
                fields = before.split()
                mounts.append((Path(_unescape_mount(fields[3])), Path(_unescape_mount(fields[4]))))
        if len(mounts) != 1:
            raise ValueError("ambiguous cgroup mount")
        root, mount = mounts[0]
        if not root.is_absolute() or not mount.is_absolute() or ".." in root.parts or ".." in mount.parts:
            raise ValueError("unsupported cgroup mount")
        current = mount / member.relative_to(root)
        cpus, memories = [], []
        for directory in (current, *current.parents):
            if not directory.is_relative_to(mount):
                break
            try:
                value = (directory / "memory.max").read_text().strip()
            except FileNotFoundError:
                value = "max"  # A controller may be absent, including at the mount root.
            if value != "max":
                used = (directory / "memory.current").read_text().strip()
                if not value.isdecimal() or not used.isdecimal():
                    raise ValueError("unsupported memory ceiling")
                memories.append(max(0, int(value) - int(used)))
            try:
                bandwidth = (directory / "cpu.max").read_text().split()
            except FileNotFoundError:
                bandwidth = ["max", "1"]
            if len(bandwidth) != 2 or not bandwidth[1].isdecimal() or int(bandwidth[1]) <= 0:
                raise ValueError("unsupported CPU ceiling")
            if bandwidth[0] != "max":
                if not bandwidth[0].isdecimal() or int(bandwidth[0]) <= 0:
                    raise ValueError("unsupported CPU quota")
                cpus.append(max(1, int(bandwidth[0]) // int(bandwidth[1])))
        return min(cpus, default=None), min(memories, default=None)
    except (OSError, ValueError, IndexError):
        return 1, 0


def worker_limit() -> int:
    """At most eight captures, bounded by affinity, host memory and cgroups."""
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    memory = _available_memory()
    cpu_ceiling, memory_ceiling = _cgroup_limits()
    if cpu_ceiling is not None:
        cpus = min(cpus, cpu_ceiling)
    if memory_ceiling is not None:
        memory = min(memory, memory_ceiling)
    return max(1, min(8, cpus, memory // (2 * 1024**3)))


def independent(entry: dict) -> bool:
    return bool(
        (entry.get("source") == "pytorch" or entry.get("pytorch_ref"))
        and not entry.get("materialized_capture")
        and not entry.get("micro_model")
        and entry.get("kind") != "model"
        and entry.get("op") != "model"
    )


def _fresh_model(entry):
    return bool(
        (entry.get("kind") == "model" or entry.get("op") == "model")
        and not entry.get("materialized_capture")
        and not entry.get("micro_model")
    )


def _model_jobs(pool, entries, prepare, workers):
    """Bound unpublished captures and drain/discard them on every consumer exit."""

    pending, active = deque(), []
    source = iter(entries)

    def submit():
        try:
            entry = next(source)
        except StopIteration:
            return False
        job = pool.submit(contextvars.copy_context().run, prepare, entry)
        pending.append((entry, job))
        active.append(job)
        return True

    def discard(job):
        try:
            _, written = job.result()
        except BaseException:
            return  # The ordinary caller owns failures; a failed preparation cleans its own stage.
        if isinstance(written, PreparedWrite):
            written.close()

    try:
        for _ in range(workers):
            if not submit():
                break
        while pending:
            entry, job = pending.popleft()
            yield entry, job
            discard(job)
            active.remove(job)
            submit()
    finally:
        for job in active:
            discard(job)  # Wait for owned captures before dropping their unpublished stages.


def ordered_jobs(entries, write, *, workers: int, prepare_model=None):
    """Yield futures in declared order, finishing each independent batch before dependent writes.

    Exceptions remain attached to their original entry for the ordinary failure ledger.
    Every thread receives its own copy of the caller's observed-facts/import context.
    """
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="phase0-capture") as pool:

        def batch_kind(entry):
            if prepare_model is not None and _fresh_model(entry):
                return "model"
            return "independent" if independent(entry) else "serial"

        for kind, batch in groupby(entries, key=batch_kind):
            if kind == "model" and workers > 1:
                yield from _model_jobs(pool, batch, prepare_model, workers)
            elif kind == "independent" and workers > 1:
                jobs = [(entry, pool.submit(contextvars.copy_context().run, write, entry)) for entry in batch]
                yield from jobs
            else:
                for entry in batch:
                    job = Future()
                    try:
                        job.set_result(write(entry))
                    except Exception as exc:  # noqa: BLE001 -- the ordinary ledger owns failures
                        job.set_exception(exc)
                    yield entry, job


# ---- selected_records --------------------------------------------------------------------------
# Stable ordinary JSON inputs; their semantic admission belongs to each consumer.


def read_mapping(path: Path, *, subject: str) -> dict:
    """Read an explicit, bounded file without following aliases or concurrent changes."""
    path = Path(path).absolute()
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents) or not path.is_file():
        raise ValueError(f"{subject} must be an ordinary file")
    before = path.stat()
    if before.st_size > 1024 * 1024:
        raise ValueError(f"{subject} exceeds the one-MiB record budget")
    raw = path.read_bytes()
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise ValueError(f"{subject} changed during reading")
    data = strict_json.loads(raw, max_bytes=1024 * 1024)
    if not isinstance(data, dict):
        raise ValueError(f"{subject} must be a mapping")
    return data
