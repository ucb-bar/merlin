"""Bound the disk a run's grade and self-check scratch can hold.

Every background grade builds each capsule in ``_qa_work/fruns_<token>/`` and every self-check in its
own raw runs root, and nothing removed either afterwards. Measured on one 12-hour run: 1-8 GB per
grade, 18 GB of retained self-checks and 24 GB of scalability staging after 3.3 hours, until the
volume filled and the grader died mid-write. Most of those bytes are build products -- a copied
compiler tree per capsule, linked ELFs and objects, C harnesses with the inputs as literals -- that
no reader opens once the grade is published.

What the readers that outlive a grade actually open, and so what is kept:

* ``capsule_result.json``, byte-identical: the correctness curriculum reuses a pass only when its
  pinned result still verifies, scheduling history prices capsules from it, and the operator audits,
  convergence report and calibration-index readers count from it;
* every small file (receipts, reports, timing records, logs up to :data:`KEEP_FILE_BYTES`), and the
  issued scalability qualification beside them;
* every capsule directory holding paid RTL evidence (a tier derived from RTL that passed or failed),
  untouched: the timing archive is ready only when every measured gSIM pass keeps its ELF, harness,
  command buffer, trace and numeric report.

L3 promotion freezes its own source snapshot before it enqueues, and calibration intake copies its
result into the run's registry while the grade runs, so neither reads a grade's scratch later. The
newest :data:`KEEP_NEWEST` grades and the one the agent's verdict was published from stay whole.

Whatever the size rule says, a path a seal, release, run record or derivation cites is kept
(:mod:`merlin.common.cited_evidence`): the run's own records, plus the trees ``MERLIN_CITED_ROOTS``
and the storage contract add -- study ledgers and papers that quote a grade's files by path. When
that registry cannot be read nothing is removed, because a full volume is recoverable and a deleted
citation is not.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

SCHEMA = "merlin.scratch_retention.v1"
MARKER = "retention.json"
MIN_FREE_ENV = "MERLIN_GRADE_MIN_FREE_BYTES"
#: Below this much free space no new background grade is admitted.
DEFAULT_MIN_FREE_BYTES = 25 << 30
#: Grades kept whole, newest first, beside the published one.
KEEP_NEWEST = 3
#: Files up to this size are kept: receipts, reports, logs.
KEEP_FILE_BYTES = 1 << 20
#: Always kept, whatever their size; readers pin these bytes.
KEPT_NAMES = frozenset({"capsule_result.json", "issued.json", "qualification.json"})
#: Directories that hold only build products: the per-capsule compiler copy and retained programs.
BULK_DIRECTORIES = frozenset({"compiler-source", "executed"})
BULK_SUFFIXES = frozenset({".elf", ".o", ".a", ".so", ".bin"})
_RTL_EXECUTED = frozenset({"pass", "fail"})
_RESULT_BYTES_READ = 64 << 20
_TIERS_HEAD_BYTES = 4 << 20
#: A long-lived grader re-reads the citing records at most this often.
CITED_MAX_AGE_S = 600.0


def say(message: str) -> None:
    """Report one line; a full log volume drops it instead of raising into the caller."""
    try:
        print(message, flush=True)
    except (OSError, ValueError):
        pass


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
        from merlin.common import cited_evidence

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
                    pass
    try:
        path.rmdir()
    except OSError:
        shutil.rmtree(path, ignore_errors=True)
    return freed


def slim_tree(root, *, citing: tuple[Path, ...] = ()) -> int:
    """Remove build products under ``root`` and keep its evidence (see the module); returns bytes freed.

    Directory times are restored, so readers that date a check by its directories see the same times.
    Links are removed as names, never followed. ``citing`` adds trees whose records may cite ``root``.
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
            try:
                size = entry.lstat().st_size
            except OSError:
                continue
            if entry.is_symlink() or Path(name).suffix in BULK_SUFFIXES or size > KEEP_FILE_BYTES:
                freed += _remove(entry)
    for directory, (atime, mtime) in times.items():
        try:
            os.utime(directory, ns=(atime, mtime))
        except OSError:
            pass
    return freed


def slim_once(root, *, scope: str, citing: tuple[Path, ...] = ()) -> int:
    """:func:`slim_tree` a finished scratch root and mark it, so later passes skip it."""
    root = Path(root)
    marker = root / MARKER
    if root.is_symlink() or not root.is_dir() or marker.exists():
        return 0
    if not _Kept(root, citing).readable:
        return 0  # leave it unmarked so a later pass with a readable registry slims it
    started = time.time()
    freed = slim_tree(root, citing=citing)
    try:
        stat = root.stat()
        marker.write_text(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "scope": scope,
                    "slimmed_at_epoch": started,
                    "bytes_freed": freed,
                    "kept": f"capsule_result.json, files up to {KEEP_FILE_BYTES} bytes, issued qualifications, and "
                    "every capsule directory with RTL-tier evidence",
                },
                indent=2,
            )
            + "\n"
        )
        os.utime(root, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    except OSError:
        pass
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
    and promotion freezes its own source copy before it enqueues.
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
                    slim_once(path, scope="completed self-check")
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
    draining, or with no owner record, is left whole. Returns bytes freed.
    """
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
            continue
        if isinstance(document, dict) and str(document.get("token")) == entry.stem:
            records[entry.stem] = document
    newest = sorted(records, key=int, reverse=True)
    protected = set(newest[: max(0, keep_newest)])
    published = _published_token(workspace)
    if published is not None:
        protected.add(published)
    freed = 0
    for token in newest:
        record = records[token]
        if token in protected or record.get("state") not in ("complete", "deferred"):
            continue
        runs = work / f"fruns_{token}"
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


if __name__ == "__main__":  # operator use: python -m ... <run_dir> [workspace]
    _freed = prune_superseded_grades(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    say(f"freed {_freed >> 20} MiB")
