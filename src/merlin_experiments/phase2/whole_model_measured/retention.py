"""What a finished whole-model job keeps, and the regenerable build intermediates it drops.

A whole-model build writes hundreds of megabytes of per-group lowered modules and weight blobs that
the linked ELF already contains; kept per candidate they fill the disk in a few dozen measurements.
Only regenerable intermediates go -- the package snapshot, the ELFs, the program sources, every record
and log and the runs' consoles stay, so a verdict can be re-read and the build repeated -- and every
removal is recorded (path, bytes, sha256), because a deletion nobody wrote down is how a result
loses its evidence.  Deciding WHETHER a job may be slimmed is :func:`may_finalize`'s: a reference,
an unfinished job, a best (or a result within noise of it) and a digest the certifier still holds
open all keep their trees.
"""

from __future__ import annotations

import contextlib
import fnmatch
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from merlin.perf import whole_model_verdict as V

from . import jobs as J
from .identity import key_of, now, read_json, sha256_file, write_json_atomic

RETENTION_SCHEMA = "merlin_whole_model_retention_v1"

#: What a FINISHED job keeps (paths relative to its job directory; ``**`` spans directories, every
#: other segment is a shell-style pattern of one path segment).  A launch may declare its own list.
DEFAULT_RETAIN: tuple[str, ...] = (
    "job.json",
    "result.json",
    "build_record.json",
    "board_request.json",
    "local_verdict.json",
    "pre_measure_check*",
    "structure_screen*.json",
    "worker.log",
    "package/**",
    "selfcheck_out/**",
    "run*/**",
    "cell/**/*.json",
    "build*/program/*.elf",
    "build*/program/*.c",
    "build*/program/build.log",
    "build*/*.json",
    "build*/manifest.yaml",
    "build*/harness/**",
)

#: A best within this relative distance of the store's lowest keeps its build trees.
BEST_NOISE = 0.001


def _retained(parts: list[str], pattern: list[str]) -> bool:
    if not pattern:
        return not parts
    if pattern[0] == "**":
        return any(_retained(parts[i:], pattern[1:]) for i in range(len(parts) + 1))
    return bool(parts) and fnmatch.fnmatchcase(parts[0], pattern[0]) and _retained(parts[1:], pattern[1:])


def finalize_retention(job_dir: Path, retain: Any = None) -> dict[str, Any]:
    """Remove every file under ``build*/`` of a FINISHED job that ``retain`` does not name, recording
    each one in ``build_record.json["retention"]``."""
    job_dir = Path(job_dir)
    patterns = [str(p).split("/") for p in (retain or DEFAULT_RETAIN)]
    removed: list[dict[str, Any]] = []
    for build in sorted(job_dir.glob("build*")):
        if not build.is_dir() or build.is_symlink():
            continue
        for path in sorted(build.rglob("*")):
            if path.is_dir() and not path.is_symlink():
                continue
            parts = path.relative_to(job_dir).parts
            if any(_retained(list(parts), pattern) for pattern in patterns):
                continue
            entry: dict[str, Any] = {"path": "/".join(parts)}
            if path.is_symlink():
                entry.update(bytes=0, symlink_to=os.readlink(path))
            else:
                entry.update(bytes=path.stat().st_size, sha256=sha256_file(path))
            path.unlink()
            removed.append(entry)
        for directory in sorted((d for d in build.rglob("*") if d.is_dir()), key=lambda d: -len(d.parts)):
            with contextlib.suppress(OSError):
                directory.rmdir()  # only empty ones
    receipt = {
        "schema": RETENTION_SCHEMA,
        "at": now(),
        "retain": ["/".join(p) for p in patterns],
        "files_removed": len(removed),
        "bytes_removed": sum(int(r["bytes"]) for r in removed),
        "removed": removed,
    }
    record_path = job_dir / "build_record.json"
    record = read_json(record_path) or {}
    record["retention"] = receipt
    write_json_atomic(record_path, record)
    return receipt


def prune_archived_attempt(attempt_dir: Path, retain: Any = None) -> list[dict[str, Any]]:
    """:func:`finalize_retention` every job-shaped directory an archived attempt holds: itself, and
    any attempt nested inside it (a job lost twice archives its first archive inside its second)."""
    attempt_dir = Path(attempt_dir)
    roots = {attempt_dir}
    for path in attempt_dir.rglob("*"):
        if path.is_dir() and not path.is_symlink() and path.name.startswith(J.ARCHIVED_ATTEMPT_PREFIXES):
            roots.add(path)
    return [finalize_retention(root, retain) for root in sorted(roots)]


def prune_intermediates(root: Path, prunable: Any, *, keep: Path) -> dict[str, Any]:
    """Remove the build intermediates the BUILDER declared disposable, inside this job's build dir only:
    only paths the builder names, only under ``root``, never the ELF, and what was removed is recorded."""
    removed, freed = [], 0
    root = Path(root).resolve()
    for relative in prunable if isinstance(prunable, list) else []:
        for path in sorted(root.glob(str(relative))):
            resolved = path.resolve()
            if root not in resolved.parents or resolved == keep.resolve() or keep.resolve().is_relative_to(resolved):
                continue
            if path.is_dir() and not path.is_symlink():
                size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                shutil.rmtree(path)
            elif path.is_file():
                size = path.stat().st_size
                path.unlink()
            else:
                continue
            removed.append(str(path.relative_to(root)))
            freed += size
    return {"paths": len(removed), "bytes": freed, "patterns": list(prunable) if isinstance(prunable, list) else []}


def may_finalize(root: Path, job: Mapping[str, Any]) -> str | None:
    """Why a job must NOT be finalized now, or None."""
    if job.get("role") == J.ROLE_REFERENCE:
        return "the reference is every batch's control"
    if job.get("state") not in (J.DONE, J.FAILED, J.SCREEN_FAILED):
        return f"the job is {job.get('state')}"
    digest = str(job.get("package_sha256") or "")
    certifier = job.get("certifier_root")
    if certifier:
        open_job = read_json(Path(str(certifier)) / digest / "job.json")
        if open_job is not None and open_job.get("state") not in J.TERMINAL:
            return "the certifier holds this digest open"
    document = read_json(Path(root) / key_of(job) / "result.json") or {}
    cycles = V.objective_cycles(document.get("verdict"))
    if cycles is not None and document.get("timing_status") == V.TIMING_MEASURED:
        best = [
            V.objective_cycles((read_json(p.parent / "result.json") or {}).get("verdict"))
            for p in Path(root).glob("*/job.json")
        ]
        best = [c for c in best if c is not None]
        if best and cycles <= min(best) * (1 + BEST_NOISE):
            return "this result is the store's best (or within noise of it)"
    return None


def finalize_if_allowed(root: Path, job_dir: Path) -> dict[str, Any] | None:
    job = read_json(Path(job_dir) / "job.json") or {}
    why_not = may_finalize(root, job)
    if why_not is not None:
        return {"skipped": why_not}
    try:
        return finalize_retention(job_dir, job.get("retain"))
    except OSError as exc:  # recorded; a job that cannot be slimmed is still a finished job
        return {"error": f"{type(exc).__name__}: {exc}"}


def finalize_superseded(root: Path) -> list[str]:
    """Finalize every finished job whose trees were kept only because it was the best when it finished,
    and which a later result has since beaten: the reason it kept them is gone."""
    swept = []
    for job_path in sorted(Path(root).glob("*/job.json")):
        job_dir = job_path.parent
        record = read_json(job_dir / "build_record.json") or {}
        if "retention" in record or not any(d.is_dir() for d in job_dir.glob("build*")):
            continue
        receipt = finalize_if_allowed(Path(root), job_dir)
        if receipt is not None and "skipped" not in receipt and "error" not in receipt:
            swept.append(job_dir.name)
    return swept


def finalize_batch_if_allowed(batch_dir: Path, finalized: list[Mapping[str, Any] | None]) -> dict[str, Any]:
    """Remove the linked batch ELF and its per-variant objects once EVERY job the batch measured has
    been finalized; each removed file is recorded in ``batch.json["retention"]``."""
    if not finalized or any(not isinstance(f, Mapping) or "files_removed" not in f for f in finalized):
        return {"skipped": "a job of this batch was not finalized"}
    link = Path(batch_dir) / "link"
    removed: list[dict[str, Any]] = []
    for path in sorted([*link.glob("*.o"), *link.glob("*.elf")]) if link.is_dir() else []:
        if path.is_symlink() or not path.is_file():
            continue
        removed.append(
            {"path": str(path.relative_to(batch_dir)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
        path.unlink()
    receipt = {
        "schema": RETENTION_SCHEMA,
        "at": now(),
        "files_removed": len(removed),
        "bytes_removed": sum(int(r["bytes"]) for r in removed),
        "removed": removed,
    }
    record_path = Path(batch_dir) / "batch.json"
    record = read_json(record_path) or {}
    record["retention"] = receipt
    write_json_atomic(record_path, record)
    return receipt


__all__ = [
    "DEFAULT_RETAIN",
    "finalize_batch_if_allowed",
    "finalize_if_allowed",
    "finalize_retention",
    "finalize_superseded",
    "may_finalize",
    "prune_archived_attempt",
    "prune_intermediates",
]
