"""Small result indexes with immutable, content-addressed public case details.

This module is standard-library-only so the same reader can be staged for author
tools. Detail files contain the already public row, never raw grader results.
Missing details leave the compact evidence intact and cannot create a pass.
"""

from __future__ import annotations

import hashlib
import copy
import json
import math
import os
import tempfile
import threading
from pathlib import Path

SCHEMA = "merlin.phase1.result_index.v1"
MAX_DETAIL_BYTES = 64 << 20
_CACHE = {}
_LOCK = threading.RLock()


class _FullRow(dict):
    """An opened row: its later public projections must survive compaction."""


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _cache_key(path, digest):
    stat = path.stat()
    return (str(path.resolve()), digest, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _root(workspace=None, document_path=None):
    if workspace is not None:
        return Path(workspace).resolve()
    if document_path is not None:
        path = Path(document_path).resolve()
        for parent in path.parents:
            if parent.name in ("qa", ".qa_channel"):
                return parent.parent
    return Path.cwd().resolve()


def load_row(row, *, workspace=None, document_path=None, reader=None, read_only=False):
    """Open one compact row's public detail; legacy inline rows pass through.

    ``reader`` optionally supplies bounded/cached ``(JSON, raw payload digest)``
    access. Paths must remain in this workspace's public case store.
    Compact attribution overlays the stored row, so carried results keep their
    current grade label without another copy of their large diagnostic payload.
    ``read_only`` shares immutable nested payloads for trusted index publication;
    consumers that edit nested records must keep the default independent copy.
    """
    if isinstance(row, _FullRow) or not isinstance(row, dict) or not isinstance(row.get("detail"), dict):
        return row
    pointer = row["detail"]
    digest, relative = pointer.get("sha256"), pointer.get("path")
    if not isinstance(digest, str) or len(digest) != 64 or not isinstance(relative, str):
        return row
    root = _root(workspace, document_path)
    path = root / relative
    try:
        if path.is_symlink() or not path.resolve().is_relative_to(root / "qa" / "cases"):
            return row
        if path.name != f"{digest}.json" or path.stat().st_size > MAX_DETAIL_BYTES:
            return row
        if reader is not None:
            loaded = reader(path, limit=MAX_DETAIL_BYTES)
            if not isinstance(loaded, tuple) or len(loaded) != 2 or loaded[1] != digest:
                return row
            full = loaded[0]
        else:
            key = _cache_key(path, digest)
            with _LOCK:
                full = _CACHE.get(key)
            if full is None:
                payload = path.read_bytes()
                if hashlib.sha256(payload).hexdigest() != digest:
                    return row
                full = json.loads(payload)
                with _LOCK:
                    if len(_CACHE) >= 1024:
                        _CACHE.clear()
                    _CACHE[key] = full
        if not isinstance(full, dict) or full.get("capsule") != row.get("capsule"):
            return row
        # Only attribution is mutable in an index; summaries are intentionally
        # partial and must not overwrite the complete nested records.
        return _FullRow({**(full if read_only else copy.deepcopy(full)),
                         **({"graded": row["graded"]} if "graded" in row else {}),
                         "detail": pointer})
    except (OSError, ValueError, TypeError):
        return row


def hydrate(document, *, workspace=None, document_path=None, reader=None, read_only=False):
    """Return a full-row view of an index or an unchanged legacy document."""
    if not isinstance(document, dict):
        return document
    return {**document, **{
        key: [load_row(row, workspace=workspace, document_path=document_path, reader=reader, read_only=read_only)
              for row in rows]
        for key in ("per_capsule", "cases")
        if isinstance((rows := document.get(key)), list)
    }}


def _summary(row):
    keys = ("capsule", "name", "status", "pass", "screened_only", "certified_complete", "tiers",
            "failure_plane", "failure_category", "barrier_tier", "barrier_status", "barrier_cycles",
            "numerical_execution_passed", "execution_digest", "graded", "trace_status", "trace_requirement_status")
    out = {key: row[key] for key in keys if key in row}
    if isinstance(out.get("tiers"), dict):
        out["tiers"] = {key: value.get("status") if isinstance(value, dict) else value
                        for key, value in out["tiers"].items()}
    if row.get("failure_detail"):
        out["failure_detail"] = str(row["failure_detail"])[:96]
    if isinstance(row.get("failure"), dict):
        out["failure"] = {key: str(value)[:160] if key == "detail" else value
                          for key, value in row["failure"].items()
                          if key in ("plane", "category", "tier", "detail")}
    performance = row.get("performance")
    if isinstance(performance, dict):
        out["performance"] = {key: performance[key] for key in
                              ("status", "estimated_whole_soc_cycles", "array_utilization_pct", "bound_by")
                              if key in performance}
        roofline = performance.get("structural_roofline") or {}
        for key in ("array_utilization_pct", "bound_by"):
            if key not in out["performance"] and key in roofline:
                out["performance"][key] = roofline[key]
        positive = lambda value: type(value) in (int, float) and math.isfinite(value) and value > 0
        cycles = performance.get("estimated_whole_soc_cycles")
        useful, peak = roofline.get("useful_macs"), roofline.get("structural_peak_macs_per_cycle")
        if (performance.get("status") in ("PROVISIONAL", "QUALIFIED")
                and performance.get("complete_event_accounting") is True
                and all(positive(value) for value in (cycles, useful, peak))):
            out["performance"]["array_utilization_pct"] = round(100 * useful / (cycles * peak), 2)
        terms = {"compute": roofline.get("structural_compute_floor_cycles")}
        for side, service in (roofline.get("logical_request_ideal_service") or {}).items():
            if isinstance(service, dict):
                terms[f"{side} requests"] = service.get("ideal_requested_service_cycles")
        terms = {name: value for name, value in terms.items() if positive(value)}
        if terms:
            out["performance"].setdefault("bound_by", max(terms, key=terms.get))
    step = row.get("optimization_next")
    if isinstance(step, dict):
        for key in ("next_step", "use", "action", "reading"):
            if isinstance(step.get(key), str):
                out["next_step"] = " ".join(step[key].split())[:120]
                break
    return out


def compact_row(row, *, workspace):
    """Store a full public row once and return its small index entry."""
    if not isinstance(row, dict):
        return row
    full = row if isinstance(row, _FullRow) else load_row(row, workspace=workspace)
    if full is row and "detail" in row and not isinstance(row, _FullRow):
        return row  # inaccessible detail must never be replaced by its summary
    full = {key: value for key, value in full.items() if key not in ("detail", "graded")}
    pointer = row.get("detail")
    if isinstance(pointer, dict):
        path = Path(workspace) / str(pointer.get("path", ""))
        with _LOCK:
            cached = _CACHE.get(_cache_key(path, pointer.get("sha256")))
        if cached is not None and full == cached:
            return {**_summary({**full, **({"graded": row["graded"]} if "graded" in row else {})}),
                    "detail": pointer}
    payload = _encoded(full)
    digest = hashlib.sha256(payload).hexdigest()
    name = str(full.get("capsule") or full.get("name") or "unnamed")
    # Preserve ordinary case names; encode unsafe names without allowing paths.
    component = name if name not in (".", "..") and all(c.isalnum() or c in "_- ." for c in name) else hashlib.sha256(name.encode()).hexdigest()
    relative = Path("qa") / "cases" / component / f"{digest}.json"
    path = Path(workspace) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".detail-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        try:
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
        finally:
            temporary.unlink(missing_ok=True)
    with _LOCK:
        key = _cache_key(path, digest)
        if key not in _CACHE and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("public case detail digest differs")
        if len(_CACHE) >= 1024:
            _CACHE.clear()
        _CACHE[key] = copy.deepcopy(full)
    summary = _summary({**full, **({"graded": row["graded"]} if "graded" in row else {})})
    summary["detail"] = {"path": relative.as_posix(), "sha256": digest}
    return summary


def compact_document(document, *, workspace, namespace="grade"):
    """Keep top-level advice/counts and replace full case payloads with indexes."""
    if not isinstance(document, dict):
        return document
    out = {**document, "result_index": {"schema": SCHEMA, "namespace": namespace}}
    advice = {}
    for entry in (document.get("performance_headline") or {}).get("top_headroom") or ():
        if isinstance(entry, dict) and isinstance(entry.get("next_step"), dict):
            step = entry["next_step"]
            advice[entry.get("test_case")] = step.get("choice") or step.get("kind")
    for key in ("per_capsule", "cases"):
        if isinstance(document.get(key), list):
            out[key] = [compact_row(row, workspace=workspace) for row in document[key]]
            for row in out[key]:
                if isinstance(row, dict) and advice.get(row.get("capsule")) and "next_step" not in row:
                    row["next_step"] = str(advice[row["capsule"]])[:100]
    return out


def append_rows(workspace, previous, document):
    """Append only changed compact rows; the caller owns the publication lock."""
    before = {row.get("capsule"): row for row in (previous or {}).get("per_capsule") or ()
              if isinstance(row, dict)}
    changed = [row for row in document.get("per_capsule") or ()
               if isinstance(row, dict) and row != before.get(row.get("capsule"))]
    if changed:
        with (Path(workspace) / "qa" / "completed_rows.jsonl").open("ab") as stream:
            for row in changed:
                stream.write(_encoded(row) + b"\n")
