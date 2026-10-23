"""The capsule catalog: what each capsule is (op, shapes, origin, cohort) and its latest recorded result.

Definitions come from the run's staged corpus (``private_corpus_input/{public,scalability/cases}``):
``capsule.yaml`` plus, for scalability cases, ``source-shape-binding.json`` (the model, program and
source operation a shape was cut from). Cohorts are the run's own rosters
(:func:`..terminal_capsules.cohort_rosters`). Results are the newest row per capsule among
``qa/progress.json``, ``qa/verdict.json`` and ``_qa_work/fruns_*/snapshot_progress.json``, with
per-tier cycles and timings from the newest ``capsule_result.json`` and the certification ledger
``qa/tier_state.json``. Capsule names are private: public mode shows counts only.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from ...terminal_capsules import cohort_rosters
from ..io import clean, epoch, is_dir, is_file, num, obj, rows, scan_dir

TIERS = ("L0", "L1", "L2", "L3")
#: Newest ``fruns_*`` snapshot directories scanned for results (each holds one graded snapshot).
MAX_SNAPSHOTS = 24
MAX_RESULTS = 400
MAX_INVOCATIONS = 40


def _dims(shape: Any) -> str:
    if isinstance(shape, list) and all(isinstance(d, int) for d in shape):
        return "x".join(str(d) for d in shape) or "scalar"
    return "?"


def _compact(attributes: dict) -> str:
    """Small scalar and integer-list attributes as ``key=value``; tensor names and bulk omitted."""
    parts = []
    for key, value in attributes.items():
        if isinstance(value, bool) or key in ("ifm", "weight", "out", "outs", "arg_order", "semantic"):
            continue
        if isinstance(value, int):
            parts.append(f"{key}={value}")
        elif isinstance(value, list) and value and len(value) <= 4 and all(isinstance(v, int) for v in value):
            parts.append(f"{key}={'x'.join(map(str, value))}")
        elif isinstance(value, str) and len(value) <= 12 and key not in ("layout",):
            parts.append(f"{key}={value}")
    return clean(" ".join(parts), 80)


def _binding(doc: dict | None) -> dict[str, Any]:
    if not doc:
        return {}
    occurrences = doc.get("occurrences")
    shape = doc.get("shape_mkn")
    return {
        "model": clean(doc.get("model"), 40) or None,
        "program": clean(doc.get("program"), 40) or None,
        "aten_op": clean(doc.get("operation"), 60) or None,
        "occurrences": len(occurrences) if isinstance(occurrences, list) else None,
        "shape_mkn": "x".join(map(str, shape)) if isinstance(shape, list) else None,
        "node": clean(doc.get("node"), 60) or None,
        "full_model": doc.get("full_model") if isinstance(doc.get("full_model"), bool) else None,
    }


def definition(path: Path, doc: dict, binding: dict | None) -> dict[str, Any]:
    """One catalog row from a capsule's own definition."""
    operation = obj(doc.get("operation"))
    attributes = obj(operation.get("attributes"))
    inputs = rows(doc.get("inputs"), 32)
    shapes = [f"{clean(i.get('name'), 16)}:{_dims(i.get('shape'))} {clean(i.get('dtype'), 8)}".strip() for i in inputs]
    semantic = obj(doc.get("semantic"))
    origin = {
        "model": None,
        "program": None,
        "aten_op": None,
        "occurrences": None,
        "reference": clean(doc.get("source_reference"), 200) or None,
    }
    origin.update({k: v for k, v in _binding(binding).items() if v is not None})
    dtype = attributes.get("output_dtype") or obj(doc.get("numeric_policy")).get("dtype")
    requirements = doc.get("pass_requirements")
    tiers = doc.get("required_oracle_tiers")
    return {
        "name": clean(doc.get("name") or path.parent.name, 120),
        "cohort": None,
        "kind": clean(doc.get("kind"), 24) or None,
        "op": clean(operation.get("op"), 32) or None,
        "attrs": _compact(attributes),
        "shapes": shapes,
        "dtype": clean(dtype, 12) or None,
        "source_role": clean(doc.get("source_role"), 32) or None,
        "origin": origin,
        "family": clean(semantic.get("semantic_family"), 40) or None,
        "axis": clean(semantic.get("generalization_axis"), 40) or None,
        "pass_requirements": [clean(r, 48) for r in requirements if isinstance(r, str)]
        if isinstance(requirements, list)
        else [],
        "required_tiers": [t for t in tiers if isinstance(t, str)] if isinstance(tiers, list) else [],
        "tiers": {},
        "status": None,
        "numeric": None,
        "cycles": {"L2": None, "L3": None},
        "failure": {"plane": None, "category": None, "tier": None},
        "timing": {"build_s": None, "sim_active_s": None, "oracle_wait_s": None},
        "path": str(path.parent),
        "result_at": None,
        "result_source": None,
    }


def _definitions(ctx, run: Path) -> dict[str, dict[str, Any]]:
    """Parse the staged corpus once per change of its directories."""
    corpus = run / "private_corpus_input"
    roots = [corpus / "public", corpus / "scalability" / "cases"]
    stamp = tuple(os.stat(r).st_mtime_ns if is_dir(r) else None for r in roots)
    cached = ctx.state.get("definitions")
    if cached and cached[0] == stamp:
        return cached[1]
    found: dict[str, dict[str, Any]] = {}
    for root in roots:
        for entry in scan_dir(root, 2048):
            path = Path(entry.path) / "capsule.yaml"
            if not is_file(path):
                continue
            doc = ctx.reader.yaml(path, limit=1 << 20)
            if not isinstance(doc, dict):
                continue
            binding = ctx.reader.document(Path(entry.path) / "source-shape-binding.json", limit=1 << 20)
            row = definition(path, doc, binding)
            found.setdefault(row["name"], row)
    ctx.state["definitions"] = (stamp, found)
    return found


def _reader_adapter(ctx):
    """The ``read(path, yaml=, sha256=)`` shape :func:`cohort_rosters` expects, over ``ctx.reader``."""

    def read(path, *, yaml=False, sha256=None):
        path = Path(path)
        doc = ctx.reader.yaml(path) if yaml else ctx.reader.json(path)
        if not isinstance(doc, dict):
            return {}
        if sha256 is not None:
            ident = ctx.reader.identity(path)
            digests = ctx.state.setdefault("digests", {})
            if digests.get(str(path), (None,))[0] != ident:
                try:
                    digests[str(path)] = (ident, hashlib.sha256(path.read_bytes()).hexdigest())
                except OSError:
                    return {}
            if digests[str(path)][1] != sha256:
                return {}
        return doc

    return read


def _snapshots(run: Path) -> list[Path]:
    """Newest ``fruns_*`` directories first (their names end in a nanosecond stamp)."""
    work = run / "_qa_work"
    found = [Path(e.path) for e in scan_dir(work, 8192) if e.name.startswith("fruns_") and e.is_dir()]

    def stamp(path: Path) -> int:
        tail = path.name.rpartition("_")[2]
        return int(tail) if tail.isdigit() else 0

    return sorted(found, key=stamp, reverse=True)[:MAX_SNAPSHOTS]


def _apply_row(row: dict, result: dict, at: float | None, source: str) -> None:
    tiers = result.get("tiers")
    if isinstance(tiers, dict):
        row["tiers"] = {
            t: clean(v if isinstance(v, str) else obj(v).get("status"), 20) for t, v in tiers.items() if t in TIERS
        }
    row["status"] = clean(result.get("status"), 24) or None
    numeric = result.get("numeric_status") or obj(result.get("numeric")).get("status")
    row["numeric"] = clean(numeric, 20) or None
    cycles = obj(result.get("tier_cycles"))
    for tier in ("L2", "L3"):
        if num(cycles.get(tier)) is not None:
            row["cycles"][tier] = cycles[tier]
    row["failure"] = {
        "plane": clean(result.get("failure_plane"), 40) or None,
        "category": clean(result.get("failure_category"), 40) or None,
        "tier": clean(result.get("failure_tier"), 8) or None,
    }
    row["result_at"], row["result_source"] = at, source


def _results(ctx, run: Path, workspace: Path | None) -> list[tuple[float, dict, str]]:
    """Every recorded per-capsule row with the time its record was written."""
    out = []
    documents = []
    if workspace is not None:
        for name in ("progress.json", "verdict.json"):
            path = workspace / "qa" / name
            doc = ctx.reader.document(path)
            if doc:
                documents.append(
                    (epoch(doc.get("updated_at") or doc.get("graded_at")) or os.stat(path).st_mtime, doc, str(path))
                )
    for snapshot in _snapshots(run):
        path = snapshot / "snapshot_progress.json"
        doc = ctx.reader.document(path)
        if doc:
            documents.append(
                (epoch(doc.get("updated_at")) or epoch(int(snapshot.name.rpartition("_")[2] or 0)) or 0, doc, str(path))
            )
    for at, doc, source in documents:
        for result in rows(doc.get("per_capsule")):
            if isinstance(result.get("capsule"), str):
                out.append((at or 0, result, source))
    return out


def _capsule_results(ctx, run: Path) -> dict[str, Path]:
    """The newest ``capsule_result.json`` per capsule across graded snapshots."""
    newest: dict[str, Path] = {}
    for snapshot in _snapshots(run):
        for bench in scan_dir(snapshot / "runs", 16):
            for entry in scan_dir(Path(bench.path), 1024):
                path = Path(entry.path) / "capsule_result.json"
                if entry.name not in newest and is_file(path):
                    newest[entry.name] = path
                    if len(newest) >= MAX_RESULTS:
                        return newest
    return newest


def _apply_result(row: dict, doc: dict) -> None:
    tiers = obj(doc.get("tiers"))
    for tier, value in tiers.items():
        value = obj(value)
        if tier in TIERS and value.get("status") and tier not in row["tiers"]:
            row["tiers"][tier] = clean(value.get("status"), 20)
        if tier in ("L2", "L3") and num(value.get("cycles")) is not None and row["cycles"].get(tier) is None:
            row["cycles"][tier] = value["cycles"]
        timing = obj(value.get("timing"))
        for key in ("build_s", "sim_active_s", "oracle_wait_s"):
            seconds = num(timing.get(key))
            if seconds is not None:
                row["timing"][key] = round((row["timing"][key] or 0) + seconds, 3)
    failure = obj(doc.get("failure"))
    if failure and not row["failure"].get("category"):
        row["failure"] = {
            "plane": clean(failure.get("plane"), 40) or None,
            "category": clean(failure.get("category"), 40) or None,
            "tier": None,
        }
    if row["status"] is None:
        row["status"] = clean(doc.get("status"), 24) or None


def catalog(ctx) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """``(rows by name, notes)``: definitions joined with cohorts and the newest results."""
    experiment = ctx.experiment
    run = Path(experiment["run_dir"])
    workspace = Path(experiment["workspace"]) if experiment.get("workspace") else None
    notes: list[str] = []
    definitions = _definitions(ctx, run)
    table = {
        name: {
            **row,
            "tiers": {},
            "cycles": dict(row["cycles"]),
            "failure": dict(row["failure"]),
            "timing": dict(row["timing"]),
        }
        for name, row in definitions.items()
    }
    _, rosters = cohort_rosters(_reader_adapter(ctx), run)
    for cohort, names in rosters.items():
        if names is None:
            notes.append(f"{cohort} roster not recorded")
            continue
        for name in names:
            row = table.get(name)
            if row is None:
                table[name] = row = {**definition(run / name / "capsule.yaml", {"name": name}, None), "path": None}
            row["cohort"] = row["cohort"] or cohort
    newest: dict[str, tuple[float, dict, str]] = {}
    for at, result, source in _results(ctx, run, workspace):
        name = result["capsule"]
        if name not in newest or at > newest[name][0]:
            newest[name] = (at, result, source)
    for name, (at, result, source) in newest.items():
        row = table.get(name)
        if row is None:
            table[name] = row = {**definition(run / name / "capsule.yaml", {"name": name}, None), "path": None}
        _apply_row(row, result, at, source)
    for name, path in _capsule_results(ctx, run).items():
        doc = ctx.reader.document(path, limit=4 << 20)
        if doc and name in table:
            _apply_result(table[name], doc)
            table[name]["result_path"] = str(path)
    if workspace is not None:
        ledger = ctx.reader.document(workspace / "qa" / "tier_state.json", limit=16 << 20)
        for name, state in obj(ledger).items():
            row = table.get(name)
            if row is None or name.startswith("<"):
                continue
            for tier in ("L2", "L3"):
                status = obj(obj(state).get(tier)).get("status")
                if status and tier not in row["tiers"]:
                    row["tiers"][tier] = clean(status, 20)
    if not definitions:
        notes.append("staged capsule definitions not found under private_corpus_input")
    return table, notes


def _counts(table: dict[str, dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {
        k: {} for k in ("cohort", "kind", "op", "status", "family", "pass_requirements")
    }
    for row in table.values():
        for key in ("cohort", "kind", "op", "status", "family"):
            value = row.get(key) or "not recorded"
            counts[key][value] = counts[key].get(value, 0) + 1
        for token in row.get("pass_requirements") or ():
            counts["pass_requirements"][token] = counts["pass_requirements"].get(token, 0) + 1
    return counts


def read(ctx) -> dict[str, Any]:
    table, notes = catalog(ctx)
    ordered = sorted(table.values(), key=lambda r: (str(r.get("cohort") or "~"), r["name"]))
    section = {
        "capsules": ordered if ctx.operator_private else [],
        "counts": _counts(table),
        "total": len(table),
        "notes": notes,
    }
    if not ctx.operator_private:
        section["notes"] = notes + ["capsule names and per-capsule rows require operator-private mode"]
    return section


def _invocations(directory: Path) -> list[str]:
    lines = []
    generated = directory / "generated" / "invocations"
    records = []
    for entry in scan_dir(generated, 4096):
        path = Path(entry.path) / "invocation.json"
        if is_file(path):
            records.append(path)
    docs = []
    for path in records[: MAX_INVOCATIONS * 4]:
        try:
            import json

            doc = json.loads(path.read_bytes()[: 4 << 20])
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict):
            docs.append(doc)
    docs.sort(key=lambda d: d.get("started_ns") or 0)
    for doc in docs[:MAX_INVOCATIONS]:
        start, end = num(doc.get("started_ns")), num(doc.get("finished_ns"))
        seconds = f"{(end - start) / 1e9:8.2f}s" if start and end else "        ?"
        lines.append(
            f"{clean(doc.get('stage'), 32):<32} rc={doc.get('returncode')!s:<4} {seconds}"
            f"  {clean(doc.get('status'), 16)}"
        )
    if len(records) > MAX_INVOCATIONS:
        lines.append(f"... {len(records) - MAX_INVOCATIONS} more invocation records")
    return lines


def inspect(ctx, key: str) -> dict[str, Any]:
    if not ctx.operator_private:
        return {
            "title": "capsule detail",
            "sections": [],
            "paths": [],
            "notice": "capsule detail requires operator-private mode",
        }
    table, _ = catalog(ctx)
    row = table.get(key)
    if row is None:
        return {
            "title": f"capsule {clean(key, 80)}",
            "sections": [],
            "paths": [],
            "notice": "not in this run's catalog",
        }
    origin = row["origin"]
    identity = [
        f"cohort {row.get('cohort') or 'not recorded'}   kind {row.get('kind')}   op {row.get('op')}"
        f"   dtype {row.get('dtype')}",
        f"family {row.get('family')}   axis {row.get('axis')}   role {row.get('source_role')}",
        "attributes " + (row.get("attrs") or "none recorded"),
    ]
    identity += ["input " + s for s in row["shapes"]]
    provenance = [f"{k}: {v}" for k, v in origin.items() if v is not None] or ["no origin recorded"]
    tiers = [
        f"{t}: {row['tiers'].get(t, 'not recorded')}"
        + (f"  cycles {row['cycles'][t]:,}" if row["cycles"].get(t) else "")
        for t in TIERS
    ]
    tiers.append("required " + (", ".join(row["required_tiers"]) or "not recorded"))
    timing = [f"{k} {v}" for k, v in row["timing"].items() if v is not None] or ["no tier timings recorded"]
    failure = [f"{k}: {v}" for k, v in row["failure"].items() if v] or ["no failure recorded"]
    paths = [p for p in (row.get("path"), row.get("result_source"), row.get("result_path")) if p]
    sections = [
        {"title": "Identity", "lines": identity},
        {"title": "Origin", "lines": provenance},
        {"title": "Tiers", "lines": tiers},
        {"title": "Timing (summed over tiers)", "lines": timing},
        {"title": "Failure", "lines": failure},
        {"title": "Pass requirements", "lines": row["pass_requirements"] or ["none declared"]},
    ]
    if row.get("result_source") and row["result_source"].endswith(".json"):
        doc = ctx.reader.document(Path(row["result_source"]))
        for result in rows(obj(doc).get("per_capsule")):
            if result.get("capsule") == key:
                reasons = result.get("tier_reasons")
                detail = result.get("failure_detail")
                lines = [f"{t}: {clean(v, 300)}" for t, v in obj(reasons).items()]
                if detail:
                    lines.append(clean(detail, 400))
                if lines:
                    sections.append({"title": "Recorded reasons", "lines": lines})
                break
    if row.get("result_path"):
        lines = _invocations(Path(row["result_path"]).parent)
        if lines:
            sections.append({"title": "Invocations (stage, return code, wall)", "lines": lines})
    return {"title": f"capsule {row['name']}", "sections": sections, "paths": paths}


__all__ = ["catalog", "definition", "inspect", "read"]
