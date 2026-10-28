"""The seed baseline and variant promotion from the seed run's own records: no simulation at all.

``seed_prep --from-records <workspace>`` reads what the seed run already measured instead of timing
the seed again:

* per case, the newest complete timing-model row of the seed run's ``qa/perf_history.jsonl`` whose
  source is the case's graded source (the source that earned the verdict the last completed grade
  reports) or the seed's graded source: ``status: timed``;
* otherwise the calibrated fast estimate (:mod:`..feedback.fast_estimate`) of the decoded instruction
  stream the seed run's grade left for that case: ``status: fast_estimate``;
* otherwise, when the run timed the case only on another snapshot, that row, labelled with its source:
  ``status: recorded_other_snapshot``; else ``UNKNOWN`` with the reason;
* a variant is promoted only when the run recorded complete timings for the variant's own tree, every
  recorded case is faster than the seed's recorded cycles and the run recorded a passing L2 status for
  that tree; every other applicable variant is a candidate for the run to measure.

Rows keep the shape :func:`.baseline.summarize` gives, so the report and the ranked gaps are unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

from .baseline import case_family


def _rows(path: Path) -> list[dict]:
    rows = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _newest(rows):
    return max(rows, key=lambda r: float(r.get("t") or 0)) if rows else None


def _grade_dir(run_dir: Path | None, token, case: str) -> Path | None:
    if run_dir is None or not token:
        return None
    hits = sorted((run_dir / "_qa_work" / f"fruns_{token}" / "runs").glob(f"*/{case}"))
    return hits[0] if hits else None


def _timed(case: str, row: dict, status: str, evidence: str) -> dict:
    cycles, bound = row.get("cycles"), row.get("roofline_bound_cycles")
    identity = row.get("performance_identity") or {}
    return {
        "status": status, "evidence": evidence, "model_cycles": cycles,
        "roofline_cycles": bound, "bound_by": row.get("bound_by"),
        "utilization_pct": row.get("array_utilization_pct"),
        "headroom_cycles": int(cycles - bound) if cycles and bound else None,
        "model_sha256": identity.get("model_sha256"),
        "recorded": {"source_sha256": row.get("source_sha256"), "grade_token": row.get("grade_token"),
                     "evidence": row.get("evidence"), "t": row.get("t")},
    }


def baseline_rows(workspace: Path, run_dir: Path | None, cases, *, seed_source: str | None,
                  facts_sha256: str | None) -> list[dict]:
    """One baseline row per case from the seed run's records (see the module docstring)."""
    from ..feedback import fast_estimate as FE

    history = _rows(workspace / "qa" / "perf_history.jsonl")
    modeled = [r for r in history if r.get("kind") == "modeled" and isinstance(r.get("cycles"), int)]
    grade_path = workspace / "qa" / "latest_completed_grade.json"
    grade = json.loads(grade_path.read_text(encoding="utf-8")) if grade_path.is_file() else {}
    per_case = {r.get("capsule"): r for r in grade.get("per_capsule") or () if isinstance(r, dict)}
    calibrated = FE.calibration(facts_sha256, path=Path("/nonexistent-ledger")) if facts_sha256 else None
    out = []
    for case in cases:
        verdict = per_case.get(case) or {}
        graded = verdict.get("graded") or {}
        sources = {s for s in (graded.get("graded_source_sha256"), seed_source) if s}
        mine = [r for r in modeled if r.get("test_case") == case]
        row = {"case": case, "family": case_family(case),
               "numerics": (verdict.get("tiers") or {}).get("L2"), "wall_s": 0.0,
               "graded": {k: graded.get(k) for k in ("token", "graded_source_sha256", "currency")} if graded else None}
        exact = _newest([r for r in mine if r.get("source_sha256") in sources])
        root = _grade_dir(run_dir, graded.get("token"), case)
        counted = FE.census(FE.trace_for(root)) if root is not None else None
        if counted:
            row.update(commands=counted.get("commands"), fences=counted.get("fences"))
        if exact is not None:
            row.update(_timed(case, exact, "timed", "recorded complete Spike timing model of the graded source"))
        elif counted:
            structural = {k: v for k, v in (_newest(mine) or {}).items()
                          if k in ("useful_macs", "structural_peak_macs_per_cycle", "roofline_bound_cycles")}
            if "roofline_bound_cycles" in structural:
                structural["bound_cycles"] = structural.pop("roofline_bound_cycles")
            estimate = FE.estimate(counted, structural, calibrated)
            if estimate.get("status") == "ESTIMATED":
                row.update(status="fast_estimate", model_cycles=estimate["cycles"],
                           evidence=f"fast census estimate of the graded executable (band {estimate.get('low_cycles')}-"
                                    f"{estimate.get('high_cycles')})",
                           fast_estimate={k: estimate.get(k) for k in ("cycles", "low_cycles", "high_cycles",
                                                                        "lower_bound", "limited_by")})
                bound, useful = structural.get("bound_cycles"), structural.get("useful_macs")
                peak = structural.get("structural_peak_macs_per_cycle")
                row.update(roofline_cycles=bound, headroom_cycles=int(estimate["cycles"] - bound) if bound else None,
                           utilization_pct=round(100.0 * useful / (estimate["cycles"] * peak), 2)
                           if useful and peak else None)
            else:
                row.update(status="UNKNOWN", model_cycles=None, unknown_reason=estimate.get("reason"))
        if row.get("status") in (None, "UNKNOWN") and mine:
            other = _newest(mine)
            row.update(_timed(case, other, "recorded_other_snapshot",
                              f"recorded complete Spike timing model of another seed-run snapshot "
                              f"({str(other.get('source_sha256'))[:12]}); may differ from the seed"))
        if row.get("status") is None:
            row.update(status="UNKNOWN", model_cycles=None,
                       unknown_reason="the seed run recorded no timing and no decoded stream for this case"
                       + ("" if verdict else " (not graded by the seed run)"))
        row.setdefault("roofline_cycles", None)
        row.setdefault("headroom_cycles", None)
        row.setdefault("utilization_pct", None)
        out.append(row)
    return out


def promotion(workspace: Path, seed: Path, rows: dict[str, dict], variants_root: Path | None) -> dict:
    """Promote variants only on recorded evidence; every other applicable variant is a candidate."""
    from merlin.common.tree_hash import hash_tree

    from . import variants as V

    result: dict = {"basis": "the seed run's own records; no simulation", "promoted": [], "candidates": [],
                    "not_applicable": [], "examined": 0}
    if variants_root is None or not Path(variants_root).is_dir():
        return result
    history = _rows(workspace / "qa" / "perf_history.jsonl")
    by_source: dict[str, list[dict]] = {}
    for r in history:
        by_source.setdefault(r.get("source_sha256") or r.get("graded_source_sha256"), []).append(r)
    found = V.discover(Path(variants_root), seed, [p for p in (Path(variants_root).parent / "checkpoints").glob("*")
                                                   if p.is_dir()])
    statuses: dict[str, int] = {}
    for change, _merged in found:
        statuses[change.status] = statuses.get(change.status, 0) + 1
    result["discovered"] = statuses
    for change, _merged in found:
        if change.status != "applicable":
            result["not_applicable"].append({"name": change.name, "status": change.status})
            continue
        result["examined"] += 1
        tree = Path(variants_root) / change.name
        recorded = by_source.get(hash_tree(tree)["sha256"], []) if tree.is_dir() else []
        timed = {}
        for r in sorted((r for r in recorded if r.get("kind") == "modeled"), key=lambda r: float(r.get("t") or 0)):
            timed[r.get("test_case")] = r.get("cycles")
        passing = {r.get("test_case") for r in recorded if r.get("kind") == "status" and r.get("l2") == "pass"}
        entry = {"name": change.name, "files": change.files, "cases": []}
        before_total = after_total = 0
        faster = bool(timed)
        for case, cycles in sorted(timed.items()):
            seed_row = rows.get(case) or {}
            before = seed_row.get("model_cycles") if seed_row.get("status") == "timed" else None
            entry["cases"].append({"case": case, "before": before, "after": cycles,
                                   "before_evidence": seed_row.get("status"), "after_evidence": "recorded",
                                   "numerics": "pass" if case in passing else None})
            if before and cycles:
                before_total += before
                after_total += cycles
                faster = faster and cycles < before
            else:
                faster = False
        if faster and timed and set(timed) <= passing and after_total < before_total:
            entry["reason"] = (f"recorded faster {before_total:,}->{after_total:,} model cycles on every recorded case "
                               "with recorded passing numerics")
            entry["gain_cycles"] = before_total - after_total
            result["promoted"].append(entry)
        else:
            entry["reason"] = ("candidate: no recorded timing of this variant's tree; fast-estimate delta unavailable "
                               "from the records (no decoded stream attributable to the tree); measure it in the run"
                               if not timed else
                               "candidate: recorded timings do not show it faster on every case with recorded "
                               "passing numerics; measure it in the run")
            result["candidates"].append(entry)
    return result
