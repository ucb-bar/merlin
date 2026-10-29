"""Seed preparation from the seed run's own records (the default, launch-path mode).

The seed run already timed its cases on the Spike cycle model and compared its variants.
This module reads those records instead of simulating again:

* baseline cycles per case come from ``qa/perf_history.jsonl`` (complete timing-model rows) and
  the per-case detail rows of the last completed grade (``qa/cases/<case>/<sha>.json``, which also
  carry command counts); a record whose graded source is the seed's own tree hash is preferred,
  otherwise the newest record is used and labelled with the source it was measured on;
* a case recorded only with counts gets the calibrated fast estimate from those counts, labelled;
* variant decisions come from the run's A/B records (``qa/idle_tuning.json`` experiments: the
  same model and timing epoch for the baseline and the candidate, plus the numerical comparison).

Nothing here starts a simulator.  ``--measure`` in the CLI is the opt-in path that re-times.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import baseline as B

PERF_HISTORY = ("qa", "perf_history.jsonl")
LATEST_GRADE = ("qa", "latest_completed_grade.json")
IDLE_TUNING = ("qa", "idle_tuning.json")
UNCHANGED_NUMERICS = "unchanged"


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def perf_history(workspace: Path, model_sha256: str | None = None) -> dict[str, list[dict]]:
    """Complete timing-model rows per case, oldest first."""
    out: dict[str, list[dict]] = {}
    path = workspace.joinpath(*PERF_HISTORY)
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("kind") != "modeled" or not isinstance(row.get("cycles"), int):
            continue
        ident = (row.get("performance_identity") or {}).get("model_sha256")
        if model_sha256 and ident and ident != model_sha256:
            continue
        out.setdefault(row.get("test_case"), []).append(row)
    for rows in out.values():
        rows.sort(key=lambda r: float(r.get("t") or 0))
    return out


COMPLETED_ROWS = ("qa", "completed_rows.jsonl")
COUNT_KEYS = ("loads", "computes", "stores", "fences", "fences_per_output_tile", "transfer_widths",
              "capacity_fallback", "refusal_records")


def _when(stamp) -> float:
    from datetime import datetime

    try:
        return datetime.fromisoformat(str(stamp)).timestamp()
    except ValueError:
        return 0.0


def case_details(workspace: Path, cases: set[str]) -> dict[str, list[dict]]:
    """Every graded detail row the run kept for ``cases``, with the source and time it was graded on."""
    out: dict[str, list[dict]] = {}
    path = workspace.joinpath(*COMPLETED_ROWS)
    seen: set[str] = set()
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        case, detail = row.get("capsule"), (row.get("detail") or {}).get("path")
        if case not in cases or not detail or detail in seen:
            continue
        seen.add(detail)
        record = _read_json(workspace / detail)
        if isinstance(record, dict):
            graded = row.get("graded") or {}
            out.setdefault(case, []).append({"row": record, "source": graded.get("graded_source_sha256") or "",
                                             "t": _when(graded.get("graded_at"))})
    return out


def _row_from_history(case: str, rec: dict) -> dict:
    cycles, bound = rec["cycles"], rec.get("roofline_bound_cycles")
    return {
        "case": case, "family": B.case_family(case), "status": "timed", "model_cycles": cycles,
        "evidence": rec.get("evidence") or "recorded Spike timing model",
        "roofline_cycles": round(bound, 1) if isinstance(bound, (int, float)) else None,
        "bound_by": rec.get("bound_by"), "utilization_pct": rec.get("array_utilization_pct"),
        "headroom_cycles": int(cycles - bound) if isinstance(bound, (int, float)) else None,
        "model_sha256": (rec.get("performance_identity") or {}).get("model_sha256"),
        "loads": None, "computes": None, "stores": None, "fences": None, "fences_per_output_tile": None,
        "transfer_widths": {}, "capacity_fallback": None, "refusal_records": [],
    }


def baseline(seed_sha256: str, cases: list[str], workspace: Path, model_sha256: str | None = None,
             measured: dict[str, dict] | None = None) -> list[dict]:
    """One row per case from the records; ``source_match`` says whether the bytes are the seed's.

    ``measured`` rows (an earlier ``--measure`` run on these exact seed bytes) count as exact records.
    """
    if model_sha256 is None:
        # One model per baseline: the one most of the run's complete estimates were made with.
        tally: dict[str, int] = {}
        for rows_ in perf_history(workspace).values():
            for rec in rows_:
                ident = (rec.get("performance_identity") or {}).get("model_sha256")
                if ident:
                    tally[ident] = tally.get(ident, 0) + 1
        model_sha256 = max(tally, key=tally.get) if tally else None
    history = perf_history(workspace, model_sha256)
    details = case_details(workspace, set(cases))
    rows: list[dict] = []
    for case in cases:
        options: list[tuple[int, float, int, dict, str, str]] = []  # (exact, time, has counts, row, source, kind)
        for rec in history.get(case, []):
            source = rec.get("graded_source_sha256") or rec.get("source_sha256") or ""
            exact = seed_sha256 in (rec.get("graded_source_sha256"), rec.get("source_sha256"))
            options.append((int(exact), float(rec.get("t") or 0), 0, _row_from_history(case, rec), source, "history"))
        counted: list[dict] = []
        for item in details.get(case, []):
            perf = item["row"].get("performance") or {}
            model = perf.get("model_sha256")
            if model_sha256 and model and model != model_sha256:
                continue
            row = B.summarize(case, {"observed": {"performance": perf, "tiers": item["row"].get("tiers") or {}}})
            if row.get("loads") is not None:
                counted.append(row)
            options.append((int(item["source"] == seed_sha256), item["t"], int(row.get("loads") is not None),
                            row, item["source"], "grade"))
        mine = (measured or {}).get(case)
        if mine and mine.get("status") in ("timed", "fast_estimate"):
            options.append((1, float("inf"), int(mine.get("loads") is not None), dict(mine), seed_sha256, "measured"))
        usable = [o for o in options if o[3]["status"] != "UNKNOWN"]
        if not usable:
            reason = next((o[3].get("unknown_reason") or o[3].get("timing_reason") for o in options), None)
            rows.append({"case": case, "family": B.case_family(case), "status": "UNKNOWN", "model_cycles": None,
                         "unknown_reason": (reason or "no record for this case in the seed run")
                         + "; time it with --measure"})
            continue
        exact, _t, _c, row, source, kind = max(usable, key=lambda o: (o[0], o[1], o[2]))
        row = dict(row)
        if row.get("loads") is None:
            # Counts come from a graded row of the same executable cost when one exists.
            same = next((c for c in counted if c.get("model_cycles") == row.get("model_cycles")), None)
            if same:
                row.update({k: same.get(k) for k in COUNT_KEYS})
        row["record"] = kind
        row["record_source_sha256"] = source
        row["source_match"] = "seed" if exact else "older source"
        label = {"history": "recorded Spike timing model", "grade": "recorded Spike timing model (graded row)",
                 "measured": "measured on the seed bytes (--measure)"}[kind]
        if row["status"] == "fast_estimate":
            label = "fast estimate from recorded counts"
        row["evidence"] = label + ("" if exact else f", source {source[:12]}, not the seed's bytes")
        rows.append(row)
    return rows


def variant_records(workspace: Path) -> dict[str, list[dict]]:
    """The run's A/B experiments grouped by variant name (same model and epoch on both sides)."""
    doc = _read_json(workspace.joinpath(*IDLE_TUNING)) or {}
    out: dict[str, list[dict]] = {}
    for exp in doc.get("experiments") or []:
        cand = exp.get("candidate") or {}
        if cand.get("kind") != "variant" or not cand.get("name"):
            continue
        before, after = exp.get("measured_cycles_before"), exp.get("measured_cycles_after")
        if not isinstance(after, int):
            continue
        out.setdefault(cand["name"], []).append({
            "case": exp.get("test_case"), "recorded_before": before, "after": after,
            "numerics": exp.get("numerics"), "baseline_status": exp.get("baseline_status"),
            "candidate_current": exp.get("candidate_current"), "graded_at_unix_ns": exp.get("graded_at_unix_ns"),
        })
    for name, items in out.items():
        newest: dict[str, dict] = {}
        for item in sorted(items, key=lambda i: i.get("graded_at_unix_ns") or 0):
            newest[item["case"]] = item
        out[name] = list(newest.values())
    return out


def decide(change_status: str, records: list[dict], seed_rows: dict[str, dict]) -> dict:
    """Promotion decision for one variant from its records against the seed's baseline rows."""
    cases = []
    ok = bool(records)
    before_total = after_total = 0
    for item in records:
        seed = seed_rows.get(item["case"]) or {}
        before = seed.get("model_cycles") if seed.get("status") == "timed" else None
        numerics = str(item.get("numerics") or "")
        passed = numerics.startswith(UNCHANGED_NUMERICS)
        cases.append({"case": item["case"], "before": before, "after": item["after"],
                      "recorded_before": item.get("recorded_before"), "numerics": "pass" if passed else numerics})
        if before is None or not passed or item["after"] >= before:
            ok = False
        else:
            before_total += before
            after_total += item["after"]
    if change_status != "applicable":
        reason = f"change is {change_status} against the seed"
        ok = False
    elif not records:
        reason = "no A/B record in the seed run"
    elif ok:
        reason = (f"recorded faster on every compared case {before_total:,}->{after_total:,} model cycles "
                  "(after = the variant tree's record; seed = baseline map) with unchanged numerics")
    else:
        reason = "not faster than the seed on every recorded case with unchanged numerics"
    return {"promote": ok, "reason": reason, "cases": cases, "gain_cycles": before_total - after_total}
