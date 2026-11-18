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

from merlin.development.backend.campaign.seeding import baseline as B
from merlin.development.backend.campaign.seeding.variants import FAMILIES, FamilyComparison

# ---- records -----------------------------------------------------------------------------------

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

COUNT_KEYS = (
    "loads",
    "computes",
    "stores",
    "fences",
    "fences_per_output_tile",
    "transfer_widths",
    "capacity_fallback",
    "refusal_records",
)


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
            out.setdefault(case, []).append(
                {"row": record, "source": graded.get("graded_source_sha256") or "", "t": _when(graded.get("graded_at"))}
            )
    return out


def _row_from_history(case: str, rec: dict) -> dict:
    cycles, bound = rec["cycles"], rec.get("roofline_bound_cycles")
    return {
        "case": case,
        "family": B.case_family(case),
        "status": "timed",
        "model_cycles": cycles,
        "evidence": rec.get("evidence") or "recorded Spike timing model",
        "roofline_cycles": round(bound, 1) if isinstance(bound, (int, float)) else None,
        "bound_by": rec.get("bound_by"),
        "utilization_pct": rec.get("array_utilization_pct"),
        "headroom_cycles": int(cycles - bound) if isinstance(bound, (int, float)) else None,
        "model_sha256": (rec.get("performance_identity") or {}).get("model_sha256"),
        "loads": None,
        "computes": None,
        "stores": None,
        "fences": None,
        "fences_per_output_tile": None,
        "transfer_widths": {},
        "capacity_fallback": None,
        "refusal_records": [],
    }


def baseline(
    seed_sha256: str,
    cases: list[str],
    workspace: Path,
    model_sha256: str | None = None,
    measured: dict[str, dict] | None = None,
) -> list[dict]:
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
            options.append(
                (
                    int(item["source"] == seed_sha256),
                    item["t"],
                    int(row.get("loads") is not None),
                    row,
                    item["source"],
                    "grade",
                )
            )
        mine = (measured or {}).get(case)
        if mine and mine.get("status") in ("timed", "fast_estimate"):
            options.append((1, float("inf"), int(mine.get("loads") is not None), dict(mine), seed_sha256, "measured"))
        usable = [o for o in options if o[3]["status"] != "UNKNOWN"]
        if not usable:
            reason = next((o[3].get("unknown_reason") or o[3].get("timing_reason") for o in options), None)
            rows.append(
                {
                    "case": case,
                    "family": B.case_family(case),
                    "status": "UNKNOWN",
                    "model_cycles": None,
                    "unknown_reason": (reason or "no record for this case in the seed run")
                    + "; time it with --measure",
                }
            )
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
        label = {
            "history": "recorded Spike timing model",
            "grade": "recorded Spike timing model (graded row)",
            "measured": "measured on the seed bytes (--measure)",
        }[kind]
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
        out.setdefault(cand["name"], []).append(
            {
                "case": exp.get("test_case"),
                "recorded_before": before,
                "after": after,
                "numerics": exp.get("numerics"),
                "baseline_status": exp.get("baseline_status"),
                "candidate_current": exp.get("candidate_current"),
                "graded_at_unix_ns": exp.get("graded_at_unix_ns"),
            }
        )
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
        cases.append(
            {
                "case": item["case"],
                "before": before,
                "after": item["after"],
                "recorded_before": item.get("recorded_before"),
                "numerics": "pass" if passed else numerics,
            }
        )
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
        reason = (
            f"recorded faster on every compared case {before_total:,}->{after_total:,} model cycles "
            "(after = the variant tree's record; seed = baseline map) with unchanged numerics"
        )
    else:
        reason = "not faster than the seed on every recorded case with unchanged numerics"
    return {"promote": ok, "reason": reason, "cases": cases, "gain_cycles": before_total - after_total}


# ---- report ------------------------------------------------------------------------------------
# Write the seed preparation products: SEED_BASELINE.{json,md} and SEED_STRUCTURE.md.


def _num(value) -> str:
    if value is None:
        return "UNKNOWN"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _widths(value) -> str:
    if not value:
        return "-"
    if isinstance(value, dict):
        return ",".join(f"{k}:{v}" for k, v in sorted(value.items()))
    return ",".join(str(v) for v in value)


def ranked_gaps(rows: list[dict], per_family: int) -> dict[str, list[dict]]:
    """Largest absolute cycle gaps (model cycles minus roofline bound) per family."""
    out: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("headroom_cycles") is None:
            continue
        out.setdefault(row.get("family") or "other", []).append(row)
    return {fam: sorted(items, key=lambda r: -r["headroom_cycles"])[:per_family] for fam, items in sorted(out.items())}


def write_baseline(out: Path, payload: dict, per_family: int = 8) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "SEED_BASELINE.json").write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    rows = payload["baseline"]["cases"]
    lines = [
        "# Seed baseline performance map",
        "",
        f"Seed compiler: `{payload['seed']['path']}` (tree sha256 `{payload['seed']['sha256'][:16]}`).",
        f"Export: `{payload['export']['path']}`. Timing: {payload['baseline']['method']}.",
        "Operator-side preparation result: Spike cycle-model numbers only; no physical measurements.",
        f"Scope: {payload['baseline']['scope']}.",
        "",
        "## Largest cycle gaps per family",
        "",
    ]
    for family, items in ranked_gaps(rows, per_family).items():
        lines.append(f"### {family}")
        lines.append("")
        lines.append("| case | model cycles | roofline | util % | headroom | bound |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for r in items:
            lines.append(
                f"| {r['case']} | {_num(r.get('model_cycles'))} | {_num(r.get('roofline_cycles'))} | "
                f"{_num(r.get('utilization_pct'))} | {_num(r.get('headroom_cycles'))} | "
                f"{r.get('bound_by') or '-'} |"
            )
        lines.append("")
    lines += [
        "## All cases",
        "",
        "| case | family | status | model cycles | roofline | util % | headroom | fences/out tile | "
        "loads | computes | transfer widths | capacity fallback |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (r.get("family") or "", r["case"])):
        lines.append(
            f"| {r['case']} | {r.get('family') or '-'} | {r['status']} | {_num(r.get('model_cycles'))} | "
            f"{_num(r.get('roofline_cycles'))} | {_num(r.get('utilization_pct'))} | "
            f"{_num(r.get('headroom_cycles'))} | {_num(r.get('fences_per_output_tile'))} | "
            f"{_num(r.get('loads'))} | {_num(r.get('computes'))} | {_widths(r.get('transfer_widths'))} | "
            f"{_num(r.get('capacity_fallback'))} |"
        )
    lines.append("")
    promo = payload.get("promotion") or {}
    lines += ["## Variant promotion", ""]
    if not promo:
        lines.append("Not run.")
    else:
        lines.append(
            f"Variants examined: {promo.get('examined', 0)}; promoted tree: `{promo.get('promoted_tree') or 'none'}`."
        )
        lines.append("")
        for title, key in (("Promoted", "promoted"), ("Candidates", "candidates")):
            entries = promo.get(key) or []
            lines.append(f"### {title} ({len(entries)})")
            lines.append("")
            for e in entries:
                cases = "; ".join(
                    f"{c['case']} {_num(c.get('before'))}->{_num(c.get('after'))}"
                    + ("" if c.get("numerics") in (None, "pass") else f" numerics {c['numerics']}")
                    for c in e.get("cases", [])
                )
                lines.append(
                    f"- **{e['name']}** ({e.get('reason', '')}): files {', '.join(e.get('files', []))}"
                    + (f"; {cases}" if cases else "")
                )
            lines.append("")
        if promo.get("discovered"):
            lines.append(
                "Variant changes discovered: "
                + ", ".join(f"{k} {v}" for k, v in sorted(promo["discovered"].items()))
                + "."
            )
            lines.append("")
        if promo.get("not_examined"):
            lines.append(
                f"Applicable but not examined in this bounded run ({len(promo['not_examined'])}): "
                + ", ".join(promo["not_examined"])
            )
            lines.append("")
        skipped = promo.get("not_applicable") or []
        if skipped:
            lines.append(
                f"Not applicable ({len(skipped)}): " + ", ".join(f"{s['name']} ({s['status']})" for s in skipped)
            )
            lines.append("")
    (out / "SEED_BASELINE.md").write_text("\n".join(lines), encoding="utf-8")


def structure_lines(rows: list[FamilyComparison], effects: dict[str, list[str]]) -> list[str]:
    """Short actionable lines, one block per family."""
    lines: list[str] = []
    for row in rows:
        own = ", ".join(f"{name} ({n} lines)" for name, n in sorted(row.seed_author.items()))
        if row.seed_library:
            uses = "uses library " + ", ".join(m.rpartition(".")[2] for m in row.seed_library)
        else:
            uses = "uses own implementation only"
        if own:
            uses += f"; own modules {own}"
        lines.append(f"- **{row.family}**: {uses}.")
        for module, labels in sorted(row.missing_from_seed.items()):
            note = f"  - library `{module.rpartition('.')[2]}` used by {', '.join(labels)}, not by the seed"
            extra = effects.get(module)
            if extra:
                note += "; " + "; ".join(extra)
            lines.append(note + ".")
        for label, effect_lines in effects.items():
            if label.startswith(f"family:{row.family}:"):
                lines.extend(f"  - {e}" for e in effect_lines)
    return lines


def write_structure(
    out: Path, seed_label: str, compared: list[str], rows: list[FamilyComparison], effects: dict[str, list[str]]
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    head = [
        "# Seed structure by family",
        "",
        f"Seed `{seed_label}` compared with {', '.join(compared) or 'no other compilers'}. Routes are the "
        "library modules each compiler's author modules import (static analysis). Cycle effects are Spike "
        "cycle-model numbers on shared cases where measured; UNKNOWN otherwise.",
        "",
    ]
    order = {f: i for i, f in enumerate(FAMILIES)}
    rows = sorted(rows, key=lambda r: order.get(r.family, len(order)))
    (out / "SEED_STRUCTURE.md").write_text("\n".join(head + structure_lines(rows, effects)) + "\n", encoding="utf-8")
