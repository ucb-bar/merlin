#!/usr/bin/env python3
"""Distill every experiment run into one durable index, and SAY what each run is missing.

WHY THIS EXISTS. A graded run costs ~47 GB on disk and carries ~13 MB of signal: which capsules passed
which oracle tier, what it cost in tokens and wall time, the config it ran under, and the agent's own
transcript. The other 99.97% is per-capsule build trees and per-cycle simulator console — regenerable,
and it filled a 3.6 TB filesystem to 92%. Once the bulk is reclaimed, the signal is the whole record of
the experiment, and it is spread across five files per run in formats that changed as the harness did.
This walks all of it into one artifact so the history is queryable without the runs, and so that
reclaiming disk never again risks the part worth keeping.

IT REPORTS ABSENCE. Each run's entry carries a ``missing`` list naming every keep-list item that is not
there. That is the point rather than a footnote: a census that quietly omits what it could not find reads
as complete coverage, and the same "a check that could not run reported success" failure has now been
measured three times in this repo. An older run whose harness never wrote ``run_record.json`` is a
different thing from a run whose record was deleted, and only a stated absence lets you tell them apart.

Emits ``out/artifacts/runs-index/<TS>_<sha7>/`` with ``runs_index.json`` (full) and ``runs_index.csv``
(one row per run, for a spreadsheet or a quick sort).
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "merlin/python"))
from merlin.common.paths import artifacts_dir, repo_root  # noqa: E402

# The keep-list, named once. Each entry is (label, extractor) and every extractor returns None when the
# evidence is absent rather than a zero — a zero is a measurement, None is a gap.
KEEP = ("tiers", "per_tier", "tokens", "tool_calls", "wall_s", "config", "aet", "transcript", "submission")


def _yaml(p: Path):
    try:
        return yaml.safe_load(p.read_text()) or {}
    except Exception:                                    # noqa: BLE001 -- unreadable == absent
        return {}


def _json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:                                    # noqa: BLE001
        return None


def per_tier_detail(run: Path) -> dict | None:
    """Per-capsule oracle-tier verdicts, from the newest verdict that carries them.

    Read from ``qa_history/verdict_round_*.json``. Passing rows do NOT carry a ``tiers`` map — only the
    barrier fields — so both shapes are accepted; keying on ``tiers`` alone reports "L3 never ran" for a
    run in which every capsule certified.
    """
    best = None
    for f in sorted(run.glob("qa_history/verdict_round_*.json")):
        d = _json(f)
        if not isinstance(d, dict):
            continue
        rows = d.get("per_capsule") or []
        if not rows:
            continue
        detail = {}
        for row in rows:
            name = row.get("capsule")
            if not name:
                continue
            entry = {"pass": bool(row.get("pass"))}
            if row.get("tiers"):
                entry["tiers"] = row["tiers"]
            for k in ("barrier_tier", "barrier_status", "barrier_gates", "required_tiers"):
                if row.get(k) is not None:
                    entry[k] = row[k]
            detail[name] = entry
        if detail and any(("tiers" in v or "barrier_status" in v) for v in detail.values()):
            best = {"source": f.name, "capsules": detail}
    return best


def wall_seconds(run: Path) -> float | None:
    """Total agent wall time, from whichever record this run's harness generation wrote."""
    tot = 0.0
    found = False
    for f in sorted(run.glob("rounds/*.codex_summary.json")):
        d = _json(f)
        if isinstance(d, dict) and d.get("wall_s") is not None:
            tot += float(d["wall_s"]); found = True
    if found:
        return round(tot, 1)
    # Older harness generations recorded it elsewhere and under a different name. The spellings below
    # were read off the artifacts, not guessed: `wall_time_seconds` is what `cost_time_toolcalls.yaml`
    # writes, and searching only for `wall_s` reported 110 runs as having no timing when 78 of them did.
    for name in ("cost_time_toolcalls.yaml", "qa_loop_summary.yaml", "run_manifest.yaml",
                 "timing_detailed.json", "run_record.json"):
        d = _yaml(run / name) if name.endswith(".yaml") else _json(run / name)
        if isinstance(d, dict):
            for k in ("wall_time_seconds", "wall_s", "wall_seconds", "elapsed_s",
                      "duration_s", "total_wall_s", "total_seconds"):
                if d.get(k) is not None:
                    try:
                        return float(d[k])
                    except (TypeError, ValueError):
                        continue
    return None


def distill(run: Path) -> dict:
    st = _yaml(run / "qa_loop_state.yaml")
    rounds = st.get("rounds") or []
    env = _yaml(run / "environment.yaml")

    tok = sum(r.get("tokens_total") or 0 for r in rounds) or None
    calls = sum(r.get("tool_calls") or 0 for r in rounds) or None
    tiers = [{"round": r.get("round"), "n_passed": r.get("n_passed"),
              "n_capsules": r.get("n_capsules"), "all_pass": r.get("all_pass"),
              "tokens_total": r.get("tokens_total"), "tool_calls": r.get("tool_calls"),
              "conformant": ((r.get("conformance") or {}).get("conformant")
                             if isinstance(r.get("conformance"), dict) else None)}
             for r in rounds] or None
    aet = _json(run / "run_record.json") or _yaml(run / "run_manifest.yaml") or None
    pt = per_tier_detail(run)
    wall = wall_seconds(run)
    sub = sorted(str(p.relative_to(run)) for p in run.glob("_qa_work/cand_*/submission")) or None

    entry = {
        "run_id": st.get("run_id") or run.name,
        "path": str(run),
        "arm": st.get("arm") or env.get("arm"),
        "model": st.get("model"), "effort": st.get("effort"),
        "target": env.get("target"), "suite": run.parent.name,
        "n_rounds": len(rounds) or None,
        "final_score": (rounds[-1].get("n_passed") if rounds else None),
        "n_capsules": (rounds[-1].get("n_capsules") if rounds else None),
        "tiers": tiers, "per_tier": pt, "tokens": tok, "tool_calls": calls,
        "wall_s": wall, "config": env or None, "aet": aet,
        "transcript": len(list(run.glob("rounds/*"))) or None,
        "submission": sub,
    }
    entry["missing"] = [k for k in KEEP if entry.get(k) in (None, [], {})]
    return entry


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--roots", nargs="*", default=None,
                    help="checkout roots to scan (default: this repo + its sibling worktrees)")
    ap.add_argument("--out", default=None, help="output dir (default: out/artifacts/runs-index/<TS>_<sha>)")
    a = ap.parse_args(argv)

    root = Path(repo_root())
    roots = [Path(r) for r in a.roots] if a.roots else [
        root, *sorted((root / ".claude/worktrees").glob("*"))]

    runs: list[Path] = []
    for r in roots:
        d = r / "out/runs"
        if d.is_dir():
            runs += [p.parent for p in d.rglob("qa_loop_state.yaml")]
    runs = sorted(set(runs))

    entries = [distill(p) for p in runs]

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        sha = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"], cwd=root,
                             capture_output=True, text=True, timeout=30).stdout.strip() or "nosha"
    except Exception:                                    # noqa: BLE001
        sha = "nosha"
    out = Path(a.out) if a.out else artifacts_dir() / "runs-index" / f"runs_index_{ts}_{sha}"
    out.mkdir(parents=True, exist_ok=True)

    gaps: dict[str, int] = {}
    for e in entries:
        for m in e["missing"]:
            gaps[m] = gaps.get(m, 0) + 1

    doc = {"schema": "runs-index/v0", "generated_at": ts, "git_sha": sha,
           "roots": [str(r) for r in roots], "n_runs": len(entries),
           "coverage": {k: len(entries) - gaps.get(k, 0) for k in KEEP},
           "gaps": gaps, "runs": entries}
    (out / "runs_index.json").write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")

    cols = ["run_id", "arm", "target", "suite", "model", "effort", "n_rounds",
            "final_score", "n_capsules", "tokens", "tool_calls", "wall_s", "missing"]
    with (out / "runs_index.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for e in entries:
            w.writerow({**{c: e.get(c) for c in cols}, "missing": ",".join(e["missing"])})

    print(f"runs indexed: {len(entries)}  ->  {out}")
    for k in KEEP:
        have = len(entries) - gaps.get(k, 0)
        print(f"  {k:12s} {have:4d}/{len(entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
