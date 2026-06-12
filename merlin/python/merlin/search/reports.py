"""Emit scoreboards, regime maps, Pareto frontiers, and decision reports from search results.

Turns grid/evolutionary/MAP-Elites output into durable artifacts: ``regime_map.csv`` (one row
per occupied behavior cell), ``pareto_frontier.csv`` (non-dominated strategies over
exploitability vs complexity), and ``decision_report.md``.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

from merlin.dse.pareto import compute_pareto
from merlin.search.archive import archive_rows


def _csv(rows: list[dict], columns: list[str]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k) for k in columns})
    return buf.getvalue()


def regime_map_csv(archive: dict) -> str:
    cols = ["memory_abstraction", "control_abstraction", "granularity", "workload_regime",
            "strategy", "features", "correctness", "coverage", "exploitability", "speedup",
            "total"]
    return _csv(archive_rows(archive), cols)


def pareto_frontier_csv(rows: list[dict]) -> str:
    """Pareto frontier over (exploitability max, complexity min) from grid/scoreboard rows.

    ``rows`` come from ``grid.grid_search_strategies`` (each has a ``score`` and ``strategy``).
    """
    points = []
    for r in rows:
        s = r["score"]
        points.append({
            "strategy": r.get("strategy"),
            "features": r.get("features", ""),
            "exploitability": s.exploitability,
            "speedup": s.speedup,
            "complexity_penalty": s.complexity_penalty,
            "correctness": s.correctness,
        })
    # Only correct candidates are eligible for the frontier.
    legal = [p for p in points if p["correctness"] >= 1.0] or points
    front = compute_pareto(legal, ["exploitability", "complexity_penalty"], ["max", "min"])
    cols = ["strategy", "features", "exploitability", "speedup", "complexity_penalty"]
    return _csv(sorted(front, key=lambda p: -p["exploitability"]), cols)


def decision_report_md(title: str, archive: dict, grid_rows: list[dict] | None = None) -> str:
    rows = archive_rows(archive)
    rows.sort(key=lambda r: -r["total"])
    lines = [f"# Search decision report — {title}", "",
             f"Occupied behavior cells: **{len(rows)}** (portfolio of families).", "",
             "## Best per behavior cell", "",
             "| memory | control | granularity | strategy | features | exploit | total |",
             "| --- | --- | --- | --- | --- | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['memory_abstraction']} | {r['control_abstraction']} | "
                     f"{r['granularity']} | {r['strategy']} | {r['features']} | "
                     f"{r['exploitability']} | {r['total']} |")
    return "\n".join(lines) + "\n"


def build_report(archive: dict, grid_rows: list[dict] | None = None, title: str = "search",
                 out_dir=None) -> dict:
    """Assemble the search artifacts; write them under ``out_dir`` if given."""
    artifacts = {
        "regime_map.csv": regime_map_csv(archive),
        "decision_report.md": decision_report_md(title, archive, grid_rows),
    }
    if grid_rows is not None:
        artifacts["pareto_frontier.csv"] = pareto_frontier_csv(grid_rows)
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, text in artifacts.items():
            (out / name).write_text(text, encoding="utf-8")
    return artifacts
