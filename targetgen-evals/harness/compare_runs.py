"""Aggregate all runs for a target into reports."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import yaml

from harness.collect_metrics import _ALL_COLUMNS, _STRING_COLUMNS


def _load_run(run_dir: Path) -> dict | None:
    summary_path = run_dir / "metrics" / "summary_metrics.json"
    manifest_path = run_dir / "run_manifest.yaml"
    if not summary_path.exists() or not manifest_path.exists():
        return None
    with open(summary_path) as f:
        return json.load(f)


def _mean_std(values: list) -> tuple:
    nums = [v for v in values if v is not None and isinstance(v, (int, float))]
    if not nums:
        return None, None
    mean = sum(nums) / len(nums)
    if len(nums) < 2:
        return round(mean, 4), None
    variance = sum((x - mean) ** 2 for x in nums) / (len(nums) - 1)
    return round(mean, 4), round(math.sqrt(variance), 4)


def _fmt(mean, std) -> str:
    if mean is None:
        return "NA"
    if std is None:
        return str(mean)
    return f"{mean} ± {std}"


def compare(root: Path, target: str, output_dir: Path) -> int:
    runs_dir = root / "runs" / target
    if not runs_dir.exists():
        print(f"ERROR: no runs directory for target {target}: {runs_dir}", file=sys.stderr)
        return 1

    run_dirs = sorted(d for d in runs_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
    rows = []
    for rd in run_dirs:
        data = _load_run(rd)
        if data is not None:
            rows.append(data)

    if not rows:
        print(f"No validated runs found for target {target}", file=sys.stderr)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.csv").write_text(",".join(_ALL_COLUMNS) + "\n")
        (output_dir / "ablation_table.md").write_text(
            f"# Ablation Table: {target}\n\n*No validated runs.*\n"
        )
        (output_dir / "summary.md").write_text(
            f"# {target} — no validated runs yet\n"
        )
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    # metrics.csv — one row per run, full column set, NA for missing
    csv_path = output_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_ALL_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {col: row.get(col, "NA") for col in _ALL_COLUMNS}
            # Normalise None → NA for CSV
            for k, v in out.items():
                if v is None:
                    out[k] = "NA"
            writer.writerow(out)

    # ablation_table.md — one row per method, mean ± std, smoke runs excluded
    non_smoke = [r for r in rows if not r.get("is_smoke_test", True)]
    methods = sorted({r["method"] for r in non_smoke})

    numeric_cols = [
        c for c in _ALL_COLUMNS
        if c not in _STRING_COLUMNS and c not in ("is_smoke_test", "seed")
    ]

    table_lines = [
        f"# Ablation Table: {target}\n",
        "| method | seeds | " + " | ".join(numeric_cols) + " |",
        "|---|---|" + "|".join("---" for _ in numeric_cols) + "|",
    ]
    for method in methods:
        method_rows = [r for r in non_smoke if r["method"] == method]
        seeds = sorted({r["seed"] for r in method_rows})
        cells = []
        for col in numeric_cols:
            vals = [r.get(col) for r in method_rows]
            mean, std = _mean_std(vals)
            cells.append(_fmt(mean, std))
        table_lines.append(
            f"| {method} | {','.join(str(s) for s in seeds)} | "
            + " | ".join(cells) + " |"
        )

    if not methods:
        table_lines.append("| *(no real baseline runs yet)* | — |" + "|".join("NA" for _ in numeric_cols) + "|")

    (output_dir / "ablation_table.md").write_text("\n".join(table_lines) + "\n")

    # summary.md
    smoke_count = sum(1 for r in rows if r.get("is_smoke_test", True))
    real_count = len(rows) - smoke_count
    (output_dir / "summary.md").write_text(
        f"# {target} — comparison summary\n\n"
        f"- Total runs: {len(rows)} ({smoke_count} smoke, {real_count} real)\n"
        f"- Methods with real runs: {len(methods)}\n\n"
        f"See `ablation_table.md` for per-method aggregated metrics.\n"
        f"See `metrics.csv` for per-run raw data.\n"
    )

    print(f"reports written to {output_dir}")
    print(f"  {len(rows)} runs ({smoke_count} smoke, {real_count} real baseline)")
    print(f"  {csv_path}")
    print(f"  {output_dir / 'ablation_table.md'}")
    return 0
