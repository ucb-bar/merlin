#!/usr/bin/env python3
"""Drive `tools/run_heterogeneous_e2e.py` across the (model, granularity,
machines) matrix and aggregate results into one summary table.

Default matrix (mirrors PR-6 of the rosy-sundae plan):
  models       = dronet, dronet_coarse, mobilenet_v2
  granularity  = dispatch, layer
  machines     = baseline (CPU_P + CPU_E), 3-target (+ QNN_GPU),
                 4-target (+ QNN_HTA, only for int8 models)

Each (model, gran, machines) cell runs end-to-end and contributes one row
to `eval/qrb5165/heterogeneous/summary.{csv,md}`. The baseline trace for
each (model, gran) feeds into the heterogeneous correctness check via
--baseline-trace.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import shlex
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
E2E = REPO_ROOT / "tools" / "run" / "het_e2e.py"

DEFAULT_MODELS = ["dronet", "dronet_coarse", "mobilenet_v2"]
DEFAULT_GRANULARITIES = ["dispatch", "layer"]

INT8_MODELS = {"mobilenet_v2"}  # Only int8 models route to HTA (fp16 NPU).


@dataclasses.dataclass
class Cell:
    model: str
    granularity: str
    machines: tuple[str, ...]
    label: str  # "baseline", "3target", "4target"


def build_matrix(
    models: list[str], granularities: list[str], include_qnn_gpu: bool, include_qnn_hta: bool
) -> list[Cell]:
    cells = []
    for model in models:
        for gran in granularities:
            cells.append(Cell(model, gran, ("CPU_P", "CPU_E"), "baseline"))
            if include_qnn_gpu:
                cells.append(Cell(model, gran, ("CPU_P", "CPU_E", "QNN_GPU"), "3target"))
            if include_qnn_hta and model in INT8_MODELS:
                cells.append(Cell(model, gran, ("CPU_P", "CPU_E", "QNN_GPU", "QNN_HTA"), "4target"))
    return cells


def run_cell(cell: Cell, baseline_trace: pathlib.Path | None, extra_args: list[str]) -> dict:
    """Invoke run_heterogeneous_e2e.py for one cell. Returns parsed
    summary.json."""
    out_dir = REPO_ROOT / "eval" / "qrb5165" / "heterogeneous" / f"{cell.model}_{cell.granularity}_{cell.label}"
    cmd = [
        "uv",
        "run",
        "python",
        str(E2E),
        "--model",
        cell.model,
        "--granularity",
        cell.granularity,
        "--machines",
        *cell.machines,
        "--output-dir",
        str(out_dir),
    ]
    if baseline_trace and baseline_trace.exists():
        cmd.extend(["--baseline-trace", str(baseline_trace)])
    cmd.extend(extra_args)
    print(
        f"\n[matrix] cell={cell.label} model={cell.model} " f"granularity={cell.granularity} machines={cell.machines}"
    )
    print(f"[matrix] cmd: {' '.join(shlex.quote(a) for a in cmd)}")
    res = subprocess.run(cmd, check=False)
    summary_path = out_dir / "summary.json"
    if res.returncode != 0:
        return {
            "model": cell.model,
            "granularity": cell.granularity,
            "machines": list(cell.machines),
            "label": cell.label,
            "error": f"e2e returned {res.returncode}",
        }
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        s["label"] = cell.label
        return s
    return {
        "model": cell.model,
        "granularity": cell.granularity,
        "machines": list(cell.machines),
        "label": cell.label,
        "error": "missing summary.json",
    }


def render_summary(rows: list[dict], out_root: pathlib.Path) -> None:
    csv_path = out_root / "summary.csv"
    md_path = out_root / "summary.md"
    cols = [
        "model",
        "granularity",
        "label",
        "machines",
        "planned_ms",
        "observed_ms",
        "gap_pct_median",
        "correctness.sets_equal",
    ]

    def field(row: dict, key: str):
        if "." in key:
            head, tail = key.split(".", 1)
            return field(row.get(head, {}) if isinstance(row.get(head), dict) else {}, tail)
        v = row.get(key, "")
        if isinstance(v, list):
            return "+".join(v)
        return v

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(field(r, c)) for c in cols) + "\n")

    md_lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        md_lines.append("| " + " | ".join(str(field(r, c)) for c in cols) + " |")
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"[matrix] wrote {csv_path}")
    print(f"[matrix] wrote {md_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument(
        "--granularities", nargs="+", default=DEFAULT_GRANULARITIES, choices=["dispatch", "layer", "megakernel", "tile"]
    )
    p.add_argument("--no-qnn-gpu", action="store_true", help="Skip the 3-target (CPU_P+CPU_E+QNN_GPU) cells.")
    p.add_argument("--no-qnn-hta", action="store_true", help="Skip the 4-target cells.")
    p.add_argument("--output-root", type=pathlib.Path, default=REPO_ROOT / "eval" / "qrb5165" / "heterogeneous")
    # Pass-through args forwarded verbatim to run_heterogeneous_e2e.py.
    p.add_argument("--ssh-host", default="qdev")
    p.add_argument("--solver", choices=["greedy", "mosek"], default="mosek")
    p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--skip-compile-qnn", action="store_true")
    p.add_argument("--skip-profile", action="store_true")
    p.add_argument("--skip-board-run", action="store_true")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    a.output_root.mkdir(parents=True, exist_ok=True)
    cells = build_matrix(
        a.models,
        a.granularities,
        include_qnn_gpu=not a.no_qnn_gpu,
        include_qnn_hta=not a.no_qnn_hta,
    )
    print(f"[matrix] {len(cells)} cells: ")
    for c in cells:
        print(f"  - {c.label} {c.model}/{c.granularity} {c.machines}")

    extra: list[str] = [
        f"--ssh-host={a.ssh_host}",
        f"--solver={a.solver}",
        f"--repetitions={a.repetitions}",
    ]
    if a.skip_compile_qnn:
        extra.append("--skip-compile-qnn")
    if a.skip_profile:
        extra.append("--skip-profile")
    if a.skip_board_run:
        extra.append("--skip-board-run")

    baseline_trace_by_key: dict[tuple[str, str], pathlib.Path] = {}
    rows: list[dict] = []
    # Run baselines first so heterogeneous cells can reference them.
    for cell in [c for c in cells if c.label == "baseline"]:
        s = run_cell(cell, baseline_trace=None, extra_args=extra)
        rows.append(s)
        if "trace" in s and pathlib.Path(s["trace"]).exists():
            baseline_trace_by_key[(cell.model, cell.granularity)] = pathlib.Path(s["trace"])
    for cell in [c for c in cells if c.label != "baseline"]:
        baseline = baseline_trace_by_key.get((cell.model, cell.granularity))
        rows.append(run_cell(cell, baseline_trace=baseline, extra_args=extra))

    render_summary(rows, a.output_root)
    return 0 if all("error" not in r for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
