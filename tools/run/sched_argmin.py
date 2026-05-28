#!/usr/bin/env python3
"""Build an argmin heterogeneous schedule from CPU/GPU/HTA per-dispatch measurements.

The existing plot tool prefers HTA > GPU > CPU when any data is available,
which is WRONG when the accelerator is slower than CPU (e.g. yolov8 has
many small float dispatches where QNN_GPU's per-invocation setup
dominates). This tool picks the actual fastest target per dispatch:

  chosen = argmin(cpu_ms, gpu_ms, hta_ms)

Inputs:
  --cpu-trace CSV from dispatch_flow_runner (op, invoke_us, ...)
  --gpu-csv   CSV from tools/sweep on board (name, mean_ms, ...)
  --hta-csv   CSV from tools/sweep on board (island, source, mean_ms, ...)
  --canonical-from-cpu  canonical-dispatch is derived from CPU-trace op name
                        (default: strip "_call_NNN" suffix).
  --out-dir   writes:
              schedule.json     — per-dispatch target + ms
              schedule.csv      — tabular summary
              speedup_table.md  — per-target breakdown

Notes:
- HTA timings are matched by `source_dispatch` (canonical), then applied
  to every call site of that canonical dispatch.
- A dispatch missing measurements for a target is treated as +inf for
  that target.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re

_CALL_RE = re.compile(r"^(dispatch_\d+)_call_\d+$")


def _canonical(op_name: str) -> str:
    m = _CALL_RE.match(op_name)
    return m.group(1) if m else op_name


def _strip_async_prefix(name: str) -> str:
    return name.replace("main_graph$async_", "")


def load_cpu(path: pathlib.Path) -> list[dict]:
    rows = list(csv.DictReader(path.open()))
    out = []
    for r in rows:
        out.append(
            {
                "op": r["op"],
                "canonical": _canonical(r["op"]),
                "cpu_ms": float(r["invoke_us"]) / 1000.0,
            }
        )
    return out


def load_gpu(path: pathlib.Path) -> dict[str, float]:
    if not path or not path.is_file():
        return {}
    by_canonical = {}
    for r in csv.DictReader(path.open()):
        if r.get("status") and r["status"] != "ok":
            continue
        mean = r.get("mean_ms")
        if not mean:
            continue
        canonical = _strip_async_prefix(r["name"])
        by_canonical[canonical] = float(mean)
    return by_canonical


def load_hta(path: pathlib.Path) -> dict[str, float]:
    if not path or not path.is_file():
        return {}
    by_canonical = {}
    for r in csv.DictReader(path.open()):
        if r.get("status") and r["status"] != "ok":
            continue
        canonical = r.get("source") or r.get("source_dispatch")
        mean = r.get("mean_ms")
        if not canonical or not mean:
            continue
        by_canonical[canonical] = float(mean)
    return by_canonical


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cpu-trace", type=pathlib.Path, required=True)
    p.add_argument("--gpu-csv", type=pathlib.Path, default=None)
    p.add_argument("--hta-csv", type=pathlib.Path, default=None)
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = p.parse_args(argv)

    cpu_rows = load_cpu(args.cpu_trace)
    gpu_by_canonical = load_gpu(args.gpu_csv)
    hta_by_canonical = load_hta(args.hta_csv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    schedule = {
        "dispatches": {},
        "metadata": {
            "cpu_trace": str(args.cpu_trace),
            "gpu_csv": str(args.gpu_csv) if args.gpu_csv else None,
            "hta_csv": str(args.hta_csv) if args.hta_csv else None,
            "policy": "argmin(cpu_ms, gpu_ms, hta_ms)",
        },
    }
    table_rows = []
    target_counts = {"CPU": 0, "QNN_GPU": 0, "QNN_HTA": 0}
    target_totals = {"CPU": 0.0, "QNN_GPU": 0.0, "QNN_HTA": 0.0}
    total_cpu_only_ms = 0.0
    total_chosen_ms = 0.0

    for row in cpu_rows:
        cpu_ms = row["cpu_ms"]
        canonical = row["canonical"]
        gpu_ms = gpu_by_canonical.get(canonical)
        hta_ms = hta_by_canonical.get(canonical)
        candidates = [("CPU", cpu_ms)]
        if gpu_ms is not None:
            candidates.append(("QNN_GPU", gpu_ms))
        if hta_ms is not None:
            candidates.append(("QNN_HTA", hta_ms))
        chosen_target, chosen_ms = min(candidates, key=lambda c: c[1])
        target_counts[chosen_target] += 1
        target_totals[chosen_target] += chosen_ms
        total_cpu_only_ms += cpu_ms
        total_chosen_ms += chosen_ms
        schedule["dispatches"][row["op"]] = {
            "canonical": canonical,
            "cpu_ms": cpu_ms,
            "gpu_ms": gpu_ms,
            "hta_ms": hta_ms,
            "chosen_target": chosen_target,
            "chosen_ms": chosen_ms,
        }
        table_rows.append(
            {
                "op": row["op"],
                "canonical": canonical,
                "cpu_ms": f"{cpu_ms:.3f}",
                "gpu_ms": f"{gpu_ms:.3f}" if gpu_ms is not None else "",
                "hta_ms": f"{hta_ms:.3f}" if hta_ms is not None else "",
                "chosen_target": chosen_target,
                "chosen_ms": f"{chosen_ms:.3f}",
            }
        )

    (args.out_dir / "schedule.json").write_text(json.dumps(schedule, indent=2))
    with (args.out_dir / "schedule.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        w.writeheader()
        w.writerows(table_rows)

    speedup_md = [
        "# Argmin Heterogeneous Schedule",
        "",
        f"- CPU-only invoke total: **{total_cpu_only_ms:.1f} ms**",
        f"- Heterogeneous invoke total: **{total_chosen_ms:.1f} ms**",
        f"- Speedup: **{total_cpu_only_ms / max(total_chosen_ms, 1e-9):.2f}×**",
        "",
        "## Picks per target",
        "",
        "| Target | Count | Total ms |",
        "|---|---:|---:|",
    ]
    for t in ("CPU", "QNN_GPU", "QNN_HTA"):
        speedup_md.append(f"| {t} | {target_counts[t]} | {target_totals[t]:.1f} |")
    speedup_md.append("")
    speedup_md.append(f"GPU dispatches with data: {len(gpu_by_canonical)}")
    speedup_md.append(f"HTA dispatches with data: {len(hta_by_canonical)}")
    (args.out_dir / "speedup_table.md").write_text("\n".join(speedup_md) + "\n")

    print(
        json.dumps(
            {
                "total_cpu_only_ms": round(total_cpu_only_ms, 2),
                "total_chosen_ms": round(total_chosen_ms, 2),
                "speedup": round(total_cpu_only_ms / max(total_chosen_ms, 1e-9), 3),
                "target_counts": target_counts,
                "out_dir": str(args.out_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
