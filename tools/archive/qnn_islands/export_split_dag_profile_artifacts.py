#!/usr/bin/env python3
"""Build an explicit split-DAG scheduling artifact for HTA conv islands."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections.abc import Iterable

INF_US = 1.0e9
MACHINES = ["CPU", "GPU", "HTA"]


def _load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def _dispatch_name(name: str) -> str | None:
    matches = re.findall(r"dispatch_(\d+)", name)
    if not matches:
        return None
    return f"dispatch_{int(matches[0])}"


def _read_hta_islands(profile_path: pathlib.Path) -> dict[str, dict]:
    data = _load(profile_path)
    out: dict[str, dict] = {}
    for key, row in data.get("dispatches", {}).items():
        dispatch = _dispatch_name(key)
        if dispatch is None:
            continue
        cell = row.get("qnn_hta", {})
        prof = cell.get("profile") if isinstance(cell, dict) else None
        if not prof or prof.get("mean_us", 0) <= 0:
            continue
        out[dispatch] = {
            "mean_us": float(prof["mean_us"]),
            "setup_us": float(prof.get("setup_us", 0.0)),
            "source_key": key,
        }
    return out


def _split_boundary(cpu_us: float, hta_us: float) -> tuple[float, float]:
    """Conservative placeholder until CPU boundary kernels are measured.

    We preserve the original CPU dispatch total as the non-HTA residual when
    HTA is faster. This keeps the split candidate from claiming a speedup
    that was not measured while still making data dependencies explicit.
    """
    residual = max(cpu_us - hta_us, 0.0)
    return residual * 0.5, residual * 0.5


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-workload", type=pathlib.Path, required=True)
    parser.add_argument("--base-processing", type=pathlib.Path, required=True)
    parser.add_argument("--base-transfer", type=pathlib.Path, required=True)
    parser.add_argument("--hta-profile", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)

    base_workload = _load(args.base_workload)
    base_processing = _load(args.base_processing)
    base_transfer = _load(args.base_transfer)
    hta = _read_hta_islands(args.hta_profile)
    split_dispatches = sorted(d for d in hta if d in base_processing)

    out_workload = {"dispatches": {}, "machines": MACHINES, "split_metadata": {}}
    out_processing: dict[str, list[float]] = {}
    replacement: dict[str, str] = {}

    next_id = 0
    for name, row in sorted(
        base_workload["dispatches"].items(),
        key=lambda kv: int(kv[1].get("id", 0)),
    ):
        deps = [replacement.get(dep, dep) for dep in row.get("dependencies", [])]
        if name not in split_dispatches:
            new_row = dict(row)
            new_row["id"] = next_id
            new_row["dependencies"] = deps
            out_workload["dispatches"][name] = new_row
            out_processing[name] = base_processing[name]
            replacement[name] = name
            next_id += 1
            continue

        cpu_us = float(base_processing[name][0])
        hta_us = float(hta[name]["mean_us"])
        pre_us, post_us = _split_boundary(cpu_us, hta_us)
        pre = f"{name}_pre_cpu"
        island = f"{name}_hta_conv"
        post = f"{name}_post_cpu"
        out_workload["dispatches"][pre] = {
            "id": next_id,
            "dependencies": deps,
            "op_summary": f"{row.get('op_summary', '')}::pre_boundary",
            "infeasible_machines": ["GPU", "HTA"],
            "split_parent": name,
        }
        out_processing[pre] = [pre_us, INF_US, INF_US]
        next_id += 1
        out_workload["dispatches"][island] = {
            "id": next_id,
            "dependencies": [pre],
            "op_summary": f"{row.get('op_summary', '')}::hta_conv_island",
            "infeasible_machines": ["CPU", "GPU"],
            "split_parent": name,
        }
        out_processing[island] = [INF_US, INF_US, hta_us]
        next_id += 1
        out_workload["dispatches"][post] = {
            "id": next_id,
            "dependencies": [island],
            "op_summary": f"{row.get('op_summary', '')}::post_boundary",
            "infeasible_machines": ["GPU", "HTA"],
            "split_parent": name,
        }
        out_processing[post] = [post_us, INF_US, INF_US]
        out_workload["split_metadata"][name] = {
            "pre": pre,
            "island": island,
            "post": post,
            "cpu_whole_us": cpu_us,
            "hta_island_us": hta_us,
            "pre_boundary_us": pre_us,
            "post_boundary_us": post_us,
            "boundary_model": "residual_half_from_cpu_whole_minus_hta_island",
            "island_profile": hta[name],
        }
        replacement[name] = post
        next_id += 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "workload.json").write_text(json.dumps(out_workload, indent=2))
    (args.out_dir / "processing_times.json").write_text(json.dumps(out_processing, indent=2))
    (args.out_dir / "transfer_times.json").write_text(json.dumps(base_transfer, indent=2))
    summary = {
        "base_dispatches": len(base_workload["dispatches"]),
        "split_dispatches": split_dispatches,
        "expanded_nodes": len(out_workload["dispatches"]),
        "boundary_model": "residual_half_from_cpu_whole_minus_hta_island",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
