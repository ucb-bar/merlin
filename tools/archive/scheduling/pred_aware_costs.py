"""Build a predecessor-aware cost map from a scheduler-runner trace.

Reads a trace CSV (columns: dispatch_key, target, run_us, ...) and the
matching schedule.json (which provides per-dispatch dependency lists +
machine assignments), and emits a per-(op, current target, predecessor
target) cost table that the MOSEK MILP can ingest as
processing_times_by_pred. Each (i, k_pred, k_curr) cell defaults to the
2D `mean_time_us` from the original profiled_manifest.json when no
matching observation exists for that combination.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", type=pathlib.Path, required=True, help="trace.csv emitted by merlin-dispatch-scheduler")
    p.add_argument("--schedule", type=pathlib.Path, required=True, help="combined_schedule.json (drives dependencies)")
    p.add_argument(
        "--profiled-manifest", type=pathlib.Path, required=True, help="profiled_manifest.json with 2D fallback costs"
    )
    p.add_argument(
        "--out", type=pathlib.Path, required=True, help="output profiled_manifest.json with cost_by_pred map"
    )
    p.add_argument(
        "--machines", nargs="+", default=["CPU_P", "CPU_E"], help="machine ordering for the MOSEK cost tensor"
    )
    args = p.parse_args()

    trace = pd.read_csv(args.trace)
    sched = json.loads(args.schedule.read_text())
    prof = json.loads(args.profiled_manifest.read_text())
    machines = args.machines

    # Build a dispatch_id -> (target, run_us) lookup directly from the trace.
    # Trace keys are always `<job_prefix>dispatch_<id>` so we extract the int
    # ID, which is the universal anchor across DISPATCH / LAYER / MEGAKERNEL
    # granularities (the layer/MK manifests carry parent_dispatch_ids).
    import re

    did_target: dict[int, str] = {}
    did_run: dict[int, float] = {}
    did_pred_target: dict[int, str] = {}  # predecessor of dispatch_id at runtime
    # We also need predecessor data: walk trace rows in start_us order and
    # for each dispatch find the most-recent finished dispatch on the same
    # logical chain (we approximate via the highest end_us < start_us OR
    # via the schedule's dependencies list when keys match).
    for _, row in trace.iterrows():
        m = re.search(r"dispatch_(\d+)", str(row.dispatch_key))
        if not m:
            continue
        did = int(m.group(1))
        did_target[did] = row.target
        did_run[did] = float(row.run_us)
    # Predecessor target via the schedule's dependency graph: for each chunk
    # in the schedule, its first dependency's last constituent dispatch's
    # target is the upstream cluster.
    sched_disps = sched["dispatches"]
    for chunk_name, entry in sched_disps.items():
        deps = entry.get("dependencies", []) or []
        if not deps:
            continue
        pred_chunk = sched_disps.get(deps[0])
        if not pred_chunk:
            continue
        pred_id = pred_chunk.get("id")
        if pred_id is None or pred_id not in did_target:
            continue
        chunk_id = entry.get("id")
        if chunk_id is None:
            continue
        did_pred_target[chunk_id] = did_target[pred_id]

    # Aggregate by (chunk_id, current_target, predecessor_target). For LAYER
    # / MEGAKERNEL the chunk_id == manifest id; we'll pick up parent_dispatch_
    # ids in the patch loop below.
    sums: dict[tuple[int, str, str], list[float]] = {}
    for chunk_name, entry in sched_disps.items():
        chunk_id = entry.get("id")
        if chunk_id is None:
            continue
        cur = entry.get("hardware_target")
        if cur is None:
            continue
        # For the chunk's run cost, sum its constituent dispatches' run_us
        # observed on the trace. (For DISPATCH granularity that's just one.)
        parent_ids = entry.get("parent_dispatch_ids", []) or [chunk_id]
        if any(pid not in did_run for pid in parent_ids):
            continue
        total_run = sum(did_run[pid] for pid in parent_ids)
        if total_run == 0:
            continue
        deps = entry.get("dependencies", []) or []
        if not deps:
            cell = (chunk_id, cur, "_cold_")
        else:
            pred_chunk = sched_disps.get(deps[0], {})
            pred_id = pred_chunk.get("id")
            pred_tgt = did_target.get(pred_id) if pred_id is not None else None
            if pred_tgt is None:
                continue
            cell = (chunk_id, cur, pred_tgt)
        sums.setdefault(cell, []).append(total_run)

    # Convert to mean per cell.
    cells: dict[tuple[int, str, str], float] = {k: sum(v) / len(v) for k, v in sums.items()}

    # Patch profiled_manifest entries: add per-(machine, predecessor_machine)
    # cost matrix as a JSON dict we can round-trip back into Operation. For
    # LAYER / MEGAKERNEL chunks the dispatch-trace data needs to roll up to
    # the chunk's constituent dispatches: the chunk's effective duration on
    # `cur` with predecessor on `pred_tgt` is the sum of its constituent
    # dispatch run_us values when run on `cur` (predecessor on prior chunk's
    # last dispatch's target).
    # cells is keyed by chunk_id directly (pre-aggregated above), so the
    # patch loop is a simple lookup per (chunk_id, k_curr, k_pred).
    patched = 0
    for name, e in prof["dispatches"].items():
        cid = e["id"]
        cost_by_pred: dict[str, float] = {}
        for k_curr in machines:
            for k_pred in machines + ["_cold_"]:
                v = cells.get((cid, k_curr, k_pred))
                if v is not None:
                    cost_by_pred[f"{k_pred}->{k_curr}"] = float(v)
        if cost_by_pred:
            e["cost_by_pred"] = cost_by_pred
            patched += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(prof, indent=2))
    print(f"patched {patched} of {len(prof['dispatches'])} entries with cost_by_pred")
    return 0


if __name__ == "__main__":
    sys.exit(main())
