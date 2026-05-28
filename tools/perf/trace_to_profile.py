#!/usr/bin/env python3
"""Fold a scheduler-runner trace CSV's observed run_us back into the
profiled_manifest.json, closing the iterative profile/schedule/run loop.

The board_roundtrip.py pre-pass measures per-(dispatch, machine) cost in
isolation via iree-benchmark-module. Those numbers are accurate per-dispatch
but ~3x optimistic vs in-scheduler execution because the bench runs each
VMFB hot-cache and the scheduler chains 20+ different VMFBs back-to-back
(cache eviction + scheduler queue/mutex overhead between dispatches).

This tool reads a trace.csv (output of merlin-dispatch-scheduler) and
overwrites the profiles[<machine>].mean_time_us in the manifest with the
median run_us observed for that dispatch on that target. A second
schedule built from the updated manifest gives planned ≈ observed within
scheduler-overhead variance, which is the regime XPU-RT's cost-model
assumptions actually hold in.

Usage:
    tools/trace_to_profile.py \\
        --trace-csv eval/.../trace.csv \\
        --manifest eval/.../breakdowns/profiled_manifest.json \\
        [--write]                # in-place update (omit for dry-run)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--trace-csv", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path, help="profiled_manifest.json to update in place.")
    p.add_argument(
        "--write",
        action="store_true",
        help="Persist the update. Without this flag, the tool " "prints the proposed delta and exits.",
    )
    p.add_argument(
        "--smoothing",
        type=float,
        default=1.0,
        help="EMA factor in [0, 1] applied to the new cost: "
        "  new_cost = alpha * trace + (1 - alpha) * old_cost. "
        "1.0 = full overwrite (default, prior behaviour). "
        "0.5 = average old/new. Use < 1 to dampen the "
        "greedy scheduler's iteration-to-iteration "
        "oscillation when concurrency-dependent costs feed "
        "back into the cost matrix.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.trace_csv)
    if "run_us" not in df.columns or "dispatch_key" not in df.columns:
        print("trace CSV missing required columns", file=sys.stderr)
        return 2

    # The trace's dispatch_key may include a replica prefix (e.g.
    # mlp3_dispatch_15) when produced from a multi-job run. Strip the
    # leading <replica>_ token so it matches the manifest's keys.
    def strip_replica(k: str) -> str:
        idx = k.find("_dispatch_")
        if idx < 0:
            return k
        return k[idx + 1 :]

    df["base_key"] = df["dispatch_key"].astype(str).apply(strip_replica)

    # Median observed run_us per (base_key, target).
    medians = df.groupby(["base_key", "target"])["run_us"].median().reset_index()

    manifest = json.loads(args.manifest.read_text())
    dispatches = manifest.get("dispatches", {})
    if not dispatches:
        print("manifest has no dispatches block", file=sys.stderr)
        return 2

    print(f"{'dispatch':<14}{'target':<8}{'old_us':>10}{'new_us':>10}" f"{'delta_pct':>12}")
    changes = 0
    for row in medians.itertuples(index=False):
        entry = dispatches.get(row.base_key)
        if entry is None:
            print(f"  (skip {row.base_key} — not in manifest)")
            continue
        profs = entry.setdefault("profiles", {})
        old = profs.get(row.target, {}).get("mean_time_us", 0.0)
        observed = float(row.run_us)
        # Apply EMA smoothing if requested.
        if old and 0.0 < args.smoothing < 1.0:
            new = args.smoothing * observed + (1 - args.smoothing) * old
        else:
            new = observed
        if old:
            delta_pct = 100.0 * (new - old) / old
        else:
            delta_pct = float("inf")
        print(f"{row.base_key:<14}{row.target:<8}{old:>10.1f}{new:>10.1f}" f"{delta_pct:>12.1f}")
        prof = profs.setdefault(row.target, {})
        prof["mean_time_us"] = new
        prof["source"] = f"scheduler-runner trace median " f"(EMA alpha={args.smoothing:.2f})"
        changes += 1

    if not args.write:
        print(f"\n[dry-run] {changes} profiles would be updated; pass --write " "to persist.")
        return 0

    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nupdated {changes} profiles -> {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
