#!/usr/bin/env python3
"""Build a schedule that forces every dispatch onto a single target.

Used as part of the iterative profile/schedule/run loop:
  1. Start with bench-derived costs.
  2. Force-all-on-CPU_P → run → fold trace back via trace_to_profile.py.
  3. Force-all-on-CPU_E → run → fold trace back.
  4. Now both columns of the cost matrix reflect in-scheduler conditions.
  5. Re-run the real scheduler (merlin_adapter.py schedule).

Usage:
    tools/force_target_schedule.py \\
        --base eval/.../breakdowns/manifest.json \\
        --target CPU_P --vmfb-prefix dispatch \\
        --out eval/.../forced_cpu_p.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument(
        "--base",
        required=True,
        type=Path,
        help="breakdown manifest.json (or any json with a "
        "dispatches block keyed by dispatch_<N> with "
        "id+dependencies+module_name+op_summary).",
    )
    p.add_argument("--target", required=True, help="Force this hardware_target on every dispatch.")
    p.add_argument("--vmfb-prefix", default="", help="Optional path prefix for vmfb_path values.")
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base = json.loads(args.base.read_text())
    src = base.get("dispatches", {})
    out_dispatches = {}
    for k, e in src.items():
        out_dispatches[k] = {
            "id": e["id"],
            "subid": e.get("subid"),
            "ordinal": e.get("ordinal", 1),
            "total": e.get("total", 1),
            "hardware_target": args.target,
            "start_time": 0.0,
            "duration": 0.0,
            "dependencies": e.get("dependencies", []),
            "op_summary": e.get("op_summary", ""),
            "job_name": "dronet",
            "module_name": e.get("module_name", e.get("op_summary", k)),
            "vmfb_path": (f"{args.vmfb_prefix.rstrip('/')}/{k}.vmfb" if args.vmfb_prefix else f"{k}.vmfb"),
        }
    payload = {
        "schema_version": 1,
        "machines": [args.target],
        "device_map": {args.target: "@device_a"},
        "source": "tools/force_target_schedule.py",
        "dispatches": out_dispatches,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"forced {args.target}: {len(out_dispatches)} dispatches -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
