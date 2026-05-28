#!/usr/bin/env python3
"""Compose a multi-job schedule from a single-model breakdown manifest.

Takes a base schedule (e.g. dronet's dispatch_schedule.json) and replicates
it N times under distinct job_names ("dronet0", "mlp0", "mlp1", ...) with
optional time offsets so the workloads run concurrently. Optionally swaps
in LLM-generated / custom-kernel VMFBs for specific dispatches to exercise
the kernel-embedding path.

Mirrors the structure of /scratch2/agustin/merlin/tmp/example_schedule.json
so the existing plot_dispatch_trace.py renders the result identically.

Usage:
    tools/compose_multi_schedule.py \\
        --base dispatch_schedule.json \\
        --base-name dronet \\
        --instances "dronet:1,mlp:8" \\
        --vmfb-dir /root/iree_run/dronet/breakdowns \\
        --kernel-swap dispatch_15:custom_elementwise_2048.vmfb \\
        --out combined_schedule.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--base", required=True, type=Path, help="Base schedule.json to replicate.")
    p.add_argument("--base-name", default="dronet", help="Job name applied to the first replica.")
    p.add_argument(
        "--instances",
        default="dronet:1",
        help="Comma-list of <name>:<count> pairs. The first " "matching --base-name uses the base job name.",
    )
    p.add_argument("--stagger-ms", type=float, default=0.0, help="Per-instance start-time stagger in ms.")
    p.add_argument(
        "--vmfb-dir",
        type=str,
        default=None,
        help="Override the vmfb_path with this base directory " "joined to each dispatch's basename.",
    )
    p.add_argument(
        "--kernel-swap",
        action="append",
        default=[],
        help="Repeatable <dispatch_key>:<vmfb_path> override "
        "(applies to every replica). Use this to swap in "
        "LLM-generated / custom kernels.",
    )
    p.add_argument(
        "--kernel-tag",
        action="append",
        default=[],
        help="Repeatable <dispatch_key>:<variant> tag attached "
        "to that dispatch (every replica). Examples: "
        "<key>:llm, <key>:tile, <key>:megakernel. The tag "
        "is recorded as kernel_variant in the schedule and "
        "carried through to the trace CSV.",
    )
    p.add_argument("--out", required=True, type=Path)
    return p.parse_args()


def load_base(path: Path) -> dict:
    return json.loads(path.read_text())


def build_replica(
    base: dict,
    replica_name: str,
    base_name: str,
    start_offset_ms: float,
    vmfb_dir: str | None,
    kernel_swaps: dict[str, str],
    kernel_tags: dict[str, str],
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    base_dispatches = base.get("dispatches", {})
    for k, d in base_dispatches.items():
        new_key = f"{replica_name}_{k}"
        new_d = dict(d)
        # Keep the replica name verbatim so distinct instances get distinct
        # colors in the plot (mlp0, mlp1, ... rather than collapsing to "mlp").
        new_d["job_name"] = replica_name
        # Start time + dependency rewriting.
        new_d["start_time"] = d.get("start_time", 0.0) + start_offset_ms
        # planned_start_us (in some schedules) and start_time_us
        if "start_time_us" in d:
            new_d["start_time_us"] = d["start_time_us"] + start_offset_ms * 1000.0
        new_d["dependencies"] = [f"{replica_name}_{dep}" for dep in d.get("dependencies", [])]
        if "time_dependency" in d and d["time_dependency"]:
            new_d["time_dependency"] = f"{replica_name}_{d['time_dependency']}"
        # vmfb_path: kernel swap > vmfb-dir override > base value.
        # Strip any directory from the base value before applying swaps.
        base_path = d.get("vmfb_path", "")
        base_basename = Path(base_path).name if base_path else f"{k}.vmfb"
        if k in kernel_swaps:
            new_d["vmfb_path"] = kernel_swaps[k]
        elif vmfb_dir:
            new_d["vmfb_path"] = f"{vmfb_dir.rstrip('/')}/{base_basename}"
        if k in kernel_tags:
            new_d["kernel_variant"] = kernel_tags[k]
        out[new_key] = new_d
    return out


def main() -> int:
    args = parse_args()
    base = load_base(args.base)

    instance_specs: list[tuple[str, int]] = []
    for spec in args.instances.split(","):
        spec = spec.strip()
        if not spec:
            continue
        name, _, count = spec.partition(":")
        instance_specs.append((name, int(count) if count else 1))

    swap_map: dict[str, str] = {}
    for s in args.kernel_swap:
        k, _, v = s.partition(":")
        swap_map[k] = v
    tag_map: dict[str, str] = {}
    for s in args.kernel_tag:
        k, _, v = s.partition(":")
        tag_map[k] = v

    composed: dict[str, dict] = {}
    instance_index = 0
    for name, count in instance_specs:
        for i in range(count):
            replica = f"{name}{i}"
            offset_ms = instance_index * args.stagger_ms
            composed.update(build_replica(base, replica, args.base_name, offset_ms, args.vmfb_dir, swap_map, tag_map))
            instance_index += 1

    payload = {
        "schema_version": 1,
        "machines": base.get("machines", ["CPU_P", "CPU_E"]),
        "device_map": base.get(
            "device_map",
            {
                "CPU_P": "@device_a",
                "CPU_E": "@device_b",
            },
        ),
        "source": "tools/compose_multi_schedule.py",
        "dispatches": composed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"composed {len(composed)} dispatches across " f"{instance_index} instances -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
