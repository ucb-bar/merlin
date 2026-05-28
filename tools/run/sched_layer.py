#!/usr/bin/env python3
"""LAYER-granularity scheduling without compiler-side chunk_compile.

Trick: each LAYER chunk is a sequence of DISPATCH-level chunks the
scheduler treats atomically. The LAYER's cost on machine M is the sum
of constituent dispatch costs on M (with cross-machine transfers
counted iff a constituent's predecessor lives outside the layer). The
schedule is solved at LAYER granularity, then expanded back to a
DISPATCH-level schedule.json that the existing scheduler_bin can run.
The runtime ordering within a layer follows the dispatches' topological
order; all dispatches in a single LAYER share the layer's hardware
target.

Inputs:
  - chunk_manifest.json (from tools/chunk_extractor.py extract --level=layer)
  - profiled_manifest.json (from tools/board_roundtrip.py)

Output:
  - layer_profiled_manifest.json — LAYER-aggregated cost matrix in the
    same schema as profiled_manifest.json so merlin_adapter.py schedule
    consumes it transparently.
  - <layer-merlin-dir>/breakdowns/manifest.json — synthesised
    DISPATCH-level manifest where each chunk = one LAYER (multi-dispatch)
    so the scheduler treats it atomically.
  - <layer-merlin-dir>/breakdowns/profiled_manifest.json — profile data
    for the LAYER chunks.

Usage:
    tools/layer_schedule.py build \\
        --chunk-manifest /tmp/dronet_layer/chunk_manifest.json \\
        --dispatch-merlin-dir eval/qrb5165/dronet \\
        --out eval/qrb5165/dronet_layer
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys

_LOG = logging.getLogger("layer_schedule")


def _load_json(p: pathlib.Path) -> dict:
    return json.loads(p.read_text())


def aggregate_costs(
    chunk_manifest: dict,
    profiled_manifest: dict,
) -> tuple[dict, dict]:
    """Return (layer_dispatches_dict, layer_processing_times_per_machine).

    layer_dispatches_dict: schema-compatible with breakdown manifest.json,
    one entry per layer-chunk. Carries the constituent dispatch ids in
    `parent_dispatch_ids` so the runtime can iterate them in order.

    Layer cost on machine M = sum_{d in chunk} cost[d, M].
    """
    dispatches = profiled_manifest.get("dispatches", {})
    layer_chunks = chunk_manifest.get("chunks", {})
    layer_dispatches: dict[str, dict] = {}
    # Discover machines from any dispatch's profile.
    machines: list[str] = []
    for e in dispatches.values():
        for m in e.get("profiles", {}) or {}:
            if m not in machines:
                machines.append(m)
    if not machines:
        raise RuntimeError("profiled_manifest has no profiles per machine")

    # Order chunks by smallest parent dispatch id to keep deterministic.
    sorted_chunks = sorted(
        layer_chunks.items(),
        key=lambda kv: min(kv[1].get("parent_dispatch_ids", [10**9])),
    )
    chunk_order: dict[str, int] = {k: i for i, (k, _) in enumerate(sorted_chunks)}

    for new_idx, (chunk_name, c) in enumerate(sorted_chunks):
        member_names = c.get("parent_dispatch_names", [])
        # Sum costs per machine. Drop machines where any member is missing
        # a profile (so the cost matrix entry becomes infeasible).
        per_machine: dict[str, float] = {}
        for m in machines:
            tot = 0.0
            ok = True
            for member in member_names:
                e = dispatches.get(member)
                if e is None:
                    ok = False
                    break
                v = (e.get("profiles", {}) or {}).get(m, {}).get("mean_time_us")
                if v is None:
                    ok = False
                    break
                tot += float(v)
            if ok:
                per_machine[m] = tot
        # Compose schema-compatible entry.
        layer_dispatches[f"layer_{new_idx}"] = {
            "id": new_idx,
            "subid": None,
            "ordinal": 1,
            "total": 1,
            "op_summary": c.get("op_summary", ""),
            "module_name": "",
            "inputs": list(c.get("inputs", [])),
            "outputs": list(c.get("outputs", [])),
            # Renumber dependencies using the new chunk ordering.
            "dependencies": [
                f"layer_{chunk_order[f'chunk_{d}']}" for d in c.get("deps", []) if f"chunk_{d}" in chunk_order
            ],
            # Carry the constituent dispatch info so the runtime can
            # iterate them in topological order on the assigned target.
            "parent_dispatch_ids": list(c.get("parent_dispatch_ids", [])),
            "parent_dispatch_names": list(member_names),
            "profiles": {m: {"mean_time_us": v} for m, v in per_machine.items()},
            # When a downstream consumer wants a single VMFB to invoke for
            # the layer (PR 3), it should set vmfb_path here. Until then
            # the runtime needs to be extended with a "multi-vmfb chunk"
            # mode (open follow-on noted in plan).
            "vmfb_path": "",
        }

    return layer_dispatches, machines


def cmd_build(args: argparse.Namespace) -> int:
    chunk_manifest = _load_json(args.chunk_manifest)
    if chunk_manifest.get("level") != "layer":
        _LOG.warning(
            "chunk_manifest level is %s, not 'layer' — proceeding " "but this tool is targeted at LAYER granularity",
            chunk_manifest.get("level"),
        )
    src_breakdowns = args.dispatch_merlin_dir / "breakdowns"
    profiled = src_breakdowns / "profiled_manifest.json"
    if not profiled.exists():
        _LOG.error("missing %s — run board_roundtrip.py first", profiled)
        return 2
    profiled_manifest = _load_json(profiled)

    layer_dispatches, machines = aggregate_costs(chunk_manifest, profiled_manifest)

    out_breakdowns = args.out / "breakdowns"
    out_breakdowns.mkdir(parents=True, exist_ok=True)
    # Write a breakdowns/manifest.json compatible with merlin_adapter, but
    # keyed by chunk (each "dispatch" entry IS a layer).
    manifest_out = {
        "schema_version": 1,
        "source": "tools/layer_schedule.py",
        "level": "layer",
        "num_dispatches": len(layer_dispatches),
        "dispatches": {k: {kk: vv for kk, vv in v.items() if kk != "profiles"} for k, v in layer_dispatches.items()},
    }
    (out_breakdowns / "manifest.json").write_text(json.dumps(manifest_out, indent=2) + "\n")
    profiled_out = {
        "schema_version": 1,
        "source": "tools/layer_schedule.py",
        "level": "layer",
        "num_dispatches": len(layer_dispatches),
        "dispatches": layer_dispatches,
    }
    (out_breakdowns / "profiled_manifest.json").write_text(json.dumps(profiled_out, indent=2) + "\n")

    _LOG.info("built %d layer chunks across %d machines (%s)", len(layer_dispatches), len(machines), machines)
    _LOG.info("manifest -> %s", out_breakdowns / "manifest.json")
    _LOG.info("profiled -> %s", out_breakdowns / "profiled_manifest.json")
    return 0


def cmd_expand(args: argparse.Namespace) -> int:
    """Expand a LAYER-granularity schedule.json back into a DISPATCH-level
    schedule.json the existing scheduler_runner.cc can run.

    For each layer with start_time T and assigned target M, distribute
    its constituent dispatches over [T, T + layer_duration] in topo order.
    Each constituent gets the SAME hardware_target M and a start_time
    that respects intra-layer dependencies + the original per-dispatch
    cost on M.
    """
    layer_sched = _load_json(args.layer_schedule)
    sched_dispatches = layer_sched.get("dispatches", {})

    src_breakdowns = args.dispatch_merlin_dir / "breakdowns"
    src_profiled = _load_json(src_breakdowns / "profiled_manifest.json")
    src_dispatches = src_profiled.get("dispatches", {})

    # Pull constituent-dispatch names from the LAYER profiled_manifest
    # (the schedule.json from merlin_adapter doesn't carry that field).
    layer_profiled_path = (
        (args.layer_merlin_dir or args.layer_schedule.parent.parent) / "breakdowns" / "profiled_manifest.json"
    )
    if not layer_profiled_path.exists():
        _LOG.error(
            "missing %s — pass --layer-merlin-dir to point at the " "dir produced by `layer_schedule.py build`",
            layer_profiled_path,
        )
        return 2
    layer_profiled = _load_json(layer_profiled_path)
    layer_meta = layer_profiled.get("dispatches", {})
    # Merge the schedule's hardware_target / start_time onto the layer
    # meta so we have one source-of-truth dict per layer.
    layer_dispatches: dict[str, dict] = {}
    for name, sched_entry in sched_dispatches.items():
        meta = dict(layer_meta.get(name, {}))
        meta.update(sched_entry)
        layer_dispatches[name] = meta

    out_dispatches: dict[str, dict] = {}
    layer_to_target: dict[str, str] = {n: e["hardware_target"] for n, e in layer_dispatches.items()}

    for layer_name, layer in layer_dispatches.items():
        layer_start_us = float(layer.get("start_time_us", 0.0))
        layer_target = layer["hardware_target"]
        layer_skipped = bool(layer.get("skipped", False))
        members = list(layer.get("parent_dispatch_names", []))
        # Topologically sort the members using src_profiled deps.
        ordered = _topo_sort_subset(members, src_dispatches)
        # Distribute over time on the assigned target.
        cur_t_us = layer_start_us
        for m_name in ordered:
            e = src_dispatches[m_name]
            cost_us = (e.get("profiles", {}) or {}).get(layer_target, {}).get("mean_time_us", 0.0)
            duration_ms = float(cost_us) / 1000.0
            entry = {
                "id": e["id"],
                "subid": e.get("subid"),
                "ordinal": 1,
                "total": 1,
                "hardware_target": layer_target,
                "start_time_us": cur_t_us,
                "start_time": cur_t_us / 1000.0,
                "duration": duration_ms,
                # The expanded schedule's per-dispatch deps must match
                # the per-dispatch breakdown manifest (NOT the layer's
                # cross-layer deps, which are already enforced via the
                # layer's start_time).
                "dependencies": list(e.get("dependencies", [])),
                "op_summary": e.get("op_summary", ""),
                "job_name": "",
                "module_name": e.get("module_name", ""),
                "vmfb_path": pathlib.Path(e.get("executable", f"{m_name}.vmfb")).name,
            }
            if layer_skipped:
                entry["skipped"] = True
            if "deadline_us" in layer:
                entry["deadline_us"] = layer["deadline_us"]
            out_dispatches[m_name] = entry
            cur_t_us += float(cost_us)

    payload = {
        "schema_version": 1,
        "machines": layer_sched.get("machines", []),
        "device_map": layer_sched.get("device_map", {}),
        "source": "tools/layer_schedule.py expand",
        "dispatches": out_dispatches,
    }
    args.out_dispatch_schedule.parent.mkdir(parents=True, exist_ok=True)
    args.out_dispatch_schedule.write_text(json.dumps(payload, indent=2) + "\n")
    _LOG.info(
        "expanded %d layers -> %d per-dispatch entries -> %s",
        len(layer_dispatches),
        len(out_dispatches),
        args.out_dispatch_schedule,
    )
    return 0


def _topo_sort_subset(members: list[str], dispatches: dict) -> list[str]:
    member_set = set(members)
    in_deg = {m: 0 for m in members}
    for m in members:
        for d in dispatches.get(m, {}).get("dependencies", []):
            if d in member_set:
                in_deg[m] += 1
    ready = sorted([m for m, d in in_deg.items() if d == 0])
    out: list[str] = []
    while ready:
        n = ready.pop(0)
        out.append(n)
        for c in members:
            if n in dispatches.get(c, {}).get("dependencies", []):
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    ready.append(c)
                    ready.sort()
    if len(out) != len(members):
        _LOG.warning("topo cycle in layer members; falling back to input order")
        return list(members)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s_b = sub.add_parser("build", help="aggregate DISPATCH costs to LAYER")
    s_b.add_argument("--chunk-manifest", required=True, type=pathlib.Path)
    s_b.add_argument("--dispatch-merlin-dir", required=True, type=pathlib.Path)
    s_b.add_argument("--out", required=True, type=pathlib.Path, help="output dir; gets a breakdowns/ subdir")
    s_b.set_defaults(func=cmd_build)

    s_e = sub.add_parser(
        "expand",
        help="expand a LAYER schedule.json into a "
        "per-dispatch schedule.json the existing "
        "scheduler_runner.cc can run",
    )
    s_e.add_argument("--layer-schedule", required=True, type=pathlib.Path)
    s_e.add_argument(
        "--layer-merlin-dir",
        type=pathlib.Path,
        default=None,
        help="LAYER merlin dir (the --out of `build`). When "
        "omitted, defaults to the schedule.json's "
        "grandparent dir. Provides parent_dispatch_names "
        "per layer chunk, which the schedule.json itself "
        "does not carry.",
    )
    s_e.add_argument(
        "--dispatch-merlin-dir",
        required=True,
        type=pathlib.Path,
        help="provides the per-dispatch profiled_manifest.json " "with module_name + vmfb_path + costs.",
    )
    s_e.add_argument("--out-dispatch-schedule", required=True, type=pathlib.Path)
    s_e.set_defaults(func=cmd_expand)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
