"""DSE reporting: per-strategy scoreboard, and the hardware-only vs interface-aware capstone.

The capstone answers the thesis question directly: sweeping the *hardware* space alone (with
the interface fixed to the opaque baseline) buys latency only by enlarging DRAM bandwidth /
SRAM (area); adding the *interface* axis lets the DSE instead expose a small resident store +
accumulator-commit unit and reach lower latency at equal-or-less area — so the best design
changes category. Produces a scoreboard, decision report, and Pareto CSVs.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

from merlin.common import paths
from merlin.dse.exploitability import exploitability
from merlin.dse.hardware_space import (area_proxy, build_hardware_space, default_cost_model)
from merlin.dse.interface_space import baseline_only, build_interface_space
from merlin.dse.pareto import compute_pareto, frontier_dominates
from merlin.dse.strategy import default_strategies, evaluate_strategy


def scoreboard(rpv: dict, strategies=None, cost_model: dict | None = None) -> list[dict]:
    """One row per strategy: cycles, energy, speedup vs baseline, exploitability, area, decision."""
    cm = cost_model or default_cost_model()
    strats = strategies or default_strategies()
    evals = {s.id: evaluate_strategy(s, rpv, cm) for s in strats}
    base = next((e for e in evals.values() if e["variant_class"] == "baseline"), None)
    oracle = next((e for e in evals.values() if e["variant_class"] == "oracle"), None)
    base_cyc = base["cycles"] if base else max(e["cycles"] for e in evals.values())
    oracle_speedup = (base_cyc / oracle["cycles"]) if oracle and oracle["cycles"] else 1.0

    rows = []
    for s in strats:
        e = evals[s.id]
        speedup = base_cyc / e["cycles"] if e["cycles"] else 1.0
        rows.append({
            "strategy": s.id,
            "variant_class": s.variant_class,
            "features": ";".join(s.interface_features),
            "cycles": round(e["cycles"], 1),
            "energy": e["energy"],
            "speedup_vs_baseline": round(speedup, 3),
            "exploitability": round(exploitability(speedup, oracle_speedup), 3),
            "area": area_proxy(cm, s.interface_features),
        })
    return rows


def hardware_vs_interface(rpv: dict, hw_grid: dict | None = None) -> dict:
    """Compare the hardware-only and interface-aware design frontiers (latency vs area)."""
    hw_space = build_hardware_space(hw_grid)
    base_iface = baseline_only()
    # Buildable interfaces only — the oracle is a bound, not a candidate hardware design.
    all_iface = build_interface_space(
        variant_classes=["baseline", "hardware_managed", "software_visible"])
    # A resident interface is only feasible where the resident store fits the packed weight.
    need_bytes = int(rpv["metrics"].get("pack_bytes", 0)) * int(
        rpv["metrics"].get("distinct_weights", 1))

    def _feasible(s, cm) -> bool:
        if "resident_packed_tensor" in s.interface_features:
            return cm.get("resident_store_bytes", 0) >= need_bytes
        return True

    def cells(strategies):
        out = []
        for cm in hw_space:
            for s in strategies:
                if not _feasible(s, cm):
                    continue
                e = evaluate_strategy(s, rpv, cm)
                out.append({
                    "cycles": round(e["cycles"], 1),
                    "area": area_proxy(cm, s.interface_features),
                    "strategy": s.id,
                    "variant_class": s.variant_class,
                    "features": list(s.interface_features),
                    "dram_bytes_per_cycle": cm["dram_bytes_per_cycle"],
                    "resident_store_bytes": cm["resident_store_bytes"],
                    "dispatch_fixed_cycles": cm["dispatch_fixed_cycles"],
                })
        return out

    hw_only = cells(base_iface)
    iface = cells(all_iface)
    objs, modes = ["cycles", "area"], ["min", "min"]
    hw_front = compute_pareto(hw_only, objs, modes)
    iface_front = compute_pareto(iface, objs, modes)

    hw_best = min(hw_front, key=lambda p: p["cycles"])
    iface_best = min(iface_front, key=lambda p: p["cycles"])
    return {
        "hardware_only_frontier": sorted(hw_front, key=lambda p: p["area"]),
        "interface_aware_frontier": sorted(iface_front, key=lambda p: p["area"]),
        "interface_dominates_hardware_only": frontier_dominates(iface_front, hw_front, objs, modes),
        "hardware_only_best": hw_best,
        "interface_aware_best": iface_best,
        "best_interface_changes_category": hw_best["strategy"] != iface_best["strategy"],
    }


def recommended_hw_features(capstone: dict) -> dict:
    """From the interface-aware frontier, the minimal-useful hardware features to build."""
    front = capstone["interface_aware_frontier"]
    resident_pts = [p for p in front if "resident_packed_tensor" in p["features"]]
    accumulator_pts = [p for p in front if "accumulator_commit" in p["features"]]
    rec: dict = {"required_contracts": sorted(set(capstone["interface_aware_best"]["features"]))}
    if resident_pts:
        store = sorted(p["resident_store_bytes"] for p in resident_pts if p["resident_store_bytes"])
        if store:
            rec["resident_store_bytes"] = {"min_useful": store[0], "saturation": store[-1]}
    if accumulator_pts:
        rec["accumulator_commit_unit"] = True
    return rec


def _csv(rows: list[dict], columns: list[str]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k) for k in columns})
    return buf.getvalue()


def build_report(rpv: dict, workload: str = "vla_action_chunk_decode",
                 cost_model: dict | None = None, hw_grid: dict | None = None,
                 out_dir: str | Path | None = None) -> dict:
    """Produce the scoreboard, capstone frontiers, and a decision report; write artifacts."""
    board = scoreboard(rpv, cost_model=cost_model)
    capstone = hardware_vs_interface(rpv, hw_grid)
    rec = recommended_hw_features(capstone)

    board_csv = _csv(board, ["strategy", "variant_class", "features", "cycles", "energy",
                             "speedup_vs_baseline", "exploitability", "area"])
    front_cols = ["area", "cycles", "strategy", "variant_class", "dram_bytes_per_cycle",
                  "resident_store_bytes", "dispatch_fixed_cycles"]
    decision = _decision_md(workload, board, capstone, rec)

    artifacts = {
        "scoreboard.csv": board_csv,
        "pareto_hardware_only.csv": _csv(capstone["hardware_only_frontier"], front_cols),
        "pareto_interface_aware.csv": _csv(capstone["interface_aware_frontier"], front_cols),
        "decision_report.md": decision,
    }
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, text in artifacts.items():
            (out / name).write_text(text, encoding="utf-8")
    return {"scoreboard": board, "capstone": capstone, "recommended_hw_features": rec,
            "artifacts": artifacts}


def _decision_md(workload: str, board: list[dict], capstone: dict, rec: dict) -> str:
    hw_b, if_b = capstone["hardware_only_best"], capstone["interface_aware_best"]
    lines = [
        f"# DSE decision report — {workload}", "",
        "## Per-strategy scoreboard", "",
        "| strategy | class | cycles | speedup | exploitability | area |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in board:
        lines.append(f"| {r['strategy']} | {r['variant_class']} | {r['cycles']} | "
                     f"{r['speedup_vs_baseline']} | {r['exploitability']} | {r['area']} |")
    lines += [
        "", "## Hardware-only vs interface-aware", "",
        f"- hardware-only best: **{hw_b['strategy']}** "
        f"(cycles={hw_b['cycles']}, area={hw_b['area']}, "
        f"dram_bpc={hw_b['dram_bytes_per_cycle']}, resident={hw_b['resident_store_bytes']})",
        f"- interface-aware best: **{if_b['strategy']}** "
        f"(cycles={if_b['cycles']}, area={if_b['area']}, "
        f"dram_bpc={if_b['dram_bytes_per_cycle']}, resident={if_b['resident_store_bytes']})",
        f"- interface-aware frontier dominates hardware-only: "
        f"**{capstone['interface_dominates_hardware_only']}**",
        f"- best design changes category: **{capstone['best_interface_changes_category']}**",
        "", "## Recommended hardware features", "",
        "```yaml", _yaml_block(rec), "```", "",
    ]
    return "\n".join(lines)


def _yaml_block(d: dict) -> str:
    from merlin.common.yaml import dump_yaml
    return dump_yaml(d).rstrip()
