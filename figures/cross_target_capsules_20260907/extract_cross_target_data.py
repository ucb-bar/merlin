#!/usr/bin/env python3
"""Freeze comparable campaign/capsule telemetry for the 2026-09-07 runs."""

from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

RUNS = {
    "atlas": {
        "run_id": "merlincirct_atlas_fresh_func_20260907",
        "latest_verdict": "verdict_round_00.json",
        "results": "runs_00",
    },
    "gemmini": {
        "run_id": "merlincirct_gemmini_universal_smallread_resnet50_20260907",
        "latest_verdict": "verdict_round_01.json",
        "results": "runs_01",
    },
    "radiance": {
        "run_id": "merlincirct_radiance_pr1_cohort29_20260907",
        "latest_verdict": "verdict_round_01.json",
        "results": "runs_01",
    },
}


def product(shape):
    out = 1
    for value in shape or []:
        if isinstance(value, int):
            out *= value
    return out


def shape_regime(name: str, shapes: list[list[int]]) -> str:
    low = name.lower()
    for key, label in (
        ("odd_tail", "odd tail"), ("partial", "partial tile"),
        ("sub_tile", "sub-tile"), ("aligned", "aligned"),
        ("tall_skinny", "tall-skinny"), ("wide_skinny", "wide-skinny"),
        ("gemv", "GEMV-like"), ("square", "square"),
        ("projection", "projection"), ("spill", "spill/stress"),
    ):
        if key in low:
            return label
    dims = [d for s in shapes for d in s if isinstance(d, int) and d > 0]
    if not dims:
        return "unspecified"
    if max(dims) / min(dims) >= 8:
        return "skewed"
    if all(d % 16 == 0 for d in dims):
        return "aligned"
    return "irregular"


def family_of(spec: dict) -> str:
    family = (spec.get("semantic") or {}).get("semantic_family")
    if family:
        return str(family).replace("_", " ")
    op = str((spec.get("operation") or {}).get("op", "other")).lower()
    if any(x in op for x in ("matmul", "gemm", "gemv", "conv", "linear")):
        return "contraction"
    if any(x in op for x in ("norm", "softmax")):
        return "normalization"
    if any(x in op for x in ("reduce", "pool")):
        return "reduction"
    if any(x in op for x in ("move", "transpose", "reshape", "flatten")):
        return "movement"
    return op.replace("_", " ")


def run_dir(target: str, run_id: str) -> Path:
    return ROOT / "out/runs" / target / "capsule-bench/merlin_assisted" / run_id


def workspace(target: str, run_id: str) -> Path:
    return ROOT / "merlin/experiments/capsule_bench/targets" / target / "_qa_ws" / run_id / "workspace"


def capsule_index(ws: Path) -> dict[str, Path]:
    result = {}
    for group in ("isa", "layers", "model", "model_slices"):
        for path in (ws / group).glob("*/capsule.yaml"):
            result.setdefault(path.parent.name, path)
    return result


def extract_target(target: str, cfg: dict) -> tuple[dict, list[dict], list[dict], list[dict]]:
    rd = run_dir(target, cfg["run_id"])
    ws = workspace(target, cfg["run_id"])
    verdict_path = rd / "qa_history" / cfg["latest_verdict"]
    verdict = json.loads(verdict_path.read_text())
    timing = json.loads((rd / "timing_detailed.json").read_text())
    graded_capsules = [x for x in verdict["per_capsule"] if x.get("status") != "budget_exhausted"]
    if len(graded_capsules) != verdict["n_capsules"]:
        raise ValueError(f"{target}: expected {verdict['n_capsules']} graded capsules, found {len(graded_capsules)}")
    tools = timing.get("tools", {}).get("by_tool", {})
    llm = timing.get("llm") or {}
    llm_tokens = llm.get("tokens") or {}
    llm_rates = llm.get("token_rates") or {}
    turnaround = llm.get("client_observed_turnaround_s") or {}
    input_chars = sum(x.get("input_chars", 0) or 0 for x in tools.values())
    output_chars = sum(x.get("output_chars", 0) or 0 for x in tools.values())
    l3_pass = sum(1 for x in graded_capsules if x.get("tiers", {}).get("L3") == "pass")
    l3_pass_carried = sum(
        1 for x in graded_capsules
        if x.get("tiers", {}).get("L3") == "pass" and "L3" in (x.get("tier_reuse") or {}).get("carried", [])
    )
    l3_pass_executed = sum(
        1 for x in graded_capsules
        if x.get("tiers", {}).get("L3") == "pass" and "L3" in (x.get("tier_reuse") or {}).get("executed", [])
    )
    summary = {
        "target": target,
        "run_id": cfg["run_id"],
        "latest_verdict": cfg["latest_verdict"],
        "graded_at": verdict["graded_at"],
        "passed": verdict["n_passed"],
        "capsules": verdict["n_capsules"],
        "pass_fraction": verdict["n_passed"] / verdict["n_capsules"],
        "l3_pass": l3_pass,
        "l3_pass_executed": l3_pass_executed,
        "l3_pass_carried": l3_pass_carried,
        "think_generate_s": timing["think_generate_s"],
        "tool_wait_s": timing["tool_and_wait_s"],
        "measured_span_s": timing["measured_span_s"],
        "sessions": timing["sessions"],
        "tool_calls": timing["tool_calls_matched"],
        "tool_calls_unterminated": timing["tool_calls_unterminated"],
        "tool_concurrency_overlap_s": timing["tool_concurrency_overlap_s"],
        "input_chars": input_chars,
        "output_chars": output_chars,
        "provider_usage_events_available": timing["tokens"].get("available", False),
        "token_source": "response-level rollout snapshots",
        "tokens_total": llm_tokens.get("tokens_total"),
        "tokens_fresh_input": llm_tokens.get("tokens_fresh_input"),
        "tokens_cache_read": llm_tokens.get("tokens_cache_read"),
        "tokens_cache_write": llm_tokens.get("tokens_cache_write"),
        "tokens_output": llm_tokens.get("tokens_output"),
        "tokens_reasoning": llm_tokens.get("tokens_reasoning"),
        "cache_read_share": llm_tokens.get("cache_read_share_of_input"),
        "output_tokens_per_client_turnaround_s": llm_rates.get("output_tokens_per_client_observed_turnaround_s"),
        "turnaround_p50_s": turnaround.get("p50"),
        "turnaround_p95_s": turnaround.get("p95"),
    }
    tool_rows = []
    for tool_name, values in tools.items():
        tool_rows.append({"target": target, "tool": tool_name, **values})

    idx = capsule_index(ws)
    result_root = rd / "_qa_work" / cfg["results"] / "runs" / f"{target}-capsule-bench"
    capsules = []
    missing_specs = []
    for observed in graded_capsules:
        name = observed["capsule"]
        spec_path = idx.get(name)
        if spec_path is None:
            missing_specs.append(name)
            spec = {"name": name, "kind": "unknown", "inputs": []}
        else:
            spec = yaml.safe_load(spec_path.read_text())
        inputs = spec.get("inputs") or []
        shapes = [x.get("shape") or [] for x in inputs]
        dtypes = sorted({str(x.get("dtype", "unknown")) for x in inputs})
        result_path = result_root / name / "capsule_result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        row = {
            "target": target,
            "capsule": name,
            "status": observed.get("status", "unknown"),
            "numeric_status": observed.get("numeric_status"),
            "failure_plane": observed.get("failure_plane"),
            "kind": spec.get("kind", "unknown"),
            "family": family_of(spec),
            "operation": (spec.get("operation") or {}).get("op", "unknown"),
            "shape_regime": shape_regime(name, shapes),
            "shape_signature": " + ".join("x".join(map(str, s)) for s in shapes if s) or "scalar/unknown",
            "dtypes": "+".join(dtypes),
            "input_elements": sum(product(s) for s in shapes),
            "max_tensor_elements": max([product(s) for s in shapes] or [0]),
            "required_tiers": "+".join(spec.get("required_oracle_tiers") or []),
            "l3_evidence": (
                "carried pass" if observed.get("tiers", {}).get("L3") == "pass" and "L3" in (observed.get("tier_reuse") or {}).get("carried", [])
                else "fresh pass" if observed.get("tiers", {}).get("L3") == "pass" and "L3" in (observed.get("tier_reuse") or {}).get("executed", [])
                else "pass (origin unspecified)" if observed.get("tiers", {}).get("L3") == "pass"
                else "failed" if observed.get("tiers", {}).get("L3") == "fail"
                else "not passed"
            ),
        }
        for tier in ("L2", "L3"):
            tier_data = (result.get("tiers") or {}).get(tier) or {}
            timing_data = tier_data.get("timing") or {}
            row[f"{tier.lower()}_status"] = tier_data.get("status")
            row[f"{tier.lower()}_wall_s"] = timing_data.get("adapter_wall_s")
            row[f"{tier.lower()}_sim_s"] = timing_data.get("sim_active_s")
            row[f"{tier.lower()}_engine"] = tier_data.get("engine")
            row[f"{tier.lower()}_cycles"] = tier_data.get("cycles")
        capsules.append(row)
    summary["missing_capsule_specs"] = missing_specs

    history = []
    for path in sorted((rd / "qa_history").glob("verdict*.json"), key=lambda p: p.stat().st_mtime):
        data = json.loads(path.read_text())
        if not data.get("gradeable") or not data.get("n_capsules"):
            continue
        history.append({
            "target": target,
            "verdict": path.name,
            "graded_at": data["graded_at"],
            "passed": data["n_passed"],
            "capsules": data["n_capsules"],
            "pass_fraction": data["n_passed"] / data["n_capsules"],
        })
    return summary, capsules, history, tool_rows


def write_csv(path: Path, rows: list[dict]):
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    summaries, capsules, history, tools = [], [], [], []
    for target, cfg in RUNS.items():
        summary, target_capsules, target_history, target_tools = extract_target(target, cfg)
        summaries.append(summary)
        capsules.extend(target_capsules)
        history.extend(target_history)
        tools.extend(target_tools)
    payload = {
        "snapshot_at": datetime.now(timezone.utc).isoformat(),
        "comparability_note": "Active campaigns with different corpora and simulator engines; cross-target timings are observational, not a causal compiler ablation.",
        "campaigns": summaries,
        "capsules": capsules,
        "history": history,
        "tools": tools,
    }
    (HERE / "cross_target_snapshot.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_csv(HERE / "capsules.csv", capsules)
    write_csv(HERE / "campaigns.csv", summaries)
    write_csv(HERE / "tools.csv", tools)
    print(json.dumps({"campaigns": len(summaries), "capsules": len(capsules), "history": len(history), "tools": len(tools)}, indent=2))


if __name__ == "__main__":
    main()
