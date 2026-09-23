"""Run timing reports from retained evidence; no target selection at import time."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from merlin_experiments.phase1.context import InvocationContext, add_context_arguments, resolve_context
from merlin_experiments.phase1.telemetry import evidence, timeline


def transcript_paths(run_dir: Path) -> list[Path]:
    """The transcripts of one run, preferring the per-round files (they cannot interleave rounds).

    Falls back to the concatenated ``transcript.jsonl``, which ``decompose`` segments anyway.
    """
    run_dir = Path(run_dir)
    per_round = sorted((run_dir / "rounds").glob("round_*.transcript.jsonl"))
    if per_round:
        return per_round
    single = run_dir / "transcript.jsonl"
    return [single] if single.is_file() else []


def circt_gate(run_dir: Path) -> dict:
    """Prescreen-gate tally from the run's own gate log (absent log -> zeros, which is what it means)."""
    skips = ran = 0
    log = Path(run_dir) / "circt_gate_log.jsonl"
    if log.is_file():
        for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if not isinstance(rec, dict):
                continue
            skips += int(bool(rec.get("sim_skipped")))
            ran += int(not rec.get("sim_skipped"))
    return {"sims_skipped": skips, "sims_run": ran}


_TIMING_FIELDS = ("build_s", "sim_active_s", "oracle_wait_s", "adapter_wall_s")


def oracle_timing(run_dir: Path) -> dict:
    """Every L-tier invocation and a lossless-by-status aggregate.

    The source record's own ``engine`` is authoritative; tier names are fidelity levels, not simulator
    names.  Missing phase timings stay missing and mark sums as lower bounds.  Repeated capsule results
    are separate paid invocations and therefore intentionally remain separate rows.
    """
    run_dir = Path(run_dir)
    rows = []
    for result_path in sorted(run_dir.rglob("capsule_result.json")):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        capsule = result.get("capsule") or result_path.parent.name
        for tier, tier_result in sorted((result.get("tiers") or {}).items()):
            if not isinstance(tier_result, dict):
                continue
            timing = tier_result.get("timing")
            timing = dict(timing) if isinstance(timing, dict) else None
            rows.append(
                {
                    "source": str(result_path.relative_to(run_dir)),
                    "capsule": capsule,
                    "tier": tier,
                    "status": tier_result.get("status"),
                    "engine": tier_result.get("engine"),
                    "measured_now": tier_result.get("measured_now"),
                    "timing": timing,
                    "timing_complete": bool(
                        timing and all(isinstance(timing.get(field), (int, float)) for field in _TIMING_FIELDS)
                    ),
                }
            )
    by_tier = {}
    for tier in sorted({row["tier"] for row in rows}):
        selected = [row for row in rows if row["tier"] == tier]
        timed = [row for row in selected if row["timing"] is not None]
        field_values = {
            field: [float(row["timing"][field]) for row in timed if isinstance(row["timing"].get(field), (int, float))]
            for field in _TIMING_FIELDS
        }
        adapter = field_values["adapter_wall_s"]
        by_tier[tier] = {
            "records": len(selected),
            "timed_records": len(timed),
            "complete_timing_records": sum(row["timing_complete"] for row in selected),
            "missing_timing_records": len(selected) - len(timed),
            "statuses": dict(sorted(Counter(str(row["status"]) for row in selected).items())),
            "engines": dict(sorted(Counter(str(row["engine"]) for row in selected if row["engine"]).items())),
            "totals_s": {field: (round(sum(values), 6) if values else None) for field, values in field_values.items()},
            "fields_measured": {field: len(values) for field, values in field_values.items()},
            "totals_are_lower_bounds": any(len(values) < len(selected) for values in field_values.values()),
            "adapter_wall_distribution_s": {
                "mean": round(sum(adapter) / len(adapter), 6) if adapter else None,
                "p50": timeline._percentile(adapter, 0.50),
                "p95": timeline._percentile(adapter, 0.95),
                "max": max(adapter) if adapter else None,
            },
        }
    return {
        "capsule_result_files": len({row["source"] for row in rows}),
        "tier_records": len(rows),
        "by_tier": by_tier,
        "per_invocation": rows,
        "note": (
            "one row per capsule_result tier; repeated grading attempts remain separate paid "
            "invocations. Null timings are unknown/not-run, never zero."
        ),
    }


def _codex_round_integrity(run_dir: Path, timing: dict, tokens: dict, reconciliation: dict, resources: dict) -> dict:
    """Completeness gate for finished Codex rounds; an active turn is explicitly incomplete."""
    failures = []
    rounds = Path(run_dir) / "rounds"
    stems = sorted({p.name.split(".transcript.jsonl")[0] for p in rounds.glob("round_[0-9][0-9].transcript.jsonl")})
    details = []
    for stem in stems:
        required = {
            "prompt": rounds / f"{stem}.prompt.txt",
            "raw": rounds / f"{stem}.codex_events.raw.jsonl",
            "timestamped": rounds / f"{stem}.codex_events.timestamped.jsonl",
            "normalized": rounds / f"{stem}.transcript.jsonl",
            "final": rounds / f"{stem}.final.txt",
            "summary": rounds / f"{stem}.codex_summary.json",
        }
        missing = [name for name, path in required.items() if not path.is_file()]
        summary = {}
        if required["summary"].is_file():
            try:
                summary = json.loads(required["summary"].read_text(encoding="utf-8"))
            except (OSError, ValueError):
                missing.append("summary_readable")
        round_failures = [f"{stem}:missing_{name}" for name in missing]
        if summary and summary.get("usage_complete") is not True:
            round_failures.append(f"{stem}:usage_incomplete")
        if summary and summary.get("unknown_types"):
            round_failures.append(f"{stem}:unknown_driver_event_types")
        rollout = list(rounds.glob(f"{stem}.codex_rollout_snapshot/**/*.jsonl"))
        if not rollout:
            round_failures.append(f"{stem}:rollout_snapshot_missing")
        failures.extend(round_failures)
        details.append(
            {
                "round": stem,
                "complete": not round_failures,
                "failures": round_failures,
                "turns_started": summary.get("turns_started"),
                "turns_usage_reported": summary.get("turns_usage_reported"),
                "driver_wall_s": summary.get("wall_s"),
            }
        )
    if not stems:
        failures.append("no_completed_round_transcript")
    if timing.get("method") == "unknown":
        failures.append("agent_timing_unavailable")
    if timing.get("tool_calls_unterminated"):
        failures.append("unterminated_tool_calls")
    if timing.get("tool_results_unpaired"):
        failures.append("unpaired_tool_results")
    if not tokens.get("available"):
        failures.append("provider_token_usage_unavailable")
    if reconciliation.get("complete") is not True:
        failures.extend(reconciliation.get("failures") or ["stream_reconciliation_incomplete"])
    if not any((Path(run_dir) / "agent_evidence_snapshot").rglob("*")):
        failures.append("agent_visible_evidence_snapshot_missing")
    if resources.get("available") is not True:
        failures.append("resource_samples_missing")
    return {
        "complete": not failures,
        "failures": failures,
        "rounds": details,
        "policy": (
            "formal completion requires exact prompt/final, raw+stamped+normalized streams, "
            "sealed rollout, complete usage, known timing, and paired tools"
        ),
    }


def decompose_run(run_dir: Path) -> dict:
    """The unified process/oracle telemetry record for one run dir."""
    run_dir = Path(run_dir)
    paths = transcript_paths(run_dir)
    if not paths:
        rec = {
            "method": "unknown",
            "think_generate_s": None,
            "tool_and_wait_s": None,
            "think_pct": None,
            "measured_span_s": None,
            "unavailable_reason": f"no transcript found under {run_dir}",
        }
    else:
        rec = timeline.decompose(timeline.read_events(paths))
    rec["transcripts"] = [p.name for p in paths]
    rec["circt_gate"] = circt_gate(run_dir)
    rec["llm"] = evidence.rollout_telemetry(run_dir)
    rec["oracle"] = oracle_timing(run_dir)
    rec["artifacts"] = evidence.artifact_inventory(run_dir)
    rec["stream_reconciliation"] = evidence.stream_reconciliation(run_dir)
    rec["resources"] = evidence.resource_telemetry(run_dir)
    normalized_tokens = rec.get("tokens") or {}
    integrity_tokens = normalized_tokens
    if not integrity_tokens.get("available") and rec["llm"].get("available"):
        integrity_tokens = {"available": True, **rec["llm"]["tokens"]}
    rec["telemetry_integrity"] = _codex_round_integrity(
        run_dir, rec, integrity_tokens, rec["stream_reconciliation"], rec["resources"]
    )
    rec["generated_at"] = datetime.now(UTC).isoformat()
    rec["measurement_limits"] = {
        "provider_server_latency": "unavailable: provider request-start/TTFT is not emitted",
        "true_decode_token_rate": "unavailable: no per-token timestamp stream",
        "historical_cpu_rss_io": "unavailable unless sampled during the run; cannot be backfilled",
        "available_substitute": (
            "arrival-stamped agent/tool wall, client-observed response turnaround, "
            "provider token buckets, and exact L-tier phase timings"
        ),
    }
    return rec


def write_run_timing(run_dir: Path) -> Path:
    """Seal volatile evidence and write ``<run_dir>/timing_detailed.json``."""
    run_dir = Path(run_dir)
    evidence.snapshot_codex_rollouts(run_dir)
    evidence.snapshot_agent_evidence(run_dir)
    out = run_dir / "timing_detailed.json"
    out.write_text(json.dumps(decompose_run(run_dir), indent=2))
    return out


def _fmt(v, unit="s"):
    return "UNKNOWN" if v is None else f"{v:g}{unit}"


def report_run(run_dir: Path, *, write: bool = False) -> dict:
    rec = decompose_run(run_dir)
    print(f"== {run_dir}")
    print(f"  method            : {rec['method']}")
    if rec.get("unavailable_reason"):
        print(f"  UNAVAILABLE       : {rec['unavailable_reason']}")
    print(f"  think+generate    : {_fmt(rec['think_generate_s'])}")
    print(f"  tool and wait     : {_fmt(rec['tool_and_wait_s'])}")
    print(f"  think share       : {_fmt(rec['think_pct'], '%')}")
    print(
        f"  measured span     : {_fmt(rec.get('measured_span_s'))}"
        f"  (sessions={rec.get('sessions')}, between={_fmt(rec.get('between_session_s'))})"
    )
    if rec["method"] == "arrival_stamps":
        print(
            f"  tool calls        : {rec['tool_calls_matched']} matched, "
            f"{rec['tool_calls_unterminated']} unterminated, "
            f"{rec['tool_results_unpaired']} unpaired results"
        )
        print(
            f"  sum of call durs  : {_fmt(rec['tool_call_seconds_sum'])} "
            f"(overlap {_fmt(rec['tool_concurrency_overlap_s'])} — concurrent tool calls)"
        )
    if write:
        print(f"  wrote {write_run_timing(run_dir)}")
    return rec


# --- legacy cross-arm view ------------------------------------------------------------------------
# The original purpose of this file: the per-tier SIM cost the operator paid, alongside the agent split.
# Kept, but the agent split now comes from `decompose` so it is correct for every driver.

TIER_TOOL = {"L2": "spike", "L3": "verilator/VCS", "L4": "verilator/VCS"}


def harvest_arm(run_dir: Path, target: str) -> dict:
    """Agent split + EXACT per-tier sim wall for one arm's run dir."""
    import yaml

    run_dir = Path(run_dir)
    state = run_dir / "qa_loop_state.yaml"
    st = yaml.safe_load(state.read_text()) if state.is_file() else {}
    active = ((st or {}).get("cumulative") or {}).get("active_wall_s", 0.0)
    sims = {
        "spike": {"runs": 0, "build_s": 0.0, "sim_s": 0.0},
        "verilator/VCS": {"runs": 0, "build_s": 0.0, "sim_s": 0.0},
    }
    for cr in (run_dir / "_qa_work").glob(f"runs_*/runs/{target}-capsule-bench/*/capsule_result.json"):
        try:
            r = json.loads(cr.read_text())
        except ValueError:
            continue
        for tier, tv in (r.get("tiers") or {}).items():
            tm = (tv or {}).get("timing") or {}
            tool = TIER_TOOL.get(tier)
            if tool and tm:
                sims[tool]["runs"] += 1
                sims[tool]["build_s"] += tm.get("build_s") or 0.0
                sims[tool]["sim_s"] += tm.get("sim_active_s") or 0.0
    return {
        "active_wall_min": round(active / 60, 1),
        "agent_session": decompose_run(run_dir),
        "tool_wall_exact": {
            tool: {
                "runs": v["runs"],
                "total_s": round(v["sim_s"] + v["build_s"], 2),
                "per_run_s": round((v["sim_s"] + v["build_s"]) / max(v["runs"], 1), 3),
            }
            for tool, v in sims.items()
        },
    }


def _legacy_arms(arm_runs: dict[str, tuple[str, str]], *, context: InvocationContext) -> int:
    out_dir = context.reports / "timing"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {label: harvest_arm(context.runs / sub / rid, context.target) for rid, (sub, label) in arm_runs.items()}
    (out_dir / "timing_detailed.json").write_text(json.dumps(res, indent=2))
    for label, t in res.items():
        a = t["agent_session"]
        print(
            f"  {label:14s} active={t['active_wall_min']:>7} min  think={_fmt(a['think_generate_s'])} "
            f"tool={_fmt(a['tool_and_wait_s'])} ({a['method']})"
        )
    print(f"wrote {out_dir}/timing_detailed.json")
    return 0


def main(argv: list[str] | None = None, *, context: InvocationContext | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_context_arguments(ap)
    ap.add_argument(
        "--run-dir", action="append", default=[], help="a run directory (repeatable); prints its think/tool split"
    )
    ap.add_argument("--write", action="store_true", help="also write <run-dir>/timing_detailed.json")
    ap.add_argument("--arms", action="store_true", help="legacy cross-arm view (needs the experiment env)")
    ap.add_argument(
        "--arm", action="append", default=[], metavar="RUN_ID=SUBDIR:LABEL", help="arm to include in --arms"
    )
    args = ap.parse_args(argv)
    if args.run_dir:
        for d in args.run_dir:
            report_run(Path(d), write=args.write)
        return 0
    if args.arms:
        arms = {}
        for spec in args.arm:
            rid, _, rest = spec.partition("=")
            sub, _, label = rest.partition(":")
            arms[rid] = (sub, label or rid)
        if not arms:
            ap.error("--arms needs at least one --arm RUN_ID=SUBDIR:LABEL")
        return _legacy_arms(arms, context=resolve_context(args, ap, context))
    ap.error("give --run-dir DIR (or --arms with --arm specs)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
