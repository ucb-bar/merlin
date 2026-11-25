"""Explain complete-model checks using their existing routed-call profiles.

This reader does not run models or grant numerical, placement or hardware
qualification. A call interval includes its ABI bridge and implementation.
Gaps include host work and profiling bookkeeping, not just dispatch overhead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "merlin.model_bottlenecks.v1"
COUNTERS = {"raw_counter_ticks", "spike_retired_instruction_proxy"}


def _integer(value, name):
    if type(value) is not int or not 0 <= value < 1 << 64:
        raise ValueError(f"{name} must be a nonnegative 64-bit integer")
    return value


def _profile(row):
    """Reconcile an existing full-build summary or a complete group trace."""
    profile = row.get("group_profile")
    if profile is None:
        return None
    if not isinstance(profile, dict):
        raise ValueError("group profile must be an object")
    ticks = _integer(row.get("cycles_functional", row.get("cycles")), "forward counter")
    if "cycles_functional" in row and "cycles" in row and row["cycles"] != ticks:
        raise ValueError("forward counter aliases disagree")
    dropped = _integer(profile.get("dropped"), "dropped call count")
    calls = _integer(profile.get("calls"), "call count")
    groups = profile.get("groups")
    details = []
    if groups is not None:
        if not isinstance(groups, list) or len(groups) != calls:
            raise ValueError("group detail roster differs from call count")
        for index, group in enumerate(groups):
            if _integer(group.get("index"), "call index") != index or not isinstance(group.get("name"), str):
                raise ValueError("group detail indices or names are incomplete")
            details.append({"index": index, "name": group["name"],
                            "ticks": _integer(group.get("cycles"), "call interval"),
                            "gap_before": _integer(group.get("gap_before"), "preceding gap")})
        group_ticks = sum(group["ticks"] for group in details)
        gap_ticks = sum(group["gap_before"] for group in details) + _integer(profile.get("tail_gap"), "tail gap")
        for field, total in (("group_cycles", group_ticks), ("gap_cycles", gap_ticks)):
            if field in profile and _integer(profile[field], field) != total:
                raise ValueError("profile summary disagrees with detailed intervals")
    else:
        group_ticks = _integer(profile.get("group_cycles"), "routed-call intervals")
        gap_ticks = _integer(profile.get("gap_cycles"), "between-call intervals")
    if group_ticks + gap_ticks != ticks:
        raise ValueError("routed-call and gap intervals do not reconcile with the forward")
    if calls == 0 and group_ticks:
        raise ValueError("routed-call intervals exist without a recorded call")
    return {"forward_ticks": ticks, "call_ticks": group_ticks, "gap_ticks": gap_ticks,
            "calls": calls, "dropped": dropped, "complete": dropped == 0, "groups": details}


def _failure_details(report):
    result = []
    for program in report.get("programs", []):
        for kernel in program.get("kernels", []):
            if kernel.get("status") not in {"pass", "passed", "built", "exact", "ok"} and kernel.get("detail"):
                detail = str(kernel["detail"])
                dependency = "z3" in detail.lower() and any(word in detail.lower() for word in ("unavailable", "modulenotfound"))
                result.append({"kind": "proof_runtime_unavailable" if dependency else "kernel_not_completed",
                               "program": program.get("program"), "symbol": kernel.get("symbol"),
                               "form": kernel.get("form"), "detail": detail,
                               "numerical_failure_established": False})
    for failure in report.get("failures", []):
        detail = str(failure.get("detail") or failure.get("message") or "")
        if detail and not any(row["detail"] == detail for row in result):
            dependency = "z3" in detail.lower() and any(word in detail.lower() for word in ("unavailable", "modulenotfound"))
            result.append({"kind": "proof_runtime_unavailable" if dependency else "check_not_completed",
                           "detail": detail, "numerical_failure_established": False})
    return result


def analyze(document, *, expected_steps, input_id, counter_kind):
    """Project one selected input without merging runs, counters or numerical tracks.

    ``expected_steps`` comes from the selected models configuration. Counters are
    explicitly raw ticks or Spike's instruction proxy. Physical timing remains
    the measurement controller's responsibility, never a caller-supplied flag.
    """
    if counter_kind not in COUNTERS or not isinstance(input_id, str) or not input_id:
        raise ValueError("select an explicit diagnostic counter and input")
    if not expected_steps or any(not isinstance(k, str) or not k or type(v) is not int or v <= 0
                                 for k, v in expected_steps.items()):
        raise ValueError("select the complete positive program/step roster")
    report = document.get("report", document)
    if report.get("schema") != "merlin.model_check.report.v1":
        raise ValueError("select an ordinary model_check report")
    expected = {(program, step) for program, count in expected_steps.items() for step in range(count)}
    seen, executed, intervals, program_names = set(), set(), [], set()
    for program in report.get("programs", []):
        name = program.get("program")
        if name not in expected_steps or name in program_names:
            raise ValueError("unexpected or duplicate model program")
        program_names.add(name)
        for row in program.get("runs", []):
            if row.get("input") != input_id:
                continue
            step = row.get("step")
            if "program" in row and row["program"] != name:
                raise ValueError("nested run belongs to a different program")
            if type(step) is not int or (name, step) not in expected or (name, step) in seen:
                raise ValueError("unexpected or duplicate recurrent step")
            seen.add((name, step))
            if row.get("status") != "completed":
                continue
            executed.add((name, step))
            profile = _profile(row)
            if profile is not None:
                intervals.append({"program": name, "step": step, **profile})
    complete = len(intervals) == len(expected) and all(row["complete"] for row in intervals)
    forward = sum(row["forward_ticks"] for row in intervals)
    calls = sum(row["call_ticks"] for row in intervals)
    gaps = sum(row["gap_ticks"] for row in intervals)
    dropped = sum(row["dropped"] for row in intervals)
    breakdown = sorted([
        {"region": "routed_calls", "ticks": calls,
         "meaning": ("Recorded routed calls only, a lower bound because calls were dropped" if dropped else
                     "ABI bridge and emitted kernel, including any internal preparation and waits")},
        {"region": "unattributed" if dropped else "between_calls", "ticks": gaps,
         "meaning": ("Unattributed interval including dropped routed calls" if dropped else
                     "Host work and profiler bookkeeping between routed calls")},
    ], key=lambda row: (-row["ticks"], row["region"])) if intervals else []
    blockers = _failure_details(report)
    priorities = []
    if blockers:
        proof_unavailable = any(row["kind"] == "proof_runtime_unavailable" for row in blockers)
        priorities.append({"kind": "restore_build_prerequisites" if proof_unavailable else "inspect_failed_check",
                           "basis": "recorded check or build refusal",
                           "next_check": ("replay the same interface with the required proof runtime visible" if proof_unavailable else
                                          "inspect the retained failure and distinguish unsupported lowering, proof limits and numerical mismatches")})
    if not complete:
        priorities.append({"kind": "complete_execution_profile", "basis": "missing or dropped forward intervals",
                           "next_check": "run every declared program and recurrent step with complete profiling"})
    if complete and forward:
        dominant = breakdown[0]["region"]
        priorities.append({"kind": "inspect_" + dominant, "basis": "largest reconciled diagnostic interval",
                           "next_check": "join these intervals to operation and emitted-code evidence before selecting a fix"})
    return {
        "schema": SCHEMA, "model": report.get("model"), "input": input_id,
        "source_sha256": document.get("source_sha256"), "submission": report.get("submission"),
        "status": "profile_observed" if complete else "blocked" if blockers else "incomplete",
        "counter_kind": counter_kind, "physical_timing_qualified": False,
        "functional_gates": report.get("gates", {}), "all_declared_outputs_verified": False,
        "model_quality": "not_established_by_this_report", "placement": "not_established_by_this_report",
        "expected_steps": len(expected), "recorded_steps": len(seen), "profiled_steps": len(intervals),
        "missing_steps": [{"program": p, "step": s} for p, s in sorted(expected - seen)],
        "missing_execution_steps": [{"program": p, "step": s} for p, s in sorted(expected - executed)],
        "missing_profile_steps": [{"program": p, "step": s} for p, s in sorted(expected - {
            (row["program"], row["step"]) for row in intervals})],
        "profile_complete": complete, "intervals": intervals,
        "forward_sum_ticks": forward if intervals else None, "breakdown": breakdown,
        "ranked_calls": sorted([{"program": row["program"], "step": row["step"], **group}
                                for row in intervals for group in row["groups"]],
                               key=lambda row: (-row["ticks"], row["program"], row["step"], row["index"])),
        "blockers": blockers, "next_actions": priorities,
        "limits": ["Forward sums exclude inter-step transfers and do not establish complete session latency.",
                   "Routed-call time is not pure accelerator time. Gaps are not pure dispatch overhead.",
                   "Call counts and static command differences do not establish traffic or avoidable latency.",
                   "No optimization saving or causal explanation is inferred from these intervals."],
    }


def feedback(report):
    """Bounded agent-facing scalars only, with no reference tensors or private paths."""
    return {"schema": "merlin.model_bottlenecks.feedback.v1", "status": report["status"],
            "counter_kind": report["counter_kind"], "physical_timing_qualified": False,
            "profile_complete": report["profile_complete"],
            "steps": {key: report[key] for key in ("expected_steps", "recorded_steps", "profiled_steps")},
            "breakdown": report["breakdown"], "next_actions": report["next_actions"],
            "limits": report["limits"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--models-sha256", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--counter-kind", choices=sorted(COUNTERS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    from merlin.model_check.models import load

    raw, config = args.report.read_bytes(), args.models.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.report_sha256 or hashlib.sha256(config).hexdigest() != args.models_sha256:
        raise ValueError("selected report or models bytes differ from their pins")
    model = load(args.models, source_bytes=config).model(args.model)
    result = analyze(json.loads(raw), expected_steps={p.name: p.steps for p in model.programs},
                     input_id=args.input, counter_kind=args.counter_kind)
    if result["model"] != args.model:
        raise ValueError("report belongs to another selected model")
    result["inputs"] = {"report": {"path": str(args.report.resolve()), "sha256": args.report_sha256},
                        "models": {"path": str(args.models.resolve()), "sha256": args.models_sha256}}
    if args.report.read_bytes() != raw or args.models.read_bytes() != config:
        raise ValueError("selected inputs changed during analysis")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "bottlenecks.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.out / "agent-feedback.json").write_text(json.dumps(feedback(result), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
