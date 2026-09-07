"""Read-only Phase-2 receipt dashboard. This never launches or qualifies a candidate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any


def matching_processes(run_root: Path, source_snapshot: str | None, *, proc: Path = Path("/proc")) -> list[dict]:
    """Inspect argv privately; publish only exact matching launcher identity and elapsed time."""
    allowed_scripts = {str((Path(source_snapshot) / "merlin/experiments/gemmini_perf_bench/scripts/launch_global_agent_experiment.py").absolute())} if source_snapshot else set()
    expected_output = os.path.abspath(run_root)
    rows = []
    try:
        uptime = float((proc / "uptime").read_text().split()[0])
        ticks = os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError):
        return rows
    for entry in proc.iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
            if not argv or not allowed_scripts:
                continue
            # Interpret only launcher/output path candidates, never arbitrary
            # private arguments. Relative argv belongs to this process's cwd,
            # not the dashboard's; inaccessible cwd cannot establish a match.
            def process_path(value: str) -> str:
                path = Path(value)
                if not path.is_absolute():
                    cwd = (entry / "cwd").readlink()
                    if not cwd.is_absolute():
                        raise ValueError("process cwd is not absolute")
                    path = cwd / path
                return os.path.normpath(path)
            scripts = [token for token in argv
                       if token.rsplit("/", 1)[-1] == "launch_global_agent_experiment.py"]
            if not any(process_path(token) in allowed_scripts for token in scripts):
                continue
            outputs = [argv[i + 1] for i, token in enumerate(argv[:-1]) if token == "--output"]
            if not any(process_path(token) == expected_output for token in outputs):
                continue
            # stat comm may itself contain spaces/parentheses; fields after its final ')'.
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] == "Z":
                continue
            rows.append({"pid": int(entry.name), "state": fields[0],
                         "elapsed_wall_seconds": max(0, uptime - int(fields[19]) / ticks),
                         "match": "exact snapshot launcher and output argument"})
        except (OSError, ValueError, UnicodeError, IndexError):
            continue  # A process may exit between reads; absence is not a terminal verdict.
    return rows


def collect_status(run_root: Path, *, milestones: Path | None = None,
                   source_snapshot: Path | None = None,
                   process_reader=matching_processes) -> dict[str, Any]:
    run_root = run_root.resolve()
    evidence, warnings = [], []

    def read(path):
        if not path.is_file():
            return {}
        try:
            raw = path.read_bytes()
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("receipt is not an object")
            evidence.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
            return value
        except (OSError, ValueError) as exc:
            warnings.append({"path": str(path), "reason": type(exc).__name__})
            return {}

    launch = read(run_root / "launch.json")
    explicit_snapshot = str(source_snapshot.absolute()) if source_snapshot is not None else None
    if explicit_snapshot and launch.get("source_snapshot") not in (None, explicit_snapshot):
        raise ValueError("explicit source snapshot differs from the launch receipt")
    snapshot = launch.get("source_snapshot") or explicit_snapshot
    folder = run_root / "global_iterations"
    experiment = read(folder / "experiment.json")
    sequence = read(folder / "agent_sequence.json")
    terminal_failure = read(run_root / "terminal_failure.json")
    terminal_failure = terminal_failure if terminal_failure.get("schema") == "global_launch_terminal_failure_v1" else {}
    live = process_reader(run_root, snapshot)
    iterations = [read(path) for path in sorted(folder.glob("iteration_*.json"))]
    iterations = [row for row in iterations if row.get("schema") == "global_perf_iteration_v1"]
    latest = iterations[-1] if iterations else {}
    failure = latest.get("analysis", {}).get("failure") or {}
    failure_summary = {key: failure.get(key) for key in ("type", "reason")} if failure else None
    if failure:
        # Follow only a worker directory explicitly named by this exact iteration's
        # host error, not an arbitrary newest worker or a free path from candidate JSON.
        for path in sorted((run_root / "host_analysis_workers").glob("*/result.json")):
            if str(path.parent) in str(failure.get("reason", "")):
                worker_failure = read(path).get("failure")
                if isinstance(worker_failure, dict):
                    failure_summary = {key: worker_failure.get(key) for key in ("type", "reason")}
                    failure_summary["worker_receipt"] = str(path)
                    break
        failure_summary["reason"] = str(failure_summary.get("reason", ""))[:1500]
    audits = [read(path) for path in sorted(folder.glob("agent_round_*.json"))]
    failures = [read(path) for path in sorted(folder.glob("continuation_failure_*.json"))]
    continuations = [read(path) for path in sorted(folder.glob("continuation_*.json"))
                     if not path.name.startswith("continuation_failure_")]
    summaries = [read(path) for path in sorted((run_root / "rounds").glob("round_*.codex_summary.json"))]
    checkpoints = sequence.get("checkpoints") or [row["checkpoint"] for row in continuations if row.get("checkpoint")]
    if not checkpoints and (folder / "initial_seed_candidate.json").is_file():
        seed = read(folder / "initial_seed_candidate.json")
        checkpoints = [{"path": str(folder / "initial_seed_candidate.json"), "round": -1,
                        "candidate_sha256": seed.get("candidate_sha256"),
                        "role": "initial_verified_seed_not_an_authored_result"}]
    checkpoint = checkpoints[-1] if checkpoints else None
    retained = None
    if checkpoint:
        checkpoint_path = Path(checkpoint.get("path", ""))
        # Do not follow a candidate-supplied path outside this run's evidence directory.
        if checkpoint_path.parent.resolve() == folder.resolve():
            seal = read(checkpoint_path)
            if (not checkpoint.get("sha256") or not checkpoint_path.is_file()
                    or hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != checkpoint["sha256"]):
                warnings.append({"reason": "retained checkpoint receipt digest absent or mismatched"})
                seal = {}
            bound_path = Path(seal.get("iteration_record", ""))
            if bound_path.parent.resolve() == folder.resolve() and bound_path.is_file():
                raw = bound_path.read_bytes()
                if hashlib.sha256(raw).hexdigest() == seal.get("iteration_record_sha256"):
                    row = read(bound_path)
                    if row.get("candidate_sha256") == seal.get("candidate_sha256") == checkpoint.get("candidate_sha256"):
                        retained = row
            if retained is None:
                warnings.append({"reason": "retained checkpoint iteration binding unavailable or mismatched"})
    comparison = {"status": "UNKNOWN", "reason": "no bound retained checkpoint and seed analysis"}
    if retained and iterations:
        from merlin.perf.structural_delta import compare_full_model_structure
        comparison = compare_full_model_structure(iterations[0]["analysis"], retained["analysis"])
    measured = {"full_model_speedup": "UNPROVEN", "scope": "short mechanism receipts only; not model timing",
                "retained_probe_receipts": (retained or {}).get("probe_receipts", []),
                "retained_controlled_context_receipts": (retained or {}).get("context_receipts", []),
                "retained_paired_context_receipts": (retained or {}).get("paired_context_receipts", [])}
    terminal_status = sequence.get("status")
    if terminal_failure:
        state = "terminal_receipt:failed"
    elif terminal_status:
        state = f"terminal_receipt:{terminal_status}"
    elif live:
        state = "live_launcher_observed"
    elif failure_summary and not audits and latest.get("readiness", {}).get("status") == "blocked":
        state = "blocked_initial_analysis; launcher terminal receipt absent"
    else:
        state = "no_matching_live_launcher_observed; terminal status unknown"
    prepared = sorted((run_root / "agent_workspaces").glob("round_*"))
    phase1 = launch.get("phase1") or experiment.get("phase1_qualification") or {}
    notes = read(milestones.resolve()) if milestones else {}
    return {"schema": "phase2_read_only_status_v1", "refreshed_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root), "state": state, "live_processes": live,
        "objective": launch.get("objective", experiment.get("capsule", "UNKNOWN")),
        "source_snapshot": snapshot,
        "source_snapshot_origin": "launch_receipt" if launch.get("source_snapshot") else "explicit_host_startup_path" if snapshot else "UNKNOWN",
        "terminal_receipt_present": bool(terminal_status or terminal_failure),
        "launcher_terminal_failure": terminal_failure or None,
        "latest_analysis_failure": failure_summary,
        "source_snapshot_files_sha256": launch.get("source_snapshot_files_sha256"),
        "round": {"latest_prepared_workspace": prepared[-1].name if prepared else None,
                  "latest_terminal_audit": {key: audits[-1].get(key) for key in ("round", "status", "agent_exit_code")} if audits else None,
                  "note": "prepared workspace is not proof that authoring started",
                  "completed_audits": len(audits), "failed_continuations": len(failures)},
        "iterations": {"processed": len(iterations), "latest_index": latest.get("iteration"),
            "latest_candidate_sha256": latest.get("candidate_sha256"),
            "latest_readiness": latest.get("readiness", {}).get("status")},
        "retained_checkpoint": checkpoint, "retained_iteration_binding_verified": retained is not None,
        "budget": {"total_authoring_seconds": launch.get("total_authoring_seconds"),
            "round_seconds": launch.get("round_seconds"), "iteration_seconds": launch.get("iteration_seconds"),
            "authoring_seconds_reserved": sequence.get("authoring_seconds_reserved"),
            "completed_transport_wall_seconds": sum(row.get("wall_s", 0) for row in summaries),
            "current_authoring_elapsed_seconds": "UNKNOWN",
            "note": "launcher wall time includes setup; reserved budget is not measured authoring time"},
        "structural_seed_to_retained": comparison, "measured_evidence": measured,
        "development_gaps": {"full_model_numerical_qualification": "UNPROVEN",
            "full_model_speedup": "UNPROVEN", "phase1_passed": phase1.get("public_passed"),
            "phase1_total": phase1.get("public_total"), "frozen_phase1_gap_ids": phase1.get("known_functional_gap_ids", []),
            "latest_promotion_blockers": latest.get("readiness", {}).get("promotion_blockers", []),
            "checkpoint_scope": "authored/structural checkpoint is not semantic or performance promotion"},
        "milestones": notes, "evidence": evidence, "warnings": warnings}


def render_markdown(status: dict) -> str:
    row, budget = status["iterations"], status["budget"]
    lines = ["# Phase-2 progress", "", f"Updated: {status['refreshed_at']}", "",
        f"Run: `{status['run_root']}`", f"Objective: `{status['objective']}`", f"State: **{status['state']}**", "",
        f"Round: {status['round']['latest_prepared_workspace'] or 'not prepared'}; terminal audit: {status['round']['latest_terminal_audit']}",
        f"Processed iterations: {row['processed']}; latest readiness: {row['latest_readiness']}",
        f"Retained checkpoint: `{(status['retained_checkpoint'] or {}).get('candidate_sha256', 'none')}`", "",
        f"Budget: {budget['total_authoring_seconds']} s authoring; {budget['round_seconds']} s/round; {budget['iteration_seconds']} s/action.",
        f"Completed transport wall time: {budget['completed_transport_wall_seconds']:.1f} s. Current authoring elapsed: UNKNOWN.",
        "Launcher elapsed includes preparation; a launch record alone does not establish liveness.", "",
        "## Structural changes: seed → retained checkpoint", "",
        "Static work estimates only—not measured traffic or end-to-end speedup.", ""]
    if status.get("latest_analysis_failure"):
        lines[8:8] = ["Recorded analysis failure: " + status["latest_analysis_failure"]["reason"], ""]
    metrics = status["structural_seed_to_retained"].get("metrics", {})
    changed = [(name, value) for name, value in metrics.items() if value.get("delta") not in (None, 0)]
    if changed:
        lines += ["| Metric | Before | After | Delta |", "|---|---:|---:|---:|"]
        lines += [f"| {name} | {value.get('before')} | {value.get('after')} | {value.get('delta')} |" for name, value in changed]
    else:
        lines.append("No bound retained structural change available.")
    lines += ["", "Full-model speedup: **UNPROVEN**. Short probe results do not establish full-model timing.",
        "Authored checkpoints are not numerical/performance promotion. Frozen Phase-1 qualification is unchanged.", "",
        "## Development notes", "", "```json", json.dumps(status["milestones"], indent=2), "```", "",
        "Detailed gaps, exact receipt paths/hashes and scoped probe references: [status.json](status.json).", ""]
    return "\n".join(lines)


def write_dashboard(output: Path, status: dict) -> None:
    """Replace each complete document atomically; both carry the same refresh identity."""
    output.mkdir(parents=True, exist_ok=True)
    documents = {"status.json": json.dumps(status, indent=2, sort_keys=True) + "\n",
                 "STATUS.md": render_markdown(status)}
    for name, contents in documents.items():
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output,
                                         prefix=f".{name}.", delete=False) as temporary:
            temporary.write(contents)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary.name, output / name)


def refresh_loop(refresh, *, watch_seconds: float = 0, interval_seconds: float = 30,
                 monotonic=time.monotonic, sleep=time.sleep) -> None:
    if not 0 <= watch_seconds <= 3600 or not 1 <= interval_seconds <= 60:
        raise ValueError("watch must be 0..3600 seconds; interval must be 1..60 seconds")
    deadline = monotonic() + watch_seconds
    first = True
    while True:
        if not first and monotonic() >= deadline:
            return
        first = False
        status = refresh()
        if (watch_seconds == 0 or
                (status["terminal_receipt_present"] and not status["live_processes"])):
            return
        remaining = deadline - monotonic()
        if remaining <= 0:
            return
        sleep(min(interval_seconds, remaining))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--milestones", type=Path, help="Separate host-authored JSON notes; never a qualification receipt")
    parser.add_argument("--source-snapshot", type=Path, help="Exact host-known startup source path; must match launch.json when present")
    parser.add_argument("--watch-seconds", type=float, default=0, help="Bounded refresh duration, maximum 3600; default one shot")
    parser.add_argument("--interval-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.output.resolve() == args.run_root.resolve() or args.run_root.resolve() in args.output.resolve().parents:
        parser.error("dashboard output must be outside the immutable run directory")
    def refresh():
        status = collect_status(args.run_root, milestones=args.milestones, source_snapshot=args.source_snapshot)
        write_dashboard(args.output, status)
        return status
    refresh_loop(refresh, watch_seconds=args.watch_seconds, interval_seconds=args.interval_seconds)
    print(args.output / "STATUS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
