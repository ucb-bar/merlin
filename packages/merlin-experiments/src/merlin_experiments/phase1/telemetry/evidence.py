"""Retained process evidence, exact stream reconciliation and resource observations."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from merlin_experiments.phase1.telemetry import timeline


def _role(path: Path) -> str:
    name = path.name
    if "agent_evidence_snapshot" in str(path):
        return "authoritative_agent_visible_broker_or_selfcheck_evidence"
    if "codex_events.raw" in name:
        return "authoritative_provider_cli_stream"
    if "codex_events.timestamped" in name:
        return "arrival_stamped_provider_cli_stream"
    if "codex_rollout" in str(path):
        return "authoritative_codex_rollout_full_io_and_incremental_usage"
    if name.endswith(".prompt.txt"):
        return "exact_agent_input_prompt"
    if name.endswith(".final.txt"):
        return "exact_agent_final_output"
    if name.endswith(".transcript.jsonl"):
        return "normalized_transcript_outputs_may_be_clipped"
    if name.endswith(".codex_summary.json"):
        return "driver_session_summary"
    if name.endswith(".stage_ledger.json"):
        return "grader_stage_artifact_ledger"
    if name.endswith("stderr.log"):
        return "driver_stderr"
    if name.endswith(".sh"):
        return "sandbox_launch_script"
    return "run_telemetry_support"


def artifact_inventory(run_dir: Path) -> dict:
    """SHA/size inventory of every process-telemetry artifact (not the bulky simulator corpus)."""
    run_dir = Path(run_dir)
    candidates = []
    rounds = run_dir / "rounds"
    if rounds.is_dir():
        for path in sorted(rounds.rglob("*")):
            if path.is_file():
                candidates.append(path)
    evidence = run_dir / "agent_evidence_snapshot"
    if evidence.is_dir():
        candidates.extend(path for path in sorted(evidence.rglob("*")) if path.is_file())
    for name in (
        "transcript.jsonl",
        "environment.yaml",
        "cost_time_toolcalls.yaml",
        "qa_loop_state.yaml",
        "qa_loop_summary.yaml",
    ):
        path = run_dir / name
        if path.is_file():
            candidates.append(path)
    files = []
    for path in candidates:
        try:
            data = path.read_bytes()
        except OSError:
            continue
        files.append(
            {
                "path": str(path.relative_to(run_dir)),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "role": _role(path),
            }
        )
    return {
        "files": files,
        "authoritative_io": [
            row["path"]
            for row in files
            if row["role"].startswith("authoritative") or row["role"].startswith("exact_agent")
        ],
        "note": (
            "raw provider/rollout files are authoritative for full tool I/O; normalized transcripts "
            "are an analysis view and may clip large outputs"
        ),
    }


def _repo_from_run(run_dir: Path) -> Path | None:
    for parent in (run_dir, *run_dir.parents):
        if (parent / "out" / "artifacts" / "cache" / "codex_home").is_dir():
            return parent
    return None


def snapshot_codex_rollouts(run_dir: Path) -> list[Path]:
    """Copy the isolated Codex rollout logs into the durable run before cache cleanup.

    These logs are the only artifact containing incremental response usage plus full apply-patch bodies;
    the CLI's top-level event stream intentionally summarizes file changes.  No auth/state database is
    copied: only session JSONL.
    """
    run_dir = Path(run_dir)
    repo = _repo_from_run(run_dir)
    if repo is None:
        return []
    cache = repo / "out" / "artifacts" / "cache" / "codex_home"
    copied = []
    for home in sorted(cache.glob(f"{run_dir.name}_r*")):
        suffix = home.name.rsplit("_r", 1)[-1]
        dest = run_dir / "rounds" / f"round_{suffix}.codex_rollout_snapshot"
        for source in sorted((home / "sessions").rglob("*.jsonl")):
            relative = source.relative_to(home / "sessions")
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(target)
    return copied


def snapshot_agent_evidence(run_dir: Path) -> list[Path]:
    """Seal agent-visible broker/self-check evidence that otherwise lives in a mutable workspace."""
    run_dir = Path(run_dir)
    repo = _repo_from_run(run_dir)
    copied = []
    from merlin_experiments.phase1.workspaces import workspace_candidates

    # The archived pointer also works with relocated MERLIN_OUT_ROOT, where no repository ancestor
    # or Codex cache exists. Use only the first surviving workspace to avoid mixing run identities.
    workspaces = [workspace for workspace in workspace_candidates(run_dir, repo=repo) if workspace.is_dir()][:1]
    dest_root = run_dir / "agent_evidence_snapshot"
    for workspace in workspaces:
        for name in (".qa_channel", "selfcheck_out"):
            source = workspace / name
            if not source.exists():
                continue
            dest = dest_root / name
            if source.is_dir():
                shutil.copytree(source, dest, dirs_exist_ok=True)
                copied.extend(path for path in dest.rglob("*") if path.is_file())
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
                copied.append(dest)
    return copied


def rollout_telemetry(run_dir: Path) -> dict:
    """Incremental Codex response usage and client-observed response intervals.

    Codex records a completion timestamp and per-response usage, but no provider request-start/TTFT or
    token-delta stream.  The interval begins at the preceding task-start/tool-result observed by the
    client.  It is deliberately called turnaround, never API latency or decode throughput.
    """
    paths = sorted((Path(run_dir) / "rounds").glob("round_*.codex_rollout_snapshot/**/*.jsonl"))
    responses = []
    total = {"input_total": 0, "fresh_input": 0, "cache_read": 0, "cache_write": 0, "output": 0, "reasoning": 0}
    for path in paths:
        pending_start = None
        try:
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
                if line.startswith("{")
            ]
        except (OSError, ValueError):
            continue
        for record in records:
            stamp = timeline._stamp(record.get("timestamp"))
            kind = record.get("type")
            payload = record.get("payload") or {}
            payload_type = payload.get("type") if isinstance(payload, dict) else None
            if kind == "event_msg" and payload_type == "task_started" and stamp is not None:
                pending_start = stamp
            elif (
                kind == "response_item"
                and payload_type in {"custom_tool_call_output", "function_call_output"}
                and stamp is not None
            ):
                pending_start = stamp
            elif kind == "token_usage_record" and isinstance(payload, dict):
                usage = payload.get("usage") or {}
                if not isinstance(usage, dict):
                    continue
                input_total = int(usage.get("input_tokens", 0) or 0)
                cache_read = int(usage.get("cached_input_tokens", 0) or 0)
                cache_write = int(usage.get("cache_write_input_tokens", 0) or 0)
                output = int(usage.get("output_tokens", 0) or 0)
                reasoning = int(usage.get("reasoning_output_tokens", 0) or 0)
                fresh = max(input_total - cache_read - cache_write, 0)
                elapsed = max(stamp - pending_start, 0.0) if stamp is not None and pending_start is not None else None
                row = {
                    "source": str(path.relative_to(run_dir)),
                    "response_id": payload.get("response_id"),
                    "turn_id": payload.get("turn_id"),
                    "completed_at": record.get("timestamp"),
                    "client_observed_turnaround_s": round(elapsed, 6) if elapsed is not None else None,
                    "tokens": {
                        "input_total": input_total,
                        "fresh_input": fresh,
                        "cache_read": cache_read,
                        "cache_write": cache_write,
                        "output": output,
                        "reasoning": reasoning,
                    },
                    "output_tokens_per_client_observed_turnaround_s": (
                        output / elapsed if elapsed and elapsed > 0 else None
                    ),
                }
                responses.append(row)
                for key, value in row["tokens"].items():
                    total[key] += value
    turnaround = [
        row["client_observed_turnaround_s"] for row in responses if row["client_observed_turnaround_s"] is not None
    ]
    input_traffic = total["fresh_input"] + total["cache_read"] + total["cache_write"]
    response_wall = sum(turnaround) if turnaround else None
    return {
        "available": bool(responses),
        "rollout_files": [str(path.relative_to(run_dir)) for path in paths],
        "responses": responses,
        "tokens": {
            "tokens_input_total": total["input_total"],
            "tokens_fresh_input": total["fresh_input"],
            "tokens_cache_read": total["cache_read"],
            "tokens_cache_write": total["cache_write"],
            "tokens_output": total["output"],
            "tokens_reasoning": total["reasoning"],
            "tokens_total": total["input_total"] + total["output"],
            "cache_read_share_of_input": (total["cache_read"] / input_traffic if input_traffic else None),
            "cache_write_share_of_input": (total["cache_write"] / input_traffic if input_traffic else None),
            "reasoning_is_subset_of_output": True,
        }
        if responses
        else {},
        "client_observed_turnaround_s": {
            "sum": round(sum(turnaround), 6) if turnaround else None,
            "mean": round(sum(turnaround) / len(turnaround), 6) if turnaround else None,
            "p50": timeline._percentile(turnaround, 0.50),
            "p95": timeline._percentile(turnaround, 0.95),
            "max": max(turnaround) if turnaround else None,
        },
        "token_rates": {
            "output_tokens_per_client_observed_turnaround_s": (
                total["output"] / response_wall if response_wall else None
            ),
            "all_provider_tokens_per_client_observed_turnaround_s": (
                (total["input_total"] + total["output"]) / response_wall if response_wall else None
            ),
            "denominator_s": round(response_wall, 6) if response_wall else None,
            "note": "client-observed response turnaround; not server decode throughput",
        },
        "latency_limit": (
            "Codex exposes response-completion timestamps but not request-start, TTFT, or "
            "token-delta timestamps. Turnaround is client-observed from the prior task/tool "
            "input; it is not server latency, TTFT, or true decode tokens/s."
        ),
    }


def stream_reconciliation(run_dir: Path) -> dict:
    """Prove that the arrival-stamped mirror contains every raw Codex event in the same order."""
    run_dir = Path(run_dir)
    rounds = run_dir / "rounds"
    rows, failures = [], []
    for raw_path in sorted(rounds.glob("round_*.codex_events.raw.jsonl")):
        stem = raw_path.name.replace(".codex_events.raw.jsonl", "")
        stamped_path = rounds / f"{stem}.codex_events.timestamped.jsonl"
        try:
            raw_bytes = raw_path.read_bytes()
            stamped_bytes = stamped_path.read_bytes()
            raw_lines = raw_bytes.splitlines()
            stamped_lines = stamped_bytes.splitlines()
            raw_events = [json.loads(line) for line in raw_lines]
            stamped = [json.loads(line) for line in stamped_lines]
            events_equal = len(raw_events) == len(stamped) and all(
                row.get("event") == event for row, event in zip(stamped, raw_events)
            )
            seq_contiguous = [row.get("seq") for row in stamped] == list(range(1, len(stamped) + 1))
            newline_terminated = raw_bytes.endswith(b"\n") and stamped_bytes.endswith(b"\n")
            row_failures = []
            if not events_equal:
                row_failures.append("raw_and_stamped_events_differ")
            if not seq_contiguous:
                row_failures.append("stamped_sequence_not_contiguous")
            if not newline_terminated:
                row_failures.append("stream_not_newline_terminated")
            rows.append(
                {
                    "round": stem,
                    "raw_events": len(raw_events),
                    "stamped_events": len(stamped),
                    "events_equal": events_equal,
                    "seq_contiguous": seq_contiguous,
                    "newline_terminated": newline_terminated,
                    "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                    "stamped_sha256": hashlib.sha256(stamped_bytes).hexdigest(),
                    "failures": row_failures,
                }
            )
            failures.extend(f"{stem}:{failure}" for failure in row_failures)
        except Exception as exc:  # noqa: BLE001 — corruption is an integrity result, not a crash
            failure = f"{stem}:reconciliation_failed:{type(exc).__name__}"
            failures.append(failure)
            rows.append({"round": stem, "failures": [failure]})
    if not rows:
        failures.append("no_raw_stream_to_reconcile")
    return {"complete": not failures, "failures": failures, "rounds": rows}


def resource_telemetry(run_dir: Path) -> dict:
    """Summarize retained procfs samples while keeping each JSONL as the source of truth."""
    paths = sorted((Path(run_dir) / "rounds").glob("round_*.resource_samples.jsonl"))
    paths += sorted((Path(run_dir) / "rounds").glob("round_*.live_resource_samples.jsonl"))
    records = []
    for path in paths:
        try:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("{")]
        except (OSError, ValueError):
            rows = []
        stamps = [timeline._stamp(row.get("sampled_at")) for row in rows]
        stamps = [stamp for stamp in stamps if stamp is not None]
        records.append(
            {
                "source": str(path.relative_to(run_dir)),
                "samples": len(rows),
                "observed_span_s": (round(max(stamps) - min(stamps), 6) if len(stamps) >= 2 else 0.0),
                "rss_bytes_peak": max((int(row.get("rss_bytes", 0)) for row in rows), default=None),
                "rss_bytes_mean": (
                    round(sum(int(row.get("rss_bytes", 0)) for row in rows) / len(rows), 3) if rows else None
                ),
                "virtual_bytes_peak": max((int(row.get("virtual_bytes", 0)) for row in rows), default=None),
                "processes_peak": max((int(row.get("processes", 0)) for row in rows), default=None),
                "threads_peak": max((int(row.get("threads", 0)) for row in rows), default=None),
                "observed_cpu_seconds_peak": max(
                    (float(row.get("user_cpu_s", 0)) + float(row.get("system_cpu_s", 0)) for row in rows), default=None
                ),
                "observed_read_bytes_peak": max((int(row.get("read_bytes", 0)) for row in rows), default=None),
                "observed_write_bytes_peak": max((int(row.get("write_bytes", 0)) for row in rows), default=None),
            }
        )
    return {
        "available": any(row["samples"] for row in records),
        "streams": records,
        "sampling_note": (
            "5 s procfs snapshots of the live descendant tree. Peak cumulative CPU/I/O "
            "is a lower bound because a short-lived child may exit between samples; raw "
            "samples are retained for alternate analyses."
        ),
    }
