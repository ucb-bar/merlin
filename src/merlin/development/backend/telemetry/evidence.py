"""Retained process evidence, exact stream reconciliation and resource observations."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import threading
from collections.abc import Mapping
from pathlib import Path

from merlin.development.backend.session.inputs.implementation_inventory import fingerprint
from merlin.development.backend.telemetry import timeline
from merlin.development.targetgen.authoring.semantic_search import SearchLimits, search_linalg_inventory
from merlin.development.targetgen.corpus.capsule_io import discover_capsules
from merlin.development.targetgen.sandbox import grants as BW
from merlin.target.contract.interfaces.linalg import is_linalg_on_tensors, parse_linalg_mlir
from merlin.target.spec.instructions import validate_normalized_instruction_model

# ---- evidence ----------------------------------------------------------------------------------


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
    copied = []
    # Retain explicitly matched custom-home evidence before its lease ends.
    # Usage discovery already checked raw thread identity and exact workspace.
    for actor in declared_home_usage(run_dir)["threads"]:
        source = Path(actor["source"])
        if source.is_relative_to(run_dir):
            continue  # Already a retained snapshot; never copy onto itself.
        session = next((parent for parent in source.parents if parent.name == "sessions"), None)
        if session is None:
            continue
        suffix = session.parent.name.removeprefix(f"{run_dir.name}_r")
        if not suffix.isdigit() or int(suffix) not in actor["rounds"]:
            continue
        target = run_dir / "rounds" / f"round_{suffix}.codex_rollout_snapshot" / source.relative_to(session)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(target)
    copied.extend(_snapshot_thread_rollouts(run_dir, already=set(copied)))
    repo = _repo_from_run(run_dir)
    if repo is None:
        return copied
    cache = repo / "out" / "artifacts" / "cache" / "codex_home"
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


def _snapshot_thread_rollouts(run_dir: Path, *, already: set[Path]) -> list[Path]:
    """Retain the rollout of every lead thread and every sub-agent descended from one.

    A helper's file carries two ``session_meta`` rows (its own, then its parent's copied history), so
    the exact-identity usage selection above never retains it; a lead without a usage sample is not
    retained there either. Lineage comes from each thread's own metadata. A failure here is reported
    and never fails the evidence seal.
    """
    from merlin.development.backend.telemetry import author_agents

    copied = []
    try:
        records = author_agents.threads(run_dir)["records"].values()
    except Exception as exc:  # noqa: BLE001 -- accounting evidence must not gate the telemetry seal
        print(f"[telemetry] thread rollouts not retained: {type(exc).__name__}: {exc}", file=sys.stderr)
        return copied
    for record in records:
        source = Path(record["rollout"]["path"])
        if source.is_relative_to(run_dir):
            continue  # Already a retained snapshot; never copy onto itself.
        session = next((parent for parent in source.parents if parent.name == "sessions"), None)
        suffix = session.parent.name.removeprefix(f"{run_dir.name}_r") if session is not None else ""
        if not suffix.isdigit():
            continue
        target = run_dir / "rounds" / f"round_{suffix}.codex_rollout_snapshot" / source.relative_to(session)
        if target in already:
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        except OSError as exc:
            print(f"[telemetry] rollout {source.name} not retained: {exc}", file=sys.stderr)
            continue
        copied.append(target)
    return copied


def snapshot_agent_evidence(run_dir: Path) -> list[Path]:
    """Seal agent-visible broker/self-check evidence that otherwise lives in a mutable workspace."""
    run_dir = Path(run_dir)
    repo = _repo_from_run(run_dir)
    copied = []
    from merlin.development.backend.session.workspaces import workspace_candidates

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


def _json_lines(path: Path):
    """Read complete provider records; a live trailing partial row is unavailable."""
    try:
        lines = path.read_bytes().splitlines(keepends=True)
    except OSError:
        return []
    records = []
    for line in lines:
        if not line.endswith(b"\n"):
            continue
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError):
            continue
        if isinstance(row, dict):
            records.append(row)
    return records


def declared_home_usage(run_dir: Path) -> dict:
    """Exact run-declared homes, raw thread identity, and cumulative live usage.

    No global cache search or account/balance interpretation. Unmatched actors
    are excluded rather than silently attributed to this experiment.
    """
    import yaml

    run_dir = Path(run_dir)
    unavailable = {"available": False, "threads": [], "tokens": {}, "cost_usd": None}
    try:
        env = yaml.safe_load((run_dir / "environment.yaml").read_text())
    except (OSError, ValueError, yaml.YAMLError):
        return unavailable
    if not isinstance(env, dict) or env.get("run_id", run_dir.name) != run_dir.name:
        return unavailable
    declared = [
        (env.get("codex_runtime_policy") or {}).get("home_root"),
        ((env.get("run_config") or {}).get("codex_runtime") or {}).get("home_root"),
    ]
    declared = [v for v in declared if v is not None]
    workspace = env.get("workspace_path")
    if (
        not declared
        or any(not isinstance(v, str) for v in declared)
        or len(set(declared)) != 1
        or not isinstance(workspace, str)
    ):
        return unavailable
    home = Path(declared[0])
    if not home.is_absolute() or not Path(workspace).is_absolute():
        return unavailable
    selected = {}
    for raw in sorted((run_dir / "rounds").glob("round_*.codex_events.raw.jsonl")):
        suffix = raw.name.split(".")[0].removeprefix("round_")
        if not suffix.isdigit():
            continue
        ids = {row.get("thread_id") for row in _json_lines(raw) if row.get("type") == "thread.started"}
        ids = {v for v in ids if isinstance(v, str) and v}
        files = list((home / f"{run_dir.name}_r{suffix}" / "sessions").rglob("*.jsonl"))
        files += list((run_dir / "rounds" / f"round_{suffix}.codex_rollout_snapshot").rglob("*.jsonl"))
        for path in sorted(files):
            records = _json_lines(path)
            metas = [row.get("payload") for row in records if row.get("type") == "session_meta"]
            if len(metas) != 1 or not isinstance(metas[0], dict):
                continue
            meta = metas[0]
            actor = meta.get("id")
            if actor not in ids or meta.get("cwd") != workspace:
                continue
            start = timeline._stamp(records[0].get("timestamp")) if records else None
            seen = set()
            latest = None
            for row in records:
                payload = row.get("payload") or {}
                if (
                    row.get("type") != "event_msg"
                    or not isinstance(payload, dict)
                    or payload.get("type") != "token_count"
                ):
                    continue
                info = payload.get("info") or {}
                usage = info.get("total_token_usage") if isinstance(info, dict) else None
                if not isinstance(usage, dict):
                    continue
                names = (
                    "input_tokens",
                    "cached_input_tokens",
                    "output_tokens",
                    "reasoning_output_tokens",
                    "total_tokens",
                )
                if any(type(usage.get(k)) is not int or usage[k] < 0 for k in names):
                    continue
                write = usage.get("cache_write_input_tokens", 0)
                if type(write) is not int or write < 0:
                    continue
                fingerprint = tuple(usage[k] for k in names) + (write,)
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                if (
                    usage["cached_input_tokens"] + write > usage["input_tokens"]
                    or usage["reasoning_output_tokens"] > usage["output_tokens"]
                    or usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]
                ):
                    continue
                stamp = timeline._stamp(row.get("timestamp"))
                if stamp is None:
                    continue
                latest = {"stamp": stamp, "at": row.get("timestamp"), "usage": usage, "write": write}
            if latest is None:
                continue
            prior = selected.get(actor)
            if prior and prior["stamp"] > latest["stamp"]:
                prior["rounds"].add(int(suffix))
                continue
            selected[actor] = {**latest, "start": start, "source": str(path), "rounds": {int(suffix)}}
            if prior:
                selected[actor]["rounds"].update(prior["rounds"])
                starts = [v for v in (start, prior["start"]) if v is not None]
                selected[actor]["start"] = min(starts) if starts else None
    threads = []
    totals = {k: 0 for k in ("input_total", "fresh_input", "cache_read", "cache_write", "output", "reasoning", "total")}
    for actor, item in sorted(selected.items()):
        usage = item["usage"]
        buckets = {
            "input_total": usage["input_tokens"],
            "fresh_input": usage["input_tokens"] - usage["cached_input_tokens"] - item["write"],
            "cache_read": usage["cached_input_tokens"],
            "cache_write": item["write"],
            "output": usage["output_tokens"],
            "reasoning": usage["reasoning_output_tokens"],
            "total": usage["total_tokens"],
        }
        span = item["stamp"] - item["start"] if item["start"] is not None else None
        span = span if span is not None and span > 0 else None
        threads.append(
            {
                "thread_id": actor,
                "rounds": sorted(item["rounds"]),
                "source": item["source"],
                "sampled_through": item["at"],
                "tokens": buckets,
                "usage_sample_span_s": span,
                "cost_usd": None,
                "input_tokens_per_client_observed_s": buckets["input_total"] / span if span else None,
                "output_tokens_per_client_observed_s": buckets["output"] / span if span else None,
                "reasoning_share_of_output": buckets["reasoning"] / buckets["output"] if buckets["output"] else None,
            }
        )
        for key in totals:
            totals[key] += buckets[key]
    return {
        "available": bool(threads),
        "threads": threads,
        "cost_usd": None,
        "tokens": {
            "tokens_input_total": totals["input_total"],
            "tokens_input_traffic": totals["input_total"],
            "tokens_fresh_input": totals["fresh_input"],
            "tokens_cache_read": totals["cache_read"],
            "tokens_cache_write": totals["cache_write"],
            "tokens_output": totals["output"],
            "tokens_reasoning": totals["reasoning"],
            "tokens_total": totals["total"],
            "reasoning_is_subset_of_output": True,
        }
        if threads
        else {},
        "usage_complete": False,
        "scope": "latest cumulative provider usage per exactly matched thread; volatile observation, not billing",
        "rate_note": "client-observed usage-sample traffic rates; not provider decode throughput or TTFT",
        "unattributed_agents": "excluded; no subagent attribution without an exact observed thread identity",
    }


def rollout_telemetry(run_dir: Path) -> dict:
    """Incremental Codex response usage and client-observed response intervals.

    Codex records a completion timestamp and per-response usage, but no provider request-start/TTFT or
    token-delta stream.  The interval begins at the preceding task-start/tool-result observed by the
    client.  It is deliberately called turnaround, never API latency or decode throughput.
    """
    live = declared_home_usage(run_dir)
    if live["available"]:
        return live
    paths = sorted((Path(run_dir) / "rounds").glob("round_*.codex_rollout_snapshot/**/*.jsonl"))
    responses = []
    seen_responses: set = set()
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
                # A forked sub-agent rollout may repeat a parent row; one response is counted once.
                response_id = payload.get("response_id")
                if response_id is not None and response_id in seen_responses:
                    continue
                seen_responses.add(response_id)
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


# ---- semantic_diagnostics ----------------------------------------------------------------------
# Host-private semantic-search census over a reviewed public capsule snapshot.
#
# The census is advisory evidence. It never selects a treatment, serves an agent
# tool or prompt, or contributes to a Phase 1 grade or certification decision.

_SCHEMA = "merlin.phase1.semantic_search_diagnostic.v1"

_RECEIPT = "semantic_search_diagnostic.json"

_MAX_PAYLOAD_OPS = 32

_MAX_LINALG_BYTES = 2 * 1024 * 1024

_SEARCH_LIMITS = SearchLimits(timeout_ms=250, solver_timeout_ms=50, max_candidates=64)


def _ordinary_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"semantic-search input is not an ordinary file: {path}")
    return path.read_bytes()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _assert_host_private_location(run_dir: Path, *, workspace: Path, public_root: Path) -> None:
    """Keep the owner receipt out of every directly bound agent input tree.

    Mode 0600 is insufficient: the sandboxed agent runs under the operator's UID.
    Resolve parent symlinks before testing both containment directions so a
    configured run root cannot alias an agent-visible workspace or public view.
    """
    run = run_dir.resolve()
    for label, root in (("workspace", workspace), ("public corpus", public_root)):
        visible = root.resolve()
        if run.is_relative_to(visible) or visible.is_relative_to(run):
            raise RuntimeError(f"semantic-search host-private run directory overlaps the agent {label}")


def assert_private_mounts(argv: list[str], run_dir: Path) -> None:
    """Reject a final agent mount plan that exposes run metadata at any bind alias.

    Call after *all* runtime/toolchain binds and answer masks have been composed,
    immediately before launching an agent. A source bound at another destination
    needs checking too; testing only the receipt's original path misses aliases.
    """
    private_trees = [path.resolve(strict=True) for path in (run_dir / "qa_history",) if path.exists()]
    private = [
        path.resolve(strict=True)
        for path in (run_dir, run_dir / "environment.yaml", run_dir / _RECEIPT)
        if path.exists()
    ] + private_trees
    if not private:
        raise RuntimeError("semantic-search host-private run directory is missing before agent launch")
    if any(BW.is_exposed(argv, path) for path in private):
        raise RuntimeError("semantic-search host-private run metadata is visible in the agent sandbox")
    for state, source, destination in BW._mounts(argv):
        if state != "expose":
            continue
        source_path = Path(source).resolve()
        for path in private:
            if path.is_relative_to(source_path):
                alias = Path(destination) / path.relative_to(source_path)
                if BW.is_exposed(argv, alias):
                    raise RuntimeError(
                        "semantic-search host-private run metadata is visible through an agent bind alias"
                    )
        # A direct bind of a private descendant does not include its parent at
        # the alias. Protect history trees without rejecting intentionally
        # public frozen views elsewhere below the host-owned run directory.
        if any(source_path.is_relative_to(tree) for tree in private_trees) and BW.is_exposed(argv, Path(destination)):
            raise RuntimeError("host-private QA history is visible through an agent bind alias")


def _public_linalg_files(public_root: Path, contract_root: Path) -> list[tuple[str, Path]]:
    """Select declared linalg files from public capsules in the frozen view only."""
    selected = []
    for capsule in discover_capsules(public_root, labels={"public"}, contract=contract_root):
        relative = capsule.get("linalg_mlir")
        if relative is None:
            continue
        relpath = Path(str(relative))
        directory = Path(capsule["__dir__"])
        if relpath.is_absolute() or ".." in relpath.parts or not relpath.parts:
            raise RuntimeError("public capsule declares an escaping linalg input")
        if not directory.is_relative_to(public_root):
            raise RuntimeError("public capsule escaped the frozen public view")
        path = directory / relpath
        if not path.resolve(strict=True).is_relative_to(directory.resolve(strict=True)):
            raise RuntimeError("public capsule linalg input escaped its capsule directory")
        selected.append((path.relative_to(public_root).as_posix(), path))
    names = [name for name, _ in selected]
    if len(set(names)) != len(names):
        raise RuntimeError("public linalg input names are not unique")
    return sorted(selected)


def create(
    run_dir: Path,
    *,
    workspace: Path,
    model_path: Path | None,
    public_root: Path,
    contract_root: Path,
) -> dict:
    """Write an owner-only receipt when the reviewed release has a model."""
    _assert_host_private_location(run_dir, workspace=workspace, public_root=public_root)
    if model_path is None:
        return {"schema": _SCHEMA, "status": "unavailable", "reason": "reviewed release has no instruction model"}

    model_bytes = _ordinary_bytes(model_path)
    model_sha = _sha256(model_bytes)
    public_sha = fingerprint(public_root)
    rows = []
    try:
        model = json.loads(model_bytes)
        if not isinstance(model, Mapping):
            raise ValueError("instruction model must be a JSON object")
        # Phase 0 emits this one non-selectable stub when no OOT instruction
        # description was selected. Every other model must carry the reviewed
        # SW-spec/CIRCT-fact identities and a self-consistent normalized body.
        stub_fields = {"schema", "target", "status", "unknowns", "instructions"}
        if model.get("status") == "UNKNOWN" and model.get("instructions") == [] and set(model) == stub_fields:
            if (
                model.get("schema") != "merlin.instruction_semantics.v1"
                or not isinstance(model.get("target"), str)
                or not model["target"]
                or not isinstance(model.get("unknowns"), list)
                or not model["unknowns"]
            ):
                raise ValueError("Phase 0 instruction-model stub is malformed")
        else:
            model = validate_normalized_instruction_model(model)
        model_error = None
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        model = None
        model_error = f"{type(exc).__name__}: {exc}"
    try:
        selected = _public_linalg_files(public_root, contract_root)
        selection_error = None
    except Exception as exc:  # noqa: BLE001 — optional discovery is diagnostic, not Phase 1 admission
        selected = []
        selection_error = f"{type(exc).__name__}: {exc}"
    searched_ops = 0
    for name, path in selected:
        row = {"capsule_linalg": name}
        try:
            raw = _ordinary_bytes(path)
        except Exception as exc:  # noqa: BLE001 — the native corpus owner retains admission authority
            row.update(status="unavailable", reason=f"{type(exc).__name__}: {exc}")
            rows.append(row)
            continue
        row["linalg_sha256"] = _sha256(raw)
        if model_error is not None:
            row.update(status="unavailable", reason=model_error)
        elif len(raw) > _MAX_LINALG_BYTES:
            row.update(status="limit_reached", reason="declared linalg input exceeds diagnostic byte limit")
        else:
            try:
                text = raw.decode("utf-8")
                if not is_linalg_on_tensors(text):
                    raise ValueError("declared linalg input lacks linalg-on-tensors provenance")
                parsed = parse_linalg_mlir(text)
                operations = parsed["ops"]
                if not any(str(op.get("operation", "")).startswith("linalg.") for op in operations):
                    raise ValueError("declared linalg input has no linalg payload operation")
                if searched_ops + len(operations) > _MAX_PAYLOAD_OPS:
                    row.update(status="limit_reached", reason="public payload operation budget exhausted")
                else:
                    searched_ops += len(operations)
                    row.update(
                        status="observed",
                        inventory=search_linalg_inventory(parsed, model, limits=_SEARCH_LIMITS),
                    )
            except Exception as exc:  # noqa: BLE001 — diagnostic failure cannot change the experiment treatment
                row.update(status="diagnostic_error", reason=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    if fingerprint(public_root) != public_sha or _sha256(_ordinary_bytes(model_path)) != model_sha:
        raise RuntimeError("semantic-search inputs changed during diagnostic collection")
    receipt = {
        "schema": _SCHEMA,
        "scope": "host-private public capsule linalg files; advisory only; no compiler or execution verdict",
        "instruction_model_sha256": model_sha,
        "public_corpus_sha256": public_sha,
        "selection_error": selection_error,
        "limits": {
            "max_payload_operations": _MAX_PAYLOAD_OPS,
            "max_linalg_bytes": _MAX_LINALG_BYTES,
            "search_timeout_ms_per_operation": _SEARCH_LIMITS.timeout_ms,
            "solver_timeout_ms": _SEARCH_LIMITS.solver_timeout_ms,
        },
        "inputs": len(rows),
        "rows": rows,
    }
    data = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode("utf-8")
    destination = run_dir / _RECEIPT
    with os.fdopen(os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        stream.write(data)
    return {
        "schema": _SCHEMA,
        "status": "recorded",
        "instruction_model_sha256": model_sha,
        "public_corpus_sha256": public_sha,
        "receipt_sha256": _sha256(data),
        "inputs": len(rows),
    }


def verify(record: Mapping, run_dir: Path, *, workspace: Path, model_path: Path | None, public_root: Path) -> None:
    """Refuse drift of a new run's optional model, public inputs, or receipt."""
    _assert_host_private_location(run_dir, workspace=workspace, public_root=public_root)
    if not isinstance(record, Mapping) or record.get("schema") != _SCHEMA:
        raise RuntimeError("semantic-search diagnostic identity is missing or malformed")
    if record.get("status") == "unavailable":
        if model_path is not None or (run_dir / _RECEIPT).exists():
            raise RuntimeError("semantic-search diagnostic availability changed on resume")
        return
    if record.get("status") != "recorded" or model_path is None:
        raise RuntimeError("semantic-search diagnostic model selection changed on resume")
    model_sha = _sha256(_ordinary_bytes(model_path))
    if model_sha != record.get("instruction_model_sha256"):
        raise RuntimeError("semantic-search instruction model changed on resume")
    public_sha = fingerprint(public_root)
    if public_sha != record.get("public_corpus_sha256"):
        raise RuntimeError("semantic-search public corpus changed on resume")
    receipt_path = run_dir / _RECEIPT
    receipt = _ordinary_bytes(receipt_path)
    if receipt_path.stat().st_mode & 0o077:
        raise RuntimeError("semantic-search diagnostic receipt is not owner-only")
    if _sha256(receipt) != record.get("receipt_sha256"):
        raise RuntimeError("semantic-search diagnostic receipt changed on resume")
    try:
        document = json.loads(receipt)
    except json.JSONDecodeError as exc:
        raise RuntimeError("semantic-search diagnostic receipt is malformed") from exc
    if (
        document.get("schema") != _SCHEMA
        or document.get("instruction_model_sha256") != model_sha
        or document.get("public_corpus_sha256") != public_sha
        or document.get("inputs") != record.get("inputs")
        or not isinstance(document.get("rows"), list)
    ):
        raise RuntimeError("semantic-search diagnostic receipt identity is inconsistent")


# ---- resilient_output --------------------------------------------------------------------------
# Keep a long-lived process alive when its log cannot be written.
#
# The runner and its brokers write their logs to a file on the run's own volume. When that volume
# filled, every ``print`` raised ``OSError`` -- including the ones inside the ``except`` blocks that
# were there to report a failed grade -- so the grading threads and the self-check broker died while
# the agent kept working, and the run went three hours without feedback. A log line is never worth a
# grader: :func:`install` wraps ``sys.stdout``/``sys.stderr`` so a failed write or flush drops that
# text, counts it, and says so in one line once a write succeeds again.


class ResilientStream:
    """A text stream whose ``write``/``flush`` report an ``OSError`` as dropped text, never raise it."""

    def __init__(self, stream) -> None:
        self._stream = stream
        self._lock = threading.Lock()
        self.dropped = 0
        self.reason = ""

    def _failed(self, error: OSError, *, dropped: int) -> None:
        with self._lock:
            self.dropped += dropped
            self.reason = os.strerror(error.errno) if error.errno else type(error).__name__

    def write(self, text):
        with self._lock:
            dropped, reason = self.dropped, self.reason
        try:
            if dropped:
                self._stream.write(f"[output] {dropped} writes dropped: {reason}\n")
                with self._lock:
                    self.dropped -= dropped
            return self._stream.write(text)
        except OSError as error:
            self._failed(error, dropped=1)
            return len(text)

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        try:
            self._stream.flush()
        except OSError as error:
            self._failed(error, dropped=0)  # Buffered text is retried by the next flush.

    def __getattr__(self, name):
        return getattr(self._stream, name)


def install() -> None:
    """Wrap this process's ``sys.stdout`` and ``sys.stderr`` once; child processes are unaffected."""
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if stream is not None and not isinstance(stream, ResilientStream):
            setattr(sys, name, ResilientStream(stream))


__all__ = [
    "ResilientStream",
    "install",
]
