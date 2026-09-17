#!/usr/bin/env python3
"""Atomically seal one complete 4x4 CPU-host campaign without outcome-based selection."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable

import yaml

from merlin.benchharness.host_agent import (
    _submission_package_digest,
    _submission_source_digest,
)
from merlin.compare.host_experiment import HostExperimentSpec


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _assert_campaign_not_excluded(runs_root: Path, campaign_run_id: str) -> None:
    """Treat any controller exclusion pathname, including a symlink, as irrevocable."""
    exclusion = runs_root / ".campaign_exclusions" / f"{campaign_run_id}.json"
    if exclusion.exists() or exclusion.is_symlink():
        raise ValueError(
            "controller-excluded campaign cannot be finalized or promoted: "
            f"{campaign_run_id}")


_TOKEN_FIELDS = ("input_tokens", "cached_input_tokens", "cache_write_input_tokens",
                 "output_tokens", "reasoning_output_tokens", "uncached_input_tokens")

_TRUSTED_SEARCH_CHECKS = {
    "private_ledger", "search_record_present", "selected_policy_present",
    "exact_final_search_file_set", "heldout_never_opened", "six_balanced_paired_measurements",
    "screen_evaluations_budget", "confirmation_requests_budget", "package_builds_budget",
    "compiler_invocations_budget", "spike_checks_budget", "k1_programs_budget",
    "search_wall_deadline", "submission_source_unchanged", "deterministic_replay",
    "controller_private_prebuild", "private_prebuild_snapshots_present",
    "private_prebuild_snapshot_identity", "private_prebuild_logs_valid",
    "controller_private_shape_corpus", "evaluations_bind_private_shape_corpus",
    "terminal_receipt_associations",
    "all_requests_have_passing_terminal_receipts", "broker_terminal_timing",
    "search_record_matches_replay", "selected_policy_matches_replay",
    "independent_convergence_sweep",
    "staged_spike_k1_policy", "manifest_policy_inside_submission",
    "submission_policy_byte_match",
}

_LEVEL_CHECKS = {
    "L0": {"version", "capsule_sha256", "requested_mode", "actual_mode",
           "fallback_forbidden", "harts", "vlen_bits", "source_sha256",
           "transformations", "vlen_policy", "tail_policy", "kernel_symbol", "no_main",
           "no_process_wrappers", "trusted_receipt_isolation", "no_constructors",
           "source_size", "lowered_changed",
           "c_syntax", "mlir_verifier"},
    "L1": {"numeric_correctness", "memory_safety", "post_codegen_random_seeds",
           "trusted_parent_receipts"},
    "L2": {"rvv_correctness", "instruction_evidence", "tail_case", "vlen_256",
           "cycle_measurement"},
    "L3": {"exact_mode", "no_fallback", "numeric_correctness", "trusted_parent_receipt",
           "per_call_correctness", "csr_vlen",
           "exact_affinity", "exact_task_count", "active_harts", "wall_time", "peak_rss",
           "audit_attribution", "upload_integrity"},
}

_EARLY_FAILURE_PREFIXES = {
    "L0": ("compiler invocation failed", "compiler invocation timed out",
           "compiler omitted outputs", "metadata is invalid:",
           "C syntax check timed out", "MLIR verifier timed out"),
    "L1": ("L0 scalar artifact failed", "trusted native build failed",
           "trusted native build timed out", "trusted native execution timed out"),
    "L2": ("L0 RVV artifact failed", "Spike build timed out", "Spike build failed",
           "Spike execution timed out"),
    "L3": ("L0 scalar artifact failed", "L0 rvv artifact failed",
           "L0 rvv_multicore artifact failed", "K1 cross-build timed out",
           "K1 cross-build failed"),
}


def _nonblank_line_count(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


def _exact_attempt_streams(directory: Path, attempts: list[dict[str, Any]], *, label: str
                           ) -> list[dict[str, Any]]:
    """Require the complete, immutable raw evidence set copied by record_codex_trajectory."""
    if not directory.is_dir():
        raise ValueError(f"missing retained {label} evidence directory: {directory}")
    expected = []
    for ordinal, attempt in enumerate(attempts):
        if not isinstance(attempt, dict) or int(attempt.get("index", -1)) != ordinal:
            raise ValueError("run_result attempts must have consecutive zero-based indices")
        expected.append(f"attempt_{ordinal:04d}.jsonl")
    actual = sorted(path.name for path in directory.iterdir() if path.is_file())
    if actual != expected or any(path.is_dir() or path.is_symlink() for path in directory.iterdir()):
        raise ValueError(
            f"retained {label} evidence must contain exactly one non-symlink JSONL stream per attempt")
    return [{"name": name, "sha256": _sha256(directory / name),
             "nonblank_lines": _nonblank_line_count(directory / name)} for name in expected]


def _as_nonnegative_int(value: Any, *, where: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{where} must be an integer, not boolean")
    try:
        output = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} must be an integer") from exc
    if output < 0:
        raise ValueError(f"{where} must be non-negative")
    return output


def _token_totals(path: Path, *, run_result: dict[str, Any], reconciliation: dict[str, Any]
                  ) -> dict[str, int]:
    """Verify ledger rows are per-turn deltas, not cumulative usage snapshots.

    AET's token ledger is one row per normalized Codex turn.  We prove that interpretation against
    both retained Chia attempts and AET's independently computed reconciliation before summing it.
    This makes campaign totals auditable even when an arm retried.
    """
    attempts = run_result.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("run_result must retain at least one Codex attempt")
    expected_turns: list[dict[str, Any]] = []
    for ordinal, attempt in enumerate(attempts):
        if not isinstance(attempt, dict) or int(attempt.get("index", -1)) != ordinal:
            raise ValueError("run_result attempts must have consecutive zero-based indices")
        turns = attempt.get("turns")
        if not isinstance(turns, list):
            raise ValueError("run_result attempt has no turn ledger")
        if any(not isinstance(turn, dict) for turn in turns):
            raise ValueError("run_result turns must be objects")
        expected_turns.extend(turns)
    parsed: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        parsed.append(row)
    if len(parsed) != len(expected_turns):
        raise ValueError("AET token ledger turn count does not match retained Chia attempts")
    if [row.get("turn") for row in parsed] != list(range(len(parsed))):
        raise ValueError("AET token ledger must use consecutive per-turn delta ordinals")
    totals = {name: 0 for name in _TOKEN_FIELDS}
    uncached_total = 0
    for ordinal, (row, expected) in enumerate(zip(parsed, expected_turns)):
        required_fields = _TOKEN_FIELDS
        missing = [name for name in required_fields
                   if name not in row or row[name] is None or
                   name not in expected or expected[name] is None]
        if missing:
            raise ValueError(
                f"token turn {ordinal} is missing required full-fidelity field(s): {missing}")
        for name in _TOKEN_FIELDS:
            value, expected_value = row.get(name), expected.get(name)
            if value is not None:
                value = _as_nonnegative_int(value, where=f"token ledger turn {ordinal}.{name}")
            if expected_value is not None:
                expected_value = _as_nonnegative_int(
                    expected_value, where=f"run_result turn {ordinal}.{name}")
            if value != expected_value:
                raise ValueError(
                    f"AET token ledger is not the per-turn delta recorded by Chia at turn {ordinal}")
            if value is not None:
                totals[name] += value
        cached = row.get("cached_input_tokens") or 0
        cache_write = row.get("cache_write_input_tokens") or 0
        input_tokens = row.get("input_tokens")
        output_tokens = row.get("output_tokens")
        reasoning = row.get("reasoning_output_tokens") or 0
        if input_tokens is not None and cached + cache_write > input_tokens:
            raise ValueError("AET token ledger violates cache subset semantics")
        if output_tokens is not None and reasoning > output_tokens:
            raise ValueError("AET token ledger violates reasoning subset semantics")
        uncached = row.get("uncached_input_tokens")
        derived_uncached = None if input_tokens is None else max(input_tokens - cached - cache_write, 0)
        if uncached != derived_uncached:
            raise ValueError("AET token ledger has non-delta uncached-input accounting")
        if uncached is not None:
            uncached_total += _as_nonnegative_int(uncached, where="uncached_input_tokens")

    ledger = reconciliation.get("token_ledger")
    if not isinstance(ledger, dict) or ledger.get("num_turns") != len(parsed):
        raise ValueError("AET reconciliation does not attest the exact token ledger turn count")
    checks = ledger.get("checks")
    if not isinstance(checks, dict) or ledger.get("all_match") is not True or ledger.get(
            "subset_invariants_hold") is not True:
        raise ValueError("AET reconciliation did not validate token ledger accounting")
    expected_totals = {
        "input_tokens": None,  # retained in full; reconciliation exposes its non-overlapping form.
        "cached_input_tokens": checks.get("cache_read", {}).get("ledger"),
        "cache_write_input_tokens": checks.get("cache_write", {}).get("ledger"),
        "output_tokens": checks.get("output", {}).get("ledger"),
        "reasoning_output_tokens": checks.get("reasoning", {}).get("ledger"),
    }
    if checks.get("uncached_input", {}).get("ledger") != uncached_total:
        raise ValueError("AET reconciliation disagrees with per-turn uncached-input totals")
    for name, expected_total in expected_totals.items():
        if expected_total is not None and totals[name] != _as_nonnegative_int(
                expected_total, where=f"AET reconciliation {name}"):
            raise ValueError(f"AET reconciliation disagrees with per-turn {name} totals")
    return totals


def _tool_rows(path: Path, *, token_path: Path, raw_directory: Path,
               timestamped_directory: Path, run_result: dict[str, Any], model: str
               ) -> list[dict[str, Any]]:
    """Re-derive AET's normalized tool ledger from the retained lossless streams.

    This closes the gap between an arbitrary ``tools.jsonl`` and the evidence that AET imported.
    The run-result attempt records independently attest how many tool calls Codex/Chia observed.
    """
    attempts = run_result.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("run_result must retain attempts before tools can be reconciled")
    expected_attempt_tools: list[dict[str, Any]] = []
    for attempt in attempts:
        tools = attempt.get("tools") if isinstance(attempt, dict) else None
        if not isinstance(tools, list) or any(not isinstance(tool, dict) for tool in tools):
            raise ValueError("run_result attempt has no valid tool ledger")
        expected_attempt_tools.extend(tools)
    parsed: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{path}:{line_number} is a blank tool-ledger row")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        parsed.append(row)
    try:
        from aet.trajectory.importers.codex import import_codex_run
        from aet.trajectory.reconcile import token_ledger_rows, tool_ledger_rows
        _, normalized = import_codex_run(
            raw_directory, timestamped=timestamped_directory, model=model,
            billing_mode="subscription", run_id="campaign-finalizer-replay")
        rederived = tool_ledger_rows(normalized)
        rederived_tokens = token_ledger_rows(normalized)
    except Exception as exc:
        raise ValueError(f"cannot deterministically rederive AET tool ledger from retained streams: {exc}") from exc
    if parsed != rederived:
        raise ValueError("AET tools.jsonl differs from deterministic retained-stream rederivation")
    retained_tokens = [json.loads(line) for line in token_path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if retained_tokens != rederived_tokens:
        raise ValueError("AET token_ledger.jsonl differs from deterministic retained-stream rederivation")
    # The two recorders serialize different detail fields, but their stable tool identities must
    # agree exactly and neither side can hide a dropped/inserted invocation.
    attempt_identities = [(tool.get("item_id"), tool.get("item_type"))
                          for tool in expected_attempt_tools]
    rederived_identities = [(tool.get("item_id"), tool.get("kind")) for tool in rederived]
    if (len(set(attempt_identities)) != len(attempt_identities) or
            len(set(rederived_identities)) != len(rederived_identities) or
            set(attempt_identities) != set(rederived_identities)):
        raise ValueError("AET tool ledger identities do not match retained Chia attempts")
    return parsed


def _finite_nonnegative(value: Any, *, where: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{where} must be a finite non-negative number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} must be a finite non-negative number") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{where} must be a finite non-negative number")
    return number


def _same_seconds(actual: Any, expected: Any, *, where: str) -> float:
    actual_number = _finite_nonnegative(actual, where=where)
    expected_number = _finite_nonnegative(expected, where=f"expected {where}")
    if not math.isclose(actual_number, expected_number, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError(f"{where} differs from its retained authoritative evidence")
    return actual_number


def _driver_total_wall(timing: dict[str, Any]) -> float:
    """Validate a driver-emitted monotonic interval, independent of summary metrics."""
    if timing.get("version") != 1 or timing.get("authority") != "driver_monotonic_ns":
        raise ValueError("driver wall timing has no recognized authority")
    start = timing.get("start_monotonic_ns")
    end = timing.get("end_monotonic_ns")
    if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
        raise ValueError("driver wall timing ticks must be integer monotonic nanoseconds")
    if start < 0 or end < start:
        raise ValueError("driver wall timing monotonic interval is invalid")
    return _same_seconds(timing.get("wall_seconds"), (end - start) / 1e9,
                         where="driver wall_seconds")


def _timing_evidence(*, summary: dict[str, Any], run_result: dict[str, Any],
                     grader_result: dict[str, Any], search_seal: dict[str, Any],
                     driver_wall_timing: dict[str, Any]) -> dict[str, float]:
    active = _same_seconds(summary.get("active_wall_seconds"), run_result.get("active_wall_s"),
                           where="active_wall_seconds")
    grader = _same_seconds(summary.get("grader_wall_seconds"), grader_result.get("wall_seconds"),
                           where="grader_wall_seconds")
    search = _same_seconds(summary.get("trusted_search_wall_seconds"),
                           _finite_nonnegative(search_seal.get("trusted_broker_wall_ns", 0),
                                               where="trusted_search_seal wall_ns") / 1e9,
                           where="trusted_search_wall_seconds")
    wall = _same_seconds(summary.get("wall_seconds"), _driver_total_wall(driver_wall_timing),
                         where="wall_seconds")
    # Search runs concurrently with the agent; grading follows it.  Requiring this lower bound
    # catches impossible negative/corrupted totals without claiming the parallel phases add.
    if wall + 1e-6 < max(active, search) + grader:
        raise ValueError("wall_seconds is smaller than the recorded agent/search plus grader duration")
    return {"active_wall_seconds": active, "grader_wall_seconds": grader,
            "trusted_search_wall_seconds": search, "wall_seconds": wall}


def _expected_grader_layout(spec: HostExperimentSpec) -> dict[str, Any]:
    grader_path = spec._repo_path(spec.grading["grader"])
    module_spec = importlib.util.spec_from_file_location("cpu_host_completion_grader", grader_path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError("cannot load the frozen CPU-host grader for result validation")
    grader = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(grader)
    corpus = spec._repo_path(spec.development_corpus["materialized_capsules"]).resolve()
    splits = grader._validate_corpus(
        corpus / "public" / "train.jsonl", corpus / "public" / "validation.jsonl",
        corpus / "sealed" / "heldout.jsonl")
    semantic = grader._select_semantic_coverage(splits["heldout"])
    tails = grader._select_tail_coverage(splits["heldout"], split_name="heldout")
    multicore = grader._select_multicore(splits["heldout"], split_name="heldout")
    by_id = {row["id"]: row for row in [*semantic, *tails, multicore]}
    selected = [by_id[name] for name in sorted(by_id)]
    return {
        "rows": by_id, "buffer_plans": {
            row["id"]: grader._buffer_plan(row) for row in selected},
        "selected": [row["id"] for row in selected],
        "tails": [row["id"] for row in tails], "multicore": multicore["id"],
        "records": {
            "L0": [(row["id"], mode) for row in selected for mode in ("scalar", "rvv")] +
                  [(multicore["id"], "rvv_multicore")],
            "L1": [(row["id"], "scalar") for row in selected],
            "L2": [(row["id"], "rvv") for row in tails],
            "L3": ([(row["id"], mode) for row in tails for mode in ("scalar", "rvv")] +
                   [(multicore["id"], "rvv_multicore")]),
        },
    }


def _require_exact_record_fields(record: dict[str, Any], expected: set[str], *, where: str) -> None:
    if set(record) != expected:
        raise ValueError(f"{where} does not match the exact grader producer schema")


def _require_int(value: Any, *, where: str, nonzero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or (nonzero and value == 0):
        qualifier = " non-zero" if nonzero else ""
        raise ValueError(f"{where} must be an integer{qualifier}")
    return value


def _validate_build_logs(logs: Any, *, where: str) -> list[dict[str, Any]]:
    if not isinstance(logs, list):
        raise ValueError(f"{where} must retain the grader build-log list")
    for index, log in enumerate(logs):
        if not isinstance(log, dict) or set(log) != {"returncode", "wall_seconds", "stderr_tail"}:
            raise ValueError(f"{where}[{index}] does not match the grader producer schema")
        _require_int(log["returncode"], where=f"{where}[{index}].returncode")
        _finite_nonnegative(log["wall_seconds"], where=f"{where}[{index}].wall_seconds")
        if not isinstance(log["stderr_tail"], str):
            raise ValueError(f"{where}[{index}].stderr_tail must be text")
    return logs


def _validate_submission_build_failure(
        result: dict[str, Any], *, expected_commands: list[Any]) -> None:
    expected = {"version", "status", "failure_class", "implemented_levels", "reason",
                "build_failure", "wall_seconds"}
    _require_exact_record_fields(result, expected, where="grader submission build failure")
    if (result["version"] != 1 or result["status"] != "treatment_build_fail" or
            result["failure_class"] != "treatment_build_fail" or
            result["implemented_levels"] != list(_LEVEL_CHECKS) or
            result["reason"] not in {"submission build failed", "submission build timed out",
                                     "submission build did not produce a valid compiler"}):
        raise ValueError("grader submission build failure has an invalid typed envelope")
    _finite_nonnegative(result["wall_seconds"], where="grader build-failure wall_seconds")
    evidence = result["build_failure"]
    if not isinstance(evidence, dict):
        raise ValueError("grader submission build failure lacks retained evidence")
    commands = evidence.get("commands")
    stage = _require_int(evidence.get("failed_stage_index"), where="submission build stage")
    if not isinstance(commands, list) or not commands or stage != len(commands) - 1:
        raise ValueError("grader submission build stage does not match retained commands")
    if ([log.get("command") for log in commands if isinstance(log, dict)] !=
            expected_commands[:len(commands)]):
        raise ValueError("grader submission build logs differ from the sealed manifest")
    for index, log in enumerate(commands):
        if (not isinstance(log, dict) or not isinstance(log.get("command"), list) or
                not log["command"] or any(not isinstance(part, str) for part in log["command"]) or
                not isinstance(log.get("stdout_tail"), str) or
                not isinstance(log.get("stderr_tail"), str)):
            raise ValueError("grader submission build log is malformed")
        _finite_nonnegative(log.get("wall_seconds"), where="submission build command wall")
        if index < stage and _require_int(
                log.get("returncode"), where="prior submission build returncode") != 0:
            raise ValueError("submission build failure has an earlier failed command")
    terminal = commands[-1]
    if result["reason"] == "submission build timed out":
        if (set(evidence) != {"commands", "failed_stage_index", "timeout_seconds"} or
                evidence["timeout_seconds"] != 1800 or terminal.get("returncode") is not None or
                terminal.get("outcome") != "timeout"):
            raise ValueError("submission build timeout evidence is inconsistent")
    elif result["reason"] == "submission build failed":
        if (set(evidence) != {"commands", "failed_stage_index", "returncode"} or
                _require_int(evidence["returncode"], where="submission build returncode",
                             nonzero=True) != terminal.get("returncode")):
            raise ValueError("submission build failure evidence is inconsistent")
    else:
        if (set(evidence) != {"commands", "failed_stage_index", "contract_error"} or
                not isinstance(evidence["contract_error"], str) or
                _require_int(terminal.get("returncode"),
                             where="terminal submission build returncode") != 0):
            raise ValueError("submission build output-contract evidence is inconsistent")


def _is_digest(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64 and
            all(character in "0123456789abcdef" for character in value))


def _validate_check_record(
        level_name: str, record: dict[str, Any], *, expected_row: dict[str, Any],
        expected_buffer_plan: dict[str, Any],
        l0_records: dict[tuple[str, str], dict[str, Any]]) -> None:
    """Require the exact public record shape emitted after a grader actually ran its checks."""
    base = {"capsule", "family", "mode", "checks"}
    if level_name == "L0":
        expected = base | {"returncode", "wall_seconds", "stdout_tail", "stderr_tail", "ok",
                           "metadata", "source_sha256", "syntax_stderr", "verifier_stderr",
                           "buffer_plan", "source_size_bytes", "input_mlir_sha256",
                           "lowered_mlir_sha256", "syntax_returncode", "verifier_returncode"}
        _require_exact_record_fields(record, expected, where="grader L0 checked record")
        if _require_int(record["returncode"], where="grader L0 returncode") != 0:
            raise ValueError("grader L0 checked record has non-zero compiler return code")
        _finite_nonnegative(record["wall_seconds"], where="grader L0 wall_seconds")
        if any(not isinstance(record[name], str) for name in (
                "stdout_tail", "stderr_tail", "syntax_stderr", "verifier_stderr")):
            raise ValueError("grader L0 checked record log tails must be text")
        if not isinstance(record["ok"], bool) or record["ok"] != all(record["checks"].values()):
            raise ValueError("grader L0 ok flag disagrees with its checks")
        if (not isinstance(record["metadata"], dict) or
                record["buffer_plan"] != expected_buffer_plan or
                not _is_digest(record["source_sha256"]) or
                not _is_digest(record["input_mlir_sha256"]) or
                not _is_digest(record["lowered_mlir_sha256"])):
            raise ValueError("grader L0 checked record lacks metadata/source/buffer evidence")
        metadata = record["metadata"]
        mode = str(record["mode"])
        vector_mode = mode != "scalar"
        expected_harts = int(expected_row["core_count"]) if mode == "rvv_multicore" else 1
        source_size = _require_int(record["source_size_bytes"], where="grader L0 source size")
        metadata_checks = {
            "version": metadata.get("version") == 1,
            "capsule_sha256": metadata.get("capsule_sha256") == expected_row["sha256"],
            "requested_mode": metadata.get("requested_mode") == mode,
            "actual_mode": metadata.get("actual_mode") == mode,
            "fallback_forbidden": metadata.get("fallback_used") is False,
            "harts": (not isinstance(metadata.get("harts"), bool) and
                      metadata.get("harts") == expected_harts),
            "vlen_bits": (not isinstance(metadata.get("vlen_bits"), bool) and
                          metadata.get("vlen_bits") == 256),
            "source_sha256": metadata.get("source_sha256") == record["source_sha256"],
            "transformations": isinstance(metadata.get("transformations"), list) and
                               bool(metadata.get("transformations")),
            "vlen_policy": (metadata.get("vlen_policy") == "not_applicable" if not vector_mode
                            else metadata.get("vlen_policy") in {
                                "scalable_vl", "runtime_verified_fixed"}),
            "tail_policy": (metadata.get("tail_policy") == "not_applicable" if not vector_mode
                            else metadata.get("tail_policy") in {"dynamic_vl", "explicit_mask"}),
            "source_size": 0 <= source_size <= 8 * 1024 * 1024,
            "lowered_changed": record["lowered_mlir_sha256"] != record["input_mlir_sha256"],
            "c_syntax": _require_int(
                record["syntax_returncode"], where="grader L0 syntax returncode") == 0,
            "mlir_verifier": _require_int(
                record["verifier_returncode"], where="grader L0 verifier returncode") == 0,
        }
        if any(record["checks"][name] != value for name, value in metadata_checks.items()):
            raise ValueError("grader L0 checks disagree with retained metadata/tool evidence")
        return

    if level_name == "L1":
        expected = base | {"status", "build_wall_seconds", "build_stderr_tail",
                           "build_returncode", "trials", "receipt_nonce"}
        _require_exact_record_fields(record, expected, where="grader L1 checked record")
        _finite_nonnegative(record["build_wall_seconds"], where="grader L1 build_wall_seconds")
        if (_require_int(record["build_returncode"], where="grader L1 build_returncode") != 0 or
                not isinstance(record["build_stderr_tail"], str)):
            raise ValueError("grader L1 checked record lacks a successful retained native build")
        trials = record["trials"]
        nonce = _require_int(record["receipt_nonce"], where="grader L1 receipt nonce")
        if nonce < 1:
            raise ValueError("grader L1 receipt nonce must be positive")
        if not isinstance(trials, list) or len(trials) != 3:
            raise ValueError("grader L1 must retain exactly three randomized trials")
        for index, trial in enumerate(trials):
            if not isinstance(trial, dict) or set(trial) != {
                    "seed", "returncode", "wall_seconds", "stdout_tail", "stderr_tail", "ok"}:
                raise ValueError(f"grader L1 trial {index} does not match the producer schema")
            if (_require_int(trial["seed"], where=f"grader L1 trial {index}.seed") < 1 or
                    not isinstance(trial["ok"], bool) or
                    not isinstance(trial["stdout_tail"], str) or
                    not isinstance(trial["stderr_tail"], str)):
                raise ValueError(f"grader L1 trial {index} has invalid evidence types")
            _require_int(trial["returncode"], where=f"grader L1 trial {index}.returncode")
            _finite_nonnegative(trial["wall_seconds"], where=f"grader L1 trial {index}.wall_seconds")
            expected_line = (f"MERLIN_TRUSTED_RESULT version=1 seed={trial['seed']} "
                             f"nonce={nonce} memory=1 numeric=1")
            trusted_lines = [line for line in trial["stdout_tail"].splitlines()
                             if line.startswith("MERLIN_TRUSTED_RESULT ")]
            if trial["ok"] != (trial["returncode"] == 0 and trusted_lines == [expected_line]):
                raise ValueError(f"grader L1 trial {index} ok flag disagrees with retained output")
        trials_ok = all(trial["ok"] for trial in trials)
        if record["checks"] != {"numeric_correctness": trials_ok, "memory_safety": trials_ok,
                                "post_codegen_random_seeds": True,
                                "trusted_parent_receipts": trials_ok}:
            raise ValueError("grader L1 checks disagree with randomized trial evidence")
        return

    if level_name == "L2":
        expected = base | {"status", "tail_case", "seed", "vector_instructions",
                           "vector_dataflow", "kernel_text_sha256",
                           "linked_vector_dataflow", "executed_vector_dataflow",
                           "required_pc_trace_lines", "spike_trace_sha256", "trusted_receipt",
                           "receipt_nonce",
                           "build_logs", "spike_cycles", "spike_returncode", "wall_seconds",
                           "stdout_tail", "stderr_tail"}
        _require_exact_record_fields(record, expected, where="grader L2 checked record")
        if (not isinstance(record["tail_case"], bool) or
                _require_int(record["seed"], where="grader L2 seed") < 1 or
                _require_int(record["receipt_nonce"], where="grader L2 receipt nonce") < 1 or
                _require_int(record["spike_cycles"], where="grader L2 spike_cycles") < 0 or
                not isinstance(record["vector_instructions"], list) or
                any(not isinstance(value, str) for value in record["vector_instructions"]) or
                not isinstance(record["vector_dataflow"], dict) or
                not isinstance(record["linked_vector_dataflow"], dict) or
                not isinstance(record["executed_vector_dataflow"], bool) or
                not isinstance(record["trusted_receipt"], bool) or
                not isinstance(record["required_pc_trace_lines"], list) or
                any(not isinstance(value, str) for value in record["required_pc_trace_lines"]) or
                not _is_digest(record["kernel_text_sha256"]) or
                not _is_digest(record["spike_trace_sha256"]) or
                not isinstance(record["stdout_tail"], str) or
                not isinstance(record["stderr_tail"], str)):
            raise ValueError("grader L2 checked record has invalid execution evidence")
        logs = _validate_build_logs(record["build_logs"], where="grader L2 build_logs")
        if len(logs) != 6 or any(log["returncode"] != 0 for log in logs):
            raise ValueError("grader L2 checked record lacks all six successful build stages")
        _finite_nonnegative(record["wall_seconds"], where="grader L2 wall_seconds")
        def validate_dataflow(dataflow, *, linked=False):
            if (set(dataflow) != {"version", "function_found", "useful",
                                 "source_vector_loads", "computed_vector_registers",
                                 "output_vector_stores", "output_scalar_stores",
                                 "output_scalar_overwrites",
                                 "required_execution_pcs", "vector_instructions"} or
                    dataflow.get("version") != 1 or
                    dataflow.get("function_found") is not True or
                    not isinstance(dataflow.get("useful"), bool) or
                    any(not isinstance(dataflow.get(name), list) for name in (
                        "source_vector_loads", "computed_vector_registers",
                        "output_vector_stores", "output_scalar_stores",
                        "output_scalar_overwrites",
                        "required_execution_pcs", "vector_instructions")) or
                    any(not isinstance(value, str) for name in (
                        "source_vector_loads", "computed_vector_registers",
                        "output_vector_stores", "output_scalar_stores",
                        "output_scalar_overwrites", "vector_instructions")
                        for value in dataflow[name]) or
                    any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                        for value in dataflow["required_execution_pcs"])):
                raise ValueError("grader L2 useful-vector dataflow evidence is malformed")
            if not linked and dataflow["vector_instructions"] != record["vector_instructions"]:
                raise ValueError("grader L2 object vector mnemonics disagree")

        dataflow = record["vector_dataflow"]
        linked_dataflow = record["linked_vector_dataflow"]
        validate_dataflow(dataflow)
        validate_dataflow(linked_dataflow, linked=True)
        retained_trace_pcs = {int(value, 16) for line in record["required_pc_trace_lines"]
                              for value in re.findall(r"0x([0-9a-fA-F]+)", line)}
        executed_vector_dataflow = bool(
            linked_dataflow["required_execution_pcs"] and
            set(linked_dataflow["required_execution_pcs"]) <= retained_trace_pcs)
        if record["executed_vector_dataflow"] != executed_vector_dataflow:
            raise ValueError("grader L2 executed-vector evidence disagrees with retained trace lines")
        receipt_pattern = re.compile(
            rf"^MERLIN_TRUSTED_RESULT version=1 seed={record['seed']} "
            rf"nonce={record['receipt_nonce']} vlenb=32 cycles=([1-9][0-9]*) calls=20$")
        trusted_matches = [receipt_pattern.fullmatch(line) for line in record["stdout_tail"].splitlines()
                           if line.startswith("MERLIN_TRUSTED_RESULT ")]
        trusted_receipt = len(trusted_matches) == 1 and trusted_matches[0] is not None
        if record["trusted_receipt"] != trusted_receipt:
            raise ValueError("grader L2 trusted receipt disagrees with retained Spike stdout")
        cycle_match = trusted_matches[0] if trusted_receipt else None
        parsed_cycles = int(cycle_match.group(1)) if cycle_match else 0
        if parsed_cycles != record["spike_cycles"]:
            raise ValueError("grader L2 cycle count differs from retained Spike stdout")
        expected_checks = {
            "rvv_correctness": _require_int(
                record["spike_returncode"], where="grader L2 spike_returncode") == 0 and
                               trusted_receipt,
            "instruction_evidence": (dataflow["useful"] and linked_dataflow["useful"] and
                                     executed_vector_dataflow),
            "tail_case": record["tail_case"] or expected_row["family"] == "runtime_parallel",
            "vlen_256": "vlenb=32" in record["stdout_tail"],
            "cycle_measurement": record["spike_cycles"] > 0,
        }
        if record["checks"] != expected_checks:
            raise ValueError("grader L2 checks disagree with retained Spike evidence")
        return

    if level_name == "L3":
        expected = base | {"status", "harts", "build_wall_seconds", "build_stderr_tail",
                           "build_returncode", "seed", "receipt_nonce", "metrics", "monitor",
                           "kernel_text_sha256", "local_sha256",
                           "remote_sha256", "board_wall_seconds", "ssh_returncode",
                           "ssh_stderr_tail"}
        _require_exact_record_fields(record, expected, where="grader L3 checked record")
        harts = _require_int(record["harts"], where="grader L3 harts")
        if harts < 1 or _require_int(
                record["build_returncode"], where="grader L3 build_returncode") != 0:
            raise ValueError("grader L3 checked record lacks a valid hart count/build")
        _finite_nonnegative(record["build_wall_seconds"], where="grader L3 build_wall_seconds")
        _finite_nonnegative(record["board_wall_seconds"], where="grader L3 board_wall_seconds")
        if (_require_int(record["seed"], where="grader L3 seed") < 1 or
                _require_int(record["receipt_nonce"], where="grader L3 receipt nonce") < 1 or
                not isinstance(record["build_stderr_tail"], str) or
                not isinstance(record["ssh_stderr_tail"], str) or
                not _is_digest(record["kernel_text_sha256"]) or
                not _is_digest(record["local_sha256"]) or not _is_digest(record["remote_sha256"])):
            raise ValueError("grader L3 checked record has invalid retained build/upload evidence")
        ssh_returncode = _require_int(record["ssh_returncode"], where="grader L3 ssh_returncode")
        metrics, monitor = record["metrics"], record["monitor"]
        metric_fields = {"vlenb", "affinity_count", "wall_ns", "time_ticks", "calls",
                         "peak_rss_kb", "pinned_hart_mask", "worker_hart_mask",
                         "productive_worker_hart_mask", "pthread_create_attempts",
                         "pthread_creates", "pthread_create_failures", "pthread_completions",
                         "pthread_affinity_attempts", "pthread_affinity_successes",
                         "pthread_affinity_failures", "minimum_worker_cpu_ns", "audit_call",
                         "audit_wall_ns", "audit_time_ticks", "correctness_checks"}
        metric_fields |= {"counterfactual_create_attempts", "counterfactual_creates",
                          "counterfactual_create_failures", "counterfactual_suppressed_starts",
                          "counterfactual_worker_dependence"}
        metric_fields |= {"audit_serialized_callbacks", "audit_output_elements",
                          "audit_output_coverage", "audit_owner_min_elements",
                          "audit_owner_max_elements", "audit_ownership_violations",
                          "audit_balanced_shards"}
        required_metric_fields = metric_fields
        if (not isinstance(metrics, dict) or
                not required_metric_fields <= set(metrics) <= metric_fields or
                any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in metrics.values())):
            raise ValueError("grader L3 metrics do not match trusted harness output")
        monitor_fields = {"version", "returncode", "timed_out", "wall_ns", "requested_harts",
                          "max_tasks", "tids_observed", "active_tids", "cpus_observed",
                          "active_cpus", "affinity_samples", "peak_rss_kb", "child_stdout",
                          "child_stderr", "pinned_affinities_observed",
                          "pinned_runtime_cpus", "running_cpus_observed",
                          "max_simultaneous_running_cpus"}
        if not isinstance(monitor, dict) or set(monitor) != monitor_fields:
            raise ValueError("grader L3 monitor does not match the trusted monitor schema")
        list_monitor = {"cpus_observed", "active_cpus", "affinity_samples",
                        "pinned_affinities_observed", "pinned_runtime_cpus",
                        "running_cpus_observed"}
        integer_monitor = monitor_fields - {
            "timed_out", *list_monitor, "child_stdout", "child_stderr"}
        if (any(isinstance(monitor[name], bool) or not isinstance(monitor[name], int)
                for name in integer_monitor) or not isinstance(monitor["timed_out"], bool) or
                any(not isinstance(monitor[name], list) for name in list_monitor) or
                not isinstance(monitor["child_stdout"], str) or
                not isinstance(monitor["child_stderr"], str)):
            raise ValueError("grader L3 monitor evidence has invalid field types")
        for name in list_monitor - {"affinity_samples"}:
            values = monitor[name]
            if (any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in values) or values != sorted(set(values))):
                raise ValueError("grader L3 monitor CPU evidence is not canonical")
        if (any(not isinstance(value, str) for value in monitor["affinity_samples"]) or
                monitor["affinity_samples"] != sorted(set(monitor["affinity_samples"]))):
            raise ValueError("grader L3 monitor affinity evidence is not canonical")
        if (monitor["version"] != 1 or monitor["requested_harts"] != harts or
                monitor["timed_out"] is not False):
            raise ValueError("grader L3 monitor identity differs from the requested run")
        parsed_metrics: dict[str, int] = {}
        for line in monitor["child_stdout"].splitlines():
            match = re.fullmatch(r"K1_METRIC ([a-z_]+) ([0-9]+)", line)
            if match:
                parsed_metrics[match.group(1)] = int(match.group(2))
        if parsed_metrics != metrics:
            raise ValueError("grader L3 metrics differ from retained monitor child stdout")
        l0 = l0_records.get((str(record["capsule"]), str(record["mode"])), {})
        metadata = l0.get("metadata") if isinstance(l0.get("metadata"), dict) else {}
        expected_affinity = "0" if harts == 1 else f"0-{harts-1}"
        expected_cpus = list(range(harts))
        expected_hart_mask = (1 << harts) - 1
        measured_calls = metrics["calls"]
        per_call_correctness = (
            measured_calls >= 20 and 1 <= metrics["audit_call"] <= 20 and
            metrics["correctness_checks"] == measured_calls + 1 and
            metrics["audit_wall_ns"] > 0 and metrics["audit_time_ticks"] > 0)
        counter_accounting = (
            metrics["pthread_create_attempts"] ==
            metrics["pthread_creates"] + metrics["pthread_create_failures"] and
            metrics["pthread_affinity_attempts"] ==
            metrics["pthread_affinity_successes"] + metrics["pthread_affinity_failures"])
        output_count = _require_int(
            expected_buffer_plan["output_count"], where="grader L3 output count")
        floor_count = output_count // harts
        ceil_count = (output_count + harts - 1) // harts
        shard_attribution = (
            output_count >= harts and metrics["audit_output_elements"] == output_count and
            metrics["audit_serialized_callbacks"] == harts - 1 and
            metrics["audit_output_coverage"] == output_count and
            metrics["audit_ownership_violations"] == 0 and
            metrics["audit_owner_min_elements"] >= floor_count and
            metrics["audit_owner_max_elements"] <= ceil_count and
            metrics["audit_balanced_shards"] == 1)
        if harts == 1:
            audit_attribution = counter_accounting and shard_attribution and all(
                metrics[name] == 0 for name in (
                "pinned_hart_mask", "worker_hart_mask", "productive_worker_hart_mask",
                "pthread_create_attempts", "pthread_creates", "pthread_create_failures",
                "pthread_completions", "pthread_affinity_attempts",
                "pthread_affinity_successes", "pthread_affinity_failures",
                "minimum_worker_cpu_ns", "counterfactual_create_attempts",
                "counterfactual_creates", "counterfactual_create_failures",
                "counterfactual_suppressed_starts")) and metrics[
                    "counterfactual_worker_dependence"] == 1
        else:
            worker_mask = expected_hart_mask & ~1
            audit_attribution = counter_accounting and shard_attribution and (
                metrics["pinned_hart_mask"] == expected_hart_mask and
                metrics["worker_hart_mask"] == worker_mask and
                metrics["productive_worker_hart_mask"] == worker_mask and
                metrics["pthread_create_attempts"] == harts - 1 and
                metrics["pthread_creates"] == harts - 1 and
                metrics["pthread_create_failures"] == 0 and
                metrics["pthread_completions"] == harts - 1 and
                metrics["pthread_affinity_attempts"] == harts and
                metrics["pthread_affinity_successes"] == harts and
                metrics["pthread_affinity_failures"] == 0 and
                metrics["minimum_worker_cpu_ns"] >= 100 and
                metrics["counterfactual_create_attempts"] == harts - 1 and
                metrics["counterfactual_creates"] == harts - 1 and
                metrics["counterfactual_create_failures"] == 0 and
                metrics["counterfactual_suppressed_starts"] == harts - 1 and
                metrics["counterfactual_worker_dependence"] == 1)
        expected_receipt = (
            f"MERLIN_TRUSTED_RESULT version=1 seed={record['seed']} "
            f"nonce={record['receipt_nonce']} memory=1 numeric=1")
        receipt_lines = [line for line in monitor["child_stdout"].splitlines()
                         if line.startswith("MERLIN_TRUSTED_RESULT ")]
        trusted_receipt = receipt_lines == [expected_receipt]
        expected_checks = {
            "exact_mode": metadata.get("actual_mode") == record["mode"],
            "no_fallback": metadata.get("fallback_used") is False,
            "numeric_correctness": monitor["returncode"] == 0 and trusted_receipt,
            "trusted_parent_receipt": trusted_receipt,
            "per_call_correctness": per_call_correctness,
            "csr_vlen": metrics.get("vlenb") == 32,
            "exact_affinity": metrics.get("affinity_count") == harts and
                              monitor["affinity_samples"] == [expected_affinity],
            "exact_task_count": 2 <= monitor["max_tasks"] <= harts + 1 and
                                monitor["tids_observed"] >= harts + 1,
            "active_harts": audit_attribution and ((
                monitor["pinned_affinities_observed"] == expected_cpus and
                monitor["pinned_runtime_cpus"] == expected_cpus
            ) if harts == 1 else True),
            "audit_attribution": audit_attribution,
            "wall_time": metrics.get("wall_ns", 0) > 0 and monitor["wall_ns"] > 0,
            "peak_rss": max(metrics.get("peak_rss_kb", 0), monitor["peak_rss_kb"]) > 0,
            "upload_integrity": record["remote_sha256"] == record["local_sha256"],
        }
        expected_status = "pass" if all(record["checks"].values()) else "fail"
        if (ssh_returncode != 0 or record["checks"] != expected_checks or
                record["status"] != expected_status):
            raise ValueError("grader L3 checks/status evidence disagree with retained K1 evidence")
        return
    raise ValueError(f"unknown grader level {level_name}")


def _validate_early_failure(
        level_name: str, record: dict[str, Any], *, l0_outcomes: dict[tuple[str, str], str],
        expected_buffer_plan: dict[str, Any] | None = None
        ) -> None:
    """Validate the exact grader-produced negative shape and its upstream cause.

    Reasons are not evidence by themselves.  Dependent L1--L3 failures must point to a failed L0
    record for the same capsule/mode; independent compile/link failures retain the producer's
    return code/timing/log fields.  Missing tools, transport, upload, and monitor failures are not
    accepted here: the grader emits them as campaign-invalid ``error`` outcomes.
    """
    base = {"capsule", "family", "mode", "status", "reason"}
    reason = record["reason"]
    capsule, mode = str(record["capsule"]), str(record["mode"])
    if level_name == "L0":
        if reason in {"C syntax check timed out", "MLIR verifier timed out"}:
            expected = {
                "capsule", "family", "mode", "ok", "reason", "returncode",
                "wall_seconds", "stdout_tail", "stderr_tail", "checks", "metadata",
                "source_sha256", "source_size_bytes", "input_mlir_sha256",
                "lowered_mlir_sha256", "buffer_plan", "timeout_seconds",
                "timed_out_stage", "syntax_returncode", "verifier_returncode",
                "syntax_stderr", "verifier_stderr",
            }
            _require_exact_record_fields(record, expected, where="grader L0 tool timeout")
            if (record["ok"] is not False or record["returncode"] != 0 or
                    record["timeout_seconds"] != 60 or
                    not isinstance(record["checks"], dict) or
                    set(record["checks"]) != _LEVEL_CHECKS["L0"] or
                    record["checks"].get("mlir_verifier") is not False or
                    record["buffer_plan"] != expected_buffer_plan):
                raise ValueError("grader L0 tool timeout evidence is inconsistent")
            expected_stage = ("c_syntax" if reason == "C syntax check timed out"
                              else "mlir_verifier")
            if (record["timed_out_stage"] != expected_stage or
                    record["checks"].get("c_syntax") is not
                    (expected_stage == "mlir_verifier") or
                    record["verifier_returncode"] is not None or
                    (record["syntax_returncode"] is not None if expected_stage == "c_syntax"
                     else _require_int(record["syntax_returncode"],
                                       where="pre-verifier syntax returncode") != 0)):
                raise ValueError("grader L0 tool timeout stage evidence is inconsistent")
            if any(not isinstance(record[name], str) for name in (
                    "stdout_tail", "stderr_tail", "syntax_stderr", "verifier_stderr")):
                raise ValueError("grader L0 tool timeout log tails must be text")
            _finite_nonnegative(record["wall_seconds"], where="grader L0 timeout wall_seconds")
            if (not isinstance(record["metadata"], dict) or
                    not _is_digest(record["source_sha256"]) or
                    not _is_digest(record["input_mlir_sha256"]) or
                    not _is_digest(record["lowered_mlir_sha256"]) or
                    _require_int(record["source_size_bytes"], where="grader L0 source size") < 0):
                raise ValueError("grader L0 tool timeout lacks artifact evidence")
            return
        if reason == "compiler invocation timed out":
            _require_exact_record_fields(
                record, {"capsule", "family", "mode", "ok", "reason", "timeout_seconds",
                         "wall_seconds"}, where="grader L0 compiler timeout")
            if (record["ok"] is not False or record["timeout_seconds"] != 300):
                raise ValueError("grader L0 compiler timeout evidence is inconsistent")
            _finite_nonnegative(record["wall_seconds"], where="grader L0 timeout wall_seconds")
            return
        base = {"capsule", "family", "mode", "ok", "reason", "returncode",
                "wall_seconds", "stdout_tail", "stderr_tail"}
        _require_exact_record_fields(record, base, where="grader L0 early failure")
        if record["ok"] is not False:
            raise ValueError("grader L0 early failure must set ok=false")
        returncode = _require_int(record["returncode"], where="grader L0 returncode")
        _finite_nonnegative(record["wall_seconds"], where="grader L0 wall_seconds")
        if not isinstance(record["stdout_tail"], str) or not isinstance(record["stderr_tail"], str):
            raise ValueError("grader L0 early failure must retain stdout/stderr text")
        if reason == "compiler invocation failed":
            if returncode == 0:
                raise ValueError("grader L0 compiler failure has a zero return code")
        else:
            if returncode != 0:
                raise ValueError("grader L0 post-invocation failure has a non-zero return code")
            omitted = {
                "compiler omitted outputs ['kernel.c']",
                "compiler omitted outputs ['lowered.mlir']",
                "compiler omitted outputs ['metadata.json']",
                "compiler omitted outputs ['kernel.c', 'lowered.mlir']",
                "compiler omitted outputs ['kernel.c', 'metadata.json']",
                "compiler omitted outputs ['lowered.mlir', 'metadata.json']",
                "compiler omitted outputs ['kernel.c', 'lowered.mlir', 'metadata.json']",
            }
            if not (reason in omitted or reason.startswith("metadata is invalid:")):
                raise ValueError("grader L0 failure reason is not producer-reachable")
        return

    upstream_reason = "L0 RVV artifact failed" if level_name == "L2" else f"L0 {mode} artifact failed"
    if reason == upstream_reason:
        extra = {"tail_case"} if level_name == "L2" else ({"harts"} if level_name == "L3" else set())
        _require_exact_record_fields(record, base | extra, where=f"grader {level_name} upstream failure")
        if l0_outcomes.get((capsule, mode)) != "fail":
            raise ValueError(
                f"grader {level_name} claims an L0 dependency failure but matching L0 passed")
        if level_name == "L2" and not isinstance(record["tail_case"], bool):
            raise ValueError("grader L2 tail_case must be boolean")
        if level_name == "L3" and _require_int(
                record["harts"], where="grader L3 harts") < 1:
            raise ValueError("grader L3 harts must be positive")
        return

    if level_name == "L1" and reason in {
            "trusted native build timed out", "trusted native build failed"}:
        extra = {"build_logs", "failed_stage_index"}
        if reason == "trusted native build timed out":
            extra.add("timeout_seconds")
        else:
            extra |= {"build_wall_seconds", "build_stderr_tail", "build_returncode"}
        _require_exact_record_fields(
            record, base | extra, where="grader L1 build failure")
        stage = _require_int(record["failed_stage_index"],
                             where="grader L1 failed_stage_index")
        logs = _validate_build_logs(record["build_logs"], where="grader L1 build_logs")
        if stage < 0:
            raise ValueError("grader L1 failed_stage_index must be non-negative")
        if reason == "trusted native build timed out":
            if record["timeout_seconds"] != 120 or len(logs) != stage:
                raise ValueError("grader L1 timeout evidence is inconsistent with its failed stage")
        else:
            _finite_nonnegative(record["build_wall_seconds"],
                                where="grader L1 build_wall_seconds")
            _require_int(record["build_returncode"],
                         where="grader L1 build_returncode", nonzero=True)
            if (not isinstance(record["build_stderr_tail"], str) or
                    len(logs) != stage + 1 or not logs or logs[-1]["returncode"] == 0):
                raise ValueError("grader L1 build-failure evidence is inconsistent")
            if (record["build_returncode"] != logs[-1]["returncode"] or
                    record["build_stderr_tail"] != logs[-1]["stderr_tail"] or
                    record["build_wall_seconds"] < sum(log["wall_seconds"] for log in logs)):
                raise ValueError("grader L1 aggregate build evidence contradicts its stage logs")
        return

    if level_name == "L1" and reason == "trusted native execution timed out":
        expected = base | {"build_wall_seconds", "build_stderr_tail", "build_returncode",
                           "trials", "timed_out_trial_index", "timed_out_seed",
                           "timeout_seconds"}
        _require_exact_record_fields(record, expected, where="grader L1 execution timeout")
        if (_require_int(record["build_returncode"], where="grader L1 build_returncode") != 0 or
                not isinstance(record["build_stderr_tail"], str) or
                record["timeout_seconds"] != 45):
            raise ValueError("grader L1 execution timeout lacks a successful retained build")
        _finite_nonnegative(record["build_wall_seconds"], where="grader L1 build_wall_seconds")
        trials = record["trials"]
        index = _require_int(record["timed_out_trial_index"], where="grader L1 timeout trial")
        if (_require_int(record["timed_out_seed"], where="grader L1 timeout seed") < 1 or
                not isinstance(trials, list) or index != len(trials) or not 0 <= index < 3):
            raise ValueError("grader L1 timeout trial index is inconsistent")
        for trial_index, trial in enumerate(trials):
            if not isinstance(trial, dict) or set(trial) != {
                    "seed", "returncode", "wall_seconds", "stdout_tail", "stderr_tail", "ok"}:
                raise ValueError(f"grader L1 completed trial {trial_index} is malformed")
            if (_require_int(trial["seed"], where="grader L1 trial seed") < 1 or
                    not isinstance(trial["ok"], bool) or not isinstance(trial["stdout_tail"], str) or
                    not isinstance(trial["stderr_tail"], str)):
                raise ValueError(f"grader L1 completed trial {trial_index} has invalid types")
            _require_int(trial["returncode"], where="grader L1 trial returncode")
            _finite_nonnegative(trial["wall_seconds"], where="grader L1 trial wall_seconds")
        return

    if level_name == "L2" and reason in {"Spike build timed out", "Spike build failed"}:
        extra = {"tail_case", "build_logs", "failed_stage_index"}
        if reason == "Spike build timed out":
            extra.add("timeout_seconds")
        _require_exact_record_fields(record, base | extra, where="grader L2 build failure")
        if not isinstance(record["tail_case"], bool):
            raise ValueError("grader L2 tail_case must be boolean")
        stage = _require_int(record["failed_stage_index"], where="grader L2 failed_stage_index")
        logs = _validate_build_logs(record["build_logs"], where="grader L2 build_logs")
        if stage < 0:
            raise ValueError("grader L2 failed_stage_index must be non-negative")
        if reason == "Spike build timed out":
            if record["timeout_seconds"] != 120 or len(logs) != stage:
                raise ValueError("grader L2 timeout evidence is inconsistent with its failed stage")
        elif len(logs) != stage + 1 or not logs or logs[-1]["returncode"] == 0:
            raise ValueError("grader L2 failure logs do not end at a non-zero failed stage")
        return

    if level_name == "L2" and reason == "Spike execution timed out":
        expected = base | {"tail_case", "seed", "vector_instructions", "build_logs",
                           "vector_dataflow", "kernel_text_sha256",
                           "timeout_seconds", "wall_seconds"}
        _require_exact_record_fields(record, expected, where="grader L2 execution timeout")
        if (not isinstance(record["tail_case"], bool) or
                _require_int(record["seed"], where="grader L2 seed") < 1 or
                not isinstance(record["vector_instructions"], list) or
                any(not isinstance(value, str) for value in record["vector_instructions"]) or
                not isinstance(record["vector_dataflow"], dict) or
                (record["kernel_text_sha256"] is not None and
                 not _is_digest(record["kernel_text_sha256"])) or
                record["timeout_seconds"] != 180):
            raise ValueError("grader L2 execution timeout evidence is malformed")
        logs = _validate_build_logs(record["build_logs"], where="grader L2 build_logs")
        if len(logs) != 6 or any(log["returncode"] != 0 for log in logs):
            raise ValueError("grader L2 execution timeout lacks six successful build stages")
        _finite_nonnegative(record["wall_seconds"], where="grader L2 timeout wall_seconds")
        return

    if level_name == "L3" and reason in {"K1 cross-build timed out", "K1 cross-build failed"}:
        extra = {"harts", "build_logs", "failed_stage_index"}
        if reason == "K1 cross-build timed out":
            extra.add("timeout_seconds")
        else:
            extra |= {"build_wall_seconds", "build_stderr_tail", "build_returncode"}
        _require_exact_record_fields(record, base | extra, where="grader L3 build failure")
        if _require_int(record["harts"], where="grader L3 harts") < 1:
            raise ValueError("grader L3 harts must be positive")
        stage = _require_int(record["failed_stage_index"],
                             where="grader L3 failed_stage_index")
        logs = _validate_build_logs(record["build_logs"], where="grader L3 build_logs")
        if stage < 0:
            raise ValueError("grader L3 failed_stage_index must be non-negative")
        if reason == "K1 cross-build timed out":
            if record["timeout_seconds"] != 180 or len(logs) != stage:
                raise ValueError("grader L3 timeout evidence is inconsistent with its failed stage")
        else:
            _finite_nonnegative(record["build_wall_seconds"], where="grader L3 build_wall_seconds")
            _require_int(record["build_returncode"], where="grader L3 build_returncode", nonzero=True)
            if (not isinstance(record["build_stderr_tail"], str) or
                    len(logs) != stage + 1 or not logs or logs[-1]["returncode"] == 0):
                raise ValueError("grader L3 build-failure evidence is inconsistent")
            if (record["build_returncode"] != logs[-1]["returncode"] or
                    record["build_stderr_tail"] != logs[-1]["stderr_tail"] or
                    record["build_wall_seconds"] < sum(log["wall_seconds"] for log in logs)):
                raise ValueError("grader L3 aggregate build evidence contradicts its stage logs")
        return
    raise ValueError(f"grader {level_name} record has no recognized substantive failure evidence")


def _grader_status_from_levels(grader_result: dict[str, Any], *, layout: dict[str, Any]) -> str:
    """Derive the terminal grader outcome from the exact heldout roster and L0--L3 checks."""
    if (grader_result.get("version") != 1 or
            grader_result.get("implemented_levels") != ["L0", "L1", "L2", "L3"] or
            grader_result.get("selected_capsules") != layout["selected"] or
            grader_result.get("tail_capsules") != layout["tails"] or
            grader_result.get("multicore_capsule") != layout["multicore"]):
        raise ValueError("grader result does not bind the exact frozen heldout selection")
    levels = grader_result.get("levels")
    if not isinstance(levels, dict) or list(levels) != ["L0", "L1", "L2", "L3"]:
        raise ValueError("grader result must retain exact ordered L0--L3 evidence")
    derived_levels: list[str] = []
    derived_records: dict[str, dict[tuple[str, str], str]] = {}
    l0_evidence: dict[tuple[str, str], dict[str, Any]] = {}
    for level_name, level in levels.items():
        if not isinstance(level, dict) or not isinstance(level.get("records"), list) or not level["records"]:
            raise ValueError(f"grader {level_name} has no substantive records")
        expected_records = layout["records"][level_name]
        actual_records = [(record.get("capsule"), record.get("mode"))
                          for record in level["records"] if isinstance(record, dict)]
        if actual_records != expected_records or len(set(actual_records)) != len(actual_records):
            raise ValueError(f"grader {level_name} records differ from the frozen heldout roster")
        record_statuses = []
        for record in level["records"]:
            checks = record.get("checks") if isinstance(record, dict) else None
            early_tool_timeout = (
                level_name == "L0" and record.get("reason") in {
                    "C syntax check timed out", "MLIR verifier timed out"})
            expected_row = layout["rows"][record["capsule"]]
            if record.get("family") != expected_row["family"]:
                raise ValueError(f"grader {level_name} record family differs from heldout")
            if isinstance(checks, dict) and checks and not early_tool_timeout:
                if (set(checks) != _LEVEL_CHECKS[level_name] or
                        any(not isinstance(value, bool) for value in checks.values())):
                    raise ValueError(f"grader {level_name} record has no closed required checks")
                _validate_check_record(
                    level_name, record, expected_row=expected_row,
                    expected_buffer_plan=layout["buffer_plans"][record["capsule"]],
                    l0_records=l0_evidence)
                derived = "pass" if all(checks.values()) else "fail"
            else:
                reason = record.get("reason")
                allowed = _EARLY_FAILURE_PREFIXES.get(level_name, ())
                declared_failure = (record.get("status") == "fail" or record.get("ok") is False)
                if (not declared_failure or not isinstance(reason, str) or
                        not any(reason.startswith(prefix) for prefix in allowed)):
                    raise ValueError(f"grader {level_name} record has no recognized failure evidence")
                _validate_early_failure(
                    level_name, record, l0_outcomes=derived_records.get("L0", {}),
                    expected_buffer_plan=layout["buffer_plans"][record["capsule"]])
                derived = "fail"
            declared = record.get("status")
            if declared is None and isinstance(record.get("ok"), bool):
                declared = "pass" if record["ok"] else "fail"
            if declared != derived:
                raise ValueError(f"grader {level_name} record status disagrees with its checks")
            record_statuses.append(derived)
            derived_records.setdefault(level_name, {})[(record["capsule"], record["mode"])] = derived
            if level_name == "L0":
                l0_evidence[(record["capsule"], record["mode"])] = record
        if level_name == "L0":
            source_change = level.get("scalar_rvv_source_change")
            if (not isinstance(source_change, dict) or set(source_change) != set(layout["selected"]) or
                    any(not isinstance(value, bool) for value in source_change.values())):
                raise ValueError("grader L0 scalar/RVV source-change evidence is incomplete")
            rederived_source_change = {
                capsule: (l0_evidence[(capsule, "scalar")].get("source_sha256") !=
                          l0_evidence[(capsule, "rvv")].get("source_sha256"))
                for capsule in layout["selected"]
            }
            if source_change != rederived_source_change:
                raise ValueError("grader L0 scalar/RVV source-change mapping is not rederived")
            if not all(source_change.values()):
                record_statuses.append("fail")
        level_status = "pass" if all(value == "pass" for value in record_statuses) else "fail"
        if level.get("status") != level_status:
            raise ValueError(f"grader {level_name} status disagrees with its records")
        derived_levels.append(level_status)
    if (levels["L1"].get("authority") != "native_scalar_reference_with_asan_ubsan_and_guards" or
            levels["L2"].get("authority") != "spike_rv64gcv_vlen256" or
            levels["L3"].get("authority") != "spacemit_k1_linux_csr_and_proc_monitor" or
            set((levels["L0"].get("scalar_rvv_source_change") or {})) != set(layout["selected"])):
        raise ValueError("grader level authorities/coverage metadata are incomplete")
    status = "pass" if all(value == "pass" for value in derived_levels) else "fail"
    if grader_result.get("status") != status:
        raise ValueError("grader top-level status disagrees with L0--L3 evidence")
    return status


def _validate_then_publish(path: Path, value: dict[str, Any], *,
                           before_publish: Callable[[], None] | None = None
                           ) -> HostExperimentSpec:
    """Round-trip and preflight private bytes before making the final path visible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(value, stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        completed = HostExperimentSpec.from_yaml(temporary)
        check = completed.preflight(check_environment=False, require_frozen=True)
        if not check.ready:
            raise ValueError(f"completed campaign round-trip is NO_GO: {check.to_dict()}")
        if before_publish is not None:
            before_publish()
        os.replace(temporary, path)
        return HostExperimentSpec.from_yaml(path)
    finally:
        temporary.unlink(missing_ok=True)


def _expected_launch_plan(spec: HostExperimentSpec, launch: dict[str, Any]
                          ) -> dict[tuple[str, int], dict[str, Any]]:
    """Derive every launch identity from the frozen run-plan convention.

    The launcher assigns one campaign id and a single base seed; repeat ``r`` is always base+r.
    An arm/repeat cannot substitute an arbitrary directory or seed after the protocol is frozen.
    """
    campaign = str(launch.get("campaign_run_id", ""))
    if not campaign:
        raise ValueError("launch record has no campaign_run_id")
    scheme = "{campaign_run_id}__{arm}__r{repeat:02d}__seed{seed:03d}"
    if launch.get("run_id_scheme") != scheme:
        raise ValueError("launch record has no recognized deterministic run_id_scheme")
    launch_seed = launch.get("launch_seed")
    if isinstance(launch_seed, bool):
        raise ValueError("launch_seed must be an integer")
    try:
        launch_seed = int(launch_seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("launch record has no integer launch_seed") from exc
    if launch_seed != int(spec.agent["launch_seed"]):
        raise ValueError("launch_seed differs from the frozen campaign seed")
    plan: dict[tuple[str, int], dict[str, Any]] = {}
    for frozen in spec.agent["launch_plan"]:
        arm, repeat, seed = str(frozen["arm"]), int(frozen["repeat"]), int(frozen["seed"])
        ordinal = int(frozen["ordinal"])
        run_id = scheme.format(campaign_run_id=campaign, arm=arm, repeat=repeat, seed=seed)
        plan[arm, repeat] = {"ordinal": ordinal, "arm": arm, "repeat": repeat,
                             "seed": seed, "run_id": run_id}
    return plan


def _verify_launch_rows(*, label: str, rows: list[Any], expected: dict[tuple[str, int], dict[str, Any]],
                        runs_root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    if len(rows) != len(expected):
        raise ValueError(f"{label} must contain the exact four arms x four blocks frozen launch plan")
    if [row.get("ordinal") if isinstance(row, dict) else None for row in rows] != list(
            range(len(expected))):
        raise ValueError(f"{label} differs from the exact frozen launch chronology")
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{label} row is not an object")
        key = str(row.get("arm", "")), int(row.get("repeat", -1))
        frozen = expected.get(key)
        if frozen is None:
            raise ValueError(f"{label} contains an unplanned arm/repeat: {key}")
        if key in result:
            raise ValueError(f"{label} duplicates a frozen arm/repeat: {key}")
        for name in ("ordinal", "arm", "repeat", "seed", "run_id"):
            if row.get(name) != frozen[name]:
                raise ValueError(f"{label} {name} differs from deterministic frozen launch plan")
        run_dir = Path(str(row.get("run_dir", ""))).resolve()
        if run_dir != runs_root / frozen["run_id"]:
            raise ValueError(f"{label} run_dir differs from deterministic frozen launch plan")
        result[key] = row
    if set(result) != set(expected):
        raise ValueError(f"{label} does not cover the exact frozen 4x4 launch plan")
    return result


def _validate_block_boundaries(spec: HostExperimentSpec, launch: dict[str, Any],
                               launch_record: Path) -> list[dict[str, Any]]:
    """Validate four retained washout/requalification receipts before scoring carryover."""
    records = launch.get("block_boundaries")
    repeats = int(spec.agent["repeats"])
    if not isinstance(records, list) or len(records) != repeats:
        raise ValueError("launch must retain one K1 requalification receipt per Williams block")
    root = launch_record.parent / "block_boundaries"
    expected_files = [f"{block:02d}.json" for block in range(repeats)]
    actual_files = sorted(path.name for path in root.iterdir()) if root.is_dir() else []
    if actual_files != expected_files or any(path.is_symlink() for path in root.iterdir()):
        raise ValueError("block-boundary receipt directory differs from the frozen Williams design")
    environment = dict(spec.search_space_config()["board_environment"])
    limit = int(environment["settle_attempts"])
    interval = float(environment["settle_interval_seconds"])
    validated = []
    for block, record in enumerate(records):
        path = (root / f"{block:02d}.json").resolve()
        expected_record = {
            "block": block,
            "first_ordinal": block * len(spec.arms),
            "path": str(path),
            "sha256": _sha256(path),
        }
        if record != expected_record:
            raise ValueError("launch block-boundary index differs from retained receipt bytes")
        receipt = _json(path)
        attempts = receipt.get("attempts")
        if (receipt.get("version") != 1 or
                receipt.get("authority") != "frozen_k1_board_environment_gate" or
                receipt.get("block") != block or
                receipt.get("first_ordinal") != block * len(spec.arms) or
                receipt.get("mandatory_washout_seconds") != interval or
                receipt.get("stabilization_attempt_limit") != limit or
                receipt.get("board_environment") != environment or
                not isinstance(attempts, list) or not 1 <= len(attempts) <= limit or
                receipt.get("qualifying_attempt_index") != len(attempts) - 1 or
                receipt.get("ready") is not True):
            raise ValueError("block-boundary receipt does not implement the frozen K1 washout gate")
        qualifying = attempts[-1]
        evidence = qualifying.get("evidence") if isinstance(qualifying, dict) else None
        if (qualifying.get("verdict") != "GO" or qualifying.get("ready") is not True or
                not isinstance(evidence, dict) or
                evidence.get("protocol_inputs_sha256") != spec.freeze["protocol_inputs_sha256"] or
                evidence.get("analysis_plan_sha256") != spec.analysis["sha256"] or
                evidence.get("k1_board_state_ready") is not True or
                not isinstance(evidence.get("k1_board_state_probe"), dict)):
            raise ValueError("block-boundary receipt has no exact qualifying board-state evidence")
        validated.append(expected_record)
    return validated


def _complete_campaign_reserved(
        source: Path, launch_record: Path, output: Path) -> HostExperimentSpec:
    source, launch_record, output = source.resolve(), launch_record.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite campaign completion: {output}")
    spec = HostExperimentSpec.from_yaml(source)
    if spec.status != "protocol_frozen":
        raise ValueError("campaign completion accepts only a protocol_frozen source")
    protocol_check = spec.preflight(check_environment=False, require_frozen=True)
    if not protocol_check.ready:
        raise ValueError(f"frozen protocol no longer verifies: {protocol_check.to_dict()}")
    expected_arm_inputs = protocol_check.evidence.get("arm_workspace_inputs", {})
    grader_layout = _expected_grader_layout(spec)
    launch = _json(launch_record)
    launch_version = launch.get("version")
    if (launch_version not in {2, 3} or launch.get("sequential") is not True or
            launch.get("terminal_failure_policy") != "record_and_continue" or
            launch.get("retry_terminal_outcomes") is not False):
        raise ValueError("launch record does not implement the frozen record-and-continue policy")
    if launch.get("protocol_inputs_sha256") != spec.freeze.get("protocol_inputs_sha256"):
        raise ValueError("launch record belongs to a different frozen protocol")
    if launch.get("environment_manifest_sha256") != spec.environment.get("sha256"):
        raise ValueError("launch record belongs to a different frozen environment")
    if (launch.get("analysis_plan_sha256") != spec.analysis.get("sha256") or
            launch.get("provider_sampling_seeded") is not False or
            launch.get("launch_seed_role") != "campaign_metadata_only_not_provider_sampling"):
        raise ValueError("launch record differs from the frozen unseeded-provider analysis plan")
    runs_root = Path(str(launch.get("runs_root", ""))).resolve()
    if not runs_root.is_dir():
        raise ValueError("launch record has no existing canonical runs_root")
    campaign_run_id = str(launch.get("campaign_run_id", ""))
    _assert_campaign_not_excluded(runs_root, campaign_run_id)
    canonical_launch = runs_root / campaign_run_id / "contracts" / "launch.json"
    if launch_record != canonical_launch or not canonical_launch.is_file():
        raise ValueError("launch record is outside its canonical launcher run directory")
    claim_path = Path(str(launch.get("authorization_claim", ""))).resolve()
    expected_claim = runs_root / ".protocol_claims" / (
        f"{spec.freeze['protocol_inputs_sha256']}.json")
    claim = _json(claim_path) if claim_path == expected_claim and claim_path.is_file() else {}
    if (claim_path != expected_claim or not claim_path.is_file() or
            launch.get("authorization_claim_sha256") != _sha256(claim_path) or
            claim.get("version") != 1 or claim.get("status") != "bound" or
            claim.get("protocol_inputs_sha256") != spec.freeze.get("protocol_inputs_sha256") or
            claim.get("environment_manifest_sha256") != spec.environment.get("sha256") or
            claim.get("analysis_plan_sha256") != spec.analysis.get("sha256") or
            claim.get("campaign_run_id") != campaign_run_id or
            claim.get("spec_path") != str(source) or
            claim.get("spec_sha256") != _sha256(source)):
        raise ValueError("launch record has no exact bound one-shot protocol authorization claim")

    planned = launch.get("planned")
    results = launch.get("results")
    if not isinstance(planned, list) or not isinstance(results, list):
        raise ValueError("launch record requires planned and results lists")
    expected_plan = _expected_launch_plan(spec, launch)
    block_boundaries = _validate_block_boundaries(spec, launch, launch_record)
    cells = claim_path.with_name(f"{claim_path.stem}.cells")
    expected_cell_files = [f"{ordinal:02d}.consumed.json" for ordinal in range(len(expected_plan))]
    actual_cell_files = sorted(path.name for path in cells.iterdir()) if cells.is_dir() else []
    if actual_cell_files != expected_cell_files or any(path.is_symlink() for path in cells.iterdir()):
        raise ValueError("one-shot authorization does not contain every exact consumed launch cell")
    for frozen in expected_plan.values():
        receipt = _json(cells / f"{int(frozen['ordinal']):02d}.consumed.json")
        expected_receipt = {
            "version": 1, "status": "authorized",
            "protocol_inputs_sha256": spec.freeze["protocol_inputs_sha256"],
            "environment_manifest_sha256": spec.environment["sha256"],
            "analysis_plan_sha256": spec.analysis["sha256"],
            "campaign_run_id": campaign_run_id, "ordinal": int(frozen["ordinal"]),
            "arm": frozen["arm"], "repeat": int(frozen["repeat"]),
            "seed": int(frozen["seed"]), "run_id": frozen["run_id"],
        }
        if receipt != expected_receipt:
            raise ValueError("consumed arm authorization differs from the frozen launch cell")
    planned_by_key = _verify_launch_rows(
        label="launch planned", rows=planned, expected=expected_plan, runs_root=runs_root)
    results_by_key = _verify_launch_rows(
        label="launch results", rows=results, expected=expected_plan, runs_root=runs_root)

    order = {arm.id: arm.order for arm in spec.arms}
    rows = []
    aggregate_tokens = {name: 0 for name in _TOKEN_FIELDS}
    aggregate_time = {name: 0.0 for name in (
        "active_wall_seconds", "grader_wall_seconds", "trusted_search_wall_seconds",
        "wall_seconds")}
    aggregate_tools = 0
    seen_run_ids: set[str] = set()
    for key in sorted(expected_plan, key=lambda value: (order[value[0]], value[1])):
        result = results_by_key[key]
        planned_row = planned_by_key[key]
        if result.get("returncode") not in {0, 1} or result.get("run_identity_ok") is not True:
            raise ValueError(f"campaign run has no complete pass/fail observation: {result}")
        if launch_version == 3 and (
                result.get("attempted") is not True or result.get("executed") is not True or
                result.get("cell_status") != "executed" or
                result.get("terminal_class") not in {
                    "graded_pass", "graded_fail", "treatment_search_fail",
                    "treatment_build_fail", "treatment_agent_fail"}):
            raise ValueError(f"campaign has a v3 non-executed or harness-invalid cell: {result}")
        run_id = str(result.get("run_id", ""))
        run_dir = Path(str(result.get("run_dir", ""))).resolve()
        if (not run_id or run_id in seen_run_ids or run_dir.name != run_id or
                run_dir.parent != runs_root):
            raise ValueError("campaign run IDs must be unique and match their directory")
        seen_run_ids.add(run_id)
        run_record = _json(run_dir / "run_record.json")
        summary = _json(run_dir / "metrics" / "summary_metrics.json")
        preflight_path = run_dir / "contracts" / "preflight.json"
        preflight = _json(preflight_path)
        input_lock_path = run_dir / "contracts" / "workspace_input_lock.json"
        input_lock = _json(input_lock_path)
        input_audit_path = run_dir / "contracts" / "workspace_input_audit.json"
        input_audit = _json(input_audit_path)
        canonical_lock = json.dumps(
            input_lock, sort_keys=True, separators=(",", ":")).encode("utf-8")
        input_lock_sha = hashlib.sha256(canonical_lock).hexdigest()
        arm_input = (preflight.get("evidence", {}).get("arm_workspace_inputs", {}).get(
            str(result["arm"]), {}))
        frozen_environment = preflight.get("evidence", {}).get("frozen_environment", {})
        if (preflight.get("ready") is not True or preflight.get("errors") != [] or
                preflight.get("blockers") != [] or
                preflight.get("evidence", {}).get("protocol_inputs_sha256") !=
                spec.freeze.get("protocol_inputs_sha256") or
                not isinstance(frozen_environment, dict) or
                frozen_environment.get("manifest_sha256") != spec.environment.get("sha256") or
                frozen_environment.get("capture_complete") is not True or
                frozen_environment.get("local_identity_matches") is not True or
                frozen_environment.get("k1_identity_matches") is not True or
                arm_input != expected_arm_inputs.get(str(result["arm"])) or
                arm_input.get("input_lock_sha256") != input_lock_sha or
                arm_input.get("file_count") != len(input_lock) or
                input_audit.get("ok") is not True or
                input_audit.get("input_lock_sha256") != input_lock_sha or
                input_audit.get("changed_or_missing") != [] or input_audit.get("unexpected") != []):
            raise ValueError(f"run preflight/workspace identity differs from frozen protocol: {run_id}")
        compiler_seal_path = run_dir / "contracts" / "compiler_seal.json"
        search_seal_path = run_dir / "contracts" / "trusted_search_seal.json"
        compiler_seal = _json(compiler_seal_path)
        search_seal = _json(search_seal_path)
        reconciliation_path = run_dir / "metrics" / "codex_reconciliation.json"
        reconciliation = _json(reconciliation_path)
        token_path = run_dir / "metrics" / "token_ledger.jsonl"
        tools_path = run_dir / "agent" / "tools.jsonl"
        grader_result_path = run_dir / "metrics" / "grader_result.json"
        grader_result = _json(grader_result_path)
        driver_timing_path = run_dir / "metrics" / "driver_wall_timing.json"
        driver_timing = _json(driver_timing_path)
        archive = run_dir / "artifacts" / "compiler_submission"
        search_required = "deterministic_candidate_search" in next(
            arm for arm in spec.arms if arm.id == result["arm"]).capabilities
        terminal_path = run_dir / "contracts" / "terminal_outcome.json"
        if launch_version == 3:
            terminal = _json(terminal_path)
            outcome = str(terminal.get("terminal_class", ""))
            if (terminal.get("version") != 1 or terminal.get("run_id") != run_id or
                    terminal.get("arm") != result["arm"] or
                    outcome not in {"graded_pass", "graded_fail", "treatment_search_fail",
                                    "treatment_build_fail", "treatment_agent_fail"} or
                    result.get("terminal_class") != outcome or
                    summary.get("terminal_class") != outcome or
                    terminal.get("paper_evidence_eligible") is not
                    (outcome in {"graded_pass", "graded_fail"}) or
                    terminal.get("promotion_eligible") is not (outcome == "graded_pass")):
                raise ValueError(f"run has no valid typed terminal outcome: {run_id}")
            expected_terminal_checks = {
                "agent_success": summary.get("agent_success"),
                "agent_failure_class": summary.get("agent_failure_class"),
                "workspace_input_audit": input_audit.get("ok"),
                "aet_reconciled": reconciliation.get("ok"),
                "trusted_search_status": search_seal.get("status"),
                "compiler_seal_status": compiler_seal.get("status"),
                "compiler_seal_failure_class": compiler_seal.get("failure_class"),
                "grader_returncode": summary.get("grader_returncode"),
                "grader_status": grader_result.get("status"),
                "grader_failure_class": grader_result.get("failure_class"),
            }
            if terminal.get("checks") != expected_terminal_checks:
                raise ValueError(f"typed terminal checks differ from retained evidence: {run_id}")
        else:
            terminal = None
            outcome = ""
        required_integrity = {
            "workspace_inputs_unchanged": True, "aet_reconciled": True,
            "billing_mode": spec.agent["billing"],
        }
        if any(summary.get(name) != value for name, value in required_integrity.items()):
            raise ValueError(f"run summary has no complete reconciled observation: {run_id}")
        grader_status = summary.get("grader_status")
        grader_returncode = summary.get("grader_returncode")
        agent_noncompletion = (
            outcome == "treatment_agent_fail" and summary.get("agent_success") is False)
        search_treatment_failure = (
            not agent_noncompletion and
            outcome in {"treatment_search_fail", "treatment_build_fail",
                        "treatment_agent_fail"} and search_seal.get("status") == "fail")
        if search_treatment_failure:
            if (not search_required or result.get("returncode") != 1 or
                    summary.get("trusted_search_status") != "fail" or
                    summary.get("compiler_seal_status") != "not_run" or
                    grader_status != "not_run" or grader_returncode != 2 or
                    search_seal.get("status") != "fail" or
                    search_seal.get("failure_class") != outcome or
                    compiler_seal.get("status") != "not_run" or
                    grader_result.get("status") != "not_run" or archive.exists()):
                raise ValueError(f"typed treatment search failure is inconsistent: {run_id}")
        elif agent_noncompletion:
            if (result.get("returncode") != 1 or
                    search_seal.get("status") != ("fail" if search_required else "not_required") or
                    (search_required and search_seal.get("failure_class") not in {
                        "treatment_search_fail", "treatment_build_fail", "treatment_agent_fail"}) or
                    compiler_seal.get("status") != "not_run" or
                    compiler_seal.get("failure_class") != "treatment_agent_fail" or
                    grader_status != "not_run" or grader_returncode != 2 or
                    grader_result.get("status") != "not_run" or archive.exists()):
                raise ValueError(f"typed reconciled agent failure is inconsistent: {run_id}")
        elif outcome in {"treatment_build_fail", "treatment_agent_fail"}:
            if (summary.get("agent_success") is not True or result.get("returncode") != 1 or
                    compiler_seal.get("status") != "sealed" or
                    grader_returncode != 1 or grader_status != outcome or
                    grader_result.get("failure_class") != outcome or not archive.is_dir()):
                raise ValueError(f"typed grader treatment failure is inconsistent: {run_id}")
        else:
            if _grader_status_from_levels(grader_result, layout=grader_layout) != grader_status:
                raise ValueError(f"run summary and grader outcome disagree: {run_id}")
            derived = ("graded_pass" if
                       (grader_status, grader_returncode, result.get("returncode")) ==
                       ("pass", 0, 0) else
                       "graded_fail" if
                       (grader_status, grader_returncode, result.get("returncode")) ==
                       ("fail", 1, 1) else "")
            if not derived:
                raise ValueError(f"run has no recognized complete pass/fail outcome: {run_id}")
            if launch_version == 2:
                outcome = derived
            elif outcome != derived:
                raise ValueError(f"typed outcome and grader outcome disagree: {run_id}")
            if summary.get("compiler_seal_status") != "sealed":
                raise ValueError(f"graded outcome has no sealed compiler: {run_id}")
        expected_search = "pass" if search_required else "not_required"
        expected_summary_search = (
            "fail" if agent_noncompletion and search_required else expected_search)
        if (not search_treatment_failure and
                summary.get("trusted_search_status") != expected_summary_search):
            raise ValueError(f"trusted search summary is invalid for {run_id}")
        if search_required and not (search_treatment_failure or agent_noncompletion):
            checks = search_seal.get("checks")
            search_dir = archive / "search"
            ledger_index = run_dir / "metrics" / "trusted_search_ledger" / "index.json"
            if (search_seal.get("version") != 1 or search_seal.get("status") != "pass" or
                    not isinstance(checks, dict) or set(checks) != _TRUSTED_SEARCH_CHECKS or
                    any(value is not True for value in checks.values()) or
                    not (search_dir / "search_record.json").is_file() or
                    not (search_dir / "selected_policy.json").is_file() or
                    not ledger_index.is_file() or
                    search_seal.get("search_record_sha256") !=
                    _sha256(search_dir / "search_record.json") or
                    search_seal.get("selected_policy_sha256") !=
                    _sha256(search_dir / "selected_policy.json") or
                    search_seal.get("trusted_ledger_sha256") != _sha256(ledger_index) or
                    _as_nonnegative_int(search_seal.get("trusted_evaluation_count"),
                                        where="trusted search evaluation count") < 1 or
                    _as_nonnegative_int(search_seal.get("trusted_evaluation_wall_ns"),
                                        where="trusted search evaluation wall ns") < 1 or
                    _as_nonnegative_int(search_seal.get("trusted_broker_wall_ns"),
                                        where="trusted search broker wall ns") < 1):
                raise ValueError(f"trusted search seal is incomplete or inconsistent for {run_id}")
        elif not search_required and search_seal != {
                "version": 1, "status": "not_required", "arm": result["arm"]}:
            raise ValueError(f"non-search arm has a fabricated search seal: {run_id}")
        if not (search_treatment_failure or agent_noncompletion) and (
                compiler_seal.get("status") != "sealed" or reconciliation.get("ok") is not True or
                not archive.is_dir() or
                _submission_source_digest(archive) != compiler_seal.get("compiler_source_sha256") or
                _submission_package_digest(archive) != compiler_seal.get("compiler_package_sha256")):
            raise ValueError(f"compiler archive/seal/AET reconciliation failed for {run_id}")
        if not (search_treatment_failure or agent_noncompletion) and (
                compiler_seal.get("search_status") != expected_search or
                compiler_seal.get("selected_policy_sha256") !=
                search_seal.get("selected_policy_sha256", compiler_seal.get("policy_sha256")) or
                compiler_seal.get("search_record_sha256") !=
                search_seal.get("search_record_sha256")):
            raise ValueError(f"compiler seal is not cross-bound to trusted search: {run_id}")
        build = grader_result.get("build")
        build_commands = build.get("commands") if isinstance(build, dict) else None
        grader_compiler_seal = grader_result.get("compiler_seal")
        grader_contracts = grader_result.get("contracts")
        expected_contracts = {
            "target_contract": {"path": str(spec._repo_path(spec.target_contract).resolve()),
                                "sha256": _sha256(spec._repo_path(spec.target_contract))},
            "dialect_plan": {"path": str(spec._repo_path(spec.dialect_plan).resolve()),
                             "sha256": _sha256(spec._repo_path(spec.dialect_plan))},
        }
        expected_grader_search = ({
            "status": "pass",
            "checks": {name: True for name in (
                "driver_verified", "policy_byte_match", "independent_convergence_sweep",
                "deterministic_replay", "heldout_never_opened")},
            "seal_sha256": _sha256(search_seal_path),
        } if search_required else {"status": "not_required"})
        archived_manifest = yaml.safe_load((archive / "manifest.yaml").read_text(encoding="utf-8")) \
            if archive.is_dir() else {}
        manifest_build = archived_manifest.get("build") if isinstance(archived_manifest, dict) else None
        expected_build_commands = []
        if isinstance(manifest_build, dict):
            expected_build_commands.append(manifest_build.get("command"))
            if manifest_build.get("then") is not None:
                expected_build_commands.append(manifest_build.get("then"))
        if outcome in {"graded_pass", "graded_fail"} and (
                not isinstance(build_commands, list) or not expected_build_commands or
                len(build_commands) != len(expected_build_commands) or
                any(not isinstance(row, dict) or row.get("returncode") != 0
                    for row in build_commands) or
                [row.get("command") for row in build_commands] != expected_build_commands or
                build.get("policy_sha256") != compiler_seal.get("policy_sha256") or
                grader_compiler_seal != {
                    "status": "pass",
                    "checks": {name: True for name in (
                        "sealed", "policy_sha256", "compiler_source_sha256",
                        "compiler_package_sha256")},
                    "seal_sha256": _sha256(compiler_seal_path),
                } or grader_contracts != expected_contracts or
                grader_result.get("trusted_search") != expected_grader_search):
            raise ValueError(f"grader envelope is not bound to compiler/search/contracts: {run_id}")
        if outcome == "treatment_build_fail" and not search_treatment_failure:
            _validate_submission_build_failure(
                grader_result, expected_commands=expected_build_commands)
        if (outcome == "treatment_agent_fail" and not search_treatment_failure and
                not agent_noncompletion):
            _require_exact_record_fields(
                grader_result,
                {"version", "status", "failure_class", "implemented_levels", "reason",
                 "wall_seconds"}, where="grader invalid-submission treatment outcome")
            if (grader_result["version"] != 1 or
                    grader_result["status"] != "treatment_agent_fail" or
                    grader_result["failure_class"] != "treatment_agent_fail" or
                    grader_result["implemented_levels"] != list(_LEVEL_CHECKS) or
                    not isinstance(grader_result["reason"], str)):
                raise ValueError("grader invalid-submission outcome has an invalid envelope")
            _finite_nonnegative(
                grader_result["wall_seconds"], where="invalid-submission grader wall_seconds")
        expected_run_record = {
            "run_id": run_id, "project": "merlin", "suite": "k1_cpu/cpu-host-compiler",
            "target": "k1_cpu", "method": result["arm"], "seed": result["seed"],
            "experiment": spec.label, "arm": result["arm"], "model": spec.agent["model"],
            "billing_mode": spec.agent["billing"],
            "environment_manifest_sha256": spec.environment["sha256"], "spec": str(source),
            "analysis_plan_sha256": spec.analysis["sha256"],
            "provider_sampling_seeded": False,
        }
        if any(run_record.get(name) != value for name, value in expected_run_record.items()):
            raise ValueError(f"run identity does not match launch record: {run_id}")
        run_result_path = run_dir / "agent" / "run_result.json"
        run_result = _json(run_result_path)
        expected_agent_status = "failed" if agent_noncompletion else "completed"
        resolved_model_ok = (run_result.get("resolved_model") in
                             ({None, spec.agent["model"]} if agent_noncompletion else
                              {spec.agent["model"]}))
        if (not resolved_model_ok or run_result.get("requested_model") != spec.agent["model"] or
                run_result.get("status") != expected_agent_status or
                run_result.get("usage_complete") is not True):
            raise ValueError(f"retained Codex model/usage identity differs from frozen arm: {run_id}")
        attempts = run_result.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            raise ValueError(f"run has no retained Codex attempts: {run_id}")
        if agent_noncompletion:
            backend_failures = {"AuthenticationError", "BillingError", "RateLimitError",
                                "ServerError", "InvalidRequestError"}
            failure_classes = {str(attempt.get("failure_class") or "")
                               for attempt in attempts if isinstance(attempt, dict)}
            if (failure_classes & backend_failures or
                    summary.get("agent_failure_class") != "treatment_agent_fail"):
                raise ValueError(f"backend/auth failure was mislabeled as treatment: {run_id}")
        raw_streams = _exact_attempt_streams(
            run_dir / "agent" / "aet_raw", attempts, label="AET raw")
        timestamped_streams = _exact_attempt_streams(
            run_dir / "agent" / "aet_timestamped", attempts, label="AET timestamped")
        raw_event_count = sum(row["nonblank_lines"] for row in raw_streams)
        timestamped_event_count = sum(row["nonblank_lines"] for row in timestamped_streams)
        raw_report = reconciliation.get("raw_events")
        if (not isinstance(raw_report, dict) or raw_report.get("reconciled") is not True or
                raw_report.get("raw_event_count") != raw_event_count or
                timestamped_event_count != raw_event_count):
            raise ValueError(f"retained raw/timestamped streams do not reconcile for {run_id}")
        tokens = _token_totals(token_path, run_result=run_result, reconciliation=reconciliation)
        for name, value in tokens.items():
            aggregate_tokens[name] += value
        tools = _tool_rows(
            tools_path, token_path=token_path,
            raw_directory=run_dir / "agent" / "aet_raw",
            timestamped_directory=run_dir / "agent" / "aet_timestamped",
            run_result=run_result, model=str(run_result.get("resolved_model") or spec.agent["model"]))
        tool_count = len(tools)
        aggregate_tools += tool_count
        timing = _timing_evidence(
            summary=summary, run_result=run_result, grader_result=grader_result,
            search_seal=search_seal, driver_wall_timing=driver_timing)
        for name, value in timing.items():
            aggregate_time[name] += value
        rows.append({
            "ordinal": int(result["ordinal"]),
            "arm": result["arm"], "repeat": int(result["repeat"]),
            "seed": int(result["seed"]), "run_id": run_id, "run_dir": str(run_dir),
            "outcome": outcome,
            "launch_plan_sha256": hashlib.sha256(json.dumps(
                planned_row, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
            "preflight_sha256": _sha256(preflight_path),
            "workspace_input_lock_sha256": _sha256(input_lock_path),
            "workspace_input_audit_sha256": _sha256(input_audit_path),
            "terminal_outcome_sha256": (_sha256(terminal_path)
                                        if launch_version == 3 else None),
            "compiler_seal": compiler_seal, "search_seal_sha256": _sha256(search_seal_path),
            "summary_sha256": _sha256(run_dir / "metrics" / "summary_metrics.json"),
            "grader_result_sha256": _sha256(grader_result_path),
            "driver_wall_timing_sha256": _sha256(driver_timing_path),
            "reconciliation_sha256": _sha256(reconciliation_path),
            "run_result_sha256": _sha256(run_result_path),
            "codex_attempts": len(attempts),
            "aet_raw_streams": raw_streams,
            "aet_timestamped_streams": timestamped_streams,
            "token_ledger_sha256": _sha256(token_path), "tool_ledger_sha256": _sha256(tools_path),
            "tokens": tokens, "tool_calls": tool_count, "timing_seconds": timing,
        })

    rule = spec.freeze["selection"]
    selected = next(row for row in rows if row["arm"] == rule["primary_arm"] and row[
        "repeat"] == int(rule["primary_repeat_index"]))
    selected_package = Path(selected["run_dir"]) / "artifacts" / "compiler_submission"
    promoted = selected["outcome"] == "graded_pass"
    if promoted:
        selected_manifest = yaml.safe_load(
            (selected_package / "manifest.yaml").read_text(encoding="utf-8"))
        if not isinstance(selected_manifest, dict):
            raise ValueError("selected compiler manifest is not a mapping")
        selected_policy = (selected_package / str(selected_manifest.get("policy", ""))).resolve()
        if (not selected_policy.is_relative_to(selected_package.resolve()) or
                not selected_policy.is_file() or
                _sha256(selected_policy) != selected["compiler_seal"]["policy_sha256"]):
            raise ValueError("selected archived policy bytes differ from compiler seal")
    outcome_counts: dict[str, int] = {}
    for row in rows:
        outcome_counts[row["outcome"]] = outcome_counts.get(row["outcome"], 0) + 1
    promotion = ({"status": "promoted", "predeclared_run_id": selected["run_id"]}
                 if promoted else {
                     "status": "ineligible", "predeclared_run_id": selected["run_id"],
                     "reason": f"predeclared primary outcome is {selected['outcome']}",
                 })
    campaign_record = {
        "version": 1, "campaign_run_id": launch.get("campaign_run_id"),
        "launch_record": str(launch_record), "launch_record_sha256": _sha256(launch_record),
        "expected_run_count": len(expected_plan), "completed_run_count": len(rows), "runs": rows,
        "analysis_plan_sha256": spec.analysis["sha256"],
        "block_boundary_receipts": block_boundaries,
        "outcome_counts": outcome_counts, "retries_after_observed_failure": 0,
        "selection": {**rule,
                      "predeclared_run_id": selected["run_id"],
                      "selected_run_id": selected["run_id"] if promoted else None,
                      "selection_outcome_fields_used": [], "heldout_outcome_used": False},
        "promotion": promotion,
        "telemetry": {"tokens": aggregate_tokens, "tool_calls": aggregate_tools,
                      "time_seconds": aggregate_time},
    }
    campaign_sha = hashlib.sha256(json.dumps(
        campaign_record, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["status"] = "campaign_complete" if promoted else "campaign_complete_unpromoted"
    output_fields = ({
        "selected_policy_sha256": selected["compiler_seal"]["policy_sha256"],
        "runtime_sha256": selected["compiler_seal"]["compiler_package_sha256"],
        "compiler_sha256": selected["compiler_seal"]["compiler_source_sha256"],
        "selected_compiler_package": str(selected_package),
        "selected_run_id": selected["run_id"],
    } if promoted else {
        "selected_policy_sha256": "unresolved", "runtime_sha256": "unresolved",
        "compiler_sha256": "unresolved", "selected_compiler_package": "unresolved",
        "selected_run_id": "unresolved",
    })
    raw["freeze"].update({**output_fields, "campaign_record_sha256": campaign_sha,
                          "campaign_record": campaign_record})
    # A controller may publish a tombstone while the retained evidence or temporary output is being
    # audited.  Recheck after round-trip validation and immediately before the atomic publication.
    return _validate_then_publish(
        output, raw,
        before_publish=lambda: _assert_campaign_not_excluded(runs_root, campaign_run_id))


def complete_campaign(source: Path, launch_record: Path, output: Path) -> HostExperimentSpec:
    """Publish once under an exclusive reservation; never replace another finalizer's result."""
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    reservation = output.with_name(f".{output.name}.completion.lock")
    try:
        descriptor = os.open(reservation, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(
            f"another campaign finalizer owns the output reservation: {reservation}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()}\n")
            stream.flush()
            os.fsync(stream.fileno())
        if output.exists():
            raise FileExistsError(f"refusing to overwrite campaign completion: {output}")
        return _complete_campaign_reserved(source, launch_record, output)
    finally:
        reservation.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--launch-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    completed = complete_campaign(args.spec, args.launch_record, args.output)
    print(yaml.safe_dump({"status": completed.status, "output": str(args.output.resolve()),
                         "selected_run_id": completed.freeze["selected_run_id"]}, sort_keys=False), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
