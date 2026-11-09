#!/usr/bin/env python3
"""Deterministic, holdout-blind staged search for generic compiler policies.

This file is copied verbatim into Arm 3/4 workspaces. It intentionally depends only on Python,
PyYAML, and the staged trusted-evaluator shim. Spike cycle/correctness screening is cheap and
content-stable; the single deterministic top survivor per incumbent receives exactly six balanced K1
measurement pairs on all six frozen confirmation families in train and validation. There is no held-out CLI
option.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _median_paired_speedup(
        baseline_elapsed: list[int], baseline_calls: list[int],
        candidate_elapsed: list[int], candidate_calls: list[int]) -> float:
    return statistics.median([
        (base_elapsed / base_calls) / (candidate_elapsed_value / candidate_calls_value)
        for base_elapsed, base_calls, candidate_elapsed_value, candidate_calls_value in zip(
            baseline_elapsed, baseline_calls, candidate_elapsed, candidate_calls, strict=True)])


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def _coverage_key(row: dict[str, Any]) -> tuple[str, ...]:
    family = str(row["family"])
    if family == "contraction":
        return family, str(row["operation"]), str(row["dtype"]), str(row["layout"])
    if family in {"elementwise_map", "reduction", "movement_layout", "fusion_epilogue"}:
        return family, str(row["operation"]), str(row["dtype"])
    if family == "runtime_parallel":
        return family, str(row["operation"]), str(row["core_count"])
    raise ValueError(f"unsupported capsule family {family!r}")


def select_semantic_sample(rows: list[dict[str, Any]], *, per_family: int | None = None,
                           families: list[str] | None = None
                           ) -> list[dict[str, Any]]:
    """Stable content-minimum semantic buckets, optionally capped equally by family."""
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("split") not in {"train", "validation"}:
            raise ValueError("beam search accepts public train/validation capsules only")
        buckets.setdefault(_coverage_key(row), []).append(row)
    observed_families = {key[0] for key in buckets}
    required = {"contraction", "elementwise_map", "reduction", "movement_layout",
                "fusion_epilogue", "runtime_parallel"}
    if observed_families != required:
        raise ValueError(f"public split family set differs: got {sorted(observed_families)}")
    representatives = [min(buckets[key], key=lambda row: str(row["sha256"]))
                       for key in sorted(buckets)]
    if per_family is None:
        return representatives
    if per_family < 1:
        raise ValueError("per-family sample count must be positive")
    selected_families = required if families is None else set(families)
    if not selected_families or not selected_families <= required:
        raise ValueError(f"selected sample families are invalid: {sorted(selected_families)}")
    selected = []
    for family in sorted(selected_families):
        family_rows = [row for row in representatives if row["family"] == family]
        selected.extend(sorted(family_rows, key=lambda row: str(row["sha256"]))[:per_family])
    return selected


def _canonical_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(actions, key=lambda row: (int(row["stage"]), str(row["group"]), str(row["id"])))


def _candidate(actions: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = _canonical_actions(actions)
    payload = [{key: value for key, value in action.items() if key != "evidence"}
               for action in normalized]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {"version": 1, "candidate_sha256": hashlib.sha256(encoded).hexdigest(),
            "actions": normalized}


def _expand(parent: dict[str, Any], actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    used_groups = {str(action["group"]) for action in parent["actions"]}
    children: dict[str, dict[str, Any]] = {}
    for action in actions:
        if str(action["group"]) in used_groups:
            continue
        child = _candidate([*parent["actions"], action])
        children[child["candidate_sha256"]] = child
    return [children[key] for key in sorted(children)]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8")


def _marginal_action(parent: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    parent_ids = {str(action["id"]) for action in parent["actions"]}
    added = [action for action in candidate["actions"] if str(action["id"]) not in parent_ids]
    if len(added) != 1:
        raise ValueError("each search evaluation must add exactly one action to its parent")
    return added[0]


def _evaluate(parent: dict[str, Any], candidate: dict[str, Any], split: str,
              capsules: list[dict[str, Any]],
              evaluator: list[str], root: Path, repeats: int,
              minimum_pairwise_wins: int = 5) -> dict[str, Any]:
    work = (root / "confirm" / parent["candidate_sha256"][:16] /
            candidate["candidate_sha256"][:16] / split)
    work.mkdir(parents=True, exist_ok=False)
    policy_path, parent_path = work / "policy.json", work / "parent_policy.json"
    capsule_path = work / "capsules.jsonl"
    output_path = work / "observations.jsonl"
    policy_path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    parent_path.write_text(json.dumps(parent, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_jsonl(capsule_path, capsules)
    command = [*evaluator, "--phase", "confirm", "--policy", str(policy_path),
               "--parent-policy", str(parent_path),
               "--capsules", str(capsule_path),
               "--split", split, "--repeats", str(repeats), "--output", str(output_path)]
    started = time.monotonic_ns()
    proc = subprocess.run(command, capture_output=True, text=True, timeout=7200)
    elapsed = time.monotonic_ns() - started
    (work / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (work / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(f"evaluator failed for {candidate['candidate_sha256']}/{split}")
    if not output_path.is_file():
        raise RuntimeError("evaluator produced no observations.jsonl")
    observations = _jsonl(output_path)
    expected = {str(row["id"]): row for row in capsules}
    actual = [str(row.get("capsule_id", "")) for row in observations]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError("evaluator output does not cover the exact selected capsule set once")
    speedups: list[float] = []
    paired_speedups_by_capsule: dict[str, list[float]] = {}
    affected_speedups: list[float] = []
    failures: list[str] = []
    families: set[str] = set()
    marginal = _marginal_action(parent, candidate)
    affected = set(marginal.get("affected_families") or {
        "contraction", "elementwise_map", "reduction", "movement_layout",
        "fusion_epilogue", "runtime_parallel"})
    for row in observations:
        capsule = expected[str(row["capsule_id"])]
        if str(row.get("family")) != str(capsule["family"]):
            failures.append(f"{row['capsule_id']}: family mismatch")
        baseline_elapsed = row.get("baseline_elapsed_ns")
        baseline_calls = row.get("baseline_calls")
        candidate_elapsed = row.get("candidate_elapsed_ns")
        candidate_calls = row.get("candidate_calls")
        sample_arrays = (baseline_elapsed, baseline_calls, candidate_elapsed, candidate_calls)
        if (any(not isinstance(values, list) or len(values) != repeats
                for values in sample_arrays)):
            failures.append(f"{row['capsule_id']}: needs exactly {repeats} paired samples")
            continue
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
               for values in sample_arrays for value in values):
            failures.append(
                f"{row['capsule_id']}: elapsed-time/call samples must be positive integers")
            continue
        if row.get("correctness_ok") is not True:
            failures.append(f"{row['capsule_id']}: correctness failed")
        baseline_digest = row.get("baseline_code_sha256")
        candidate_digest = row.get("candidate_code_sha256")
        family = str(capsule["family"])
        if not _is_sha256(baseline_digest) or not _is_sha256(candidate_digest):
            failures.append(f"{row['capsule_id']}: emitted-code digests are not SHA-256")
        elif baseline_digest == candidate_digest and family in affected:
            failures.append(f"{row['capsule_id']}: affected-family emitted code did not change")
        paired_speedups = [
            (base_elapsed / base_calls) / (candidate_elapsed_value / candidate_calls_value)
            for base_elapsed, base_calls, candidate_elapsed_value, candidate_calls_value in zip(
                baseline_elapsed, baseline_calls, candidate_elapsed, candidate_calls, strict=True)]
        # Pairing and order balance are part of the measurement contract.  Taking the median of
        # within-pair throughput ratios preserves that design; a ratio of two independent medians
        # would discard the common board condition and input seed.
        speedup = statistics.median(paired_speedups)
        paired_speedups_by_capsule[str(row["capsule_id"])] = paired_speedups
        if family in affected and sum(value > 1.0 for value in paired_speedups) < minimum_pairwise_wins:
            failures.append(
                f"{row['capsule_id']}: fewer than {minimum_pairwise_wins}/{repeats} paired wins")
        speedups.append(speedup)
        if family in affected:
            affected_speedups.append(speedup)
        families.add(family)
    return {
        "parent_candidate_sha256": parent["candidate_sha256"],
        "candidate_sha256": candidate["candidate_sha256"], "split": split,
        "command": command, "wall_ns": elapsed, "observations_sha256": _sha256(output_path),
        "capsules": len(capsules), "families": sorted(families), "failures": failures,
        "median_speedup": statistics.median(speedups) if speedups else 0.0,
        "minimum_speedup": min(speedups) if speedups else 0.0,
        "affected_families": sorted(affected),
        "affected_median_speedup": (statistics.median(affected_speedups)
                                    if affected_speedups else 0.0),
        "per_capsule_speedup": speedups,
        "per_capsule_paired_speedups": paired_speedups_by_capsule,
    }


def _screen(parent: dict[str, Any], candidate: dict[str, Any], capsules: list[dict[str, Any]],
            evaluator: list[str],
            root: Path) -> dict[str, Any]:
    work = (root / "screen" / parent["candidate_sha256"][:16] /
            candidate["candidate_sha256"][:16] / "train")
    work.mkdir(parents=True, exist_ok=False)
    policy_path, parent_path = work / "policy.json", work / "parent_policy.json"
    capsule_path = work / "capsules.jsonl"
    output_path = work / "observations.jsonl"
    policy_path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    parent_path.write_text(json.dumps(parent, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_jsonl(capsule_path, capsules)
    command = [*evaluator, "--phase", "screen", "--policy", str(policy_path),
               "--parent-policy", str(parent_path),
               "--capsules", str(capsule_path), "--split", "train", "--repeats", "1",
               "--output", str(output_path)]
    started = time.monotonic_ns()
    proc = subprocess.run(command, capture_output=True, text=True, timeout=7200)
    elapsed = time.monotonic_ns() - started
    (work / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (work / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode or not output_path.is_file():
        raise RuntimeError(f"screen failed for {candidate['candidate_sha256']}")
    observations = _jsonl(output_path)
    expected = {str(row["id"]): row for row in capsules}
    actual = [str(row.get("capsule_id", "")) for row in observations]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError("screen output does not cover the exact selected capsule set once")
    speedups, affected_speedups, failures, families = [], [], [], set()
    marginal = _marginal_action(parent, candidate)
    affected = set(marginal.get("affected_families") or {
        "contraction", "elementwise_map", "reduction", "movement_layout",
        "fusion_epilogue", "runtime_parallel"})
    for row in observations:
        capsule = expected[str(row["capsule_id"])]
        baseline, measured = row.get("baseline_cycles"), row.get("candidate_cycles")
        if str(row.get("family")) != str(capsule["family"]):
            failures.append(f"{row['capsule_id']}: family mismatch")
        if not isinstance(baseline, int) or baseline <= 0 or not isinstance(measured, int) or measured <= 0:
            failures.append(f"{row['capsule_id']}: Spike cycle measurement is invalid")
            continue
        if row.get("correctness_ok") is not True:
            failures.append(f"{row['capsule_id']}: correctness failed")
        baseline_digest, candidate_digest = (row.get("baseline_code_sha256"),
                                              row.get("candidate_code_sha256"))
        family = str(capsule["family"])
        if not _is_sha256(baseline_digest) or not _is_sha256(candidate_digest):
            failures.append(f"{row['capsule_id']}: emitted-code digests are not SHA-256")
        elif baseline_digest == candidate_digest and family in affected:
            failures.append(f"{row['capsule_id']}: affected-family emitted code did not change")
        speedup = baseline / measured
        speedups.append(speedup)
        if family in affected:
            affected_speedups.append(speedup)
        families.add(family)
    return {
        "parent_candidate_sha256": parent["candidate_sha256"],
        "candidate_sha256": candidate["candidate_sha256"], "split": "train",
        "phase": "screen", "command": command, "wall_ns": elapsed,
        "observations_sha256": _sha256(output_path), "capsules": len(capsules),
        "families": sorted(families), "failures": failures,
        "median_speedup": statistics.median(speedups) if speedups else 0.0,
        "minimum_speedup": min(speedups) if speedups else 0.0,
        "affected_families": sorted(affected),
        "affected_median_speedup": (statistics.median(affected_speedups)
                                    if affected_speedups else 0.0),
        "per_capsule_speedup": speedups,
    }


def _eligible_train(result: dict[str, Any], minimum_families: int) -> bool:
    return not result["failures"] and len(result["families"]) >= minimum_families


def _promotion_eligible(train: dict[str, Any], validation: dict[str, Any],
                        minimum_families: int, margin: float) -> bool:
    """Apply the frozen, ratio-symmetric confirmation rule.

    ``margin`` is the calibrated upper multiplicative A/A tolerance. A reciprocal lower bound is
    the corresponding symmetric limit in log-throughput space; ``1 - margin`` would silently allow
    a larger regression than the improvement required for promotion.
    """
    lower_bound = 1.0 / (1.0 + margin)
    return (
        _eligible_train(validation, minimum_families)
        and float(train["affected_median_speedup"]) > 1.0 + margin
        and float(validation["affected_median_speedup"]) > 1.0 + margin
        and float(validation["minimum_speedup"]) >= lower_bound
    )


def _rank_key(candidate: dict[str, Any], result: dict[str, Any]) -> tuple[Any, ...]:
    # Negative values produce descending performance under ordinary ascending sort. The digest is
    # the final tie-break, making the beam invariant to evaluator completion or filesystem order.
    return (-float(result["affected_median_speedup"]), -float(result["minimum_speedup"]),
            candidate["candidate_sha256"])


def _load_space(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("version") != 1 or raw.get("status") != "frozen_definition":
        raise ValueError("optimization space must be a frozen version-1 mapping")
    actions = raw.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("optimization space has no actions")
    ids = [str(action.get("id", "")) for action in actions]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("optimization action ids must be present and unique")
    allowed_classes = {"pass", "heuristic", "flag", "knob"}
    if any(action.get("action_class") not in allowed_classes or not action.get("group")
           or not isinstance(action.get("stage"), int) for action in actions):
        raise ValueError("every action needs a group, integer stage, and legal action_class")
    all_families = ["contraction", "elementwise_map", "reduction", "movement_layout",
                    "fusion_epilogue", "runtime_parallel"]
    confirmation_families = raw.get("confirmation_families")
    if confirmation_families is None:  # backward-compatible with pre-staged fixture spaces
        confirmation_families = all_families
        raw["confirmation_families"] = confirmation_families
    if (not isinstance(confirmation_families, list) or not confirmation_families
            or len(set(confirmation_families)) != len(confirmation_families)
            or not set(confirmation_families) <= set(all_families)
            or len(confirmation_families) < int(raw.get("minimum_families", 0))):
        raise ValueError("optimization space confirmation_families are invalid")
    for action in actions:
        affected = action.get("affected_families")
        if affected is None:  # compatibility for small deterministic test spaces
            affected = list(confirmation_families)
            action["affected_families"] = affected
        if (not isinstance(affected, list) or not affected or
                len(affected) != len(set(affected)) or
                not set(affected) <= set(all_families) or
                not set(affected) <= set(confirmation_families)):
            raise ValueError(f"action {action.get('id')} has invalid affected_families")
    return raw


def run_search(*, space_path: Path, train_path: Path, validation_path: Path,
               evaluator: list[str], output: Path) -> dict[str, Any]:
    started = time.monotonic_ns()
    space = _load_space(space_path)
    train_all, validation_all = _jsonl(train_path), _jsonl(validation_path)
    if any(row.get("split") != "train" for row in train_all):
        raise ValueError("--train contains a non-train capsule")
    if any(row.get("split") != "validation" for row in validation_all):
        raise ValueError("--validation contains a non-validation capsule")
    screen_train = select_semantic_sample(
        train_all, per_family=int(space["screen_samples_per_family"]))
    confirm_train = select_semantic_sample(
        train_all, per_family=int(space["confirmation_samples_per_family"]),
        families=list(space["confirmation_families"]))
    confirm_validation = select_semantic_sample(
        validation_all, per_family=int(space["confirmation_samples_per_family"]),
        families=list(space["confirmation_families"]))
    output.mkdir(parents=True, exist_ok=False)
    evaluations = output / "evaluations"; evaluations.mkdir()
    confirmation_width = int(space["confirmation_width"])
    repeats = int(space["measurement_repeats"])
    minimum_pairwise_wins = int(
        (space.get("selection") or {}).get("minimum_pairwise_wins", max(1, repeats - 1)))
    margin, minimum_families = float(space["noise_margin"]), int(space["minimum_families"])
    required_empty = int(space["required_empty_sweeps"])
    incumbent = _candidate([])
    accepted: list[dict[str, Any]] = []
    sweeps: list[dict[str, Any]] = []
    empty_sweeps = 0
    cache: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    def screen(parent: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
        key = parent["candidate_sha256"], candidate["candidate_sha256"], "train", "screen"
        if key not in cache:
            cache[key] = _screen(parent, candidate, screen_train, evaluator, evaluations)
        return cache[key]

    def confirm(parent: dict[str, Any], candidate: dict[str, Any], split: str) -> dict[str, Any]:
        key = parent["candidate_sha256"], candidate["candidate_sha256"], split, "confirm"
        if key not in cache:
            capsules = confirm_train if split == "train" else confirm_validation
            cache[key] = _evaluate(
                parent, candidate, split, capsules, evaluator, evaluations, repeats,
                minimum_pairwise_wins)
        return cache[key]

    for sweep_index in range(int(space["max_sweeps"])):
        expanded = _expand(incumbent, space["actions"])
        screened = []
        for candidate in expanded:
            result = screen(incumbent, candidate)
            if _eligible_train(result, minimum_families):
                screened.append((candidate, result))
        screened.sort(key=lambda pair: _rank_key(pair[0], pair[1]))
        confirmation_candidates = [candidate for candidate, _ in
                                   screened[:confirmation_width]]
        confirmed = []
        for candidate in confirmation_candidates:
            train_result = confirm(incumbent, candidate, "train")
            if _eligible_train(train_result, minimum_families):
                confirmed.append((candidate, train_result))
        confirmed.sort(key=lambda pair: _rank_key(pair[0], pair[1]))
        promoted = []
        for candidate, train_result in confirmed:
            validation_result = confirm(incumbent, candidate, "validation")
            accepted_now = _promotion_eligible(
                train_result, validation_result, minimum_families, margin)
            if accepted_now:
                promoted.append((candidate, train_result, validation_result))
        promoted.sort(key=lambda row: (
            -float(row[2]["affected_median_speedup"]),
            -float(row[1]["affected_median_speedup"]),
            row[0]["candidate_sha256"]))
        winner = promoted[0] if promoted else None
        sweeps.append({"sweep": sweep_index, "incumbent": incumbent["candidate_sha256"],
                       "screened": [row[0]["candidate_sha256"] for row in screened],
                       "confirmed": [row[0]["candidate_sha256"] for row in confirmed],
                       "promoted": [row[0]["candidate_sha256"] for row in promoted],
                       "winner": winner[0]["candidate_sha256"] if winner else None})
        if winner:
            incumbent = winner[0]
            accepted.append({"candidate": incumbent,
                             "train": winner[1], "validation": winner[2]})
            empty_sweeps = 0
        else:
            empty_sweeps += 1
            if empty_sweeps >= required_empty:
                break
    converged = empty_sweeps >= required_empty
    # Keep the policy bytes identical to the candidate evaluated by the trusted broker. Convergence
    # belongs in search_record.json; adding a final-only status field would let a compiler branch on
    # bytes that were never measured.
    final_policy = dict(incumbent)
    (output / "selected_policy.json").write_text(
        json.dumps(final_policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record = {
        "version": 1, "status": "converged" if converged else "not_converged",
        "space_sha256": _sha256(space_path), "train_sha256": _sha256(train_path),
        "validation_sha256": _sha256(validation_path), "heldout_visible": False,
        "sample_counts": {
            "screen_train": len(screen_train),
            "confirmation_train": len(confirm_train),
            "confirmation_validation": len(confirm_validation),
        },
        "selection_policy": "spike_screen_then_k1_confirmation",
        "acceptance_thresholds": {
            "calibrated_upper_margin": margin,
            "affected_train_median_strictly_above": 1.0 + margin,
            "affected_validation_median_strictly_above": 1.0 + margin,
            "validation_minimum_at_least": 1.0 / (1.0 + margin),
            "minimum_pairwise_wins_per_affected_capsule": minimum_pairwise_wins,
        },
        "confirmation_families": list(space["confirmation_families"]),
        "confirmation_width": confirmation_width,
        "measurement_repeats": repeats, "accepted": accepted, "sweeps": sweeps,
        "empty_sweeps": empty_sweeps, "required_empty_sweeps": required_empty,
        "selected_policy_sha256": _sha256(output / "selected_policy.json"),
        "wall_ns": time.monotonic_ns() - started,
    }
    (output / "search_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("evaluator", nargs=argparse.REMAINDER,
                        help="evaluator argv after -- (shell strings are not accepted)")
    args = parser.parse_args(argv)
    evaluator = list(args.evaluator)
    if evaluator and evaluator[0] == "--":
        evaluator.pop(0)
    if not evaluator:
        parser.error("an evaluator argument array is required after --")
    result = run_search(space_path=args.space.resolve(), train_path=args.train.resolve(),
                        validation_path=args.validation.resolve(), evaluator=evaluator,
                        output=args.output.resolve())
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "converged" else 2


if __name__ == "__main__":
    raise SystemExit(main())
