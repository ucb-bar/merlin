"""Render claim-safe Arm1--4 figures from a completed CPU-host campaign.

The campaign uses subscription billing, so this module does not invent currency.  It reports the
predeclared resource endpoints that are actually retained at full fidelity: provider tokens,
driver-observed wall time, tool calls, compiler/grader outcomes, and generic-development search
evidence.  Every summary uses all four scheduled Williams blocks and every whisker is the observed
minimum--maximum range.

The outcome figure deliberately reads only paths and digests sealed into the completed campaign
record.  It never reads paper holdouts, and it never turns an absent search result into a zero or a
baseline speedup.  Search ratios are the final accepted action's controller-measured *marginal*
ratio against its parent policy; they are not multiplied into an invented whole-policy speedup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from merlin.benchharness.host_agent import _submission_package_digest
from merlin.common.paths import artifacts_dir
from merlin.compare.host_experiment import HostExperimentSpec


_TOKEN_FIELDS = {
    "input_tokens", "cached_input_tokens", "cache_write_input_tokens", "output_tokens",
    "reasoning_output_tokens", "uncached_input_tokens",
}
_TIMING_FIELDS = {
    "active_wall_seconds", "grader_wall_seconds", "trusted_search_wall_seconds", "wall_seconds",
}
_OUTCOMES = {
    "graded_pass", "graded_fail", "treatment_agent_fail", "treatment_build_fail",
    "treatment_search_fail",
}
_LEVELS = ("L0", "L1", "L2", "L3")
_SEARCH_CAPABILITY = "deterministic_candidate_search"
_ARM_LABELS = {
    "arm1_raw_cpp": "Arm 1\nRaw C++",
    "arm2_cpp_scaffold": "Arm 2\nC++ scaffold",
    "arm3_generated_cpu_dialect": "Arm 3\nGenerated CPU dialect",
    "arm4_agentic_pass_authoring": "Arm 4\nAgentic passes",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _nonnegative(value: object, *, label: str, integer: bool = False) -> float | int:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or value < 0):
        raise ValueError(f"{label} must be a non-negative {'integer' if integer else 'number'}")
    if integer and not isinstance(value, int):
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _load_campaign(path: Path) -> tuple[HostExperimentSpec, Mapping[str, Any]]:
    spec = HostExperimentSpec.from_yaml(path)
    if spec.status not in {"campaign_complete", "campaign_complete_unpromoted"}:
        raise ValueError("Arm figures require a completed sixteen-cell campaign")
    preflight = spec.preflight(check_environment=False, probe_board=False)
    if not preflight.ready:
        raise ValueError(f"completed CPU-host campaign is not locally verifiable: {preflight.to_dict()}")
    record = spec.freeze.get("campaign_record")
    if not isinstance(record, Mapping):
        raise ValueError("completed campaign has no embedded campaign record")
    if (_canonical_sha256(record) != spec.freeze.get("campaign_record_sha256")
            or record.get("analysis_plan_sha256") != spec.analysis.get("sha256")):
        raise ValueError("completed campaign record differs from its frozen identity")
    return spec, record


def _campaign_rows(spec: HostExperimentSpec,
                   record: Mapping[str, Any]) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    arms = [arm.id for arm in spec.arms]
    raw_rows = record.get("runs")
    if (not isinstance(raw_rows, list) or len(raw_rows) != 16
            or record.get("completed_run_count") != 16 or record.get("expected_run_count") != 16):
        raise ValueError("Arm resource reporting requires all sixteen predeclared cells")
    grouped = {arm: [] for arm in arms}
    seen: set[tuple[str, int]] = set()
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"campaign run {index} is not a mapping")
        arm, repeat = str(raw.get("arm", "")), raw.get("repeat")
        if arm not in grouped or isinstance(repeat, bool) or not isinstance(repeat, int):
            raise ValueError(f"campaign run {index} has an invalid arm/repeat")
        if (arm, repeat) in seen or repeat not in range(4):
            raise ValueError("campaign resource rows do not cover each arm/block exactly once")
        seen.add((arm, repeat))
        tokens, timing = raw.get("tokens"), raw.get("timing_seconds")
        if not isinstance(tokens, Mapping) or set(tokens) != _TOKEN_FIELDS:
            raise ValueError(f"campaign run {index} has incomplete full-fidelity token accounting")
        if not isinstance(timing, Mapping) or set(timing) != _TIMING_FIELDS:
            raise ValueError(f"campaign run {index} has incomplete timing accounting")
        checked_tokens = {
            name: int(_nonnegative(value, label=f"run {index} tokens.{name}", integer=True))
            for name, value in tokens.items()
        }
        if (checked_tokens["cached_input_tokens"] + checked_tokens["cache_write_input_tokens"]
                + checked_tokens["uncached_input_tokens"] != checked_tokens["input_tokens"]):
            raise ValueError("campaign input-token subsets do not sum to provider input tokens")
        if checked_tokens["reasoning_output_tokens"] > checked_tokens["output_tokens"]:
            raise ValueError("campaign reasoning tokens exceed provider output tokens")
        checked_timing = {
            name: float(_nonnegative(value, label=f"run {index} timing_seconds.{name}"))
            for name, value in timing.items()
        }
        tool_calls = int(_nonnegative(raw.get("tool_calls"), label=f"run {index} tool_calls",
                                      integer=True))
        outcome = str(raw.get("outcome", ""))
        if outcome not in _OUTCOMES:
            raise ValueError(f"campaign run {index} has an unknown terminal outcome")
        run_id, run_dir = raw.get("run_id"), raw.get("run_dir")
        if not isinstance(run_id, str) or not run_id or not isinstance(run_dir, str) or not run_dir:
            raise ValueError(f"campaign run {index} has no retained run identity")
        if not _is_sha256(raw.get("grader_result_sha256")):
            raise ValueError(f"campaign run {index} has no frozen grader-result digest")
        if not isinstance(raw.get("compiler_seal"), Mapping):
            raise ValueError(f"campaign run {index} has no frozen compiler seal")
        if not _is_sha256(raw.get("search_seal_sha256")):
            raise ValueError(f"campaign run {index} has no frozen search-seal digest")
        grouped[arm].append({
            "repeat": repeat, "tokens": checked_tokens, "timing_seconds": checked_timing,
            "tool_calls": tool_calls, "outcome": outcome, "run_id": run_id,
            "run_dir": run_dir, "grader_result_sha256": raw["grader_result_sha256"],
            "compiler_seal": dict(raw["compiler_seal"]),
            "search_seal_sha256": raw["search_seal_sha256"],
        })
    expected = {(arm, repeat) for arm in arms for repeat in range(4)}
    if seen != expected:
        raise ValueError("campaign resource rows omit a predeclared arm/block cell")
    for arm in arms:
        grouped[arm].sort(key=lambda row: row["repeat"])
    return arms, grouped


def _summary(values: list[float | int]) -> dict[str, float | int]:
    if len(values) != 4:
        raise ValueError("Arm summaries require exactly four scheduled cells")
    return {"median": statistics.median(values), "minimum": min(values), "maximum": max(values),
            "every_cell": values}


def _summaries(arms: list[str], grouped: Mapping[str, list[dict[str, Any]]]
               ) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for arm in arms:
        rows = grouped[arm]
        summaries[arm] = {
            "provider_tokens": _summary([
                row["tokens"]["input_tokens"] + row["tokens"]["output_tokens"] for row in rows]),
            "reasoning_tokens": _summary([
                row["tokens"]["reasoning_output_tokens"] for row in rows]),
            "cell_wall_seconds": _summary([
                row["timing_seconds"]["wall_seconds"] for row in rows]),
            "agent_active_seconds": _summary([
                row["timing_seconds"]["active_wall_seconds"] for row in rows]),
            "grader_seconds": _summary([
                row["timing_seconds"]["grader_wall_seconds"] for row in rows]),
            "tool_calls": _summary([row["tool_calls"] for row in rows]),
            "terminal_outcomes": [row["outcome"] for row in rows],
        }
    return summaries


def _optional_summary(values: list[float | int | None]) -> dict[str, Any]:
    if len(values) != 4:
        raise ValueError("optional Arm summaries require exactly four scheduled cells")
    available = [value for value in values if value is not None]
    if not available:
        return {"median": None, "minimum": None, "maximum": None,
                "available_cells": 0, "every_cell": values}
    return {"median": statistics.median(available), "minimum": min(available),
            "maximum": max(available), "available_cells": len(available),
            "every_cell": values}


def _bound_json(run_dir: Path, relative: str, expected_sha256: object, *, label: str
                ) -> tuple[dict[str, Any], Path]:
    if not _is_sha256(expected_sha256):
        raise ValueError(f"{label} has no frozen SHA-256")
    path = run_dir / relative
    if not path.is_file() or path.is_symlink() or _sha256(path) != expected_sha256:
        raise ValueError(f"{label} differs from the completed campaign record")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value, path


def _validate_search_candidate(value: object, space: Mapping[str, Any], *, label: str
                               ) -> dict[str, Any]:
    if (not isinstance(value, Mapping) or set(value) != {
            "version", "candidate_sha256", "actions"} or value.get("version") != 1
            or not isinstance(value.get("actions"), list)):
        raise ValueError(f"{label} is not a closed version-1 search candidate")
    actions = list(value["actions"])
    allowed_rows = space.get("actions")
    if not isinstance(allowed_rows, list) or not allowed_rows:
        raise ValueError("frozen optimization space has no action catalogue")
    allowed = {str(action.get("id", "")): action for action in allowed_rows
               if isinstance(action, Mapping)}
    if len(allowed) != len(allowed_rows) or "" in allowed:
        raise ValueError("frozen optimization-space action ids are invalid")
    ids: list[str] = []
    for action in actions:
        if not isinstance(action, Mapping):
            raise ValueError(f"{label} contains a non-mapping action")
        action_id = str(action.get("id", ""))
        if action_id not in allowed or dict(action) != dict(allowed[action_id]):
            raise ValueError(f"{label} contains an action outside the frozen optimization space")
        ids.append(action_id)
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} repeats a selected action")
    canonical = sorted(actions, key=lambda action: (
        int(action["stage"]), str(action["group"]), str(action["id"])))
    if actions != canonical:
        raise ValueError(f"{label} actions are not in frozen canonical order")
    payload = [{key: item for key, item in action.items() if key != "evidence"}
               for action in canonical]
    digest = _canonical_sha256(payload)
    if value.get("candidate_sha256") != digest:
        raise ValueError(f"{label} candidate digest differs from its action list")
    return {"version": 1, "candidate_sha256": digest,
            "actions": [dict(action) for action in actions]}


def _positive_ratio(value: object, *, label: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or value <= 0):
        raise ValueError(f"{label} must be a finite positive ratio")
    return float(value)


def _search_metrics(run_dir: Path, row: Mapping[str, Any], package: Path,
                    space: Mapping[str, Any]) -> dict[str, Any]:
    seal, seal_path = _bound_json(
        run_dir, "contracts/trusted_search_seal.json", row["search_seal_sha256"],
        label=f"{row['run_id']} trusted search seal")
    status = seal.get("status")
    if status != "pass":
        if status != "fail":
            raise ValueError(f"{row['run_id']} search arm has no pass/fail trusted seal")
        return {"status": "unavailable", "reason": "trusted_search_failed",
                "action_count": None, "train_speedup": None,
                "validation_speedup": None, "accepted_marginals": [],
                "search_seal": {"path": str(seal_path), "sha256": row["search_seal_sha256"]}}
    record_sha, policy_sha = seal.get("search_record_sha256"), seal.get(
        "selected_policy_sha256")
    if not _is_sha256(record_sha) or not _is_sha256(policy_sha):
        raise ValueError(f"{row['run_id']} passing search seal lacks result digests")
    record, record_path = _bound_json(
        package, "search/search_record.json", record_sha,
        label=f"{row['run_id']} search record")
    policy, policy_path = _bound_json(
        package, "search/selected_policy.json", policy_sha,
        label=f"{row['run_id']} selected search policy")
    selected = _validate_search_candidate(
        policy, space, label=f"{row['run_id']} selected search policy")
    if (record.get("version") != 1 or record.get("status") != "converged"
            or record.get("heldout_visible") is not False
            or record.get("selected_policy_sha256") != policy_sha):
        raise ValueError(f"{row['run_id']} search record is not converged and holdout-blind")
    accepted = record.get("accepted")
    if not isinstance(accepted, list) or len(accepted) != len(selected["actions"]):
        raise ValueError(f"{row['run_id']} accepted search path differs from selected actions")
    parent = _validate_search_candidate(
        {"version": 1, "candidate_sha256": _canonical_sha256([]), "actions": []},
        space, label=f"{row['run_id']} empty parent policy")
    marginals: list[dict[str, Any]] = []
    for index, entry in enumerate(accepted):
        if not isinstance(entry, Mapping):
            raise ValueError(f"{row['run_id']} accepted search entry {index} is malformed")
        candidate = _validate_search_candidate(
            entry.get("candidate"), space,
            label=f"{row['run_id']} accepted candidate {index}")
        parent_ids = {str(action["id"]) for action in parent["actions"]}
        candidate_ids = {str(action["id"]) for action in candidate["actions"]}
        added_ids = candidate_ids - parent_ids
        if not parent_ids < candidate_ids or len(added_ids) != 1:
            raise ValueError(f"{row['run_id']} accepted search path is not width-one nested")
        split_values: dict[str, float] = {}
        for split in ("train", "validation"):
            result = entry.get(split)
            if (not isinstance(result, Mapping) or result.get("split") != split
                    or result.get("parent_candidate_sha256") != parent["candidate_sha256"]
                    or result.get("candidate_sha256") != candidate["candidate_sha256"]
                    or result.get("failures") != []):
                raise ValueError(
                    f"{row['run_id']} accepted {split} result is not bound to its policy step")
            split_values[split] = _positive_ratio(
                result.get("affected_median_speedup"),
                label=f"{row['run_id']} accepted {split} marginal speedup")
        marginals.append({"action_id": next(iter(added_ids)),
                          "candidate_sha256": candidate["candidate_sha256"],
                          "train_speedup": split_values["train"],
                          "validation_speedup": split_values["validation"]})
        parent = candidate
    if parent != selected:
        raise ValueError(f"{row['run_id']} selected policy differs from accepted search path")
    final = marginals[-1] if marginals else None
    return {
        "status": "pass" if final is not None else "no_accepted_action",
        "reason": None if final is not None else "converged_without_an_accepted_action",
        "action_count": len(selected["actions"]),
        "train_speedup": None if final is None else final["train_speedup"],
        "validation_speedup": None if final is None else final["validation_speedup"],
        "accepted_marginals": marginals,
        "search_seal": {"path": str(seal_path), "sha256": row["search_seal_sha256"]},
        "search_record": {"path": str(record_path), "sha256": record_sha},
        "selected_policy": {"path": str(policy_path), "sha256": policy_sha},
    }


def _campaign_outcomes(spec: HostExperimentSpec, arms: list[str],
                       grouped: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    space = spec.search_space_config()
    if not isinstance(space, Mapping) or not space:
        raise ValueError("completed campaign has no readable frozen optimization space")
    space_path = spec._repo_path(spec.search.get("space", ""))
    if (not space_path.is_file() or _sha256(space_path) != spec.search.get("space_sha256")):
        raise ValueError("frozen optimization space differs from completed campaign")
    search_arms = {arm.id for arm in spec.arms if _SEARCH_CAPABILITY in arm.capabilities}
    if not search_arms or not search_arms <= set(arms):
        raise ValueError("completed campaign has no valid deterministic-search arm set")
    cells: list[dict[str, Any]] = []
    for arm in arms:
        for row in grouped[arm]:
            run_dir = Path(row["run_dir"]).resolve()
            grader, grader_path = _bound_json(
                run_dir, "metrics/grader_result.json", row["grader_result_sha256"],
                label=f"{row['run_id']} grader result")
            if row["outcome"] in {"graded_pass", "graded_fail"}:
                levels = grader.get("levels")
                if (not isinstance(levels, Mapping) or list(levels) != list(_LEVELS)
                        or any(not isinstance(levels[level], Mapping)
                               or levels[level].get("status") not in {"pass", "fail"}
                               for level in _LEVELS)):
                    raise ValueError(f"{row['run_id']} has incomplete L0--L3 grader outcomes")
                level_status = {level: str(levels[level]["status"]) for level in _LEVELS}
                expected_top = "pass" if row["outcome"] == "graded_pass" else "fail"
                if (grader.get("status") != expected_top
                        or (expected_top == "pass" and set(level_status.values()) != {"pass"})
                        or (expected_top == "fail" and "fail" not in level_status.values())):
                    raise ValueError(f"{row['run_id']} terminal outcome disagrees with L0--L3")
            else:
                if isinstance(grader.get("levels"), Mapping) and grader["levels"]:
                    raise ValueError(f"{row['run_id']} treatment failure fabricates reached levels")
                level_status = {level: "not_reached" for level in _LEVELS}

            compiler_seal = row["compiler_seal"]
            package = run_dir / "artifacts" / "compiler_submission"
            package_size: int | None = None
            package_ref: dict[str, Any] | None = None
            if compiler_seal.get("status") == "sealed":
                package_digest = compiler_seal.get("compiler_package_sha256")
                if (not _is_sha256(package_digest) or not package.is_dir()
                        or package.is_symlink()
                        or _submission_package_digest(package) != package_digest):
                    raise ValueError(f"{row['run_id']} compiler package differs from its seal")
                entries = sorted(package.rglob("*"))
                if any(path.is_symlink() for path in entries):
                    raise ValueError(f"{row['run_id']} compiler package contains a symlink")
                regular_files = [path for path in entries if path.is_file()]
                package_size = sum(path.stat().st_size for path in regular_files)
                manifest = yaml.safe_load((package / "manifest.yaml").read_text(encoding="utf-8"))
                if not isinstance(manifest, Mapping):
                    raise ValueError(f"{row['run_id']} compiler manifest is not a mapping")
                policy_relative = Path(str(manifest.get("policy", "")))
                policy_path = (package / policy_relative).resolve()
                if (policy_relative.is_absolute() or ".." in policy_relative.parts
                        or not policy_path.is_relative_to(package.resolve())
                        or not policy_path.is_file() or policy_path.is_symlink()
                        or _sha256(policy_path) != compiler_seal.get("policy_sha256")):
                    raise ValueError(f"{row['run_id']} compiler policy differs from its seal")
                package_ref = {"path": str(package), "sha256": package_digest,
                               "bytes": package_size, "files": len(regular_files),
                               "policy_sha256": compiler_seal["policy_sha256"]}
            elif package.exists():
                raise ValueError(f"{row['run_id']} unsealed compiler has a post-completion package")

            if arm in search_arms:
                search = _search_metrics(run_dir, row, package, space)
                if search["status"] in {"pass", "no_accepted_action"} and package_ref is None:
                    raise ValueError(f"{row['run_id']} passing search has no sealed compiler package")
                if search["status"] in {"pass", "no_accepted_action"} and (
                        compiler_seal.get("selected_policy_sha256") !=
                        search["selected_policy"]["sha256"]
                        or compiler_seal.get("policy_sha256") !=
                        search["selected_policy"]["sha256"]):
                    raise ValueError(
                        f"{row['run_id']} compiler policy differs from trusted selected policy")
            else:
                seal, seal_path = _bound_json(
                    run_dir, "contracts/trusted_search_seal.json", row["search_seal_sha256"],
                    label=f"{row['run_id']} non-search seal")
                if seal != {"version": 1, "status": "not_required", "arm": arm}:
                    raise ValueError(f"{row['run_id']} non-search arm has search evidence")
                search = {"status": "not_applicable", "reason": "arm_has_no_deterministic_search",
                          "action_count": None, "train_speedup": None,
                          "validation_speedup": None, "accepted_marginals": [],
                          "search_seal": {"path": str(seal_path),
                                          "sha256": row["search_seal_sha256"]}}
            cells.append({
                "arm": arm, "repeat": row["repeat"], "run_id": row["run_id"],
                "outcome": row["outcome"], "levels": level_status,
                "grader_result": {"path": str(grader_path),
                                  "sha256": row["grader_result_sha256"]},
                "compiler_package": package_ref, "search": search,
            })

    by_arm = {arm: sorted((cell for cell in cells if cell["arm"] == arm),
                          key=lambda cell: cell["repeat"]) for arm in arms}
    level_counts = {arm: {level: {
        state: sum(cell["levels"][level] == state for cell in by_arm[arm])
        for state in ("pass", "fail", "not_reached")}
        for level in _LEVELS} for arm in arms}
    outcome_counts = {arm: {outcome: sum(cell["outcome"] == outcome for cell in by_arm[arm])
                            for outcome in sorted(_OUTCOMES)} for arm in arms}
    package_sizes = {arm: _optional_summary([
        None if cell["compiler_package"] is None else cell["compiler_package"]["bytes"]
        for cell in by_arm[arm]]) for arm in arms}
    action_counts = {arm: _optional_summary([
        cell["search"]["action_count"] for cell in by_arm[arm]]) for arm in arms}
    speedups = {arm: {
        split: _optional_summary([cell["search"][f"{split}_speedup"] for cell in by_arm[arm]])
        for split in ("train", "validation")} for arm in arms}
    return {
        "scope": "generic_development_only_no_paper_holdouts",
        "speedup_estimand": "final_accepted_action_marginal_ratio_vs_parent_policy",
        "levels": level_counts, "terminal_outcomes": outcome_counts,
        "compiler_package_size_bytes": package_sizes,
        "selected_policy_action_count": action_counts,
        "generic_train_validation_speedup": speedups, "cells": cells,
    }


def _range_bar(ax: Any, x: int, summary: Mapping[str, Any], *, color: str, style: Any,
               label_format: str) -> None:
    median = float(summary["median"])
    minimum, maximum = float(summary["minimum"]), float(summary["maximum"])
    style.vbars(ax, [x], [median], color, width=0.58)
    ax.errorbar([x], [median], yerr=[[median - minimum], [maximum - median]], fmt="none",
                ecolor=style.INK, elinewidth=1.4, capsize=3.5, capthick=1.4, zorder=6)
    ax.text(x, median, label_format.format(median), ha="center", va="bottom",
            fontsize=8.0, fontweight="bold", color=style.GOLD)


def _render_outcome_figure(output: Path, arms: list[str], outcomes: Mapping[str, Any], *,
                           plt: Any, np: Any, style: Any) -> list[Path]:
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 2, figsize=(15.2, 10.0))
    axes = axes.flatten()
    labels = [_ARM_LABELS.get(arm, arm) for arm in arms]
    x = np.arange(len(arms), dtype=float)
    for ax in axes:
        style.style_ax(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)

    # Every level bar has height four: pass + fail + treatment-not-reached.  Not-reached is hatched
    # because it is a lifecycle state, not another performance series.
    ax = axes[0]
    width = 0.18
    level_offsets = np.linspace(-0.30, 0.30, len(_LEVELS))
    for level_index, (level, offset) in enumerate(zip(_LEVELS, level_offsets, strict=True)):
        for arm_index, arm in enumerate(arms):
            xpos = x[arm_index] + offset
            counts = outcomes["levels"][arm][level]
            style.block_shadow(ax, xpos - width / 2, 0, width, 4, z=2.4)
            bottom = 0
            for state, color, hatch in (
                    ("pass", style.NAVY, ""), ("fail", style.MAUVE, ""),
                    ("not_reached", style.SLATE, "///")):
                value = int(counts[state])
                if value:
                    ax.bar([xpos], [value], width, bottom=bottom, color=color,
                           edgecolor=style.INK, linewidth=1.0, hatch=hatch or None, zorder=3)
                bottom += value
            ax.text(xpos, 4.05, level, ha="center", va="bottom", fontsize=6.8,
                    color=style.INK)
    ax.set_ylim(0, 4.65)
    ax.set_yticks(range(5))
    ax.set_ylabel("scheduled cells (n=4 per arm)")
    style.title(ax, "Generic compiler gates — L0 through L3")
    ax.legend(handles=[
        Patch(facecolor=style.NAVY, edgecolor=style.INK, label="pass"),
        Patch(facecolor=style.MAUVE, edgecolor=style.INK, label="fail"),
        Patch(facecolor=style.SLATE, edgecolor=style.INK, hatch="///", label="not reached"),
    ], fontsize=7.8, ncol=3, loc="lower left")

    ax = axes[1]
    split_style = {"train": (style.GOLD, -0.12), "validation": (style.NAVY, 0.12)}
    ratios: list[float] = []
    for arm_index, arm in enumerate(arms):
        has_ratio = False
        for split, (color, offset) in split_style.items():
            summary = outcomes["generic_train_validation_speedup"][arm][split]
            if summary["median"] is None:
                continue
            has_ratio = True
            median = float(summary["median"])
            minimum, maximum = float(summary["minimum"]), float(summary["maximum"])
            ratios.extend((minimum, maximum))
            xpos = x[arm_index] + offset
            ax.errorbar([xpos], [median], yerr=[[median - minimum], [maximum - median]],
                        fmt="o", markersize=7, color=color, ecolor=style.INK,
                        elinewidth=1.3, capsize=3.0, zorder=5)
            ax.text(xpos, maximum, f"{median:.2f}×", ha="center", va="bottom",
                    fontsize=7.2, fontweight="bold", color=color)
        if not has_ratio:
            statuses = {cell["search"]["status"] for cell in outcomes["cells"]
                        if cell["arm"] == arm}
            text = ("N/A" if statuses == {"not_applicable"} else
                    "NO ACCEPTED\nACTION" if "no_accepted_action" in statuses else
                    "SEARCH\nUNAVAILABLE")
            ax.text(x[arm_index], 0.82, text, ha="center", va="bottom", fontsize=6.5,
                    color=style.MAUVE)
    ax.axhline(1.0, color=style.INK, ls="--", lw=1.0, alpha=0.7)
    ax.set_ylim(min([0.78, *(value * 0.92 for value in ratios)]),
                max([1.22, *(value * 1.10 for value in ratios)]))
    ax.set_ylabel("parent / accepted child latency (×)")
    style.title(ax, "Final accepted marginal — generic K1 train/validation")
    ax.legend(handles=[
        Patch(facecolor=style.GOLD, edgecolor=style.INK, label="train"),
        Patch(facecolor=style.NAVY, edgecolor=style.INK, label="validation"),
    ], fontsize=8.0, ncol=2, loc="upper left")

    colors = [style.MAUVE, style.GOLD, style.SLATE, style.NAVY]
    ax = axes[2]
    for arm_index, (arm, color) in enumerate(zip(arms, colors, strict=True)):
        summary = outcomes["compiler_package_size_bytes"][arm]
        if summary["median"] is None:
            ax.scatter([x[arm_index]], [0], marker="x", s=42, color=color, zorder=5)
            ax.text(x[arm_index], 0, "UNAVAILABLE", rotation=90, ha="center", va="bottom",
                    fontsize=6.2, color=color)
            continue
        scaled = {key: (float(value) / 1024.0 if key in {"median", "minimum", "maximum"}
                        else value) for key, value in summary.items()}
        _range_bar(ax, arm_index, scaled, color=color, style=style,
                   label_format="{:,.1f} KiB")
        if summary["available_cells"] != 4:
            ax.text(arm_index, 0, f"{summary['available_cells']}/4 sealed", rotation=90,
                    ha="center", va="bottom", fontsize=6.2, color=style.INK)
    ax.set_ylabel("sealed compiler-package bytes (KiB)")
    style.title(ax, "Compiler package size")

    ax = axes[3]
    for arm_index, (arm, color) in enumerate(zip(arms, colors, strict=True)):
        summary = outcomes["selected_policy_action_count"][arm]
        if summary["median"] is None:
            ax.scatter([x[arm_index]], [0], marker="x", s=42, color=color, zorder=5)
            ax.text(x[arm_index], 0, "N/A", ha="center", va="bottom", fontsize=7.0,
                    color=color)
            continue
        _range_bar(ax, arm_index, summary, color=color, style=style, label_format="{:,.1f}")
        if float(summary["median"]) == 0:
            ax.scatter([x[arm_index]], [0], marker="o", s=36, color=color, zorder=5)
            ax.text(x[arm_index], 0, "0", ha="center", va="bottom", fontsize=8.0,
                    fontweight="bold", color=style.GOLD)
    ax.set_ylabel("actions in converged selected policy")
    style.title(ax, "Deterministic generic-search result")

    fig.suptitle("CPU-host compiler experiment — generic outcomes across Arm1–4",
                 fontfamily=style.SERIF, fontsize=17, color=style.INK, y=1.01)
    fig.text(0.5, 0.01,
             "descriptive Williams 4×4 campaign; generic train/validation only; no paper holdout "
             "is read; whiskers = observed min–max over available sealed cells",
             ha="center", fontsize=8.2, color=style.INK)
    fig.tight_layout(rect=(0, 0.045, 1, 0.98))
    stem = output / "arm1_4_compiler_outcomes"
    written = []
    for suffix, kwargs in ((".png", {"dpi": 180}), (".svg", {})):
        path = stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", facecolor=style.BG, **kwargs)
        written.append(path)
    plt.close(fig)
    return written


def generate_cpu_host_resource_figure(campaign_path: str | Path, *,
                                      output_dir: str | Path | None = None) -> Path:
    """Render provenance-bound Arm1--4 resource and generic-outcome figures.

    The historical function name is retained as the public API/entry point.  Both views are now
    generated together so a paper run cannot publish agent cost while silently omitting treatment
    effectiveness or lifecycle failures.
    """
    campaign_path = Path(campaign_path).resolve()
    spec, record = _load_campaign(campaign_path)
    arms, grouped = _campaign_rows(spec, record)
    summaries = _summaries(arms, grouped)
    outcome_summaries = _campaign_outcomes(spec, arms, grouped)
    campaign_sha = _sha256(campaign_path)
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = (artifacts_dir() / "paper-figures" / "k1-cpu-host" /
                  f"{stamp}_{campaign_sha[:8]}")
    else:
        output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from merlin.plotting import merlin_plotstyle as style
    style.use_merlin_style()

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.5))
    colors = [style.MAUVE, style.GOLD, style.SLATE, style.NAVY]
    labels = [_ARM_LABELS.get(arm, arm) for arm in arms]
    for ax in axes:
        style.style_ax(ax)
        ax.set_xticks(np.arange(len(arms)))
        ax.set_xticklabels(labels, fontsize=8.5)
    for index, (arm, color) in enumerate(zip(arms, colors)):
        _range_bar(axes[0], index, summaries[arm]["provider_tokens"], color=color, style=style,
                   label_format="{:,.0f}")
        reasoning = summaries[arm]["reasoning_tokens"]["median"]
        axes[0].text(index, 0, f"reasoning med. {reasoning:,.0f}", rotation=90,
                     ha="center", va="bottom", fontsize=6.4, color=style.INK)
        _range_bar(axes[1], index, summaries[arm]["cell_wall_seconds"], color=color, style=style,
                   label_format="{:,.1f}s")
        active = summaries[arm]["agent_active_seconds"]["median"]
        grader = summaries[arm]["grader_seconds"]["median"]
        axes[1].text(index, 0, f"agent {active:,.1f}s / grader {grader:,.1f}s", rotation=90,
                     ha="center", va="bottom", fontsize=6.4, color=style.INK)
        _range_bar(axes[2], index, summaries[arm]["tool_calls"], color=color, style=style,
                   label_format="{:,.0f}")
        passes = summaries[arm]["terminal_outcomes"].count("graded_pass")
        axes[2].text(index, 0, f"{passes}/4 graded pass", rotation=90,
                     ha="center", va="bottom", fontsize=6.4, color=style.INK)
    axes[0].set_ylabel("provider input + output tokens per cell")
    axes[1].set_ylabel("driver-observed cell wall time (seconds)")
    axes[2].set_ylabel("AET-reconciled tool calls per cell")
    style.title(axes[0], "Token cost")
    style.title(axes[1], "Time cost")
    style.title(axes[2], "Tool-interaction cost")
    fig.suptitle("CPU-host compiler experiment — descriptive Arm1–4 resource cost",
                 fontfamily=style.SERIF, fontsize=17, color=style.INK, y=1.01)
    fig.text(0.5, 0.01,
             "median across four scheduled Williams blocks; whiskers = observed min–max; "
             "subscription_notional supplies no per-run currency amount",
             ha="center", fontsize=8.2, color=style.INK)
    fig.tight_layout(rect=(0, 0.055, 1, 0.97))
    stem = output / "arm1_4_resource_cost"
    written = []
    for suffix, kwargs in ((".png", {"dpi": 180}), (".svg", {})):
        path = stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", facecolor=style.BG, **kwargs)
        written.append(path)
    plt.close(fig)
    written += _render_outcome_figure(
        output, arms, outcome_summaries, plt=plt, np=np, style=style)

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "campaign": str(campaign_path), "campaign_file_sha256": campaign_sha,
            "campaign_record_sha256": spec.freeze["campaign_record_sha256"],
            "analysis_plan_sha256": spec.analysis["sha256"],
        },
        "claim_scope": (
            "descriptive_small_n_all_sixteen_predeclared_cells_"
            "generic_development_only"),
        "billing_mode": spec.agent["billing"],
        "currency_cost": {"status": "not_available", "reason": "subscription_notional"},
        "resource_summaries": summaries,
        # Compatibility for consumers of the original resource-only manifest.
        "summaries": summaries,
        "outcome_summaries": outcome_summaries,
        "figures": [{"path": path.name, "sha256": _sha256(path)} for path in written],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True,
                        help="completed CPU-host experiment YAML from complete_campaign.py")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    output = generate_cpu_host_resource_figure(args.campaign, output_dir=args.output_dir)
    print(json.dumps({"output_dir": str(output), "manifest": str(output / "manifest.json")},
                     indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
