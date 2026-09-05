"""Render claim-safe generic CPU-host beam-search trees from frozen campaign evidence.

This reader is intentionally separate from the Arm1--4 outcome summary.  A tree is emitted only
when a completed campaign binds all three sources needed to reconstruct it: the frozen public
action catalogue, the converged search record, and the trusted broker ledger with terminal
request/receipt associations.  Candidate policies are reconstructed from the catalogue; paths in
broker requests are never dereferenced and controller-private capsule identities are never copied
to the figure manifest.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from merlin.benchharness.host_agent import (
    _submission_package_digest,
    _terminal_receipt_summary,
)
from merlin.common.paths import artifacts_dir
from merlin.plotting.cpu_host_experiment_figures import (
    _ARM_LABELS,
    _SEARCH_CAPABILITY,
    _bound_json,
    _campaign_rows,
    _canonical_sha256,
    _is_sha256,
    _load_campaign,
    _positive_ratio,
    _sha256,
    _validate_search_candidate,
)


_ALL_FAMILIES = {
    "contraction", "elementwise_map", "reduction", "movement_layout",
    "fusion_epilogue", "runtime_parallel",
}


def _candidate(actions: list[dict[str, Any]], space: Mapping[str, Any], *, label: str
               ) -> dict[str, Any]:
    canonical = sorted(actions, key=lambda action: (
        int(action["stage"]), str(action["group"]), str(action["id"])))
    payload = [{key: value for key, value in action.items() if key != "evidence"}
               for action in canonical]
    return _validate_search_candidate({
        "version": 1, "candidate_sha256": _canonical_sha256(payload),
        "actions": canonical,
    }, space, label=label)


def _expand(parent: Mapping[str, Any], space: Mapping[str, Any], *, label: str
            ) -> dict[str, dict[str, Any]]:
    actions = space.get("actions")
    if not isinstance(actions, list):
        raise ValueError("frozen optimization space has no action catalogue")
    used_groups = {str(action["group"]) for action in parent["actions"]}
    children: dict[str, dict[str, Any]] = {}
    for action in actions:
        if not isinstance(action, Mapping) or str(action.get("group", "")) in used_groups:
            continue
        child = _candidate([*parent["actions"], dict(action)], space, label=label)
        children[child["candidate_sha256"]] = child
    return dict(sorted(children.items()))


def _safe_ledger_file(ledger: Path, relative: object, expected_sha256: object, *, label: str
                      ) -> Path:
    if not isinstance(relative, str) or not relative or not _is_sha256(expected_sha256):
        raise ValueError(f"{label} has no closed relative path and SHA-256")
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"{label} escapes the trusted ledger")
    path = (ledger / raw).resolve()
    if (not path.is_relative_to(ledger.resolve()) or not path.is_file() or path.is_symlink()
            or _sha256(path) != expected_sha256):
        raise ValueError(f"{label} differs from the digest-bound trusted ledger")
    return path


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _observation_rows(path: Path, *, expected_count: int, label: str
                      ) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{label}:{line_number} is not a JSON object")
        rows.append(value)
    if len(rows) != expected_count:
        raise ValueError(f"{label} does not contain the frozen capsule sample count")
    # IDs are used only to prove exact-once coverage.  They are deliberately discarded here and
    # never returned to the caller or written to the manifest.
    identities = [row.get("capsule_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in identities) or (
            len(set(identities)) != len(identities)):
        raise ValueError(f"{label} does not cover each capsule identity exactly once")
    return rows


def _affected_families(action: Mapping[str, Any]) -> set[str]:
    raw = action.get("affected_families")
    if raw is None:
        return set(_ALL_FAMILIES)
    if (not isinstance(raw, list) or not raw or
            any(not isinstance(value, str) or value not in _ALL_FAMILIES for value in raw)):
        raise ValueError(f"action {action.get('id')} has invalid affected_families")
    return set(raw)


def _aggregate_observation(
    path: Path,
    *,
    evaluation: Mapping[str, Any],
    parent_sha256: str,
    candidate_sha256: str,
    action: Mapping[str, Any],
    expected_count: int,
    repeats: int,
    minimum_pairwise_wins: int,
    label: str,
) -> dict[str, Any]:
    phase, split = evaluation.get("phase"), evaluation.get("split")
    if (phase, split) not in {("screen", "train"), ("confirm", "train"),
                              ("confirm", "validation")}:
        raise ValueError(f"{label} has a forbidden phase/split")
    expected_repeats = 1 if phase == "screen" else repeats
    if evaluation.get("measurement_repeats") != expected_repeats:
        raise ValueError(f"{label} has the wrong measurement repeat count")
    rows = _observation_rows(path, expected_count=expected_count, label=label)
    affected = _affected_families(action)
    speedups: list[float] = []
    affected_speedups: list[float] = []
    families: set[str] = set()
    failures = 0
    for row in rows:
        if (row.get("parent_candidate_sha256") != parent_sha256 or
                row.get("candidate_sha256") != candidate_sha256):
            raise ValueError(f"{label} observation is not bound to its parent/child policy")
        family = row.get("family")
        if not isinstance(family, str) or family not in _ALL_FAMILIES:
            raise ValueError(f"{label} observation has an unknown generic family")
        families.add(family)
        if row.get("correctness_ok") is not True:
            failures += 1
        baseline_code, candidate_code = (
            row.get("baseline_code_sha256"), row.get("candidate_code_sha256"))
        if not _is_sha256(baseline_code) or not _is_sha256(candidate_code):
            raise ValueError(f"{label} observation has malformed executable-code digests")
        if family in affected and baseline_code == candidate_code:
            failures += 1
        if phase == "screen":
            if (row.get("screen_authority") != "spike_rv64gcv_mcycle_trusted_harness" or
                    row.get("code_digest_authority") !=
                    "compiled_kernel_object_text_section"):
                raise ValueError(f"{label} observation has no trusted Spike authority")
            baseline = _positive_integer(row.get("baseline_cycles"), label=f"{label} baseline")
            measured = _positive_integer(row.get("candidate_cycles"), label=f"{label} candidate")
            speedup = baseline / measured
        else:
            if (row.get("timing_authority") !=
                    "spacemit_k1_elapsed_ns_div_completed_calls" or
                    row.get("code_digest_authority") !=
                    "measured_k1_kernel_object_text_section"):
                raise ValueError(f"{label} observation has no trusted K1 timing authority")
            arrays: list[list[int]] = []
            for name in ("baseline_elapsed_ns", "baseline_calls",
                         "candidate_elapsed_ns", "candidate_calls"):
                values = row.get(name)
                if (not isinstance(values, list) or len(values) != repeats or
                        any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                            for value in values)):
                    raise ValueError(f"{label} does not contain {repeats} positive {name}")
                arrays.append(values)
            paired = [(base_elapsed / base_calls) / (candidate_elapsed / candidate_calls)
                      for base_elapsed, base_calls, candidate_elapsed, candidate_calls in zip(
                          *arrays, strict=True)]
            speedup = statistics.median(paired)
            if family in affected and sum(value > 1.0 for value in paired) < minimum_pairwise_wins:
                failures += 1
        if not math.isfinite(speedup) or speedup <= 0:
            raise ValueError(f"{label} contains a non-positive speedup")
        speedups.append(float(speedup))
        if family in affected:
            affected_speedups.append(float(speedup))
    if not speedups or not affected_speedups:
        raise ValueError(f"{label} has no observations for the marginal action families")
    return {
        "phase": phase, "split": split, "capsule_count": len(rows),
        "family_count": len(families), "failure_count": failures,
        "median_speedup": statistics.median(speedups),
        "minimum_speedup": min(speedups),
        "affected_median_speedup": statistics.median(affected_speedups),
        "observations_sha256": evaluation["observations_sha256"],
        "evaluation_wall_ns": evaluation["wall_ns"],
    }


def _added_action(parent: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
    parent_ids = {str(action["id"]) for action in parent["actions"]}
    added = [dict(action) for action in child["actions"]
             if str(action["id"]) not in parent_ids]
    if len(added) != 1:
        raise ValueError("beam candidate does not add exactly one frozen action")
    return added[0]


def _evaluation_key(parent: str, candidate: str, split: str, phase: str) -> str:
    return f"{parent}:{candidate}:{split}:{phase}"


def _metric_matches(actual: object, expected: float, *, label: str) -> None:
    value = _positive_ratio(actual, label=label)
    if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{label} differs from trusted observation aggregation")


def _verify_search_cell(spec: Any, space: Mapping[str, Any], row: Mapping[str, Any]
                        ) -> dict[str, Any]:
    run_dir = Path(row["run_dir"]).resolve()
    seal, seal_path = _bound_json(
        run_dir, "contracts/trusted_search_seal.json", row["search_seal_sha256"],
        label=f"{row['run_id']} trusted search seal")
    if seal.get("status") == "fail":
        terminal_failures = {
            "treatment_search_fail", "treatment_build_fail", "treatment_agent_fail",
        }
        package = run_dir / "artifacts" / "compiler_submission"
        if (row["outcome"] not in terminal_failures or
                seal.get("failure_class") not in terminal_failures or
                row["compiler_seal"].get("status") != "not_run" or package.exists()):
            raise ValueError(
                f"{row['run_id']} unavailable beam cell is not a typed treatment failure")
        return {
            "arm": row["arm"], "repeat": row["repeat"], "run_id": row["run_id"],
            "outcome": row["outcome"], "status": "unavailable",
            "reason": "trusted_search_failed",
            "search_seal": {"path": str(seal_path), "sha256": row["search_seal_sha256"]},
        }
    if seal.get("status") != "pass" or row["outcome"] not in {"graded_pass", "graded_fail"}:
        raise ValueError(f"{row['run_id']} search lifecycle and terminal outcome disagree")

    package = run_dir / "artifacts" / "compiler_submission"
    compiler_seal = row["compiler_seal"]
    if (compiler_seal.get("status") != "sealed" or package.is_symlink() or
            not package.is_dir() or
            not _is_sha256(compiler_seal.get("compiler_package_sha256")) or
            _submission_package_digest(package) != compiler_seal["compiler_package_sha256"]):
        raise ValueError(f"{row['run_id']} beam source package differs from its compiler seal")
    record_sha = seal.get("search_record_sha256")
    policy_sha = seal.get("selected_policy_sha256")
    record, record_path = _bound_json(
        package, "search/search_record.json", record_sha,
        label=f"{row['run_id']} search record")
    policy, policy_path = _bound_json(
        package, "search/selected_policy.json", policy_sha,
        label=f"{row['run_id']} selected policy")
    selected = _validate_search_candidate(
        policy, space, label=f"{row['run_id']} selected policy")
    if (compiler_seal.get("search_record_sha256") != record_sha or
            compiler_seal.get("selected_policy_sha256") != policy_sha or
            compiler_seal.get("policy_sha256") != policy_sha):
        raise ValueError(f"{row['run_id']} compiler seal is not bound to the selected policy")

    space_path = spec._repo_path(spec.search.get("space", ""))
    if (record.get("version") != 1 or record.get("status") != "converged" or
            record.get("heldout_visible") is not False or
            record.get("selection_policy") != "spike_screen_then_k1_confirmation" or
            record.get("space_sha256") != _sha256(space_path) or
            record.get("selected_policy_sha256") != policy_sha):
        raise ValueError(f"{row['run_id']} search record is not a frozen holdout-blind result")

    ledger = run_dir / "metrics" / "trusted_search_ledger"
    if not ledger.is_dir() or ledger.is_symlink():
        raise ValueError(f"{row['run_id']} trusted ledger is not a retained regular directory")
    index_path = _safe_ledger_file(
        ledger, "index.json", seal.get("trusted_ledger_sha256"),
        label=f"{row['run_id']} trusted ledger index")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(index, dict):
        raise ValueError(f"{row['run_id']} trusted ledger index is not a mapping")
    evaluations = index.get("evaluations")
    if (index.get("version") != 1 or
            index.get("authority") != "trusted_spacemit_k1_outside_agent_sandbox" or
            index.get("heldout_opened") is not False or
            index.get("space_sha256") != spec.search.get("space_sha256") or
            not isinstance(evaluations, dict) or not evaluations):
        raise ValueError(f"{row['run_id']} trusted ledger identity is incomplete")
    receipt_summary = _terminal_receipt_summary(ledger, index)
    if (receipt_summary != {"integrity": True, "all_pass": True, "failure_classes": []}):
        raise ValueError(f"{row['run_id']} request/receipt associations are incomplete")
    if (seal.get("trusted_evaluation_count") != len(evaluations) or
            seal.get("trusted_evaluation_wall_ns") != sum(
                value.get("wall_ns", -1) for value in evaluations.values()
                if isinstance(value, Mapping))):
        raise ValueError(f"{row['run_id']} ledger totals differ from the trusted seal")

    repeats = _positive_integer(record.get("measurement_repeats"), label="measurement_repeats")
    if index.get("measurement_repeats") != repeats:
        raise ValueError(f"{row['run_id']} ledger/search repeat counts differ")
    sample_counts = record.get("sample_counts")
    if not isinstance(sample_counts, Mapping):
        raise ValueError(f"{row['run_id']} search record has no frozen sample counts")
    expected_counts = {
        ("screen", "train"): _positive_integer(
            sample_counts.get("screen_train"), label="screen_train sample count"),
        ("confirm", "train"): _positive_integer(
            sample_counts.get("confirmation_train"), label="confirmation_train sample count"),
        ("confirm", "validation"): _positive_integer(
            sample_counts.get("confirmation_validation"),
            label="confirmation_validation sample count"),
    }
    thresholds = record.get("acceptance_thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError(f"{row['run_id']} search record has no acceptance thresholds")
    margin = float(thresholds.get("calibrated_upper_margin", -1))
    minimum_pairwise_wins = _positive_integer(
        thresholds.get("minimum_pairwise_wins_per_affected_capsule"),
        label="minimum pairwise wins")
    minimum_families = _positive_integer(space.get("minimum_families"),
                                         label="minimum_families")
    confirmation_width = _positive_integer(record.get("confirmation_width"),
                                           label="confirmation_width")
    if (not math.isfinite(margin) or margin < 0 or
            confirmation_width != space.get("confirmation_width")):
        raise ValueError(f"{row['run_id']} search thresholds differ from frozen space")

    def load_evaluation(parent: dict[str, Any], child: dict[str, Any], split: str,
                        phase: str) -> dict[str, Any]:
        key = _evaluation_key(parent["candidate_sha256"], child["candidate_sha256"], split, phase)
        raw = evaluations.get(key)
        if (not isinstance(raw, Mapping) or raw.get("parent_candidate_sha256") !=
                parent["candidate_sha256"] or raw.get("candidate_sha256") !=
                child["candidate_sha256"] or raw.get("split") != split or
                raw.get("phase") != phase or not isinstance(raw.get("wall_ns"), int) or
                raw["wall_ns"] < 0):
            raise ValueError(f"{row['run_id']} lacks trusted evaluation {key}")
        artifact = _safe_ledger_file(
            ledger, raw.get("observations"), raw.get("observations_sha256"),
            label=f"{row['run_id']} trusted observation {key}")
        used_keys.add(key)
        return _aggregate_observation(
            artifact, evaluation=raw, parent_sha256=parent["candidate_sha256"],
            candidate_sha256=child["candidate_sha256"],
            action=_added_action(parent, child), expected_count=expected_counts[phase, split],
            repeats=repeats, minimum_pairwise_wins=minimum_pairwise_wins,
            label=f"{row['run_id']} trusted observation {key}")

    empty = _candidate([], space, label=f"{row['run_id']} empty policy")
    incumbent = empty
    sweeps = record.get("sweeps")
    accepted = record.get("accepted")
    if not isinstance(sweeps, list) or not sweeps or not isinstance(accepted, list):
        raise ValueError(f"{row['run_id']} search progression is absent")
    used_keys: set[str] = set()
    public_sweeps: list[dict[str, Any]] = []
    accepted_index = 0
    trailing_empty = 0
    for sweep_index, sweep in enumerate(sweeps):
        if (not isinstance(sweep, Mapping) or sweep.get("sweep") != sweep_index or
                sweep.get("incumbent") != incumbent["candidate_sha256"]):
            raise ValueError(f"{row['run_id']} sweep {sweep_index} has an invalid incumbent")
        children = _expand(
            incumbent, space, label=f"{row['run_id']} sweep {sweep_index} candidate")
        screen_metrics = {digest: load_evaluation(incumbent, child, "train", "screen")
                          for digest, child in children.items()}
        eligible_screen = sorted(
            (digest for digest, metric in screen_metrics.items()
             if metric["failure_count"] == 0 and metric["family_count"] >= minimum_families),
            key=lambda digest: (-screen_metrics[digest]["affected_median_speedup"],
                                -screen_metrics[digest]["minimum_speedup"], digest))
        if sweep.get("screened") != eligible_screen:
            raise ValueError(f"{row['run_id']} sweep {sweep_index} screened ranking is incomplete")
        confirm_requested = eligible_screen[:confirmation_width]
        train_metrics = {digest: load_evaluation(
            incumbent, children[digest], "train", "confirm") for digest in confirm_requested}
        eligible_confirm = sorted(
            (digest for digest, metric in train_metrics.items()
             if metric["failure_count"] == 0 and metric["family_count"] >= minimum_families),
            key=lambda digest: (-train_metrics[digest]["affected_median_speedup"],
                                -train_metrics[digest]["minimum_speedup"], digest))
        if sweep.get("confirmed") != eligible_confirm:
            raise ValueError(f"{row['run_id']} sweep {sweep_index} confirmed ranking is incomplete")
        validation_metrics = {digest: load_evaluation(
            incumbent, children[digest], "validation", "confirm")
            for digest in eligible_confirm}
        promoted = [digest for digest in eligible_confirm
                    if validation_metrics[digest]["failure_count"] == 0
                    and validation_metrics[digest]["family_count"] >= minimum_families
                    and train_metrics[digest]["affected_median_speedup"] > 1.0 + margin
                    and validation_metrics[digest]["affected_median_speedup"] > 1.0 + margin
                    and validation_metrics[digest]["minimum_speedup"] >= 1.0 / (1.0 + margin)]
        promoted.sort(key=lambda digest: (
            -validation_metrics[digest]["affected_median_speedup"],
            -train_metrics[digest]["affected_median_speedup"], digest))
        winner = promoted[0] if promoted else None
        if sweep.get("promoted") != promoted or sweep.get("winner") != winner:
            raise ValueError(f"{row['run_id']} sweep {sweep_index} promotion differs from ledger")

        node_rows: list[dict[str, Any]] = []
        for digest, child in children.items():
            status = ("winner" if digest == winner else "promoted" if digest in promoted else
                      "confirmed" if digest in eligible_confirm else
                      "screened" if digest in eligible_screen else "pruned")
            action = _added_action(incumbent, child)
            node = {
                "candidate_sha256": digest, "parent_candidate_sha256": incumbent[
                    "candidate_sha256"],
                "added_action_id": str(action["id"]),
                "added_action_class": str(action.get("action_class", "")),
                "status": status, "screen": screen_metrics[digest],
                "train_confirmation": train_metrics.get(digest),
                "validation_confirmation": validation_metrics.get(digest),
            }
            node_rows.append(node)
        public_sweeps.append({
            "sweep": sweep_index, "incumbent_candidate_sha256": incumbent[
                "candidate_sha256"], "winner_candidate_sha256": winner,
            "candidate_count": len(children), "nodes": node_rows,
        })

        if winner is None:
            trailing_empty += 1
        else:
            trailing_empty = 0
            if accepted_index >= len(accepted) or not isinstance(accepted[accepted_index], Mapping):
                raise ValueError(f"{row['run_id']} winner has no accepted record")
            entry = accepted[accepted_index]
            accepted_candidate = _validate_search_candidate(
                entry.get("candidate"), space,
                label=f"{row['run_id']} accepted candidate {accepted_index}")
            if accepted_candidate != children[winner]:
                raise ValueError(f"{row['run_id']} accepted candidate differs from sweep winner")
            for split, metric in (("train", train_metrics[winner]),
                                  ("validation", validation_metrics[winner])):
                result = entry.get(split)
                if (not isinstance(result, Mapping) or result.get("failures") != [] or
                        result.get("parent_candidate_sha256") != incumbent["candidate_sha256"] or
                        result.get("candidate_sha256") != winner or
                        result.get("observations_sha256") != metric["observations_sha256"]):
                    raise ValueError(f"{row['run_id']} accepted {split} result is not ledger-bound")
                for key in ("median_speedup", "minimum_speedup", "affected_median_speedup"):
                    _metric_matches(result.get(key), metric[key],
                                    label=f"{row['run_id']} accepted {split} {key}")
            accepted_index += 1
            incumbent = children[winner]

    required_empty = _positive_integer(record.get("required_empty_sweeps"),
                                       label="required_empty_sweeps")
    if (record.get("empty_sweeps") != trailing_empty or trailing_empty < required_empty or
            accepted_index != len(accepted) or incumbent != selected):
        raise ValueError(f"{row['run_id']} search did not end in an independently empty sweep")
    if used_keys != set(evaluations):
        raise ValueError(f"{row['run_id']} trusted ledger contains evaluations outside the tree")

    receipt_counts = {key: 0 for key in evaluations}
    for receipt in index["terminal_receipts"].values():
        receipt_counts[str(receipt["evaluation_key"])] += 1
    return {
        "arm": row["arm"], "repeat": row["repeat"], "run_id": row["run_id"],
        "outcome": row["outcome"], "status": "pass",
        "privacy": "controller_private_capsule_identities_omitted",
        "search_seal": {"path": str(seal_path), "sha256": row["search_seal_sha256"]},
        "search_record": {"path": str(record_path), "sha256": record_sha},
        "selected_policy": {"path": str(policy_path), "sha256": policy_sha,
                            "action_ids": [str(action["id"]) for action in selected["actions"]]},
        "trusted_ledger_index": {"path": str(index_path),
                                 "sha256": seal["trusted_ledger_sha256"]},
        "request_receipt_associations": {
            "integrity": True, "all_pass": True,
            "terminal_receipt_count": sum(receipt_counts.values()),
            "evaluation_request_multiplicity": receipt_counts,
        },
        "evaluation_count": len(evaluations), "sweeps": public_sweeps,
    }


def _render_tree(output: Path, cell: Mapping[str, Any], *, plt: Any, style: Any) -> list[Path]:
    from matplotlib.lines import Line2D

    sweeps = cell["sweeps"]
    maximum_width = max(len(sweep["nodes"]) for sweep in sweeps)
    fig, ax = plt.subplots(figsize=(max(12.5, (len(sweeps) + 1) * 2.2),
                                    max(6.8, maximum_width * 0.72)))
    style.style_ax(ax, grid=None)
    colors = {
        "winner": style.NAVY, "promoted": style.GOLD, "confirmed": style.SAGE,
        "screened": style.SLATE, "pruned": style.MAUVE,
    }
    coordinates = {sweeps[0]["incumbent_candidate_sha256"]: (0.0, 0.0)}
    ax.scatter([0], [0], s=720, color=style.BLUE, edgecolor=style.INK, linewidth=1.4, zorder=5)
    ax.text(0, 0, "ROOT\n1.00×", color="white", ha="center", va="center",
            fontsize=8, fontweight="bold", zorder=6)
    for sweep in sweeps:
        x = float(sweep["sweep"] + 1)
        nodes = sweep["nodes"]
        positions = [index - (len(nodes) - 1) / 2 for index in range(len(nodes))]
        parent_xy = coordinates.get(sweep["incumbent_candidate_sha256"])
        if parent_xy is None:
            raise ValueError("rendered tree lacks the verified incumbent coordinate")
        for y, node in zip(positions, nodes, strict=True):
            ax.plot([parent_xy[0], x], [parent_xy[1], y], color=style.INK,
                    lw=0.8, alpha=0.42, zorder=1)
            color = colors[node["status"]]
            ax.scatter([x], [y], s=590, color=color, edgecolor=style.INK,
                       linewidth=1.25, zorder=4)
            action_label = textwrap.fill(node["added_action_id"], width=16)
            label = f"{action_label}\nS {node['screen']['affected_median_speedup']:.2f}×"
            if node["train_confirmation"] is not None:
                label += f"\nK1 {node['train_confirmation']['affected_median_speedup']:.2f}×"
            if node["validation_confirmation"] is not None:
                label += f" / {node['validation_confirmation']['affected_median_speedup']:.2f}×"
            ax.text(x, y, label, color="white", ha="center", va="center",
                    fontsize=6.5, fontweight="bold", zorder=5)
            coordinates[node["candidate_sha256"]] = (x, y)
    ax.set_xticks(range(len(sweeps) + 1))
    ax.set_xticklabels(["root", *[f"sweep {index}" for index in range(len(sweeps))]])
    ax.set_yticks([])
    ax.set_xlabel("deterministic one-action expansion (S = trusted Spike; K1 = train / validation)")
    style.title(ax, f"{_ARM_LABELS.get(cell['arm'], cell['arm']).replace(chr(10), ' ')} · "
                    f"Williams block {cell['repeat'] + 1}")
    ax.legend(handles=[Line2D([0], [0], marker="o", linestyle="", markersize=9,
                              markerfacecolor=color, markeredgecolor=style.INK, label=status)
                       for status, color in colors.items()], ncol=5, fontsize=7.5,
              loc="upper center", bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Generic compiler beam search — reconstructed from sealed evidence",
                 fontfamily=style.SERIF, fontsize=16, color=style.INK, y=1.01)
    fig.text(0.5, 0.01,
             "all legal children shown; no paper holdout or controller-private capsule identity",
             ha="center", fontsize=8.2, color=style.INK)
    fig.tight_layout(rect=(0, 0.065, 1, 0.97))
    stem = output / f"beam_{cell['arm']}_r{cell['repeat'] + 1:02d}"
    written: list[Path] = []
    for suffix, kwargs in ((".png", {"dpi": 180}), (".svg", {})):
        path = stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", facecolor=style.BG, **kwargs)
        written.append(path)
    plt.close(fig)
    return written


def _render_coverage(output: Path, cells: list[Mapping[str, Any]], *, plt: Any, np: Any,
                     style: Any) -> list[Path]:
    arms = sorted({str(cell["arm"]) for cell in cells})
    matrix = np.zeros((len(arms), 4), dtype=float)
    labels: list[list[str]] = [["" for _ in range(4)] for _ in arms]
    for arm_index, arm in enumerate(arms):
        by_repeat = {int(cell["repeat"]): cell for cell in cells if cell["arm"] == arm}
        if set(by_repeat) != set(range(4)):
            raise ValueError(f"beam coverage lacks one of four cells for {arm}")
        for repeat, cell in by_repeat.items():
            if cell["status"] == "pass":
                matrix[arm_index, repeat] = 1
                labels[arm_index][repeat] = (
                    f"VERIFIED\n{cell['evaluation_count']} evals / {len(cell['sweeps'])} sweeps")
            else:
                labels[arm_index][repeat] = "UNAVAILABLE\ntrusted search failed"
    fig, ax = plt.subplots(figsize=(10.8, 4.0))
    ax.imshow(matrix, cmap=plt.matplotlib.colors.ListedColormap([style.MAUVE, style.NAVY]),
              vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4), [f"Williams block {index}" for index in range(1, 5)])
    ax.set_yticks(range(len(arms)), [_ARM_LABELS.get(arm, arm).replace("\n", " ") for arm in arms])
    for arm_index in range(len(arms)):
        for repeat in range(4):
            ax.text(repeat, arm_index, labels[arm_index][repeat], ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_color(style.INK)
    style.title(ax, "Claim-safe beam-tree coverage")
    fig.suptitle("CPU-host generic search — completed campaign coverage",
                 fontfamily=style.SERIF, fontsize=16, color=style.INK, y=1.02)
    fig.tight_layout()
    stem = output / "arm3_4_beam_coverage"
    written: list[Path] = []
    for suffix, kwargs in ((".png", {"dpi": 180}), (".svg", {})):
        path = stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", facecolor=style.BG, **kwargs)
        written.append(path)
    plt.close(fig)
    return written


def generate_cpu_host_beam_figures(campaign_path: str | Path, *,
                                   output_dir: str | Path | None = None) -> Path:
    """Verify and render every available Arm3/4 beam tree in a completed campaign."""
    campaign_path = Path(campaign_path).resolve()
    spec, record = _load_campaign(campaign_path)
    arms, grouped = _campaign_rows(spec, record)
    space_path = spec._repo_path(spec.search.get("space", ""))
    space = spec.search_space_config()
    if (not isinstance(space, Mapping) or not space or not space_path.is_file() or
            _sha256(space_path) != spec.search.get("space_sha256")):
        raise ValueError("frozen optimization space differs from completed campaign")
    search_arms = [arm.id for arm in spec.arms if _SEARCH_CAPABILITY in arm.capabilities]
    if len(search_arms) != 2 or any(arm not in arms for arm in search_arms):
        raise ValueError("beam plots require the frozen Arm3/Arm4 deterministic-search cells")

    # Verify every cell before allocating output.  A passing seal with a partial tree is an error;
    # a typed treatment search failure is retained as explicit unavailable coverage.
    cells = [_verify_search_cell(spec, space, {**row, "arm": arm})
             for arm in search_arms for row in grouped[arm]]
    campaign_sha = _sha256(campaign_path)
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = (artifacts_dir() / "paper-figures" / "k1-cpu-host-beam" /
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

    written = _render_coverage(output, cells, plt=plt, np=np, style=style)
    for cell in cells:
        if cell["status"] == "pass":
            written += _render_tree(output, cell, plt=plt, style=style)
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "campaign": str(campaign_path), "campaign_file_sha256": campaign_sha,
            "campaign_record_sha256": spec.freeze["campaign_record_sha256"],
            "analysis_plan_sha256": spec.analysis["sha256"],
            "optimization_space": str(space_path),
            "optimization_space_sha256": spec.search["space_sha256"],
        },
        "claim_scope": "generic_development_beam_progression_no_paper_holdout",
        "privacy": {
            "controller_private_capsule_identities": "omitted",
            "broker_request_workspace_paths": "not_dereferenced_or_exported",
        },
        "structural_policy": (
            "passing cells require a complete reconstructed expansion and exact trusted-ledger "
            "evaluation/request/receipt association; typed search failures remain unavailable"),
        "cells": cells,
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
    output = generate_cpu_host_beam_figures(args.campaign, output_dir=args.output_dir)
    print(json.dumps({"output_dir": str(output), "manifest": str(output / "manifest.json")},
                     indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
