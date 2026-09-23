"""Verify global portfolio checkpoints and their bounded scientific evidence.

Callers supply the current host policy and shared compiler source owner. This
module admits recorded evidence without discovering a checkout or executing a
compiler, simulator, or archived controller.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.benchharness import hash_tree
from merlin.common.digest import sha256_bytes

from . import broker_evidence as BE
from . import contracts as P2_CONTRACTS
from . import emission_analysis as EA
from . import static_identity as SI
from .mechanism_evidence import verify_checkpoint_catalog, verify_checkpoint_work_order
from .revision_journal import RevisionJournal


@dataclass(frozen=True)
class CheckpointVerificationContext:
    """Current verifier identity and explicit shared compiler implementation."""

    host_policy: Mapping[str, Any]
    compiler_shared_source_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "host_policy", copy.deepcopy(dict(self.host_policy)))
        object.__setattr__(self, "compiler_shared_source_root", Path(self.compiler_shared_source_root).resolve())


def known_structural_host_work_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """Lexicographic same-unit deltas; never convert bytes or operations into cycles."""
    plans = [
        (analysis.get("diagnostics") or {}).get("verified_global_plan_emission") or {} for analysis in (before, after)
    ]
    host = [plan.get("host_activity") or {} for plan in plans]
    byte_fields = ("load_payload_bytes", "store_payload_bytes", "static_allocation_payload_bytes")
    byte_known = all(type(row.get(field)) is int and row[field] >= 0 for row in host for field in byte_fields)
    byte_delta = sum(abs(host[1][field] - host[0][field]) for field in byte_fields) if byte_known else None
    operations = [row.get("dynamic_operations") for row in host]
    operations_are_maps = all(isinstance(row, Mapping) for row in operations)
    categories = set(operations[0]) | set(operations[1]) if operations_are_maps else set()
    operations_known = operations_are_maps and all(
        isinstance(category, str) and type(row.get(category, 0)) is int and row.get(category, 0) >= 0
        for row in operations
        for category in categories
    )
    operation_delta = (
        sum(abs(operations[1].get(category, 0) - operations[0].get(category, 0)) for category in categories)
        if operations_known
        else None
    )
    tasks = [plan.get("tasks") for plan in plans]
    task_delta = abs(tasks[1] - tasks[0]) if all(type(value) is int for value in tasks) else None
    return {
        "host_payload_bytes_absolute_delta": byte_delta,
        "host_dynamic_operations_absolute_delta": operation_delta,
        "planned_task_count_absolute_delta": task_delta,
        "ranking_policy": (
            "lexicographic_known_host_payload_bytes_then_known_dynamic_operations_then_"
            "planned_task_count; units are never added or converted to cycles"
        ),
    }


def select_changed_portfolio_contexts(contexts) -> dict[str, Any]:
    """One selection policy for live artifacts and independently verified receipt records."""
    candidates = []
    for index, (before, after) in enumerate(contexts):
        prior_binding, current_binding = before["member_binding"], after["member_binding"]
        if prior_binding["capsule_sha256"] != current_binding["capsule_sha256"]:
            raise ValueError("portfolio member identity changed across candidate revisions")
        if prior_binding["source_sha256"] != current_binding["source_sha256"]:
            raise ValueError("portfolio member source changed across candidate revisions")
        changed_fields = [
            field
            for field in ("plan_digest", "lowered_sha256", "command_buffer_sha256")
            if prior_binding[field] != current_binding[field]
        ]
        # A plan-only metadata delta has no changed lowered/command artifact from which to
        # extract and compile a source witness. It is not silently attributed to another unit.
        if not {"lowered_sha256", "command_buffer_sha256"}.intersection(changed_fields):
            continue
        delta = known_structural_host_work_delta(before["analysis"], after["analysis"])
        values = [
            delta["host_payload_bytes_absolute_delta"],
            delta["host_dynamic_operations_absolute_delta"],
            delta["planned_task_count_absolute_delta"],
        ]
        rank = tuple(item for value in values for item in (value is not None, value if value is not None else -1))
        candidates.append(
            (
                rank,
                -index,
                {
                    "schema": "global_changed_portfolio_member_selection_v1",
                    "portfolio_index": index,
                    "capsule": current_binding["capsule"],
                    "capsule_sha256": current_binding["capsule_sha256"],
                    "changed_artifact_fields": changed_fields,
                    "structural_host_work_delta": delta,
                    "previous": copy.deepcopy(prior_binding),
                    "current": copy.deepcopy(current_binding),
                    "performance_inference": "none",
                },
            )
        )
    if not candidates:
        raise ValueError("no portfolio member has a changed emitted artifact")
    winner = max(candidates, key=lambda item: (item[0], item[1]))
    selection = winner[2]
    known = [
        name
        for name in (
            "host_payload_bytes_absolute_delta",
            "host_dynamic_operations_absolute_delta",
            "planned_task_count_absolute_delta",
        )
        if selection["structural_host_work_delta"][name] is not None
    ]
    selection["selection_basis"] = (
        "known_structural_host_work_lexicographic" if known else "stable_portfolio_order_no_known_structural_host_work"
    )
    selection["known_ranking_metrics"] = known
    selection["stable_portfolio_order_tie_break_applied"] = (
        sum(rank == winner[0] for rank, _order, _record in candidates) > 1
    )
    return selection


def paired_context_decision_feedback(
    record: Mapping[str, Any], receipt: Mapping[str, Any], *, target_sha256: str
) -> dict[str, Any]:
    """Join measured motif evidence to its actual model region without predicting model cycles."""
    analysis = record["analysis"]
    diag = analysis["diagnostics"]
    expected = {
        "graph_digest": diag["captured_logical_graph"]["logical_dispatch_digest"],
        "plan_digest": diag["verified_global_plan_emission"]["plan_digest"],
        "compiler_digest": record["compiler_dependencies"]["compiler_implementation_sha256"],
        "target_digest": target_sha256,
    }
    if (
        receipt["binding"] != expected
        or diag["verified_global_plan_emission"].get("status") != "verified"
        or receipt.get("scope") != "controlled_fixed_work_slice"
        or receipt.get("model_artifact_sha256") != analysis["emission"]["candidate_lowered_sha256"]
        or receipt.get("global_cost_validated") is not False
        or receipt.get("global_speedup_proven") is not False
    ):
        raise ValueError("paired decision feedback is not bound to the current full-model revision")
    proof = receipt["projection_proof"]
    if (
        proof.get("status") != "same_work_projection_verified"
        or proof.get("after_artifact_sha256") != receipt["model_artifact_sha256"]
        or P2_CONTRACTS.document_sha256(proof["work_contract"]) != proof["work_contract_sha256"]
    ):
        raise ValueError("paired decision feedback has no same-work contract")
    executions = receipt["executions"]
    profiles = {arm: executions[arm].get("counter_profile") or {} for arm in ("before", "after")}
    missing = []
    engine_hashes = [executions[arm].get("engine_provenance", {}).get("binary_sha256") for arm in ("before", "after")]
    if not BE._is_sha256(engine_hashes[0]) or engine_hashes[0] != engine_hashes[1]:
        missing.append("same_execution_engine_unproved")
    for arm in ("before", "after"):
        profile, execution = profiles[arm], executions[arm]
        counters = [profile.get(key) for key in ("active_union_cycles", "idle_cycles", "overlap_any_engine_cycles")]
        busy = profile.get("busy_cycles_by_engine_token")
        total = execution.get("total_compute_cycles")
        if (
            execution.get("correct") is not True
            or execution.get("warmup_runs") != 1
            or execution.get("measured_runs") != 1
            or type(total) is not int
            or total <= 0
            or profile.get("kind") != "joint_engine_busy_cycles"
            or profile.get("partition_proof", {}).get("status") != "proved"
            or profile.get("layout", {}).get("complete") is not True
            or not isinstance(busy, Mapping)
            or not busy
            or any(type(value) is not int or value < 0 for value in counters)
            or any(type(value) is not int or not 0 <= value <= total for value in busy.values())
            or counters[0] + counters[1] != total
            or counters[2] > counters[0]
        ):
            missing.append(arm + "_bounded_joint_counter_evidence_missing")
    if not missing:
        if (
            profiles["before"]["partition_proof"] != profiles["after"]["partition_proof"]
            or profiles["before"]["layout"] != profiles["after"]["layout"]
            or profiles["before"]["busy_cycles_by_engine_token"].keys()
            != profiles["after"]["busy_cycles_by_engine_token"].keys()
        ):
            missing.append("counter_semantics_changed_between_arms")
    observation: dict[str, Any] = {}
    status = "counter_evidence_unknown"
    next_step = "Resolve only the missing motif evidence relevant to this schedule; no model-cost projection."
    if not missing:
        observation = {
            "before_cycles": executions["before"]["total_compute_cycles"],
            "after_cycles": executions["after"]["total_compute_cycles"],
            "overlap_delta_cycles": profiles["after"]["overlap_any_engine_cycles"]
            - profiles["before"]["overlap_any_engine_cycles"],
            "busy_delta_by_resource": {
                key: profiles["after"]["busy_cycles_by_engine_token"][key] - value
                for key, value in profiles["before"]["busy_cycles_by_engine_token"].items()
            },
        }
        observation["cycle_delta"] = observation["after_cycles"] - observation["before_cycles"]
        if observation["overlap_delta_cycles"] == 0 and not any(observation["busy_delta_by_resource"].values()):
            status = "no_observed_overlap_or_busy_work_change"
            next_step = (
                "Do not count this schedule as a demonstrated latency-hiding gain. Use the bound "
                "region's movement, dependency and buffer evidence to choose a different transformation; "
                "a cycle-only difference is not statistical confirmation or model speedup."
            )
        else:
            status = "controlled_resource_change_observed"
            next_step = (
                "Inspect the measured resource tradeoff for this exact region; verify actual buffer "
                "capacity and dependency/repetition contracts before any global cost projection."
            )
    inventory = (analysis.get("optimization_brief") or {}).get("package_inventory") or {}
    surfaces = [
        {key: surface[key] for key in ("id", "path", "symbol") if key in surface}
        for surface in inventory.get("surfaces", [])
        if set(surface.get("effects", ())) & {"issue", "latency_hiding"}
    ]
    work = proof["work_contract"]
    return {
        "schema": "global_paired_decision_feedback_v1",
        "status": status,
        "binding": dict(receipt["binding"]),
        "candidate_sha256": record["candidate_sha256"],
        "model_artifact_sha256": receipt["model_artifact_sha256"],
        "source_task_index": work.get("source_task_index"),
        "source_op_indices": work.get("source_op_indices", []),
        "timed_command_count": work.get("timed_command_count"),
        "same_declared_movement_work": bool(work.get("timed_command_multiset")),
        "physical_movement_bytes": None,
        "observation": observation,
        "missing": missing,
        "next_step": next_step,
        "edit_surfaces": surfaces,
        "scope": "controlled_fixed_work_slice",
        "full_model_cycles": None,
        "full_model_cost_selection": "UNKNOWN",
        "global_speedup_proven": False,
        "buffer_capacity_contract": "UNPROVED",
        "pipeline_projection_admitted": False,
    }


def load_historical_reference(
    path: Path, sha256: str, *, target: str | None = None, candidate_roots: Sequence[Path] = ()
) -> tuple[bytes, dict[str, Any]]:
    """Read one explicit host bundle, never discover receipts or mint timing authority."""
    from merlin.perf.historical_reference import reference_summary

    if (
        not BE._is_sha256(sha256)
        or path.is_symlink()
        or not path.is_file()
        or any(parent.is_symlink() for parent in path.parents)
        or any(path.resolve().is_relative_to(root.resolve()) for root in candidate_roots)
    ):
        raise ValueError("historical reference requires a pinned regular host file outside candidate writes")
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("historical reference exceeds the host bundle byte limit")
    raw = path.read_bytes()
    if sha256_bytes(raw) != sha256:
        raise ValueError("historical reference bundle digest changed")

    def unique_fields(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate historical reference bundle field")
            result[key] = value
        return result

    bundle = json.loads(raw, object_pairs_hook=unique_fields)
    if (
        not isinstance(bundle, dict)
        or bundle.get("schema") != "historical_reference_bundle_v1"
        or not isinstance(bundle.get("records"), list)
    ):
        raise ValueError("invalid historical reference bundle schema")
    summary = reference_summary(bundle["records"])
    if bundle.get("summary") != summary:
        raise ValueError("historical reference recorded summary differs from bound records")
    if target is not None and any(row.get("target") != target for row in bundle["records"]):
        raise ValueError("historical reference target differs from the experiment")
    compact = {
        key: summary[key]
        for key in (
            "reference_count",
            "target_cycle_authority",
            "warm_calibration",
            "physical_roofline_complete",
            "full_model_cycle_ordering",
            "next_action",
        )
    }
    compact.update(
        {
            "schema": "historical_reference_agent_brief_v1",
            "engine_group_count": len(summary["engine_groups"]),
            "workload_count": len({row["workload"] for row in bundle["records"]}),
            "exact_positive_command_work_count": len(
                {
                    row["reference_sha256"]
                    for row in bundle["records"]
                    if type((row.get("work") or {}).get("exact_macs")) is int and row["work"]["exact_macs"] > 0
                }
            ),
            "missing_contracts": sorted({item for row in bundle["records"] for item in row["missing_contracts"]}),
            "artifact": {"path": "/perf-control/historical_reference.json", "sha256": sha256},
            "scope": "historical engine-relative references only; not warm rates, peaks or target-cycle authority",
        }
    )
    return raw, compact


def consume_authoring_checkpoint(path: Path, *, context: CheckpointVerificationContext) -> dict[str, Any]:
    """Verify a blocked, exact portfolio checkpoint without granting promotion authority."""
    document = P2_CONTRACTS.mapping_file(path)
    if (
        document.get("schema") != "global_authoring_checkpoint_v1"
        or document.get("promotion_status") != "blocked_authoring_checkpoint"
        or document.get("consumer") != "next_bounded_global_authoring_round_only"
        or document.get("full_model_timing_status") != "UNMEASURED"
        or document.get("full_model_cycles") is not None
        or document.get("global_speedup_proven") is not False
    ):
        raise ValueError("invalid global authoring checkpoint or unsupported performance claim")
    if context.host_policy != document.get("host_verification_policy"):
        raise ValueError("authoring checkpoint host verification policy changed")
    experiment_record = P2_CONTRACTS.mapping_file(path.parent / "experiment.json")
    if any(
        document.get(field) != experiment_record.get(field)
        for field in (
            "baseline_sha256",
            "target_sha256",
            "optimization_baseline_sha256",
            "optimization_baseline",
            "phase1_qualification",
            "portfolio",
            "portfolio_sha256",
        )
    ):
        raise ValueError("authoring checkpoint experiment binding changed")
    historical = document.get("historical_reference")
    if historical != experiment_record.get("historical_reference"):
        raise ValueError("authoring checkpoint historical reference changed")
    if historical is not None:
        _, summary = load_historical_reference(Path(historical["path"]), historical["sha256"])
        if summary != historical.get("summary"):
            raise ValueError("authoring checkpoint historical reference summary changed")
    authority = document.get("compiler_edit_authority")
    if authority is not None:
        from merlin.perf.compiler_edit_scope import inspect_compiler_edits, validate_edit_contract

        initial = path.parent / "edit_scope_seed"
        if (
            P2_CONTRACTS.mapping_file(path.parent / "compiler_edit_authority.json") != authority
            or Path(authority["seed_path"]).resolve() != initial.resolve()
            or initial.is_symlink()
            or hash_tree(initial)["sha256"] != authority["initial_candidate_sha256"]
            or P2_CONTRACTS.document_sha256(authority["contract"]) != authority["contract_document_sha256"]
        ):
            raise ValueError("authoring checkpoint edit authority changed")
        validate_edit_contract(authority["contract"], initial)
        if (
            inspect_compiler_edits(initial, Path(document["candidate_path"]), authority["contract"])["status"]
            != "allowed"
        ):
            raise ValueError("authoring checkpoint exceeds its host-frozen edit authority")
    verify_checkpoint_catalog(path.parent, document.get("compiler_mechanism_catalog"), authority)
    candidate = Path(document["candidate_path"])
    if (
        candidate.is_symlink()
        or candidate.parent.resolve() != path.parent.resolve()
        or document.get("candidate_read_only") is not True
        or hash_tree(candidate)["sha256"] != document.get("candidate_sha256")
    ):
        raise ValueError("authoring checkpoint candidate bytes changed")
    if SI.compiler_dependency_record(candidate, shared_source_root=context.compiler_shared_source_root) != document.get(
        "compiler_dependencies"
    ):
        raise ValueError("authoring checkpoint compiler dependencies changed")
    iteration_path = Path(document["iteration_record"])
    if (
        iteration_path.is_symlink()
        or iteration_path.parent.resolve() != path.parent.resolve()
        or P2_CONTRACTS.sha256_file(iteration_path) != document.get("iteration_record_sha256")
    ):
        raise ValueError("authoring checkpoint iteration receipt changed")
    iteration = P2_CONTRACTS.mapping_file(iteration_path)
    analysis = iteration.get("analysis") or {}
    readiness = iteration.get("readiness") or {}
    portfolio = iteration.get("portfolio") or {}
    portfolio_identity = document.get("portfolio") or {}
    members, identities = portfolio.get("members"), portfolio_identity.get("members")
    verify_checkpoint_work_order(
        path.parent,
        document.get("compiler_mechanism_work_order"),
        document.get("mechanism_work_order_analysis"),
        document.get("compiler_mechanism_catalog"),
        authority,
        portfolio_identity,
    )
    if (
        iteration.get("candidate_sha256") != document.get("candidate_sha256")
        or iteration.get("compiler_dependencies") != document.get("compiler_dependencies")
        or iteration.get("compiler_mechanism_catalog") != document.get("compiler_mechanism_catalog")
        or iteration.get("compiler_mechanism_work_order") != document.get("compiler_mechanism_work_order")
        or iteration.get("mechanism_work_order_analysis") != document.get("mechanism_work_order_analysis")
        or iteration.get("round_mechanism_attribution") != document.get("round_mechanism_attribution")
        or P2_CONTRACTS.document_sha256(analysis) != document.get("analysis_sha256")
        or readiness != document.get("readiness")
        or readiness.get("status") != "blocked"
        or P2_CONTRACTS.document_sha256(portfolio_identity) != document.get("portfolio_sha256")
        or portfolio.get("portfolio_sha256") != document.get("portfolio_sha256")
        or P2_CONTRACTS.document_sha256(portfolio) != document.get("portfolio_iteration_sha256")
        or portfolio.get("candidate_sha256") != document.get("candidate_sha256")
        or not isinstance(members, list)
        or not isinstance(identities, list)
        or not members
        or len(members) != len(identities)
        or portfolio.get("members_total") != len(members)
        or portfolio.get("members_ready") != document.get("portfolio_members_ready")
        or document.get("portfolio_members_total") != len(members)
        or not 0 <= document.get("portfolio_members_ready", -1) <= len(members)
        or portfolio.get("full_model_simulation_allowed") is not False
    ):
        raise ValueError("authoring checkpoint portfolio binding or blocked readiness changed")
    for index, (identity, member) in enumerate(zip(identities, members, strict=True)):
        if member.get("identity") != identity:
            raise ValueError("authoring checkpoint portfolio identity changed")
        member_analysis = analysis if index == 0 else member.get("analysis") or {}
        if index == 0 and (
            member.get("analysis_ref") != "/analysis" or member.get("static_comparison_ref") != "/static_comparison"
        ):
            raise ValueError("authoring checkpoint primary portfolio aliases changed")
        if (
            member_analysis.get("candidate_sha256") != document.get("candidate_sha256")
            or member_analysis.get("workload", {}).get("capsule_sha256") != identity.get("capsule_sha256")
            or EA.global_iteration_readiness(member_analysis) != member.get("readiness")
        ):
            raise ValueError("authoring checkpoint lacks exact member analysis evidence")
    return document


def consume_round_checkpoint(path: Path, *, context: CheckpointVerificationContext) -> dict[str, Any]:
    schema = P2_CONTRACTS.mapping_file(path).get("schema")
    if schema == "global_perf_candidate_v1":
        return consume_global_candidate(path, context=context)
    if schema == "global_authoring_checkpoint_v1":
        return consume_authoring_checkpoint(path, context=context)
    raise ValueError("unsupported global round checkpoint schema")


def consume_global_candidate(path: Path, *, context: CheckpointVerificationContext) -> dict[str, Any]:
    """Verify the distinct macro handoff without turning it into a measured speedup claim."""
    document = P2_CONTRACTS.mapping_file(path)
    if (
        document.get("schema") != "global_perf_candidate_v1"
        or document.get("full_model_timing_status") != "UNMEASURED"
        or document.get("global_speedup_proven") is not False
    ):
        raise ValueError("invalid global candidate receipt or unsupported full-model timing claim")
    if context.host_policy != document.get("host_verification_policy"):
        raise ValueError("global candidate host verification policy changed")
    historical = document.get("historical_reference")
    if historical is not None:
        _, summary = load_historical_reference(Path(historical["path"]), historical["sha256"])
        if summary != historical.get("summary"):
            raise ValueError("sealed historical reference summary changed")
    authority = document.get("compiler_edit_authority")
    if authority is not None:
        from merlin.perf.compiler_edit_scope import inspect_compiler_edits, validate_edit_contract

        initial = path.parent / "edit_scope_seed"
        if (
            P2_CONTRACTS.mapping_file(path.parent / "compiler_edit_authority.json") != authority
            or Path(authority["seed_path"]).resolve() != initial.resolve()
            or initial.is_symlink()
            or hash_tree(initial)["sha256"] != authority["initial_candidate_sha256"]
            or P2_CONTRACTS.document_sha256(authority["contract"]) != authority["contract_document_sha256"]
        ):
            raise ValueError("sealed candidate edit authority changed")
        validate_edit_contract(authority["contract"], initial)
        if (
            inspect_compiler_edits(initial, Path(document["candidate_path"]), authority["contract"])["status"]
            != "allowed"
        ):
            raise ValueError("sealed compiler exceeds its host-frozen edit authority")
    verify_checkpoint_catalog(path.parent, document.get("compiler_mechanism_catalog"), authority)
    qualification = document.get("phase1_qualification")
    if qualification is not None:
        if qualification.get("submission_sha256") != document.get("baseline_sha256"):
            raise ValueError("sealed Phase-1 compiler identity changed")
        frozen_root = Path(qualification["run_dir"])
        for name, digest in qualification["evidence_sha256"].items():
            source = frozen_root / name
            if (
                source.is_symlink()
                or frozen_root.resolve() not in source.resolve().parents
                or P2_CONTRACTS.sha256_file(source) != digest
            ):
                raise ValueError("frozen Phase-1 qualification evidence changed after sealing")
    comparison = document.get("optimization_baseline")
    if comparison is not None:
        experiment_record = P2_CONTRACTS.mapping_file(path.parent / "experiment.json")
        comparison_path = Path(comparison["path"])
        if (
            comparison != experiment_record.get("optimization_baseline")
            or document.get("baseline_sha256") != experiment_record.get("baseline_sha256")
            or document.get("optimization_baseline_sha256") != comparison.get("sha256")
            or comparison.get("schema") != "global_optimization_baseline_v1"
            or comparison.get("objective_numerical_qualification") != "UNPROVEN"
            or comparison.get("phase1_regraded") is not False
            or comparison_path.is_symlink()
            or hash_tree(comparison_path)["sha256"] != comparison.get("sha256")
            or SI.compiler_dependency_record(comparison_path, shared_source_root=context.compiler_shared_source_root)
            != comparison.get("compiler_dependencies")
        ):
            raise ValueError("sealed optimization comparison baseline identity or scope changed")
        if (
            comparison.get("selection") == "explicit_host_seed"
            and comparison_path.resolve() != (path.parent / "optimization_baseline").resolve()
        ):
            raise ValueError("sealed optimization baseline escaped its immutable experiment snapshot")
    candidate = Path(document["candidate_path"])
    if (
        candidate.is_symlink()
        or candidate.parent.resolve() != path.parent.resolve()
        or hash_tree(candidate)["sha256"] != document.get("candidate_sha256")
    ):
        raise ValueError("sealed global candidate bytes changed")
    if SI.compiler_dependency_record(candidate, shared_source_root=context.compiler_shared_source_root) != document.get(
        "compiler_dependencies"
    ):
        raise ValueError("sealed compiler shared implementation dependencies changed")
    iteration_path = Path(document["iteration_record"])
    if (
        iteration_path.is_symlink()
        or iteration_path.parent.resolve() != path.parent.resolve()
        or P2_CONTRACTS.sha256_file(iteration_path) != document.get("iteration_record_sha256")
    ):
        raise ValueError("global iteration receipt changed or escaped its experiment")
    iteration = P2_CONTRACTS.mapping_file(iteration_path)
    analysis = iteration.get("analysis") or {}
    experiment_record = P2_CONTRACTS.mapping_file(path.parent / "experiment.json")
    portfolio_identity = document.get("portfolio")
    portfolio = iteration.get("portfolio") or {}
    verify_checkpoint_work_order(
        path.parent,
        document.get("compiler_mechanism_work_order"),
        document.get("mechanism_work_order_analysis"),
        document.get("compiler_mechanism_catalog"),
        authority,
        portfolio_identity,
    )
    if (
        iteration.get("compiler_mechanism_catalog") != document.get("compiler_mechanism_catalog")
        or iteration.get("compiler_mechanism_work_order") != document.get("compiler_mechanism_work_order")
        or iteration.get("mechanism_work_order_analysis") != document.get("mechanism_work_order_analysis")
        or iteration.get("round_mechanism_attribution") != document.get("round_mechanism_attribution")
    ):
        raise ValueError("sealed compiler mechanism attribution changed")
    if (
        not isinstance(portfolio_identity, Mapping)
        or P2_CONTRACTS.document_sha256(portfolio_identity) != document.get("portfolio_sha256")
        or experiment_record.get("portfolio") != portfolio_identity
        or experiment_record.get("portfolio_sha256") != document.get("portfolio_sha256")
        or portfolio.get("portfolio_sha256") != document.get("portfolio_sha256")
        or P2_CONTRACTS.document_sha256(portfolio) != document.get("portfolio_iteration_sha256")
        or portfolio.get("candidate_sha256") != document.get("candidate_sha256")
    ):
        raise ValueError("sealed complete-model portfolio identity or iteration changed")
    identities = portfolio_identity.get("members")
    members = portfolio.get("members")
    if (
        not isinstance(identities, list)
        or not isinstance(members, list)
        or len(identities) != len(members)
        or not members
        or portfolio.get("members_total") != len(members)
        or portfolio.get("members_ready") != len(members)
        or portfolio.get("full_model_simulation_allowed") is not False
    ):
        raise ValueError("sealed complete-model portfolio coverage is incomplete")
    for index, (identity, member) in enumerate(zip(identities, members, strict=True)):
        if member.get("identity") != identity:
            raise ValueError("sealed portfolio member identity changed")
        if index == 0:
            if member.get("analysis_ref") != "/analysis" or member.get("static_comparison_ref") != "/static_comparison":
                raise ValueError("sealed primary portfolio aliases changed")
            member_analysis = analysis
        else:
            member_analysis = member.get("analysis") or {}
        if (
            member_analysis.get("candidate_sha256") != document.get("candidate_sha256")
            or member_analysis.get("workload", {}).get("capsule_sha256") != identity.get("capsule_sha256")
            or EA.global_iteration_readiness(member_analysis)["status"] != "ready_for_probe_admission"
        ):
            raise ValueError("sealed portfolio member lacks bound graph/plan/artifact evidence")
    if comparison is not None and (
        iteration.get("optimization_baseline") != comparison
        or analysis.get("optimization_baseline") != comparison
        or iteration.get("optimization_baseline_sha256") != comparison["sha256"]
        or iteration.get("baseline_sha256") != document["baseline_sha256"]
    ):
        raise ValueError("sealed iteration changed its optimization comparison baseline")
    if (
        P2_CONTRACTS.document_sha256(analysis) != document.get("analysis_sha256")
        or analysis.get("candidate_sha256") != document.get("candidate_sha256")
        or EA.global_iteration_readiness(analysis)["status"] != "ready_for_probe_admission"
    ):
        raise ValueError("global candidate lacks current verified graph/plan/artifact evidence")
    expected = {
        "compiler_digest": document["compiler_dependencies"]["compiler_implementation_sha256"],
        "target_digest": document["target_sha256"],
        "graph_digest": analysis["diagnostics"]["captured_logical_graph"]["logical_dispatch_digest"],
        "plan_digest": analysis["diagnostics"]["verified_global_plan_emission"]["plan_digest"],
    }
    for receipt in document.get("context_receipts") or []:
        context_path = Path(receipt["path"])
        if (
            context_path.is_symlink()
            or context_path.parent.resolve() != path.parent.resolve()
            or P2_CONTRACTS.sha256_file(context_path) != receipt.get("sha256")
        ):
            raise ValueError("optional controlled source-prefix receipt changed")
        context = P2_CONTRACTS.mapping_file(context_path)
        if (
            context.get("binding") != expected
            or context.get("scope") != "controlled_source_prefix"
            or context.get("model_artifact_sha256") != analysis["emission"]["candidate_lowered_sha256"]
            or context.get("full_model_cycles") is not None
            or context.get("global_cost_validated") is not False
            or context.get("global_speedup_proven") is not False
            or context.get("calibration_admissible") is not False
        ):
            raise ValueError("controlled source-prefix evidence is stale or overstates its scope")
    expected_decision_feedback = None
    preparation_digests = set()
    for field, schema in (
        ("source_contraction_preparation_receipts", "global_source_contraction_preparation_receipt_v1"),
        ("source_pair_receipts", "global_source_contraction_execution_receipt_v1"),
    ):
        for reference in document.get(field, []):
            receipt_path = Path(reference["path"])
            if (
                receipt_path.is_symlink()
                or receipt_path.parent.resolve() != path.parent.resolve()
                or P2_CONTRACTS.sha256_file(receipt_path) != reference.get("sha256")
            ):
                raise ValueError("source-pair receipt changed or escaped its experiment")
            receipt = P2_CONTRACTS.mapping_file(receipt_path)
            if (
                receipt.get("schema") != schema
                or receipt.get("binding") != expected
                or receipt.get("host_verifier_policy_sha256") != document["host_verification_policy"]["sha256"]
                or receipt.get("full_model_numerics_qualified") is not False
                or receipt.get("global_speedup_proven") is not False
            ):
                raise ValueError("source-pair evidence is stale or overstates its scope")
            if field == "source_contraction_preparation_receipts":
                if receipt.get("numerical_pass") is not False or receipt.get("runtime_admitted") is not False:
                    raise ValueError("source preparation cannot qualify numerical execution")
                preparation_digests.add(reference["sha256"])
            elif receipt.get("preparation_sha256") not in preparation_digests:
                raise ValueError("source execution has no sealed preparation receipt")
    for receipt in document.get("paired_context_receipts") or []:
        pair_path = Path(receipt["path"])
        if (
            pair_path.is_symlink()
            or pair_path.parent.resolve() != path.parent.resolve()
            or P2_CONTRACTS.sha256_file(pair_path) != receipt.get("sha256")
        ):
            raise ValueError("paired controlled-context receipt changed")
        pair = P2_CONTRACTS.mapping_file(pair_path)
        prior_path = Path(pair["previous_iteration_record"])
        if (
            prior_path.is_symlink()
            or prior_path.parent.resolve() != path.parent.resolve()
            or P2_CONTRACTS.sha256_file(prior_path) != pair["previous_iteration_record_sha256"]
        ):
            raise ValueError("paired preceding iteration evidence changed")
        prior = P2_CONTRACTS.mapping_file(prior_path)
        prior_diag = prior["analysis"]["diagnostics"]
        prior_binding = {
            "compiler_digest": prior["compiler_dependencies"]["compiler_implementation_sha256"],
            "target_digest": document["target_sha256"],
            "graph_digest": prior_diag["captured_logical_graph"]["logical_dispatch_digest"],
            "plan_digest": prior_diag["verified_global_plan_emission"]["plan_digest"],
        }
        proof = pair["projection_proof"]
        if (
            pair.get("binding") != expected
            or pair.get("previous_binding") != prior_binding
            or pair.get("model_artifact_sha256") != analysis["emission"]["candidate_lowered_sha256"]
            or pair.get("previous_model_artifact_sha256") != prior["analysis"]["emission"]["candidate_lowered_sha256"]
            or pair.get("scope") != "controlled_fixed_work_slice"
            or pair.get("full_model_cycles") is not None
            or any(
                pair.get(key) is not False
                for key in ("global_cost_validated", "global_speedup_proven", "calibration_admissible")
            )
            or P2_CONTRACTS.document_sha256(proof["work_contract"]) != proof["work_contract_sha256"]
            or P2_CONTRACTS.document_sha256(pair["deterministic_input_contract"])
            != pair["deterministic_input_contract_sha256"]
        ):
            raise ValueError("paired context evidence is stale or overstates its scope")
        if "decision_feedback" in pair:
            feedback = paired_context_decision_feedback(iteration, pair, target_sha256=document["target_sha256"])
            if pair["decision_feedback"] != feedback:
                raise ValueError("paired measured decision feedback differs from its bound raw evidence")
            expected_decision_feedback = {**feedback, "receipt": dict(receipt)}
    if document.get("decision_feedback") != expected_decision_feedback:
        raise ValueError("sealed measured decision context does not match its paired receipt")
    for receipt in document.get("semantic_receipts") or []:
        semantic_path = Path(receipt["path"])
        if (
            semantic_path.is_symlink()
            or semantic_path.parent.resolve() != path.parent.resolve()
            or P2_CONTRACTS.sha256_file(semantic_path) != receipt.get("sha256")
        ):
            raise ValueError("optional changed-region semantic receipt changed")
        semantic = P2_CONTRACTS.mapping_file(semantic_path)
        verify_changed_region_semantic_receipt(
            semantic,
            iteration=iteration,
            portfolio_identity=portfolio_identity,
            target_sha256=document["target_sha256"],
            experiment_root=path.parent,
        )
    for receipt in document.get("probe_receipts") or []:
        probe_path = Path(receipt["path"])
        if (
            probe_path.is_symlink()
            or probe_path.parent.resolve() != path.parent.resolve()
            or P2_CONTRACTS.sha256_file(probe_path) != receipt.get("sha256")
        ):
            raise ValueError("optional mechanism probe receipt changed")
        probe = P2_CONTRACTS.mapping_file(probe_path)
        if (
            probe.get("binding") != expected
            or probe.get("scope") != "mechanism_probe_only"
            or probe.get("full_model_cycles") is not None
            or probe.get("warmup_runs") != 1
            or probe.get("measured_runs") != 1
        ):
            raise ValueError("optional probe evidence is stale or has the wrong measurement scope")
    return document


def recorded_portfolio_contexts(
    row: Mapping[str, Any], *, portfolio_identity: Mapping[str, Any], target_sha256: str, arm: str
) -> list[dict[str, Any]]:
    """Reconstruct every member from a pinned iteration, never from a semantic claim."""
    identities = portfolio_identity.get("members")
    portfolio = row.get("portfolio") or {}
    members = portfolio.get("members")
    candidate_sha256 = row.get("candidate_sha256")
    compiler_sha256 = (row.get("compiler_dependencies") or {}).get("compiler_implementation_sha256")
    if (
        row.get("schema") != "global_perf_iteration_v1"
        or not BE._is_sha256(candidate_sha256)
        or not BE._is_sha256(compiler_sha256)
        or row.get("readiness", {}).get("status") != "ready_for_probe_admission"
        or not isinstance(identities, list)
        or not identities
        or not isinstance(members, list)
        or len(members) != len(identities)
        or portfolio.get("members_total") != len(identities)
        or portfolio.get("members_ready") != len(identities)
        or portfolio.get("candidate_sha256") != candidate_sha256
        or portfolio.get("portfolio_sha256") != P2_CONTRACTS.document_sha256(portfolio_identity)
    ):
        raise ValueError("semantic preceding/current portfolio record is incomplete or substituted")
    contexts = []
    for index, (identity, member) in enumerate(zip(identities, members, strict=True)):
        if member.get("identity") != identity or (
            index == 0
            and (
                member.get("analysis_ref") != "/analysis" or member.get("static_comparison_ref") != "/static_comparison"
            )
        ):
            raise ValueError("semantic portfolio member order or identity changed")
        analysis = RevisionJournal.portfolio_member_analysis(row, index)
        diagnostics = analysis.get("diagnostics") or {}
        graph = diagnostics.get("captured_logical_graph") or {}
        plan = diagnostics.get("verified_global_plan_emission") or {}
        emission = analysis.get("emission") or {}
        if (
            analysis.get("candidate_sha256") != candidate_sha256
            or analysis.get("workload", {}).get("capsule_sha256") != identity.get("capsule_sha256")
            or EA.global_iteration_readiness(analysis).get("status") != "ready_for_probe_admission"
            or plan.get("candidate_sha256") != candidate_sha256
            or plan.get("status") != "verified"
            or plan.get("logical_dispatch_digest") != graph.get("logical_dispatch_digest")
            or any(
                plan.get(field) != emission.get(field)
                for field in ("candidate_lowered_sha256", "candidate_command_buffer_sha256")
            )
        ):
            raise ValueError("semantic portfolio source/plan/artifact record changed")
        binding = {
            "schema": "global_portfolio_member_artifact_binding_v1",
            "arm": arm,
            "portfolio_index": index,
            "capsule": identity.get("capsule"),
            "capsule_sha256": identity.get("capsule_sha256"),
            "source_sha256": plan.get("source_sha256"),
            "candidate_sha256": candidate_sha256,
            "compiler_implementation_sha256": compiler_sha256,
            "target_sha256": target_sha256,
            "logical_dispatch_digest": graph.get("logical_dispatch_digest"),
            "plan_digest": plan.get("plan_digest"),
            "lowered_sha256": emission.get("candidate_lowered_sha256"),
            "command_buffer_sha256": emission.get("candidate_command_buffer_sha256"),
        }
        if any(
            not BE._is_sha256(binding[key])
            for key in (
                "capsule_sha256",
                "source_sha256",
                "target_sha256",
                "logical_dispatch_digest",
                "plan_digest",
                "lowered_sha256",
                "command_buffer_sha256",
            )
        ):
            raise ValueError("semantic portfolio artifact identity is missing")
        contexts.append({"analysis": analysis, "member_binding": binding})
    return contexts


def verify_changed_region_semantic_receipt(
    semantic: Mapping[str, Any],
    *,
    iteration: Mapping[str, Any],
    portfolio_identity: Mapping[str, Any],
    target_sha256: str,
    experiment_root: Path,
) -> dict[str, Any]:
    """Verify both ordered portfolios and independently rerun the member-selection policy.

    Legacy v1 lacks a pinned previous portfolio. It cannot establish secondary-member or
    selection evidence and is deliberately refused by this verifier, including supplements.
    """
    if semantic.get("schema") != "global_changed_region_semantic_receipt_v2":
        raise ValueError("legacy semantic receipt lacks pinned prior portfolio; requalify under v2")
    number = iteration.get("iteration")
    previous_value = semantic.get("previous_iteration_record")
    if type(number) is not int or number < 1 or not isinstance(previous_value, str):
        raise ValueError("semantic receipt has no valid preceding iteration record")
    previous_path = Path(previous_value)
    expected_path = experiment_root / f"iteration_{number - 1:04d}.json"
    if (
        not previous_path.is_absolute()
        or previous_path.is_symlink()
        or not previous_path.is_file()
        or previous_path.stat().st_mode & 0o222
        or previous_path != expected_path.absolute()
        or previous_path.resolve() != expected_path.absolute()
        or P2_CONTRACTS.sha256_file(previous_path) != semantic.get("previous_iteration_record_sha256")
    ):
        raise ValueError("semantic preceding iteration record is absent, changed, linked, or cross-run")
    previous = P2_CONTRACTS.mapping_file(previous_path)
    if (
        previous.get("iteration") != number - 1
        or semantic.get("iteration") != number
        or (iteration.get("static_comparison") or {}).get("previous_iteration") != number - 1
        or P2_CONTRACTS.document_sha256(previous.get("portfolio"))
        != semantic.get("previous_portfolio_iteration_sha256")
    ):
        raise ValueError("semantic preceding portfolio digest or iteration changed")
    prior_snapshot_value = previous.get("submitted_snapshot")
    if not isinstance(prior_snapshot_value, str):
        raise ValueError("semantic preceding compiler snapshot is absent")
    prior_snapshot = Path(prior_snapshot_value)
    if (
        not prior_snapshot.is_absolute()
        or prior_snapshot.is_symlink()
        or not prior_snapshot.is_dir()
        or prior_snapshot.stat().st_mode & 0o222
        or prior_snapshot.parent != experiment_root.absolute()
        or prior_snapshot.resolve() != prior_snapshot
        or any(item.is_symlink() for item in prior_snapshot.rglob("*"))
        or hash_tree(prior_snapshot)["sha256"] != previous.get("candidate_sha256")
    ):
        raise ValueError("semantic preceding compiler snapshot is changed or cross-run")
    policies = []
    for record in (previous, iteration):
        policy = copy.deepcopy(record.get("analysis_reuse_binding") or {})
        digest = policy.pop("sha256", None)
        schema = policy.pop("schema", None)
        if (
            schema != "global_static_analysis_reuse_binding_v1"
            or P2_CONTRACTS.document_sha256(policy) != digest
            or policy.get("candidate_sha256") != record.get("candidate_sha256")
            or policy.get("compiler_dependencies") != record.get("compiler_dependencies")
            or policy.get("target_sha256") != target_sha256
            or policy.get("portfolio_sha256") != P2_CONTRACTS.document_sha256(portfolio_identity)
        ):
            raise ValueError("semantic preceding/current analysis policy binding changed")
        policy.pop("candidate_sha256")
        policy.pop("compiler_dependencies")
        policies.append(policy)
    if policies[0] != policies[1]:
        raise ValueError("semantic preceding/current portfolios have different experiment policies")
    before = recorded_portfolio_contexts(
        previous, portfolio_identity=portfolio_identity, target_sha256=target_sha256, arm="previous"
    )
    after = recorded_portfolio_contexts(
        iteration, portfolio_identity=portfolio_identity, target_sha256=target_sha256, arm="current"
    )
    selection = select_changed_portfolio_contexts(list(zip(before, after, strict=True)))
    index = selection["portfolio_index"]
    binding = {
        "selection": selection,
        "previous": before[index]["member_binding"],
        "current": after[index]["member_binding"],
    }
    current = binding["current"]
    probe = {
        "compiler_digest": current["compiler_implementation_sha256"],
        "target_digest": target_sha256,
        "graph_digest": current["logical_dispatch_digest"],
        "plan_digest": current["plan_digest"],
    }
    if (
        semantic.get("portfolio_member_binding") != binding
        or semantic.get("binding") != probe
        or semantic.get("evidence", {}).get("portfolio_member_binding") != binding
        or semantic.get("previous_artifact_sha256") != binding["previous"]["lowered_sha256"]
        or semantic.get("current_artifact_sha256") != current["lowered_sha256"]
        or semantic.get("scope") != "selected changed mechanism and tested reduced domain only"
        or semantic.get("full_model_numerics_qualified") is not False
        or semantic.get("global_speedup_proven") is not False
        or semantic.get("full_model_cycles") is not None
    ):
        raise ValueError("semantic portfolio selection or evidence is stale or substituted")
    return binding
