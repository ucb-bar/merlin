#!/usr/bin/env python3
"""Compile real model candidates and consume global-plan evidence without full-model simulation.

The macro search has its own receipt schema. It cannot produce a microbenchmark promotion record
or stop because a microbenchmark plateaued. Static accounting is the objective; optional probe
observations calibrate specific mechanisms and never become measured model latency.
"""
from __future__ import annotations

import argparse
import ast
import copy
import concurrent.futures
import hashlib
import json
import math
import shutil
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import perf_agent_stage as PAS
from merlin.benchharness import hash_tree
from merlin.perf.execution_policy import ITERATION_MAX_SECONDS
from merlin.perf.mechanism_probe import ProbeBinding, ProbeObservation, require_probe_admission


_GIB = 1024 ** 3
_PORTFOLIO_WORKER_HEADROOM_BYTES = 16 * _GIB


@dataclass(frozen=True)
class FrozenPhase1:
    """Explicit launch inputs for reusing an existing qualification, never rerunning it."""

    runs_root: Path
    run_id: str
    submission_sha256: str
    waiver_predicates: tuple[str, ...]
    expected_public_passed: int
    expected_public_total: int
    expected_gap_ids: tuple[str, ...]

    def verify(self, baseline: Path) -> dict[str, Any]:
        frozen = PAS.PC.inspect_functional_run(
            self.runs_root, self.run_id, self.submission_sha256, waive=self.waiver_predicates)
        if hash_tree(baseline)["sha256"] != frozen.digest:
            raise ValueError("macro baseline is not the compiler from the frozen qualification")
        score = frozen.public_score
        gaps = sorted(str(row.get("capsule")) for row in score["per_capsule"]
                      if row.get("status") != "pass")
        if ((score.get("n_passed"), score.get("n_capsules"))
                != (self.expected_public_passed, self.expected_public_total)
                or gaps != sorted(self.expected_gap_ids)
                or len(gaps) != self.expected_public_total - self.expected_public_passed):
            raise ValueError("frozen qualification counts or exact known functional gap IDs changed")
        files = ("environment.yaml", "qa_loop_summary.yaml", "freeze.json", "run_manifest.yaml",
                 "grading_public/score_capsule.json", "grading_hidden/score_capsule.json")
        return {
            "schema": "global_frozen_phase1_binding_v1", "run_id": frozen.run_id,
            "run_dir": str(frozen.run_dir), "submission_sha256": frozen.digest,
            "public_passed": score["n_passed"], "public_total": score["n_capsules"],
            "known_functional_gap_ids": gaps,
            "waiver_predicates": list(self.waiver_predicates),
            "observed_deviations": [row.to_dict() for row in frozen.deviations],
            "evidence_sha256": {name: PAS._sha256_file(frozen.run_dir / name) for name in files},
            "qualification_action": "read_existing_frozen_receipts_only",
        }


def sentinel_identity(sentinel: PAS.StageE2ESentinel, *, role: str) -> dict[str, Any]:
    """Stable public identity for one complete-model member of a global portfolio."""
    if role not in ("primary", "training"):
        raise ValueError("portfolio member role must be primary or training")
    return {
        "capsule": sentinel.capsule,
        "capsule_sha256": sentinel.capsule_sha256,
        "required_lanes": list(sentinel.required_lanes),
        "required_tiers": list(sentinel.required_tiers),
        "role": role,
        "analysis": "full_graph_compile_and_static_only",
        "full_model_simulation_allowed": False,
    }


def portfolio_member_analysis_allocation(
        remaining_seconds: float,
        remaining_sentinels: Sequence[PAS.StageE2ESentinel], *,
        emission_seconds_by_capsule_sha256: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Allocate one member's bounded analysis time from generic frozen-source complexity.

    Every remaining graph receives an equal chance floor capped at 60 seconds. Remaining time is
    weighted by exact prior baseline-emission measurements when available. Missing measurements are
    projected from frozen interface bytes and the median observed seconds/byte; an entirely cold
    cache uses interface bytes directly. Recomputing after actual elapsed time rolls surplus forward
    while preserving declared portfolio order and the outer iteration deadline.
    """
    if not remaining_sentinels:
        raise ValueError("portfolio allocation requires at least one remaining member")
    remaining_seconds = max(0.0, remaining_seconds)

    interface_sizes: list[int] = []
    for sentinel in remaining_sentinels:
        source = Path(sentinel.frozen_source_path)
        descriptor = PAS._mapping_file(source / "capsule.yaml", yaml_file=True)
        interface = source / str(descriptor.get("interface_mlir") or "capsule.interface.mlir")
        if interface.is_symlink() or not interface.is_file():
            raise ValueError(f"frozen portfolio member has no real interface MLIR: {sentinel.capsule}")
        interface_sizes.append(max(1, interface.stat().st_size))

    members = len(remaining_sentinels)
    measurements = dict(emission_seconds_by_capsule_sha256 or {})
    for digest, seconds in measurements.items():
        if (not PAS._is_sha256(digest) or isinstance(seconds, bool)
                or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or seconds < 0):
            raise ValueError("portfolio emission-cost measurement is malformed")
    known_rates = sorted(
        measurements[sentinel.capsule_sha256] / size
        for sentinel, size in zip(remaining_sentinels, interface_sizes, strict=True)
        if sentinel.capsule_sha256 in measurements)
    median_rate = (None if not known_rates else
                   known_rates[len(known_rates) // 2] if len(known_rates) % 2 else
                   0.5 * (known_rates[len(known_rates) // 2 - 1]
                          + known_rates[len(known_rates) // 2]))
    estimates = [
        measurements.get(sentinel.capsule_sha256,
                         size * median_rate if median_rate is not None else float(size))
        for sentinel, size in zip(remaining_sentinels, interface_sizes, strict=True)]
    weights = [max(float(value), 1e-9) for value in estimates]
    chance_floor = (remaining_seconds if members == 1 else
                    min(60.0, remaining_seconds / (2 * members)))
    weighted_budget = max(0.0, remaining_seconds - chance_floor * members)
    allocated = (remaining_seconds if members == 1 else
                 chance_floor + weighted_budget * weights[0] / sum(weights))
    first_digest = remaining_sentinels[0].capsule_sha256
    return {
        "schema": "portfolio_analysis_allocation_v1",
        "policy": "bounded_equal_chance_floor_plus_measured_emission_cost_with_rolling_surplus",
        "allocated_seconds": allocated,
        "remaining_seconds": remaining_seconds,
        "remaining_members": members,
        "interface_bytes": interface_sizes[0],
        "remaining_interface_bytes": sum(interface_sizes),
        "chance_floor_seconds": chance_floor,
        "estimated_emission_seconds": estimates[0],
        "emission_cost_basis": ("exact_cached_baseline_emission"
                                if first_digest in measurements else
                                "interface_bytes_scaled_by_measured_median"
                                if median_rate is not None else "frozen_interface_bytes_proxy"),
        "known_emission_measurements": len(known_rates),
    }


def portfolio_analysis_concurrency(*, requested_workers: int, members: int,
                                   memory_available_bytes: int,
                                   minimum_memory_available_bytes: int) -> dict[str, Any]:
    """Admit bounded member parallelism from current host headroom above the launch guard."""
    if min(requested_workers, members) < 1 or min(
            memory_available_bytes, minimum_memory_available_bytes) < 0:
        raise ValueError("portfolio concurrency inputs are invalid")
    headroom = memory_available_bytes - minimum_memory_available_bytes
    if headroom < 0:
        raise TimeoutError("host memory is below the portfolio analysis admission floor")
    memory_workers = max(1, headroom // _PORTFOLIO_WORKER_HEADROOM_BYTES)
    admitted = min(requested_workers, members, memory_workers)
    return {
        "schema": "portfolio_analysis_concurrency_v1",
        "requested_workers": requested_workers,
        "admitted_workers": admitted,
        "members": members,
        "memory_available_bytes": memory_available_bytes,
        "minimum_memory_available_bytes": minimum_memory_available_bytes,
        "per_worker_headroom_bytes": _PORTFOLIO_WORKER_HEADROOM_BYTES,
        "policy": "host_memory_headroom_bounded_concurrent_member_analysis",
    }


def portfolio_concurrent_schedule(cost_seconds: Sequence[float], workers: int) -> dict[str, Any]:
    """Longest-estimated members first; retain declared order as the tie breaker and output order."""
    if workers < 1 or not cost_seconds:
        raise ValueError("portfolio concurrent schedule requires workers and member costs")
    costs = [float(value) for value in cost_seconds]
    if any(not math.isfinite(value) or value < 0 for value in costs):
        raise ValueError("portfolio concurrent schedule cost is malformed")
    admitted = min(workers, len(costs))
    order = sorted(range(len(costs)), key=lambda index: (-costs[index], index))
    loads = [0.0] * admitted
    assignments = []
    for index in order:
        worker = min(range(admitted), key=lambda item: (loads[item], item))
        assignments.append({"member_index": index, "worker": worker,
                            "estimated_seconds": costs[index]})
        loads[worker] += costs[index]
    return {"schema": "portfolio_concurrent_schedule_v1", "submission_order": order,
            "workers": admitted, "worker_estimated_seconds": loads,
            "projected_wall_seconds": max(loads), "policy": "longest_estimated_member_first"}


def _host_memory_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise ValueError("host MemAvailable is unavailable")


def compiler_dependency_content_sha256(record: Mapping[str, Any]) -> str:
    """Portable identity for compiler-affecting dependency bytes, excluding snapshot paths."""
    required = ("candidate_sha256", "shared_sources", "selected_lazy_exports")
    if any(name not in record for name in required) or not PAS._is_sha256(record["candidate_sha256"]):
        raise ValueError("compiler dependency record is incomplete")
    return PAS._document_sha256({name: record[name] for name in required})


def seed_baseline_emission_cache_from_run(*, cache_binding: Mapping[str, Any],
                                          seed_run: Path, baseline: Path,
                                          sentinels: Sequence[PAS.StageE2ESentinel],
                                          target: str,
                                          compiler_api_schema: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Import only exact successful baseline-emission bytes from a prior host-owned run."""
    from merlin.targetgen import oot_runner as OR

    seed_run = Path(seed_run).resolve()
    if seed_run.is_symlink() or not seed_run.is_dir():
        raise ValueError("baseline emission seed run is absent or linked")
    experiment = PAS._mapping_file(seed_run / "global_iterations/experiment.json")
    dependencies = (experiment.get("optimization_baseline") or {}).get("compiler_dependencies")
    baseline_sha256 = hash_tree(baseline)["sha256"]
    if (experiment.get("target") != target
            or experiment.get("optimization_baseline_sha256") != baseline_sha256
            or not isinstance(dependencies, Mapping)
            or compiler_dependency_content_sha256(dependencies)
            != cache_binding.get("compiler_dependencies_sha256")):
        raise ValueError("baseline emission seed run compiler, target or dependencies differ")
    by_capsule = {sentinel.capsule_sha256: sentinel for sentinel in sentinels}
    if len(by_capsule) != len(sentinels):
        raise ValueError("baseline emission seed portfolio identities are not unique")
    package = OR.load_package(baseline)
    entrypoints = OR.analysis_emission_entrypoints(package)
    imported: dict[str, dict[str, Any]] = {}
    workers = seed_run / "host_analysis_workers"
    for worker in sorted(workers.iterdir()) if workers.is_dir() else ():
        required = [worker / name for name in (
            "request.json", "baseline_emission.json", "baseline_lowered.mlir")]
        command_buffer_path = worker / "compiler_scratch/baseline/command_buffer.json"
        if any(path.is_symlink() or not path.is_file() for path in (*required, command_buffer_path)):
            continue
        request = PAS._mapping_file(required[0])
        sentinel_record = request.get("sentinel") or {}
        capsule_sha256 = sentinel_record.get("capsule_sha256")
        sentinel = by_capsule.get(capsule_sha256)
        if sentinel is None or capsule_sha256 in imported:
            continue
        baseline_value = request.get("baseline")
        worker_baseline = Path(baseline_value) if isinstance(baseline_value, str) else None
        if (request.get("kwargs", {}).get("target") != target
                or worker_baseline is None or not worker_baseline.is_absolute()
                or worker_baseline.is_symlink() or not worker_baseline.is_dir()
                or hash_tree(worker_baseline)["sha256"] != baseline_sha256):
            raise ValueError("baseline emission seed worker changed compiler or target")
        source_root = Path(sentinel.frozen_source_path)
        descriptor = PAS._mapping_file(source_root / "capsule.yaml", yaml_file=True)
        interface = source_root / str(descriptor.get("interface_mlir") or "capsule.interface.mlir")
        copied_interface = worker / "compiler_scratch/baseline/interface.mlir"
        if (copied_interface.is_symlink() or not copied_interface.is_file()
                or copied_interface.read_bytes() != interface.read_bytes()):
            raise ValueError("baseline emission seed worker source bytes differ")
        emission = PAS._mapping_file(required[1])
        rows = emission.get("entrypoints")
        if (emission.get("schema") != "compiler_emission_diagnostics_v1"
                or not isinstance(rows, list)
                or [row.get("command") for row in rows] != list(entrypoints)
                or any(row.get("returncode") != 0 for row in rows)):
            raise ValueError("baseline emission seed worker did not complete exact entrypoints")
        lowered_text = required[2].read_text(encoding="utf-8")
        command_buffer_text = command_buffer_path.read_text(encoding="utf-8")
        command_buffer = json.loads(command_buffer_text)
        if not isinstance(command_buffer, Mapping) or command_buffer.get("declined") is not None:
            raise ValueError("baseline emission seed worker produced no admitted command buffer")
        PAS.validate_whole_program_schema(command_buffer, compiler_api_schema, arm="seed")
        identity = PAS.baseline_emission_cache_identity(
            baseline_sha256=baseline_sha256, capsule_sha256=capsule_sha256,
            source_sha256=PAS._sha256(interface.read_bytes()), target=target,
            compiler_dependencies_sha256=str(cache_binding["compiler_dependencies_sha256"]),
            compiler_api_schema=compiler_api_schema, entrypoints=entrypoints)
        elapsed = required[2].stat().st_mtime - required[0].stat().st_mtime
        worker_receipt = (PAS._mapping_file(worker / "receipt.json")
                          if (worker / "receipt.json").is_file() else {})
        analysis_status = worker_receipt.get("status")
        analysis_wall = (worker_receipt.get("wall_seconds")
                         if analysis_status in ("completed", "timeout") else None)
        imported[capsule_sha256] = PAS.store_baseline_emission_cache(
            cache_binding, identity, lowered_text=lowered_text,
            command_buffer_text=command_buffer_text, emission_wall_seconds=max(0.0, elapsed),
            observed_analysis_wall_seconds=(
                float(analysis_wall) if isinstance(analysis_wall, (int, float))
                and not isinstance(analysis_wall, bool) and math.isfinite(analysis_wall)
                and analysis_wall >= 0 else None),
            observed_analysis_status=(analysis_status if isinstance(analysis_wall, (int, float))
                                      and not isinstance(analysis_wall, bool)
                                      and math.isfinite(analysis_wall)
                                      and analysis_wall >= 0 else None))
    missing = sorted(set(by_capsule) - imported.keys())
    if missing:
        raise ValueError(f"baseline emission seed run lacks portfolio members: {missing}")
    return [imported[sentinel.capsule_sha256] for sentinel in sentinels]


def host_verification_policy_record() -> dict[str, Any]:
    """Bind the interpretation of structural evidence to current host implementation bytes."""
    root = PAS.repo_root() / "merlin" / "python" / "merlin"
    sources = [Path(PAS.__file__), Path(__file__), Path(PAS.TC.__file__),
               Path(PAS.whole_program_schema_record()["path"]),
               root / "targetgen/sandbox/answer_surfaces.py", root / "targetgen/sandbox/bwrap.py",
               root / "targetgen/sandbox/build_dependencies.py",
               root / "targetgen/sandbox/executable_dependencies.py",
               root / "targetgen/contract/build_service.py", root / "targetgen/contract/build_recipe.py",
               root / "perf/compiler_plan_evidence.py", root / "perf/task_cfg_evidence.py",
               root / "perf/task_instruction_evidence.py",
               root / "perf/task_route_presence.py",
               root / "perf/storage_encoding.py", root / "perf/structural_transitions.py",
               root / "perf/physical_transition_evidence.py",
               root / "perf/external_objective.py",
               root / "perf/model_placement.py", root / "perf/model_macs.py",
               root / "runtime/storage_binding.py",
               root / "runtime/prepack_authority.py", root / "runtime/captured_constants.py",
               root / "frontends/argument_identity.py",
               root / "perf/host_cfg_activity.py",
               root / "perf/analysis_worker.py",
               root / "perf/isolated_probe_provider.py", root / "perf/primitive_probe.py",
               root / "perf/instruction_motif.py",
               root / "perf/probe_relevance.py", root / "perf/structural_delta.py",
               root / "perf/completion_delta.py",
               root / "perf/context_probe.py",
               root / "perf/context_program.py", root / "perf/controlled_context_provider.py",
               root / "perf/fixed_work_context.py", root / "perf/paired_context_provider.py",
               root / "perf/static_imports.py",
               root / "perf/compiler_edit_scope.py",
               root / "perf/agent_guidance.py", root / "kernels/cca_contract.py",
               root / "perf/host_region_qualifier.py", root / "perf/host_source_witness.py",
               root / "perf/source_convolution_witness.py", root / "perf/source_convolution_preparation.py",
               root / "perf/source_program_pair.py",
               root / "perf/source_contraction_witness.py", root / "perf/source_contraction_preparation.py",
               root / "perf/source_program_pair_provider.py",
               root / "perf/source_initializer_elision.py",
               root / "targetgen/conv_geometry.py", root / "targetgen/capsule_golden.py",
               root / "runtime/commandbuffer.py", root / "runtime/tensor.py",
               root / "perf/host_physical_transition_qualifier.py",
               root / "perf/native_host_witness_runner.py",
               root / "perf/mechanism_probe.py",
               root / "perf/historical_reference.py", root / "perf/harvest.py",
               root / "perf/work_volume.py",
               root / "perf/execution_policy.py", root / "targetgen/rocc/decode.py"]
    hashes = {str(path.resolve()): PAS._sha256_file(path) for path in sources}
    return {"schema": "global_host_verification_policy_v1", "sources": hashes,
            "sha256": PAS._document_sha256(hashes)}


def controlled_context_capability(analysis: Mapping[str, Any], *, provider_installed: bool | None
                                 ) -> dict[str, Any]:
    """Distinguish adapter installation from a supported motif in this exact emitted revision."""
    diag = analysis.get("diagnostics") or {}
    context = diag.get("queued_movement_context") or {}
    plan = diag.get("verified_global_plan_emission") or {}
    artifact = (analysis.get("emission") or {}).get("candidate_lowered_sha256")
    bound = (bool(artifact) and plan.get("status") == "verified"
             and plan.get("candidate_lowered_sha256") == artifact
             and context.get("schema") == "queued_movement_context_candidates_v1"
             and context.get("artifact_sha256") == artifact)
    motifs = context.get("motifs") if bound else None
    count = len(motifs) if isinstance(motifs, list) else None
    return {"provider_installed": provider_installed, "artifact_sha256": artifact,
            "current_supported_motif_count": count,
            "current_candidate_status": ("UNKNOWN" if count is None else
                "no_extracted_supported_motifs" if count == 0 else "motifs_pending_admission"),
            "available": bool(provider_installed and count),
            "admission_verified": False,
            "licence": "installation is not applicability; absent motifs do not prove absence of accelerator work"}


def paired_context_decision_feedback(record: Mapping[str, Any], receipt: Mapping[str, Any], *, target_sha256: str
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
    if (receipt["binding"] != expected or diag["verified_global_plan_emission"].get("status") != "verified"
            or receipt.get("scope") != "controlled_fixed_work_slice"
            or receipt.get("model_artifact_sha256") != analysis["emission"]["candidate_lowered_sha256"]
            or receipt.get("global_cost_validated") is not False
            or receipt.get("global_speedup_proven") is not False):
        raise ValueError("paired decision feedback is not bound to the current full-model revision")
    proof = receipt["projection_proof"]
    if (proof.get("status") != "same_work_projection_verified"
            or proof.get("after_artifact_sha256") != receipt["model_artifact_sha256"]
            or PAS._document_sha256(proof["work_contract"]) != proof["work_contract_sha256"]):
        raise ValueError("paired decision feedback has no same-work contract")
    executions = receipt["executions"]
    profiles = {arm: executions[arm].get("counter_profile") or {} for arm in ("before", "after")}
    missing = []
    engine_hashes = [executions[arm].get("engine_provenance", {}).get("binary_sha256")
                     for arm in ("before", "after")]
    if not PAS._is_sha256(engine_hashes[0]) or engine_hashes[0] != engine_hashes[1]:
        missing.append("same_execution_engine_unproved")
    for arm in ("before", "after"):
        profile, execution = profiles[arm], executions[arm]
        counters = [profile.get(key) for key in
                    ("active_union_cycles", "idle_cycles", "overlap_any_engine_cycles")]
        busy = profile.get("busy_cycles_by_engine_token")
        total = execution.get("total_compute_cycles")
        if (execution.get("correct") is not True or execution.get("warmup_runs") != 1
                or execution.get("measured_runs") != 1 or type(total) is not int or total <= 0
                or profile.get("kind") != "joint_engine_busy_cycles"
                or profile.get("partition_proof", {}).get("status") != "proved"
                or profile.get("layout", {}).get("complete") is not True
                or not isinstance(busy, Mapping) or not busy
                or any(type(value) is not int or value < 0 for value in counters)
                or any(type(value) is not int or not 0 <= value <= total for value in busy.values())
                or counters[0] + counters[1] != total or counters[2] > counters[0]):
            missing.append(arm + "_bounded_joint_counter_evidence_missing")
    if not missing:
        if (profiles["before"]["partition_proof"] != profiles["after"]["partition_proof"]
                or profiles["before"]["layout"] != profiles["after"]["layout"]
                or profiles["before"]["busy_cycles_by_engine_token"].keys()
                    != profiles["after"]["busy_cycles_by_engine_token"].keys()):
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
            "busy_delta_by_resource": {key: profiles["after"]["busy_cycles_by_engine_token"][key]-value
                                       for key, value in profiles["before"]["busy_cycles_by_engine_token"].items()},
        }
        observation["cycle_delta"] = observation["after_cycles"]-observation["before_cycles"]
        if observation["overlap_delta_cycles"] == 0 and not any(observation["busy_delta_by_resource"].values()):
            status = "no_observed_overlap_or_busy_work_change"
            next_step = ("Do not count this schedule as a demonstrated latency-hiding gain. Use the bound "
                         "region's movement, dependency and buffer evidence to choose a different transformation; "
                         "a cycle-only difference is not statistical confirmation or model speedup.")
        else:
            status = "controlled_resource_change_observed"
            next_step = ("Inspect the measured resource tradeoff for this exact region; verify actual buffer "
                         "capacity and dependency/repetition contracts before any global cost projection.")
    inventory = (analysis.get("optimization_brief") or {}).get("package_inventory") or {}
    surfaces = [{key: surface[key] for key in ("id", "path", "symbol") if key in surface}
                for surface in inventory.get("surfaces", [])
                if set(surface.get("effects", ())) & {"issue", "latency_hiding"}]
    work = proof["work_contract"]
    return {"schema": "global_paired_decision_feedback_v1", "status": status,
            "binding": dict(receipt["binding"]), "candidate_sha256": record["candidate_sha256"],
            "model_artifact_sha256": receipt["model_artifact_sha256"],
            "source_task_index": work.get("source_task_index"),
            "source_op_indices": work.get("source_op_indices", []),
            "timed_command_count": work.get("timed_command_count"),
            "same_declared_movement_work": bool(work.get("timed_command_multiset")), "physical_movement_bytes": None,
            "observation": observation, "missing": missing, "next_step": next_step,
            "edit_surfaces": surfaces, "scope": "controlled_fixed_work_slice",
            "full_model_cycles": None, "full_model_cost_selection": "UNKNOWN",
            "global_speedup_proven": False, "buffer_capacity_contract": "UNPROVED",
            "pipeline_projection_admitted": False}


def storage_encoding_agent_summary(record: Mapping[str, Any], *, complete_evidence: str, arm: str = "candidate"
                                   ) -> dict[str, Any] | None:
    """Expose unresolved storage obligations without copying every tensor contract into a prompt."""
    if arm not in {"candidate", "baseline"}:
        raise ValueError("storage summary arm must be candidate or baseline")
    plan_key = "verified_global_plan_emission" if arm == "candidate" else "verified_baseline_global_plan_emission"
    analysis = record.get("analysis") or {}
    diagnostics = analysis.get("diagnostics") or {}
    plan = diagnostics.get(plan_key) or {}
    if not isinstance(plan, Mapping) or "storage_encodings" not in plan:
        return None
    contracts = plan["storage_encodings"]
    emission = analysis.get("emission") or {}
    graph = diagnostics.get("captured_logical_graph") or {}
    compiler_sha = (analysis.get("candidate_sha256") if arm == "candidate"
                    else record.get("optimization_baseline_sha256"))
    expected = {"candidate_sha256": compiler_sha,
        "logical_dispatch_digest": graph.get("logical_dispatch_digest"),
        "candidate_lowered_sha256": emission.get(arm + "_lowered_sha256"),
        "candidate_command_buffer_sha256": emission.get(arm + "_command_buffer_sha256")}
    bound = (plan.get("status") == "verified" and graph.get("status") == "verified"
        and (record.get("candidate_sha256") if arm == "candidate" else
             (record.get("optimization_baseline") or {}).get("sha256")) == compiler_sha
        and PAS._is_sha256(plan.get("plan_digest"))
        and all(PAS._is_sha256(value) and plan.get(key) == value for key, value in expected.items()))
    count = len(contracts) if isinstance(contracts, Mapping) else None
    summary: dict[str, Any] = {"schema": "agent_storage_encoding_summary_v1", "arm": arm,
        "status": "source_bound_contracts_verified" if bound and count else
                  "no_checked_storage_contracts" if bound and count == 0 else "UNKNOWN",
        "reported_contract_count": count,
        "source_plan_binding_verified": bound,
        "binding": {**expected, "plan_digest": plan.get("plan_digest")},
        "details": {"path": complete_evidence, "canonical_record_sha256": PAS._document_sha256(record),
            "json_pointer": "/analysis/diagnostics/" + plan_key + "/storage_encodings",
            "encoding_map_sha256": PAS._document_sha256(contracts)},
        "scope": "declared storage map only; caller materialization and actual consumer addresses are separate proofs",
        "full_model_numerics_qualified": False, "global_cost_validated": False}
    transitions = plan.get("physical_transition_evidence")
    if isinstance(transitions, Mapping):
        summary["physical_transition_evidence"] = {
            "status": transitions.get("status", "UNKNOWN") if bound else "UNKNOWN",
            "source_plan_binding_verified": bound,
            "details": {"path": complete_evidence,
                "json_pointer": "/analysis/diagnostics/" + plan_key + "/physical_transition_evidence",
                "evidence_sha256": PAS._document_sha256(transitions)},
            "scope": "explicit emitted copy/address proof only; not arbitrary consumer numerics or timing"}
    for key in ("caller_materialization", "emitted_consumer_addressing"):
        pending = 0
        histogram: dict[str, int] = {}
        for row in contracts.values() if isinstance(contracts, Mapping) else ():
            value = row.get(key) if isinstance(row, Mapping) else None
            label = value[:200] if isinstance(value, str) else "UNKNOWN"
            histogram[label] = histogram.get(label, 0) + 1
            pending += isinstance(value, str) and value.startswith("requires ")
        summary[key] = {"status": "UNRESOLVED" if bound and pending else "UNKNOWN",
            "reported_pending_count": pending if count is not None else None,
            "other_or_unknown_count": count - pending if count is not None else None,
            "reported_status_counts": dict(sorted(histogram.items())[:3]),
            "verified": False}
    summary["obligation_examples"] = [{"tensor": name,
        "caller_materialization": row.get("caller_materialization", "UNKNOWN"),
        "emitted_consumer_addressing": row.get("emitted_consumer_addressing", "UNKNOWN")}
        for name, row in (list(contracts.items())[:3] if isinstance(contracts, Mapping) else ())
        if isinstance(row, Mapping)]
    return summary


def load_historical_reference(path: Path, sha256: str, *, target: str | None = None,
                              candidate_roots: Sequence[Path] = ()) -> tuple[bytes, dict[str, Any]]:
    """Read one explicit host bundle, never discover receipts or mint timing authority."""
    from merlin.perf.historical_reference import reference_summary

    if (not PAS._is_sha256(sha256) or path.is_symlink() or not path.is_file()
            or any(parent.is_symlink() for parent in path.parents)
            or any(path.resolve().is_relative_to(root.resolve()) for root in candidate_roots)):
        raise ValueError("historical reference requires a pinned regular host file outside candidate writes")
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("historical reference exceeds the host bundle byte limit")
    raw = path.read_bytes()
    if PAS._sha256(raw) != sha256:
        raise ValueError("historical reference bundle digest changed")
    def unique_fields(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate historical reference bundle field")
            result[key] = value
        return result
    bundle = json.loads(raw, object_pairs_hook=unique_fields)
    if (not isinstance(bundle, dict) or bundle.get("schema") != "historical_reference_bundle_v1"
            or not isinstance(bundle.get("records"), list)):
        raise ValueError("invalid historical reference bundle schema")
    summary = reference_summary(bundle["records"])
    if bundle.get("summary") != summary:
        raise ValueError("historical reference recorded summary differs from bound records")
    if target is not None and any(row.get("target") != target for row in bundle["records"]):
        raise ValueError("historical reference target differs from the experiment")
    compact = {key: summary[key] for key in ("reference_count", "target_cycle_authority",
        "warm_calibration", "physical_roofline_complete", "full_model_cycle_ordering", "next_action")}
    compact.update({"schema": "historical_reference_agent_brief_v1",
        "engine_group_count": len(summary["engine_groups"]),
        "workload_count": len({row["workload"] for row in bundle["records"]}),
        "exact_positive_command_work_count": len({row["reference_sha256"] for row in bundle["records"]
            if type((row.get("work") or {}).get("exact_macs")) is int
            and row["work"]["exact_macs"] > 0}),
        "missing_contracts": sorted({item for row in bundle["records"] for item in row["missing_contracts"]}),
        "artifact": {"path": "/perf-control/historical_reference.json", "sha256": sha256},
        "scope": "historical engine-relative references only; not warm rates, peaks or target-cycle authority"})
    return raw, compact


def agent_analysis_view(record: Mapping[str, Any], *, complete_evidence: str,
                        context_provider_installed: bool | None = None) -> dict[str, Any]:
    """Keep priorities in the immediate response; the complete, unpruned graph stays available."""
    from merlin.perf.structural_delta import compare_machine_objects

    view = copy.deepcopy(dict(record))
    brief = view.setdefault("analysis", {}).setdefault("optimization_brief", {})
    brief.pop("package_inventory", None)
    if record.get("historical_reference") is not None:
        brief["historical_reference"] = copy.deepcopy(record["historical_reference"]["summary"])
    comparison_seed = record.get("optimization_baseline")
    if comparison_seed is not None:
        brief["optimization_comparison_seed"] = {key: comparison_seed[key] for key in (
            "selection", "sha256", "reason", "scope", "objective_numerical_qualification")}
    # Surface the existing host comparison before prioritizing raw IR byte reductions.
    # This is presentation only: source/object binding and equality belong to
    # compare_full_model_structure, not to a second classifier in the prompt adapter.
    structural = (record.get("static_comparison") or {}).get("structural_change") or {}
    objects = structural.get("machine_object_comparison") or {}
    object_status = objects.get("status") if structural.get("status") == "compared" else "UNKNOWN"
    messages = {
        "identical": (
            "IR-only change at the audited kernel-object boundary; byte-identical target object; "
            "no demonstrated machine-code saving. Do not prioritize pre-LLVM payload reductions "
            "as hardware traffic savings. Seek a change that survives code generation or separately "
            "measure the relevant compiler/caller effect."),
        "different": (
            "The audited target object changed; this alone does not demonstrate less dynamic work "
            "or a performance benefit. Inspect the changed machine mechanism before pricing it."),
        "UNKNOWN": (
            "No bound cross-revision kernel-object comparison is available. Pre-LLVM metric "
            "reductions do not establish machine-code savings."),
    }
    if object_status not in messages:
        object_status = "UNKNOWN"
    brief["machine_code_evidence_priority"] = {
        "status": object_status,
        "comparison_arm": "preceding_analyzed_revision",
        "previous_iteration": (record.get("static_comparison") or {}).get("previous_iteration"),
        "message": ("Relative to the preceding analyzed revision only: " + messages[object_status]
                    + " This does not erase earlier changes relative to the optimization baseline."),
        "comparison": copy.deepcopy(objects),
        "evidence_pointer": "/static_comparison/structural_change/machine_object_comparison",
        "scope": "relocatable kernel object only; caller, setup, linked ELF and timing are separate",
        "global_performance_benefit": "UNKNOWN",
        "numerical_equivalence": "UNKNOWN",
    }
    analysis = record.get("analysis") or {}
    brief["optimization_baseline_machine_code_comparison"] = {
        **compare_machine_objects(analysis, analysis, before_arm="baseline"),
        "comparison_arm": "optimization_baseline",
        "message": "Cumulative object-byte comparison with the compiled optimization baseline, not the last edit. "
                   "Different bytes do not establish less dynamic work or a speedup.",
        "evidence_pointers": ["/analysis/emission", "/analysis/diagnostics/machine_artifact_activity"],
        "global_performance_benefit": "UNKNOWN", "numerical_equivalence": "UNKNOWN",
    }
    graph = view.get("analysis", {}).get("diagnostics", {}).get("captured_logical_graph", {})
    for key in ("dispatch_program", "nodes", "buffers"):
        graph.pop(key, None)
    graph["complete_unpruned_evidence"] = complete_evidence
    storage = storage_encoding_agent_summary(record, complete_evidence=complete_evidence)
    if storage is not None:
        plan = view["analysis"]["diagnostics"]["verified_global_plan_emission"]
        plan.pop("storage_encodings", None)
        plan["storage_encoding_summary"] = storage
        brief["storage_encoding_obligations"] = copy.deepcopy(storage)
    baseline_storage = storage_encoding_agent_summary(record, complete_evidence=complete_evidence, arm="baseline")
    if baseline_storage is not None:
        baseline_plan = view["analysis"]["diagnostics"]["verified_baseline_global_plan_emission"]
        baseline_plan.pop("storage_encodings", None)
        baseline_plan["storage_encoding_summary"] = baseline_storage
    view["controlled_context_capability"] = controlled_context_capability(
        record.get("analysis") or {}, provider_installed=context_provider_installed)
    # Measurements belong in the immediate decision context, not only in an optional raw attachment.
    if record.get("decision_feedback"):
        feedback = record["decision_feedback"]
        view["measurement_driven_next_step"] = feedback["next_step"]
        brief["scoped_mechanism_coverage"] = {
            "scope": feedback["scope"], "status": feedback["status"],
            "occupancy_and_latency_hiding": feedback["observation"],
            "source_task_index": feedback["source_task_index"],
            "source_op_indices": feedback["source_op_indices"],
            "same_declared_movement_work": feedback["same_declared_movement_work"],
            "physical_movement_bytes": None, "full_model_occupancy": "UNKNOWN",
            "full_model_contention": "UNKNOWN", "full_model_cycle_ordering": "UNMEASURED",
            "buffer_capacity_contract": feedback["buffer_capacity_contract"],
            "pipeline_projection_admitted": False,
        }
    portfolio = view.get("portfolio") or {}
    for index, member in enumerate(portfolio.get("members") or ()):
        member_analysis = member.get("analysis")
        if not isinstance(member_analysis, Mapping):
            continue
        member_brief = member_analysis.get("optimization_brief") or {}
        ranked = []
        for action in (member_brief.get("ranked_actions") or ())[:4]:
            ranked.append({key: copy.deepcopy(action.get(key)) for key in (
                "rank", "kind", "status", "detail", "evidence", "required_effects")})
            ranked[-1]["edit_surfaces"] = [
                {key: copy.deepcopy(surface.get(key)) for key in (
                    "id", "path", "symbol", "scope", "effects")}
                for surface in (action.get("edit_surfaces") or ())]
        diagnostics = member_analysis.get("diagnostics") or {}
        member["analysis"] = {
            "schema": "portfolio_member_agent_summary_v1",
            "candidate_sha256": member_analysis.get("candidate_sha256"),
            "failure": copy.deepcopy(member_analysis.get("failure")),
            "optimization_brief": {
                "objective": copy.deepcopy(member_brief.get("objective")),
                "ranked_actions": ranked,
                "gap_coverage": copy.deepcopy(member_brief.get("gap_coverage")),
                "global_planner_wiring": copy.deepcopy(member_brief.get("global_planner_wiring")),
                "mechanism_coverage": copy.deepcopy(member_brief.get("mechanism_coverage")),
            },
            "global_signals": {key: copy.deepcopy(diagnostics.get(key)) for key in (
                "lower_bound", "barriers", "ordering_signals", "queued_movement_context",
                "structural_levels")},
            "complete_unpruned_evidence": {
                "path": complete_evidence,
                "json_pointer": f"/portfolio/members/{index}/analysis",
            },
        }
    return view


def portfolio_action_digest(record: Mapping[str, Any], *, complete_evidence: str,
                            edit_contract: Mapping[str, Any] | None) -> dict[str, Any]:
    """Resolve every portfolio member to one compact, action-oriented host view.

    The iteration record intentionally stores the primary analysis by JSON reference while
    secondary members are inline.  That is efficient archival structure but a poor navigation
    surface for an authoring agent.  Resolve the reference here and expose only bounded totals and
    edit surfaces that exactly occur in the host-frozen authority.
    """
    portfolio = record.get("portfolio") or {}
    members = portfolio.get("members") or ()
    authorized = {
        (row.get("surface_id"), row.get("path"), row.get("symbol"))
        for row in ((edit_contract or {}).get("existing_symbols") or ())
        if isinstance(row, Mapping)
    }
    rows = []
    for index, member in enumerate(members):
        analysis = record.get("analysis") if index == 0 else member.get("analysis")
        if not isinstance(analysis, Mapping):
            analysis = {}
        diagnostics = analysis.get("diagnostics") or {}
        arm = (diagnostics.get("arms") or {}).get("candidate") or {}
        representation = arm.get("representation_activity") or {}
        movement = arm.get("movement") or {}
        plan = diagnostics.get("verified_global_plan_emission") or {}
        host = plan.get("host_activity") or {}
        placement = (diagnostics.get("model_contraction_placement") or {}).get("candidate") or {}
        task_kinds: dict[str, int] = {}
        for kind in (plan.get("declared_task_kinds") or {}).values():
            task_kinds[str(kind)] = task_kinds.get(str(kind), 0) + 1
        actions = []
        for action in ((analysis.get("optimization_brief") or {}).get("ranked_actions") or ())[:4]:
            surfaces = []
            for surface in action.get("edit_surfaces") or ():
                key = (surface.get("id"), surface.get("path"), surface.get("symbol"))
                if key not in authorized:
                    continue
                surfaces.append({key_name: copy.deepcopy(surface.get(key_name)) for key_name in (
                    "id", "path", "symbol", "scope", "effects")})
                surfaces[-1]["authority"] = "exact_host_frozen_existing_symbol"
            actions.append({key_name: copy.deepcopy(action.get(key_name)) for key_name in (
                "rank", "kind", "status", "detail", "evidence", "required_effects")})
            actions[-1]["authorized_edit_surfaces"] = surfaces
        rows.append({
            "identity": copy.deepcopy(member.get("identity")),
            "readiness": copy.deepcopy(member.get("readiness")),
            "totals": {
                "logical_graph_status": (diagnostics.get("captured_logical_graph") or {}).get("status"),
                "logical_dispatches": (diagnostics.get("captured_logical_graph") or {}).get("dispatches"),
                "accelerator_macs": arm.get("macs"),
                "accelerator_work_exact": arm.get("exact"),
                "movement_known_bytes": movement.get("known_bytes"),
                "movement_known_bytes_in": movement.get("known_bytes_in"),
                "movement_known_bytes_out": movement.get("known_bytes_out"),
                "command_counts": copy.deepcopy(representation.get("command_counts")),
                "task_kind_counts": task_kinds,
                "lane_counts": copy.deepcopy((representation.get("placement") or {}).get("lane_counts")),
                "lane_transitions": (representation.get("placement") or {}).get(
                    "adjacent_lane_transitions"),
                "contraction_count": placement.get("contraction_count"),
                "contraction_macs_by_lane": copy.deepcopy(placement.get("macs_by_lane")),
                "host_dynamic_operations": copy.deepcopy(host.get("dynamic_operations")),
                "host_load_payload_bytes": host.get("load_payload_bytes"),
                "host_store_payload_bytes": host.get("store_payload_bytes"),
                "host_static_allocation_payload_bytes": host.get("static_allocation_payload_bytes"),
            },
            "top_ranked_actions": actions,
            "complete_unpruned_evidence": {
                "path": complete_evidence,
                "json_pointer": "/analysis" if index == 0 else
                                f"/portfolio/members/{index}/analysis",
            },
        })
    return {"schema": "portfolio_action_digest_v1",
            "portfolio_sha256": portfolio.get("portfolio_sha256"),
            "candidate_sha256": record.get("candidate_sha256"),
            "members": rows,
            "selection": "per-model Pareto evidence; totals are never summed across models",
            "timing_status": "UNMEASURED_FULL_MODEL"}


def compiler_dependency_record(candidate: Path, *, shared_source_root: Path | None = None) -> dict[str, Any]:
    """Hash candidate code plus its statically resolved trusted Merlin import closure.

    This does not import candidate modules to discover dependencies. Shared helpers remain shared,
    but an edit to one invalidates the compiler identity just as an edit inside the package does.
    External Python/toolchain installations belong to the launch environment identity separately.
    """
    from merlin.perf.static_imports import imported_attribute_paths, resolve_lazy_export

    root = (shared_source_root or PAS.merlin_dir() / "python" / "merlin").resolve()
    if not root.is_dir() or not (root / "__init__.py").is_file():
        raise ValueError("compiler shared-source root must identify an existing Merlin package")
    pending: list[tuple[Path, str]] = [(path, "") for path in candidate.rglob("*.py")
                                     if "__pycache__" not in path.parts]
    scanned: set[Path] = set()
    shared: dict[str, str] = {}
    requested: set[str] = set()
    lazy_exports: dict[str, str] = {}

    def enqueue(module: str) -> None:
        if module != "merlin" and not module.startswith("merlin."):
            return
        if module in requested:
            return
        requested.add(module)
        parts = module.split(".")[1:]
        choices = (root.joinpath(*parts).with_suffix(".py"),
                   root.joinpath(*parts) / "__init__.py") if parts else (root / "__init__.py",)
        for source in choices:
            if source.is_file():
                if source.is_symlink() or (source.resolve() != root and root not in source.resolve().parents):
                    raise ValueError("shared compiler import escapes the trusted Merlin source root")
                pending.append((source, module))
                for count in range(len(parts)):
                    parent = root.joinpath(*parts[:count]) / "__init__.py"
                    if parent.is_file():
                        pending.append((parent, ".".join(("merlin", *parts[:count]))))
                return
        # A from-import may name an exported class rather than a source file. Resolve just
        # that requested lazy symbol, without importing __init__ or granting sibling modules.
        package, _, symbol = module.rpartition(".")
        initializer = root.joinpath(*package.split(".")[1:]) / "__init__.py"
        if not initializer.is_file():
            return
        if initializer.is_symlink() or root not in initializer.resolve().parents:
            raise ValueError("shared compiler import escapes the trusted Merlin source root")
        resolution = resolve_lazy_export(initializer.read_bytes(), package=package, symbol=symbol)
        if resolution.status == "unresolved":
            raise ValueError(f"unresolved shared lazy import {module}: {resolution.reason}")
        if resolution.status == "resolved":
            assert resolution.module is not None
            lazy_exports[module] = resolution.module
            enqueue(package)
            enqueue(resolution.module)

    while pending:
        source, module = pending.pop()
        source = source.resolve()
        if source in scanned:
            continue
        scanned.add(source)
        payload = source.read_bytes()
        if root in source.parents:
            shared[source.relative_to(root).as_posix()] = hashlib.sha256(payload).hexdigest()
        tree = ast.parse(payload, filename=str(source))
        package = module if source.name == "__init__.py" else module.rpartition(".")[0]
        bindings: dict[str, set[str]] = {}
        import_functions: set[str] = {"__import__"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    enqueue(alias.name)
                    bindings.setdefault(alias.asname or alias.name.split(".")[0], set()).add(
                        alias.name if alias.asname else alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                name = node.module or ""
                if node.level:
                    if not package:
                        continue  # candidate-local relative import; all its .py files are scanned
                    prefix = package.split(".")[:len(package.split(".")) - node.level + 1]
                    name = ".".join((*prefix, name)) if name else ".".join(prefix)
                enqueue(name)
                for alias in node.names:
                    enqueue(name + "." + alias.name)
                    if alias.name != "*":
                        bindings.setdefault(alias.asname or alias.name, set()).add(name + "." + alias.name)
                    if name == "importlib" and alias.name == "import_module":
                        import_functions.add(alias.asname or alias.name)
        for path in sorted(imported_attribute_paths(tree, bindings)):
            enqueue(path)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and node.args
                  and isinstance(node.args[0], ast.Constant)
                  and isinstance(node.args[0].value, str)
                  and ((isinstance(node.func, ast.Name) and node.func.id in import_functions)
                       or (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"))):
                enqueue(node.args[0].value)
    body = {"candidate_sha256": hash_tree(candidate)["sha256"],
            "shared_source_root": str(root), "shared_sources": dict(sorted(shared.items())),
            "selected_lazy_exports": dict(sorted(lazy_exports.items()))}
    return {"schema": "compiler_implementation_dependencies_v1", **body,
            "compiler_implementation_sha256": PAS._document_sha256(body),
            "scope": "candidate and static Merlin source import closure; toolchain identity is separate"}


class GlobalPerfExperiment:
    """Host controller shared by interactive/agent search and the compile-only readiness CLI.

    ``analyzer`` and ``plan_verifier`` are host integrations, never candidate-imported entrypoints.
    A candidate can suggest a transformation but cannot assert its own equivalence or objective.
    Every submitted edit invokes full-model compilation again, including candidates later rejected.
    """

    def __init__(self, *, baseline: Path, baseline_sha256: str,
                 sentinel: PAS.StageE2ESentinel, target: str, target_sha256: str,
                 portfolio_sentinels: Sequence[PAS.StageE2ESentinel] = (),
                 output: Path, timeout_s: int = 300,
                 target_descriptor: Path | None = None,
                 phase1: FrozenPhase1 | None = None,
                 optimization_baseline: Path | None = None,
                 optimization_baseline_sha256: str | None = None,
                 optimization_baseline_reason: str = "host-selected immutable optimization comparison seed",
                 historical_reference_path: Path | None = None,
                 historical_reference_sha256: str | None = None,
                 compiler_shared_source_root: Path | None = None,
                 baseline_emission_cache: Path | None = None,
                 baseline_emission_seed_runs: Sequence[Path] = (),
                 portfolio_analysis_workers: int = 1,
                 minimum_memory_available_bytes: int = 0,
                 analyzer: Callable[..., Mapping[str, Any]] = PAS.analyze_whole_model_emission,
                 plan_verifier: Callable[..., Mapping[str, Any]] | None = None):
        if not 0 < timeout_s <= ITERATION_MAX_SECONDS:
            raise ValueError("a global iteration must fit the 600-second wall budget")
        if not PAS._is_sha256(target_sha256):
            raise ValueError("the target descriptor must have an exact SHA-256 identity")
        if hash_tree(baseline)["sha256"] != baseline_sha256:
            raise ValueError("frozen Phase-1 compiler digest mismatch")
        if (optimization_baseline is None) != (optimization_baseline_sha256 is None):
            raise ValueError("optimization baseline requires both an explicit path and SHA-256")
        explicit_comparison = optimization_baseline is not None
        comparison_source = optimization_baseline if explicit_comparison else baseline
        comparison_sha256 = optimization_baseline_sha256 if explicit_comparison else baseline_sha256
        if (not PAS._is_sha256(comparison_sha256)
                or hash_tree(comparison_source)["sha256"] != comparison_sha256):
            raise ValueError("optimization baseline digest mismatch")
        if not optimization_baseline_reason.strip():
            raise ValueError("optimization baseline selection requires a reason")
        if explicit_comparison and (comparison_source.is_symlink()
                or any(path.is_symlink() for path in comparison_source.rglob("*"))):
            raise ValueError("optimization baseline must contain real immutable source files")
        if output.exists():
            raise ValueError("global experiment output must be fresh")
        if (historical_reference_path is None) != (historical_reference_sha256 is None):
            raise ValueError("historical reference requires both explicit path and SHA-256")
        reference_raw, reference_brief = (load_historical_reference(
            historical_reference_path, historical_reference_sha256, target=target)
            if historical_reference_path is not None else (None, None))
        self.historical_reference_source = historical_reference_path
        self.historical_reference = None
        self.baseline, self.baseline_sha256 = baseline, baseline_sha256
        self.compiler_shared_source_root = compiler_shared_source_root
        self.baseline_dependencies = self._compiler_dependencies(baseline)
        members = (sentinel, *tuple(portfolio_sentinels))
        member_hashes = [member.capsule_sha256 for member in members]
        if (any(not PAS._is_sha256(value) for value in member_hashes)
                or len(set(member_hashes)) != len(member_hashes)):
            raise ValueError("complete-model portfolio members require distinct exact identities")
        self.sentinel, self.portfolio_sentinels = sentinel, members
        self.target, self.target_sha256 = target, target_sha256
        self.target_descriptor = target_descriptor
        if target_descriptor is not None and PAS._sha256_file(target_descriptor) != target_sha256:
            raise ValueError("target descriptor digest mismatch")
        self.output, self.timeout_s = output, timeout_s
        if portfolio_analysis_workers < 1 or minimum_memory_available_bytes < 0:
            raise ValueError("portfolio concurrency policy is invalid")
        self.portfolio_analysis_workers = portfolio_analysis_workers
        self.minimum_memory_available_bytes = minimum_memory_available_bytes
        self.analyzer, self.plan_verifier = analyzer, plan_verifier
        self.phase1 = phase1
        self.phase1_binding = phase1.verify(baseline) if phase1 is not None else None
        self.host_policy = host_verification_policy_record()
        self.iterations: list[dict[str, Any]] = []
        self._analysis_lock = threading.Lock()
        self._artifacts: dict[str, Any] = {}
        self._previous_artifacts: Mapping[str, Any] | None = None
        self._baseline_artifacts: Mapping[str, Any] | None = None
        self._portfolio_baseline_artifacts: dict[str, Mapping[str, Any]] = {}
        self._optimization_baseline_sandbox: Mapping[str, Any] | None = None
        self._optimization_baseline_sandbox_sha256: str | None = None
        self._compiler_sandboxes: dict[int, Mapping[str, Any]] = {}
        self.completion_contract: Any | None = None
        self.edit_contract: dict[str, Any] | None = None
        self.edit_guidance_inventory = None
        self.output.mkdir(parents=True)
        if reference_raw is not None:
            reference_path = self.output / "historical_reference.json"
            with reference_path.open("xb") as stream:
                stream.write(reference_raw)
            reference_path.chmod(0o444)
            self.historical_reference = {"path": str(reference_path.resolve()),
                "sha256": historical_reference_sha256, "summary": reference_brief}
        self._historical_reference_binding_sha256 = PAS._document_sha256(self.historical_reference)
        self.optimization_baseline = baseline
        if explicit_comparison:
            self.optimization_baseline = self.output / "optimization_baseline"
            shutil.copytree(comparison_source, self.optimization_baseline,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            if hash_tree(self.optimization_baseline)["sha256"] != comparison_sha256:
                raise ValueError("optimization baseline changed while capturing its immutable snapshot")
            for path in self.optimization_baseline.rglob("*"):
                path.chmod(path.stat().st_mode & ~0o222)
            self.optimization_baseline.chmod(0o555)
        self.optimization_baseline_sha256 = comparison_sha256
        self.optimization_baseline_binding = {
            "schema": "global_optimization_baseline_v1",
            "selection": "explicit_host_seed" if explicit_comparison else "frozen_phase1_compiler",
            "path": str(self.optimization_baseline.resolve()), "sha256": comparison_sha256,
            "compiler_dependencies": self._compiler_dependencies(self.optimization_baseline),
            "reason": optimization_baseline_reason if explicit_comparison else "legacy frozen compiler comparison",
            "scope": "optimization comparison only; does not replace or extend frozen Phase-1 qualification",
            "objective_numerical_qualification": "UNPROVEN", "phase1_regraded": False,
        }
        self.optimization_baseline_binding_sha256 = PAS._document_sha256(self.optimization_baseline_binding)
        portfolio_identity = [sentinel_identity(member, role=(
            "primary" if index == 0 else "training"))
            for index, member in enumerate(self.portfolio_sentinels)]
        self.portfolio_identity = {"schema": "full_model_optimization_portfolio_v1",
            "members": portfolio_identity,
            "selection": "multi_model_pareto_without_invented_static_cycle_total",
            "execution": "bounded_host_admitted_analysis_with_deterministic_record_order",
            "holdout_policy": "separate_post_authoring_evaluation",
            "micro_graphs": "smoke_and_mechanism_calibration_only"}
        self.portfolio_identity_sha256 = PAS._document_sha256(self.portfolio_identity)
        self.baseline_emission_cache_binding = None
        self.baseline_emission_cache_seeds: list[dict[str, Any]] = []
        if baseline_emission_cache is not None:
            cache_root = Path(baseline_emission_cache).resolve()
            if (cache_root.is_relative_to(self.baseline.resolve())
                    or cache_root.is_relative_to(self.optimization_baseline.resolve())):
                raise ValueError("baseline emission cache cannot be inside a compiler tree")
            self.baseline_emission_cache_binding = {
                "schema": "baseline_emission_cache_binding_v1",
                "root": str(cache_root),
                "compiler_dependencies_sha256": compiler_dependency_content_sha256(
                    self.optimization_baseline_binding["compiler_dependencies"]),
            }
            schema = PAS.whole_program_schema_record()
            for seed_run in baseline_emission_seed_runs:
                entries = seed_baseline_emission_cache_from_run(
                    cache_binding=self.baseline_emission_cache_binding,
                    seed_run=seed_run, baseline=self.optimization_baseline,
                    sentinels=self.portfolio_sentinels, target=self.target,
                    compiler_api_schema=schema)
                self.baseline_emission_cache_seeds.append({
                    "run": str(Path(seed_run).resolve()),
                    "entries": [{"key": row["key"],
                                 "capsule_sha256": row["identity"]["capsule_sha256"],
                                 "emission_wall_seconds": row["emission_wall_seconds"]}
                                for row in entries],
                })
        elif baseline_emission_seed_runs:
            raise ValueError("baseline emission seed runs require a cache root")
        self._write("experiment.json", {
            "schema": "global_perf_experiment_v1", "objective": "full_model_graph_and_global_plan",
            "historical_reference": self.historical_reference,
            "baseline_sha256": baseline_sha256, "target": target,
            "optimization_baseline_sha256": comparison_sha256,
            "optimization_baseline": self.optimization_baseline_binding,
            "baseline_compiler_dependencies": self.baseline_dependencies,
            "compiler_shared_source_root": self.baseline_dependencies["shared_source_root"],
            "target_sha256": target_sha256, "capsule": sentinel.capsule,
            "capsule_sha256": sentinel.capsule_sha256,
            "portfolio": self.portfolio_identity,
            "portfolio_sha256": self.portfolio_identity_sha256,
            "baseline_emission_cache": self.baseline_emission_cache_binding,
            "baseline_emission_cache_seeds": self.baseline_emission_cache_seeds,
            "portfolio_analysis_workers": self.portfolio_analysis_workers,
            "minimum_memory_available_bytes": self.minimum_memory_available_bytes,
            "maximum_iteration_seconds": timeout_s,
            "full_model_simulation_allowed": False, "probe_measurements_required": False,
            "firesim_stage": "optional_post_freeze_validation",
            "phase1_action": "reuse_exact_snapshot", "micro_plateau_stops_search": False,
            "launch_scope": "qualified_macro_experiment" if self.phase1_binding else "development_readiness_only",
            "phase1_qualification": self.phase1_binding, "host_verification_policy": self.host_policy,
        })

    def _baseline_emission_observations(self) -> dict[str, dict[str, Any]]:
        """Read bounded exact cache receipts without loading large emitted artifacts."""
        if self.baseline_emission_cache_binding is None:
            return {}
        from merlin.targetgen import oot_runner as OR

        package = OR.load_package(self.optimization_baseline)
        entrypoints = OR.analysis_emission_entrypoints(package)
        schema = PAS.whole_program_schema_record()
        observations: dict[str, dict[str, Any]] = {}
        for sentinel in self.portfolio_sentinels:
            source = Path(sentinel.frozen_source_path)
            descriptor = PAS._mapping_file(source / "capsule.yaml", yaml_file=True)
            interface = source / str(descriptor.get("interface_mlir") or "capsule.interface.mlir")
            identity = PAS.baseline_emission_cache_identity(
                baseline_sha256=self.optimization_baseline_sha256,
                capsule_sha256=sentinel.capsule_sha256,
                source_sha256=PAS._sha256(interface.read_bytes()), target=self.target,
                compiler_dependencies_sha256=self.baseline_emission_cache_binding[
                    "compiler_dependencies_sha256"],
                compiler_api_schema=schema, entrypoints=entrypoints)
            cached = PAS.baseline_emission_cache_observation(
                self.baseline_emission_cache_binding, identity)
            if cached is not None:
                observations[sentinel.capsule_sha256] = cached
        return observations

    def _baseline_emission_costs(self) -> dict[str, float]:
        return {digest: row["emission_wall_seconds"]
                for digest, row in self._baseline_emission_observations().items()}

    def _portfolio_analysis_cost_estimates(self) -> list[float]:
        """Estimate changed-candidate member wall from exact prior receipts, without model constants."""
        observations = self._baseline_emission_observations()
        interface_sizes = []
        for sentinel in self.portfolio_sentinels:
            source = Path(sentinel.frozen_source_path)
            descriptor = PAS._mapping_file(source / "capsule.yaml", yaml_file=True)
            interface = source / str(descriptor.get("interface_mlir") or "capsule.interface.mlir")
            interface_sizes.append(max(1, interface.stat().st_size))
        measured = []
        for sentinel in self.portfolio_sentinels:
            row = observations.get(sentinel.capsule_sha256)
            if row is None:
                measured.append(None)
            elif row.get("observed_analysis_wall_seconds") is not None:
                measured.append(float(row["observed_analysis_wall_seconds"]))
            else:
                # A changed candidate replaces the cached baseline emission with one candidate
                # emission and reruns both host audits.  Two emission durations plus a fixed
                # generic audit allowance is deliberately conservative until a completed receipt
                # provides the exact whole-member wall observation.
                measured.append(float(row["emission_wall_seconds"]) * 2.0 + 60.0)
        known_rates = sorted(
            value / size for value, size in zip(measured, interface_sizes, strict=True)
            if value is not None)
        median_rate = (None if not known_rates else known_rates[len(known_rates) // 2]
                       if len(known_rates) % 2 else
                       0.5 * (known_rates[len(known_rates) // 2 - 1]
                              + known_rates[len(known_rates) // 2]))
        return [float(value if value is not None else
                      size * median_rate if median_rate is not None else size)
                for value, size in zip(measured, interface_sizes, strict=True)]

    def mandatory_analysis_reserve_seconds(self, maximum_seconds: float) -> dict[str, Any]:
        """Reserve a measured, generic concurrent-validation window before round finalization."""
        observations = self._baseline_emission_observations()
        if observations:
            concurrency = portfolio_analysis_concurrency(
                requested_workers=self.portfolio_analysis_workers,
                members=len(self.portfolio_sentinels),
                memory_available_bytes=_host_memory_available_bytes(),
                minimum_memory_available_bytes=self.minimum_memory_available_bytes)
            schedule = portfolio_concurrent_schedule(
                self._portfolio_analysis_cost_estimates(), concurrency["admitted_workers"])
            estimate = schedule["projected_wall_seconds"]
            basis = "resource_admitted_schedule_of_exact_or_conservative_member_receipts"
        else:
            estimate = min(float(self.timeout_s), max(60.0, maximum_seconds * 0.5))
            basis = "cold_cache_half_tool_window"
            concurrency = None
            schedule = None
        reserve = min(float(maximum_seconds), math.ceil(estimate * 1.05 + 5.0))
        return {"schema": "mandatory_portfolio_analysis_reserve_v1",
                "seconds": reserve, "estimated_member_wall_seconds": estimate,
                "basis": basis, "observed_members": len(observations),
                "analysis_concurrency": concurrency,
                "analysis_schedule": schedule,
                "execution": "concurrent_shared_deadline",
                "scope": "host tool-window admission; not a performance estimate"}

    def _write(self, name: str, record: Mapping[str, Any]) -> Path:
        path = self.output / name
        payload = PAS._canonical_json(record)
        with path.open("xb") as stream:
            stream.write(payload)
        path.chmod(0o444)
        return path

    def _compiler_dependencies(self, candidate: Path) -> dict[str, Any]:
        return compiler_dependency_record(candidate, shared_source_root=self.compiler_shared_source_root)

    def stage_historical_reference(self, control: Path, *, workspace: Path) -> None:
        """Use the existing read-only /perf-control grant, never the compiler sandbox."""
        self._check_inputs()
        if self.historical_reference is None:
            return
        if self.historical_reference_source.resolve().is_relative_to(workspace.resolve()):
            raise ValueError("historical reference source is in the writable author workspace")
        source = Path(self.historical_reference["path"])
        raw, summary = load_historical_reference(source, self.historical_reference["sha256"],
            target=self.target, candidate_roots=(workspace,))
        if summary != self.historical_reference["summary"] or control.resolve().is_relative_to(workspace.resolve()):
            raise ValueError("historical reference summary or control boundary changed")
        path = control / "historical_reference.json"
        with path.open("xb") as stream:
            stream.write(raw)
        path.chmod(0o444)

    def _check_inputs(self) -> None:
        if host_verification_policy_record() != self.host_policy:
            raise ValueError("host verification policy changed during the global experiment")
        if (PAS._document_sha256(self.portfolio_identity) != self.portfolio_identity_sha256
                or self.portfolio_identity["members"] != [sentinel_identity(member, role=(
                    "primary" if index == 0 else "training"))
                    for index, member in enumerate(self.portfolio_sentinels)]):
            raise ValueError("complete-model portfolio identity changed during global search")
        if PAS._document_sha256(self.historical_reference) != self._historical_reference_binding_sha256:
            raise ValueError("historical reference binding changed")
        if self.historical_reference is not None:
            path = Path(self.historical_reference["path"])
            if path.is_symlink() or PAS._sha256_file(path) != self.historical_reference["sha256"]:
                raise ValueError("retained historical reference bytes changed")
        if self.edit_contract is not None:
            if (PAS._document_sha256(self.edit_contract) != self.edit_scope_binding["contract_document_sha256"]
                    or hash_tree(self.edit_scope_seed)["sha256"] != self.edit_scope_binding["initial_candidate_sha256"]):
                raise ValueError("host-frozen compiler edit authority changed")
            guidance = self.edit_scope_binding.get("guidance_inventory_sha256")
            if guidance is not None and (self.edit_guidance_inventory is None or
                    PAS._document_sha256(self.edit_guidance_inventory.to_dict()) != guidance or
                    PAS._document_sha256(self.edit_scope_binding.get("guidance_inventory")) != guidance):
                raise ValueError("host-frozen compiler guidance changed")
        if self.phase1 is not None and self.phase1.verify(self.baseline) != self.phase1_binding:
            raise ValueError("frozen Phase-1 qualification or waivers changed during global search")
        if (self.target_descriptor is not None
                and PAS._sha256_file(self.target_descriptor) != self.target_sha256):
            raise ValueError("target descriptor changed during global search")
        if hash_tree(self.baseline)["sha256"] != self.baseline_sha256:
            raise ValueError("frozen compiler changed during global search")
        if self._compiler_dependencies(self.baseline) != self.baseline_dependencies:
            raise ValueError("frozen baseline shared compiler dependencies changed during global search")
        if (PAS._document_sha256(self.optimization_baseline_binding) != self.optimization_baseline_binding_sha256
                or hash_tree(self.optimization_baseline)["sha256"] != self.optimization_baseline_sha256
                or self._compiler_dependencies(self.optimization_baseline)
                != self.optimization_baseline_binding["compiler_dependencies"]):
            raise ValueError("immutable optimization baseline or shared compiler dependencies changed")
        for member in self.portfolio_sentinels:
            source = Path(member.frozen_source_path)
            if PAS._exact_tree_record(source)["sha256"] != member.capsule_sha256:
                raise ValueError(
                    f"complete-model objective changed during global search: {member.capsule}")

    def freeze_edit_scope(self, candidate: Path, contract: Mapping[str, Any], *,
                          source_pins: Mapping[str, str] | None = None,
                          host_surface_declarations: Sequence[Mapping[str, Any]] | None = None
                          ) -> dict[str, Any]:
        """Host startup only: retain initial source and its externally approved edit authority."""
        from merlin.perf.compiler_edit_scope import validate_edit_contract
        if self.iterations or self.edit_contract is not None:
            raise ValueError("compiler edit authority must be frozen once before candidate execution")
        if any(path.is_symlink() for path in candidate.rglob("*")):
            raise ValueError("initial compiler edit source contains a symlink")
        validated = validate_edit_contract(contract, candidate)
        guidance = None
        if host_surface_declarations is not None:
            guidance = PAS.inspect_compiler_package(candidate,
                host_surface_declarations=host_surface_declarations)
            authorized = {(owner["surface_id"], owner["path"], owner["symbol"])
                          for owner in validated["existing_symbols"]}
            if any((surface.id, surface.path, surface.symbol) not in authorized
                   for surface in guidance.surfaces):
                raise ValueError("host guidance surface lies outside frozen edit authority")
        if source_pins is not None:
            actual = {path.relative_to(candidate).as_posix(): PAS._sha256_file(path)
                      for path in candidate.rglob("*") if path.is_file()
                      and ".git" not in path.relative_to(candidate).parts
                      and "__pycache__" not in path.parts and path.suffix != ".pyc"}
            if actual != dict(source_pins):
                raise ValueError("host edit catalog source-file pins do not match the initial compiler")
        seed = self.output / "edit_scope_seed"
        shutil.copytree(candidate, seed, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for path in seed.rglob("*"):
            if path.is_symlink():
                raise ValueError("initial compiler edit source contains a symlink")
            path.chmod(path.stat().st_mode & ~0o222)
        seed.chmod(0o555)
        self.edit_contract, self.edit_scope_seed = validated, seed
        self.edit_guidance_inventory = guidance
        self.edit_scope_binding = {"schema": "host_frozen_compiler_edit_authority_v1",
            "initial_candidate_sha256": hash_tree(seed)["sha256"],
            "contract_document_sha256": PAS._document_sha256(validated),
            "contract": validated, "seed_path": str(seed), "source_pins_checked": source_pins is not None}
        if guidance is not None:
            self.edit_scope_binding["guidance_inventory"] = guidance.to_dict()
            self.edit_scope_binding["guidance_inventory_sha256"] = PAS._document_sha256(guidance.to_dict())
        self._write("compiler_edit_authority.json", self.edit_scope_binding)
        return copy.deepcopy(self.edit_scope_binding)

    def validate_candidate_scope(self, candidate: Path) -> dict[str, Any]:
        from merlin.perf.compiler_edit_scope import inspect_compiler_edits
        if self.edit_contract is None:
            return {"status": "unconfigured_development_only"}
        self._check_inputs()
        result = inspect_compiler_edits(self.edit_scope_seed, candidate, self.edit_contract)
        if result["status"] != "allowed":
            self._write(f"edit_scope_refusal_{time.time_ns()}.json", {
                **result, "candidate_sha256": hash_tree(candidate)["sha256"]})
            raise ValueError("candidate edit exceeds host-frozen authority: " + str(result["violations"]))
        return result

    def inspect_optimization_surfaces(self, candidate: Path) -> dict[str, Any]:
        """Expose current AST locations with host-frozen semantics, never self-granted permissions."""
        self._check_inputs()
        self.validate_candidate_scope(candidate)
        if self.edit_guidance_inventory is None:
            return PAS.inspect_compiler_package(candidate).to_dict()
        inventory = PAS.inspect_compiler_package(candidate, host_surface_declarations=[
            surface.to_dict() for surface in self.edit_guidance_inventory.surfaces])
        return {**inventory.to_dict(), "host_guidance_binding": {
            "inventory_sha256": self.edit_scope_binding["guidance_inventory_sha256"],
            "contract_document_sha256": self.edit_scope_binding["contract_document_sha256"],
            "permission_scope": "unchanged host-frozen edit contract",
            "source_scope": "current candidate AST locations; semantics from frozen host inventory"}}

    def analyze(self, candidate: Path, *, hypothesis: str, timeout_s: int | None = None) -> dict[str, Any]:
        """Serialize submitted revisions; concurrent identical requests reuse the completed result."""
        started = time.monotonic()
        budget = self.timeout_s if timeout_s is None else min(self.timeout_s, timeout_s)
        if budget <= 0 or not self._analysis_lock.acquire(timeout=budget):
            raise TimeoutError("whole-model request exhausted its budget waiting for the active analysis")
        try:
            remaining = budget - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("whole-model request has no budget after waiting for the active analysis")
            return self._analyze_locked(candidate, hypothesis=hypothesis, timeout_s=remaining)
        finally:
            self._analysis_lock.release()

    def _analyze_portfolio_member(self, submitted: Path, *, candidate_sha256: str,
                                  sentinel: PAS.StageE2ESentinel, timeout_s: float,
                                  scope: Mapping[str, Any], primary: bool = False,
                                  analyzer_override: Callable[..., Mapping[str, Any]] | None = None,
                                  ) -> tuple[dict[str, Any], dict[str, Any], Mapping[str, Any] | None]:
        """Compile one full graph with member-local artifacts and analyzer state."""
        from merlin.perf.analysis_worker import IsolatedAnalysisWorker

        retained: dict[str, Any] = {}
        analyzer = analyzer_override or self.analyzer
        kwargs: dict[str, Any] = {
            "timeout_s": timeout_s, "target": self.target,
            "peak_macs_per_cycle": None, "achievable_macs_per_cycle": None,
            "host_verifier_policy_sha256": self.host_policy["sha256"],
        }
        if self.plan_verifier is not None:
            kwargs["global_plan_verifier"] = self.plan_verifier
        if analyzer is PAS.analyze_whole_model_emission or isinstance(analyzer, IsolatedAnalysisWorker):
            kwargs["compiler_api_schema"] = PAS.whole_program_schema_record()
            kwargs["artifact_sink"] = retained.update
            kwargs["baseline_artifacts"] = (
                self._baseline_artifacts if primary else
                self._portfolio_baseline_artifacts.get(sentinel.capsule_sha256))
            kwargs["baseline_emission_cache"] = self.baseline_emission_cache_binding
        try:
            if timeout_s <= 0:
                raise TimeoutError("portfolio member has no remaining iteration budget")
            analysis = dict(analyzer(
                self.optimization_baseline, submitted, sentinel, **kwargs))
            analysis["compiler_edit_scope"] = copy.deepcopy(scope)
            if self.edit_guidance_inventory is not None:
                from merlin.perf.agent_guidance import guidance_for_emission_analysis
                analysis["optimization_brief"] = guidance_for_emission_analysis(
                    analysis["diagnostics"], self.edit_guidance_inventory)
                if primary:
                    analysis["optimization_brief"]["compiler_edit_contract_template"] = copy.deepcopy(
                        self.edit_contract)
                    analysis["optimization_brief"]["host_guidance_binding"] = {
                        "inventory_sha256": self.edit_scope_binding["guidance_inventory_sha256"],
                        "initial_candidate_sha256": self.edit_scope_binding["initial_candidate_sha256"],
                        "contract_document_sha256": self.edit_scope_binding["contract_document_sha256"],
                        "permission_scope": "unchanged host-frozen edit contract",
                        "mapping_scope": ("host-declared semantics; AST/component ownership checked on "
                                          "frozen seed; not proof of emitted effect"),
                    }
        except Exception as exc:
            analysis = {
                "schema": "host_owned_whole_model_emission_failure_v1",
                "candidate_sha256": candidate_sha256,
                "workload": {"capsule_sha256": sentinel.capsule_sha256},
                "diagnostics": {"arms": {"candidate": {"status": "emission_failed"}}},
                "failure": {"type": type(exc).__name__, "reason": str(exc)[:20000]},
                "timing_status": "UNMEASURED",
            }
        analysis["optimization_baseline"] = copy.deepcopy(self.optimization_baseline_binding)
        if analysis.get("candidate_sha256") != candidate_sha256:
            raise ValueError(f"portfolio analysis is not bound to candidate bytes: {sentinel.capsule}")
        if analysis.get("workload", {}).get("capsule_sha256") != sentinel.capsule_sha256:
            raise ValueError(f"portfolio analysis substituted its objective: {sentinel.capsule}")
        return analysis, retained, copy.deepcopy(getattr(analyzer, "completed_sandboxes", None))

    def _member_analyzer(self) -> Callable[..., Mapping[str, Any]]:
        """Give concurrent isolated members independent mutable worker bookkeeping."""
        from merlin.perf.analysis_worker import IsolatedAnalysisWorker
        if isinstance(self.analyzer, IsolatedAnalysisWorker):
            return IsolatedAnalysisWorker(
                stage_path=self.analyzer.stage_path,
                sandbox_factory=self.analyzer.sandbox_factory,
                output=self.analyzer.output)
        return self.analyzer

    def _analyze_locked(self, candidate: Path, *, hypothesis: str,
                        timeout_s: float | None = None) -> dict[str, Any]:
        """One real full-graph compilation per edit, recorded even when readiness is blocked."""
        started = time.monotonic()
        self._check_inputs()
        if not hypothesis.strip():
            raise ValueError("each iteration must state the global transformation hypothesis")
        budget_seconds = self.timeout_s if timeout_s is None else min(self.timeout_s, timeout_s)
        if budget_seconds <= 0:
            raise ValueError("full-model analysis has no remaining iteration budget")
        self.validate_candidate_scope(candidate)
        if self.historical_reference_source is not None and (
                self.historical_reference_source.resolve().is_relative_to(candidate.resolve())
                or Path(self.historical_reference["path"]).is_relative_to(candidate.resolve())):
            raise ValueError("historical reference is inside candidate-writable source")
        dependencies_before = self._compiler_dependencies(candidate)
        before = hash_tree(candidate)["sha256"]
        if (self.iterations and self.iterations[-1]["candidate_sha256"] == before
                and self.iterations[-1]["compiler_dependencies"] == dependencies_before):
            return {**copy.deepcopy(self.iterations[-1]), "exact_analysis_reused": True}
        # Analyze immutable submitted bytes. The agent may keep authoring while this request runs;
        # the result names this snapshot, and _current still refuses a newer unanalysed revision.
        submitted = self.output / f"submission_{len(self.iterations):04d}"
        shutil.copytree(candidate, submitted, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        if hash_tree(submitted)["sha256"] != before:
            raise ValueError("candidate changed while its analysis snapshot was being captured")
        for path in submitted.rglob("*"):
            if not path.is_symlink():
                path.chmod(path.stat().st_mode & ~0o222)
        submitted.chmod(0o555)
        scope = self.validate_candidate_scope(submitted)
        remaining = budget_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("global input verification and snapshot exhausted the iteration budget")
        after = hash_tree(submitted)["sha256"]
        costs = self._baseline_emission_costs()
        concurrency = portfolio_analysis_concurrency(
            requested_workers=self.portfolio_analysis_workers,
            members=len(self.portfolio_sentinels),
            memory_available_bytes=_host_memory_available_bytes(),
            minimum_memory_available_bytes=self.minimum_memory_available_bytes)
        member_results: list[tuple[dict[str, Any], dict[str, Any], Mapping[str, Any] | None,
                                   dict[str, Any], float] | None] = [
            None for _ in self.portfolio_sentinels]
        # Worker count one and worker count N use the same absolute portfolio deadline.  The old
        # one-worker fallback assigned a local weighted slice to each declared-order member; a
        # long-running member could be killed even though the portfolio still had ample time.
        # LPT scheduling still bounds total wall time and gives long observed members
        # first access to the deadline.  Each queued member receives the exact remaining outer
        # budget when it actually starts; results are restored to declared portfolio order below.
        planning = [portfolio_member_analysis_allocation(
            remaining, self.portfolio_sentinels[index:],
            emission_seconds_by_capsule_sha256=costs)
            for index in range(len(self.portfolio_sentinels))]
        deadline = time.monotonic() + remaining
        cost_estimates = self._portfolio_analysis_cost_estimates()
        schedule = portfolio_concurrent_schedule(
            cost_estimates, concurrency["admitted_workers"])
        concurrency["schedule"] = schedule
        concurrency["submission_order_capsule_sha256"] = [
            self.portfolio_sentinels[index].capsule_sha256
            for index in schedule["submission_order"]]

        def analyze_index(index: int):
            member_started = time.monotonic()
            timeout = max(0.0, deadline - member_started)
            allocation = {**planning[index],
                "planning_allocated_seconds": planning[index]["allocated_seconds"],
                "allocated_seconds": timeout,
                "policy": "shared_portfolio_deadline_with_measured_cost_lpt_admission"}
            result = self._analyze_portfolio_member(
                submitted, candidate_sha256=after,
                sentinel=self.portfolio_sentinels[index], timeout_s=timeout,
                scope=scope, primary=index == 0,
                analyzer_override=self._member_analyzer())
            return (*result, allocation, time.monotonic() - member_started)

        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrency["admitted_workers"],
            thread_name_prefix="phase2-model")
        futures = {index: executor.submit(analyze_index, index)
                   for index in schedule["submission_order"]}
        try:
            for index in range(len(self.portfolio_sentinels)):
                future = futures[index]
                wait = max(0.01, deadline - time.monotonic() + 1.0)
                member_results[index] = future.result(timeout=wait)
        except Exception:
            for future in futures.values():
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        if any(row is None for row in member_results):
            raise RuntimeError("portfolio analysis did not return every deterministic member result")
        primary = member_results[0]
        assert primary is not None
        analysis, retained, completed_sandboxes, primary_allocation, primary_elapsed = primary
        self._check_inputs()
        analysis["optimization_baseline"] = copy.deepcopy(self.optimization_baseline_binding)
        dependencies_after = self._compiler_dependencies(submitted)
        if dependencies_before != dependencies_after:
            raise ValueError("compiler implementation dependencies changed during full-model analysis")
        if before != after or analysis.get("candidate_sha256") != after:
            raise ValueError("full-model analysis is not bound to the current candidate bytes")
        if analysis.get("workload", {}).get("capsule_sha256") != self.sentinel.capsule_sha256:
            raise ValueError("analysis substituted the frozen complete-model objective")
        primary_readiness = PAS.global_iteration_readiness(analysis)
        if (isinstance(completed_sandboxes, Mapping)
                and Path(completed_sandboxes["candidate"]["package_path"]).resolve() == submitted.resolve()):
            self._compiler_sandboxes[len(self.iterations)] = copy.deepcopy(completed_sandboxes)
            comparison_policy = completed_sandboxes.get("baseline")
            if (primary_readiness["status"] == "ready_for_probe_admission"
                    and isinstance(comparison_policy, Mapping)
                    and Path(comparison_policy["package_path"]).resolve()
                    == self.optimization_baseline.resolve()
                    and comparison_policy.get("compiler_dependencies")
                    == self.optimization_baseline_binding["compiler_dependencies"]):
                self._optimization_baseline_sandbox = copy.deepcopy(comparison_policy)
                self._optimization_baseline_sandbox_sha256 = PAS._document_sha256(comparison_policy)
        previous_members = {
            member["identity"]["capsule_sha256"]: member
            for member in ((self.iterations[-1].get("portfolio") or {}).get("members") or ())
        } if self.iterations else {}
        portfolio_rows: list[dict[str, Any]] = []
        for index, sentinel in enumerate(self.portfolio_sentinels[1:], start=1):
            member_result = member_results[index]
            assert member_result is not None
            member_analysis, member_artifacts, _, member_allocation, member_elapsed = member_result
            member_readiness = PAS.global_iteration_readiness(member_analysis)
            previous_member = previous_members.get(sentinel.capsule_sha256)
            member_comparison = self._compare_analyses(
                previous_member.get("analysis") if previous_member else None,
                member_analysis,
                previous_iteration=self.iterations[-1]["iteration"] if previous_member else None)
            portfolio_rows.append({
                "identity": sentinel_identity(sentinel, role="training"),
                "status": ("completed" if member_readiness["status"]
                           == "ready_for_probe_admission" else "failed"),
                "analysis": member_analysis, "readiness": member_readiness,
                "static_comparison": member_comparison,
                "analysis_allocation": member_allocation,
                "elapsed_seconds": member_elapsed,
                "timing_status": "UNMEASURED_FULL_MODEL",
            })
            baseline_artifacts = member_artifacts.pop("baseline_artifacts", None)
            if baseline_artifacts is not None:
                self._portfolio_baseline_artifacts[sentinel.capsule_sha256] = baseline_artifacts
        self._check_inputs()
        if (hash_tree(submitted)["sha256"] != after
                or self._compiler_dependencies(submitted) != dependencies_after):
            raise ValueError("submitted compiler bytes changed during portfolio analysis")
        readiness = copy.deepcopy(primary_readiness)
        portfolio_blockers = [
            f"portfolio:{member['identity']['capsule']}:{blocker}"
            for member in portfolio_rows for blocker in member["readiness"]["blockers"]
        ]
        if portfolio_blockers:
            readiness["status"] = "blocked"
            readiness["blockers"] = [*readiness["blockers"], *portfolio_blockers]
        readiness["portfolio_sha256"] = self.portfolio_identity_sha256
        readiness["portfolio_members_ready"] = sum(
            row["readiness"]["status"] == "ready_for_probe_admission"
            for row in ({"readiness": primary_readiness}, *portfolio_rows))
        readiness["portfolio_members_total"] = len(self.portfolio_sentinels)
        readiness["selection"] = "multi_model_pareto_without_invented_static_cycle_total"
        relative_semantics: Mapping[str, Any] = {"status": "unavailable_target_completion_contract",
                                               "numerical_equivalence": False}
        remaining = budget_seconds - (time.monotonic() - started)
        if self.completion_contract is not None and remaining > 0:
            from merlin.perf.completion_delta import qualify_relative_completion_delta
            relative_semantics = qualify_relative_completion_delta(
                previous_analysis=self.iterations[-1]["analysis"] if self.iterations else None,
                current_analysis=analysis, previous_artifacts=self._artifacts,
                current_artifacts=retained, contract=self.completion_contract,
                timeout_seconds=min(30, remaining))
        elapsed = time.monotonic() - started
        if elapsed > budget_seconds:
            readiness = {**readiness, "status": "blocked",
                         "blockers": [*readiness["blockers"], "iteration_wall_budget_exceeded"]}
        record = {
            "schema": "global_perf_iteration_v1", "iteration": len(self.iterations),
            "candidate_path": str(candidate.resolve()), "candidate_sha256": after,
            "submitted_snapshot": str(submitted.resolve()),
            "compiler_dependencies": dependencies_after,
            "baseline_sha256": self.baseline_sha256,
            "optimization_baseline_sha256": self.optimization_baseline_sha256,
            "optimization_baseline": copy.deepcopy(self.optimization_baseline_binding),
            "hypothesis": hypothesis, "analysis": analysis, "readiness": readiness,
            "historical_reference": copy.deepcopy(self.historical_reference),
            "elapsed_seconds": elapsed, "timing_status": "UNMEASURED_FULL_MODEL",
            "allocated_seconds": budget_seconds,
            "probe_receipts": [], "global_performance_claim": "unproven",
            "relative_semantic_evidence": relative_semantics,
        }
        record["static_comparison"] = self._compare(record)
        record["portfolio"] = {
            "schema": "full_model_portfolio_iteration_v1",
            "portfolio_sha256": self.portfolio_identity_sha256,
            "candidate_sha256": after,
            "members": [{
                "identity": sentinel_identity(self.sentinel, role="primary"),
                "status": ("completed" if primary_readiness["status"]
                           == "ready_for_probe_admission" else "failed"),
                "analysis_ref": "/analysis", "readiness": primary_readiness,
                "static_comparison_ref": "/static_comparison",
                "analysis_allocation": primary_allocation,
                "elapsed_seconds": primary_elapsed,
                "timing_status": "UNMEASURED_FULL_MODEL",
            }, *portfolio_rows],
            "members_ready": readiness["portfolio_members_ready"],
            "members_total": readiness["portfolio_members_total"],
            "selection": readiness["selection"],
            "analysis_allocation_policy": "shared_portfolio_deadline_with_measured_cost_lpt_admission",
            "analysis_concurrency": concurrency,
            "full_model_simulation_allowed": False,
        }
        self._write(f"iteration_{record['iteration']:04d}.json", record)
        self.iterations.append(record)
        self._previous_artifacts, self._artifacts = self._artifacts, retained
        if "baseline_artifacts" in retained:
            self._baseline_artifacts = retained.pop("baseline_artifacts")
        return copy.deepcopy(record)

    def _compare(self, current: Mapping[str, Any]) -> dict[str, Any]:
        """Compare full-model counters without inventing a total from partial accounting."""
        previous = self.iterations[-1] if self.iterations else None
        return self._compare_analyses(
            previous["analysis"] if previous else None, current["analysis"],
            previous_iteration=previous["iteration"] if previous else None,
            previous_feedback=previous.get("decision_feedback") if previous else None)

    @staticmethod
    def _compare_analyses(previous: Mapping[str, Any] | None,
                          current: Mapping[str, Any], *, previous_iteration: int | None,
                          previous_feedback: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Apply the same target-neutral structural comparison to every portfolio member."""
        def counters(analysis: Mapping[str, Any]) -> dict[str, int | float | None]:
            diag = analysis.get("diagnostics") or {}
            arm = (diag.get("arms") or {}).get("candidate") or {}
            movement = arm.get("movement") or {}
            plan = diag.get("verified_global_plan_emission") or {}
            return {
                "command_buffer_macs": arm.get("macs") if arm.get("exact") is True else None,
                "full_model_contraction_macs": ((diag.get("model_contraction_placement") or {})
                                                .get("candidate") or {}).get("total_contraction_macs"),
                "command_buffer_declared_movement_bytes": movement.get("exact_bytes")
                if isinstance(movement.get("exact_bytes"), (int, float))
                and not isinstance(movement.get("exact_bytes"), bool) else None,
                "dispatches": plan.get("emitted_dispatches", plan.get("tasks")),
            }
        left = counters(previous) if previous else {}
        right = counters(current)
        changes = {name: right[name] - left[name] for name in right
                   if isinstance(right[name], (int, float))
                   and isinstance(left.get(name), (int, float))}
        from merlin.perf.structural_delta import compare_full_model_structure
        return {
            "structural_change": compare_full_model_structure(previous, current)
            if previous else {"status": "initial_observation", "cycle_selection": "UNMEASURED"},
            "previous_iteration": previous_iteration,
            "candidate_totals": right, "candidate_minus_previous": changes,
            "unknown_metrics": [name for name, value in right.items() if value is None],
            "selection": "requires_global_cost_evidence",
            "previous_revision_mechanism_feedback": (
                {"applies_to_current_revision": False, "evidence": copy.deepcopy(previous_feedback)}
                if previous_feedback else None),
            "licence": "structural accounting only; fewer dispatches do not prove fewer cycles",
        }

    def _matching_current(self, candidate: Path, *, require_ready: bool) -> dict[str, Any]:
        """Return the exact analyzed revision, optionally requiring promotion readiness.

        A blocked analysis is still valuable authoring evidence: it binds the candidate bytes to
        every portfolio member and tells the next compiler round what remains unsupported.  It is
        never sufficient for probes, execution, or the promotable global-candidate seal.
        """
        self._check_inputs()
        self.validate_candidate_scope(candidate)
        if not self.iterations:
            raise ValueError("compile the complete-model graph before requesting a probe or sealing")
        digest = hash_tree(candidate)["sha256"]
        dependencies = self._compiler_dependencies(candidate)
        matching_bytes = [item for item in reversed(self.iterations)
                          if item["candidate_sha256"] == digest]
        if not matching_bytes:
            raise ValueError("candidate changed: recompile its full graph and global plan")
        row = next((item for item in matching_bytes
                    if item["compiler_dependencies"] == dependencies), None)
        if row is None:
            raise ValueError("shared compiler dependencies changed: recompile the full graph and plan")
        if require_ready and row["readiness"]["status"] != "ready_for_probe_admission":
            raise ValueError("global iteration is not ready: " + ", ".join(row["readiness"]["blockers"]))
        return row

    def _current(self, candidate: Path) -> dict[str, Any]:
        return self._matching_current(candidate, require_ready=True)

    def current_artifacts(self, candidate: Path) -> Mapping[str, Any]:
        """Host-only artifacts retained from exactly the current full-model invocation."""
        row = self._current(candidate)
        if (not self._artifacts or self._artifacts["candidate_sha256"] != row["candidate_sha256"]
                or self._artifacts["candidate_lowered_sha256"]
                != row["analysis"]["emission"]["candidate_lowered_sha256"]):
            raise ValueError("current analysis has no retained emitted artifacts")
        return self._artifacts

    def current_probe_binding(self, candidate: Path) -> ProbeBinding:
        row = self._current(candidate)
        diag = row["analysis"]["diagnostics"]
        return ProbeBinding(
            graph_digest=diag["captured_logical_graph"]["logical_dispatch_digest"],
            plan_digest=diag["verified_global_plan_emission"]["plan_digest"],
            compiler_digest=row["compiler_dependencies"]["compiler_implementation_sha256"],
            target_digest=self.target_sha256)

    def previous_artifacts(self, candidate: Path) -> Mapping[str, Any]:
        """Host-only preceding submitted artifact; never infer a previous revision from input IR."""
        self._current(candidate)
        if len(self.iterations) < 2 or not self._previous_artifacts:
            raise ValueError("changed-region qualification requires two submitted full-model revisions")
        previous = self.iterations[-2]
        if (self._previous_artifacts.get("candidate_sha256") != previous["candidate_sha256"]
                or self._previous_artifacts.get("candidate_lowered_sha256")
                != previous["analysis"]["emission"]["candidate_lowered_sha256"]):
            raise ValueError("previous full-model artifact identity does not match its analysis")
        return self._previous_artifacts

    def previous_probe_binding(self, candidate: Path) -> ProbeBinding:
        """Bind a paired diagnostic to the actual preceding verified submitted revision."""
        self.previous_artifacts(candidate)
        previous = self.iterations[-2]
        if previous["readiness"]["status"] != "ready_for_probe_admission":
            raise ValueError("paired diagnostic requires a verified preceding global plan")
        diag = previous["analysis"]["diagnostics"]
        return ProbeBinding(
            graph_digest=diag["captured_logical_graph"]["logical_dispatch_digest"],
            plan_digest=diag["verified_global_plan_emission"]["plan_digest"],
            compiler_digest=previous["compiler_dependencies"]["compiler_implementation_sha256"],
            target_digest=self.target_sha256)

    def optimization_baseline_artifacts(self, candidate: Path) -> Mapping[str, Any]:
        """Exact optimization-comparison bytes, not a preceding candidate or Phase-1 verdict.

        The baseline cache does not carry a verified global-plan receipt. This accessor
        verifies artifact identity only; a mechanism qualifier must independently check
        source/plan/implementation correspondence before accepting semantic evidence.
        """
        row = self._current(candidate)
        artifacts = self._baseline_artifacts
        expected = {"baseline_sha256": self.optimization_baseline_sha256,
                    "capsule_sha256": self.sentinel.capsule_sha256, "target": self.target}
        if not isinstance(artifacts, Mapping) or artifacts.get("identity") != expected:
            raise ValueError("no exact retained optimization-baseline artifacts")
        emission = row["analysis"]["emission"]
        for field, recorded, analyzed in (("lowered_text", "lowered_sha256", "baseline_lowered_sha256"),
                                         ("command_buffer_text", "command_buffer_sha256", "baseline_command_buffer_sha256")):
            text = artifacts.get(field)
            if (not isinstance(text, str) or not PAS._is_sha256(artifacts.get(recorded))
                    or PAS._sha256(text.encode()) != artifacts[recorded]
                    or artifacts[recorded] != emission.get(analyzed)):
                raise ValueError("retained optimization-baseline artifact bytes changed")
        interface = Path(self.current_artifacts(candidate)["interface"])
        source_root = Path(self.sentinel.frozen_source_path).resolve(strict=True)
        if (interface.is_symlink() or not interface.is_file()
                or not interface.resolve(strict=True).is_relative_to(source_root)):
            raise ValueError("optimization-baseline interface is outside the frozen objective")
        source_sha = PAS._sha256_file(interface)
        graph = row["analysis"]["diagnostics"]["captured_logical_graph"]
        if graph.get("source_sha256") != source_sha:
            raise ValueError("optimization-baseline source is not the analyzed full objective")
        buffer = json.loads(artifacts["command_buffer_text"])
        if not isinstance(buffer, dict) or buffer.get("declined") is not None:
            raise ValueError("optimization baseline did not emit a usable command buffer")
        # Decode from checked bytes, rather than exporting mutable/unbound cached trace objects.
        return {"arm": "optimization_baseline", "interface": interface,
                "source_sha256": source_sha, "lowered_text": artifacts["lowered_text"],
                "command_buffer_text": artifacts["command_buffer_text"], "command_buffer": buffer,
                "compiler_sha256": self.optimization_baseline_sha256,
                "lowered_sha256": artifacts["lowered_sha256"],
                "command_buffer_sha256": artifacts["command_buffer_sha256"],
                "structural_plan_status": "UNVERIFIED", "numerical_qualification": "UNPROVEN"}

    def optimization_baseline_artifact_binding(self, candidate: Path) -> Mapping[str, Any]:
        """Identity-only comparison binding; deliberately not a verified ``ProbeBinding``."""
        artifact = self.optimization_baseline_artifacts(candidate)
        row = self._current(candidate)
        return {"schema": "optimization_baseline_artifact_binding_v1", "arm": "optimization_baseline",
                "compiler_sha256": self.optimization_baseline_sha256,
                "compiler_dependencies": copy.deepcopy(self.optimization_baseline_binding["compiler_dependencies"]),
                "optimization_baseline_binding_sha256": self.optimization_baseline_binding_sha256,
                "capsule_sha256": self.sentinel.capsule_sha256, "source_sha256": artifact["source_sha256"],
                "logical_dispatch_digest": row["analysis"]["diagnostics"]["captured_logical_graph"]["logical_dispatch_digest"],
                "target_sha256": self.target_sha256, "lowered_sha256": artifact["lowered_sha256"],
                "command_buffer_sha256": artifact["command_buffer_sha256"],
                "structural_plan_status": "UNVERIFIED", "numerical_qualification": "UNPROVEN",
                "phase1_qualification_extended": False}

    def compile_probe_candidate(self, candidate: Path, interface: Path, scratch: Path,
                                *, timeout_s: float, emit_command_buffer: bool = False):
        """Compile a host-selected separate probe under the same answer-masked compiler policy."""
        return self._compile_probe_revision(candidate, interface, scratch, timeout_s=timeout_s,
                                            emit_command_buffer=emit_command_buffer, previous=False)

    def compile_previous_probe_candidate(self, candidate: Path, interface: Path, scratch: Path,
                                         *, timeout_s: float, emit_command_buffer: bool = False):
        """Compile the same witness using the exact preceding analyzed compiler and masks."""
        return self._compile_probe_revision(candidate, interface, scratch, timeout_s=timeout_s,
                                            emit_command_buffer=emit_command_buffer, previous=True)

    def compile_optimization_baseline_probe_candidate(self, candidate: Path, interface: Path, scratch: Path,
                                                      *, timeout_s: float, emit_command_buffer: bool = False):
        """Compile a separate host-selected probe with the actual optimization-comparison arm."""
        return self._compile_probe_revision(candidate, interface, scratch, timeout_s=timeout_s,
            emit_command_buffer=emit_command_buffer, previous=False, optimization_baseline=True)

    def _compile_probe_revision(self, candidate: Path, interface: Path, scratch: Path,
                                *, timeout_s: float, emit_command_buffer: bool, previous: bool,
                                optimization_baseline: bool = False):
        from merlin.perf.analysis_worker import IsolatedAnalysisWorker, run_sandboxed_entrypoint, _read_output
        from merlin.targetgen import oot_runner
        started = time.monotonic()
        self._current(candidate)
        if not isinstance(self.analyzer, IsolatedAnalysisWorker):
            raise ValueError("short candidate compilation requires the production isolated compiler policy")
        if optimization_baseline:
            import math
            if not math.isfinite(timeout_s) or not 0 < timeout_s <= ITERATION_MAX_SECONDS:
                raise ValueError("optimization-baseline probe needs a finite bounded timeout")
            timeout_s = min(timeout_s, self.timeout_s - self._current(candidate)["elapsed_seconds"])
            if timeout_s <= 0:
                raise TimeoutError("optimization-baseline probe exhausted its iteration budget")
        if previous and optimization_baseline:
            raise ValueError("preceding candidate and optimization baseline are distinct arms")
        binding = (self.optimization_baseline_artifact_binding(candidate) if optimization_baseline else
                   self.previous_probe_binding(candidate) if previous else self.current_probe_binding(candidate))
        selected = self.iterations[-2] if previous else self.iterations[-1]

        def check_revision():
            self._current(candidate)
            if optimization_baseline:
                if self.optimization_baseline_artifact_binding(candidate) != binding:
                    raise ValueError("optimization-baseline witness binding changed")
                return
            actual = self.previous_probe_binding(candidate) if previous else self.current_probe_binding(candidate)
            submitted = Path(selected["submitted_snapshot"])
            if (actual != binding or hash_tree(submitted)["sha256"] != selected["candidate_sha256"]
                    or self._compiler_dependencies(submitted) != selected["compiler_dependencies"]):
                raise ValueError("selected short-witness compiler identity changed")

        check_revision()
        scratch.mkdir()
        source = scratch / "interface.mlir"
        source.write_bytes(interface.read_bytes())
        sandbox = (self._probe_sandbox(candidate, scratch, optimization_baseline=True) if optimization_baseline
                   else self._probe_sandbox(candidate, scratch, previous=previous))
        package = oot_runner.load_package(Path(sandbox["package_path"]))
        lowered = run_sandboxed_entrypoint(
            package, "lower_target_to_llvm", source,
            sandbox=sandbox, timeout_s=timeout_s - (time.monotonic() - started))
        if not emit_command_buffer:
            check_revision()
            return lowered
        buffer_result = None
        command_buffer = None
        if lowered.returncode == 0:
            buffer_path = scratch / "command_buffer.json"
            buffer_result = run_sandboxed_entrypoint(
                package, "emit_command_buffer", source, buffer_path,
                sandbox=sandbox, timeout_s=timeout_s - (time.monotonic() - started))
            if buffer_result.returncode == 0:
                command_buffer = json.loads(_read_output(buffer_path))
                if not isinstance(command_buffer, dict):
                    raise ValueError("probe command buffer must be a JSON object")
        check_revision()
        return {"lowered": lowered, "command_buffer_emission": buffer_result,
                "command_buffer": command_buffer}

    def _probe_sandbox(self, candidate: Path, scratch: Path, *, previous: bool = False,
                       optimization_baseline: bool = False) -> Mapping[str, Any]:
        """Reuse the successful analysis policy for its immutable compiler, plus public scratch."""
        row = self._current(candidate)
        if previous and optimization_baseline:
            raise ValueError("preceding candidate and optimization baseline are distinct arms")
        if optimization_baseline:
            self.optimization_baseline_artifact_binding(candidate)
            sandbox = self._optimization_baseline_sandbox
            if (not isinstance(sandbox, Mapping) or self._optimization_baseline_sandbox_sha256 is None
                    or PAS._document_sha256(sandbox) != self._optimization_baseline_sandbox_sha256):
                raise ValueError("optimization baseline has no intact recorded answer-masked policy")
            expected_package = self.optimization_baseline
            expected_dependencies = self.optimization_baseline_binding["compiler_dependencies"]
        elif previous:
            self.previous_probe_binding(candidate)
            row = self.iterations[-2]
            completed = self._compiler_sandboxes.get(row["iteration"])
            if completed is None:
                raise ValueError("preceding compiler has no retained successful answer-masked policy")
        else:
            completed = getattr(self.analyzer, "completed_sandboxes", None)
        if not optimization_baseline:
            if completed is None:
                return self.analyzer.sandbox_factory(self.optimization_baseline, candidate, scratch)["candidate"]
            sandbox = completed["candidate"]
            expected_package = Path(row["submitted_snapshot"])
            expected_dependencies = row["compiler_dependencies"]
        if (Path(sandbox["package_path"]).resolve() != expected_package.resolve()
                or sandbox.get("compiler_dependencies") != expected_dependencies):
            raise ValueError("prepared sandbox is not bound to the current immutable compiler")
        directory = scratch.resolve(strict=True)
        if scratch.is_symlink() or not directory.is_dir():
            raise ValueError("probe scratch must be a real dedicated directory")
        for surface in sandbox["answer_surfaces"]:
            answer = Path(surface["path"]).resolve()
            if (directory == answer or directory in answer.parents
                    or surface["kind"] == "dir" and answer in directory.parents):
                raise ValueError("new probe scratch would expose a masked answer surface")
        prefix = list(sandbox["command_prefix"])
        boundary = sandbox["bwrap_argv_length"]
        if not isinstance(boundary, int) or not 0 < boundary < len(prefix):
            raise ValueError("prepared policy has no exact bwrap/payload boundary")
        return {**sandbox, "command_prefix": [*prefix[:boundary], "--bind", str(directory),
                                               str(directory), *prefix[boundary:]],
                "reuse_scope": "exact analyzed compiler and existing answer masks; public scratch only"}

    def native_probe_policy(self, candidate: Path, scratch: Path) -> dict[str, Any]:
        """Host-provider-only copy of the exact existing policy; this grants nothing new."""
        from merlin.perf.analysis_worker import IsolatedAnalysisWorker
        row = self._current(candidate)
        if not isinstance(self.analyzer, IsolatedAnalysisWorker):
            raise ValueError("native probes require the production isolated compiler policy")
        scratch = scratch.resolve(strict=True)
        if not scratch.is_dir():
            raise ValueError("native probe scratch must be an existing dedicated directory")
        cache = getattr(self, "_native_probe_policies", None)
        if cache is None:
            self._native_probe_policies = cache = {}
        key = (row["compiler_dependencies"]["compiler_implementation_sha256"], str(candidate.resolve()), str(scratch))
        if key not in cache:
            cache[key] = self._probe_sandbox(candidate, scratch)
        return copy.deepcopy(cache[key])

    def run_native_probe(self, candidate: Path, scratch: Path, argv: Sequence[str],
                         *, timeout_s: float, _build_dependencies=None, _execution_dependencies=None):
        """Run a host-built changed-region witness inside the existing answer-free policy.

        The host provider selects the runner and owns its reference comparison. Native candidate
        instructions never execute in the host interpreter or inherit access to host answers.
        """
        import math
        import subprocess
        from merlin.perf.analysis_worker import IsolatedAnalysisWorker, _kill_group
        started = time.monotonic()
        row = self._current(candidate)
        if not isinstance(self.analyzer, IsolatedAnalysisWorker):
            raise ValueError("native probes require the production isolated compiler policy")
        if not math.isfinite(timeout_s) or timeout_s <= 0 or not argv:
            raise ValueError("native probes require a positive finite deadline and an argv")
        if _execution_dependencies is not None and (_build_dependencies is not None or timeout_s > 60):
            raise ValueError("raw-engine runtime requires a separate action bounded to 60 seconds")
        scratch = scratch.resolve(strict=True)
        if not scratch.is_dir():
            raise ValueError("native probe scratch must be an existing dedicated directory")
        sandbox = self.native_probe_policy(candidate, scratch)
        import contextlib
        import tempfile
        with contextlib.ExitStack() as scope:
            prefix = list(sandbox["command_prefix"])
            if _build_dependencies is not None:
                from merlin.targetgen.sandbox.build_dependencies import HostBuildDependencies
                if type(_build_dependencies) is not HostBuildDependencies:
                    raise ValueError("build grants require an exact host-owned dependency capability")
                # This fresh source overlay is private to the host and never
                # cached or served writable to a candidate/native invocation.
                overlay = Path(scope.enter_context(tempfile.TemporaryDirectory(prefix="merlin_build_sources_")))
                prefix = _build_dependencies.extend(sandbox, argv, overlay_root=overlay,
                                                     overlay_builder=compiler_dependency_mounts)
            if _execution_dependencies is not None:
                from merlin.targetgen.sandbox.executable_dependencies import HostExecutableDependencies
                if type(_execution_dependencies) is not HostExecutableDependencies:
                    raise ValueError("runtime grants require an exact trusted executable capability")
                prefix = _execution_dependencies.extend(sandbox, argv)
            if not prefix or Path(prefix[0]).name != "bwrap" or "--clearenv" not in prefix:
                raise ValueError("native probe requires the existing clear-environment bwrap policy")
            remaining = min(timeout_s, self.timeout_s - row["elapsed_seconds"]) - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("native probe iteration wall-clock budget exhausted")
            command = [*prefix, *map(str, argv)]
            process = subprocess.Popen(command, cwd=str(scratch), stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, start_new_session=True)
            try:
                stdout, stderr = process.communicate(timeout=remaining)
                result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
            finally:
                _kill_group(process.pid)
                process.wait()
                if _build_dependencies is not None:
                    _build_dependencies.revalidate(argv)
                if _execution_dependencies is not None:
                    _execution_dependencies.revalidate(argv)
        self._current(candidate)
        return result

    def charge_probe_preparation(self, candidate: Path, elapsed_seconds: float) -> None:
        import math
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("probe preparation time must be finite and nonnegative")
        row = self.iterations[-1]
        row["elapsed_seconds"] += elapsed_seconds
        receipt = {"schema": "global_probe_preparation_v1", "iteration": row["iteration"],
                   "candidate_sha256": row["candidate_sha256"], "elapsed_seconds": elapsed_seconds}
        self._write(f"probe_preparation_{time.time_ns()}.json", receipt)
        self._current(candidate)

    def prepare_source_convolution(self, candidate: Path, *, comparison_arm: str,
                                   timeout_s: float) -> dict[str, Any]:
        """Broker preparation feedback only: never a semantic pass or simulator admission."""
        import math
        from merlin.perf.source_convolution_preparation import prepare_source_convolution
        if comparison_arm not in {"optimization_baseline", "previous"}:
            raise ValueError("comparison_arm must explicitly select optimization_baseline or previous")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("source-convolution preparation needs a finite positive budget")
        started = time.monotonic()
        row = self._current(candidate)
        binding = self.current_probe_binding(candidate)
        budget = min(60.0, float(timeout_s), self.timeout_s-row["elapsed_seconds"])
        try:
            descriptor = PAS._mapping_file(Path(self.sentinel.frozen_source_path)/"capsule.yaml", yaml_file=True)
            entry = descriptor.get("entry")
            if not isinstance(entry, str) or not entry:
                raise ValueError("frozen objective descriptor has no explicit source entry")
            remaining = budget-(time.monotonic()-started)
            if remaining <= 0:
                raise TimeoutError("source-convolution preparation exhausted remaining iteration budget")
            evidence = prepare_source_convolution(candidate=candidate, experiment=self,
                comparison_arm=comparison_arm, entry=entry, output=self.output/"source_convolution_preparations",
                timeout_s=remaining)
            if self.current_probe_binding(candidate) != binding:
                raise ValueError("compiler/source/target changed during source-convolution preparation")
            allowed = []
            if self.edit_contract is not None:
                inventory = (self.edit_guidance_inventory.to_dict()
                             if self.edit_guidance_inventory is not None else
                             PAS.inspect_compiler_package(self.edit_scope_seed).to_dict())
                surfaces = {row["id"]: row for row in inventory.get("surfaces", [])}
                effects = {"placement", "movement", "layout", "encoding", "issue", "tiling", "dtype", "quantization"}
                for owner in self.edit_contract["existing_symbols"]:
                    surface = surfaces.get(owner["surface_id"], {})
                    if (surface.get("path") == owner["path"] and surface.get("symbol") == owner["symbol"]
                            and effects.intersection(surface.get("effects", []))):
                        allowed.append({**owner, "effects": surface["effects"],
                                        "matching_effects": sorted(effects.intersection(surface["effects"]))})
            elapsed = time.monotonic()-started
            if elapsed > budget:
                raise TimeoutError("source-convolution preparation exceeded shared action budget")
            receipt = {"schema": "global_source_convolution_preparation_receipt_v1",
                "iteration": row["iteration"], "binding": binding.to_dict(), "comparison_arm": comparison_arm,
                "host_verifier_policy_sha256": self.host_policy["sha256"],
                "status": "runtime_pending" if evidence.get("status") == "prepared" else "UNKNOWN",
                "evidence": evidence, "elapsed_seconds": elapsed, "numerical_pass": False,
                "runtime_admitted": False, "runtime_recipe_binding": "NOT_PREPARED",
                "full_model_numerics_qualified": False, "global_speedup_proven": False,
                "allowed_edit_surfaces": allowed,
                "edit_authority_sha256": (self.edit_scope_binding["contract_document_sha256"]
                                           if self.edit_contract is not None else None),
                "surface_scope": "effect-matched subset of host-frozen AST authority; not proof of compiler reachability",
                "remaining_obligations": ["verify emitted target instruction mechanism", "bind complete-program runtime recipe/tools/ELF",
                    "admit both complete reduced programs within remaining 60-second runtime action", "warm1/measured1 independent output checks"],
                "next_action": "Inspect prepared source/route or missing proof; preparation is not qualify-changed-region success"}
            path = self._write(f"source_convolution_preparation_{row['iteration']:04d}_{time.time_ns()}.json", receipt)
            row.setdefault("preparation_receipts", []).append({"path": str(path), "sha256": PAS._sha256_file(path)})
            return receipt
        finally:
            self.charge_probe_preparation(candidate, time.monotonic()-started)

    def prepare_source_contraction(self, candidate: Path, *, comparison_arm: str, source_op_index: int,
                                   max_m: int, max_n: int, max_k: int, timeout_s: float) -> dict[str, Any]:
        """Prepare a selected current-source pair, without granting runtime or semantic authority."""
        import math
        from merlin.perf.source_contraction_preparation import prepare_source_contraction
        if (comparison_arm not in {"optimization_baseline", "previous"}
                or type(source_op_index) is not int or source_op_index < 0
                or any(type(value) is not int or not 1 <= value <= 4096 for value in (max_m, max_n, max_k))
                or isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0):
            raise ValueError("invalid explicit source-contraction selection or budget")
        started = time.monotonic()
        row = self._current(candidate)
        binding = self.current_probe_binding(candidate)
        budget = min(60.0, float(timeout_s), self.timeout_s-row["elapsed_seconds"])
        try:
            descriptor = PAS._mapping_file(Path(self.sentinel.frozen_source_path)/"capsule.yaml", yaml_file=True)
            entry = descriptor.get("entry")
            if entry is None:
                # Older frozen descriptors omit entry. Select only a UNIQUE
                # function from their already hash-bound source, never a guessed name.
                from merlin.frontends.linalg_mlir import parse_mlir_text
                artifact = self.current_artifacts(candidate)
                source = Path(artifact["interface"]).read_text()
                source_sha = row["analysis"]["diagnostics"]["captured_logical_graph"].get("source_sha256")
                if len(source.encode()) > 2_000_000 or PAS._sha256(source.encode()) != source_sha:
                    raise ValueError("entry inference requires exact bounded frozen source bytes")
                module = parse_mlir_text(source)
                functions = [op for op in module.body.block.ops if op.name == "func.func"]
                if len(functions) != 1:
                    raise ValueError("ambiguous frozen source requires an explicit host entry")
                entry = functions[0].sym_name.data
            if not isinstance(entry, str) or not entry:
                raise ValueError("frozen objective has no valid source entry")
            remaining = budget-(time.monotonic()-started)
            if remaining <= 0:
                raise TimeoutError("source contraction preparation exhausted its action budget")
            evidence = prepare_source_contraction(candidate=candidate, experiment=self,
                comparison_arm=comparison_arm, source_op_index=source_op_index, entry=entry,
                max_m=max_m, max_n=max_n, max_k=max_k, output=self.output/"source_contraction_preparations",
                timeout_s=remaining)
            if self.current_probe_binding(candidate) != binding or time.monotonic()-started > budget:
                raise ValueError("preparation binding changed or shared deadline expired")
            receipt = {"schema": "global_source_contraction_preparation_receipt_v1",
                "iteration": row["iteration"], "binding": binding.to_dict(), "comparison_arm": comparison_arm,
                "host_verifier_policy_sha256": self.host_policy["sha256"], "evidence": evidence,
                "status": "runtime_pending" if evidence.get("status") == "prepared" else "UNKNOWN",
                "elapsed_seconds": time.monotonic()-started, "numerical_pass": False, "runtime_admitted": False,
                "full_model_numerics_qualified": False, "global_speedup_proven": False,
                "emitted_route_correspondence": evidence.get("emitted_route_correspondence", "UNKNOWN"),
                "task_route_feedback": copy.deepcopy(evidence.get("task_route_feedback") or {
                    "status": "UNKNOWN", "scope": "no bound full/short task route comparison available",
                    "timing_calibration_admissible": False}),
                "next_action": "qualify-source-contraction with this exact preparation_sha256, if host provider available"}
            path = self._write(f"source_contraction_preparation_{row['iteration']:04d}_{time.time_ns()}.json", receipt)
            reference = {"path": str(path), "sha256": PAS._sha256_file(path)}
            row.setdefault("source_contraction_preparation_receipts", []).append(reference)
            return {**receipt, "preparation_sha256": reference["sha256"]}
        finally:
            self.charge_probe_preparation(candidate, time.monotonic()-started)

    def qualify_source_contraction(self, candidate: Path, *, preparation_sha256: str,
                                   provider: Callable[..., Mapping[str, Any]], timeout_s: float) -> dict[str, Any]:
        """Run only a current hash-bound host preparation; never accept an agent-selected path."""
        import math
        if (not isinstance(preparation_sha256, str) or len(preparation_sha256) != 64
                or any(c not in "0123456789abcdef" for c in preparation_sha256)
                or isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0):
            raise ValueError("invalid source-pair preparation identity or budget")
        started = time.monotonic()
        row = self._current(candidate)
        binding = self.current_probe_binding(candidate)
        budget = min(60.0, float(timeout_s), self.timeout_s-row["elapsed_seconds"])
        try:
            references = [ref for ref in row.get("source_contraction_preparation_receipts", [])
                          if ref["sha256"] == preparation_sha256]
            if len(references) != 1:
                raise ValueError("preparation is not a unique current-iteration host receipt")
            path = Path(references[0]["path"])
            if path.is_symlink() or PAS._sha256_file(path) != preparation_sha256:
                raise ValueError("prepared source-pair receipt changed")
            preparation = json.loads(path.read_text())
            if (preparation.get("schema") != "global_source_contraction_preparation_receipt_v1"
                    or preparation.get("binding") != binding.to_dict()
                    or preparation.get("host_verifier_policy_sha256") != self.host_policy["sha256"]
                    or preparation.get("status") != "runtime_pending"):
                raise ValueError("prepared source pair is stale, unavailable, or from another host policy")
            remaining = budget-(time.monotonic()-started)
            if remaining <= 0:
                raise TimeoutError("source-pair action exhausted its iteration budget")
            evidence = dict(provider(candidate=candidate, experiment=self,
                                     prepared=preparation["evidence"], timeout_s=remaining))
            if (self.current_probe_binding(candidate) != binding or PAS._sha256_file(path) != preparation_sha256
                    or time.monotonic()-started > budget):
                raise ValueError("source-pair binding changed or shared deadline expired")
            receipt = {"schema": "global_source_contraction_execution_receipt_v1", "iteration": row["iteration"],
                "binding": binding.to_dict(), "preparation_sha256": preparation_sha256,
                "host_verifier_policy_sha256": self.host_policy["sha256"], "evidence": evidence,
                "elapsed_seconds": time.monotonic()-started, "full_model_numerics_qualified": False,
                "global_speedup_proven": False,
                "task_route_feedback": copy.deepcopy(evidence.get("task_route_feedback") or {
                    "status": "UNKNOWN", "scope": "no bound full/short task route comparison available",
                    "timing_calibration_admissible": False}),
                "scope": "complete reduced source pair only; changed full-model task relevance requires separate proof"}
            result_path = self._write(f"source_contraction_execution_{row['iteration']:04d}_{time.time_ns()}.json", receipt)
            row.setdefault("source_pair_receipts", []).append(
                {"path": str(result_path), "sha256": PAS._sha256_file(result_path)})
            return receipt
        finally:
            self.charge_probe_preparation(candidate, time.monotonic()-started)

    def qualify_changed_region(self, candidate: Path, *, provider: Callable[..., Mapping[str, Any]],
                               timeout_s: float) -> dict[str, Any]:
        """Record a host-selected semantic mechanism witness without upgrading model timing."""
        row = self._current(candidate)
        binding = self.current_probe_binding(candidate)
        previous = self.previous_artifacts(candidate)
        current = self.current_artifacts(candidate)
        started = time.monotonic()
        remaining = min(float(timeout_s), self.timeout_s - row["elapsed_seconds"])
        if remaining <= 0:
            raise TimeoutError("changed-region qualification exhausted its iteration budget")
        try:
            evidence = dict(provider(candidate=candidate, experiment=self, timeout_s=remaining))
            if self.current_probe_binding(candidate) != binding:
                raise ValueError("compiler or global plan changed during semantic qualification")
            elapsed = time.monotonic() - started
            if elapsed > remaining:
                raise TimeoutError("changed-region semantic qualification exceeded its wall budget")
            receipt = {
                "schema": "global_changed_region_semantic_receipt_v1",
                "iteration": row["iteration"], "binding": binding.to_dict(),
                "previous_artifact_sha256": previous["candidate_lowered_sha256"],
                "current_artifact_sha256": current["candidate_lowered_sha256"],
                "evidence": evidence, "elapsed_seconds": elapsed,
                "scope": "selected changed mechanism and tested reduced domain only",
                "full_model_numerics_qualified": False, "global_speedup_proven": False,
                "full_model_cycles": None,
            }
            path = self._write(f"semantic_{row['iteration']:04d}_{time.time_ns()}.json", receipt)
            row.setdefault("semantic_receipts", []).append(
                {"path": str(path), "sha256": PAS._sha256_file(path)})
            return receipt
        finally:
            # Charge failures too; the provider never charges these same seconds a second time.
            self.charge_probe_preparation(candidate, time.monotonic() - started)

    def profile_controlled_context(self, candidate: Path, *, provider: Callable[..., Mapping[str, Any]],
                                   timeout_s: float) -> dict[str, Any]:
        """Time one bounded actual source prefix, without claiming containing-model equivalence."""
        started = time.monotonic()
        row = self._current(candidate)
        binding = self.current_probe_binding(candidate)
        captured = self.current_artifacts(candidate)
        budget = min(60.0, float(timeout_s), self.timeout_s - row["elapsed_seconds"])

        def remaining() -> float:
            value = budget - (time.monotonic() - started)
            if value <= 0:
                raise TimeoutError("controlled prefix preparation plus warm/measured execution exceeded 60s budget")
            return value

        try:
            action = provider(candidate=candidate, experiment=self, timeout_s=remaining())
            inputs = action["controlled_context_inputs"]
            if (inputs["binding"] != binding or inputs["scope"] != "controlled_source_prefix"
                    or inputs["model_artifact_sha256"] != captured["candidate_lowered_sha256"]):
                raise ValueError("controlled source prefix is not bound to the current model and compiler")
            source, prepared = inputs["source_slice"], inputs["prepared"]
            indices = source["timed_source_instruction_indices"]
            if (source["source_artifact_sha256"] != captured["candidate_lowered_sha256"]
                    or not 0 < len(indices) <= 32 or len(indices) != len(set(indices))
                    or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices)
                    or len(source["source_instruction_indices"]) >= len(captured["decoded_trace"]["instructions"])
                    or source.get("full_model_executed") is not False
                    or source.get("full_layer_executed") is not False):
                raise ValueError("controlled prefix must be a separate bounded subset of emitted model instructions")
            if (prepared["source_artifact_sha256"] != source["slice_source_sha256"]
                    or prepared["measurement_scope"] != "controlled_source_prefix"
                    or prepared.get("simulator_executed") is not False
                    or not 0 < prepared["timed_instruction_count"] <= 32
                    or not 0 < prepared["host_input_bytes"] <= 65536
                    or not 0 < prepared["output_storage_bytes"] <= 65536
                    or source["source_task_compute_pair_count"] <= 1
                    or source["executed_compute_pair_count"] != 1):
                raise ValueError("controlled executable does not match the extracted source prefix")
            elf = Path(prepared["workdir"]) / "primitive.elf"
            if PAS._sha256_file(elf) != prepared["elf_sha256"]:
                raise ValueError("controlled prepared executable changed")
            observed = dict(action["execute"](timeout_s=remaining()))
            for key in ("elf_sha256", "wrapper_sha256", "primitive_mlir_sha256", "domain_digest",
                        "source_artifact_sha256", "measurement_scope"):
                if observed.get(key) != prepared[key]:
                    raise ValueError(f"controlled execution changed its admitted {key}")
            if (observed.get("correct") is not True or observed.get("warmup_runs") != 1
                    or observed.get("measured_runs") != 1 or observed.get("full_model_executed") is not False
                    or observed.get("full_source_probe_executed") is not False
                    or not isinstance(observed.get("total_compute_cycles"), int)
                    or observed["total_compute_cycles"] <= 0):
                raise ValueError("controlled execution lacks a correct warm1/measured1 prefix receipt")
            if self.current_probe_binding(candidate) != binding or PAS._sha256_file(elf) != prepared["elf_sha256"]:
                raise ValueError("compiler or controlled executable changed during measurement")
            remaining()
            receipt = {"schema": "global_controlled_context_receipt_v1", "binding": binding.to_dict(),
                       "iteration": row["iteration"], "source_slice": source, "execution": observed,
                       "model_artifact_sha256": captured["candidate_lowered_sha256"],
                       "scope": "controlled_source_prefix", "elapsed_seconds": time.monotonic()-started,
                       "full_model_cycles": None, "global_cost_validated": False,
                       "calibration_admissible": False, "global_speedup_proven": False}
            path = self._write(f"context_{row['iteration']:04d}_{time.time_ns()}.json", receipt)
            row.setdefault("context_receipts", []).append({"path": str(path), "sha256": PAS._sha256_file(path)})
            return receipt
        finally:
            self.charge_probe_preparation(candidate, time.monotonic()-started)

    def compare_controlled_context(self, candidate: Path, *, provider: Callable[..., Mapping[str, Any]],
                                   timeout_s: float) -> dict[str, Any]:
        """Compare identical bounded work in two schedules, never projected model latency."""
        started = time.monotonic()
        row = self._current(candidate)
        bindings = {"before": self.previous_probe_binding(candidate),
                    "after": self.current_probe_binding(candidate)}
        artifacts = {"before": self.previous_artifacts(candidate),
                     "after": self.current_artifacts(candidate)}
        if bindings["before"].graph_digest != bindings["after"].graph_digest:
            raise ValueError("paired context requires the same complete logical graph")
        budget = min(60.0, float(timeout_s), self.timeout_s - row["elapsed_seconds"])

        def remaining() -> float:
            value = budget - (time.monotonic() - started)
            if value <= 0:
                raise TimeoutError("paired context preparation and both executions exceeded their total budget")
            return value

        def positive_int(value: Any, limit: int) -> bool:
            return type(value) is int and 0 < value <= limit

        try:
            action = provider(candidate=candidate, experiment=self, timeout_s=remaining())
            inputs = action["paired_context_inputs"]
            proof = inputs["projection_proof"]
            contract = inputs["work_contract_sha256"]
            input_contract = inputs["deterministic_input_contract_sha256"]
            if (proof.get("schema") != "controlled_fixed_work_projection_v1"
                    or proof.get("status") != "same_work_projection_verified"
                    or proof.get("scope") != "controlled_fixed_work_slice"
                    or proof.get("future_computes_omitted_symmetrically") is not True
                    or proof.get("global_cost_validated") is not False
                    or proof.get("work_contract_sha256") != contract
                    or PAS._document_sha256(proof["work_contract"]) != contract
                    or not isinstance(input_contract, str) or len(input_contract) != 64
                    or PAS._document_sha256(inputs["deterministic_input_contract"]) != input_contract):
                raise ValueError("paired context lacks a hash-bound same-work projection")
            executable_paths: dict[str, Path] = {}
            for arm in ("before", "after"):
                raw_sha = artifacts[arm]["candidate_lowered_sha256"]
                entry = inputs["arms"][arm]
                source, prepared = entry["source_slice"], entry["prepared"]
                indices = source["timed_source_instruction_indices"]
                if (inputs[arm + "_binding"] != bindings[arm]
                        or inputs[arm + "_model_artifact_sha256"] != raw_sha
                        or proof[arm + "_artifact_sha256"] != raw_sha
                        or proof[arm + "_timed_indices"] != indices
                        or source["source_artifact_sha256"] != raw_sha
                        or source["work_contract_sha256"] != contract
                        or entry["work_contract_sha256"] != contract
                        or entry["deterministic_input_contract_sha256"] != input_contract
                        or not positive_int(len(indices), 32) or len(set(indices)) != len(indices)
                        or any(type(i) is not int or i < 0 for i in indices)
                        or max(indices) >= len(artifacts[arm]["decoded_trace"]["instructions"])
                        or len(source["source_instruction_indices"]) >= len(artifacts[arm]["decoded_trace"]["instructions"])
                        or source.get("full_model_executed") is not False
                        or source.get("full_layer_executed") is not False
                        or source["source_task_compute_pair_count"] <= 1
                        or source["executed_compute_pair_count"] != 1):
                    raise ValueError("paired context changed source identity, work, or extraction domain")
                if (prepared["source_artifact_sha256"] != source["slice_source_sha256"]
                        or prepared["measurement_scope"] != "controlled_fixed_work_slice"
                        or prepared.get("simulator_executed") is not False
                        or not positive_int(prepared["timed_instruction_count"], 32)
                        or not positive_int(prepared["host_input_bytes"], 65536)
                        or not positive_int(prepared["output_storage_bytes"], 65536)):
                    raise ValueError("paired context executable is not a bounded fixed-work slice")
                elf = Path(prepared["workdir"]) / "primitive.elf"
                if PAS._sha256_file(elf) != prepared["elf_sha256"]:
                    raise ValueError("paired context prepared executable changed")
                executable_paths[arm] = elf
            executions = {}
            for arm in ("before", "after"):
                prepared = inputs["arms"][arm]["prepared"]
                observed = dict(action["execute"](arm=arm, timeout_s=remaining()))
                for key in ("elf_sha256", "wrapper_sha256", "primitive_mlir_sha256", "domain_digest",
                            "source_artifact_sha256", "measurement_scope"):
                    if observed.get(key) != prepared[key]:
                        raise ValueError(f"paired execution changed its admitted {arm} {key}")
                if (observed.get("correct") is not True or observed.get("warmup_runs") != 1
                        or observed.get("measured_runs") != 1
                        or observed.get("full_model_executed") is not False
                        or observed.get("full_source_probe_executed") is not False
                        or not positive_int(observed.get("total_compute_cycles"), 2**63-1)):
                    raise ValueError("paired context requires correct warm1/measured1 evidence per arm")
                executions[arm] = observed
            if (self.current_probe_binding(candidate) != bindings["after"]
                    or self.previous_probe_binding(candidate) != bindings["before"]):
                raise ValueError("paired context compiler or graph binding changed")
            for arm, elf in executable_paths.items():
                if PAS._sha256_file(elf) != inputs["arms"][arm]["prepared"]["elf_sha256"]:
                    raise ValueError("paired context executable changed during measurement")
            remaining()
            cycles = {arm: item["total_compute_cycles"] for arm, item in executions.items()}
            receipt = {"schema": "global_paired_controlled_context_receipt_v1",
                "iteration": row["iteration"], "binding": bindings["after"].to_dict(),
                "previous_binding": bindings["before"].to_dict(),
                "model_artifact_sha256": artifacts["after"]["candidate_lowered_sha256"],
                "previous_model_artifact_sha256": artifacts["before"]["candidate_lowered_sha256"],
                "scope": "controlled_fixed_work_slice", "projection_proof": proof,
                "previous_iteration_record": str((self.output / f"iteration_{row['iteration']-1:04d}.json").resolve()),
                "previous_iteration_record_sha256": PAS._sha256_file(
                    self.output / f"iteration_{row['iteration']-1:04d}.json"),
                "deterministic_input_contract": inputs["deterministic_input_contract"],
                "deterministic_input_contract_sha256": input_contract, "executions": executions,
                "cycles": cycles, "after_minus_before_cycles": cycles["after"]-cycles["before"],
                "elapsed_seconds": time.monotonic()-started, "full_model_cycles": None,
                "global_cost_validated": False, "global_speedup_proven": False,
                "calibration_admissible": False, "statistical_confirmation": "not_performed"}
            feedback = paired_context_decision_feedback(row, receipt, target_sha256=self.target_sha256)
            receipt["decision_feedback"] = feedback
            path = self._write(f"paired_context_{row['iteration']:04d}_{time.time_ns()}.json", receipt)
            row.setdefault("paired_context_receipts", []).append(
                {"path": str(path), "sha256": PAS._sha256_file(path)})
            row["decision_feedback"] = {**feedback, "receipt": {
                "path": str(path), "sha256": PAS._sha256_file(path)}}
            return receipt
        finally:
            self.charge_probe_preparation(candidate, time.monotonic()-started)

    def measure_probe(self, candidate: Path, *, admission_inputs: Mapping[str, Any],
                      execute: Callable[..., ProbeObservation], timeout_s: int | None = None
                      ) -> dict[str, Any]:
        """Execute only an independently extracted, equivalent short mechanism after admission."""
        row = self._current(candidate)
        diag = row["analysis"]["diagnostics"]
        plan = diag["verified_global_plan_emission"]
        binding = ProbeBinding(
            graph_digest=diag["captured_logical_graph"]["logical_dispatch_digest"],
            plan_digest=plan["plan_digest"],
            compiler_digest=row["compiler_dependencies"]["compiler_implementation_sha256"],
            target_digest=self.target_sha256)
        inputs = dict(admission_inputs)
        if "current_binding" in inputs:
            raise ValueError("the host derives current probe bindings; callers cannot replace them")
        if inputs["model"].artifact_digest != row["analysis"]["emission"]["candidate_lowered_sha256"]:
            raise ValueError("model mechanism evidence names a different compiled artifact")
        admitted = require_probe_admission(current_binding=binding, **inputs)
        if not admitted.admitted:
            raise ValueError("probe exceeds fast iteration domain: " + admitted.reason)
        if (row["elapsed_seconds"] + float(admitted.estimated_seconds or 0)
                > self.timeout_s):
            raise ValueError("compile plus warm/measured probe pair exceeds iteration wall budget")
        remaining = min(inputs["budget"].timeout_seconds,
                        self.timeout_s - row["elapsed_seconds"],
                        self.timeout_s if timeout_s is None else timeout_s)
        if float(admitted.estimated_seconds or 0) > remaining:
            raise ValueError("probe pair exceeds the remaining broker action budget")
        started = time.monotonic()
        try:
            observed = execute(timeout_s=remaining)
        finally:
            # Failed, timed-out, or subsequently rejected observations consume the same
            # iteration budget as valid measurements. Charge the originating row even
            # if the callback changed the candidate; never turn retries into free time.
            elapsed = time.monotonic() - started
            row["elapsed_seconds"] += elapsed
        if not isinstance(observed, ProbeObservation) or observed.evidence != inputs["probe"]:
            raise ValueError("executed probe does not match the admitted artifact and mechanism")
        self._current(candidate)
        if elapsed > remaining or row["elapsed_seconds"] > self.timeout_s:
            raise ValueError("probe execution exceeded its bounded iteration budget")
        receipt = {
            "schema": "global_mechanism_probe_receipt_v1", "iteration": row["iteration"],
            "binding": binding.to_dict(), "artifact_digest": observed.evidence.artifact_digest,
            "mechanism": observed.evidence.signature.to_dict(),
            "evidence": observed.evidence.to_dict(), "compute_receipt": observed.receipt.to_dict(),
            "counter_uncertainty_cycles": observed.counter_uncertainty_cycles,
            "target_timing_authority": (observed.timing_authority.to_evidence()
                                        if observed.timing_authority is not None else None),
            "observed_timing_identity": (asdict(observed.observed_timing_identity)
                                         if observed.observed_timing_identity is not None else None),
            "timing_scope": "raw probe observation; target-cycle inference requires separately validated calibration",
            "resource_profile": observed.resource_profile,
            "total_compute_cycles": observed.receipt.total_compute_cycles,
            "warmup_runs": 1, "measured_runs": 1, "elapsed_seconds": elapsed,
            "scope": "mechanism_probe_only", "full_model_cycles": None,
        }
        from merlin.perf.probe_relevance import classify_probe_relevance
        try:
            receipt["relevance"] = classify_probe_relevance(
                previous_analysis=self.iterations[-2]["analysis"] if len(self.iterations) > 1 else None,
                current_analysis=row["analysis"], previous_artifacts=self._previous_artifacts,
                current_artifacts=self._artifacts, signature=observed.evidence.signature)
        except ValueError as exc:
            receipt["relevance"] = {"status": "UNKNOWN", "reason": str(exc),
                                    "global_cost_validated": False,
                                    "scope": "isolated admitted calibration only"}
        path = self._write(f"probe_{row['iteration']:04d}_{len(row['probe_receipts']):04d}.json", receipt)
        row["probe_receipts"].append({"path": str(path), "sha256": PAS._sha256_file(path)})
        return receipt

    def seal(self, candidate: Path, *, name: str = "global_candidate") -> Path:
        """Seal verified global artifacts, independently of microbenchmark feedback/plateaus."""
        row = self._current(candidate)
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("checkpoint name must be one safe path component")
        snapshot = self.output / ("sealed_submission" if name == "global_candidate" else name + "_submission")
        submitted = Path(row["submitted_snapshot"])
        # Seal the exact host-captured source used for analysis. Authoring may leave Python caches
        # afterward; they are not compiler inputs and must not be copied into the review artifact.
        PAS.assert_candidate_sealable(submitted)
        if hash_tree(submitted)["sha256"] != row["candidate_sha256"]:
            raise ValueError("analyzed global submission changed before sealing")
        shutil.copytree(submitted, snapshot)
        if hash_tree(snapshot)["sha256"] != row["candidate_sha256"]:
            raise ValueError("global candidate changed while making its sealed snapshot")
        for item in sorted(snapshot.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            item.chmod(item.stat().st_mode & ~0o222)
        snapshot.chmod(snapshot.stat().st_mode & ~0o222)
        return self._write(name + ".json", {
            "schema": "global_perf_candidate_v1", "candidate_sha256": row["candidate_sha256"],
            "historical_reference": copy.deepcopy(self.historical_reference),
            "candidate_path": str(snapshot.resolve()), "iteration": row["iteration"],
            "candidate_read_only": True,
            "phase1_qualification": self.phase1_binding, "host_verification_policy": self.host_policy,
            "compiler_edit_authority": self.edit_scope_binding if self.edit_contract is not None else None,
            "compiler_dependencies": row["compiler_dependencies"],
            "analysis_sha256": PAS._document_sha256(row["analysis"]),
            "iteration_record": str((self.output / f"iteration_{row['iteration']:04d}.json").resolve()),
            "iteration_record_sha256": PAS._sha256_file(
                self.output / f"iteration_{row['iteration']:04d}.json"),
            "baseline_sha256": self.baseline_sha256, "target_sha256": self.target_sha256,
            "optimization_baseline_sha256": self.optimization_baseline_sha256,
            "optimization_baseline": self.optimization_baseline_binding,
            "capsule_sha256": self.sentinel.capsule_sha256,
            "portfolio": copy.deepcopy(self.portfolio_identity),
            "portfolio_sha256": self.portfolio_identity_sha256,
            "portfolio_iteration_sha256": PAS._document_sha256(row["portfolio"]),
            "probe_receipts": row["probe_receipts"],
            "semantic_receipts": row.get("semantic_receipts", []),
            "context_receipts": row.get("context_receipts", []),
            "paired_context_receipts": row.get("paired_context_receipts", []),
            "source_contraction_preparation_receipts": row.get("source_contraction_preparation_receipts", []),
            "source_pair_receipts": row.get("source_pair_receipts", []),
            "decision_feedback": row.get("decision_feedback"),
            "full_model_timing_status": "UNMEASURED", "global_speedup_proven": False,
            "promotion_status": "unqualified_candidate_for_review",
            "promotion_blockers": row["readiness"]["promotion_blockers"],
            "consumer": "global_plan_review_and_optional_post_freeze_validation",
        })

    def checkpoint_authoring(self, candidate: Path, *, name: str) -> Path:
        """Preserve an exact blocked portfolio revision for the next authoring round only.

        This deliberately has a different schema and consumer from :meth:`seal`.  It cannot be
        used for probes, promotion, or any performance claim; its only purpose is to let a bounded
        sequence repair a compiler that does not yet lower every training model.
        """
        row = self._matching_current(candidate, require_ready=False)
        if row["readiness"]["status"] != "blocked":
            raise ValueError("authoring checkpoints are only for blocked portfolio revisions")
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("checkpoint name must be one safe path component")
        snapshot = self.output / (name + "_submission")
        submitted = Path(row["submitted_snapshot"])
        PAS.assert_candidate_sealable(submitted)
        if hash_tree(submitted)["sha256"] != row["candidate_sha256"]:
            raise ValueError("analyzed authoring submission changed before checkpointing")
        shutil.copytree(submitted, snapshot)
        if hash_tree(snapshot)["sha256"] != row["candidate_sha256"]:
            raise ValueError("authoring candidate changed while making its checkpoint")
        for item in sorted(snapshot.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            item.chmod(item.stat().st_mode & ~0o222)
        snapshot.chmod(snapshot.stat().st_mode & ~0o222)
        iteration_path = self.output / f"iteration_{row['iteration']:04d}.json"
        return self._write(name + ".json", {
            "schema": "global_authoring_checkpoint_v1",
            "candidate_sha256": row["candidate_sha256"],
            "candidate_path": str(snapshot.resolve()), "candidate_read_only": True,
            "iteration": row["iteration"], "readiness": copy.deepcopy(row["readiness"]),
            "portfolio_members_ready": row["portfolio"]["members_ready"],
            "portfolio_members_total": row["portfolio"]["members_total"],
            "historical_reference": copy.deepcopy(self.historical_reference),
            "phase1_qualification": self.phase1_binding,
            "host_verification_policy": self.host_policy,
            "compiler_edit_authority": self.edit_scope_binding if self.edit_contract is not None else None,
            "compiler_dependencies": row["compiler_dependencies"],
            "analysis_sha256": PAS._document_sha256(row["analysis"]),
            "iteration_record": str(iteration_path.resolve()),
            "iteration_record_sha256": PAS._sha256_file(iteration_path),
            "baseline_sha256": self.baseline_sha256, "target_sha256": self.target_sha256,
            "optimization_baseline_sha256": self.optimization_baseline_sha256,
            "optimization_baseline": self.optimization_baseline_binding,
            "capsule_sha256": self.sentinel.capsule_sha256,
            "portfolio": copy.deepcopy(self.portfolio_identity),
            "portfolio_sha256": self.portfolio_identity_sha256,
            "portfolio_iteration_sha256": PAS._document_sha256(row["portfolio"]),
            "full_model_timing_status": "UNMEASURED", "full_model_cycles": None,
            "global_speedup_proven": False,
            "promotion_status": "blocked_authoring_checkpoint",
            "promotion_blockers": row["readiness"]["blockers"],
            "consumer": "next_bounded_global_authoring_round_only",
        })

    def run(self, propose: Callable[[Sequence[Mapping[str, Any]]], tuple[Path, str] | None],
            *, iterations: int) -> tuple[dict[str, Any], ...]:
        """Drive repeated agent proposals through the mandatory full-model compilation boundary."""
        if iterations < 1:
            raise ValueError("iterations must be positive")
        for _ in range(iterations):
            proposal = propose(copy.deepcopy(tuple(self.iterations)))
            if proposal is None:
                break
            candidate, hypothesis = proposal
            self.analyze(candidate, hypothesis=hypothesis)
        return copy.deepcopy(tuple(self.iterations))


def consume_authoring_checkpoint(path: Path) -> dict[str, Any]:
    """Verify a blocked, exact portfolio checkpoint without granting promotion authority."""
    document = PAS._mapping_file(path)
    if (document.get("schema") != "global_authoring_checkpoint_v1"
            or document.get("promotion_status") != "blocked_authoring_checkpoint"
            or document.get("consumer") != "next_bounded_global_authoring_round_only"
            or document.get("full_model_timing_status") != "UNMEASURED"
            or document.get("full_model_cycles") is not None
            or document.get("global_speedup_proven") is not False):
        raise ValueError("invalid global authoring checkpoint or unsupported performance claim")
    if host_verification_policy_record() != document.get("host_verification_policy"):
        raise ValueError("authoring checkpoint host verification policy changed")
    experiment_record = PAS._mapping_file(path.parent / "experiment.json")
    if any(document.get(field) != experiment_record.get(field) for field in (
            "baseline_sha256", "target_sha256", "optimization_baseline_sha256",
            "optimization_baseline", "phase1_qualification", "portfolio", "portfolio_sha256")):
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
        if (PAS._mapping_file(path.parent / "compiler_edit_authority.json") != authority
                or Path(authority["seed_path"]).resolve() != initial.resolve()
                or initial.is_symlink()
                or hash_tree(initial)["sha256"] != authority["initial_candidate_sha256"]
                or PAS._document_sha256(authority["contract"])
                    != authority["contract_document_sha256"]):
            raise ValueError("authoring checkpoint edit authority changed")
        validate_edit_contract(authority["contract"], initial)
        if inspect_compiler_edits(
                initial, Path(document["candidate_path"]), authority["contract"])["status"] != "allowed":
            raise ValueError("authoring checkpoint exceeds its host-frozen edit authority")
    candidate = Path(document["candidate_path"])
    if (candidate.is_symlink() or candidate.parent.resolve() != path.parent.resolve()
            or document.get("candidate_read_only") is not True
            or hash_tree(candidate)["sha256"] != document.get("candidate_sha256")):
        raise ValueError("authoring checkpoint candidate bytes changed")
    if compiler_dependency_record(candidate) != document.get("compiler_dependencies"):
        raise ValueError("authoring checkpoint compiler dependencies changed")
    iteration_path = Path(document["iteration_record"])
    if (iteration_path.is_symlink() or iteration_path.parent.resolve() != path.parent.resolve()
            or PAS._sha256_file(iteration_path) != document.get("iteration_record_sha256")):
        raise ValueError("authoring checkpoint iteration receipt changed")
    iteration = PAS._mapping_file(iteration_path)
    analysis = iteration.get("analysis") or {}
    readiness = iteration.get("readiness") or {}
    portfolio = iteration.get("portfolio") or {}
    portfolio_identity = document.get("portfolio") or {}
    members, identities = portfolio.get("members"), portfolio_identity.get("members")
    if (iteration.get("candidate_sha256") != document.get("candidate_sha256")
            or iteration.get("compiler_dependencies") != document.get("compiler_dependencies")
            or PAS._document_sha256(analysis) != document.get("analysis_sha256")
            or readiness != document.get("readiness") or readiness.get("status") != "blocked"
            or PAS._document_sha256(portfolio_identity) != document.get("portfolio_sha256")
            or portfolio.get("portfolio_sha256") != document.get("portfolio_sha256")
            or PAS._document_sha256(portfolio) != document.get("portfolio_iteration_sha256")
            or portfolio.get("candidate_sha256") != document.get("candidate_sha256")
            or not isinstance(members, list) or not isinstance(identities, list)
            or not members or len(members) != len(identities)
            or portfolio.get("members_total") != len(members)
            or portfolio.get("members_ready") != document.get("portfolio_members_ready")
            or document.get("portfolio_members_total") != len(members)
            or not 0 <= document.get("portfolio_members_ready", -1) <= len(members)
            or portfolio.get("full_model_simulation_allowed") is not False):
        raise ValueError("authoring checkpoint portfolio binding or blocked readiness changed")
    for index, (identity, member) in enumerate(zip(identities, members, strict=True)):
        if member.get("identity") != identity:
            raise ValueError("authoring checkpoint portfolio identity changed")
        member_analysis = analysis if index == 0 else member.get("analysis") or {}
        if index == 0 and (member.get("analysis_ref") != "/analysis"
                or member.get("static_comparison_ref") != "/static_comparison"):
            raise ValueError("authoring checkpoint primary portfolio aliases changed")
        if (member_analysis.get("candidate_sha256") != document.get("candidate_sha256")
                or member_analysis.get("workload", {}).get("capsule_sha256")
                    != identity.get("capsule_sha256")
                or PAS.global_iteration_readiness(member_analysis) != member.get("readiness")):
            raise ValueError("authoring checkpoint lacks exact member analysis evidence")
    return document


def _consume_round_checkpoint(path: Path) -> dict[str, Any]:
    schema = PAS._mapping_file(path).get("schema")
    if schema == "global_perf_candidate_v1":
        return consume_global_candidate(path)
    if schema == "global_authoring_checkpoint_v1":
        return consume_authoring_checkpoint(path)
    raise ValueError("unsupported global round checkpoint schema")


def consume_global_candidate(path: Path) -> dict[str, Any]:
    """Verify the distinct macro handoff without turning it into a measured speedup claim."""
    document = PAS._mapping_file(path)
    if (document.get("schema") != "global_perf_candidate_v1"
            or document.get("full_model_timing_status") != "UNMEASURED"
            or document.get("global_speedup_proven") is not False):
        raise ValueError("invalid global candidate receipt or unsupported full-model timing claim")
    if host_verification_policy_record() != document.get("host_verification_policy"):
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
        if (PAS._mapping_file(path.parent / "compiler_edit_authority.json") != authority
                or Path(authority["seed_path"]).resolve() != initial.resolve()
                or initial.is_symlink() or hash_tree(initial)["sha256"] != authority["initial_candidate_sha256"]
                or PAS._document_sha256(authority["contract"]) != authority["contract_document_sha256"]):
            raise ValueError("sealed candidate edit authority changed")
        validate_edit_contract(authority["contract"], initial)
        if inspect_compiler_edits(initial, Path(document["candidate_path"]), authority["contract"])["status"] != "allowed":
            raise ValueError("sealed compiler exceeds its host-frozen edit authority")
    qualification = document.get("phase1_qualification")
    if qualification is not None:
        if qualification.get("submission_sha256") != document.get("baseline_sha256"):
            raise ValueError("sealed Phase-1 compiler identity changed")
        frozen_root = Path(qualification["run_dir"])
        for name, digest in qualification["evidence_sha256"].items():
            source = frozen_root / name
            if (source.is_symlink() or frozen_root.resolve() not in source.resolve().parents
                    or PAS._sha256_file(source) != digest):
                raise ValueError("frozen Phase-1 qualification evidence changed after sealing")
    comparison = document.get("optimization_baseline")
    if comparison is not None:
        experiment_record = PAS._mapping_file(path.parent / "experiment.json")
        comparison_path = Path(comparison["path"])
        if (comparison != experiment_record.get("optimization_baseline")
                or document.get("baseline_sha256") != experiment_record.get("baseline_sha256")
                or document.get("optimization_baseline_sha256") != comparison.get("sha256")
                or comparison.get("schema") != "global_optimization_baseline_v1"
                or comparison.get("objective_numerical_qualification") != "UNPROVEN"
                or comparison.get("phase1_regraded") is not False
                or comparison_path.is_symlink()
                or hash_tree(comparison_path)["sha256"] != comparison.get("sha256")
                or compiler_dependency_record(comparison_path) != comparison.get("compiler_dependencies")):
            raise ValueError("sealed optimization comparison baseline identity or scope changed")
        if (comparison.get("selection") == "explicit_host_seed"
                and comparison_path.resolve() != (path.parent / "optimization_baseline").resolve()):
            raise ValueError("sealed optimization baseline escaped its immutable experiment snapshot")
    candidate = Path(document["candidate_path"])
    if (candidate.is_symlink() or candidate.parent.resolve() != path.parent.resolve()
            or hash_tree(candidate)["sha256"] != document.get("candidate_sha256")):
        raise ValueError("sealed global candidate bytes changed")
    if compiler_dependency_record(candidate) != document.get("compiler_dependencies"):
        raise ValueError("sealed compiler shared implementation dependencies changed")
    iteration_path = Path(document["iteration_record"])
    if (iteration_path.is_symlink() or iteration_path.parent.resolve() != path.parent.resolve()
            or PAS._sha256_file(iteration_path) != document.get("iteration_record_sha256")):
        raise ValueError("global iteration receipt changed or escaped its experiment")
    iteration = PAS._mapping_file(iteration_path)
    analysis = iteration.get("analysis") or {}
    experiment_record = PAS._mapping_file(path.parent / "experiment.json")
    portfolio_identity = document.get("portfolio")
    portfolio = iteration.get("portfolio") or {}
    if (not isinstance(portfolio_identity, Mapping)
            or PAS._document_sha256(portfolio_identity) != document.get("portfolio_sha256")
            or experiment_record.get("portfolio") != portfolio_identity
            or experiment_record.get("portfolio_sha256") != document.get("portfolio_sha256")
            or portfolio.get("portfolio_sha256") != document.get("portfolio_sha256")
            or PAS._document_sha256(portfolio) != document.get("portfolio_iteration_sha256")
            or portfolio.get("candidate_sha256") != document.get("candidate_sha256")):
        raise ValueError("sealed complete-model portfolio identity or iteration changed")
    identities = portfolio_identity.get("members")
    members = portfolio.get("members")
    if (not isinstance(identities, list) or not isinstance(members, list)
            or len(identities) != len(members) or not members
            or portfolio.get("members_total") != len(members)
            or portfolio.get("members_ready") != len(members)
            or portfolio.get("full_model_simulation_allowed") is not False):
        raise ValueError("sealed complete-model portfolio coverage is incomplete")
    for index, (identity, member) in enumerate(zip(identities, members, strict=True)):
        if member.get("identity") != identity:
            raise ValueError("sealed portfolio member identity changed")
        if index == 0:
            if (member.get("analysis_ref") != "/analysis"
                    or member.get("static_comparison_ref") != "/static_comparison"):
                raise ValueError("sealed primary portfolio aliases changed")
            member_analysis = analysis
        else:
            member_analysis = member.get("analysis") or {}
        if (member_analysis.get("candidate_sha256") != document.get("candidate_sha256")
                or member_analysis.get("workload", {}).get("capsule_sha256")
                != identity.get("capsule_sha256")
                or PAS.global_iteration_readiness(member_analysis)["status"]
                != "ready_for_probe_admission"):
            raise ValueError("sealed portfolio member lacks bound graph/plan/artifact evidence")
    if comparison is not None and (iteration.get("optimization_baseline") != comparison
            or analysis.get("optimization_baseline") != comparison
            or iteration.get("optimization_baseline_sha256") != comparison["sha256"]
            or iteration.get("baseline_sha256") != document["baseline_sha256"]):
        raise ValueError("sealed iteration changed its optimization comparison baseline")
    if (PAS._document_sha256(analysis) != document.get("analysis_sha256")
            or analysis.get("candidate_sha256") != document.get("candidate_sha256")
            or PAS.global_iteration_readiness(analysis)["status"] != "ready_for_probe_admission"):
        raise ValueError("global candidate lacks current verified graph/plan/artifact evidence")
    expected = {
        "compiler_digest": document["compiler_dependencies"]["compiler_implementation_sha256"],
        "target_digest": document["target_sha256"],
        "graph_digest": analysis["diagnostics"]["captured_logical_graph"]["logical_dispatch_digest"],
        "plan_digest": analysis["diagnostics"]["verified_global_plan_emission"]["plan_digest"],
    }
    for receipt in document.get("context_receipts") or []:
        context_path = Path(receipt["path"])
        if (context_path.is_symlink() or context_path.parent.resolve() != path.parent.resolve()
                or PAS._sha256_file(context_path) != receipt.get("sha256")):
            raise ValueError("optional controlled source-prefix receipt changed")
        context = PAS._mapping_file(context_path)
        if (context.get("binding") != expected or context.get("scope") != "controlled_source_prefix"
                or context.get("model_artifact_sha256") != analysis["emission"]["candidate_lowered_sha256"]
                or context.get("full_model_cycles") is not None
                or context.get("global_cost_validated") is not False
                or context.get("global_speedup_proven") is not False
                or context.get("calibration_admissible") is not False):
            raise ValueError("controlled source-prefix evidence is stale or overstates its scope")
    expected_decision_feedback = None
    preparation_digests = set()
    for field, schema in (("source_contraction_preparation_receipts", "global_source_contraction_preparation_receipt_v1"),
                          ("source_pair_receipts", "global_source_contraction_execution_receipt_v1")):
        for reference in document.get(field, []):
            receipt_path = Path(reference["path"])
            if (receipt_path.is_symlink() or receipt_path.parent.resolve() != path.parent.resolve()
                    or PAS._sha256_file(receipt_path) != reference.get("sha256")):
                raise ValueError("source-pair receipt changed or escaped its experiment")
            receipt = PAS._mapping_file(receipt_path)
            if (receipt.get("schema") != schema or receipt.get("binding") != expected
                    or receipt.get("host_verifier_policy_sha256") != document["host_verification_policy"]["sha256"]
                    or receipt.get("full_model_numerics_qualified") is not False
                    or receipt.get("global_speedup_proven") is not False):
                raise ValueError("source-pair evidence is stale or overstates its scope")
            if field == "source_contraction_preparation_receipts":
                if receipt.get("numerical_pass") is not False or receipt.get("runtime_admitted") is not False:
                    raise ValueError("source preparation cannot qualify numerical execution")
                preparation_digests.add(reference["sha256"])
            elif receipt.get("preparation_sha256") not in preparation_digests:
                raise ValueError("source execution has no sealed preparation receipt")
    for receipt in document.get("paired_context_receipts") or []:
        pair_path = Path(receipt["path"])
        if (pair_path.is_symlink() or pair_path.parent.resolve() != path.parent.resolve()
                or PAS._sha256_file(pair_path) != receipt.get("sha256")):
            raise ValueError("paired controlled-context receipt changed")
        pair = PAS._mapping_file(pair_path)
        prior_path = Path(pair["previous_iteration_record"])
        if (prior_path.is_symlink() or prior_path.parent.resolve() != path.parent.resolve()
                or PAS._sha256_file(prior_path) != pair["previous_iteration_record_sha256"]):
            raise ValueError("paired preceding iteration evidence changed")
        prior = PAS._mapping_file(prior_path)
        prior_diag = prior["analysis"]["diagnostics"]
        prior_binding = {
            "compiler_digest": prior["compiler_dependencies"]["compiler_implementation_sha256"],
            "target_digest": document["target_sha256"],
            "graph_digest": prior_diag["captured_logical_graph"]["logical_dispatch_digest"],
            "plan_digest": prior_diag["verified_global_plan_emission"]["plan_digest"],
        }
        proof = pair["projection_proof"]
        if (pair.get("binding") != expected or pair.get("previous_binding") != prior_binding
                or pair.get("model_artifact_sha256") != analysis["emission"]["candidate_lowered_sha256"]
                or pair.get("previous_model_artifact_sha256") != prior["analysis"]["emission"]["candidate_lowered_sha256"]
                or pair.get("scope") != "controlled_fixed_work_slice"
                or pair.get("full_model_cycles") is not None
                or any(pair.get(key) is not False for key in
                       ("global_cost_validated", "global_speedup_proven", "calibration_admissible"))
                or PAS._document_sha256(proof["work_contract"]) != proof["work_contract_sha256"]
                or PAS._document_sha256(pair["deterministic_input_contract"])
                    != pair["deterministic_input_contract_sha256"]):
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
        if (semantic_path.is_symlink() or semantic_path.parent.resolve() != path.parent.resolve()
                or PAS._sha256_file(semantic_path) != receipt.get("sha256")):
            raise ValueError("optional changed-region semantic receipt changed")
        semantic = PAS._mapping_file(semantic_path)
        if (semantic.get("binding") != expected
                or semantic.get("current_artifact_sha256") != analysis["emission"]["candidate_lowered_sha256"]
                or semantic.get("full_model_numerics_qualified") is not False
                or semantic.get("global_speedup_proven") is not False
                or semantic.get("full_model_cycles") is not None):
            raise ValueError("optional semantic evidence is stale or overstates its scope")
    for receipt in document.get("probe_receipts") or []:
        probe_path = Path(receipt["path"])
        if (probe_path.is_symlink() or probe_path.parent.resolve() != path.parent.resolve()
                or PAS._sha256_file(probe_path) != receipt.get("sha256")):
            raise ValueError("optional mechanism probe receipt changed")
        probe = PAS._mapping_file(probe_path)
        if (probe.get("binding") != expected or probe.get("scope") != "mechanism_probe_only"
                or probe.get("full_model_cycles") is not None or probe.get("warmup_runs") != 1
                or probe.get("measured_runs") != 1):
            raise ValueError("optional probe evidence is stale or has the wrong measurement scope")
    return document


def write_semantic_supplement(*, original_candidate_receipt: Path, previous_iteration_receipt: Path,
                              semantic_receipt: Path, output: Path) -> dict[str, Any]:
    """Add scoped evidence to an immutable checkpoint without relabeling its original verdict."""
    if output.exists() or output.is_symlink():
        raise ValueError("semantic supplement output must be fresh")
    pointers = {name: {"path": str(path.resolve()), "sha256": PAS._sha256_file(path)}
                for name, path in (("original_candidate_receipt", original_candidate_receipt),
                                   ("previous_iteration_receipt", previous_iteration_receipt),
                                   ("semantic_receipt", semantic_receipt))}
    document = {"schema": "global_semantic_supplement_v1", **pointers,
                "host_verification_policy": host_verification_policy_record(),
                "analysis_action": "reuse_existing_bound_immutable_graph_receipts",
                "original_verdict_modified": False, "full_model_numerics_qualified": False,
                "global_speedup_proven": False, "full_model_cycles": None}
    PAS._write_json(output, document)
    return consume_semantic_supplement(output)


def validate_optimization_baseline_resume(checkpoint: Mapping[str, Any], *,
                                          optimization_baseline_sha256: str) -> None:
    """Legacy checkpoints used Phase 1 for comparison; a new segment must retain that choice."""
    previous = checkpoint.get("optimization_baseline_sha256", checkpoint.get("baseline_sha256"))
    if not PAS._is_sha256(optimization_baseline_sha256) or previous != optimization_baseline_sha256:
        raise ValueError("resume checkpoint optimization baseline differs; select a new experiment explicitly")


def verify_retained_global_checkpoint(original_path: Path) -> dict[str, Any]:
    """Consume a prior segment under its own immutable implementation, never relabel its policy."""
    import os
    import subprocess
    import sys
    original = PAS._mapping_file(original_path)
    launch = PAS._mapping_file(original_path.parent.parent / "launch.json")
    snapshot = Path(launch["source_snapshot"])
    scripts = snapshot / "merlin/experiments/gemmini_perf_bench/scripts"
    old_controller = scripts / "run_global_perf_experiment.py"
    if (original["host_verification_policy"]["sources"].get(str(old_controller.resolve()))
            != PAS._sha256_file(old_controller)):
        raise ValueError("original checkpoint verifier implementation changed")
    env = {**os.environ, "MERLIN_REPO_ROOT": str(snapshot), "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": os.pathsep.join((str(snapshot / "merlin/python"), str(scripts)))}
    code = ("import sys; from pathlib import Path; import run_global_perf_experiment as G; "
            "G.consume_global_candidate(Path(sys.argv[1])); print('original_checkpoint_verified')")
    checked = subprocess.run([sys.executable, "-c", code, str(original_path)], env=env,
                             capture_output=True, text=True, timeout=30)
    if checked.returncode != 0:
        raise ValueError("original immutable checkpoint no longer verifies: " + checked.stderr[-2000:])
    return original


def consume_semantic_supplement(path: Path) -> dict[str, Any]:
    """Verify old checkpoint under its own policy, and new mechanism evidence under this policy."""
    document = PAS._mapping_file(path)
    if (document.get("schema") != "global_semantic_supplement_v1"
            or document.get("host_verification_policy") != host_verification_policy_record()
            or document.get("analysis_action") != "reuse_existing_bound_immutable_graph_receipts"
            or document.get("full_model_cycles") is not None
            or any(document.get(key) is not False for key in
                   ("original_verdict_modified", "full_model_numerics_qualified", "global_speedup_proven"))):
        raise ValueError("semantic supplement policy or scope changed")
    loaded = {}
    for key in ("original_candidate_receipt", "previous_iteration_receipt", "semantic_receipt"):
        pointer = document[key]
        source = Path(pointer["path"])
        if source.is_symlink() or PAS._sha256_file(source) != pointer["sha256"]:
            raise ValueError("semantic supplement source receipt changed")
        loaded[key] = PAS._mapping_file(source)
    original_path = Path(document["original_candidate_receipt"]["path"])
    original = verify_retained_global_checkpoint(original_path)
    final_analysis = PAS._mapping_file(Path(original["iteration_record"]))["analysis"]
    previous = loaded["previous_iteration_receipt"]
    semantic = loaded["semantic_receipt"]
    expected = {"compiler_digest": original["compiler_dependencies"]["compiler_implementation_sha256"],
                "target_digest": original["target_sha256"],
                "graph_digest": final_analysis["diagnostics"]["captured_logical_graph"]["logical_dispatch_digest"],
                "plan_digest": final_analysis["diagnostics"]["verified_global_plan_emission"]["plan_digest"]}
    if (semantic.get("binding") != expected
            or semantic.get("current_artifact_sha256") != final_analysis["emission"]["candidate_lowered_sha256"]
            or semantic.get("previous_artifact_sha256") != previous["analysis"]["emission"]["candidate_lowered_sha256"]
            or previous["analysis"]["diagnostics"]["captured_logical_graph"]["logical_dispatch_digest"] != expected["graph_digest"]
            or semantic.get("full_model_cycles") is not None
            or semantic.get("global_speedup_proven") is not False
            or semantic.get("full_model_numerics_qualified") is not False):
        raise ValueError("supplement semantic evidence is stale or overstates its scope")
    return {**document, "candidate_sha256": original["candidate_sha256"],
            "semantic_status": semantic["evidence"].get("status"),
            "original_checkpoint_verified": True}


def verify_global_broker_receipts(path: Path, *, actions: Sequence[PAS.BrokerAction],
                                 audit: Mapping[str, Any]) -> dict[str, Any]:
    """Join macro agent tool calls without importing the legacy mandatory micro timing rule."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("global broker receipts are absent or linked")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    # Requests receive indices at admission, but concurrent requests finish out of order.
    rows.sort(key=lambda row: row.get("index", -1))
    registry = {action.name: action for action in actions}
    for index, row in enumerate(rows):
        if (row.get("index") != index or row.get("receipt_schema_version") != 1
                or row.get("action") not in registry
                or row.get("state") not in ("complete", "rejected")
                or any(not PAS._is_sha256(row.get(key)) for key in (
                    "bindings_command_sha256", "stdout_sha256", "stderr_sha256", "argv_sha256"))):
            raise ValueError("global broker receipt schema is invalid")
    observed = [(row["action"], row["bindings_command_sha256"]) for row in rows]
    invocations = audit.get("broker_invocations")
    if (not isinstance(invocations, list) or observed != [
            (row.get("action"), row.get("bindings_sha256")) for row in invocations]):
        raise ValueError("global broker receipts do not match exact agent invocations")
    required = {action.name for action in actions if action.required}
    succeeded = {row["action"] for row in rows if row.get("state") == "complete"
                 and row.get("returncode") == 0}
    if required - succeeded:
        raise ValueError("global broker required actions did not complete: " + str(sorted(required - succeeded)))
    if PAS.DEVELOPMENT_FEEDBACK_ACTION in succeeded:
        raise ValueError("a legacy micro sweep cannot be macro experiment evidence")
    return {"path": str(path.resolve()), "sha256": PAS._sha256_file(path),
            "count": len(rows), "required_actions": sorted(required),
            "successful_actions": sorted(succeeded), "all_required_succeeded": True}


def rebind_compiler_sandbox(cached: Mapping[str, Any], *, package: Path, scratch: Path,
                            dependencies: Mapping[str, Any]) -> dict[str, Any]:
    """Reuse exact deny masks; move only host-owned compiler/scratch sources, retaining aliases.

    The existing virtual cwd/import alias remains bound to the new immutable compiler bytes.
    Extra exact-path mounts make the worker's fresh absolute entrypoint/input paths available.
    No parent directory, environment, toolchain, or answer grant is enlarged.
    """
    previous = cached["compiler_dependencies"]
    if any(previous.get(key) != dependencies.get(key) for key in
           ("shared_source_root", "shared_sources", "selected_lazy_exports")):
        raise ValueError("cached compiler policy shared dependency closure changed")
    package, scratch = package.absolute(), scratch.absolute()
    if any(path.is_symlink() or not path.is_dir() or path.resolve() != path for path in (package, scratch)):
        raise ValueError("cached compiler policy requires real exact candidate and scratch directories")
    if package == scratch or package in scratch.parents or scratch in package.parents:
        raise ValueError("compiler and scratch grants must be independent")
    for surface in cached["answer_surfaces"]:
        answer = Path(surface["path"]).resolve()
        for path in (package, scratch):
            if path == answer or path in answer.parents or (surface["kind"] == "dir" and answer in path.parents):
                raise ValueError("cached compiler rebind overlaps a masked answer surface")
    if compiler_dependency_record(package, shared_source_root=Path(dependencies["shared_source_root"])) != dependencies:
        raise ValueError("cached compiler rebind source identity changed")
    for directory, digest in cached.get("overlay_trees", {}).items():
        if PAS._exact_tree_record(Path(directory))["sha256"] != digest:
            raise ValueError("cached read-only compiler overlay changed")
    prefix = list(cached["command_prefix"])
    boundary = cached["bwrap_argv_length"]
    old_package, old_scratch = str(cached["package_path"]), str(cached["scratch_path"])
    masks = []
    surface_paths = {row["path"] for row in cached["answer_surfaces"]}
    for index, token in enumerate(prefix[:boundary]):
        if token == "--tmpfs" and prefix[index+1] in surface_paths:
            masks.append((index, tuple(prefix[index:index+2])))
        elif token == "--ro-bind" and prefix[index+1] == "/dev/null" and prefix[index+2] in surface_paths:
            masks.append((index, tuple(prefix[index:index+3])))
        elif token in ("--ro-bind", "--bind"):
            for before, after in ((old_package, str(package)), (old_scratch, str(scratch))):
                source = prefix[index+1]
                if source == before or source.startswith(before + "/"):
                    prefix[index+1] = after + source[len(before):]
    if not masks:
        raise ValueError("cached compiler policy has no retained explicit answer masks")
    if any(tuple(prefix[index:index+len(mask)]) != mask for index, mask in masks):
        raise ValueError("cached compiler rebind changed an answer mask")
    insertion = min(index for index, _ in masks)
    additions = ["--ro-bind", str(package), str(package), "--bind", str(scratch), str(scratch)]
    prefix[insertion:insertion] = additions
    return {**cached, "package_path": str(package), "scratch_path": str(scratch),
            "compiler_dependencies": dict(dependencies), "command_prefix": prefix,
            "bwrap_argv_length": boundary + len(additions),
            "policy_reuse": {"status": "exact_masks_and_shared_closure_rebound",
                "cached_prefix_sha256": PAS._document_sha256(cached["command_prefix"]),
                "retained_virtual_candidate_alias": old_package,
                "rebound_candidate_sha256": dependencies["candidate_sha256"],
                "source_grants_changed": [old_package, old_scratch],
                "answer_masks_changed": False}}


def global_compiler_sandbox_factory(*, target_experiment: Any, agent_inputs: Any,
                                   frozen_functional: Any, frozen_corpus_manifest: Path,
                                   qualification_baseline: Path | None = None):
    """Extend the existing inner policy with only exact compiler inputs and dedicated scratch."""
    cache: dict[str, tuple[str, dict[str, Any]]] = {}
    def factory(baseline: Path, candidate: Path, scratch: Path) -> dict[str, Any]:
        from dataclasses import replace
        started = time.monotonic()
        PAS.verify_answer_free_agent_inputs(agent_inputs)
        surfaces = PAS.answer_surfaces(target_experiment)
        surfaces_record = [{"path": str(surface.path), "kind": surface.kind} for surface in surfaces]
        fixed = {"host_policy": host_verification_policy_record(), "surfaces": surfaces_record,
                 "descriptor": PAS._sha256_file(target_experiment.path),
                 "agent_inputs": PAS._sha256_file(agent_inputs.manifest_path),
                 "frozen_inputs": PAS._sha256_file(frozen_functional.marker),
                 "frozen_corpus": PAS._sha256_file(frozen_corpus_manifest),
                 "baseline_sha256": hash_tree(baseline)["sha256"],
                 "qualification_baseline_sha256": hash_tree(
                     qualification_baseline if qualification_baseline is not None else baseline)["sha256"]}
        result = {}
        for arm, package in (("baseline", baseline), ("candidate", candidate)):
            dependencies = compiler_dependency_record(package)
            key = PAS._document_sha256({**fixed, "shared_source_root": dependencies["shared_source_root"],
                "shared_sources": dependencies["shared_sources"],
                "selected_lazy_exports": dependencies.get("selected_lazy_exports", {})})
            if arm in cache and cache[arm][0] == key:
                result[arm] = rebind_compiler_sandbox(cache[arm][1], package=package, scratch=scratch,
                                                     dependencies=dependencies)
                result[arm]["policy_setup_elapsed_seconds"] = time.monotonic()-started
                continue
            policy = PAS.inner_execution_policy(
                target_experiment, package, agent_inputs, frozen_functional,
                qualification_baseline if qualification_baseline is not None else baseline,
                frozen_corpus_manifest)
            argv = [*policy.argv, "--ro-bind", str(package), str(package),
                    "--bind", str(scratch), str(scratch)]
            # Frozen Phase-1 contract mounts remain untouched. The current
            # whole-program compiler API has a distinct, explicit namespace.
            schema = PAS.whole_program_schema_record()
            argv.extend(("--ro-bind", schema["path"], "/compiler-api/command_buffer.schema.json",
                         "--setenv", "MERLIN_COMMAND_BUFFER_SCHEMA", "/compiler-api/command_buffer.schema.json"))
            overlay_root = scratch.parent / f"{scratch.name}.{arm}_dependency_overlays"
            argv = compiler_dependency_mounts(
                argv, dependencies, overlay_root)
            # Compiler imports never unmask an evaluator or answer surface, even when a static
            # import closure conservatively includes branches that this compiler does not execute.
            argv = PAS.BW.apply_answer_masks(argv, surfaces)
            gaps = PAS.BW.coverage_gap(argv, surfaces)
            if gaps:
                raise ValueError("compiler dependency grants expose answer surfaces")
            policy = replace(policy, argv=tuple(argv), candidate_writable=False)
            prefix = PAS.inner_command(policy, target_experiment, package, ["PAYLOAD_MARKER"], 600)[:-1]
            result[arm] = {"package_path": str(package), "command_prefix": prefix,
                           "scratch_path": str(scratch),
                           "compiler_dependencies": dependencies, "bwrap_argv_length": len(policy.argv),
                           "answer_surfaces": surfaces_record,
                           "overlay_trees": {str(overlay_root): PAS._exact_tree_record(overlay_root)["sha256"]}
                           if overlay_root.exists() else {},
                           "policy_setup_elapsed_seconds": time.monotonic()-started}
            cache[arm] = (key, copy.deepcopy(result[arm]))
        return result
    return factory


def compiler_dependency_mounts(argv: Sequence[str], dependencies: Mapping[str, Any],
                               overlay_root: Path) -> list[str]:
    """Merge new exact helpers into already granted directories without widening those grants.

    bwrap cannot create a new file mountpoint beneath a read-only directory mount. Only that
    existing granted directory is copied, extended with verified helper bytes, and re-bound RO;
    the caller reapplies every original answer mask afterward.
    """
    result = list(argv)
    root = Path(dependencies["shared_source_root"])
    directory_mounts = [(index, Path(result[index + 1]), Path(result[index + 2]))
                        for index, option in enumerate(result[:-2])
                        if option == "--ro-bind" and Path(result[index + 1]).is_dir()]
    additions: dict[int, list[tuple[Path, Path]]] = {}
    direct: list[Path] = []
    for relative, digest in dependencies["shared_sources"].items():
        source = root / relative
        if source.is_symlink() or PAS._sha256_file(source) != digest:
            raise ValueError("shared compiler dependency changed before sandbox binding")
        containing = [(index, original, destination) for index, original, destination in directory_mounts
                      if destination in source.parents]
        if containing:
            index, original, destination = max(containing, key=lambda row: (len(row[2].parts), row[0]))
            inside = source.relative_to(destination)
            if not (original / inside).exists():
                additions.setdefault(index, []).append((source, inside))
                continue
        direct.append(source)
    for index, sources in additions.items():
        overlay = overlay_root / str(index)
        shutil.copytree(result[index + 1], overlay, symlinks=True)
        PAS._make_writable(overlay)
        for source, inside in sources:
            destination = overlay / inside
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if PAS._sha256_file(source) != PAS._sha256_file(destination):
                raise ValueError("compiler helper changed while creating its read-only overlay")
        for path in overlay.rglob("*"):
            if not path.is_symlink():
                path.chmod(path.stat().st_mode & ~0o222)
        overlay.chmod(0o555)
        result[index + 1] = str(overlay)
    for source in direct:
        result.extend(("--ro-bind", str(source), str(source)))
    return result


def configure_global_analysis(experiment: GlobalPerfExperiment, *, target_experiment: Any,
                              agent_inputs: Any, frozen_functional: Any,
                              frozen_corpus_manifest: Path, stage_root: Path) -> None:
    """One production analysis policy for authoring and deterministic checkpoint validation."""
    import importlib
    from merlin.runtime.backends.base import get_backend
    from merlin.perf.analysis_worker import IsolatedAnalysisWorker
    if experiment.completion_contract is None:
        try:
            backend = get_backend(experiment.target)
            adapter = importlib.import_module(backend.__name__ + "." + experiment.target + "_completion_contract")
            experiment.completion_contract = adapter.derive_completion_contract()
        except (ImportError, ValueError, KeyError):
            pass
    if experiment.analyzer is PAS.analyze_whole_model_emission:
        sandbox_factory = global_compiler_sandbox_factory(
            target_experiment=target_experiment, agent_inputs=agent_inputs,
            frozen_functional=frozen_functional,
            frozen_corpus_manifest=frozen_corpus_manifest,
            qualification_baseline=experiment.baseline)
        experiment.analyzer = IsolatedAnalysisWorker(
            stage_path=Path(PAS.__file__), sandbox_factory=sandbox_factory,
            output=stage_root / "host_analysis_workers")


def _retryable_capacity_failure(stage_root: Path, round_index: int,
                                terminal: Mapping[str, Any]) -> bool:
    """Recognize a host-recorded capacity refusal that did no candidate work.

    This is a continuation decision, never an acceptance: the next round starts from the last
    consumed checkpoint. Keep the classifier narrow so arbitrary agent/tool failures cannot bypass
    the required-action gate.
    """
    if (terminal.get("agent_exit_code", 0) == 0
            or terminal.get("audit", {}).get("clean") is not True
            or terminal.get("audit", {}).get("broker_invocations") not in (None, [])):
        return False
    path = stage_root / "rounds" / f"round_{round_index:02d}.codex_summary.json"
    if path.is_symlink() or not path.is_file():
        return False
    try:
        summary = PAS._mapping_file(path)
    except (OSError, ValueError):
        return False
    if (summary.get("exit_code") != terminal.get("agent_exit_code")
            or summary.get("timed_out") is not False
            or summary.get("turns_started") != 1
            or summary.get("turns_usage_reported") != 0):
        return False
    errors = summary.get("errors")
    return (isinstance(errors, list) and bool(errors)
            and all(isinstance(value, str)
                    and "selected model is at capacity" in value.lower() for value in errors))


def _retryable_unchanged_round_failure(terminal: Mapping[str, Any], *,
                                       checkpoint_sha256: str) -> bool:
    """Continue after a clean no-change round whose mandatory edit analysis never ran.

    This admits no candidate: the next round is reconstructed from the already consumed checkpoint.
    It only prevents a correctly reverted late proposal from terminating the whole bounded sequence.
    A host-recorded timeout (124) is eligible only with a clean transcript; missing mandatory
    analysis remains a refused round, not fabricated evidence for a new candidate. The sequence
    separately revalidates its source policy and consumes the previous checkpoint before resuming.
    """
    broker = terminal.get("broker_evidence") or {}
    audit = terminal.get("audit") or {}
    reason = broker.get("reason")
    hits = audit.get("hits")
    clean_or_recorded_command_error = (audit.get("clean") is True or (
        isinstance(hits, list) and bool(hits)
        and all(isinstance(row, Mapping) and row.get("kind") == "invalid_broker_invocation"
                for row in hits)
        and broker.get("all_required_succeeded") is True))
    broker_failure_is_bounded = (broker.get("all_required_succeeded") is True or (
        broker.get("status") == "refused" and isinstance(reason, str)
        and reason.startswith("global broker required actions did not complete:")))
    eligible_exit = (terminal.get("agent_exit_code") == 0 or (
        terminal.get("agent_exit_code") == 124 and audit.get("clean") is True))
    return (eligible_exit
            and clean_or_recorded_command_error
            and terminal.get("candidate_sha256") == checkpoint_sha256
            and broker_failure_is_bounded)


def _ready_portfolio_members(row: Mapping[str, Any]) -> frozenset[str]:
    """Exact objective identities whose current candidate analysis is promotion-ready."""
    members = (row.get("portfolio") or {}).get("members") or ()
    return frozenset(
        str((member.get("identity") or {}).get("capsule_sha256"))
        for member in members
        if (member.get("readiness") or {}).get("status") == "ready_for_probe_admission"
    )


def _prior_round_context(stage_root: Path, round_index: int) -> dict[str, Any]:
    """Expose prior agents' own bounded summaries without treating them as host evidence."""
    rows = []
    for index in range(round_index):
        final_path = stage_root / "rounds" / f"round_{index:02d}.final.txt"
        audit_path = stage_root / "global_iterations" / f"agent_round_{index:04d}.json"
        row: dict[str, Any] = {"round": index}
        if final_path.exists() or final_path.is_symlink():
            if final_path.is_symlink() or not final_path.is_file():
                raise ValueError(f"prior round final is not a regular file: {final_path}")
            payload = final_path.read_bytes()
            row["agent_summary_sha256"] = hashlib.sha256(payload).hexdigest()
            row["agent_summary"] = (payload.decode("utf-8") if len(payload) <= 64 * 1024
                                    else None)
            row["agent_summary_omitted"] = len(payload) > 64 * 1024
        if audit_path.exists() or audit_path.is_symlink():
            audit = PAS._mapping_file(audit_path)
            row["host_round_audit"] = {key: copy.deepcopy(audit.get(key)) for key in (
                "status", "candidate_sha256", "agent_exit_code", "refusal_reasons")}
            row["host_round_audit_sha256"] = PAS._sha256_file(audit_path)
        if len(row) > 1:
            rows.append(row)
    return {"schema": "global_prior_round_context_v1", "rounds": rows,
            "interpretation": ("agent summaries are untrusted search memory; host round audits are "
                               "status evidence; neither admits a candidate or proves a speedup")}


def _agent_finalization_reserve_seconds(round_timeout_s: int) -> int:
    """Close the tool broker before the outer turn deadline so Codex can emit final telemetry."""
    if round_timeout_s < 120:
        return 0
    if round_timeout_s < 300:
        return 30
    # A refused last analysis can still require a source revert, a compact evidence summary, and
    # Codex's own final telemetry flush.  Two minutes proved insufficient in a real 600-second
    # authoring round: the agent reverted at the boundary, emitted its final message, and was killed
    # before the transport wrote the final artifact.  Reserve three minutes at the maximum round
    # size; this changes only when tools close, never which candidate can be accepted.
    return min(180, max(60, round_timeout_s // 3))


def run_global_agent_sequence(experiment: GlobalPerfExperiment, candidate: Path, *,
                              run_round: Callable[..., Mapping[str, Any]], stage_root: Path,
                              max_rounds: int, total_authoring_seconds: int,
                              round_seconds: int = 600,
                              on_round_failure: str = "stop") -> dict[str, Any]:
    """Bounded sustained search: exact consumed checkpoints, no silent live-handle retries.

    Authoring budget is reserved per invocation, not reconstructed from optimistic agent
    telemetry. Setup/analysis wall time is separately retained by each iteration. A checkpoint
    certifies structural/provenance review, never a measured global improvement.
    """
    if (min(max_rounds, total_authoring_seconds, round_seconds) <= 0 or round_seconds > 600
            or on_round_failure not in ("stop", "resume-last-checkpoint")):
        raise ValueError("invalid sustained global authoring bounds or continuation policy")
    experiment._check_inputs()
    policy = copy.deepcopy(experiment.host_policy)
    experiment.analyze(candidate, hypothesis="Bind initial seed for safe checkpoint continuation")
    initial_row = experiment._matching_current(candidate, require_ready=False)
    initial_ready = initial_row["readiness"]["status"] == "ready_for_probe_admission"
    initial_seal = (experiment.seal(candidate, name="initial_seed_candidate") if initial_ready
                    else experiment.checkpoint_authoring(candidate, name="initial_seed_authoring"))
    initial = _consume_round_checkpoint(initial_seal)
    spent, failures = 0, []
    checkpoints = [{"round": -1, "role": (
                        "initial_verified_seed_not_an_authored_result" if initial_ready
                        else "initial_blocked_authoring_seed"),
                    "path": str(initial_seal), "sha256": PAS._sha256_file(initial_seal),
                    "candidate_sha256": initial["candidate_sha256"],
                    "checkpoint_schema": initial["schema"],
                    "promotion_ready": initial_ready}]
    current = candidate
    for index in range(max_rounds):
        budget = min(round_seconds, total_authoring_seconds - spent)
        if budget <= 0:
            break
        experiment._check_inputs()
        if experiment.host_policy != policy:
            raise ValueError("sustained segment source policy changed")
        if index > 0:
            checkpoint = _consume_round_checkpoint(Path(checkpoints[-1]["path"]))
            current = PAS.fresh_round_workspace(Path(checkpoint["candidate_path"]),
                stage_root / "agent_workspaces" / f"round_{index:02d}", checkpoint["candidate_sha256"])
        ready_before = _ready_portfolio_members(
            experiment._matching_current(current, require_ready=False))
        spent += budget
        try:
            authored = run_round(current, round_index=index, round_timeout_s=budget)
            if authored.get("status") != "authored":
                raise ValueError("round callback did not return an authored audited checkpoint")
            row = experiment._matching_current(current, require_ready=False)
            lost = sorted(ready_before - _ready_portfolio_members(row))
            if lost:
                raise ValueError(
                    "round regressed previously verified portfolio members: " + ", ".join(lost))
            ready = row["readiness"]["status"] == "ready_for_probe_admission"
            sealed = (experiment.seal(current, name=f"round_{index:04d}_candidate") if ready
                      else experiment.checkpoint_authoring(
                          current, name=f"round_{index:04d}_authoring"))
            consumed = _consume_round_checkpoint(sealed)
            checkpoint = {"round": index, "path": str(sealed), "sha256": PAS._sha256_file(sealed),
                          "candidate_sha256": consumed["candidate_sha256"],
                          "checkpoint_schema": consumed["schema"], "promotion_ready": ready}
            checkpoints.append(checkpoint)
            experiment._write(f"continuation_{index:04d}.json", {
                "schema": "global_round_continuation_v1", "status": "checkpoint_consumed",
                "checkpoint": checkpoint, "authoring_seconds_reserved": spent,
                "promotion_ready": ready, "global_speedup_proven": False})
        except Exception as exc:
            failure = {"round": index, "exception": type(exc).__name__, "reason": str(exc),
                       "draft_path": str(current), "draft_sha256": hash_tree(current)["sha256"],
                       "last_good_checkpoint": checkpoints[-1] if checkpoints else None,
                       "authoring_seconds_reserved": spent, "live_handle_restarted": False}
            # Never classify arbitrary provenance/audit errors as recoverable author failures.
            # The production round writes its terminal audit before raising on a nonzero exit.
            audit_path = experiment.output / f"agent_round_{index:04d}.json"
            terminal = PAS._mapping_file(audit_path) if audit_path.is_file() else {}
            capacity_retry = _retryable_capacity_failure(stage_root, index, terminal)
            unchanged_retry = (bool(checkpoints) and _retryable_unchanged_round_failure(
                terminal, checkpoint_sha256=checkpoints[-1]["candidate_sha256"]))
            recoverable = (on_round_failure == "resume-last-checkpoint" and bool(checkpoints)
                and (unchanged_retry or (
                    terminal.get("agent_exit_code", 0) != 0
                    and terminal.get("audit", {}).get("clean") is True
                    and (terminal.get("broker_evidence", {}).get("all_required_succeeded") is True
                         or capacity_retry))))
            failure["retryable_capacity_failure"] = capacity_retry
            failure["retryable_unchanged_round_failure"] = unchanged_retry
            try:
                experiment._check_inputs()
                if checkpoints:
                    _consume_round_checkpoint(Path(checkpoints[-1]["path"]))
            except Exception:
                recoverable = False
            failure["recovery"] = "next_budgeted_round_from_consumed_checkpoint" if recoverable else "stop"
            failures.append(failure)
            experiment._write(f"continuation_failure_{index:04d}.json", failure)
            if not recoverable:
                raise
    result = {"schema": "global_agent_sequence_v1", "status": "budget_complete",
              "authoring_seconds_reserved": spent, "maximum_rounds": max_rounds,
              "on_round_failure": on_round_failure, "checkpoints": checkpoints, "failures": failures,
              "candidate": str(current), "last_good_checkpoint": checkpoints[-1] if checkpoints else None,
              "promotion_ready": bool(checkpoints and checkpoints[-1]["promotion_ready"]),
              "host_verification_policy": policy, "global_speedup_proven": False}
    experiment._write("agent_sequence.json", result)
    return result


def run_global_agent_round(
        experiment: GlobalPerfExperiment, candidate: Path, *, target_experiment: Any,
        workspace: Path, stage_root: Path, agent_inputs: PAS.AgentInputSnapshot,
        frozen_functional: PAS.FrozenFunctionalInputs, frozen_corpus_manifest: Path,
        model: str, resolved_model: str, effort: str, codex_binary: Path,
        round_index: int, round_timeout_s: int, max_tool_calls: int,
        global_probe_provider: Callable[..., Mapping[str, Any]] | None = None,
        global_semantic_provider: Callable[..., Mapping[str, Any]] | None = None,
        global_context_provider: Callable[..., Mapping[str, Any]] | None = None,
        global_paired_context_provider: Callable[..., Mapping[str, Any]] | None = None,
        global_source_pair_provider: Callable[..., Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Run a real sandboxed authoring round with the full graph as the mandatory objective.

    The caller prepares the existing frozen answer-free grants and explicit model/budgets. This
    reuses the production Codex transport, credential-free tool sandbox, transcript audit and
    telemetry, but never routes the result to the PR/PQ/PK microbenchmark consumer.
    """
    PAS.verify_answer_free_agent_inputs(agent_inputs)
    if experiment.phase1_binding is None:
        raise ValueError("paid macro authoring requires exact existing Phase-1 qualification and waivers")
    if experiment.edit_contract is None:
        raise ValueError("paid macro authoring requires a host-frozen compiler edit authority")
    experiment.validate_candidate_scope(candidate)
    if candidate.parent.resolve() != workspace.resolve():
        raise ValueError("the macro candidate must live in its isolated agent workspace")
    if min(round_timeout_s, max_tool_calls) <= 0:
        raise ValueError("macro agent round budgets must be positive")
    actions = PAS.build_action_registry(candidate, target_experiment, global_optimization=True)
    inner = PAS.inner_execution_policy(target_experiment, candidate, agent_inputs,
                                      frozen_functional, experiment.baseline, frozen_corpus_manifest)
    PAS.run_required_tool_probes(inner, target_experiment, candidate)
    configure_global_analysis(experiment, target_experiment=target_experiment, agent_inputs=agent_inputs,
        frozen_functional=frozen_functional, frozen_corpus_manifest=frozen_corpus_manifest, stage_root=stage_root)
    initial = experiment.analyze(candidate, hypothesis="Inspect the current complete-model global plan")
    finalization_reserve_s = _agent_finalization_reserve_seconds(round_timeout_s)
    broker_window_s = round_timeout_s - finalization_reserve_s
    mandatory_analysis_reserve = experiment.mandatory_analysis_reserve_seconds(broker_window_s)
    mandatory_analysis_reserve_s = mandatory_analysis_reserve["seconds"]
    authoring_tool_window_s = max(0.0, broker_window_s - mandatory_analysis_reserve_s)
    prior_round_context = _prior_round_context(stage_root, round_index)
    portfolio_names = ", ".join(member.capsule for member in experiment.portfolio_sentinels)
    text = (
        "Optimize the fixed portfolio of real full-model graphs and their global plans. "
        f"The training portfolio is: {portfolio_names}. Read STAGE_CONTEXT.json. "
        "It contains a concise initial view; INITIAL_FULL_MODEL_EVIDENCE.json contains the complete "
        "unpruned graph and immutable analysis copy when a transformation needs those details. "
        "Start with portfolio_action_digest: it resolves the primary and secondary member records "
        "into one per-model readiness, work, movement, dispatch, placement and authorized-action view. "
        "The initial exact analysis is already available there; unchanged reanalysis reuses it. "
        "For continuation rounds, prior_round_context in STAGE_CONTEXT.json contains earlier agents' "
        "own untrusted summaries plus host refusal status. Use it as search memory and do not repeat "
        "a disproved or unfinished hypothesis; it is not correctness or performance evidence. "
        "The optimization_surfaces_schema there is the authoritative Phase-2 optional manifest "
        "extension (the frozen Phase-1 schema predates it). Scope must be flag, knob, heuristic, "
        "pass, or codegen, not cca; specify path and exact AST symbol plus every required field. "
        "The host_frozen_edit_authority is the edit permission contract: edit only its exact AST "
        "symbols and explicitly listed helper-extension directories. Candidate manifest entries "
        "describe changes but cannot authorize additional files or symbols. Preserve manifest "
        "execution controls. Imports in approved owning files remain subject to the masked shared "
        "dependency policy. If a needed compiler lever has no approved owner, report that specific "
        "missing surface instead of expanding your own authority. Before editing, state a compact "
        "work order using the contract's required fields: surface/source-operation IDs, current "
        "plan digest, hypothesis, expected emitted delta, semantic obligations, cheap validation "
        "and stop/revert condition. Recompile the entire model portfolio with "
        "analyze-whole-model after each proposed transformation and before finalizing. "
        "Preserve verified emission for every member that is already ready and repair explicitly "
        "blocked members; every member must be verified before promotion or measurement. Prefer "
        "transformations that "
        "improve several model families or remove a shared global bottleneck; do not specialize a "
        "compiler rule to any capsule or model name. Compare each model only with its own prior "
        "revision and use a Pareto decision; never sum unlike models into a fabricated cycle score. "
        "Use full-graph work, movement, representation, residency, synchronization and dispatch "
        "accounting to select transformations. Preserve every source operation and dependency; "
        "capture parsing alone is not candidate emission. Unknown cost remains UNKNOWN. "
        "Short mechanism-equivalent probes may calibrate uncertain costs only through "
        "profile-reduced-global-witness with host-derived identity/equivalence admission. "
        "Exactly one warm invocation precedes exactly one measured compute-cycle invocation. "
        "profile-controlled-context preserves a bounded actual queued-load source prefix and "
        "reports controlled occupancy, never full-model equivalence or global cost calibration. "
        "compare-controlled-context compares identical bounded work between the prior and current "
        "schedule; it is optional and never projects a full-model speedup. Device profiles are "
        "not required for host-only edits or models. Follow host_memory_hotspot evidence to the "
        "exact allocation/buffer identity and declared compiler surface; source attribution may "
        "still be UNKNOWN. Host semantic qualification prioritizes a changed dequantization-to-"
        "contraction mechanism when present, then supported fanout/pointwise mechanisms. "
        "Paired measurements automatically return decision_feedback joined to the exact model "
        "region and compiler surface; use measurement_driven_next_step when present. A prior "
        "revision's measured feedback is search history, not calibration for newly edited bytes. "
        "Use qualify-changed-region for host-selected semantic witnesses of an actual changed "
        "source region when available; this tests reduced mechanisms, not a full-model rerun. "
        "For host-to-convolution lowering changes, prepare-source-convolution requires exactly "
        "comparison_arm=optimization_baseline or comparison_arm=previous. It uses cached full-source "
        "proofs, compiles both reduced source programs, and returns allowed edit surfaces. "
        "Preparation never simulates, admits a runtime, or grants a numerical pass. "
        "For a contraction implementation change, prepare-source-contraction requires comparison_arm, "
        "source_op_index, max_m, max_n and max_k (decimal source index and positive reduced bounds). "
        "It preserves the source scalar semantics and actual input/initializer ABI; use its returned "
        "preparation_sha256 with qualify-source-contraction only if the complete-source-pair provider "
        "is installed. Each action has its own total 60-second budget. The pair executes complete "
        "reduced programs, not a full layer/model; successful short outputs or cycle differences do "
        "not prove the selected full-model task changed via the same route. Respect explicit UNKNOWN "
        "route relevance and never substitute an unrelated primitive or host-chain pass. "
        "Never simulate a complete layer or model during search; FireSim belongs after freeze. "
        "There is no mandatory micro GSIM sweep or micro plateau stopping rule. Preserve the "
        "frozen 92/96 Phase-1 baseline and its waivers; do not run Phase 1 again. Do not modify "
        "harnesses or evaluators. Reuse generalized compiler algorithms and target-derived facts. "
        f"Execute compiler/tools only through python3 {PAS.BROKER_NAME} ACTION [NAME=VALUE ...]. "
        "Do not run Python directly against any path in the candidate workspace, even for "
        "read-only parsing, imports, AST checks or manifest inspection; use jq, sed or rg for "
        "read-only inspection and use the declared broker action for compiler execution. "
        "Broker commands must stand alone: do not pipe them to jq, redirect, or chain them. "
        "Do not place shell or Python commands before or after a broker call in the same command. "
        "analyze-whole-model, qualify-changed-region, inspect-optimization-surfaces and "
        "profile-reduced-global-witness, profile-controlled-context and compare-controlled-context "
        "accept NO NAME=VALUE bindings; do not add HYPOTHESIS=. "
        "The response is compact and links a read-only full evidence file; inspect that file "
        "separately with jq when detailed fields are needed. Full-model analysis is the required "
        "compile action; individual candidate entrypoint smoke commands are optional in macro mode. "
        "At round end state the full-graph transformation, structural evidence, unknown costs, "
        "and remaining semantic/promotion blockers; do not claim measured full-model speedup. "
        f"The host reserves the final {mandatory_analysis_reserve_s:g} seconds of the "
        "tool window for analyze-whole-model. Other broker actions are refused once the preceding "
        f"{authoring_tool_window_s:g}-second authoring/tool window closes. The complete broker "
        f"closes after {broker_window_s} seconds, leaving "
        f"{finalization_reserve_s} seconds for the final response. Finish the final full-model "
        "analysis before that tool deadline, then emit the final response before the round "
        "deadline; a valid intermediate edit does not make a timed-out round complete.\n")
    text += (f"Phase-1 qualification compiler SHA: {experiment.baseline_sha256}. "
             f"Immutable optimization comparison compiler SHA: {experiment.optimization_baseline_sha256}. "
             f"Comparison selection reason: {experiment.optimization_baseline_binding['reason']}. "
             "The comparison seed does not replace or extend Phase-1 qualification; correctness "
             "of these complete-model objectives remains UNPROVEN.\n")
    if experiment.historical_reference is not None:
        text += ("A host-pinned public historical reference bundle is available read-only at "
                 "/perf-control/historical_reference.json; its compact coverage and missing contracts "
                 "are in the optimization brief. These are historical engine-relative references, "
                 "not warm calibration, hardware peaks or target-cycle authority. Use them to choose "
                 "relevant short probes, not to claim full-model speedup.\n")
    external_objectives = []
    for member in experiment.portfolio_sentinels:
        objective_record_path = Path(member.frozen_source_path) / "objective.json"
        if objective_record_path.is_file():
            record = PAS._mapping_file(objective_record_path)
            if record.get("schema") == "external_full_model_objective_v1":
                external_objectives.append(record)
                text += ("One portfolio member is a separately host-pinned external full-model "
                         "objective, not an addition to frozen Phase-1 qualification. Its numeric "
                         f"correctness is UNPROVEN. Inspect its normalized source under {member.capsule_path}; "
                         "do not load capture weights, references or execute normalization scripts.\n")
    prompt_path = workspace / "TASK.md"
    prompt_path.write_text(text)
    prompt = PAS.PromptArtifact(prompt_path, text, PAS._sha256(text.encode()), len(text.encode()))
    PAS._write_json(workspace / "INITIAL_FULL_MODEL_EVIDENCE.json", initial)
    initial_view = agent_analysis_view(initial, complete_evidence="INITIAL_FULL_MODEL_EVIDENCE.json",
                                      context_provider_installed=global_context_provider is not None)
    action_digest = portfolio_action_digest(
        initial, complete_evidence="INITIAL_FULL_MODEL_EVIDENCE.json",
        edit_contract=experiment.edit_contract)
    PAS._write_json(workspace / "STAGE_CONTEXT.json", {
        "mode": "global_perf_experiment_v1", "initial_whole_model_analysis": initial_view,
        "portfolio_action_digest": action_digest,
        "prior_round_context": prior_round_context,
        "host_frozen_edit_authority": experiment.edit_scope_binding if experiment.edit_contract is not None else None,
        "automatic_optimization_inventory": PAS.inspect_compiler_package(candidate).to_dict(),
        "optimization_surfaces_schema": PAS._mapping_file(
            PAS.merlin_dir() / "contract/schemas/manifest.schema.json")["properties"]["optimization_surfaces"],
        "candidate": str(candidate), "broker_actions": [action.as_dict() for action in actions],
        "phase1_sha256": experiment.baseline_sha256, "model_sha256": experiment.sentinel.capsule_sha256,
        "optimization_baseline_sha256": experiment.optimization_baseline_sha256,
        "optimization_baseline": experiment.optimization_baseline_binding,
        "full_model_portfolio": experiment.portfolio_identity,
        "full_model_portfolio_sha256": experiment.portfolio_identity_sha256,
        "portfolio_source_paths": [member.capsule_path for member in experiment.portfolio_sentinels],
        "objective_source_path": experiment.sentinel.capsule_path,
        "external_objective": external_objectives[0] if external_objectives else None,
        "external_objectives": external_objectives,
        "maximum_iteration_seconds": experiment.timeout_s,
        "maximum_round_seconds": round_timeout_s,
        "maximum_tool_window_seconds": broker_window_s,
        "maximum_non_analysis_tool_window_seconds": authoring_tool_window_s,
        "mandatory_analysis_reserve": mandatory_analysis_reserve,
        "finalization_reserve_seconds": finalization_reserve_s,
        "probes_available": global_probe_provider is not None,
        "changed_region_qualification_available": global_semantic_provider is not None,
        "controlled_context_provider_installed": global_context_provider is not None,
        "controlled_context_profile_available": initial_view["controlled_context_capability"]["available"],
        "paired_fixed_work_provider_installed": global_paired_context_provider is not None,
        "complete_source_pair_provider_installed": global_source_pair_provider is not None,
        "paired_fixed_work_comparison_available": False,
        "paired_fixed_work_comparison_status": "requires_two_bound_revisions_and_same_work_projection",
        "promotion_status": "unqualified_until_semantic_and_global_cost_evidence",
    })
    control = stage_root / "global_control" / f"round_{round_index:04d}"
    receipts = control / "receipts.jsonl"
    broker = PAS._Broker(
        inner, target_experiment, candidate, actions, receipts,
        deadline=time.monotonic() + broker_window_s, max_calls=max_tool_calls,
        max_tool_seconds=experiment.timeout_s, global_experiment=experiment,
        mandatory_analysis_reserve_seconds=mandatory_analysis_reserve_s,
        global_probe_provider=global_probe_provider, global_semantic_provider=global_semantic_provider,
        global_context_provider=global_context_provider,
        global_paired_context_provider=global_paired_context_provider,
        global_source_pair_provider=global_source_pair_provider)
    try:
        with broker.serving() as (host, port):
            PAS.stage_broker_shim(control, host=host, port=port, token=broker.token,
                                 tool_timeout_s=experiment.timeout_s, actions=actions)
            experiment.stage_historical_reference(control, workspace=workspace)
            rc, transcript, _ = PAS._codex_round(
                workspace, stage_root, prompt, target_experiment, agent_inputs, frozen_functional,
                experiment.baseline, frozen_corpus_manifest, control,
                model=model, resolved_model=resolved_model, effort=effort, round_index=round_index,
                timeout_s=round_timeout_s, codex_binary=codex_binary)
    finally:
        config = control / ".perf_broker.json"
        if config.is_file() and not config.is_symlink():
            config.chmod(0o600)
            config.unlink()
        if receipts.is_file():
            receipts.chmod(0o444)
    audit = PAS.audit_codex_transcript(transcript, target_experiment, candidate, actions)
    refusals = []
    try:
        evidence = verify_global_broker_receipts(receipts, actions=actions, audit=audit)
    except ValueError as exc:
        evidence = {"status": "refused", "reason": str(exc)}
        refusals.append(str(exc))
    try:
        current = experiment._matching_current(candidate, require_ready=False)
    except ValueError as exc:
        current = {"candidate_sha256": hash_tree(candidate)["sha256"]}
        refusals.append(str(exc))
    try:
        telemetry = PAS._round_telemetry(stage_root, round_index, model=resolved_model, agent_exit_code=rc)
    except PAS.StageGateError as exc:
        telemetry = {"complete": False, "reason": str(exc)}
        refusals.append(str(exc))
    record = {"schema": "global_agent_round_v1", "round": round_index,
              "candidate_sha256": current["candidate_sha256"], "agent_exit_code": rc,
              "audit": audit, "broker_evidence": evidence, "telemetry": telemetry,
              "authoring_readiness": copy.deepcopy(current.get("readiness")),
              "promotion_ready": (current.get("readiness", {}).get("status")
                                  == "ready_for_probe_admission"),
              "status": "authored" if rc == 0 and audit.get("clean") and not refusals else "refused",
              "refusal_reasons": refusals,
              "global_speedup_proven": False}
    experiment._write(f"agent_round_{round_index:04d}.json", record)
    if record["status"] != "authored":
        raise ValueError("macro agent round did not finish with a clean authoring audit")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--model-capsule", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--target-descriptor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--hypothesis", required=True)
    args = parser.parse_args(argv)
    source = args.model_capsule.resolve()
    descriptor = PAS._mapping_file(source / "capsule.yaml", yaml_file=True)
    sentinel = PAS.StageE2ESentinel(
        capsule=str(descriptor.get("id") or source.name), capsule_path=str(source),
        frozen_source_path=str(source),
        capsule_sha256=PAS._exact_tree_record(source)["sha256"],
        required_lanes=tuple((descriptor.get("lanes") or {}).get("require") or ()),
        required_tiers=tuple(descriptor.get("required_oracle_tiers") or ()))
    experiment = GlobalPerfExperiment(
        baseline=args.baseline, baseline_sha256=args.baseline_sha256,
        sentinel=sentinel, target=args.target,
        target_sha256=hashlib.sha256(args.target_descriptor.read_bytes()).hexdigest(),
        target_descriptor=args.target_descriptor, output=args.output, timeout_s=args.timeout_seconds)
    record = experiment.analyze(args.candidate, hypothesis=args.hypothesis)
    print(json.dumps({"readiness": record["readiness"], "elapsed_seconds": record["elapsed_seconds"],
                      "record": str(args.output / "iteration_0000.json")}, indent=2))
    return 0 if record["readiness"]["status"] == "ready_for_probe_admission" else 2


if __name__ == "__main__":
    raise SystemExit(main())
