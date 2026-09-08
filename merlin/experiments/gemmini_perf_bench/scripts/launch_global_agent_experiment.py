#!/usr/bin/env python3
"""Launch full-graph authoring with optional bounded isolated probes, never model simulation."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import perf_agent_stage as PAS
from merlin.benchharness import hash_tree
from merlin.perf.host_resources import (HostResourcePolicy, HostResourceTripwire,
                                        sample_host_memory, summarize_samples, violations)
from merlin.perf.execution_policy import (FULL_GRAPH_STATIC_ANALYSIS_MAX_SECONDS,
                                          GLOBAL_AUTHORING_ROUND_MAX_SECONDS)
from merlin.targetgen.target_experiment import load_target_experiment
from run_global_perf_experiment import (FrozenPhase1, GlobalPerfExperiment, configure_global_analysis,
                                        full_model_portfolio_identity,
                                        run_global_agent_round, run_global_agent_sequence,
                                        verify_retained_global_checkpoint,
                                        validate_optimization_baseline_resume)


_GIB = 1024 ** 3


def _mechanism_catalog_worker_arguments(path: Path | None, digest: str | None) -> tuple[str, ...]:
    """Forward the already-validated raw pin without resolving or rediscovering it."""
    if path is None:
        return ()
    return ("--mechanism-catalog", str(path), "--mechanism-catalog-sha256", str(digest))


def _mechanism_work_order_worker_arguments(
        path: Path | None, digest: str | None) -> tuple[str, ...]:
    """Forward one already-validated immutable host work-order pin exactly."""
    if path is None:
        return ()
    return ("--mechanism-work-order", str(path),
            "--mechanism-work-order-sha256", str(digest))


def _acquire_host_resource_lease(stage_root: Path):
    """Exclude another heavy compiler experiment without claiming machine-wide ownership."""
    lease_path = PAS.repo_root() / "out/artifacts/cache/host_resources/full_model_perf.lock"
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lease_path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        stage_root.mkdir(parents=True, exist_ok=True)
        PAS._write_json(stage_root / "host_resource_telemetry.json", {
            "schema": "host_resource_telemetry_v1", "status": "launch_refused",
            "reasons": ["host_resource_lease_busy"], "lease": str(lease_path),
            "worker_returncode": None})
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "stage_root": str(stage_root)}) + "\n")
    handle.flush()
    return handle


def _descendant_pids(root_pid: int, proc_root: Path = Path("/proc")) -> tuple[int, ...]:
    """Snapshot descendants without depending on a process-management package."""
    children: dict[int, list[int]] = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            line = next(row for row in (entry / "status").read_text().splitlines()
                        if row.startswith("PPid:"))
            parent = int(line.partition(":")[2].strip())
        except (OSError, StopIteration, ValueError):
            continue
        children.setdefault(parent, []).append(int(entry.name))
    result, pending = [], list(children.get(root_pid, ()))
    while pending:
        pid = pending.pop()
        result.append(pid)
        pending.extend(children.get(pid, ()))
    return tuple(result)


def _signal_worker_tree(process: subprocess.Popen, sig: signal.Signals) -> None:
    # Some nested tools create their own sessions. Signal descendants directly before the source
    # worker's group so those sessions cannot outlive a resource-triggered controller shutdown.
    for pid in reversed(_descendant_pids(process.pid)):
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        os.killpg(process.pid, sig)
    except (ProcessLookupError, PermissionError):
        try:
            process.send_signal(sig)
        except ProcessLookupError:
            pass


def _stop_worker_tree(process: subprocess.Popen, *, grace_s: float = 5.0) -> None:
    known_descendants = set(_descendant_pids(process.pid))
    for sig in (signal.SIGTERM, signal.SIGKILL):
        known_descendants.update(_descendant_pids(process.pid))
        for pid in reversed(tuple(known_descendants)):
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass
        _signal_worker_tree(process, sig)
        try:
            process.wait(timeout=grace_s)
            # A nested new-session child is reparented when the leader exits; retain its original PID.
            for pid in reversed(tuple(known_descendants)):
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            return
        except subprocess.TimeoutExpired:
            continue


def _run_resource_guarded_worker(command: list[str], environment: dict[str, str], *,
                                 stage_root: Path, policy: HostResourcePolicy,
                                 sample_seconds: float) -> int:
    """Supervise the complete experiment tree and retain only memory extrema/endpoints."""
    samples = [sample_host_memory()]
    initial_reasons = violations(samples[-1], policy)
    if initial_reasons:
        stage_root.mkdir(parents=True, exist_ok=True)
        PAS._write_json(stage_root / "host_resource_telemetry.json", {
            "schema": "host_resource_telemetry_v1", "status": "launch_refused",
            "reasons": list(initial_reasons), "policy": policy.record(),
            "summary": summarize_samples(samples), "worker_returncode": None})
        return 75
    process = subprocess.Popen(command, env=environment, start_new_session=True)
    guard = HostResourceTripwire(policy)
    stop_decision = None
    returncode = None
    try:
        while returncode is None:
            try:
                returncode = process.wait(timeout=sample_seconds)
            except subprocess.TimeoutExpired:
                samples.append(sample_host_memory())
                decision = guard.observe(samples[-1])
                if decision["status"] == "stop":
                    stop_decision = dict(decision)
                    _stop_worker_tree(process)
                    returncode = process.returncode
    except BaseException:
        _stop_worker_tree(process)
        raise
    finally:
        samples.append(sample_host_memory())
        stage_root.mkdir(parents=True, exist_ok=True)
        PAS._write_json(stage_root / "host_resource_telemetry.json", {
            "schema": "host_resource_telemetry_v1",
            "status": "resource_limit" if stop_decision is not None else "completed",
            "policy": policy.record(), "sample_period_seconds": sample_seconds,
            "summary": summarize_samples(samples), "trip": stop_decision,
            "worker_returncode": returncode})
    return 75 if stop_decision is not None else int(returncode)


def run_analysis_only(experiment, candidate: Path, *, stage_root: Path,
                      static_analysis_seed_checkpoint: Path | None = None,
                      static_analysis_seed_sha256: str | None = None,
                      **analysis_policy) -> int:
    """Use the real compile/static-analysis boundary without authoring, probes or a seal."""
    configure_global_analysis(experiment, stage_root=stage_root, **analysis_policy)
    if static_analysis_seed_checkpoint is not None:
        experiment.import_static_analysis_checkpoint(
            candidate, checkpoint=static_analysis_seed_checkpoint,
            checkpoint_sha256=static_analysis_seed_sha256)
    analysis = experiment.analyze(candidate, hypothesis="Host-requested full-objective compile/static preflight")
    result = {"schema": "global_analysis_only_v1", "readiness": analysis["readiness"],
        "candidate_sha256": analysis["candidate_sha256"],
        "portfolio_sha256": experiment.portfolio_identity_sha256,
        "portfolio_members_ready": analysis["portfolio"]["members_ready"],
        "portfolio_members_total": analysis["portfolio"]["members_total"],
        "baseline_sha256": experiment.baseline_sha256,
        "optimization_baseline_sha256": experiment.optimization_baseline_sha256,
        "iteration_record": str(experiment.output / f"iteration_{analysis['iteration']:04d}.json"),
        "phase1_rerun": False, "authoring_launched": False, "simulators_executed": False,
        "candidate_sealed": False, "objective_numerical_qualification": "UNPROVEN",
        "global_speedup_proven": False}
    PAS._write_json(stage_root / "analysis_only.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if analysis["readiness"]["status"] == "ready_for_probe_admission" else 1


def run_authoring_with_terminal_receipt(stage_root: Path, *, configure, sequence):
    """Persist pre-round/setup exceptions too; retain the original exception and fail closed."""
    stage = "configure_global_analysis"
    try:
        configure()
        stage = "initial_analysis_or_authoring_sequence"
        return sequence()
    except Exception as exc:
        PAS._write_json(stage_root / "terminal_failure.json", {
            "schema": "global_launch_terminal_failure_v1", "status": "failed",
            "stage": stage, "exception": type(exc).__name__, "reason": str(exc),
            "completed_round_receipts": len(list((stage_root / "global_iterations").glob("agent_round_*.json"))),
            "iteration_receipts": [str(path) for path in sorted((stage_root / "global_iterations").glob("iteration_*.json"))],
            "promotion_status": "unqualified", "global_speedup_proven": False,
            "phase1_rerun": False, "live_handle_restarted": False})
        raise


def load_host_guidance_declarations(contract_path: Path, catalog_receipt):
    """Load only the inventory explicitly pinned by the supplied host catalog receipt.

    Legacy catalogs without this pin retain manifest-only guidance. Mere proximity
    to the edit contract is not authority to load an unpinned semantic inventory.
    The controller subsequently checks every declaration against actual AST and
    component ownership and the unchanged edit contract.
    """
    digest = catalog_receipt.get("guidance_inventory_sha256")
    if digest is None:
        digest = (catalog_receipt.get("unchanged_catalog_file_sha256") or {}).get("inventory.json")
    if digest is None:
        return None
    path = contract_path.parent / "inventory.json"
    if not PAS._is_sha256(digest) or path.is_symlink():
        raise ValueError("host guidance inventory pin/path is invalid")
    raw = path.read_bytes()
    if len(raw) > 4_000_000 or PAS._sha256(raw) != digest:
        raise ValueError("host guidance inventory differs from pinned catalog bytes")
    inventory = json.loads(raw)
    if not isinstance(inventory, dict) or not isinstance(inventory.get("surfaces"), list):
        raise ValueError("host guidance inventory has no surface declarations")
    return inventory["surfaces"]


def main(argv: list[str] | None = None) -> int:
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-config", type=Path, required=True,
                        help="existing campaign config or suite JSON carrying frozen run identity/waivers")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--optimization-baseline", type=Path,
                        help="explicit immutable comparison compiler; never replaces frozen Phase-1 qualification")
    parser.add_argument("--optimization-baseline-sha256",
                        help="required exact compiler-tree SHA-256 for optimization-baseline")
    parser.add_argument("--optimization-baseline-reason",
                        default="host-selected immutable optimization comparison seed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-emission-cache", type=Path,
                        help="persistent host-owned exact baseline-emission cache; defaults beside output")
    parser.add_argument("--baseline-emission-cache-seed-run", type=Path, action="append", default=[],
                        help="prior run whose exact successful baseline emissions seed the cache")
    parser.add_argument("--portfolio-analysis-workers", type=int, default=4,
                        help="maximum host-admitted concurrent full-model analysis workers")
    parser.add_argument("--historical-reference", type=Path,
                        help="explicit public historical reference bundle; not target timing calibration")
    parser.add_argument("--historical-reference-sha256",
                        help="required SHA-256 of the exact host-owned historical bundle")
    objective = parser.add_mutually_exclusive_group()
    objective.add_argument("--objective-capsule",
                        help="exact public full-model objective from the existing frozen functional inputs")
    objective.add_argument("--external-objective", type=Path,
                        help="host JSON spec for one pinned already-normalized external complete-model source")
    parser.add_argument("--external-objective-sha256",
                        help="required exact SHA-256 of the external objective spec")
    parser.add_argument("--portfolio-capsule", action="append", default=[],
                        help="additional frozen full-model training objective; repeat for a portfolio")
    parser.add_argument("--portfolio-external-objective", type=Path, action="append", default=[],
                        help="pinned external full-model training member; repeat for a portfolio")
    parser.add_argument("--portfolio-external-objective-sha256", action="append", default=[],
                        help="exact SHA-256 paired by order with portfolio-external-objective")
    parser.add_argument("--round-seconds", type=int, default=600)
    parser.add_argument(
        "--iteration-seconds", type=int, default=600,
        help=("host-only full-graph compile/static-analysis ceiling; at most "
              f"{FULL_GRAPH_STATIC_ANALYSIS_MAX_SECONDS:g}s; does not extend authoring rounds or "
              "permit full-model simulation"))
    parser.add_argument("--max-tool-calls", type=int, default=40)
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--total-authoring-seconds", type=int,
                        help="total reserved authoring budget across bounded rounds; defaults to one round")
    parser.add_argument("--on-round-failure", choices=("stop", "resume-last-checkpoint"), default="stop")
    parser.add_argument("--resume-checkpoint", type=Path,
                        help="exact prior sealed candidate; starts an explicit newly frozen policy segment")
    parser.add_argument("--static-analysis-seed-checkpoint", type=Path,
                        help="explicit prior global candidate whose static analysis may be imported")
    parser.add_argument("--static-analysis-seed-sha256",
                        help="required exact SHA-256 of static-analysis-seed-checkpoint")
    parser.add_argument("--edit-contract", type=Path,
                        help="host-approved edit_contract.json with sibling receipt.json initial source-file pins")
    parser.add_argument("--mechanism-catalog", type=Path,
                        help="absolute immutable host compiler_mechanism_catalog_v1 JSON")
    parser.add_argument("--mechanism-catalog-sha256",
                        help="required exact raw-file SHA-256 for mechanism-catalog")
    parser.add_argument("--mechanism-work-order", type=Path,
                        help="absolute immutable host_prepared_mechanism_work_order_v1 JSON")
    parser.add_argument("--mechanism-work-order-sha256",
                        help="required exact raw-file SHA-256 for mechanism-work-order")
    parser.add_argument("--validation-only", action="store_true",
                        help="validate and seal an existing checkpoint without a new paid authoring round")
    parser.add_argument("--analysis-only", action="store_true",
                        help="one compile/static analysis only; no Codex, telemetry, probes or sealing")
    parser.add_argument("--comparison-candidate", type=Path,
                        help="preserved pre-edit compiler for changed-region validation")
    parser.add_argument("--compare-controlled-context", action="store_true",
                        help="compare identical bounded queued work across the two validation revisions")
    parser.add_argument("--semantic-only", action="store_true",
                        help="compile and qualify changed host regions without any device simulation")
    parser.add_argument("--probe-interface", type=Path,
                        help="host-selected separate short interface, never the full-model objective")
    parser.add_argument("--probe-runtime-receipt", type=Path,
                        help="existing exact-ELF/engine warm diagnostic used for wall-time admission")
    parser.add_argument("--probe-profile", choices=("none", "occupancy"), default="none",
                        help="optional minimal joint-busy counters on the isolated primitive only")
    parser.add_argument("--source-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--min-memory-available-gib", type=int, default=16,
                        help="refuse/stop before host MemAvailable falls below this bound")
    parser.add_argument("--max-swap-used-gib", type=int, default=2,
                        help="refuse/stop after host swap use exceeds this bound")
    parser.add_argument("--resource-sample-seconds", type=float, default=2.0,
                        help="host memory/swap supervision interval")
    parser.add_argument("--resource-trip-samples", type=int, default=2,
                        help="consecutive pressured samples required while a run is active")
    args = parser.parse_args(argv)
    if args.baseline_emission_cache is None:
        args.baseline_emission_cache = (
            args.output.resolve().parent / "_global_phase2_baseline_emission_cache_v1")
    else:
        args.baseline_emission_cache = args.baseline_emission_cache.resolve()
    if any(path.is_symlink() or not path.is_dir()
           for path in args.baseline_emission_cache_seed_run):
        parser.error("baseline-emission-cache-seed-run must name a real prior run directory")
    if (min(args.min_memory_available_gib, args.max_swap_used_gib) < 0
            or args.portfolio_analysis_workers < 1
            or args.resource_trip_samples < 1 or not math.isfinite(args.resource_sample_seconds)
            or not 0.25 <= args.resource_sample_seconds <= 30.0):
        parser.error("host resource limits/sample interval are invalid")
    resource_policy = HostResourcePolicy(
        minimum_memory_available_bytes=args.min_memory_available_gib * _GIB,
        maximum_swap_used_bytes=args.max_swap_used_gib * _GIB,
        consecutive_violations_to_stop=args.resource_trip_samples)
    if bool(args.historical_reference) != bool(args.historical_reference_sha256):
        parser.error("historical-reference requires its exact historical-reference-sha256 pin")
    if bool(args.static_analysis_seed_checkpoint) != bool(args.static_analysis_seed_sha256):
        parser.error("static-analysis seed requires both checkpoint path and exact SHA-256")
    if args.static_analysis_seed_checkpoint:
        seed = args.static_analysis_seed_checkpoint
        if (not PAS._is_sha256(args.static_analysis_seed_sha256) or seed.is_symlink()
                or not seed.is_file() or PAS._sha256_file(seed) != args.static_analysis_seed_sha256):
            parser.error("static-analysis seed checkpoint is linked, absent, or differs from its pin")
    if bool(args.mechanism_catalog) != bool(args.mechanism_catalog_sha256):
        parser.error("mechanism-catalog requires its exact mechanism-catalog-sha256 pin")
    if args.mechanism_catalog:
        catalog = args.mechanism_catalog
        if not args.edit_contract:
            parser.error("mechanism-catalog requires an explicit host edit-contract")
        if (not PAS._is_sha256(args.mechanism_catalog_sha256) or not catalog.is_absolute()
                or catalog.resolve() != catalog or catalog.is_symlink() or not catalog.is_file()
                or catalog.stat().st_mode & 0o222
                or PAS._sha256_file(catalog) != args.mechanism_catalog_sha256):
            parser.error("mechanism-catalog must be an exact immutable absolute file")
    if bool(args.mechanism_work_order) != bool(args.mechanism_work_order_sha256):
        parser.error("mechanism-work-order requires its exact mechanism-work-order-sha256 pin")
    if args.mechanism_work_order:
        work_order = args.mechanism_work_order
        if not args.mechanism_catalog:
            parser.error("mechanism-work-order requires an explicit mechanism-catalog")
        if (not PAS._is_sha256(args.mechanism_work_order_sha256)
                or not work_order.is_absolute() or work_order.resolve() != work_order
                or work_order.is_symlink() or not work_order.is_file()
                or work_order.stat().st_mode & 0o222
                or PAS._sha256_file(work_order) != args.mechanism_work_order_sha256):
            parser.error("mechanism-work-order must be an exact immutable absolute file")
    if args.validation_only and args.static_analysis_seed_checkpoint:
        parser.error("validation-only compares two revisions and cannot import an initial static seed")
    if args.historical_reference:
        from run_global_perf_experiment import load_historical_reference
        load_historical_reference(args.historical_reference, args.historical_reference_sha256,
                                  candidate_roots=(args.candidate,))
    if args.analysis_only and (args.validation_only or args.resume_checkpoint or args.comparison_candidate
            or args.probe_interface or args.probe_runtime_receipt or args.probe_profile != "none"
            or args.compare_controlled_context or args.semantic_only or args.max_rounds != 1
            or args.total_authoring_seconds is not None or args.edit_contract
            or args.mechanism_catalog or args.mechanism_work_order):
        parser.error("analysis-only excludes authoring/resume, qualification and profiling options")
    if bool(args.optimization_baseline) != bool(args.optimization_baseline_sha256):
        parser.error("optimization-baseline requires its exact optimization-baseline-sha256 pin")
    if args.optimization_baseline and (not PAS._is_sha256(args.optimization_baseline_sha256)
            or hash_tree(args.optimization_baseline)["sha256"] != args.optimization_baseline_sha256):
        parser.error("optimization-baseline digest does not match the explicit source")
    if bool(args.external_objective) != bool(args.external_objective_sha256):
        parser.error("external-objective requires its exact external-objective-sha256 pin")
    if len(args.portfolio_external_objective) != len(args.portfolio_external_objective_sha256):
        parser.error("each portfolio-external-objective requires its ordered exact SHA-256 pin")
    if len(set(args.portfolio_capsule)) != len(args.portfolio_capsule):
        parser.error("portfolio-capsule members must be distinct")
    total_authoring = args.total_authoring_seconds if args.total_authoring_seconds is not None else args.round_seconds
    if (min(args.max_rounds, total_authoring, args.round_seconds) <= 0
            or args.round_seconds > GLOBAL_AUTHORING_ROUND_MAX_SECONDS):
        parser.error(
            "authoring bounds must be positive and each round at most "
            f"{GLOBAL_AUTHORING_ROUND_MAX_SECONDS:g} seconds")
    if not 0 < args.iteration_seconds <= FULL_GRAPH_STATIC_ANALYSIS_MAX_SECONDS:
        parser.error(
            "iteration-seconds is the host-only full-graph static-analysis ceiling and must be "
            f"in (0, {FULL_GRAPH_STATIC_ANALYSIS_MAX_SECONDS:g}]")
    if args.validation_only and (args.max_rounds != 1 or args.resume_checkpoint or args.total_authoring_seconds):
        parser.error("validation-only does not launch or resume an authoring sequence")
    if args.validation_only and args.mechanism_catalog:
        parser.error("mechanism-catalog is for an authored round, not validation-only comparison")
    if args.validation_only != bool(args.comparison_candidate):
        parser.error("validation-only requires exactly one comparison-candidate")
    if args.compare_controlled_context and not args.validation_only:
        parser.error("compare-controlled-context requires validation-only and comparison-candidate")
    if args.semantic_only and (not args.validation_only or args.compare_controlled_context
                               or args.probe_interface or args.probe_runtime_receipt):
        parser.error("semantic-only requires validation-only and excludes device profiling options")
    if bool(args.probe_interface) != bool(args.probe_runtime_receipt):
        parser.error("probe interface and runtime receipt must be supplied together")
    if args.probe_profile != "none" and not args.probe_interface:
        parser.error("an isolated probe profile requires its interface and matching runtime receipt")
    config_document = PAS._mapping_file(args.campaign_config)
    config = (config_document["campaigns"][0]["config"] if "campaigns" in config_document
              else config_document.get("config", config_document))
    if not args.source_worker:
        import perf_snapshot
        source = PAS.repo_root().resolve()
        snapshot = args.output.resolve().with_name(args.output.name + ".source")
        lease = _acquire_host_resource_lease(args.output.resolve())
        if lease is None:
            return 75
        descriptor = PAS._mapping_file(Path(config["descriptor"]), yaml_file=True)
        target_name = str(descriptor["target"])
        try:
            before_snapshot = sample_host_memory()
            pressure = violations(before_snapshot, resource_policy)
            if pressure:
                args.output.resolve().mkdir(parents=True, exist_ok=True)
                PAS._write_json(args.output.resolve() / "host_resource_telemetry.json", {
                    "schema": "host_resource_telemetry_v1", "status": "launch_refused",
                    "reasons": list(pressure), "policy": resource_policy.record(),
                    "summary": summarize_samples([before_snapshot]), "worker_returncode": None})
                return 75
            perf_snapshot.create(source, snapshot, output_root=source / "out", target_name=target_name)
            environment = {**os.environ, "MERLIN_REPO_ROOT": str(snapshot),
                           "MERLIN_OUT_ROOT": str(source / "out"), "PYTHONDONTWRITEBYTECODE": "1",
                           "PYTHONPATH": os.pathsep.join(str(snapshot / item) for item in (
                               "merlin/python", "merlin/experiments/gemmini_perf_bench/scripts",
                               "merlin/experiments/capsule_bench/harness"))}
            command = [sys.executable, str(snapshot / Path(__file__).resolve().relative_to(source)),
                       "--campaign-config", str(args.campaign_config.resolve()),
                       "--candidate", str(args.candidate.resolve()), "--output", str(args.output.resolve()),
                       "--round-seconds", str(args.round_seconds), "--iteration-seconds", str(args.iteration_seconds),
                       "--max-tool-calls", str(args.max_tool_calls), "--source-worker"]
            command.extend(("--baseline-emission-cache", str(args.baseline_emission_cache)))
            command.extend(("--portfolio-analysis-workers", str(args.portfolio_analysis_workers)))
            for seed_run in args.baseline_emission_cache_seed_run:
                command.extend(("--baseline-emission-cache-seed-run", str(seed_run.resolve())))
            command.extend(("--max-rounds", str(args.max_rounds),
                            "--on-round-failure", args.on_round_failure,
                            "--min-memory-available-gib", str(args.min_memory_available_gib),
                            "--max-swap-used-gib", str(args.max_swap_used_gib),
                            "--resource-sample-seconds", str(args.resource_sample_seconds),
                            "--resource-trip-samples", str(args.resource_trip_samples)))
            if args.total_authoring_seconds is not None:
                command.extend(("--total-authoring-seconds", str(total_authoring)))
            if args.resume_checkpoint:
                command.extend(("--resume-checkpoint", str(args.resume_checkpoint.resolve())))
            if args.static_analysis_seed_checkpoint:
                command.extend(("--static-analysis-seed-checkpoint",
                                str(args.static_analysis_seed_checkpoint.resolve()),
                                "--static-analysis-seed-sha256",
                                args.static_analysis_seed_sha256))
            if args.optimization_baseline:
                command.extend(("--optimization-baseline", str(args.optimization_baseline.resolve()),
                                "--optimization-baseline-sha256", args.optimization_baseline_sha256,
                                "--optimization-baseline-reason", args.optimization_baseline_reason))
            if args.edit_contract:
                command.extend(("--edit-contract", str(args.edit_contract.resolve())))
            if args.mechanism_catalog:
                command.extend(_mechanism_catalog_worker_arguments(
                    args.mechanism_catalog, args.mechanism_catalog_sha256))
            if args.mechanism_work_order:
                command.extend(_mechanism_work_order_worker_arguments(
                    args.mechanism_work_order, args.mechanism_work_order_sha256))
            if args.historical_reference:
                command.extend(("--historical-reference", str(args.historical_reference.resolve()),
                                "--historical-reference-sha256", args.historical_reference_sha256))
            if args.probe_interface:
                command.extend(("--probe-interface", str(args.probe_interface.resolve()),
                                "--probe-runtime-receipt", str(args.probe_runtime_receipt.resolve()),
                                "--probe-profile", args.probe_profile))
            if args.validation_only:
                command.extend(("--validation-only", "--comparison-candidate",
                                str(args.comparison_candidate.resolve())))
            if args.analysis_only:
                command.append("--analysis-only")
            if args.compare_controlled_context:
                command.append("--compare-controlled-context")
            if args.semantic_only:
                command.append("--semantic-only")
            if args.objective_capsule:
                command.extend(("--objective-capsule", args.objective_capsule))
            if args.external_objective:
                command.extend(("--external-objective", str(args.external_objective.resolve()),
                                "--external-objective-sha256", args.external_objective_sha256))
            for capsule in args.portfolio_capsule:
                command.extend(("--portfolio-capsule", capsule))
            for path, digest in zip(args.portfolio_external_objective,
                                    args.portfolio_external_objective_sha256, strict=True):
                command.extend(("--portfolio-external-objective", str(path.resolve()),
                                "--portfolio-external-objective-sha256", digest))
            return _run_resource_guarded_worker(
                command, environment, stage_root=args.output.resolve(), policy=resource_policy,
                sample_seconds=args.resource_sample_seconds)
        finally:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            lease.close()
    import perf_snapshot
    snapshot_receipt = perf_snapshot.verify(PAS.repo_root())
    target = load_target_experiment(Path(config["descriptor"]))
    run_root = PAS.runs_root(target.target, "capsule-bench")
    functional = PAS.inspect_stage_functional_run(
        run_root, str(config["functional_run_id"]), str(config["functional_submission_sha256"]),
        waive=tuple(config["waive_functional_gate"]))
    gaps = tuple(sorted(str(row["capsule"]) for row in functional.public_score["per_capsule"]
                        if row.get("status") != "pass"))
    phase1 = FrozenPhase1(run_root, functional.run_id, functional.digest,
                         tuple(config["waive_functional_gate"]), 92, 96, gaps)
    phase1.verify(functional.submission_dir)
    if args.output.exists():
        raise ValueError("macro stage output must be fresh")
    stage_root = args.output.resolve()
    stage_root.mkdir(parents=True)
    base = PAS.PC.materialize_perf_workspace(functional, stage_root / "_frozen_functional")
    frozen_functional = PAS.load_frozen_functional_inputs(functional)
    from merlin.perf.external_objective import load_external_objective
    primary_external = (load_external_objective(
        args.external_objective.resolve(), spec_sha256=args.external_objective_sha256,
        max_source_bytes=64*1024*1024) if args.external_objective else None)
    portfolio_externals = [
        load_external_objective(path.resolve(), spec_sha256=digest,
                                max_source_bytes=64*1024*1024)
        for path, digest in zip(args.portfolio_external_objective,
                                args.portfolio_external_objective_sha256, strict=True)]
    external_objectives = ([primary_external] if primary_external is not None else []) + portfolio_externals
    corpus = PAS.freeze_performance_corpus(
        PAS.discover_performance_corpus(target, families="all", capsules="all"),
        stage_root / "_frozen_corpus")
    inputs = PAS.build_answer_free_agent_inputs(
        corpus, target, stage_root / "_agent_inputs",
        external_objective=(primary_external if primary_external is not None
                            and not portfolio_externals else None),
        external_objectives=(tuple(external_objectives)
                             if portfolio_externals else ()))
    sentinel = (PAS.select_external_e2e_sentinel(primary_external, inputs) if args.external_objective else
                PAS.select_e2e_sentinel(functional, frozen_functional, target,
                                        objective_capsule=args.objective_capsule))
    portfolio_sentinels = [
        PAS.select_e2e_sentinel(functional, frozen_functional, target,
                                objective_capsule=capsule)
        for capsule in args.portfolio_capsule]
    portfolio_sentinels.extend(
        PAS.select_external_e2e_sentinel(external, inputs)
        for external in portfolio_externals)
    portfolio_identity = full_model_portfolio_identity((sentinel, *portfolio_sentinels))
    portfolio_sha256 = PAS._document_sha256(portfolio_identity)
    resumed = None
    if args.resume_checkpoint:
        resumed = verify_retained_global_checkpoint(args.resume_checkpoint.resolve())
        validate_optimization_baseline_resume(resumed, optimization_baseline_sha256=(
            args.optimization_baseline_sha256 if args.optimization_baseline else functional.digest))
        resumed_portfolio_sha256 = resumed.get("portfolio_sha256")
        resume_portfolio_matches = (
            resumed_portfolio_sha256 == portfolio_sha256
            or (resumed_portfolio_sha256 is None and len(portfolio_sentinels) == 0))
        if (resumed["candidate_sha256"] != hash_tree(args.candidate)["sha256"]
                or resumed["baseline_sha256"] != functional.digest
                or resumed["target_sha256"] != target.descriptor_sha256
                or resumed["capsule_sha256"] != sentinel.capsule_sha256
                or not resume_portfolio_matches
                or resumed["phase1_qualification"] != phase1.verify(base)):
            raise ValueError("resume checkpoint candidate, objective, target or frozen qualification differs")
    candidate = PAS.fresh_round_workspace(
        args.candidate.resolve(), stage_root / "agent_workspaces" / "round_00",
        hash_tree(args.candidate)["sha256"])
    codex, resolved_model = None, None
    if not args.validation_only and not args.analysis_only:
        codex = PAS._require_executable("codex", label="Codex")
        telemetry = PAS.telemetry_preflight(
            model=str(config["model"]), price_table=Path(config["telemetry_price_table"]),
            codex_binary=codex)
        PAS._write_json(stage_root / "telemetry_preflight.json", telemetry)
        resolved_model = str(telemetry["model_resolution"]["resolved_model"])
    experiment = GlobalPerfExperiment(
        baseline=base, baseline_sha256=functional.digest, sentinel=sentinel,
        portfolio_sentinels=portfolio_sentinels,
        target=target.target, target_sha256=target.descriptor_sha256,
        target_descriptor=target.path, phase1=phase1,
        optimization_baseline=args.optimization_baseline,
        optimization_baseline_sha256=args.optimization_baseline_sha256,
        optimization_baseline_reason=args.optimization_baseline_reason,
        historical_reference_path=args.historical_reference,
        historical_reference_sha256=args.historical_reference_sha256,
        baseline_emission_cache=args.baseline_emission_cache,
        baseline_emission_seed_runs=tuple(args.baseline_emission_cache_seed_run),
        portfolio_analysis_workers=args.portfolio_analysis_workers,
        minimum_memory_available_bytes=resource_policy.minimum_memory_available_bytes,
        source_snapshot_root=PAS.repo_root(),
        source_snapshot_files_sha256=PAS._document_sha256(snapshot_receipt["files"]),
        output=stage_root / "global_iterations", timeout_s=args.iteration_seconds)
    if args.analysis_only:
        PAS._write_json(stage_root / "launch.json", {
            "schema": "global_agent_launch_v1", "mode": "analysis_only",
            "historical_reference": experiment.historical_reference,
            "source_config": str(args.campaign_config.resolve()),
            "source_config_sha256": PAS._sha256_file(args.campaign_config),
            "source_snapshot": str(PAS.repo_root()),
            "source_snapshot_files_sha256": PAS._document_sha256(snapshot_receipt["files"]),
            "phase1": experiment.phase1_binding, "baseline_sha256": experiment.baseline_sha256,
            "optimization_baseline": experiment.optimization_baseline_binding,
            "optimization_baseline_sha256": experiment.optimization_baseline_sha256,
            "candidate_source_sha256": hash_tree(args.candidate)["sha256"],
            "objective": sentinel.capsule, "capsule_sha256": sentinel.capsule_sha256,
            "portfolio": experiment.portfolio_identity,
            "portfolio_sha256": experiment.portfolio_identity_sha256,
            "baseline_emission_cache": experiment.baseline_emission_cache_binding,
            "baseline_emission_cache_seeds": experiment.baseline_emission_cache_seeds,
            "external_objective": primary_external.record() if primary_external is not None else None,
            "external_objectives": [external.record() for external in external_objectives],
            "external_objective_spec_sha256": args.external_objective_sha256,
            "portfolio_external_objective_spec_sha256": args.portfolio_external_objective_sha256,
            "full_model_simulation_allowed": False, "simulators_enabled": False,
            "static_analysis_seed": ({"path": str(args.static_analysis_seed_checkpoint.resolve()),
                "sha256": args.static_analysis_seed_sha256}
                if args.static_analysis_seed_checkpoint else None),
            "authoring_launched": False, "iteration_seconds": args.iteration_seconds,
            "host_resource_policy": resource_policy.record()})
        return run_analysis_only(experiment, candidate, stage_root=stage_root,
            target_experiment=target, agent_inputs=inputs, frozen_functional=frozen_functional,
            frozen_corpus_manifest=corpus.manifest_path,
            static_analysis_seed_checkpoint=(args.static_analysis_seed_checkpoint.resolve()
                if args.static_analysis_seed_checkpoint else None),
            static_analysis_seed_sha256=args.static_analysis_seed_sha256)
    if not args.validation_only:
        from merlin.perf.agent_guidance import build_compiler_edit_contract
        source_pins = None
        host_surfaces = None
        if args.edit_contract:
            contract = PAS._mapping_file(args.edit_contract)
            catalog_receipt = PAS._mapping_file(args.edit_contract.parent / "receipt.json")
            if catalog_receipt.get("contract_sha256") != contract.get("sha256"):
                raise ValueError("host edit catalog receipt does not bind the supplied contract")
            source_pins = catalog_receipt["source_files"]
            host_surfaces = load_host_guidance_declarations(args.edit_contract, catalog_receipt)
        else:
            contract = build_compiler_edit_contract(PAS.inspect_compiler_package(candidate))
        experiment.freeze_edit_scope(candidate, contract, source_pins=source_pins,
                                     host_surface_declarations=host_surfaces)
        if args.mechanism_catalog:
            experiment.freeze_mechanism_catalog(
                args.mechanism_catalog, args.mechanism_catalog_sha256)
        if args.mechanism_work_order:
            experiment.freeze_mechanism_work_order(
                args.mechanism_work_order, args.mechanism_work_order_sha256,
                candidate=candidate)
    provider = None
    semantic_provider = None
    context_provider = None
    paired_context_provider = None
    source_pair_provider = None
    import importlib
    from merlin.runtime.backends.base import get_backend
    from merlin.perf.host_region_qualifier import HostChangedRegionQualifier
    from merlin.perf.host_physical_transition_qualifier import HostPhysicalTransitionQualifier, ChangedRegionQualifierDispatch
    backend = get_backend(target.target)
    if not args.semantic_only and all(callable(getattr(backend, name, None)) for name in (
            "short_program_environment", "prepare_short_program_build", "prepare_short_program_execution")):
        from merlin.perf.source_program_pair_provider import SourceProgramPairProvider
        source_pair_provider = SourceProgramPairProvider(
            target=target.target, adapter=backend, output=stage_root/"source_pair_runtime")
    abi_module_name = backend.__name__ + "." + target.target + "_host_witness_abi"
    try:
        native_abi_adapter = importlib.import_module(abi_module_name)
    except ModuleNotFoundError as exc:
        if exc.name != abi_module_name:
            raise
    else:
        native_abi = native_abi_adapter.derive_native_witness_abi(target=target.target)
        lane_migration = None
        if source_pair_provider is not None:
            from merlin.perf.lane_migration_qualifier import LaneMigrationContractionQualifier
            lane_migration = LaneMigrationContractionQualifier(
                target=target.target, runtime_provider=source_pair_provider,
                abi_provenance=native_abi["abi_provenance"],
                output=stage_root / "lane_migration_witnesses")
        semantic_provider = ChangedRegionQualifierDispatch(
            physical=HostPhysicalTransitionQualifier(**native_abi, output=stage_root / "physical_transition_witnesses"),
            legacy=HostChangedRegionQualifier(**native_abi, output=stage_root / "semantic_witnesses"),
            lane_migration=lane_migration)
    primitive_adapter_name = backend.__name__ + "." + target.target + "_primitive_probe"
    try:
        adapter = importlib.import_module(primitive_adapter_name)
    except ModuleNotFoundError as exc:
        if exc.name != primitive_adapter_name:
            raise
    else:
        import inspect
        from merlin.perf.controlled_context_provider import ControlledSourcePrefixProvider
        prepare = getattr(adapter, "prepare_primitive_probe", None)
        if prepare is not None and "include_operand_movement" in inspect.signature(prepare).parameters:
            context_provider = ControlledSourcePrefixProvider(
                target=target.target, adapter=adapter, output=stage_root / "controlled_contexts")
            from merlin.perf.paired_context_provider import PairedControlledContextProvider
            paired_context_provider = PairedControlledContextProvider(
                target=target.target, adapter=adapter, output=stage_root / "paired_contexts")
    if args.probe_interface:
        import importlib
        from merlin.runtime.backends.base import get_backend
        from merlin.perf.isolated_probe_provider import IsolatedPrimitiveProbeProvider
        backend = get_backend(target.target)
        adapter = importlib.import_module(backend.__name__ + "." + target.target + "_primitive_probe")
        provider = IsolatedPrimitiveProbeProvider(
            target=target.target, short_interface=args.probe_interface, adapter=adapter,
            runtime_receipt=args.probe_runtime_receipt, output=stage_root / "isolated_probes",
            profile_counters=args.probe_profile == "occupancy")
    PAS._write_json(stage_root / "launch.json", {
        "schema": "global_agent_launch_v1", "source_config": str(args.campaign_config.resolve()),
        "historical_reference": experiment.historical_reference,
        "source_config_sha256": PAS._sha256_file(args.campaign_config),
        "source_snapshot": str(PAS.repo_root()),
        "source_snapshot_files_sha256": PAS._document_sha256(snapshot_receipt["files"]),
        "phase1": experiment.phase1_binding, "candidate_source": str(args.candidate.resolve()),
        "baseline_sha256": experiment.baseline_sha256,
        "optimization_baseline_sha256": experiment.optimization_baseline_sha256,
        "optimization_baseline": experiment.optimization_baseline_binding,
        "candidate_source_sha256": hash_tree(args.candidate)["sha256"],
        "host_resource_policy": resource_policy.record(),
        "objective": sentinel.capsule, "model": resolved_model,
        "portfolio": experiment.portfolio_identity,
        "portfolio_sha256": experiment.portfolio_identity_sha256,
        "baseline_emission_cache": experiment.baseline_emission_cache_binding,
        "baseline_emission_cache_seeds": experiment.baseline_emission_cache_seeds,
        "requested_objective_capsule": args.objective_capsule,
        "requested_portfolio_capsules": args.portfolio_capsule,
        "external_objective": primary_external.record() if primary_external is not None else None,
        "external_objectives": [external.record() for external in external_objectives],
        "external_objective_roles": (["primary"] if primary_external is not None else [])
                                    + ["training"] * len(portfolio_externals),
        "external_objective_spec_sha256": args.external_objective_sha256,
        "portfolio_external_objective_spec_sha256": args.portfolio_external_objective_sha256,
        "mode": "checkpoint_validation" if args.validation_only else "agent_authoring",
        "comparison_candidate": str(args.comparison_candidate.resolve()) if args.comparison_candidate else None,
        "comparison_candidate_sha256": hash_tree(args.comparison_candidate)["sha256"] if args.comparison_candidate else None,
        "simulators_enabled": not args.semantic_only and (provider is not None or context_provider is not None),
        "full_model_simulation_allowed": False,
        "probe_interface_sha256": provider.short_interface_sha256 if provider else None,
        "probe_runtime_receipt_sha256": provider.runtime_receipt_sha256 if provider else None,
        "probe_adapter_sha256": PAS._sha256_file(Path(provider.adapter.__file__)) if provider else None,
        "probe_profile_scope": "isolated_primitive_" + args.probe_profile,
        "changed_region_semantic_provider": type(semantic_provider).__name__ if semantic_provider else None,
        "changed_region_native_abi": semantic_provider.abi_provenance if semantic_provider else None,
        "controlled_context_provider": type(context_provider).__name__ if context_provider else None,
        "paired_context_provider": type(paired_context_provider).__name__ if paired_context_provider else None,
        "complete_source_pair_provider": type(source_pair_provider).__name__ if source_pair_provider else None,
        "validation_context_scope": ("none_semantic_only" if args.semantic_only else
                                     "controlled_fixed_work_slice" if args.compare_controlled_context else "controlled_source_prefix"),
        "iteration_seconds": args.iteration_seconds, "round_seconds": args.round_seconds,
        "maximum_rounds": args.max_rounds, "total_authoring_seconds": total_authoring,
        "on_round_failure": args.on_round_failure,
        "compiler_mechanism_catalog": copy.deepcopy(experiment.mechanism_catalog_binding),
        "compiler_mechanism_work_order": copy.deepcopy(
            experiment.mechanism_work_order_binding),
        "static_analysis_seed": ({"path": str(args.static_analysis_seed_checkpoint.resolve()),
            "sha256": args.static_analysis_seed_sha256,
            "policy": "exact_content_hit_or_cold_analysis_miss"}
            if args.static_analysis_seed_checkpoint else None),
        "resume_checkpoint": ({"path": str(args.resume_checkpoint.resolve()),
            "sha256": PAS._sha256_file(args.resume_checkpoint),
            "candidate_sha256": resumed["candidate_sha256"],
            "previous_host_policy": resumed["host_verification_policy"],
            "current_host_policy": experiment.host_policy,
            "policy_transition": ("explicit_new_segment_with_exact_static_seed_or_fresh_analysis"
                                  if args.static_analysis_seed_checkpoint else
                                  "explicit_new_segment_with_fresh_initial_model_analysis"),
            "old_verdict_modified": False} if resumed else None),
    })
    print(f"GLOBAL {'VALIDATION' if args.validation_only else 'AUTHORING'}: {stage_root} "
          f"models={len(experiment.portfolio_sentinels)} primary={sentinel.capsule} "
          f"simulation={'bounded_probes_only' if not args.semantic_only and (provider or context_provider) else 'disabled'}", flush=True)
    if args.validation_only:
        try:
            if semantic_provider is None or (not args.semantic_only and context_provider is None):
                raise ValueError("deterministic validation lacks its selected evidence providers")
            configure_global_analysis(experiment, target_experiment=target, agent_inputs=inputs,
                frozen_functional=frozen_functional, frozen_corpus_manifest=corpus.manifest_path,
                stage_root=stage_root)
            before = experiment.analyze(args.comparison_candidate.resolve(),
                                        hypothesis="Preserved pre-edit whole-model compiler")
            after = experiment.analyze(candidate, hypothesis="Validate preserved generalized compiler edits")
            semantics = experiment.qualify_changed_region(candidate, provider=semantic_provider,
                                                           timeout_s=args.iteration_seconds)
            if args.semantic_only:
                context = {}
            elif args.compare_controlled_context:
                if paired_context_provider is None:
                    raise ValueError("paired controlled-context provider is unavailable")
                context = experiment.compare_controlled_context(
                    candidate, provider=paired_context_provider, timeout_s=60)
            else:
                context = experiment.profile_controlled_context(candidate, provider=context_provider, timeout_s=60)
            sealed = experiment.seal(candidate)
            validation = {"schema": "global_checkpoint_validation_v1", "mode": "no_new_authoring",
                "phase1_action": "reuse_exact_92_of_96_with_waivers", "comparison_sha256": before["candidate_sha256"],
                "candidate_sha256": after["candidate_sha256"], "candidate_receipt": str(sealed),
                "semantic_status": semantics["evidence"].get("status"),
                "controlled_prefix_cycles": None if args.semantic_only or args.compare_controlled_context else context["execution"]["total_compute_cycles"],
                "paired_fixed_work_cycles": context.get("cycles"),
                "simulation_executed": not args.semantic_only,
                "full_model_cycles": None, "global_speedup_proven": False,
                "promotion_status": "unqualified_candidate_for_review"}
            PAS._write_json(stage_root / "validation.json", validation)
            print(json.dumps(validation, indent=2), flush=True)
            return 0
        except Exception as exc:
            PAS._write_json(stage_root / "validation_failure.json", {
                "schema": "global_checkpoint_validation_failure_v1", "exception": type(exc).__name__,
                "reason": str(exc), "global_speedup_proven": False, "phase1_rerun": False})
            raise
    def author_round(current, *, round_index, round_timeout_s):
        return run_global_agent_round(
        experiment, current, target_experiment=target, workspace=current.parent,
        stage_root=stage_root, agent_inputs=inputs, frozen_functional=frozen_functional,
        frozen_corpus_manifest=corpus.manifest_path,
        model=str(config["model"]), resolved_model=resolved_model, effort=str(config["effort"]),
        codex_binary=codex, round_index=round_index, round_timeout_s=round_timeout_s,
        max_tool_calls=args.max_tool_calls, global_probe_provider=provider,
        global_semantic_provider=semantic_provider, global_context_provider=context_provider,
        global_paired_context_provider=paired_context_provider,
        global_source_pair_provider=source_pair_provider)
    def configure_and_seed():
        configure_global_analysis(experiment, target_experiment=target,
            agent_inputs=inputs, frozen_functional=frozen_functional,
            frozen_corpus_manifest=corpus.manifest_path, stage_root=stage_root)
        if args.static_analysis_seed_checkpoint:
            experiment.import_static_analysis_checkpoint(
                candidate, checkpoint=args.static_analysis_seed_checkpoint.resolve(),
                checkpoint_sha256=args.static_analysis_seed_sha256)

    sequence = run_authoring_with_terminal_receipt(stage_root,
        configure=configure_and_seed,
        sequence=lambda: run_global_agent_sequence(experiment, candidate, run_round=author_round,
            stage_root=stage_root, max_rounds=args.max_rounds,
            total_authoring_seconds=total_authoring, round_seconds=args.round_seconds,
            on_round_failure=args.on_round_failure))
    try:
        # A bounded sequence may end with an exact blocked authoring checkpoint.  Preserve that
        # evidence for a follow-on segment, but never relabel it as the promotable global seal.
        # Ready, failure-free sequences still receive the conventional final review artifact.
        sealed = (experiment.seal(Path(sequence["candidate"]))
                  if not sequence["failures"] and sequence.get("promotion_ready") is True
                  else Path(sequence["last_good_checkpoint"]["path"]))
    except Exception as exc:
        PAS._write_json(stage_root / "terminal_failure.json", {
            "schema": "global_launch_terminal_failure_v1", "stage": "seal",
            "authoring_status": sequence["status"], "exception": type(exc).__name__,
            "reason": str(exc), "candidate_sha256": hash_tree(candidate)["sha256"],
            "promotion_status": "unqualified", "global_speedup_proven": False})
        raise
    print(json.dumps({"authoring": sequence["status"], "candidate": str(sealed),
                      "promotion": ("unqualified" if sequence.get("promotion_ready") is True
                                    else "blocked_authoring_checkpoint"),
                      "full_model_timing": "UNMEASURED"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
