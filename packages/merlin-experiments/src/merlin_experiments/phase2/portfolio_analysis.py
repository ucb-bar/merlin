"""Serialized whole-portfolio analysis, same-run reuse and scientific reevaluation.

Owns live analysis policy and publication through concrete revision authorities.
Cross-run checkpoint admission remains a separate consumer of these operations.
"""

from __future__ import annotations

import concurrent.futures
import copy
import math
import shutil
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.benchharness import hash_tree
from merlin.common.digest import sha256_bytes
from merlin.perf.execution_policy import FULL_GRAPH_STATIC_ANALYSIS_MAX_SECONDS
from merlin.perf.functional_gate import STATUS_NOT_RUN as FUNCTIONAL_GATE_NOT_RUN
from merlin.perf.functional_gate import (
    FunctionalGateConfig,
    FunctionalGateResult,
    gate_result_for_selection,
    run_functional_gate,
)
from merlin_experiments.phase2 import contracts as P2_CONTRACTS
from merlin_experiments.phase2 import emission_analysis as EA
from merlin_experiments.phase2 import emission_diagnostics as ED
from merlin_experiments.phase2 import portfolio_resources as PR
from merlin_experiments.phase2 import stage_inputs as INPUTS
from merlin_experiments.phase2 import static_cache as SC
from merlin_experiments.phase2 import static_identity as SI
from merlin_experiments.phase2.revision_session import RevisionSession
from merlin_experiments.phase2.stage_inputs import sentinel_identity


def _host_memory_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise ValueError("host MemAvailable is unavailable")


class PortfolioAnalysis:
    """Own bounded fresh and same-run reused analysis of an admitted portfolio."""

    def __init__(
        self,
        session: RevisionSession,
        *,
        timeout_s: int = 300,
        portfolio_analysis_workers: int = 1,
        minimum_memory_available_bytes: int = 0,
        analyzer: Callable[..., Mapping[str, Any]] = EA.analyze_whole_model_emission,
        plan_verifier: Callable[..., Mapping[str, Any]] | None = None,
        functional_gate: FunctionalGateConfig | None = None,
        functional_gate_runner: Callable[..., FunctionalGateResult] | None = None,
        baseline_emission_cache: Path | None = None,
        baseline_emission_seed_runs: Sequence[Path] = (),
    ) -> None:
        self.validate_timeout(timeout_s)
        if portfolio_analysis_workers < 1 or minimum_memory_available_bytes < 0:
            raise ValueError("portfolio concurrency policy is invalid")
        self.session = session
        self.timeout_s = timeout_s
        self.portfolio_analysis_workers = portfolio_analysis_workers
        self.minimum_memory_available_bytes = minimum_memory_available_bytes
        self.analyzer, self.plan_verifier = analyzer, plan_verifier
        self.functional_gate = functional_gate
        self.functional_gate_runner = functional_gate_runner or run_functional_gate
        self.completion_contract: Any | None = None
        self._analysis_lock = threading.Lock()
        self.baseline_emission_cache_binding = None
        self.baseline_emission_cache_seeds: list[dict[str, Any]] = []
        if baseline_emission_cache is not None:
            cache_root = Path(baseline_emission_cache).resolve()
            if cache_root.is_relative_to(self.session.inputs.baseline.resolve()) or cache_root.is_relative_to(
                self.session.inputs.optimization_baseline.resolve()
            ):
                raise ValueError("baseline emission cache cannot be inside a compiler tree")
            self.baseline_emission_cache_binding = {
                "schema": "baseline_emission_cache_binding_v1",
                "root": str(cache_root),
                "compiler_dependencies_sha256": SI.compiler_dependency_content_sha256(
                    self.session.inputs.optimization_baseline_binding["compiler_dependencies"]
                ),
            }
            schema = ED.whole_program_schema_record(
                self.session.inputs.contract_root / "schemas/command_buffer.schema.json"
            )
            for seed_run in baseline_emission_seed_runs:
                entries = EA.seed_baseline_emission_cache_from_run(
                    cache_binding=self.baseline_emission_cache_binding,
                    seed_run=seed_run,
                    baseline=self.session.inputs.optimization_baseline,
                    sentinels=self.session.inputs.portfolio_sentinels,
                    target=self.session.inputs.target,
                    compiler_api_schema=schema,
                )
                self.baseline_emission_cache_seeds.append(
                    {
                        "run": str(Path(seed_run).resolve()),
                        "entries": [
                            {
                                "key": row["key"],
                                "capsule_sha256": row["identity"]["capsule_sha256"],
                                "emission_wall_seconds": row["emission_wall_seconds"],
                            }
                            for row in entries
                        ],
                    }
                )
        elif baseline_emission_seed_runs:
            raise ValueError("baseline emission seed runs require a cache root")

    @staticmethod
    def validate_timeout(timeout_s: float) -> None:
        if not 0 < timeout_s <= FULL_GRAPH_STATIC_ANALYSIS_MAX_SECONDS:
            raise ValueError(
                "full-graph static analysis must fit the "
                f"{FULL_GRAPH_STATIC_ANALYSIS_MAX_SECONDS:g}-second host wall budget"
            )

    def _baseline_emission_observations(self) -> dict[str, dict[str, Any]]:
        """Read bounded exact cache receipts without loading large emitted artifacts."""
        if self.baseline_emission_cache_binding is None:
            return {}
        from merlin.targetgen import oot_runner as OR

        package = OR.load_package(self.session.inputs.optimization_baseline)
        entrypoints = OR.analysis_emission_entrypoints(package)
        schema = ED.whole_program_schema_record(
            self.session.inputs.contract_root / "schemas/command_buffer.schema.json"
        )
        observations: dict[str, dict[str, Any]] = {}
        for sentinel in self.session.inputs.portfolio_sentinels:
            source = Path(sentinel.frozen_source_path)
            descriptor = P2_CONTRACTS.mapping_file(source / "capsule.yaml", yaml_file=True)
            interface = source / str(descriptor.get("interface_mlir") or "capsule.interface.mlir")
            identity = EA.baseline_emission_cache_identity(
                baseline_sha256=self.session.inputs.optimization_baseline_sha256,
                capsule_sha256=sentinel.capsule_sha256,
                source_sha256=sha256_bytes(interface.read_bytes()),
                target=self.session.inputs.target,
                compiler_dependencies_sha256=self.baseline_emission_cache_binding["compiler_dependencies_sha256"],
                compiler_api_schema=schema,
                entrypoints=entrypoints,
            )
            cached = EA.baseline_emission_cache_observation(self.baseline_emission_cache_binding, identity)
            if cached is not None:
                observations[sentinel.capsule_sha256] = cached
        return observations

    def _baseline_emission_costs(self) -> dict[str, float]:
        return {digest: row["emission_wall_seconds"] for digest, row in self._baseline_emission_observations().items()}

    def _portfolio_analysis_cost_estimates(self) -> list[float]:
        """Estimate changed-candidate member wall from exact prior receipts, without model constants."""
        observations = self._baseline_emission_observations()
        interface_sizes = []
        for sentinel in self.session.inputs.portfolio_sentinels:
            source = Path(sentinel.frozen_source_path)
            descriptor = P2_CONTRACTS.mapping_file(source / "capsule.yaml", yaml_file=True)
            interface = source / str(descriptor.get("interface_mlir") or "capsule.interface.mlir")
            interface_sizes.append(max(1, interface.stat().st_size))
        measured = []
        for sentinel in self.session.inputs.portfolio_sentinels:
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
            value / size for value, size in zip(measured, interface_sizes, strict=True) if value is not None
        )
        median_rate = (
            None
            if not known_rates
            else known_rates[len(known_rates) // 2]
            if len(known_rates) % 2
            else 0.5 * (known_rates[len(known_rates) // 2 - 1] + known_rates[len(known_rates) // 2])
        )
        return [
            float(value if value is not None else size * median_rate if median_rate is not None else size)
            for value, size in zip(measured, interface_sizes, strict=True)
        ]

    def mandatory_analysis_reserve_seconds(self, maximum_seconds: float) -> dict[str, Any]:
        """Reserve a measured, generic concurrent-validation window before round finalization."""
        observations = self._baseline_emission_observations()
        if observations:
            concurrency = PR.portfolio_analysis_concurrency(
                requested_workers=self.portfolio_analysis_workers,
                members=len(self.session.inputs.portfolio_sentinels),
                memory_available_bytes=_host_memory_available_bytes(),
                minimum_memory_available_bytes=self.minimum_memory_available_bytes,
            )
            schedule = PR.portfolio_concurrent_schedule(
                self._portfolio_analysis_cost_estimates(), concurrency["admitted_workers"]
            )
            estimate = schedule["projected_wall_seconds"]
            basis = "resource_admitted_schedule_of_exact_or_conservative_member_receipts"
        else:
            estimate = min(float(self.timeout_s), max(60.0, maximum_seconds * 0.5))
            basis = "cold_cache_half_tool_window"
            concurrency = None
            schedule = None
        reserve = min(float(maximum_seconds), math.ceil(estimate * 1.05 + 5.0))
        return {
            "schema": "mandatory_portfolio_analysis_reserve_v1",
            "seconds": reserve,
            "estimated_member_wall_seconds": estimate,
            "basis": basis,
            "observed_members": len(observations),
            "analysis_concurrency": concurrency,
            "analysis_schedule": schedule,
            "execution": "concurrent_shared_deadline",
            "scope": "host tool-window admission; not a performance estimate",
        }

    def bind_mechanism_work_order_analysis(self, record: Mapping[str, Any]) -> dict[str, Any] | None:
        if (
            not self.session.mechanism_program.has_work_order
            and self.session.mechanism_program.work_order_binding is None
        ):
            return self.session.mechanism_program.bind_analysis(record)
        self.session.check_inputs()
        already_bound = self.session.mechanism_program.analysis_binding is not None
        immutable = None
        if already_bound:
            iteration = record.get("iteration")
            stored = next((row for row in self.session.journal.iterations if row.get("iteration") == iteration), None)
            reuse_binding = record.get("analysis_reuse_binding")
            if isinstance(stored, Mapping) and isinstance(reuse_binding, Mapping):
                immutable = self.immutable_reusable_iteration(stored, binding=reuse_binding)
        binding = self.session.mechanism_program.bind_analysis(record, immutable_iteration=immutable)
        if not already_bound:
            self.session.check_inputs()
        return binding

    def record_bound_mechanism_work_order_seed(self, candidate: Path, record: Mapping[str, Any]) -> dict[str, Any]:
        """Append an immutable no-compile seed record carrying the new work-order binding."""
        binding = self.bind_mechanism_work_order_analysis(record)
        if binding is None or record.get("mechanism_work_order_analysis") == binding:
            return copy.deepcopy(dict(record))
        started = time.monotonic()
        return self._reuse_prior_analysis(
            candidate,
            hypothesis="Bind the host work order to the initial complete-model static evidence",
            source=record,
            binding=record["analysis_reuse_binding"],
            mechanism_attribution=self.session.inspect_mechanism_round(candidate, require_semantic_edit=False),
            started=started,
            budget_seconds=self.timeout_s,
        )

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

    def _analyze_portfolio_member(
        self,
        submitted: Path,
        *,
        candidate_sha256: str,
        sentinel: INPUTS.StageE2ESentinel,
        timeout_s: float,
        scope: Mapping[str, Any],
        primary: bool = False,
        analyzer_override: Callable[..., Mapping[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], Mapping[str, Any] | None]:
        """Compile one full graph with member-local artifacts and analyzer state."""
        from merlin.perf.analysis_worker import IsolatedAnalysisWorker

        retained: dict[str, Any] = {}
        analyzer = analyzer_override or self.analyzer
        kwargs: dict[str, Any] = {
            "timeout_s": timeout_s,
            "target": self.session.inputs.target,
            "peak_macs_per_cycle": None,
            "achievable_macs_per_cycle": None,
            "host_verifier_policy_sha256": self.session.inputs.host_policy["sha256"],
        }
        if self.plan_verifier is not None:
            kwargs["global_plan_verifier"] = self.plan_verifier
        if analyzer is EA.analyze_whole_model_emission:
            kwargs["contract_root"] = self.session.inputs.contract_root
        if analyzer is EA.analyze_whole_model_emission or isinstance(analyzer, IsolatedAnalysisWorker):
            kwargs["compiler_api_schema"] = ED.whole_program_schema_record(
                self.session.inputs.contract_root / "schemas/command_buffer.schema.json"
            )
            kwargs["artifact_sink"] = retained.update
            kwargs["baseline_artifacts"] = (
                self.session.journal.baseline_artifacts
                if primary
                else self.session.journal.portfolio_baseline_artifacts.get(sentinel.capsule_sha256)
            )
            kwargs["baseline_emission_cache"] = self.baseline_emission_cache_binding
        try:
            if timeout_s <= 0:
                raise TimeoutError("portfolio member has no remaining iteration budget")
            analysis = dict(analyzer(self.session.inputs.optimization_baseline, submitted, sentinel, **kwargs))
            analysis["compiler_edit_scope"] = copy.deepcopy(scope)
            if self.session.edit_authority.guidance_inventory is not None:
                from merlin.perf.agent_guidance import guidance_for_emission_analysis

                analysis["optimization_brief"] = guidance_for_emission_analysis(
                    analysis["diagnostics"], self.session.edit_authority.guidance_inventory
                )
                if primary:
                    analysis["optimization_brief"]["compiler_edit_contract_template"] = copy.deepcopy(
                        self.session.edit_authority.contract
                    )
                    analysis["optimization_brief"]["host_guidance_binding"] = {
                        "inventory_sha256": self.session.edit_authority.binding["guidance_inventory_sha256"],
                        "initial_candidate_sha256": self.session.edit_authority.binding["initial_candidate_sha256"],
                        "contract_document_sha256": self.session.edit_authority.binding["contract_document_sha256"],
                        "permission_scope": "unchanged host-frozen edit contract",
                        "mapping_scope": (
                            "host-declared semantics; AST/component ownership checked on "
                            "frozen seed; not proof of emitted effect"
                        ),
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
        analysis["optimization_baseline"] = copy.deepcopy(self.session.inputs.optimization_baseline_binding)
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
                analysis_source=self.analyzer.analysis_source,
                contract_root=self.analyzer.contract_root,
                sandbox_factory=self.analyzer.sandbox_factory,
                output=self.analyzer.output,
                python_command=self.analyzer.python_command,
            )
        return self.analyzer

    def analysis_reuse_binding(
        self, *, candidate_sha256: str, compiler_dependencies: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Exact static-analysis inputs whose equality permits cross-iteration reuse."""
        body = {
            "candidate_sha256": candidate_sha256,
            "compiler_dependencies": copy.deepcopy(compiler_dependencies),
            "host_verification_policy_sha256": self.session.inputs.host_policy["sha256"],
            "target_sha256": self.session.inputs.target_sha256,
            "baseline_sha256": self.session.inputs.baseline_sha256,
            "optimization_baseline_binding_sha256": self.session.inputs.optimization_baseline_binding_sha256,
            "portfolio_sha256": self.session.inputs.portfolio_identity_sha256,
            "historical_reference_binding_sha256": self.session.inputs._historical_reference_binding_sha256,
            "phase1_qualification_sha256": P2_CONTRACTS.document_sha256(self.session.inputs.phase1_binding),
            "compiler_edit_authority_sha256": P2_CONTRACTS.document_sha256(self.session.edit_authority.binding),
        }
        if self.session.fast_evaluation.binding is not None:
            body["fast_evaluation_binding_sha256"] = self.session.fast_evaluation.binding_sha256
        return {
            "schema": "global_static_analysis_reuse_binding_v1",
            **body,
            "sha256": P2_CONTRACTS.document_sha256(body),
        }

    def cross_run_static_analysis_binding(
        self, *, candidate_sha256: str, compiler_dependencies: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Content-only identity for an explicitly pinned static-analysis checkpoint."""
        schema = ED.whole_program_schema_record(
            self.session.inputs.contract_root / "schemas/command_buffer.schema.json"
        )
        body = {
            "candidate_sha256": candidate_sha256,
            "candidate_compiler_dependencies_content_sha256": SI.compiler_dependency_content_sha256(
                compiler_dependencies
            ),
            "baseline_sha256": self.session.inputs.baseline_sha256,
            "baseline_compiler_dependencies_content_sha256": SI.compiler_dependency_content_sha256(
                self.session.inputs.baseline_dependencies
            ),
            "optimization_baseline": SI.portable_optimization_baseline_binding(
                self.session.inputs.optimization_baseline_binding
            ),
            "host_verification_policy_content_sha256": self.session.inputs.host_policy["sha256"],
            "target": self.session.inputs.target,
            "target_descriptor_sha256": self.session.inputs.target_sha256,
            "ordered_portfolio": copy.deepcopy(self.session.inputs.portfolio_identity),
            "portfolio_sha256": self.session.inputs.portfolio_identity_sha256,
            "phase1_qualification": SI.portable_phase1_binding(self.session.inputs.phase1_binding),
            "historical_reference": SI.portable_historical_reference(self.session.inputs.historical_reference),
            "compiler_edit_authority": SI.portable_edit_authority(self.session.edit_authority.binding),
            "compiler_api_schema": {"name": Path(schema["path"]).name, "sha256": schema["sha256"]},
            "analysis_options": {
                "schema": "global_cross_run_analysis_options_v1",
                "maximum_full_graph_static_analysis_seconds": self.timeout_s,
                "portfolio_analysis_workers": self.portfolio_analysis_workers,
                "minimum_memory_available_bytes": self.minimum_memory_available_bytes,
                "peak_macs_per_cycle": None,
                "achievable_macs_per_cycle": None,
                "full_model_simulation_allowed": False,
                "analyzer": "host_owned_whole_model_emission_with_current_readiness_v1",
            },
            "machine_build_policy": copy.deepcopy(self.session.inputs.machine_build_policy),
        }
        if self.session.fast_evaluation.binding is not None:
            body["fast_evaluation_binding"] = copy.deepcopy(self.session.fast_evaluation.binding)
        return {
            "schema": "global_cross_run_static_analysis_binding_v1",
            **body,
            "sha256": P2_CONTRACTS.document_sha256(body),
        }

    def persist_static_analysis_bundle(
        self,
        record: Mapping[str, Any],
        artifacts: Mapping[str, Any],
        *,
        portfolio_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """Persist only reusable analytical documents and primary emitted artifact bytes."""
        return SC.persist_static_analysis_bundle(
            record,
            artifacts,
            output=self.session.journal.output,
            capsule_sha256s=[member.capsule_sha256 for member in self.session.inputs.portfolio_sentinels],
            portfolio_sha256=self.session.inputs.portfolio_identity_sha256,
            portfolio_artifacts=portfolio_artifacts,
        )

    def immutable_reusable_iteration(
        self, row: Mapping[str, Any], *, binding: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        """Load and revalidate one prior ready iteration; malformed cache entries are misses."""
        return SI.immutable_reusable_iteration(
            row,
            binding=binding,
            output=self.session.journal.output,
            record_sha256=self.session.journal.record_sha256,
            compiler_shared_source_root=self.session.inputs.compiler_shared_source_root,
            capsule_sha256s=[member.capsule_sha256 for member in self.session.inputs.portfolio_sentinels],
            portfolio_sha256=self.session.inputs.portfolio_identity_sha256,
        )

    def _find_reusable_iteration(self, *, binding: Mapping[str, Any]) -> dict[str, Any] | None:
        return SI.find_reusable_iteration(
            self.session.journal.iterations,
            binding=binding,
            output=self.session.journal.output,
            record_sha256=self.session.journal.record_sha256,
            compiler_shared_source_root=self.session.inputs.compiler_shared_source_root,
            capsule_sha256s=[member.capsule_sha256 for member in self.session.inputs.portfolio_sentinels],
            portfolio_sha256=self.session.inputs.portfolio_identity_sha256,
        )

    @staticmethod
    def _reuse_allocation(source: Mapping[str, Any], *, source_iteration: int) -> dict[str, Any]:
        original = source.get("source_analysis_allocation", source)
        return {
            "schema": "portfolio_analysis_reuse_allocation_v1",
            "policy": "exact_immutable_ready_iteration_reuse_no_compilation",
            "allocated_seconds": 0.0,
            "source_iteration": source_iteration,
            "source_analysis_allocation": copy.deepcopy(original),
        }

    def _reuse_prior_analysis(
        self,
        candidate: Path,
        *,
        hypothesis: str,
        source: Mapping[str, Any],
        binding: Mapping[str, Any],
        mechanism_attribution: Mapping[str, Any] | None,
        started: float,
        budget_seconds: float,
    ) -> dict[str, Any]:
        """Append a fresh iteration around reusable static evidence from an older revision."""
        source_iteration = source["iteration"]
        result_iteration = len(self.session.journal.iterations)
        analysis = copy.deepcopy(source["analysis"])
        primary_readiness = EA.global_iteration_readiness(analysis)
        source_portfolio = source["portfolio"]
        source_members = source_portfolio["members"]
        previous_members = {
            member["identity"]["capsule_sha256"]: member
            for member in ((self.session.journal.iterations[-1].get("portfolio") or {}).get("members") or ())
        }
        portfolio_rows: list[dict[str, Any]] = []
        for sentinel, source_member in zip(
            self.session.inputs.portfolio_sentinels[1:], source_members[1:], strict=True
        ):
            member_analysis = copy.deepcopy(source_member["analysis"])
            member_readiness = EA.global_iteration_readiness(member_analysis)
            previous_member = previous_members.get(sentinel.capsule_sha256)
            portfolio_rows.append(
                {
                    "identity": sentinel_identity(sentinel, role="training"),
                    "status": ("completed" if member_readiness["status"] == "ready_for_probe_admission" else "failed"),
                    "analysis": member_analysis,
                    "readiness": member_readiness,
                    "static_comparison": self.compare_analyses(
                        previous_member.get("analysis") if previous_member else None,
                        member_analysis,
                        previous_iteration=(
                            self.session.journal.iterations[-1]["iteration"] if previous_member else None
                        ),
                    ),
                    "analysis_allocation": self._reuse_allocation(
                        source_member.get("analysis_allocation") or {}, source_iteration=source_iteration
                    ),
                    "elapsed_seconds": 0.0,
                    "timing_status": "UNMEASURED_FULL_MODEL",
                }
            )
        readiness = copy.deepcopy(primary_readiness)
        portfolio_blockers = [
            f"portfolio:{member['identity']['capsule']}:{blocker}"
            for member in portfolio_rows
            for blocker in member["readiness"]["blockers"]
        ]
        if portfolio_blockers:
            readiness["status"] = "blocked"
            readiness["blockers"] = [*readiness["blockers"], *portfolio_blockers]
        readiness["portfolio_sha256"] = self.session.inputs.portfolio_identity_sha256
        readiness["portfolio_members_ready"] = sum(
            row["readiness"]["status"] == "ready_for_probe_admission"
            for row in ({"readiness": primary_readiness}, *portfolio_rows)
        )
        readiness["portfolio_members_total"] = len(self.session.inputs.portfolio_sentinels)
        readiness["selection"] = "multi_model_pareto_without_invented_static_cycle_total"
        if readiness["status"] != "ready_for_probe_admission":
            raise ValueError("reusable static analysis no longer satisfies current readiness policy")

        current_artifacts = self.session.journal.iteration_artifacts.get(source_iteration, {})
        if current_artifacts and (
            current_artifacts.get("candidate_sha256") != source["candidate_sha256"]
            or current_artifacts.get("candidate_lowered_sha256")
            != analysis.get("emission", {}).get("candidate_lowered_sha256")
        ):
            # Retained artifacts are an in-memory convenience, not part of the immutable
            # static-analysis cache.  A stale copy removes probe eligibility; it cannot poison
            # the reused readiness result or a relative semantic comparison.
            current_artifacts = {}
        previous_artifacts = self.session.journal.artifacts
        relative_semantics: Mapping[str, Any] = {
            "status": "unavailable_target_completion_contract",
            "numerical_equivalence": False,
        }
        remaining = budget_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("exact analysis reuse verification exhausted the iteration budget")
        if self.completion_contract is not None and current_artifacts:
            from merlin.perf.completion_delta import qualify_relative_completion_delta

            relative_semantics = qualify_relative_completion_delta(
                previous_analysis=self.session.journal.iterations[-1]["analysis"],
                current_analysis=analysis,
                previous_artifacts=previous_artifacts,
                current_artifacts=current_artifacts,
                contract=self.completion_contract,
                timeout_seconds=min(30, remaining),
            )
        elapsed = time.monotonic() - started
        if elapsed > budget_seconds:
            raise TimeoutError("exact analysis reuse verification exhausted the iteration budget")
        source_compilation_iteration = (source.get("analysis_reuse") or {}).get(
            "source_compilation_iteration", source_iteration
        )
        reuse = {
            "schema": "global_exact_static_analysis_reuse_v1",
            "source_iteration": source_iteration,
            "source_compilation_iteration": source_compilation_iteration,
            "result_iteration": result_iteration,
            "binding": copy.deepcopy(binding),
            "source_iteration_record": str(
                (self.session.journal.output / f"iteration_{source_iteration:04d}.json").resolve()
            ),
            "source_analysis_sha256": P2_CONTRACTS.document_sha256(analysis),
            "full_graph_compiler_invoked": False,
            "full_model_simulation_executed": False,
            "probe_or_timing_receipts_reused": False,
            "source_elapsed_seconds": source["elapsed_seconds"],
            "reuse_verification_elapsed_seconds": elapsed,
        }
        record = {
            "schema": "global_perf_iteration_v1",
            "iteration": result_iteration,
            "candidate_path": str(candidate.resolve()),
            "candidate_sha256": source["candidate_sha256"],
            "submitted_snapshot": source["submitted_snapshot"],
            "compiler_dependencies": copy.deepcopy(source["compiler_dependencies"]),
            "analysis_reuse_binding": copy.deepcopy(binding),
            "cross_run_static_analysis_binding": self.cross_run_static_analysis_binding(
                candidate_sha256=source["candidate_sha256"], compiler_dependencies=source["compiler_dependencies"]
            ),
            "analysis_reuse": reuse,
            "exact_analysis_reused": True,
            "compiler_mechanism_catalog": copy.deepcopy(self.session.mechanism_program.catalog_binding),
            "compiler_mechanism_work_order": copy.deepcopy(self.session.mechanism_program.work_order_binding),
            "mechanism_work_order_analysis": copy.deepcopy(self.session.mechanism_program.analysis_binding),
            "round_mechanism_attribution": copy.deepcopy(mechanism_attribution),
            "baseline_sha256": self.session.inputs.baseline_sha256,
            "optimization_baseline_sha256": self.session.inputs.optimization_baseline_sha256,
            "optimization_baseline": copy.deepcopy(self.session.inputs.optimization_baseline_binding),
            "hypothesis": hypothesis,
            "analysis": analysis,
            "readiness": readiness,
            "historical_reference": copy.deepcopy(self.session.inputs.historical_reference),
            "elapsed_seconds": elapsed,
            "timing_status": "UNMEASURED_FULL_MODEL",
            "allocated_seconds": budget_seconds,
            "probe_receipts": [],
            "global_performance_claim": "unproven",
            "relative_semantic_evidence": relative_semantics,
            "fast_evaluation": copy.deepcopy(source.get("fast_evaluation")),
        }
        if source.get("static_analysis_bundle") is not None:
            record["static_analysis_bundle"] = copy.deepcopy(source["static_analysis_bundle"])
        record["static_comparison"] = self.compare(record)
        self.apply_functional_gate(record, current_artifacts, reused_from=source)
        readiness = record["readiness"]
        record["portfolio"] = {
            "schema": "full_model_portfolio_iteration_v1",
            "portfolio_sha256": self.session.inputs.portfolio_identity_sha256,
            "candidate_sha256": source["candidate_sha256"],
            "members": [
                {
                    "identity": sentinel_identity(self.session.inputs.sentinel, role="primary"),
                    "status": "completed",
                    "analysis_ref": "/analysis",
                    "readiness": primary_readiness,
                    "static_comparison_ref": "/static_comparison",
                    "analysis_allocation": self._reuse_allocation(
                        source_members[0].get("analysis_allocation") or {}, source_iteration=source_iteration
                    ),
                    "elapsed_seconds": 0.0,
                    "timing_status": "UNMEASURED_FULL_MODEL",
                },
                *portfolio_rows,
            ],
            "members_ready": readiness["portfolio_members_ready"],
            "members_total": readiness["portfolio_members_total"],
            "selection": readiness["selection"],
            "analysis_allocation_policy": "exact_immutable_ready_iteration_reuse_no_compilation",
            "analysis_concurrency": {
                "schema": "portfolio_analysis_reuse_concurrency_v1",
                "requested_workers": self.portfolio_analysis_workers,
                "admitted_workers": 0,
                "members": len(self.session.inputs.portfolio_sentinels),
                "policy": "no_workers_admitted_for_exact_immutable_analysis_reuse",
            },
            "full_model_simulation_allowed": False,
        }
        self.session.journal.publish_reuse(record, current_artifacts, source_iteration=source_iteration)
        return copy.deepcopy(record)

    def _analyze_locked(self, candidate: Path, *, hypothesis: str, timeout_s: float | None = None) -> dict[str, Any]:
        """Compile a novel edit or chronologically reuse one exact prior ready analysis."""
        started = time.monotonic()
        self.session.check_inputs()
        if not hypothesis.strip():
            raise ValueError("each iteration must state the global transformation hypothesis")
        budget_seconds = self.timeout_s if timeout_s is None else min(self.timeout_s, timeout_s)
        if budget_seconds <= 0:
            raise ValueError("full-model analysis has no remaining iteration budget")
        mechanism_attribution = self.session.inspect_mechanism_round(candidate, require_semantic_edit=False)
        if mechanism_attribution is not None and mechanism_attribution.get("status") not in ("allowed", "initial_seed"):
            self._write(f"mechanism_analysis_refusal_{time.time_ns()}.json", mechanism_attribution)
            raise ValueError(
                "candidate violates the host-frozen one-mechanism policy: "
                + str(mechanism_attribution.get("violations"))
            )
        self.session.validate_candidate_scope(candidate)
        if self.session.inputs.historical_reference_source is not None and (
            self.session.inputs.historical_reference_source.resolve().is_relative_to(candidate.resolve())
            or Path(self.session.inputs.historical_reference["path"]).is_relative_to(candidate.resolve())
        ):
            raise ValueError("historical reference is inside candidate-writable source")
        dependencies_before = self.session.inputs.compiler_dependencies(candidate)
        before = hash_tree(candidate)["sha256"]
        reuse_binding = self.analysis_reuse_binding(candidate_sha256=before, compiler_dependencies=dependencies_before)
        reusable = self._find_reusable_iteration(binding=reuse_binding)
        if reusable is not None:
            if reusable["iteration"] != self.session.journal.iterations[-1]["iteration"]:
                return self._reuse_prior_analysis(
                    candidate,
                    hypothesis=hypothesis,
                    source=reusable,
                    binding=reuse_binding,
                    mechanism_attribution=mechanism_attribution,
                    started=started,
                    budget_seconds=budget_seconds,
                )
            elapsed = time.monotonic() - started
            if elapsed > budget_seconds:
                raise TimeoutError("exact analysis reuse verification exhausted the iteration budget")
            receipt = {
                "schema": "global_exact_static_analysis_reuse_v1",
                "source_iteration": reusable["iteration"],
                "source_compilation_iteration": (reusable.get("analysis_reuse") or {}).get(
                    "source_compilation_iteration", reusable["iteration"]
                ),
                "result_iteration": reusable["iteration"],
                "binding": copy.deepcopy(reuse_binding),
                "hypothesis": hypothesis,
                "full_graph_compiler_invoked": False,
                "full_model_simulation_executed": False,
                "probe_or_timing_receipts_reused": False,
                "duplicate_current_revision": True,
                "reuse_verification_elapsed_seconds": elapsed,
            }
            receipt_path = self._write(f"analysis_reuse_{time.time_ns()}.json", receipt)
            return {
                **copy.deepcopy(self.session.journal.iterations[-1]),
                "exact_analysis_reused": True,
                "analysis_reuse_receipt": {"path": str(receipt_path), "sha256": P2_CONTRACTS.sha256_file(receipt_path)},
            }
        # Analyze immutable submitted bytes. The agent may keep authoring while this request runs;
        # the result names this snapshot, and _current still refuses a newer unanalysed revision.
        submitted = self.session.journal.output / f"submission_{len(self.session.journal.iterations):04d}"
        shutil.copytree(candidate, submitted, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        if hash_tree(submitted)["sha256"] != before:
            raise ValueError("candidate changed while its analysis snapshot was being captured")
        for path in submitted.rglob("*"):
            if not path.is_symlink():
                path.chmod(path.stat().st_mode & ~0o222)
        submitted.chmod(0o555)
        scope = self.session.validate_candidate_scope(submitted)
        remaining = budget_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("global input verification and snapshot exhausted the iteration budget")
        after = hash_tree(submitted)["sha256"]
        costs = self._baseline_emission_costs()
        concurrency = PR.portfolio_analysis_concurrency(
            requested_workers=self.portfolio_analysis_workers,
            members=len(self.session.inputs.portfolio_sentinels),
            memory_available_bytes=_host_memory_available_bytes(),
            minimum_memory_available_bytes=self.minimum_memory_available_bytes,
        )
        member_results: list[
            tuple[dict[str, Any], dict[str, Any], Mapping[str, Any] | None, dict[str, Any], float] | None
        ] = [None for _ in self.session.inputs.portfolio_sentinels]
        # Worker count one and worker count N use the same absolute portfolio deadline.  The old
        # one-worker fallback assigned a local weighted slice to each declared-order member; a
        # long-running member could be killed even though the portfolio still had ample time.
        # LPT scheduling still bounds total wall time and gives long observed members
        # first access to the deadline.  Each queued member receives the exact remaining outer
        # budget when it actually starts; results are restored to declared portfolio order below.
        planning = [
            PR.portfolio_member_analysis_allocation(
                remaining, self.session.inputs.portfolio_sentinels[index:], emission_seconds_by_capsule_sha256=costs
            )
            for index in range(len(self.session.inputs.portfolio_sentinels))
        ]
        deadline = time.monotonic() + remaining
        cost_estimates = self._portfolio_analysis_cost_estimates()
        schedule = PR.portfolio_concurrent_schedule(cost_estimates, concurrency["admitted_workers"])
        concurrency["schedule"] = schedule
        concurrency["submission_order_capsule_sha256"] = [
            self.session.inputs.portfolio_sentinels[index].capsule_sha256 for index in schedule["submission_order"]
        ]

        def analyze_index(index: int):
            member_started = time.monotonic()
            timeout = max(0.0, deadline - member_started)
            allocation = {
                **planning[index],
                "planning_allocated_seconds": planning[index]["allocated_seconds"],
                "allocated_seconds": timeout,
                "policy": "shared_portfolio_deadline_with_measured_cost_lpt_admission",
            }
            result = self._analyze_portfolio_member(
                submitted,
                candidate_sha256=after,
                sentinel=self.session.inputs.portfolio_sentinels[index],
                timeout_s=timeout,
                scope=scope,
                primary=index == 0,
                analyzer_override=self._member_analyzer(),
            )
            return (*result, allocation, time.monotonic() - member_started)

        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrency["admitted_workers"], thread_name_prefix="phase2-model"
        )
        futures = {index: executor.submit(analyze_index, index) for index in schedule["submission_order"]}
        try:
            for index in range(len(self.session.inputs.portfolio_sentinels)):
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
        self.session.check_inputs()
        analysis["optimization_baseline"] = copy.deepcopy(self.session.inputs.optimization_baseline_binding)
        dependencies_after = self.session.inputs.compiler_dependencies(submitted)
        if dependencies_before != dependencies_after:
            raise ValueError("compiler implementation dependencies changed during full-model analysis")
        if before != after or analysis.get("candidate_sha256") != after:
            raise ValueError("full-model analysis is not bound to the current candidate bytes")
        if analysis.get("workload", {}).get("capsule_sha256") != self.session.inputs.sentinel.capsule_sha256:
            raise ValueError("analysis substituted the frozen complete-model objective")
        primary_readiness = EA.global_iteration_readiness(analysis)
        if (
            isinstance(completed_sandboxes, Mapping)
            and Path(completed_sandboxes["candidate"]["package_path"]).resolve() == submitted.resolve()
        ):
            self.session.journal.register_compiler_sandboxes(len(self.session.journal.iterations), completed_sandboxes)
            comparison_policy = completed_sandboxes.get("baseline")
            if (
                primary_readiness["status"] == "ready_for_probe_admission"
                and isinstance(comparison_policy, Mapping)
                and Path(comparison_policy["package_path"]).resolve()
                == self.session.inputs.optimization_baseline.resolve()
                and comparison_policy.get("compiler_dependencies")
                == self.session.inputs.optimization_baseline_binding["compiler_dependencies"]
            ):
                self.session.journal.register_optimization_baseline_sandbox(comparison_policy)
        previous_members = (
            {
                member["identity"]["capsule_sha256"]: member
                for member in ((self.session.journal.iterations[-1].get("portfolio") or {}).get("members") or ())
            }
            if self.session.journal.iterations
            else {}
        )
        portfolio_rows: list[dict[str, Any]] = []
        portfolio_artifacts: dict[str, Mapping[str, Any]] = {self.session.inputs.sentinel.capsule_sha256: retained}
        for index, sentinel in enumerate(self.session.inputs.portfolio_sentinels[1:], start=1):
            member_result = member_results[index]
            assert member_result is not None
            member_analysis, member_artifacts, _, member_allocation, member_elapsed = member_result
            portfolio_artifacts[sentinel.capsule_sha256] = member_artifacts
            member_readiness = EA.global_iteration_readiness(member_analysis)
            previous_member = previous_members.get(sentinel.capsule_sha256)
            member_comparison = self.compare_analyses(
                previous_member.get("analysis") if previous_member else None,
                member_analysis,
                previous_iteration=self.session.journal.iterations[-1]["iteration"] if previous_member else None,
            )
            portfolio_rows.append(
                {
                    "identity": sentinel_identity(sentinel, role="training"),
                    "status": ("completed" if member_readiness["status"] == "ready_for_probe_admission" else "failed"),
                    "analysis": member_analysis,
                    "readiness": member_readiness,
                    "static_comparison": member_comparison,
                    "analysis_allocation": member_allocation,
                    "elapsed_seconds": member_elapsed,
                    "timing_status": "UNMEASURED_FULL_MODEL",
                }
            )
        fast_evaluation = self.session.fast_evaluation.evaluate(
            [
                (sentinel, member_result[0], member_result[1])
                for sentinel, member_result in zip(self.session.inputs.portfolio_sentinels, member_results, strict=True)
                if member_result is not None
            ],
            edit_contract=self.session.edit_authority.contract,
            target_descriptor=self.session.inputs.target_descriptor,
            target_sha256=self.session.inputs.target_sha256,
            portfolio_sha256=self.session.inputs.portfolio_identity_sha256,
        )
        for sentinel, member_result in zip(
            self.session.inputs.portfolio_sentinels[1:], member_results[1:], strict=True
        ):
            assert member_result is not None
            self.session.journal.retain_portfolio_baseline(sentinel.capsule_sha256, member_result[1])
        self.session.check_inputs()
        if (
            hash_tree(submitted)["sha256"] != after
            or self.session.inputs.compiler_dependencies(submitted) != dependencies_after
        ):
            raise ValueError("submitted compiler bytes changed during portfolio analysis")
        readiness = copy.deepcopy(primary_readiness)
        portfolio_blockers = [
            f"portfolio:{member['identity']['capsule']}:{blocker}"
            for member in portfolio_rows
            for blocker in member["readiness"]["blockers"]
        ]
        if portfolio_blockers:
            readiness["status"] = "blocked"
            readiness["blockers"] = [*readiness["blockers"], *portfolio_blockers]
        readiness["portfolio_sha256"] = self.session.inputs.portfolio_identity_sha256
        readiness["portfolio_members_ready"] = sum(
            row["readiness"]["status"] == "ready_for_probe_admission"
            for row in ({"readiness": primary_readiness}, *portfolio_rows)
        )
        readiness["portfolio_members_total"] = len(self.session.inputs.portfolio_sentinels)
        readiness["selection"] = "multi_model_pareto_without_invented_static_cycle_total"
        relative_semantics: Mapping[str, Any] = {
            "status": "unavailable_target_completion_contract",
            "numerical_equivalence": False,
        }
        remaining = budget_seconds - (time.monotonic() - started)
        if self.completion_contract is not None and remaining > 0:
            from merlin.perf.completion_delta import qualify_relative_completion_delta

            relative_semantics = qualify_relative_completion_delta(
                previous_analysis=self.session.journal.iterations[-1]["analysis"]
                if self.session.journal.iterations
                else None,
                current_analysis=analysis,
                previous_artifacts=self.session.journal.artifacts,
                current_artifacts=retained,
                contract=self.completion_contract,
                timeout_seconds=min(30, remaining),
            )
        elapsed = time.monotonic() - started
        if elapsed > budget_seconds:
            readiness = {
                **readiness,
                "status": "blocked",
                "blockers": [*readiness["blockers"], "iteration_wall_budget_exceeded"],
            }
        record = {
            "schema": "global_perf_iteration_v1",
            "iteration": len(self.session.journal.iterations),
            "candidate_path": str(candidate.resolve()),
            "candidate_sha256": after,
            "submitted_snapshot": str(submitted.resolve()),
            "compiler_dependencies": dependencies_after,
            "analysis_reuse_binding": self.analysis_reuse_binding(
                candidate_sha256=after, compiler_dependencies=dependencies_after
            ),
            "cross_run_static_analysis_binding": self.cross_run_static_analysis_binding(
                candidate_sha256=after, compiler_dependencies=dependencies_after
            ),
            "baseline_sha256": self.session.inputs.baseline_sha256,
            "optimization_baseline_sha256": self.session.inputs.optimization_baseline_sha256,
            "optimization_baseline": copy.deepcopy(self.session.inputs.optimization_baseline_binding),
            "compiler_mechanism_catalog": copy.deepcopy(self.session.mechanism_program.catalog_binding),
            "compiler_mechanism_work_order": copy.deepcopy(self.session.mechanism_program.work_order_binding),
            "mechanism_work_order_analysis": copy.deepcopy(self.session.mechanism_program.analysis_binding),
            "round_mechanism_attribution": copy.deepcopy(mechanism_attribution),
            "hypothesis": hypothesis,
            "analysis": analysis,
            "readiness": readiness,
            "historical_reference": copy.deepcopy(self.session.inputs.historical_reference),
            "elapsed_seconds": elapsed,
            "timing_status": "UNMEASURED_FULL_MODEL",
            "allocated_seconds": budget_seconds,
            "probe_receipts": [],
            "global_performance_claim": "unproven",
            "relative_semantic_evidence": relative_semantics,
            "fast_evaluation": fast_evaluation,
        }
        record["static_comparison"] = self.compare(record)
        self.apply_functional_gate(record, retained)
        readiness = record["readiness"]
        record["portfolio"] = {
            "schema": "full_model_portfolio_iteration_v1",
            "portfolio_sha256": self.session.inputs.portfolio_identity_sha256,
            "candidate_sha256": after,
            "members": [
                {
                    "identity": sentinel_identity(self.session.inputs.sentinel, role="primary"),
                    "status": ("completed" if primary_readiness["status"] == "ready_for_probe_admission" else "failed"),
                    "analysis_ref": "/analysis",
                    "readiness": primary_readiness,
                    "static_comparison_ref": "/static_comparison",
                    "analysis_allocation": primary_allocation,
                    "elapsed_seconds": primary_elapsed,
                    "timing_status": "UNMEASURED_FULL_MODEL",
                },
                *portfolio_rows,
            ],
            "members_ready": readiness["portfolio_members_ready"],
            "members_total": readiness["portfolio_members_total"],
            "selection": readiness["selection"],
            "analysis_allocation_policy": "shared_portfolio_deadline_with_measured_cost_lpt_admission",
            "analysis_concurrency": concurrency,
            "full_model_simulation_allowed": False,
        }
        static_bundle = self.persist_static_analysis_bundle(record, retained, portfolio_artifacts=portfolio_artifacts)
        if static_bundle is not None:
            record["static_analysis_bundle"] = static_bundle
        self.session.journal.publish_analysis(record, retained, portfolio_artifacts)
        return copy.deepcopy(record)

    def apply_functional_gate(
        self,
        record: dict[str, Any],
        artifacts: Mapping[str, Any] | None,
        *,
        reused_from: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute the objective's emitted program and bind the verdict into this iteration.

        Static counters cannot tell a -40% win from a miscompile, and until this step the loop
        never executed a candidate at all (`full_model_simulation_allowed` was False at every
        site). With a gate configured, the objective's retained `lowered_text` is built into the
        model's whole-model ELF and run on the configured simulator; the harness's own
        `MERLIN_RESULT` fields are compared to the declared expectations.

        The verdict lands in three places so nothing downstream can miss it: `functional_gate` on
        the record (the full receipt), `static_comparison.functional_gate` (what the agent reads),
        and -- only on `failed` -- a `functional_gate_failed` readiness blocker, which is what keeps
        `best_authored_candidate`, `seal` and every `require_ready` accessor from ever selecting
        the revision. `not_run` (toolchain absent, timeout, no retained artifact, no gate
        configured) never blocks and is never success; it is recorded as exactly what it is.

        An exact-reuse iteration inherits its source's verdict: the same bytes produced the same
        emission, so re-running would only re-measure the same program.
        """
        iteration = int(record["iteration"])
        lowered_sha = ((record.get("analysis") or {}).get("emission") or {}).get("candidate_lowered_sha256")
        binding = {
            "iteration": iteration,
            "candidate_sha256": record.get("candidate_sha256"),
            "candidate_lowered_sha256": lowered_sha,
        }
        if self.functional_gate is None:
            gate: dict[str, Any] = {
                "schema": "merlin_functional_gate_result_v1",
                "status": FUNCTIONAL_GATE_NOT_RUN,
                "reason": "no functional gate configured for this run (--functional-gate absent): "
                "this iteration was scored on static counters only and its numerics "
                "were NOT executed",
                "stage": "preflight",
                "configured": False,
                "simulation_executed": False,
                "excludes_candidate": False,
                **binding,
            }
        elif (
            reused_from is not None
            and isinstance(reused_from.get("functional_gate"), Mapping)
            and reused_from["functional_gate"].get("candidate_lowered_sha256") == lowered_sha
        ):
            gate = copy.deepcopy(dict(reused_from["functional_gate"]))
            gate.update(binding)
            gate["reused_from_iteration"] = reused_from.get("iteration")
        else:
            lowered = artifacts.get("lowered_text") if isinstance(artifacts, Mapping) else None
            buffer = artifacts.get("command_buffer") if isinstance(artifacts, Mapping) else None
            if not isinstance(lowered, str) or not lowered:
                gate = {
                    "schema": "merlin_functional_gate_result_v1",
                    "status": FUNCTIONAL_GATE_NOT_RUN,
                    "stage": "preflight",
                    "reason": "the objective's emitted artifact was not retained for this "
                    "iteration, so there is nothing to build",
                    "configured": True,
                    "simulation_executed": False,
                    "excludes_candidate": False,
                    **binding,
                }
            else:
                config = self.functional_gate
                workdir = self.session.journal.output / f"functional_gate_{iteration:04d}"
                try:
                    result = self.functional_gate_runner(
                        lowered,
                        buffer if isinstance(buffer, Mapping) else None,
                        model_payload_dir=config.model_payload_dir,
                        toolchain=config.toolchain,
                        gate_spec=config.gate_spec,
                        workdir=workdir,
                        timeout=config.timeout_seconds,
                        keep_elf=config.keep_elf,
                    )
                    gate = result.to_dict() if isinstance(result, FunctionalGateResult) else dict(result)
                except Exception as exc:  # a broken gate is `not_run`, visibly -- never a pass
                    gate = {
                        "schema": "merlin_functional_gate_result_v1",
                        "status": FUNCTIONAL_GATE_NOT_RUN,
                        "stage": "runner",
                        "reason": f"gate runner raised {type(exc).__name__}: {exc}",
                        "simulation_executed": False,
                        "excludes_candidate": False,
                    }
                gate.update(binding)
                gate["configured"] = True
                gate["config_sha256"] = config.source_sha256
                gate["config_path"] = str(config.source_path) if config.source_path else None
        if gate.get("status") not in ("passed", "failed", FUNCTIONAL_GATE_NOT_RUN):
            gate = {
                **gate,
                "status": FUNCTIONAL_GATE_NOT_RUN,
                "reason": f"gate returned an unknown status {gate.get('status')!r}",
            }
        excluded, why = gate_result_for_selection(gate)
        gate["excludes_candidate"] = excluded
        record["functional_gate"] = gate
        comparison = record.get("static_comparison")
        if not isinstance(comparison, dict):
            comparison = {}
            record["static_comparison"] = comparison
        comparison["functional_gate"] = {
            "status": gate["status"],
            "reason": gate.get("reason"),
            "stage": gate.get("stage"),
            "configured": bool(gate.get("configured")),
            "simulation_executed": bool(gate.get("simulation_executed")),
            "excludes_candidate": excluded,
            "verdict_line": gate.get("verdict_line"),
            "fields": dict(gate.get("fields") or {}),
            "expected": gate.get("expected"),
            "mismatches": [
                row for row in (gate.get("comparisons") or ()) if isinstance(row, Mapping) and not row.get("ok")
            ],
            "reading": (
                "the emitted program EXECUTED and matched the model's gate; the static "
                "deltas above describe a numerically valid revision"
                if gate["status"] == "passed"
                else "the emitted program EXECUTED and did NOT match the model's gate; this "
                "revision is excluded from selection regardless of its static deltas"
                if excluded
                else "the emitted program was NOT executed for this iteration; the static "
                "deltas above say nothing about its numerics"
            ),
        }
        if excluded:
            readiness = dict(record.get("readiness") or {})
            blockers = [str(item) for item in (readiness.get("blockers") or ())]
            if "functional_gate_failed" not in blockers:
                blockers.append("functional_gate_failed")
            readiness.update(status="blocked", blockers=blockers, functional_gate={"status": "failed", "reason": why})
            record["readiness"] = readiness
        return gate

    def compare(self, current: Mapping[str, Any]) -> dict[str, Any]:
        """Compare full-model counters without inventing a total from partial accounting."""
        previous = self.session.journal.iterations[-1] if self.session.journal.iterations else None
        return self.compare_analyses(
            previous["analysis"] if previous else None,
            current["analysis"],
            previous_iteration=previous["iteration"] if previous else None,
            previous_feedback=previous.get("decision_feedback") if previous else None,
        )

    @staticmethod
    def compare_analyses(
        previous: Mapping[str, Any] | None,
        current: Mapping[str, Any],
        *,
        previous_iteration: int | None,
        previous_feedback: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Apply the same target-neutral structural comparison to every portfolio member."""

        def counters(analysis: Mapping[str, Any]) -> dict[str, int | float | None]:
            diag = analysis.get("diagnostics") or {}
            arm = (diag.get("arms") or {}).get("candidate") or {}
            movement = arm.get("movement") or {}
            plan = diag.get("verified_global_plan_emission") or {}
            # Include host-lane work independently of command-buffer/contraction quantities.
            # Otherwise a host-only improvement can produce zero deltas in every reported
            # counter. Structural changes remain distinct from measured timing benefits.
            host = plan.get("host_activity") or {}
            host_ops = host.get("static_operations") or {}
            host_dyn = host.get("dynamic_operations") or {}
            return {
                "command_buffer_macs": arm.get("macs") if arm.get("exact") is True else None,
                "full_model_contraction_macs": (
                    (diag.get("model_contraction_placement") or {}).get("candidate") or {}
                ).get("total_contraction_macs"),
                "command_buffer_declared_movement_bytes": movement.get("exact_bytes")
                if isinstance(movement.get("exact_bytes"), (int, float))
                and not isinstance(movement.get("exact_bytes"), bool)
                else None,
                "dispatches": plan.get("emitted_dispatches", plan.get("tasks")),
                "host_load_payload_bytes": host.get("load_payload_bytes"),
                "host_store_payload_bytes": host.get("store_payload_bytes"),
                "host_static_allocation_payload_bytes": host.get("static_allocation_payload_bytes"),
                "host_static_allocations": host_ops.get("allocation"),
                # Dynamic counts describe executed host work, unlike payload bytes alone.
                # Inspect the operation mix before interpreting a reduction: removing integer
                # work while leaving floating work unchanged is not a measured cycle benefit.
                "host_dynamic_operations_total": (
                    sum(v for v in host_dyn.values() if isinstance(v, int) and not isinstance(v, bool))
                    if host_dyn
                    else None
                ),
                # ... and the MIX, per family, because the total alone has no gradient worth
                # following. The warning above was written as a comment and therefore reached
                # nobody: the authoring loop kept removing `integer_arithmetic` and `branch` --
                # the cheap families -- while `floating_arithmetic` sat at 164,554 untouched,
                # because one collapsed scalar cannot say WHICH family moved. These per-family
                # counters flow into `candidate_minus_previous` like every other numeric field, so
                # an edit that trades 10,000 integer operations for 100 float ones is visible as
                # what it is instead of reading as a win.
                **{
                    f"host_dynamic_operations_{family}": value
                    for family, value in sorted(host_dyn.items())
                    if isinstance(value, int) and not isinstance(value, bool)
                },
                # The ACCELERATOR's own occupancy, which none of the above represents. Movement
                # bytes say what crossed DRAM and `dispatches` counts launches, but neither says
                # how many cycles the array is occupied issuing them -- so an edit that changes
                # which operand is stationary, or how a tile is shaped, moved this term by 25%
                # while every scored field above stayed identical. A loop that cannot see it will
                # treat mesh efficiency as free and spend it without noticing.
                "mesh_issue_cycles": plan.get("mesh_issue_cycles"),
                "accelerator_fences": plan.get("fences"),
            }

        def cost_locations(analysis: Mapping[str, Any], *, limit: int = 8) -> list[dict[str, Any]]:
            """The most expensive host tasks, NAMED, with the family that dominates each.

            A magnitude with no location is not a gradient. The loop was told only how many host
            operations a whole model executes, never which of its tasks executed them, so every
            edit was a guess about where the cost was. The per-task breakdown and its
            `source_regions` already exist in `host_cfg_activity_v1`; this only stops discarding
            them. `share_of_host_dynamic_operations` is the fraction of the whole host lane, so a
            task worth working on is distinguishable from one that is already noise.
            """
            plan = (analysis.get("diagnostics") or {}).get("verified_global_plan_emission") or {}
            host = plan.get("host_activity") or {}
            tasks = host.get("tasks")
            if not isinstance(tasks, list):
                return []
            whole = sum(
                v
                for v in (host.get("dynamic_operations") or {}).values()
                if isinstance(v, int) and not isinstance(v, bool)
            )
            ranked: list[tuple[int, dict[str, Any]]] = []
            for index, task in enumerate(tasks):
                if not isinstance(task, Mapping):
                    continue
                families = {
                    name: value
                    for name, value in (task.get("dynamic_operations") or {}).items()
                    if isinstance(value, int) and not isinstance(value, bool)
                }
                total = sum(families.values())
                if not total:
                    continue
                dominant = max(families.items(), key=lambda item: (item[1], item[0]))
                regions = [str(name) for name in (task.get("source_regions") or [])]
                ranked.append(
                    (
                        total,
                        {
                            "task_index": index,
                            "dynamic_operations": total,
                            "share_of_host_dynamic_operations": round(total / whole, 6) if whole else None,
                            "dominant_family": dominant[0],
                            "dominant_family_operations": dominant[1],
                            "dynamic_operations_by_family": dict(sorted(families.items())),
                            # Truncated because a task can name hundreds of regions and this payload is
                            # read by the agent every iteration; the count is kept so the truncation is
                            # never mistaken for the whole task.
                            "source_regions": regions[:12],
                            "source_region_count": len(regions),
                        },
                    )
                )
            ranked.sort(key=lambda item: -item[0])
            return [entry for _, entry in ranked[:limit]]

        def reference_gap(analysis: Mapping[str, Any]) -> dict[str, Any]:
            """Distance to a MEASURED destination for this objective -- or the fact there is none.

            An authoring loop that sees only its own last revision can tell that it moved, never
            whether it moved far enough or toward anything. This reports the objective's own
            measured reference and the estimated distance to it, and when the objective has no
            measured reference it says SO, loudly, because that is the more important finding: a
            campaign ran seventeen iterations against an objective with no reference point and
            sealed a round at -2 host operations, and nothing in the loop could say that the model
            being optimized had no destination to be optimized toward.
            """
            from merlin.perf.target_reference import estimate_cycles, find_reference_for_capsule

            capsule = str(((analysis.get("workload") or {}).get("capsule")) or "")
            if not capsule:
                return {"status": "objective_capsule_not_declared"}
            found = find_reference_for_capsule(capsule)
            if found is None:
                return {
                    "status": "no_measured_reference_for_objective",
                    "objective_capsule": capsule,
                    "consequence": (
                        "there is no measured cycle count for this objective, so no edit made "
                        "against it can be shown to close a gap toward a known destination; the "
                        "host-operation deltas below are self-relative only"
                    ),
                    "remedy": (
                        "add a portfolio member whose capsule appears in merlin/contract/perf_reference_targets.yaml"
                    ),
                }
            name, reference = found
            plan = (analysis.get("diagnostics") or {}).get("verified_global_plan_emission") or {}
            host_dyn = (plan.get("host_activity") or {}).get("dynamic_operations") or {}
            total = sum(v for v in host_dyn.values() if isinstance(v, int) and not isinstance(v, bool))
            record: dict[str, Any] = {
                "status": "derived",
                "reference": name,
                "objective_capsule": capsule,
                "reference_whole_model_cycles": (reference.get("measured") or {}).get("whole_model_cycles"),
            }
            if total:
                # Absolute, because there is no measured host-operation count to divide by: the
                # reference measured CYCLES, not the operations its own host lane executed. So this
                # is the weaker of the two forms and is labelled as such by `basis`.
                record["cycles"] = estimate_cycles(total, reference)
            return record

        left = counters(previous) if previous else {}
        right = counters(current)
        changes = {
            name: right[name] - left[name]
            for name in right
            if isinstance(right[name], (int, float)) and isinstance(left.get(name), (int, float))
        }
        from merlin.perf.structural_delta import compare_full_model_structure

        return {
            "structural_change": compare_full_model_structure(previous, current)
            if previous
            else {"status": "initial_observation", "cycle_selection": "UNMEASURED"},
            "previous_iteration": previous_iteration,
            "candidate_totals": right,
            "candidate_minus_previous": changes,
            "host_cost_locations": cost_locations(current),
            "reference_gap": reference_gap(current),
            "unknown_metrics": [name for name, value in right.items() if value is None],
            "selection": "requires_global_cost_evidence",
            "previous_revision_mechanism_feedback": (
                {"applies_to_current_revision": False, "evidence": copy.deepcopy(previous_feedback)}
                if previous_feedback
                else None
            ),
            "licence": "structural accounting only; fewer dispatches do not prove fewer cycles",
        }

    @staticmethod
    def authored_host_cost(analysis: Mapping[str, Any] | None) -> tuple[int, int] | None:
        """The host-lane payload an analyzed revision's emitted program moves, or None.

        Ordered (bytes, allocations) so bytes decide and allocation count breaks ties. This is the
        quantity an authoring round is actually able to move -- the command buffer is frequently
        byte-identical across a real host-lane improvement -- so it is what "best" means here.
        Returns None when the analysis did not emit a verified plan, which makes such a revision
        ineligible rather than implicitly best.
        """
        if not analysis:
            return None
        plan = (analysis.get("diagnostics") or {}).get("verified_global_plan_emission") or {}
        host = plan.get("host_activity") or {}
        parts = [host.get("load_payload_bytes"), host.get("store_payload_bytes")]
        if any(not isinstance(v, int) or isinstance(v, bool) for v in parts):
            return None
        allocations = (host.get("static_operations") or {}).get("allocation")
        return (sum(parts), allocations if isinstance(allocations, int) else 0)

    def best_authored_candidate(self) -> dict[str, Any] | None:
        """The cheapest ready revision authored so far, as a sealable selection, or None.

        A round used to seal whatever the agent's workspace held when its budget expired, i.e. the
        LAST revision rather than the BEST one. Measured over two runs and five rounds that lost
        every improvement the agent found: iterations that removed host allocations were never
        sealed, four consecutive rounds sealed one neutral revision, and each round then resumed
        from that neutral seal -- so wins could not accumulate. Selecting here makes an improvement
        survive its own round.

        Only revisions that are ready AND whose preserved snapshot still hashes to the bytes the
        analysis was earned on are eligible; a snapshot that has drifted is skipped rather than
        trusted. Ties keep the latest iteration, so an equal-cost later revision still wins and the
        agent's most recent work is preferred when nothing improved.
        """
        ranked: list[tuple[tuple[int, int], int, dict[str, Any]]] = []
        excluded: list[dict[str, Any]] = []
        for row in self.session.journal.iterations:
            # A revision whose emitted program EXECUTED and failed the model's gate is never
            # "best", whatever its counters say: a -40% host-operation win that miscompiles is
            # the one outcome this selection must not seal. Checked before readiness so the
            # exclusion is explicit even when the readiness blocker was not applied.
            gate_excluded, why = gate_result_for_selection(row.get("functional_gate"))
            if gate_excluded:
                excluded.append(
                    {"iteration": int(row["iteration"]), "candidate_sha256": row.get("candidate_sha256"), "reason": why}
                )
                continue
            if (row.get("readiness") or {}).get("status") != "ready_for_probe_admission":
                continue
            cost = self.authored_host_cost(row.get("analysis"))
            if cost is None:
                continue
            snapshot = row.get("submitted_snapshot")
            if not snapshot or not Path(snapshot).is_dir():
                continue
            try:
                if hash_tree(Path(snapshot))["sha256"] != row["candidate_sha256"]:
                    continue
            except (OSError, ValueError):
                continue
            ranked.append((cost, int(row["iteration"]), row))
        if not ranked:
            return None
        cost, iteration, row = min(ranked, key=lambda item: (item[0], -item[1]))
        return {
            "iteration": iteration,
            "candidate_sha256": row["candidate_sha256"],
            "snapshot": str(Path(row["submitted_snapshot"]).resolve()),
            "host_payload_bytes": cost[0],
            "host_static_allocations": cost[1],
            "considered": len(ranked),
            "functional_gate": (row.get("functional_gate") or {}).get("status"),
            "excluded_functional_gate_failures": excluded,
        }

    def _write(self, name: str, record: Mapping[str, Any]) -> Path:
        path = self.session.journal.output / name
        payload = P2_CONTRACTS.canonical_json(record)
        with path.open("xb") as stream:
            stream.write(payload)
        path.chmod(0o444)
        return path
