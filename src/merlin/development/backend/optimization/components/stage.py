"""Re-evaluate original compile/static roles for actual observed descendant bytes.

The qualified baseline selects the immutable roster and build/runtime owners.
Its old evaluation never certifies an edited candidate. This fixed ordinary
controller retains new evidence and refuses unresolved original obligations
before numerical qualification; it grants no proof by declaration or callback.
"""

from __future__ import annotations

import shutil
import time
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from weakref import WeakKeyDictionary

from merlin.common.digest import hash_tree, sha256_bytes
from merlin.development.backend.optimization import contracts as C
from merlin.development.backend.optimization.author_interface import broker_service as B
from merlin.development.backend.optimization.author_interface import stage_inputs as SI
from merlin.development.backend.optimization.author_interface import stage_prompt as SP
from merlin.development.backend.optimization.authoring import telemetry as _merlin_telemetry_1
from merlin.development.backend.optimization.authoring import telemetry_accounting as TEL
from merlin.development.backend.optimization.authoring.transcript_audit import audit_codex_transcript
from merlin.development.backend.optimization.components import launch as L
from merlin.development.backend.optimization.components.execution import ComponentIndependentExecution
from merlin.development.backend.optimization.components.experiment import strict_tool_policy
from merlin.development.backend.optimization.components.policy import ComponentOnlyPolicy
from merlin.development.backend.optimization.contracts import StageGateError, document_sha256, exact_tree_record
from merlin.development.backend.qualification.components.component_compile_roles import (
    evaluate_component_compile_roles,
)
from merlin.development.backend.qualification.components.component_qualification import (
    qualify_component_compiler,
)
from merlin.development.backend.qualification.components.witnesses import run_component_origin_round

# ---- component_final_qualification -------------------------------------------------------------


def qualify_observed_component_descendant(inputs, lineage, *, timeout_s):
    """Use original selection owners and freshly evaluate the observed compiler."""
    baseline = inputs.qualification
    baseline.verify(candidate=inputs.edit_authority.seed)
    original = baseline.compile_role_evaluation
    evaluation = evaluate_component_compile_roles(
        roster=original.roster,
        compiler_origin=baseline.compiler_origin,
        compiler_lineage=lineage,
        candidate=inputs.candidate,
        contract_root=baseline.contract_root,
        build_service=original.build_service,
        instruction_check=original.instruction_check,
        readelf=original.readelf,
        evidence_root=inputs.stage_root / "final_compile_role_evaluation",
        timeout_s=timeout_s,
    )
    if evaluation is original:
        raise StageGateError("descendant compilation cannot reuse its baseline's static evaluation")
    evaluation.require_complete(candidate=inputs.candidate)
    return qualify_component_compiler(
        inputs.candidate,
        corpus_root=baseline.corpus_root,
        target_experiment=inputs.policy.target_experiment,
        contract_root=baseline.contract_root,
        source_root=baseline.source_root,
        evidence_root=inputs.stage_root / "final_functional_qualification",
        runtime=baseline.runtime,
        view=inputs.view,
        timeout_s=timeout_s,
        compiler_origin=baseline.compiler_origin,
        compiler_lineage=lineage,
        runtime_authority=baseline.runtime_authority,
        compile_role_evaluation=evaluation,
    )


# ---- component_stage ---------------------------------------------------------------------------
# Ordinary isolated component authoring and descendant requalification lifecycle.
#
# The launch edge retains its stable public entrypoint. This executing controller
# preserves exact original feedback owners, observed lineage and complete final
# compile/static/numeric gates; sealing a refused candidate cannot promote it.


def run_component_stage(
    launch: L.QualifiedComponentLaunch,
    *,
    model: str,
    effort: str,
    wall_budget_seconds: int,
    max_tool_calls: int,
    tool_timeout_seconds: int,
    suite: str,
    search_configuration: Path | None = None,
) -> Path:
    """Author one fresh bounded session, audit receipts, and requalify final bytes."""
    with ExitStack() as cleanup:
        return _run_component_stage(
            launch,
            model=model,
            effort=effort,
            wall_budget_seconds=wall_budget_seconds,
            max_tool_calls=max_tool_calls,
            tool_timeout_seconds=tool_timeout_seconds,
            suite=suite,
            search_configuration=search_configuration,
            cleanup=cleanup,
        )


def _run_component_stage(
    launch,
    *,
    model,
    effort,
    wall_budget_seconds,
    max_tool_calls,
    tool_timeout_seconds,
    suite,
    search_configuration,
    cleanup,
):
    if type(launch) is not L.QualifiedComponentLaunch:
        raise C.StageGateError("component authoring requires evaluated launch prerequisites")
    launch.verify()
    if model != launch.selected_model or effort != launch.selected_effort:
        raise C.StageGateError("component authoring cannot change the qualified model transport")
    if (
        any(
            type(value) is not int or value <= 0
            for value in (wall_budget_seconds, max_tool_calls, tool_timeout_seconds)
        )
        or tool_timeout_seconds > 600
    ):
        raise C.StageGateError("component authoring budgets require positive integers and a bounded tool ceiling")
    inputs = launch.inputs
    search = None
    if search_configuration is not None:
        from merlin.development.backend.optimization.search.search_configuration import (
            read_search_configuration,
        )

        search = read_search_configuration(search_configuration)
        if launch.selected_subagent_count != search.author_agents:
            raise C.StageGateError("search must use the author concurrency qualified by its launch canary")
        if not search.settings.final_reserve_seconds < wall_budget_seconds <= search.settings.hard_seconds:
            raise C.StageGateError("search wall budget must preserve its final reserve within the hard ceiling")
    if not isinstance(suite, str) or not suite.strip():
        raise C.StageGateError("component authoring requires an explicit accounting suite")
    telemetry_preflight = _merlin_telemetry_1.prepare(
        model=model,
        authoring_stage=run_component_stage,
        price_table=inputs.price_table,
        codex_binary=inputs.codex_binary,
    )
    # Readiness receipts are retained as a separate pre-authoring attempt. Use a
    # new exact policy owner for authoring rather than adopting probe feedback.
    policy = ComponentOnlyPolicy(
        candidate=inputs.candidate,
        target_experiment=inputs.policy.target_experiment,
        receipt_path=inputs.stage_root / "control-authoring" / "receipts.jsonl",
        component_corpus=inputs.policy.component_corpus,
        component_analytical=inputs.policy.component_analytical,
        component_cca=inputs.policy.component_cca,
        component_rtl=inputs.policy.component_rtl,
        services=inputs.policy.services,
        baseline_admission=inputs.policy.baseline_admission,
        independent_runtime=inputs.policy.independent_runtime,
    )
    pool = None
    if search is not None:
        pool = search.install(policy, edit_authority=inputs.edit_authority, root=inputs.stage_root / "search")
        cleanup.callback(pool.close)
        wall_budget_seconds = (
            min(wall_budget_seconds, search.settings.soft_seconds) - search.settings.final_reserve_seconds
        )
    # Copy only the immutable launch configuration with this new receipt owner.
    authored_inputs = replace(inputs, policy=policy)
    tool = L.ComponentToolPolicy(
        authored_inputs,
        strict_tool_policy(
            inputs.view,
            inputs.candidate,
            runtime=inputs.runtime,
            candidate_destination=str(inputs.candidate),
            bwrap_binary=inputs.sandbox_binary,
        ),
    )
    actions = policy.build_registry()
    prompt_inputs = SI.prepare_component_prompt_inputs(
        policy,
        corpus_manifest_path="/component-inputs/contract/performance_corpus_manifest.json",
        candidate_path=str(inputs.candidate),
        allowed_paths=(
            "/component-inputs",
            "/component-inputs/contract/performance_corpus_manifest.json",
            str(inputs.candidate),
            B.BROKER_NAME,
            str(B.BROKER_RECEIPT_MOUNT),
        ),
        wall_budget_seconds=wall_budget_seconds,
        max_tool_calls=max_tool_calls,
        tool_timeout_seconds=tool_timeout_seconds,
        broker_path=B.BROKER_NAME,
        broker_receipt_path=str(B.BROKER_RECEIPT_MOUNT),
        public_manifest_sha256=sha256_bytes(L.public_component_manifest(policy.component_corpus)),
        author_agents=search.author_agents if search is not None else None,
    )
    prompt = SP.render_component_prompt(prompt_inputs)
    broker = B.Broker(
        tool,
        policy.target_experiment,
        inputs.candidate,
        actions,
        policy.receipt_path,
        deadline=time.monotonic() + wall_budget_seconds,
        workflow=policy,
        max_calls=max_tool_calls,
        max_tool_seconds=tool_timeout_seconds,
        public_control_dir=authored_inputs.public_control_dir,
    )
    B.stage_broker_shim(
        authored_inputs.public_control_dir,
        host="",
        port=0,
        token=broker.token,
        actions=actions,
        tool_timeout_s=tool_timeout_seconds,
        socket_path="/perf-control/channel.sock",
    )
    refusals, audit, receipts, final_qualification, telemetry = [], None, None, None, None
    started = time.monotonic()
    search_observation = None
    try:
        with broker.serving(socket_path=authored_inputs.socket_path):
            rc, transcript, lineage = run_component_origin_round(
                launch,
                authored_inputs,
                model=model,
                effort=effort,
                wall_budget_seconds=wall_budget_seconds,
                prompt=prompt,
            )
        if pool is not None:
            search.verify()
            search_observation = policy.search_coordinator.status()
    finally:
        if pool is not None:
            pool.close()
    if pool is not None:
        search_observation = policy.search_coordinator.status()
    try:
        round_telemetry = _merlin_telemetry_1.collect_round(
            inputs.stage_root, 0, model=model, agent_exit_code=rc, preflight_record=telemetry_preflight
        )
        telemetry = _merlin_telemetry_1.finalize(
            inputs.stage_root,
            [{"round": 0, "telemetry": round_telemetry}],
            model=model,
            target=policy.target_experiment.target,
            suite=suite,
            run_id=inputs.stage_root.name,
            preflight_record=telemetry_preflight,
        )
        authored_inputs.verify()
        audit = audit_codex_transcript(transcript, policy.target_experiment, inputs.candidate, actions)
        if audit.get("clean") is not True:
            raise C.StageGateError("component authoring transcript audit failed")
        digest = str(hash_tree(inputs.candidate)["sha256"])
        if search is not None:
            search.verify()
            if (
                not search_observation["finalized"]
                or search_observation["selected"] is None
                or search_observation["selected"]["source_sha256"] != digest
                or search_observation["incumbent"]["source_sha256"] != digest
            ):
                raise C.StageGateError("observed final compiler differs from the measured search incumbent")
        receipts = policy.verify_receipts(policy.receipt_path, actions, audit, candidate_sha256=digest)
        if rc not in (0, C.ROUND_DEADLINE_EXIT):
            raise C.StageGateError("component model transport did not complete an admitted bounded session")
        final_qualification = qualify_observed_component_descendant(
            authored_inputs,
            lineage,
            timeout_s=tool_timeout_seconds,
        )
        final_qualification.verify()
    except Exception as exc:  # noqa: BLE001 - seal refused evidence without promotion
        refusals.append(type(exc).__name__ + ": " + str(exc)[:500])
    sealed = inputs.stage_root / "sealed_candidate"
    shutil.copytree(inputs.candidate, sealed)
    for path in (sealed, *sealed.rglob("*")):
        if path.is_symlink():
            raise C.StageGateError("component final candidate contains a symlink")
        path.chmod(path.stat().st_mode & ~0o222)
    record = inputs.stage_root / "candidate_record.json"
    C.write_json(
        record,
        {
            "schema": "merlin.component_authoring_candidate.v1",
            "workflow_id": policy.workflow_id,
            "candidate_path": str(sealed),
            "candidate_sha256": str(hash_tree(sealed)["sha256"]),
            "launch_qualification_sha256": launch.qualification_sha256,
            "functional_qualification_sha256": final_qualification.receipt_sha256 if final_qualification else None,
            "phase1_origin_sha256": inputs.qualification.compiler_origin.receipt_sha256,
            "phase2_lineage_sha256": lineage.receipt_sha256,
            "transcript": str(transcript),
            "transcript_sha256": C.sha256_file(transcript),
            "exit_code": rc,
            "audit": audit,
            "receipts": receipts,
            "wall_seconds": time.monotonic() - started,
            "telemetry": telemetry,
            **(
                {"search": search_observation, "search_configuration_sha256": search.sha256}
                if search is not None
                else {}
            ),
            "admission": {"consumable": not refusals, "refusal": "; ".join(refusals) or None},
            "final_acceptance": "NOT_ESTABLISHED",
        },
    )
    record.chmod(0o400)
    return record


# ---- component_variants ------------------------------------------------------------------------
# Retain compiler variants observed inside the controller's frozen edit scope.
#
# These snapshots are issued from the current authorized compiler, never imported
# from a compiler directory/report supplied as a tuning answer. Retaining several
# legal emissions lets an independent measurement cohort compare schedules while
# holding every variant of the same mathematical workload together.

_ISSUED = WeakKeyDictionary()


@dataclass(frozen=True, eq=False)
class ComponentVariantSnapshot:
    execution: ComponentIndependentExecution
    compiler: Path
    compiler_sha256: str
    membership_sha256: str
    execution_sha256: str
    baseline_admission_sha256: str
    edit_authority_sha256: str

    @property
    def sha256(self):
        return document_sha256(
            {
                "schema": "merlin.component_variant_snapshot.v1",
                "compiler_sha256": self.compiler_sha256,
                "membership_sha256": self.membership_sha256,
                "execution_sha256": self.execution_sha256,
                "baseline_admission_sha256": self.baseline_admission_sha256,
                "edit_authority_sha256": self.edit_authority_sha256,
            }
        )

    def verify(self):
        issued = _ISSUED.get(self)
        if issued is None or issued != self.sha256:
            raise StageGateError("component variant requires live controller-owned source capture")
        self.execution.verify()
        if (
            self.execution.sha256 != self.execution_sha256
            or self.execution.baseline_admission.sha256 != self.baseline_admission_sha256
            or self.execution.edit_authority_sha256 != self.edit_authority_sha256
        ):
            raise StageGateError("component variant belongs to another fresh source/edit authority")
        if (
            self.compiler.resolve() != self.compiler
            or self.compiler.is_symlink()
            or str(hash_tree(self.compiler)["sha256"]) != self.compiler_sha256
            or exact_tree_record(self.compiler)["sha256"] != self.membership_sha256
        ):
            raise StageGateError("component variant source membership changed after capture")
        for path in (self.compiler, *self.compiler.rglob("*")):
            if path.is_symlink() or path.stat().st_mode & 0o222:
                raise StageGateError("component variant is not a retained immutable source snapshot")
        self.execution.edit_authority.validate_candidate(self.compiler)
        return self.sha256


def capture_component_variant(*, execution, output):
    """Capture only the actual current source after fresh-origin/edit inspection."""
    if type(execution) is not ComponentIndependentExecution:
        raise StageGateError("component variant capture requires independent normal execution authority")
    execution.verify()
    output = Path(output)
    if (
        output.exists()
        or output.is_symlink()
        or output.resolve() != output
        or not output.is_relative_to(execution.output)
    ):
        raise StageGateError("component variant requires a fresh controller-owned private output")
    before = exact_tree_record(execution.candidate)["sha256"]
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    shutil.copytree(execution.candidate, output)
    if exact_tree_record(output)["sha256"] != before or exact_tree_record(execution.candidate)["sha256"] != before:
        raise StageGateError("component source changed during controlled variant capture")
    for path in output.rglob("*"):
        path.chmod(path.stat().st_mode & ~0o222)
    output.chmod(0o555)
    execution.verify()
    variant = ComponentVariantSnapshot(
        execution,
        output,
        str(hash_tree(output)["sha256"]),
        before,
        execution.sha256,
        execution.baseline_admission.sha256,
        execution.edit_authority_sha256,
    )
    _ISSUED[variant] = variant.sha256
    variant.verify()
    return variant
