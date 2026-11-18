"""Grade a frozen agent run through capsule_bench_v0 (post-agent, full-access phase).

Public/dev phase (repairable in a real run) -> freeze -> hidden/private qualification. Writes
score_capsule.json per phase, an iteration_000 snapshot, run_manifest.yaml, and final_report.md.
Runs OUTSIDE the agent sandbox (needs spike/verilator + the hidden goldens).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections.abc import Mapping
from pathlib import Path

import yaml

from merlin.common.digest import hash_tree
from merlin.development.backend.authoring.library.compiler_library import selected_child
from merlin.development.backend.context import (
    InvocationContext,
    add_context_arguments,
    readback_kwargs,
    resolve_context,
    verify_readback_record,
)
from merlin.development.backend.feedback.certification import hidden_capsule_policy as HCP
from merlin.development.backend.feedback.grading import freeze as freeze_run
from merlin.development.backend.qualification.full_models import execution_engine as PFX
from merlin.development.backend.qualification.full_models import private_full_models as PFM
from merlin.development.backend.qualification.full_models.private_facts import selected_input_facts
from merlin.development.backend.qualification.provenance import (
    private_instruction_coordinator as instruction_coordinator,
)
from merlin.development.backend.qualification.provenance import (
    private_instruction_declaration as instruction_declaration,
)
from merlin.development.targetgen.grading import model_checks as CG
from merlin.development.targetgen.grading import submission as _merlin_submission_1
from merlin.evaluation.capsule.evidence import certificates as capsule_adapters
from merlin.target.registry import identity as capsule_runner_config
from merlin.target.simulators import oracle_policy as capsule_oracle_policy

# ---- formal ------------------------------------------------------------------------------------

FORMAL_REQUIRED_TIER = "L3"


def _formal_overlap_shares_compute(environment: Mapping[str, str]) -> bool:
    """Overlap hidden and private grading only through the one shared compute pool.

    Both graders inherit the exact CPU roster, slot directory, worker caps and
    preparation affinity. Every build/simulator step leases its CPU from the same
    shared slots, so the pool divides capacity between the graders as steps finish
    and a finished private gate leaves every slot to hidden grading. No CPU is
    partitioned, removed from the roster or reserved for either grader. Separate
    legacy per-engine rosters could oversubscribe two concurrent graders, so they
    keep serial grading; an incomplete shared selection refuses as the pool does.
    """
    from merlin.common.process.cpu_pool import grading_cpus, shared_compute_enabled

    return shared_compute_enabled(environment) and len(grading_cpus(environment)) >= 2


def _private_source_freeze_for_formal(
    run_dir: Path,
    spec: Path,
    target: str,
    *,
    workspace: Path | None = None,
    repo: Path | None = None,
    descriptor: Path | None = None,
) -> dict | None:
    """Use only the host-owned fresh-run input record, never a caller's receipt."""
    environment_path = run_dir / "environment.yaml"
    expected_root = run_dir.absolute() / "private_full_model_input" / "sources"
    if environment_path.is_symlink():
        raise ValueError("formal private run environment is indirect")
    if not environment_path.is_file():
        if expected_root.exists():
            raise ValueError("fresh private authored-source freeze has no run environment")
        return None  # historical diagnostic runs have no certifiable source freeze
    environment = yaml.safe_load(environment_path.read_bytes())
    if not isinstance(environment, Mapping):
        raise ValueError("formal private run environment is malformed")
    private = environment.get("private_full_model_spec")
    if private is None:
        if expected_root.exists():
            raise ValueError("fresh private authored-source freeze was removed from the run record")
        return None
    from merlin.runtime.execution.model_inputs import file_sha256

    if (
        not isinstance(private, Mapping)
        or private.get("frozen_path") != str(spec.absolute())
        or private.get("frozen_sha256") != file_sha256(spec)
    ):
        raise ValueError("formal private model spec differs from the frozen run input")
    record = private.get("source_freeze")
    if record is None:
        if expected_root.exists():
            raise ValueError("fresh private authored-source freeze was removed from the run record")
        return None
    expected_spec = run_dir.absolute() / "private_full_model_input" / "spec.yaml"
    root = expected_root
    if (
        private.get("path") != str(expected_spec)
        or private.get("sha256") != file_sha256(expected_spec)
        or private.get("source_sha256") != private.get("sha256")
        or private.get("frozen_sha256") != private.get("sha256")
    ):
        raise ValueError("formal private model source belongs to another run")
    from merlin.development.backend.qualification.provenance import private_source_freeze
    from merlin.development.backend.session.inputs.run_record import bundle_manifest_identity

    resolved = private_source_freeze.verify(spec, record, root=root, target=target)
    effective_path = run_dir / "input_bundle_manifest.yaml"
    if effective_path.is_symlink() or not effective_path.is_file():
        raise ValueError("formal private source has no effective input bundle")
    effective = yaml.safe_load(effective_path.read_bytes())
    if not isinstance(effective, dict) or environment.get("bundle_manifest_sha256") != bundle_manifest_identity(
        effective_path, effective
    ):
        raise ValueError("formal private source differs from frozen effective bundle")
    host_inputs = {entry.get("path") for entry in effective.get("host_inputs", []) if isinstance(entry, Mapping)}
    masks = effective.get("private_validation_paths")
    denied = {entry.get("path") for entry in masks or [] if isinstance(entry, Mapping)}
    required = [{"path": original, "kind": "file"} for original, _field in resolved]
    if descriptor is not None:
        from merlin.development.backend.session.inputs.corpus_views import _reconcile_public_software_denial
        from merlin.target.registry.descriptor_loading import load_target_experiment

        selected = load_target_experiment(descriptor)
        if selected.target != target:
            raise ValueError("formal private source declaration names another target")
        required = _reconcile_public_software_denial(
            selected,
            effective,
            required,
            spec=spec,
            source_freeze=record,
            source_root=root,
        )
    if (
        str(root) not in host_inputs
        or str(root) not in denied
        or any(entry["path"] not in denied for entry in required)
        or masks != private.get("masked_paths")
    ):
        raise ValueError("formal private authored sources are not frozen and masked")
    if workspace is None or repo is None:
        raise ValueError("formal private authored sources require the frozen workspace")
    from merlin.development.targetgen.sandbox import grants as bwrap

    snapshot = environment.get("bundle_input_snapshot")
    bwrap.verify_snapshot_binding(workspace, effective, snapshot, repo=repo)
    [frozen_root] = bwrap.snapshot_input_paths(workspace, effective, [root], repo=repo)
    for row in (*record["archives"], *record["sources"]):
        relative = Path(row["frozen_path"]).relative_to(root)
        frozen_member = frozen_root / relative
        if file_sha256(frozen_member) != row.get("sha256", row.get("manifest_sha256")):
            raise ValueError("formal private authored source differs from frozen workspace input")
    return dict(record)


def _formal_model_simulator(target: str) -> dict:
    """Resolve the concrete engine for formal whole-model certification.

    A tier names fidelity, not a binary.  Operator grading already resolves its elaborated-RTL engine
    through the target-routed availability/cost policy; whole-model grading must make the identical
    decision.  A target may supply L3 through an adapter while leaving the manifest's static ``tier_sim``
    map empty, so consulting that map here rejects a perfectly valid arc-routed target before grading.
    """
    cfg = capsule_runner_config._config_for_target(target, None, "i8xi8_i32")
    if FORMAL_REQUIRED_TIER not in cfg.rtl_tiers:
        raise RuntimeError(f"formal tier {FORMAL_REQUIRED_TIER} is not an RTL tier for target {target}")
    selected = dict(capsule_oracle_policy.describe_l3_engine(target))
    if not selected.get("available"):
        raise RuntimeError(
            f"formal tier {FORMAL_REQUIRED_TIER} has no available simulator for target {target}: "
            f"{selected.get('reason') or selected!r}"
        )
    if not selected.get("engine") or selected.get("fidelity") != "elaborated_rtl":
        raise RuntimeError(f"formal simulator selection for {target} did not resolve to elaborated_rtl: {selected!r}")
    selected["selection"] = selected.get("selection") or "target_routed_l3_policy"
    return selected


def _install_formal_model_simulator(target: str) -> tuple[dict, str | None]:
    """Select and install the trusted model-tile engine, returning prior ambient state for audit."""
    selected = _formal_model_simulator(target)
    required = os.environ.get("MERLIN_REQUIRED_RTL_ENGINE", "").strip() or None
    if required is not None and selected.get("engine") != required:
        raise RuntimeError(f"formal model engine {selected.get('engine')!r} differs from required {required!r}")
    inherited = os.environ.get("MERLIN_MESH_SIM")
    os.environ["MERLIN_MESH_SIM"] = selected["engine"]
    return selected, inherited


def phase_completion(
    score: Mapping | None, *, required_tier: str = FORMAL_REQUIRED_TIER, functional_hidden: bool = False
) -> tuple[bool, list[str]]:
    """Fail-closed completion predicate for one official grading phase.

    ``functional_pass`` alone is insufficient: an empty/mis-rooted suite used to look like ``0/0`` and
    the headline can omit which tier supplied the evidence.  A formal phase therefore has to be
    non-vacuous, fully passing, gradeable, complete at the predeclared L3 tier, and RTL-backed for every
    passing capsule.  Malformed or absent fields are refusals, never inferred successes.
    """
    if not isinstance(score, Mapping):
        return False, ["score_missing_or_malformed"]

    reasons: list[str] = []
    from merlin.development.targetgen.grading.boundary import execution_denominator

    n_executable, refusal_failures = execution_denominator(score)
    reasons.extend(refusal_failures)
    n_capsules = score.get("n_capsules")
    n_passed = score.get("n_passed")
    if not isinstance(n_capsules, int) or isinstance(n_capsules, bool) or n_capsules <= 0:
        reasons.append("capsule_set_empty_or_malformed")
    if not isinstance(n_passed, int) or isinstance(n_passed, bool):
        reasons.append("pass_count_missing_or_malformed")
    elif isinstance(n_capsules, int) and not isinstance(n_capsules, bool) and n_passed != n_capsules:
        reasons.append("not_all_capsules_passed")
    if score.get("functional_pass") != 1:
        reasons.append("functional_pass_not_true")
    if score.get("gradeable") is not True:
        reasons.append("numeric_grade_not_gradeable")
    if score.get("integrity_status") != "clean":
        reasons.append("submission_integrity_not_clean")
    if score.get("numeric_all_exact") is not True:
        reasons.append("numeric_exactness_not_complete")
    # Operator capsules emit one kernel and must pass decoded instruction-trace conformance.  Whole
    # models are multi-kernel host/accelerator compositions and have no honest single model-level trace;
    # they instead carry a distinct, fail-closed accelerator-execution check.  Require both applicable
    # scopes, and require their counts to cover the entire admitted denominator -- never stamp a model
    # trace pass for an artifact that does not exist.
    scope = score.get("structural_evidence_scope")
    if not isinstance(scope, Mapping):
        reasons.append("structural_evidence_scope_missing_or_malformed")
    else:
        n_trace = scope.get("n_instruction_trace_capsules")
        n_model = scope.get("n_model_execution_capsules")
        n_host = scope.get("n_provided_host_capsules")
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in (n_trace, n_model, n_host)):
            reasons.append("structural_evidence_scope_counts_malformed")
        else:
            if n_executable is not None and n_trace + n_model + n_host != n_executable:
                reasons.append("structural_evidence_scope_denominator_mismatch")
            if n_trace > 0 and score.get("trace_all_pass") is not True:
                reasons.append("trace_conformance_not_complete")
            if n_model > 0 and score.get("model_execution_all_pass") is not True:
                reasons.append("model_execution_evidence_not_complete")
            if n_host > 0 and score.get("provided_host_execution_all_pass") is not True:
                reasons.append("provided_host_execution_evidence_not_complete")
    if score.get("structural_evidence_all_pass") is not True:
        reasons.append("structural_evidence_not_complete")

    # Cohort selection happens before the scored denominator.  Require its arithmetic and policy in the
    # formal record so a hidden source-pool capsule cannot disappear through an undocumented filter.
    # `frozen_target_capability_operand_dtype` is code-derived after freeze and excludes only a dtype
    # absent from every frozen target capability; it does not expose held-out names to the agent.
    admission = score.get("cohort_admission")
    if not isinstance(admission, Mapping):
        reasons.append("cohort_admission_missing_or_malformed")
    else:
        policy = admission.get("policy")
        if policy not in {
            "all_discovered",
            "frozen_target_capability_operand_dtype",
            "descriptor_capability_and_resource_v1",
            "descriptor_search_cohort_v1",
        }:
            reasons.append("cohort_admission_policy_invalid")
        source_n = admission.get("n_source_capsules")
        admitted_n = admission.get("n_admitted_capsules")
        capability_excluded_n = admission.get("n_capability_excluded")
        resource_excluded_n = admission.get("n_resource_excluded")
        search_excluded_n = admission.get("n_search_excluded", 0)
        if any(
            not isinstance(v, int) or isinstance(v, bool) or v < 0
            for v in (source_n, admitted_n, capability_excluded_n, resource_excluded_n, search_excluded_n)
        ):
            reasons.append("cohort_admission_counts_malformed")
        else:
            if source_n <= 0 or admitted_n <= 0:
                reasons.append("cohort_admission_vacuous")
            if source_n != (admitted_n + capability_excluded_n + resource_excluded_n + search_excluded_n):
                reasons.append("cohort_admission_arithmetic_invalid")
            if isinstance(n_capsules, int) and not isinstance(n_capsules, bool) and admitted_n != n_capsules:
                reasons.append("cohort_admission_denominator_mismatch")
        for field in ("excluded_name_set_sha256", "admitted_name_set_sha256"):
            digest = admission.get(field)
            if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                reasons.append(f"cohort_admission_{field}_invalid")
        if policy == "descriptor_capability_and_resource_v1":
            if admission.get("resource_policy") != "representative_l3_capstones_v1":
                reasons.append("cohort_admission_resource_policy_invalid")
            required_models = admission.get("required_admitted_models")
            if not isinstance(required_models, list) or not required_models:
                reasons.append("cohort_admission_required_models_missing")
        if policy == "descriptor_search_cohort_v1":
            if not isinstance(admission.get("search_policy"), str) or not admission["search_policy"]:
                reasons.append("cohort_admission_search_policy_missing")
            digest = admission.get("included_name_set_sha256")
            if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                reasons.append("cohort_admission_included_name_set_sha256_invalid")

    # The diagnostic score denominator deliberately excludes work that was never measured. Formal
    # completion must not inherit that smaller denominator: otherwise a screen budget, model timeout,
    # deferred capstone, or hidden capsule marked ineligible can disappear while the surviving subset
    # reports all-pass. Require each unmeasured count explicitly and require zero.
    unmeasured_counts = (
        "n_not_graded_ineligible",
        "n_gated_deferred",
        "n_screened_only",
        "n_budget_exhausted",
        "n_incomplete",
        "n_not_gradeable_no_oracle",
    )
    for field in unmeasured_counts:
        value = score.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            reasons.append(f"{field}_missing_or_malformed")
        elif value != 0:
            reasons.append(f"{field}_nonzero")

    tier_reached = score.get("tier_reached")
    tier_count = tier_reached.get(required_tier) if isinstance(tier_reached, Mapping) else None
    if not isinstance(tier_count, int) or isinstance(tier_count, bool):
        reasons.append(f"{required_tier}_count_missing_or_malformed")
    elif n_executable is not None and tier_count != n_executable:
        reasons.append(f"not_all_capsules_reached_{required_tier}")

    if functional_hidden:
        if required_tier != "L2":
            reasons.append("functional_hidden_requires_L2")
        reasons.extend(HCP.evidence_failures(score))
    else:
        evidence = score.get("pass_evidence")
        rtl_backed = evidence.get("rtl_backed") if isinstance(evidence, Mapping) else None
        if not isinstance(rtl_backed, int) or isinstance(rtl_backed, bool):
            reasons.append("rtl_backed_count_missing_or_malformed")
        elif n_executable is not None and rtl_backed != n_executable:
            reasons.append("not_all_capsules_rtl_backed")
    return not reasons, reasons


def _phase_manifest(score: Mapping, *, required_tier: str, functional_hidden: bool = False) -> dict:
    complete, failures = phase_completion(score, required_tier=required_tier, functional_hidden=functional_hidden)
    return {
        "passed": score.get("public_passed")
        or score.get("hidden_passed")
        or f"{score.get('n_passed')}/{score.get('n_capsules')}",
        "functional_pass": score.get("functional_pass"),
        "n_passed": score.get("n_passed"),
        "n_capsules": score.get("n_capsules"),
        "gradeable": score.get("gradeable"),
        "integrity_status": score.get("integrity_status"),
        "numeric_all_exact": score.get("numeric_all_exact"),
        "trace_all_pass": score.get("trace_all_pass"),
        "model_execution_all_pass": score.get("model_execution_all_pass"),
        "structural_evidence_all_pass": score.get("structural_evidence_all_pass"),
        "provided_host_execution_all_pass": score.get("provided_host_execution_all_pass"),
        "structural_evidence_scope": score.get("structural_evidence_scope"),
        "trace_requirement_summary": score.get("trace_requirement_summary"),
        **({"expected_refusal_scope": score["expected_refusal_scope"]} if "expected_refusal_scope" in score else {}),
        "unmeasured_counts": {
            field: score.get(field)
            for field in (
                "n_not_graded_ineligible",
                "n_gated_deferred",
                "n_screened_only",
                "n_budget_exhausted",
                "n_incomplete",
                "n_not_gradeable_no_oracle",
            )
        },
        "highest_tier": score.get("highest_tier"),
        "tier_reached": score.get("tier_reached"),
        "pass_evidence": score.get("pass_evidence"),
        "cohort_admission": score.get("cohort_admission"),
        "formal_complete": complete,
        "completion_failures": failures,
        **(
            {
                "functional_hidden_evidence": score.get("functional_hidden_evidence"),
                "validation_mode": "functional_spike",
                "required_tier": "L2",
                "claim": "functional_corroboration",
                "rtl_certified": False,
            }
            if functional_hidden
            else {}
        ),
    }


def _roots(spec: str) -> list[str]:
    """A comma-separated root list. A target's graded suite spans sibling capsule CATEGORIES, so one
    path cannot name it; the launcher resolves them from the descriptor and passes them through here."""
    return [s for s in (x.strip() for x in str(spec).split(",")) if s]


def _score(
    pkg,
    capsules,
    runs_root,
    labels,
    no_oracle,
    *,
    context: InvocationContext,
    contract=None,
    step_timeout_s=900,
    hidden_capsule_mode="rtl",
):
    # Resolve the TARGET'S OWN oracle ladder from its contract (external_backend->program_oracle,
    # chipyard->spike/verilator, else arc) — never pass None here, which historically fell back to the
    # gemmini spike/verilator MLIR-lowering oracle and mis-graded atlas (torch-mlir run_lowering.py crash).
    # `{}` = honest no-oracle (L0/L1/trace only). sim_via is self-resolved from the contract.
    adapters = {} if no_oracle else capsule_adapters.oracle_adapters(context.target, **readback_kwargs(context))
    functional_hidden = hidden_capsule_mode == "functional_spike"
    if functional_hidden:
        cfg = capsule_runner_config._config_for_target(context.target, None, "i8xi8_i32")
        if (
            labels != {"hidden"}
            or no_oracle
            or "L2" not in adapters
            or cfg.tier_sim.get("L2") != "spike"
            or "L2" in cfg.rtl_tiers
        ):
            raise RuntimeError("functional hidden policy requires the actual non-RTL Spike L2 adapter")
        adapters = {"L2": adapters["L2"]}
    return _merlin_submission_1.grade(
        pkg,
        capsules_root=_roots(capsules),
        runs_root=runs_root,
        model_snapshot_root=Path(runs_root).resolve() / ".private_model_sources",
        labels=labels,
        contract=str(Path(contract).resolve()) if contract else str(context.repo / "src/merlin/resources/contracts"),
        oracle_adapters=adapters,
        timeout=step_timeout_s,
        target=context.target,
        no_oracle=no_oracle,
        # The public set was materialized from the descriptor before the run.  The hidden
        # pool cannot reveal names there, so derive its hardware-capable cohort only here,
        # after freeze, outside the sandbox.  The score seals the source/admitted counts.
        capability_admission=labels == {"hidden"},
        **({"functional_hidden_tier": "L2"} if functional_hidden else {}),
    )


def _cost_phrase(proc: dict) -> str:
    """Render the money field so a notional figure can never be read as a spend."""
    if proc.get("estimated_cost_usd") is not None:
        return f"cost=${proc['estimated_cost_usd']}"
    notional = proc.get("subscription_notional_usd")
    if notional is not None:
        return f"cost=n/a ({proc.get('billing_mode') or 'notional'}: ${notional} notional)"
    return f"cost=n/a ({proc.get('cost_unavailable_reason') or 'no usage metadata'})"


def _verify_formal_readback_policy(run_dir: Path, context: InvocationContext) -> None:
    """Formal grading cannot select a different transport from the admitted run."""
    path = run_dir / "environment.yaml"
    if path.is_symlink():
        raise RuntimeError("formal run environment must not be indirect")
    if not path.is_file():
        verify_readback_record(context, None)
        return
    record = yaml.safe_load(path.read_bytes())
    if not isinstance(record, dict):
        raise RuntimeError("formal run environment is malformed")
    verify_readback_record(context, record.get("readback_policy"))


@selected_child
def main(argv: list[str] | None = None, *, context: InvocationContext | None = None) -> int:
    """Retain original API admission across public, hidden and private grading."""
    from merlin.development.backend.feedback.grading.formal_run import _main

    selection = argparse.ArgumentParser(add_help=False)
    add_context_arguments(selection)
    selection.add_argument("--run-dir")
    selection.add_argument("--workspace", type=Path)
    early, _ = selection.parse_known_args(argv)
    if early.run_dir is None:
        return _main(argv, context=context)
    manifest = Path(early.run_dir).absolute() / "input_bundle_manifest.yaml"
    environment_path = manifest.parent / "environment.yaml"
    if environment_path.is_symlink():
        raise ValueError("formal original run environment is indirect")
    environment = yaml.safe_load(environment_path.read_bytes()) if environment_path.is_file() else {}
    if not isinstance(environment, dict):
        raise ValueError("formal original run environment is malformed")
    if not manifest.exists() and not manifest.is_symlink():
        if environment.get("compiler_tool_profile") is not None:
            raise ValueError("formal original compiler profile archive is missing")
        return _main(argv, context=context)  # Historical unprofiled runs keep their policy.
    if manifest.is_symlink() or manifest.resolve() != manifest:
        raise ValueError("formal compiler profile requires the original ordinary archived manifest")
    bundle = yaml.safe_load(manifest.read_bytes())
    if not isinstance(bundle, dict) or bundle.get("compiler_tool_profile") is None:
        if environment.get("compiler_tool_profile") is not None:
            raise ValueError("formal original compiler profile was removed")
        return _main(argv, context=context)
    from merlin.development.backend.session.inputs.run_record import bundle_manifest_identity

    if environment.get("compiler_tool_profile") != bundle["compiler_tool_profile"] or environment.get(
        "bundle_manifest_sha256"
    ) != bundle_manifest_identity(manifest, bundle):
        raise ValueError("formal compiler profile differs from the original run record")
    if early.workspace is None:
        raise ValueError("formal compiler profile requires its original frozen workspace")
    original_context = context if context is not None else resolve_context(early, selection)
    from merlin.development.backend.feedback.execution.snapshot_evaluator import (
        selected as external_snapshot_selected,
    )
    from merlin.development.backend.feedback.execution.snapshots import SnapshotInputs
    from merlin.development.targetgen.execution.host.common_transforms import CommonTransformService

    owner_type = SnapshotInputs if external_snapshot_selected() else CommonTransformService
    owner = owner_type(workspace=early.workspace.absolute(), bundle_manifest=manifest, repo=original_context.repo)
    from merlin.development.backend.feedback.execution.profile_runtime import original_compiler_execution
    from merlin.target.registry.descriptor_loading import load_target_experiment

    target = load_target_experiment(original_context.descriptor)
    with original_compiler_execution(
        owner,
        target_experiment=target,
        candidate=manifest.parent / "submission",
        work_root=manifest.parent / "_qa_work" / "formal_profile_execution",
    ):
        # Keep the caller's context convention, including installed explicit
        # corpus requirements; profile selection does not enable legacy defaults.
        return _main(argv, context=context)
