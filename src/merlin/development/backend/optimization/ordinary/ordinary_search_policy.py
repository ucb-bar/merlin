"""Queued development feedback on the normal Phase 1 functional fork.

This adapter retains the corpus workflow's scientific and receipt owners. It
adds immutable revision dispatch, not component-runtime qualification or final
acceptance. The selected ordinary executor still performs its original fresh
Spike/numerical and certified simulator admission for every measured member.
"""

from __future__ import annotations

import copy
import json
import time
from contextlib import nullcontext
from dataclasses import asdict, is_dataclass, replace
from functools import partial
from pathlib import Path

from merlin.analysis.optimization.candidates import _census_process as CENSUS_PROCESS
from merlin.analysis.optimization.candidates import candidate_diagnostics as CANDIDATE_DIAGNOSTICS
from merlin.analysis.optimization.candidates import candidate_source as CANDIDATE_SOURCE
from merlin.analysis.optimization.candidates import cost_graph as CANDIDATE_FEATURES
from merlin.analysis.optimization.candidates import cost_graph as CANDIDATE_GRAPH
from merlin.analysis.optimization.candidates import dynamic_census as DYNAMIC_CENSUS
from merlin.analysis.optimization.candidates import dynamic_census as DYNAMIC_STREAM
from merlin.analysis.perf.measurement.execution_policy import ITERATION_MAX_SECONDS
from merlin.common.process import ownership as CORE_CHILDREN
from merlin.development.backend.brokers import compiler_call_channel as PROFILE_CHANNEL
from merlin.development.backend.feedback.candidates import observations as CANDIDATE_CENSUS
from merlin.development.backend.feedback.candidates import observations as CANDIDATE_OBSERVATIONS
from merlin.development.backend.feedback.execution import profile_runtime as PROFILE_RUNTIME
from merlin.development.backend.optimization.author_interface import functional_inputs as FI
from merlin.development.backend.optimization.author_interface import queued_search_actions as QSA
from merlin.development.backend.optimization.author_interface.broker_policy import BrokerServices
from merlin.development.backend.optimization.authoring.edit_authority import FrozenEditAuthority
from merlin.development.backend.optimization.campaign import compiler_libraries as LIBRARIES
from merlin.development.backend.optimization.campaign import compiler_library_state as LIBRARY_STATE
from merlin.development.backend.optimization.campaign import functional_boundary as PC
from merlin.development.backend.optimization.campaign import package_sandbox as _merlin_package_sandbox_1
from merlin.development.backend.optimization.campaign import pass_preparation as PREPARATION
from merlin.development.backend.optimization.contracts import (
    StageGateError,
    canonical_json,
    document_sha256,
    exact_tree_record,
    mapping_file,
    sha256_file,
)
from merlin.development.backend.optimization.diagnostics import archive_count_observation as ARCHIVE_COUNTS
from merlin.development.backend.optimization.diagnostics import archive_screen as ARCHIVE_NATIVE
from merlin.development.backend.optimization.diagnostics import census_launch as CENSUS_LAUNCH
from merlin.development.backend.optimization.diagnostics import spike_runtime as ARCHIVE_RUNTIME
from merlin.development.backend.optimization.measurement import development_feedback as DF
from merlin.development.backend.optimization.measurement import gsim_gate as GATE
from merlin.development.backend.optimization.measurement import identities as MS
from merlin.development.backend.optimization.measurement import paired_evidence as PUBLIC_MEASUREMENT
from merlin.development.backend.optimization.measurement import paired_plan as PME
from merlin.development.backend.optimization.ordinary import capsule_feedback as CAPSULE_FEEDBACK
from merlin.development.backend.optimization.ordinary import capsule_ordinary_scope as CAPSULE_SCOPE
from merlin.development.backend.optimization.ordinary import ordinary_archive_screen as ARCHIVE_SCREEN
from merlin.development.backend.optimization.ordinary import ordinary_build_runtime as BUILD
from merlin.development.backend.optimization.ordinary import ordinary_calibration as CAL
from merlin.development.backend.optimization.ordinary import ordinary_checked_source as CHECKED_SOURCE
from merlin.development.backend.optimization.ordinary import ordinary_compile_only as COMPILE_ONLY
from merlin.development.backend.optimization.ordinary import ordinary_fast_feedback as FAST
from merlin.development.backend.optimization.ordinary import ordinary_impact_selection as IMPACT
from merlin.development.backend.optimization.ordinary import (
    ordinary_measurement_selection as MEMBER_SELECTION,
)
from merlin.development.backend.optimization.ordinary import ordinary_queue_calibration as QUEUE_CAL
from merlin.development.backend.optimization.ordinary import ordinary_screen_features as SCREEN_FEATURES
from merlin.development.backend.optimization.search import corpus as CORPUS
from merlin.development.backend.optimization.search import corpus_feedback as CF
from merlin.development.backend.optimization.search import provisional_search as PROV
from merlin.development.backend.optimization.search.retained_revisions import (
    RetainedRevision,
    executing_revisions,
)
from merlin.development.targetgen.authoring import library_admission as COMPILER_LIBRARY
from merlin.development.targetgen.execution.host import public_program_timing as PUBLIC_PROGRAM
from merlin.development.targetgen.execution.native import model_policy as MODEL_POLICY
from merlin.development.targetgen.execution.native import native_model_execution as NATIVE_MODEL
from merlin.development.targetgen.execution.runner import artifact_retesting as AR
from merlin.development.targetgen.execution.runner import compile as NATIVE_BUILD
from merlin.runtime.harness import call_timing as COMPLETE_TIMING
from merlin.runtime.harness import direct_kernel as DIRECT_HARNESS
from merlin.runtime.harness import pointer_storage as LOGICAL_HARNESS
from merlin.target.registry.experiment_descriptor import TargetExperiment

# ---- ordinary_search_policy --------------------------------------------------------------------


def prepare_ordinary_search_policy(
    *,
    functional_runs_root,
    functional_run_id,
    functional_submission_sha256,
    corpus,
    baseline,
    candidate,
    target_experiment,
    receipt_path,
    work_root,
    edit_authority,
    contract_root,
    certificate_path,
    certificate_sha256,
    rtl_facts_path,
    feedback_round=0,
    tuning_call_budget=None,
    waive_functional_gate=(),
    services=BrokerServices(),
    e2e_sentinel=None,
    whole_model_inputs=None,
    native_runtime_view=None,
    artifact_retest_index=None,
    build_runtime_view=None,
    build_product_index=None,
    compiler_library_selection=None,
    compiler_pass_selection=None,
    compiler_pass_preparation=None,
    checked_source_diagnostic=None,
    capsule_phase1_owner=None,
    required_cohort_declaration=None,
):
    """Admit the same Phase 1 functional fork and ordinary corpus feedback route.

    No executable callback, backend name, imported provider or saved verdict can
    substitute for either original admission owner. An explicit live
    checked_source_diagnostic only selects L2 source witnesses in selected
    compile/Spike feedback. It does not waive this formal admission, infer public
    placements, or grant timing or promotion authority.
    """
    from merlin.development.backend.optimization.ordinary.search_policy import (
        _ISSUER,
        OrdinarySearchPolicy,
        _phase1_target,
        _real,
    )

    if (
        type(corpus) is not CORPUS.FrozenPerformanceCorpus
        or type(target_experiment) is not TargetExperiment
        or type(edit_authority) is not FrozenEditAuthority
        or not edit_authority.configured
        or type(feedback_round) is not int
        or feedback_round < 0
    ):
        raise StageGateError("ordinary search requires normal frozen corpus, target and edit authority")
    if waive_functional_gate:
        raise StageGateError("ordinary search requires an unwaived Phase 1 qualification")
    candidate = _real(candidate, label="candidate")
    baseline = _real(baseline, label="functional baseline")
    contract_root = _real(contract_root, label="contract resources")
    edit_authority.check_integrity()
    initial_digest = functional_submission_sha256
    if compiler_pass_preparation is not None:
        from merlin.development.backend.optimization.campaign.pass_preparation import verify_pair

        verify_pair(compiler_pass_preparation, compiler_pass_selection, functional_sha256=functional_submission_sha256)
        library = compiler_library_selection or COMPILER_LIBRARY.active_compiler_library()
        if library is None or compiler_pass_selection.binding.library != library.contract.sha256:
            raise StageGateError("prepared ordinary search lost its same-release reviewed compiler library")
        initial_digest = compiler_pass_preparation["candidate_seed"]["sha256"]
        if baseline != Path(compiler_pass_preparation["baseline"]["path"]):
            raise StageGateError("ordinary search baseline differs from original prepared treatment")
    elif compiler_pass_selection is not None:
        raise StageGateError("production compiler passes require original pre-freeze preparation")
    if (
        edit_authority.initial_source != candidate
        or edit_authority.binding["initial_candidate_sha256"] != initial_digest
    ):
        raise StageGateError("ordinary search edit authority must start from the selected Phase 1 functional compiler")
    capsule_scope = None
    if capsule_phase1_owner is None:
        functional = FI.inspect_stage_functional_run(
            functional_runs_root,
            functional_run_id,
            functional_submission_sha256,
            waive=frozenset(waive_functional_gate),
        )
    else:
        functional, capsule_scope = CAPSULE_SCOPE.select(
            capsule_phase1_owner,
            functional_runs_root=functional_runs_root,
            functional_run_id=functional_run_id,
            functional_submission_sha256=functional_submission_sha256,
            target_experiment=target_experiment,
            required_cohort_declaration=required_cohort_declaration,
        )
    LIBRARY_STATE.require_phase1_selection(functional, compiler_library_selection)
    _phase1_target(mapping_file(functional.run_dir / "environment.yaml", yaml_file=True), target_experiment)
    if capsule_scope is None:
        FI.require_formal_qualification(functional, target_experiment)
    original = baseline if compiler_pass_preparation is None else Path(compiler_pass_preparation["original"]["path"])
    if not _merlin_package_sandbox_1.check_fork(_merlin_package_sandbox_1.functional_fork(functional), original).ok:
        raise StageGateError("ordinary search baseline is not the original admitted functional fork")
    FI._require_read_only_tree(baseline, label="ordinary search functional baseline")
    CORPUS.verify_frozen_performance_corpus(corpus)
    feedback = DF.prepare_development_feedback(
        contract_root=contract_root,
        certificate_path=certificate_path,
        certificate_sha256=certificate_sha256,
        rtl_facts_path=rtl_facts_path,
        corpus=corpus,
        baseline=baseline,
        baseline_sha256=(
            functional.digest if compiler_pass_preparation is None else compiler_pass_preparation["baseline"]["sha256"]
        ),
        target_experiment=target_experiment,
        work_root=Path(work_root),
        tuning_call_budget=tuning_call_budget,
        functional_run_dir=functional.run_dir,
        whole_model_inputs=whole_model_inputs,
        capsule_development_scope=capsule_scope,
    )
    policy = OrdinarySearchPolicy(
        functional=functional,
        corpus=corpus,
        feedback=feedback,
        edit_authority=edit_authority,
        contract_root=contract_root,
        candidate=candidate,
        receipt_path=Path(receipt_path),
        feedback_round=feedback_round,
        services=services,
        e2e_sentinel=e2e_sentinel,
        compiler_library_selection=(
            compiler_library_selection
            if compiler_library_selection is not None
            else COMPILER_LIBRARY.active_compiler_library()
        ),
        compiler_pass_selection=compiler_pass_selection,
        compiler_pass_preparation=compiler_pass_preparation,
        checked_source_diagnostic=checked_source_diagnostic,
        _issuer=_ISSUER,
    )
    if capsule_scope is not None:
        CAPSULE_SCOPE.attach(policy, capsule_scope)
        policy._selection = canonical_json(policy._binding())
        policy.validate_search_inputs()
    if native_runtime_view is not None:
        if capsule_scope is not None:
            raise StageGateError("capsule development cannot install a GSIM runtime")
        policy.install_native_runtime(native_runtime_view, artifact_retest_index)
    elif artifact_retest_index is not None:
        raise StageGateError("normal artifact index requires its exact live native runtime owner")
    if build_runtime_view is not None:
        policy.install_build_runtime(build_runtime_view, build_product_index)
    elif build_product_index is not None:
        raise StageGateError("normal build products require their exact live original toolchain view")
    return policy
