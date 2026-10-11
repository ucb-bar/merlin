"""Unpaid, fresh-run qualification of an explicitly selected compiler submission."""

from __future__ import annotations

from pathlib import Path

from merlin.common.jsonio import canonical_sha256
from merlin.compile.model_execution_inputs import strict_tree_sha256
from merlin.targetgen.sandbox import bwrap
from merlin_experiments.corpus.preparation import private_json

from . import formal_invocation, run_inputs
from .feedback import certification, private_full_model_execution, private_full_models
from .session import PreparedRun


def execute(prepared: PreparedRun) -> int:
    """Grade through the ordinary formal engine without constructing an author.

    This uses the run-owned reviewed corpus, private source freeze and selected
    candidate copy created by ``session.prepare``. It is not evidence that an
    agent converged or that another run's grader/source identity was upgraded.
    """
    options = prepared.request.options
    if not options.qualify_submission or prepared.resuming:
        raise RuntimeError("unpaid qualification requires a fresh selected submission")
    if not isinstance(prepared.environment.get("corpus_review"), dict):
        raise RuntimeError("unpaid qualification has no reviewed corpus handoff")
    private = prepared.environment.get("private_full_model_spec")
    if not isinstance(private, dict) or private.get("source_freeze") is None:
        raise RuntimeError("unpaid qualification has no run-owned private source freeze")
    if prepared.private_full_model_spec is None or prepared.public_root is None:
        raise RuntimeError("unpaid qualification lacks selected grading roots")
    seed = prepared.environment.get("seed_submission")
    selected_source = Path(options.qualify_submission).expanduser().resolve(strict=True)
    if not isinstance(seed, dict) or seed.get("source") != str(selected_source):
        raise RuntimeError("unpaid qualification seed differs from selected submission")
    prepared.verify_inputs()
    run_inputs.verify_selected_seed_source(prepared.workspace, seed)
    submission = run_inputs.snapshot_seed_for_grade(prepared.workspace, prepared.run_dir, seed)
    initial_candidate_sha256 = strict_tree_sha256(submission)["sha256"]
    graded_source_sha256 = run_inputs.graded_source_hash(submission)
    facts = bwrap.frozen_selected_rtl_facts(prepared.workspace, prepared.bundle, repo=prepared.request.context.repo)
    returncode = formal_invocation.run(prepared, public_capsules=prepared.public_root, selected_rtl_facts=facts)
    run_inputs.verify_seed_submission(prepared.workspace, prepared.run_dir, seed)
    run_inputs.verify_selected_seed_source(prepared.workspace, seed)
    if run_inputs.graded_source_hash(submission) != graded_source_sha256:
        raise RuntimeError("unpaid qualification graded source changed during formal grading")
    target = prepared.request.context.target
    descriptor = prepared.request.context.descriptor
    grade = certification._official_grade_result(
        returncode,
        prepared.run_dir,
        required_models=private_full_models.requirements_for(descriptor),
        required_programs=private_full_models.program_requirements_for(descriptor),
        execution_gate=private_full_model_execution.gate_for(
            descriptor, required_programs=private_full_models.program_requirements_for(descriptor)
        ),
    )
    private_json(
        prepared.run_dir / "submission_qualification.json",
        {
            "schema": "merlin.phase1.submission_qualification.v1",
            "qualification_only": True,
            "authoring_converged": False,
            "target": target,
            "selected_source_tree_sha256": seed.get("selected_source_tree_sha256"),
            "initial_candidate_sha256": initial_candidate_sha256,
            "graded_source_sha256": graded_source_sha256,
            "candidate_sha256": strict_tree_sha256(submission)["sha256"],
            "implementation_sources_sha256": canonical_sha256(prepared.environment["implementation_sources"]),
            "corpus_review": prepared.environment["corpus_review"],
            "official_grade": grade,
            "scope": "new-run submission grading only; no authoring convergence or legacy-run grade transfer",
        },
    )
    return 0 if grade["complete"] else 1
