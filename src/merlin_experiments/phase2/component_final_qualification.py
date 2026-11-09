"""Re-evaluate original compile/static roles for actual observed descendant bytes.

The qualified baseline selects the immutable roster and build/runtime owners.
Its old evaluation never certifies an edited candidate. This fixed ordinary
controller retains new evidence and refuses unresolved original obligations
before numerical qualification; it grants no proof by declaration or callback.
"""

from merlin_experiments.phase1.component_compile_roles import evaluate_component_compile_roles
from merlin_experiments.phase1.component_qualification import qualify_component_compiler

from .contracts import StageGateError


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
