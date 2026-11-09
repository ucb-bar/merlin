"""Join original source-only selection and evaluated transport to fresh authoring.

No declaration, saved JSON or linked artifact can discharge an independently
required static obligation. This owner keeps these checks outside author grants.
"""

from pathlib import Path

from merlin_experiments.phase0.component_compile_sources import IndependentCompileOnlyRoster
from merlin_experiments.phase2 import contracts as C


def verify_compile_roster(roster, *, hardware, software, descriptor):
    if type(roster) is not IndependentCompileOnlyRoster:
        raise C.StageGateError("fresh Phase 1 requires a live independent source-only roster")
    if (
        roster.hardware is not hardware
        or roster.software is not software
        or roster.target_descriptor != Path(descriptor).resolve()
    ):
        raise C.StageGateError("source-only roster differs from the fresh hardware/software/target selection")
    roster.verify()
    if {member.cohort for member in roster.members} != {"functional_guard", "withheld_transfer"}:
        raise C.StageGateError("fresh source-only qualification requires original guards and withheld transfer members")
    return roster.sha256


def qualification_compile_roles(evaluation, *, origin, lineage, candidate, contract_root):
    from .component_compile_roles import ComponentCompileRoleEvaluation

    if type(evaluation) is not ComponentCompileRoleEvaluation:
        raise C.StageGateError("component qualification requires evaluated original source-only roles")
    if (
        evaluation.compiler_origin is not origin
        or evaluation.compiler_lineage is not lineage
        or evaluation.roster is not origin.inputs.compile_roster
        or evaluation.contract_root != Path(contract_root).resolve()
    ):
        raise C.StageGateError("source-only evaluation differs from the current compiler's original required selection")
    document = evaluation.verify(candidate=Path(candidate))
    binding = {
        "source_roster_sha256": evaluation.roster.sha256,
        "evaluation_sha256": evaluation.receipt_sha256,
        "receipt": str(evaluation.receipt),
        "compilation_denominator": document["compilation_denominator"],
        "static_denominator": document["static_denominator"],
    }
    failures = ["required source-only role unresolved: " + name for name in document["unresolved"]]
    return binding, failures
