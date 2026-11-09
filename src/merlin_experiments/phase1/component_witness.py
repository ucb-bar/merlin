"""Typed source-to-execution witnesses from the independently qualified verifier.

The independently qualified verifier checks actual import/build/execution records. The
generic owner checks its exact source/candidate/member authority and all stage
joins. Candidate-written pass flags and summaries are never witness authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from merlin.common.digest import is_sha256
from merlin_experiments.phase2 import contracts as C

_PARENTS = {
    "source": (),
    "fx": ("source",),
    "imported_ir": ("fx",),
    "prepared_ir": ("imported_ir",),
    "partition": ("prepared_ir",),
    "emitted_host": ("partition",),
    "emitted_device": ("partition",),
    "object": ("emitted_host", "emitted_device"),
    "elf": ("object",),
    "execution": ("elf",),
    "outputs": ("execution",),
    "effects": ("execution",),
}

# These are evaluated execution obligations, separate from the source graph's
# declared effects. A finite numerical screen alone cannot discharge them.
REQUIRED_EXECUTION_EFFECTS = (
    "layout",
    "tails",
    "alias",
    "ownership",
    "lifetime",
    "epoch",
    "effects",
    "fallback",
    "repeated_invocation",
    "host_device_link",
    "synchronization",
)


@dataclass(frozen=True)
class ComponentStageFile:
    stage: str
    path: Path
    sha256: str
    parent_sha256: tuple[tuple[str, str], ...]
    facets: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComponentStageWitness:
    """Produced only after the independent verifier reopens its execution authorities.

    Each stage pin names the actual product; parents are exact invocation inputs.
    ``effects`` binds the verifier's evaluated layouts, tails, ownership, aliases,
    lifetimes, mutation/epochs, repeated invocation and synchronization evidence.
    """

    source_program_sha256: str
    compiler_sha256: str
    member_sha256: str
    target_descriptor_sha256: str
    frontend: str
    stages: tuple[ComponentStageFile, ...]
    output_roster: tuple[str, ...]
    effect_roster: tuple[str, ...]
    reference_authority: str


def verify_component_stage_witness(
    witness: ComponentStageWitness,
    *,
    member: dict,
    candidate_sha256: str,
    target_descriptor_sha256: str,
    frontend: str,
    capsule_root: Path,
    evidence_root: Path,
    required_effects: tuple[str, ...],
) -> dict:
    """Join actual products to exact independent inputs and complete outputs."""
    if (
        type(witness) is not ComponentStageWitness
        or witness.frontend != frontend
        or frontend not in {"mlir", "pytorch"}
    ):
        raise C.StageGateError("component execution requires a typed source-to-executable witness")
    member_identity = member.get("sha256", member.get("capsule_fingerprint"))
    if (
        witness.source_program_sha256 != member["program_sha256"]
        or witness.member_sha256 != member_identity
        or witness.compiler_sha256 != candidate_sha256
        or witness.target_descriptor_sha256 != target_descriptor_sha256
        or witness.reference_authority != "original_independent_golden"
    ):
        raise C.StageGateError("component witness authority differs from the admitted source/compiler/oracle")
    if not isinstance(witness.stages, tuple) or any(type(row) is not ComponentStageFile for row in witness.stages):
        raise C.StageGateError("component witness stage membership must be immutable")
    parents_graph = dict(_PARENTS)
    if any(row.stage == "combined_emission" for row in witness.stages):
        # Some ordinary compilers convert, prepare, partition and emit in one
        # invoked pipeline. Retain its single actual product and applicability
        # witness rather than inventing unobserved intermediate products.
        facets = ("prepared_ir", "partition", "emitted_host", "emitted_device")
        combined = [row for row in witness.stages if row.stage == "combined_emission"]
        if len(combined) != 1 or combined[0].facets != facets:
            raise C.StageGateError("combined emission omits evaluated stage applicability")
        for name in facets:
            parents_graph.pop(name)
        parents_graph["combined_emission"] = ("imported_ir",) if frontend == "pytorch" else ("source",)
        parents_graph["object"] = ("combined_emission",)
    required = set(parents_graph) if frontend == "pytorch" else set(parents_graph) - {"fx", "imported_ir"}
    index = {row.stage: row for row in witness.stages}
    if len(index) != len(witness.stages) or set(index) != required:
        raise C.StageGateError("component execution witness has missing or duplicate stages")
    for name, row in index.items():
        if not is_sha256(row.sha256) or C.sha256_file(row.path) != row.sha256:
            raise C.StageGateError("component execution stage product changed")
        owner = capsule_root if name == "source" else evidence_root
        if not row.path.is_absolute() or row.path.resolve() != row.path or not row.path.is_relative_to(owner.resolve()):
            raise C.StageGateError("component execution stage escapes its private input/evidence owner")
        if name != "combined_emission" and row.facets:
            raise C.StageGateError("only an observed combined emission can join stage facets")
        parents = parents_graph[name] if name != "prepared_ir" or frontend == "pytorch" else ("source",)
        expected = tuple((parent, index[parent].sha256) for parent in parents)
        if row.parent_sha256 != expected:
            raise C.StageGateError("component stage inputs do not join the actual prior products")
    if index["source"].sha256 != member["program_sha256"]:
        raise C.StageGateError("component source product differs from the independent admitted program")
    if tuple(sorted(witness.output_roster)) != tuple(sorted(member["output_roster"])) or len(
        set(witness.output_roster)
    ) != len(witness.output_roster):
        raise C.StageGateError("component execution witness omits a complete logical output")
    if not isinstance(witness.effect_roster, tuple) or not set(required_effects).issubset(witness.effect_roster):
        raise C.StageGateError("component execution witness omits declared effects")
    return {
        "frontend": frontend,
        "source_program_sha256": witness.source_program_sha256,
        "compiler_sha256": witness.compiler_sha256,
        "member_sha256": witness.member_sha256,
        "target_descriptor_sha256": witness.target_descriptor_sha256,
        "stages": [
            {
                "stage": row.stage,
                "path": str(row.path),
                "sha256": row.sha256,
                "parent_sha256": [list(parent) for parent in row.parent_sha256],
                "facets": list(row.facets),
            }
            for row in witness.stages
        ],
        "output_roster": list(witness.output_roster),
        "effect_roster": list(witness.effect_roster),
        "reference_authority": witness.reference_authority,
        "scope": "selected private verifier's evaluated actual import/build/execution joins",
    }
