"""Typed source-to-execution witnesses from the independently qualified verifier.

The independently qualified verifier checks actual import/build/execution records. The
generic owner checks its exact source/candidate/member authority and all stage
joins. Candidate-written pass flags and summaries are never witness authority.
"""

from __future__ import annotations

import importlib
import json
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path

from merlin.common.digest import is_sha256
from merlin.common.process import invocation_records as invocation_record
from merlin.development.backend.optimization import contracts as C
from merlin.development.backend.qualification.components.origin import (
    FreshCompilerOrigin,
    _authority_identity,
    _plain,
    _readonly,
    _tree,
)

# ---- component_witness -------------------------------------------------------------------------

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


# ---- component_source_applicability ------------------------------------------------------------
# Derive limited source applicability from actual registered tensor IR.
#
# This analysis says what the mathematical source can observe. It never proves
# physical buffer ownership, placement, links, synchronization or repeated execution.
# The selected target verifier must still evaluate those complete runtime obligations.

_SUBJECTS = (
    "static_input_domain",
    "observable_input_mutation",
    "observable_input_address_alias",
    "cross_invocation_source_state",
    "source_synchronization",
)


def _plain(path: Path) -> None:
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or not path.is_file()
        or path.resolve() != path
        or any(part.is_symlink() for part in (path, *path.parents))
    ):
        raise C.StageGateError("component source analysis requires an exact direct source file")


def _analysis_sources() -> tuple[tuple[Path, str], ...]:
    """Pin the complete parser membership and the concrete analysis owners."""
    paths = {Path(__file__)}
    for name in ("merlin.compiler.dialects.xdsl_context", "merlin.compiler.dialects.xdsl_context", "xdsl"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue  # Missing parsing authority is reported UNKNOWN below.
        owner = Path(module.__file__)
        paths.add(owner)
        if name == "xdsl":
            paths.update(owner.parent.rglob("*.py"))
    for path in paths:
        _plain(path)
    return tuple((path, C.sha256_file(path)) for path in sorted(paths))


@dataclass(frozen=True)
class ComponentSourceApplicability:
    source: Path
    source_program_sha256: str
    frontend: str
    analysis_sources: tuple[tuple[Path, str], ...]
    analysis_json: str

    def verify(self) -> None:
        """Rederive source facts; a typed object or caller status is insufficient."""
        observed = evaluate_component_source_applicability(
            source=self.source,
            source_program_sha256=self.source_program_sha256,
            frontend=self.frontend,
        )
        if observed.analysis_sources != self.analysis_sources or observed.analysis_json != self.analysis_json:
            raise C.StageGateError("component source applicability or parser authority changed")

    def record(self) -> dict:
        return {
            "schema": "merlin.component_source_applicability.v1",
            "source": str(self.source),
            "source_program_sha256": self.source_program_sha256,
            "frontend": self.frontend,
            "analysis_sources": [[str(path), digest] for path, digest in self.analysis_sources],
            **json.loads(self.analysis_json),
        }


def _inspect_source(text: str) -> dict:
    from xdsl.dialects.arith import Arith
    from xdsl.dialects.builtin import AnyFloat, IndexType, IntegerType, ModuleOp, NoneAttr, TensorType
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.dialects.linalg import Linalg
    from xdsl.dialects.linalg.abstract_ops import LinalgStructuredOperation
    from xdsl.dialects.linalg.ops import YieldOp
    from xdsl.dialects.math import Math
    from xdsl.dialects.tensor import Tensor
    from xdsl.parser import Parser
    from xdsl.traits import get_effects

    from merlin.compiler.dialects.xdsl_context import make_context

    module = Parser(make_context(Arith, Tensor, Linalg, Math), text).parse_module()
    module.verify()
    functions = tuple(module.body.block.ops)
    if len(functions) != 1 or type(functions[0]) is not FuncOp:
        raise ValueError("source must contain exactly one defined mathematical function")
    function = functions[0]
    if len(function.body.blocks) != 1 or not function.body.block.args:
        raise ValueError("source requires one concrete function block with explicit tensor inputs")

    def static_tensor(value_type):
        # A rank-zero tensor has one logical element and no dimensions. Its
        # empty shape is static; it remains a tensor, not a by-value scalar ABI.
        return (
            isinstance(value_type, TensorType)
            and isinstance(value_type.encoding, NoneAttr)
            and all(dim >= 0 for dim in value_type.get_shape())
        )

    if (
        not all(static_tensor(arg.type) for arg in function.body.block.args)
        or not function.function_type.outputs.data
        or not all(static_tensor(value_type) for value_type in function.function_type.outputs.data)
    ):
        raise ValueError("source function inputs/outputs are not explicit static unencoded tensors")

    def mathematical_type(value_type):
        if isinstance(value_type, TensorType):
            return static_tensor(value_type) and isinstance(
                value_type.get_element_type(), (IntegerType, AnyFloat, IndexType)
            )
        return isinstance(value_type, (IntegerType, AnyFloat, IndexType))

    operations, unresolved = [], []
    for ordinal, op in enumerate(module.walk()):
        types = [value.type for value in (*op.operands, *op.results)]
        types += [arg.type for region in op.regions for block in region.blocks for arg in block.args]
        if not all(mathematical_type(value_type) for value_type in types):
            unresolved.append(f"operation {ordinal} {op.name} has non-mathematical or dynamic storage types")
        if type(op) in {ModuleOp, FuncOp, ReturnOp, YieldOp}:
            effect = "structural_container_or_value_publication"
        elif isinstance(op, LinalgStructuredOperation):
            # The registered structured operation contract states that memref
            # outs mutate buffers, while tensor outs return updated SSA results.
            if (
                len(op.outputs) != len(op.res)
                or not op.outputs
                or any(not static_tensor(value.type) for value in op.outputs)
                or any(lhs.type != rhs.type for lhs, rhs in zip(op.outputs, op.res, strict=True))
            ):
                unresolved.append(f"operation {ordinal} {op.name} lacks tensor-only destination semantics")
            effect = "structured_tensor_value_update"
        else:
            effects = get_effects(op)
            effect = "unknown" if effects is None else "present" if effects else "none"
            if effects is None or effects:
                unresolved.append(f"operation {ordinal} {op.name} has unknown or observable memory effects")
        operations.append(
            {
                "ordinal": ordinal,
                "operation": op.name,
                "types": [str(value_type) for value_type in types],
                "source_effect": effect,
            }
        )
    return {
        "operations": operations,
        "unresolved": unresolved,
        "inputs": [str(arg.type) for arg in function.body.block.args],
        "outputs": [str(value_type) for value_type in function.function_type.outputs.data],
    }


def evaluate_component_source_applicability(
    *,
    source: Path,
    source_program_sha256: str,
    frontend: str,
) -> ComponentSourceApplicability:
    """Observe exact source semantics; unavailable/unknown authority stays UNKNOWN."""
    _plain(source)
    if C.sha256_file(source) != source_program_sha256 or frontend not in {"mlir", "pytorch"}:
        raise C.StageGateError("component source applicability differs from the admitted source/frontend")
    before = _analysis_sources()
    observation = {"operations": [], "unresolved": [], "inputs": [], "outputs": []}
    if frontend != "mlir":
        observation["unresolved"].append("Python source requires independent actual FX capture/import correspondence")
    else:
        try:
            observation = _inspect_source(source.read_text())
        except Exception as exc:  # noqa: BLE001 - parsing/verification never substitutes for missing source facts
            observation["unresolved"].append(type(exc).__name__ + ": " + str(exc))
    pure = not observation["unresolved"]
    facts = {}
    for subject in _SUBJECTS:
        if pure:
            status = "PASS" if subject == "static_input_domain" else "N_A"
            reason = (
                "registered verified function has static unencoded tensor inputs/outputs"
                if subject == "static_input_domain"
                else "closed tensor SSA has no observable addresses, input writes, persistent state "
                "or external synchronization"
            )
        else:
            status, reason = "UNKNOWN", "source applicability authority is incomplete"
        facts[subject] = {"status": status, "reason": reason, "scope": "mathematical source only"}
    observation.update(
        {
            "facts": facts,
            "runtime_effects": {effect: "UNKNOWN" for effect in REQUIRED_EXECUTION_EFFECTS},
            "numerical_finiteness": "UNKNOWN",
            "scope": (
                "source semantics only; no physical alias/lifetime/epoch, partition, execution or numerical credit"
            ),
        }
    )
    after = _analysis_sources()
    if before != after or C.sha256_file(source) != source_program_sha256:
        raise C.StageGateError("component source or parsing authority changed during applicability analysis")
    return ComponentSourceApplicability(
        source, source_program_sha256, frontend, after, json.dumps(observation, sort_keys=True, allow_nan=False)
    )


# ---- component_lineage -------------------------------------------------------------------------
# Descendant compiler authority observed around actual Phase 2 author transport.
#
# The baseline is the issued fresh Phase 1 compiler. Cumulative edits are checked
# against its frozen edit authority, and actual author/broker observations bind the
# final membership. This proves neither correctness nor performance.

_ISSUED: dict[object, tuple] = {}


@dataclass(frozen=True)
class ComponentCompilerLineage:
    origin: FreshCompilerOrigin
    launch: object
    authored_inputs: object
    candidate: Path
    candidate_sha256: str
    receipt: Path
    receipt_sha256: str
    transcript: Path
    transcript_sha256: str
    broker_receipt: Path
    broker_receipt_sha256: str
    author_invocation: Path
    author_invocation_sha256: str
    _issuer: object = field(repr=False, compare=False)

    def verify(self, *, candidate: Path | None = None) -> dict:
        from merlin.development.backend.optimization.components.launch import QualifiedComponentLaunch

        if (
            type(self._issuer) is not object
            or self._issuer not in _ISSUED
            or type(self.origin) is not FreshCompilerOrigin
        ):
            raise C.StageGateError("compiler descendant has no observed Phase 2 authoring authority")
        if _authority_identity(self) != _ISSUED[self._issuer]:
            raise C.StageGateError("compiler descendant authority fields changed after issuance")
        if type(self.launch) is not QualifiedComponentLaunch:
            raise C.StageGateError("compiler descendant lacks an independently qualified launch")
        self.launch.verify()
        self.authored_inputs.verify()
        if self.launch.inputs.qualification.compiler_origin is not self.origin:
            raise C.StageGateError("compiler descendant differs from its fresh Phase 1 baseline authority")
        self.origin.verify(candidate=self.launch.inputs.edit_authority.seed)
        self.launch.inputs.edit_authority.validate_candidate(self.candidate)
        for path, digest in (
            (self.receipt, self.receipt_sha256),
            (self.transcript, self.transcript_sha256),
            (self.broker_receipt, self.broker_receipt_sha256),
            (self.author_invocation, self.author_invocation_sha256),
        ):
            _plain(path)
            if C.sha256_file(path) != digest:
                raise C.StageGateError("compiler descendant authoring evidence changed")
        invocation_record.verify(self.author_invocation)
        for selected in (self.candidate, self.candidate if candidate is None else candidate):
            if _tree(selected)["sha256"] != self.candidate_sha256:
                raise C.StageGateError("compiler bytes differ from the observed Phase 2 descendant")
        return C.mapping_file(self.receipt)


def run_component_origin_round(
    launch, authored_inputs, *, model: str, effort: str, wall_budget_seconds: int, prompt: str
):
    """Run the ordinary provider and bind its actual audited descendant output.

    Invoke inside the existing component broker's serving lifetime. The caller
    continues with unchanged numerical and source/effect qualification gates.
    No externally supplied transcript, receipt or candidate can mint lineage.
    """
    from merlin.development.backend.optimization.authoring.transcript_audit import audit_codex_transcript
    from merlin.development.backend.optimization.components.launch import (
        ComponentLaunchInputs,
        QualifiedComponentLaunch,
        _control_command,
        _read_paths,
        _runtime_binds,
    )
    from merlin.development.backend.providers import codex_launch_codex_agent as CA

    if type(launch) is not QualifiedComponentLaunch or type(authored_inputs) is not ComponentLaunchInputs:
        raise C.StageGateError("compiler descendant authoring needs actual qualified component inputs")
    launch.verify()
    authored_inputs.verify()
    if replace(authored_inputs, policy=launch.inputs.policy) != launch.inputs:
        raise C.StageGateError("compiler descendant authoring changed its frozen launch inputs")
    origin = launch.inputs.qualification.compiler_origin
    if type(origin) is not FreshCompilerOrigin:
        raise C.StageGateError("compiler descendant authoring requires fresh Phase 1 baseline origin")
    origin.verify(candidate=launch.inputs.edit_authority.seed)
    inputs, policy = authored_inputs, authored_inputs.policy
    owner = inputs.stage_root / "author_origin"
    if owner.exists() or owner.is_symlink():
        raise C.StageGateError("compiler descendant refuses existing author observation state")
    with invocation_record.observe_call(
        owner,
        stage="fresh_compiler_phase2_authoring",
        function=CA.run_round,
        arguments={"model": model, "effort": effort, "wall_budget_seconds": wall_budget_seconds},
        inputs=(origin.receipt,),
        dependencies=(Path(__file__),),
    ) as observation:
        rc, transcript = CA.run_round(
            inputs.candidate,
            inputs.stage_root,
            model,
            {},
            policy.target_experiment,
            "bwrap",
            0,
            wall_budget_seconds,
            effort=effort,
            prompt=prompt,
            effective_model=model,
            continue_session=True,
            continuation_prompt=prompt,
            sandbox_command=lambda inner, ws, bundle, extra_binds=None: _control_command(
                inputs, inner, ws, bundle, extra_binds=extra_binds
            ),
            codex_binary=inputs.codex_binary,
            codex_home_root=inputs.stage_root / "codex_homes",
            candidate_read_paths=_read_paths(inputs),
            runtime_binds=lambda home: _runtime_binds(inputs, home),
            require_fresh_home=True,
            **(
                {"max_concurrent_subagents": launch.selected_subagent_count}
                if launch.selected_subagent_count is not None
                else {}
            ),
        )
        observation.returned(stdout=C.canonical_json({"exit_code": rc, "transcript": str(transcript)}))
    audit = audit_codex_transcript(transcript, policy.target_experiment, inputs.candidate, policy.build_registry())
    if rc not in (0, C.ROUND_DEADLINE_EXIT) or audit.get("clean") is not True:
        raise C.StageGateError("compiler descendant authoring transport/transcript was refused")
    inputs.verify()
    digest = _tree(inputs.candidate)["sha256"]
    receipts = policy.verify_receipts(policy.receipt_path, policy.build_registry(), audit, candidate_sha256=digest)
    frozen = owner / "compiler"
    shutil.copytree(inputs.candidate, frozen)
    _readonly(frozen)
    if _tree(frozen)["sha256"] != digest:
        raise C.StageGateError("compiler descendant changed while freezing author output")
    receipt = owner / "lineage.json"
    C.write_json(
        receipt,
        {
            "schema": "merlin.fresh_compiler_lineage.v1",
            "candidate_sha256": digest,
            "phase1_origin_sha256": origin.receipt_sha256,
            "launch_qualification_sha256": launch.qualification_sha256,
            "edit_authority_sha256": C.document_sha256(inputs.edit_authority.binding),
            "transcript_sha256": C.sha256_file(transcript),
            "audit": audit,
            "broker_receipts": receipts,
            "scope": "actual Phase 2 edits of fresh Phase 1 output; correctness and performance remain unqualified",
        },
    )
    receipt.chmod(0o400)
    lineage = ComponentCompilerLineage(
        origin,
        launch,
        authored_inputs,
        frozen,
        digest,
        receipt,
        C.sha256_file(receipt),
        transcript,
        C.sha256_file(transcript),
        policy.receipt_path,
        C.sha256_file(policy.receipt_path),
        observation.path,
        C.sha256_file(observation.path),
        object(),
    )
    _ISSUED[lineage._issuer] = _authority_identity(lineage)
    lineage.verify(candidate=inputs.candidate)
    return rc, transcript, lineage
