"""Descendant compiler authority observed around actual Phase 2 author transport.

The baseline is the issued fresh Phase 1 compiler. Cumulative edits are checked
against its frozen edit authority, and actual author/broker observations bind the
final membership. This proves neither correctness nor performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from merlin.common import invocation_record
from merlin_experiments.phase1.run_inputs import copy_candidate_tree
from merlin_experiments.phase2 import contracts as C

from .component_origin import FreshCompilerOrigin, _authority_identity, _plain, _readonly, _tree

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
        from merlin_experiments.phase2.component_launch import QualifiedComponentLaunch

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
    from merlin_experiments.phase1.providers import codex_agent as CA
    from merlin_experiments.phase2.component_launch import (
        ComponentLaunchInputs,
        QualifiedComponentLaunch,
        _control_command,
        _read_paths,
        _runtime_binds,
    )
    from merlin_experiments.phase2.transcript_audit import audit_codex_transcript

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
            **({"max_concurrent_subagents": launch.selected_subagent_count}
               if launch.selected_subagent_count is not None else {}),
        )
        observation.returned(stdout=C.canonical_json({"exit_code": rc, "transcript": str(transcript)}))
    audit = audit_codex_transcript(transcript, policy.target_experiment, inputs.candidate, policy.build_registry())
    if rc not in (0, C.ROUND_DEADLINE_EXIT) or audit.get("clean") is not True:
        raise C.StageGateError("compiler descendant authoring transport/transcript was refused")
    inputs.verify()
    digest = _tree(inputs.candidate)["sha256"]
    receipts = policy.verify_receipts(policy.receipt_path, policy.build_registry(), audit, candidate_sha256=digest)
    frozen = owner / "compiler"
    copy_candidate_tree(inputs.candidate, frozen)
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
