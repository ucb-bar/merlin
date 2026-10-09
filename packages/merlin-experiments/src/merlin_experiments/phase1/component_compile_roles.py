"""Evaluate the complete original source-only roster through ordinary compilation.

Large tensor types never enter input/golden generation or execution. Actual link
and whole-ELF policy observations are retained separately from the original
static obligations, which remain UNKNOWN without independent proof producers.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from merlin.targetgen.compile_only_execution import compile_source_only, verify_compile_only_report
from merlin.targetgen.contract.build_service import BuildOnlyService
from merlin_experiments.phase2 import contracts as C
from merlin_experiments.phase2.component_instruction_audit import IndependentLinkedInstructionCheck

from . import component_qualification_evidence as E
from .component_compile_admission import verify_compile_roster
from .component_copy_proof import ComponentCopyProof, prove_component_copy
from .component_lineage import ComponentCompilerLineage
from .component_origin import FreshCompilerOrigin, _authority_identity, _plain, _tree
from .component_package_execution import qualified_package_execution

_ISSUED: dict[object, tuple] = {}
SCHEMA = "merlin.component_compile_role_evaluation.v1"


def _selection(*, roster, origin, lineage, candidate, instruction_check):
    if type(origin) is not FreshCompilerOrigin:
        raise C.StageGateError("source-only evaluation requires actual fresh compiler origin")
    if lineage is None:
        origin.verify(candidate=candidate)
    elif type(lineage) is ComponentCompilerLineage and lineage.origin is origin:
        lineage.verify(candidate=candidate)
    else:
        raise C.StageGateError("source-only evaluation has no matching observed compiler lineage")
    inputs = origin.inputs
    compiler_transport = inputs.compiler_transport
    if roster is not inputs.compile_roster:
        raise C.StageGateError("source-only evaluation changed the original preauthor required roster")
    roster_sha = verify_compile_roster(
        roster, hardware=inputs.hardware, software=inputs.software, descriptor=inputs.target_experiment.path
    )
    if (
        type(instruction_check) is not IndependentLinkedInstructionCheck
        or instruction_check.policy.command_intake.hardware is not inputs.hardware
    ):
        raise C.StageGateError("source-only evaluation requires matching independently derived whole-ELF policy")
    return {
        "source_roster_sha256": roster_sha,
        "phase1_origin_sha256": origin.receipt_sha256,
        "phase2_lineage_sha256": lineage.receipt_sha256 if lineage is not None else None,
        "instruction_selection_sha256": instruction_check.verify(),
        "target_descriptor_sha256": C.sha256_file(roster.target_descriptor),
        "compiler_transport_sha256": compiler_transport.sha256 if compiler_transport is not None else None,
    }


def _denominators(rows):
    linked = sum(row["compilation_status"] == "linked" for row in rows)
    static = sum(len(row["static_obligations"]) for row in rows)
    proved = sum(value == "PROVED" for row in rows for value in row["static_obligations"].values())
    refuted = sum(value == "REFUTED" for row in rows for value in row["static_obligations"].values())
    unresolved = []
    for row in rows:
        if row["compilation_status"] != "linked":
            unresolved.append(row["member"]["name"] + ":candidate_compilation")
        unresolved.extend(
            row["member"]["name"] + ":" + name
            for name in row["member"]["required_static_obligations"]
            if row["static_obligations"][name] != "PROVED"
        )
        if row["member"]["expectation"] == "static_refusal":
            unresolved.append(row["member"]["name"] + ":original_static_refusal")
    return {
        "compilation_denominator": {"required": len(rows), "linked_and_policy_accepted": linked},
        "static_denominator": {
            "required": static,
            "proved": proved,
            "unknown": static - proved - refuted,
            **({"refuted": refuted} if refuted else {}),
        },
        "unresolved": unresolved,
    }


@dataclass(frozen=True)
class ComponentCompileRoleEvaluation:
    """Live transport evaluation, never an authority for unresolved static roles."""

    roster: object
    compiler_origin: FreshCompilerOrigin
    compiler_lineage: ComponentCompilerLineage | None
    candidate: Path
    candidate_sha256: str
    compiler_snapshot: Path
    contract_root: Path
    contract_sha256: str
    build_service: BuildOnlyService
    instruction_check: IndependentLinkedInstructionCheck
    readelf: Path
    receipt: Path
    receipt_sha256: str
    evidence_sha256: str
    _issuer: object = field(repr=False, compare=False)
    static_proofs: tuple[ComponentCopyProof, ...] = ()
    proof_evidence_sha256: str | None = None

    def verify(self, *, candidate=None):
        if (
            type(self._issuer) is not object
            or self._issuer not in _ISSUED
            or _authority_identity(self) != _ISSUED[self._issuer]
        ):
            raise C.StageGateError("source-only evaluation was not issued around actual original member compilation")
        selected = self.candidate if candidate is None else _plain(Path(candidate))
        binding = _selection(
            roster=self.roster,
            origin=self.compiler_origin,
            lineage=self.compiler_lineage,
            candidate=selected,
            instruction_check=self.instruction_check,
        )
        for source in (selected, self.candidate, self.compiler_snapshot):
            if _tree(source)["sha256"] != self.candidate_sha256:
                raise C.StageGateError("source-only compiler or private snapshot changed")
        if _tree(self.contract_root)["sha256"] != self.contract_sha256:
            raise C.StageGateError("source-only original grading contract changed")
        if C.sha256_file(_plain(self.receipt)) != self.receipt_sha256:
            raise C.StageGateError("source-only evaluation receipt changed")
        document = C.mapping_file(self.receipt)
        if document["selection"] != binding or document["component_sources"] != E.component_sources():
            raise C.StageGateError("source-only selection or implementation source membership changed")
        if _tree(self.receipt.parent / "cases")["sha256"] != self.evidence_sha256:
            raise C.StageGateError("source-only actual product/invocation evidence changed")
        proof_root = self.receipt.parent / "proofs"
        if self.proof_evidence_sha256 is not None and _tree(proof_root)["sha256"] != self.proof_evidence_sha256:
            raise C.StageGateError("source-only static proof attempt evidence changed")
        proofs = {}
        selection = getattr(self.compiler_origin.inputs, "pointer_storage", None)
        for proof in self.static_proofs:
            if type(proof) is not ComponentCopyProof or proof.selection is not selection or proof.member.name in proofs:
                raise C.StageGateError("source-only static proof changed the original storage/member selection")
            proofs[proof.member.name] = proof
        members = self.roster.members
        rows = document["members"]
        if len(rows) != len(members):
            raise C.StageGateError("source-only evaluation lost an original mandatory member")
        self.build_service.verify(self.compiler_origin.inputs.hardware.target)
        admission = self.instruction_check.admission_service()
        for row, member in zip(rows, members, strict=True):
            expected = {name: "UNKNOWN" for name in member.required_static_obligations}
            proof = proofs.get(member.name)
            if proof is not None:
                if proof.member is not member or proof.transport_report != Path(row["transport_report"]):
                    raise C.StageGateError("source-only proof substituted an original member or transport")
                checked = proof.verify()
                for name, value in checked["facets"].items():
                    if name in expected:
                        expected[name] = value
            if row["member"] != member.record() or row["static_obligations"] != expected:
                raise C.StageGateError("source-only evaluation changed original obligations or invented static proof")
            report_path = self.receipt.parent / "cases" / member.name / "compile_only_result.json"
            if row["transport_report"] != str(report_path):
                raise C.StageGateError("source-only evaluation substituted a different member transport")
            if row["compilation_status"] == "linked":
                report = verify_compile_only_report(
                    report_path,
                    build_service=self.build_service,
                    elf_admission=admission,
                    compiler_library=self.compiler_origin.inputs.library,
                    compiler_library_root=self.compiler_origin.inputs.view.root / "compiler",
                )
                if (
                    report["inputs"]["source"]["sha256"] != member.source_sha256
                    or report["inputs"]["source"]["path"] != str(member.source)
                    or report["inputs"]["original_abi"] != member.original_abi.record()
                    or report["inputs"]["package_root"] != str(self.compiler_snapshot)
                    or report["inputs"]["contract_root"] != str(self.contract_root)
                    or report["instruction_policy"]["status"] != "accepted"
                ):
                    raise C.StageGateError("source-only transport lost its original source/compiler/ABI/policy join")
            elif row["compilation_status"] not in {"unavailable", "refused_by_instruction_policy"}:
                raise C.StageGateError("source-only evaluation invented a compilation outcome")
        if any(document[key] != value for key, value in _denominators(rows).items()):
            raise C.StageGateError("source-only evaluation changed its original required denominator")
        if set(proofs) - {member.name for member in members}:
            raise C.StageGateError("source-only evaluation has an extra static proof member")
        if document["status"] != ("incomplete" if document["unresolved"] else "complete") or any(
            document[role] != "not_attempted" for role in ("numerical_execution", "physical_effects", "performance")
        ):
            raise C.StageGateError("source-only transport cannot qualify static, numerical or physical roles")
        return document

    def require_complete(self, *, candidate=None):
        document = self.verify(candidate=candidate)
        if document["unresolved"]:
            raise C.StageGateError("required original compilation/static roles remain incomplete")
        return document


def evaluate_component_compile_roles(
    *,
    roster,
    compiler_origin,
    candidate,
    contract_root,
    build_service,
    instruction_check,
    readelf,
    evidence_root,
    timeout_s,
    compiler_lineage=None,
):
    """Run every protected original source through the candidate's ordinary route.

    No static proof callback is accepted. The fixed narrow counted-copy checker
    consumes only independently selected original storage and actual products.
    Unsupported facets remain UNKNOWN; link success cannot fill those fields.
    """
    if type(timeout_s) is not int or not 0 < timeout_s <= 600 or type(build_service) is not BuildOnlyService:
        raise C.StageGateError("source-only evaluation requires explicit bounded ordinary build services")
    candidate, contract_root, root, readelf = map(_plain, map(Path, (candidate, contract_root, evidence_root, readelf)))
    binding = _selection(
        roster=roster,
        origin=compiler_origin,
        lineage=compiler_lineage,
        candidate=candidate,
        instruction_check=instruction_check,
    )
    target = compiler_origin.inputs.hardware.target
    build_service.verify(target)
    admission = instruction_check.admission_service()
    protected = (
        candidate,
        contract_root,
        compiler_origin.inputs.view.root,
        compiler_origin.inputs.corpus_root,
        *(member.source.parent for member in roster.members),
    )
    if root.exists() or any(root.is_relative_to(path) or path.is_relative_to(root) for path in protected):
        raise C.StageGateError("source-only evaluation needs a fresh private owner outside all input grants")
    before, contract_sha = _tree(candidate)["sha256"], _tree(contract_root)["sha256"]
    implementation = E.component_sources()
    root.mkdir(parents=True, mode=0o700)
    clone = root / "compiler"
    shutil.copytree(candidate, clone)
    if _tree(clone)["sha256"] != before:
        raise C.StageGateError("source-only private compiler snapshot changed during preparation")
    cases = root / "cases"
    cases.mkdir()
    proofs_root = root / "proofs"
    proofs_root.mkdir()
    proofs = []
    rows, deadline = [], time.monotonic() + timeout_s
    with qualified_package_execution(
        candidate=clone,
        view=compiler_origin.inputs.view,
        runtime=compiler_origin.inputs.runtime,
        evidence_root=cases,
        container_transport=compiler_origin.inputs.compiler_transport,
    ):
        for member in roster.members:
            output = cases / member.name
            status, failure = "unavailable", None
            try:
                remaining = int(deadline - time.monotonic())
                if remaining <= 0:
                    raise C.StageGateError("original source-only compilation budget exhausted")
                report = compile_source_only(
                    package_dir=clone,
                    source=member.source,
                    original_abi=member.original_abi,
                    contract_root=contract_root,
                    target=target,
                    output_root=output,
                    build_service=build_service,
                    elf_admission=admission,
                    readelf=readelf,
                    timeout_s=remaining,
                    compiler_library=compiler_origin.inputs.library,
                    compiler_library_root=compiler_origin.inputs.view.root / "compiler",
                )
                status = report["compilation_status"]
            except Exception as error:  # noqa: BLE001 -- retain every original unavailable member
                failure = {"type": type(error).__name__, "detail": str(error)}
            static = {name: "UNKNOWN" for name in member.required_static_obligations}
            proof_failure = None
            selection = getattr(compiler_origin.inputs, "pointer_storage", None)
            if status == "linked" and selection is not None and member.expectation == "compile_only":
                try:
                    proof = prove_component_copy(
                        selection=selection,
                        member=member,
                        transport_report=output / "compile_only_result.json",
                        build_service=build_service,
                        instruction_check=instruction_check,
                        output_root=proofs_root / member.name,
                        timeout_s=deadline - time.monotonic(),
                        compiler_library=compiler_origin.inputs.library,
                        compiler_library_root=compiler_origin.inputs.view.root / "compiler",
                    )
                    proofs.append(proof)
                    for name, value in proof.verify()["facets"].items():
                        if name in static:
                            static[name] = value
                except Exception as error:  # noqa: BLE001 -- retain original UNKNOWN plus actual failed attempt
                    proof_failure = {"type": type(error).__name__, "detail": str(error)}
            rows.append(
                {
                    "member": member.record(),
                    "compilation_status": status,
                    "failure": failure,
                    "transport_report": str(output / "compile_only_result.json"),
                    "static_obligations": static,
                    "static_proof_failure": proof_failure,
                }
            )
    receipt = root / "compile_roles.json"
    denominator = _denominators(rows)
    C.write_json(
        proofs_root / "attempts.json",
        {
            "scope": "actual fixed checker attempts; this summary cannot issue a theorem",
            "members": [{"name": row["member"]["name"], "failure": row["static_proof_failure"]} for row in rows],
        },
    )
    C.write_json(
        receipt,
        {
            "schema": SCHEMA,
            "status": "incomplete" if denominator["unresolved"] else "complete",
            "selection": binding,
            "candidate_sha256": before,
            "contract_sha256": contract_sha,
            "component_sources": implementation,
            "members": rows,
            **denominator,
            "numerical_execution": "not_attempted",
            "physical_effects": "not_attempted",
            "performance": "not_attempted",
            "scope": (
                "original finite source-only roster; fixed conditional emitted-copy proofs only; "
                "unsupported/static resources and physical obligations UNKNOWN"
            ),
        },
    )
    evaluation = ComponentCompileRoleEvaluation(
        roster,
        compiler_origin,
        compiler_lineage,
        candidate,
        before,
        clone,
        contract_root,
        contract_sha,
        build_service,
        instruction_check,
        readelf,
        receipt,
        C.sha256_file(receipt),
        _tree(cases)["sha256"],
        object(),
        tuple(proofs),
        _tree(proofs_root)["sha256"],
    )
    _ISSUED[evaluation._issuer] = _authority_identity(evaluation)
    evaluation.verify()
    return evaluation
