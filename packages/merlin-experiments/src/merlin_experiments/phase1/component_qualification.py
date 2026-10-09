"""Evaluate the reviewed independent component domain with the ordinary grader.

Coverage declares obligations. Qualification evaluates the selected compiler and
keeps the original complete numerical, instruction and executable gates. This
host owner is absent from the component author view.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from merlin_experiments.phase2 import contracts as C
from merlin_experiments.phase2.component_experiment import ComponentView, RuntimeGrant, verify_component_view

from . import component_qualification_domain as D
from . import component_qualification_evidence as E
from .component_compile_admission import qualification_compile_roles
from .component_lineage import ComponentCompilerLineage
from .component_origin import FreshCompilerOrigin, _authority_identity
from .component_package_execution import qualified_package_execution, selected_compiler_transport

_ISSUED: dict[object, tuple] = {}


def _verify_origin(origin, lineage, candidate: Path) -> dict:
    """Only actual fresh author transport can establish compiler provenance."""
    if type(origin) is not FreshCompilerOrigin:
        raise C.StageGateError("component qualification requires issued fresh Phase 1 compiler origin")
    if lineage is None:
        origin.verify(candidate=candidate)
    elif type(lineage) is ComponentCompilerLineage and lineage.origin is origin:
        lineage.verify(candidate=candidate)
    else:
        raise C.StageGateError("component qualification descendant has no matching observed author lineage")
    return {
        "phase1_origin_sha256": origin.receipt_sha256,
        "phase2_lineage_sha256": lineage.receipt_sha256 if lineage is not None else None,
    }


def _verify_runtime(authority, origin, target_descriptor: Path):
    try:
        from merlin_experiments.phase2.component_runtime import IndependentComponentRuntime
    except ImportError as error:
        raise C.StageGateError("independently issued component runtime support is unavailable") from error

    if type(authority) is not IndependentComponentRuntime:
        raise C.StageGateError("component qualification requires independently issued target runtime support")
    authority.verify(required_roles=("grade", "stage_verifier"))
    if (
        authority.target_descriptor != Path(target_descriptor).resolve()
        or authority.hardware_intake is not origin.inputs.hardware
        or authority is not origin.inputs.execution_support
    ):
        raise C.StageGateError("component runtime differs from the fresh compiler's independent hardware selection")
    return authority.sha256


def _component_sources() -> dict:
    return E.component_sources()


def _evaluator_distribution() -> dict:
    return E.evaluator_distribution()


@dataclass(frozen=True)
class ComponentQualification:
    candidate: Path
    candidate_sha256: str
    corpus_root: Path
    coverage_sha256: str
    receipt: Path
    receipt_sha256: str
    target_descriptor: Path
    target_descriptor_sha256: str
    contract_root: Path
    contract_sha256: str
    source_root: Path
    implementation_sources_json: str
    runtime: tuple[RuntimeGrant, ...]
    view: ComponentView
    status: str
    compiler_origin: FreshCompilerOrigin
    compiler_lineage: ComponentCompilerLineage | None
    runtime_authority: object
    runtime_authority_sha256: str
    _issuer: object = field(repr=False, compare=False)
    compile_role_evaluation: object = None
    container_transport: object = None

    def verify(self, *, candidate: Path | None = None) -> dict:
        """Reopen all frozen authorities; caller hashes and booleans grant nothing."""
        from . import source_inputs
        from .component_source_applicability import evaluate_component_source_applicability

        if (
            type(self._issuer) is not object
            or self._issuer not in _ISSUED
            or _authority_identity(self) != _ISSUED[self._issuer]
            or self.status != "qualified"
        ):
            raise C.StageGateError("component compiler has no evaluated domain qualification")
        compile_roles, unresolved = qualification_compile_roles(
            self.compile_role_evaluation,
            origin=self.compiler_origin,
            lineage=self.compiler_lineage,
            candidate=self.candidate if candidate is None else candidate,
            contract_root=self.contract_root,
        )
        if unresolved:
            raise C.StageGateError("required original compilation/static roles remain incomplete")
        if C.sha256_file(self.receipt) != self.receipt_sha256:
            raise C.StageGateError("component qualification receipt changed")
        document = C.mapping_file(self.receipt)
        if document.get("compile_roles") != compile_roles:
            raise C.StageGateError("component qualification original compilation/static evidence changed")
        if document.get("status") != "qualified" or document.get("failures"):
            raise C.StageGateError("component qualification has unresolved failures")
        selected = self.candidate if candidate is None else Path(candidate)
        if _verify_origin(self.compiler_origin, self.compiler_lineage, selected) != document.get("compiler_origin"):
            raise C.StageGateError("component qualification fresh compiler provenance changed")
        if (
            _verify_runtime(self.runtime_authority, self.compiler_origin, self.target_descriptor)
            != self.runtime_authority_sha256
        ):
            raise C.StageGateError("component qualification independent runtime changed")
        transport = selected_compiler_transport(self.runtime_authority, view=self.view, runtime=self.runtime)
        if transport is not self.container_transport or document.get("compiler_transport_sha256") != (
            transport.sha256 if transport is not None else None
        ):
            raise C.StageGateError("component qualification compiler command transport changed")
        if C.exact_tree_record(selected)["sha256"] != self.candidate_sha256:
            raise C.StageGateError("component compiler changed after domain qualification")
        if C.sha256_file(self.target_descriptor) != self.target_descriptor_sha256:
            raise C.StageGateError("component qualification target selection changed")
        if C.exact_tree_record(self.contract_root)["sha256"] != self.contract_sha256:
            raise C.StageGateError("component grading contract changed")
        report, domain = D.reopen_domain(self.corpus_root, origin=self.compiler_origin)
        D.verify_receipt_domain(document, domain)
        if report["sha256"] != self.coverage_sha256:
            raise C.StageGateError("component domain membership changed")
        source_inputs.verify(
            json.loads(self.implementation_sources_json),
            repo=self.source_root,
            entrypoint=Path(__file__),
            descriptor=None,
        )
        if _component_sources() != document["component_sources"]:
            raise C.StageGateError("component ordinary compiler/grader source membership changed")
        if _evaluator_distribution() != document["evaluator_distribution"]:
            raise C.StageGateError("installed component evaluator distribution changed")
        for row in document.get("source_applicability", []):
            observed = evaluate_component_source_applicability(
                source=Path(row["source"]),
                source_program_sha256=row["source_program_sha256"],
                frontend=row["frontend"],
            )
            if observed.record() != row:
                raise C.StageGateError("component source applicability changed after qualification")
        for grant in self.runtime:
            grant.verify()
        verify_component_view(self.view)
        for row in document["result_evidence"]:
            if C.sha256_file(Path(row["path"])) != row["sha256"]:
                raise C.StageGateError("component execution evidence changed")
        if C.exact_tree_record(self.receipt.parent / "grade")["sha256"] != document["execution_tree_sha256"]:
            raise C.StageGateError("component executable or complete output evidence changed")
        E.verify_execution_evidence(
            document=document,
            compiler_root=self.receipt.parent / "compiler",
            grade_root=self.receipt.parent / "grade",
            candidate_sha256=self.candidate_sha256,
            report=report,
            corpus_root=self.corpus_root,
            descriptor_sha256=self.target_descriptor_sha256,
            preparation=D.preparation_for_origin(self.compiler_origin),
        )
        return document


def _passed(row: dict) -> bool:
    # The historical grader reports some uncertified rows as pass and attaches
    # cert_verdict. A component qualification must consume that verdict.
    return row.get("status") == "pass" and row.get("numeric") == "pass" and not row.get("cert_verdict")


def qualify_component_compiler(
    candidate: Path,
    *,
    corpus_root: Path,
    target_experiment,
    contract_root: Path,
    source_root: Path,
    evidence_root: Path,
    runtime: tuple[RuntimeGrant, ...],
    view: ComponentView,
    timeout_s: int = 600,
    max_workers: int = 1,
    compiler_origin: FreshCompilerOrigin | None = None,
    compiler_lineage: ComponentCompilerLineage | None = None,
    runtime_authority=None,
    compile_role_evaluation=None,
) -> ComponentQualification:
    """Grade every selected original mandatory member, including declared refusals.

    The candidate is cloned privately before its normal build. Goldens, held
    members and execution records never enter the public authoring view. Source
    correspondence/native program checks stay with the existing grader and its
    selected target build services; numerical policies are never replaced here.
    Explicit source preparation keeps its complete mandatory candidate roster;
    historical coverage retains guard/transfer qualification.
    """
    if type(timeout_s) is not int or not 0 < timeout_s <= 600 or type(max_workers) is not int or max_workers != 1:
        raise C.StageGateError("component qualification requires bounded execution and one scoped compiler worker")
    origin_binding = _verify_origin(compiler_origin, compiler_lineage, Path(candidate))
    runtime_digest = _verify_runtime(runtime_authority, compiler_origin, Path(target_experiment.path))
    container_transport = selected_compiler_transport(runtime_authority, view=view, runtime=runtime)
    compile_roles, compile_failures = qualification_compile_roles(
        compile_role_evaluation,
        origin=compiler_origin,
        lineage=compiler_lineage,
        candidate=Path(candidate),
        contract_root=contract_root,
    )
    from merlin_experiments.phase0.component_coverage import build_guard_link

    from . import source_inputs
    from .component_source_applicability import evaluate_component_source_applicability
    from .component_witness import REQUIRED_EXECUTION_EFFECTS, verify_component_stage_witness

    verify_component_view(view)
    if not isinstance(runtime, tuple) or not runtime or any(type(row) is not RuntimeGrant for row in runtime):
        raise C.StageGateError("component qualification requires complete selected runtime file membership")
    candidate, corpus_root = Path(candidate).resolve(), Path(corpus_root).resolve()
    source_root, contract_root = Path(source_root).resolve(), Path(contract_root).resolve()
    evidence_root = Path(evidence_root)
    if evidence_root.exists() or evidence_root.is_symlink():
        raise C.StageGateError("component qualification evidence destination must be fresh")
    if any(evidence_root.resolve().is_relative_to(root) for root in (candidate, corpus_root, contract_root)):
        raise C.StageGateError("component qualification evidence overlaps immutable inputs")
    for grant in runtime:
        grant.verify()
    before = C.exact_tree_record(candidate)["sha256"]
    contract_digest = C.exact_tree_record(contract_root)["sha256"]
    descriptor = Path(target_experiment.path).resolve()
    descriptor_digest = C.sha256_file(descriptor)
    # Target support is the independently issued runtime above. Passing the
    # descriptor to the legacy inventory would rediscover the old OOT provider.
    implementation = source_inputs.record(repo=source_root, entrypoint=Path(__file__), descriptor=None)
    component_sources = _component_sources()
    evaluator_distribution = _evaluator_distribution()
    report, domain = D.reopen_domain(corpus_root, origin=compiler_origin)
    preparation = D.preparation_for_origin(compiler_origin)
    obligations = D.selected_obligations(report, preparation=preparation)
    cohorts, guards = {row["cohort"] for row in obligations}, {"functional_guard", "withheld_transfer"}
    if not obligations or (cohorts != guards if preparation is None else not guards <= cohorts):
        raise C.StageGateError("qualification needs mandatory independent guards and withheld transfer obligations")
    members = [member for row in obligations for member in row["members"]]
    if len({member["name"] for member in members}) != len(members):
        raise C.StageGateError("component functional membership contains duplicate names")
    evidence_root.mkdir(parents=True, mode=0o700)
    clone = evidence_root / "compiler"
    shutil.copytree(candidate, clone)
    if C.exact_tree_record(clone)["sha256"] != before:
        raise C.StageGateError("private component compiler changed while preparing its exact snapshot")
    score, failures, result_evidence, stage_witnesses, source_observations = None, list(compile_failures), [], [], []
    try:
        with qualified_package_execution(
            candidate=clone,
            view=view,
            runtime=runtime,
            evidence_root=evidence_root / "grade",
            container_transport=container_transport,
        ):
            score = runtime_authority.grader(
                clone,
                capsules_root=[corpus_root / member["member"] for member in members],
                runs_root=evidence_root / "grade",
                model_snapshot_root=evidence_root / "model_sources",
                labels={"public", "hidden"},
                contract=contract_root,
                timeout=timeout_s,
                max_workers=max_workers,
                target=target_experiment.target,
            )
        rows = score.get("per_capsule") or []
        index = {row.get("capsule"): row for row in rows}
        if len(rows) != len(members) or set(index) != {member["name"] for member in members}:
            failures.append("mandatory declared-domain denominator did not execute completely")
        result_paths = sorted((evidence_root / "grade").rglob("capsule_result.json"))
        actual_results = {C.mapping_file(path).get("capsule"): C.mapping_file(path) for path in result_paths}
        for obligation in obligations:
            for member in obligation["members"]:
                row = index.get(member["name"], {})
                if obligation["expectation"] == "unsupported_program":
                    # Only the actual ordinary lowering refusal can discharge a
                    # declared unsupported case. An execution crash is not refusal.
                    actual = actual_results.get(member["name"], {})
                    # The ordinary score intentionally omits the refusal plane.
                    # Reopen its actual produced record, not an invented score
                    # field or a candidate exit-code summary.
                    refused = (
                        row.get("status") == "declined"
                        and actual.get("status") == "declined"
                        and (actual.get("failure") or {}).get("plane") == "backend_declined"
                        and isinstance((actual.get("declined") or {}).get("reason"), str)
                        and bool(actual["declined"]["reason"].strip())
                    )
                    if not refused:
                        failures.append(member["name"] + ": declared unsupported program was not refused")
                elif not _passed(row):
                    failures.append(member["name"] + ": complete numerical/executable certification failed")
        if score.get("integrity_status") != "clean":
            failures.append("candidate integrity gate did not pass")
        for path in result_paths:
            result_evidence.append({"path": str(path.resolve()), "sha256": C.sha256_file(path)})
        # Numeric/executable assertions without the grader's produced per-member
        # records are not independently evaluated evidence.
        if len(result_evidence) != len(members):
            failures.append("complete per-member execution evidence is absent")
        # Numerical agreement and decoded ISA do not establish capture fidelity,
        # host/device placement or invocation/link correspondence. Require the
        # selected private verifier to reopen those actual execution authorities.
        try:
            runtime_authority.verify(required_roles=("grade", "stage_verifier"))
            verifier = runtime_authority.stage_verifier
            result_index = {
                C.mapping_file(Path(row["path"])).get("capsule"): Path(row["path"]) for row in result_evidence
            }
            declarations = {row["id"]: row for row in report.get("declaration", {}).get("obligations", [])}
            for obligation in obligations:
                if obligation["expectation"] == "unsupported_program":
                    continue
                frontend = declarations.get(obligation["id"], {}).get("frontend", "mlir")
                for member in obligation["members"]:
                    capsule_root = corpus_root / member["member"]
                    capsule = C.mapping_file(capsule_root / "capsule.yaml", yaml_file=True)
                    effects = tuple(
                        sorted(
                            set(REQUIRED_EXECUTION_EFFECTS)
                            | set((capsule.get("component_coverage") or {}).get("generated_effects", []))
                        )
                    )
                    source_applicability = evaluate_component_source_applicability(
                        source=capsule_root / capsule.get("interface_mlir", "capsule.interface.mlir"),
                        source_program_sha256=member["program_sha256"],
                        frontend=frontend,
                    )
                    source_observations.append(source_applicability.record())
                    witness = verifier(
                        member=member,
                        result_path=result_index[member["name"]],
                        candidate_root=candidate,
                        compiler_snapshot=clone,
                        candidate_sha256=before,
                        capsule_root=capsule_root,
                        evidence_root=evidence_root / "grade",
                        target_descriptor=descriptor,
                        frontend=frontend,
                        required_effects=effects,
                        timeout_s=timeout_s,
                        source_applicability=source_applicability,
                    )
                    stage_witnesses.append(
                        {
                            "member_name": member["name"],
                            **verify_component_stage_witness(
                                witness,
                                member=member,
                                candidate_sha256=before,
                                target_descriptor_sha256=descriptor_digest,
                                frontend=frontend,
                                capsule_root=capsule_root,
                                evidence_root=evidence_root / "grade",
                                required_effects=effects,
                            ),
                        }
                    )
            runtime_authority.verify(required_roles=("grade", "stage_verifier"))
        except Exception as exc:  # noqa: BLE001 - numeric pass cannot replace absent facet evidence
            failures.append("component stage/effect witnesses UNKNOWN: " + type(exc).__name__ + ": " + str(exc))
    except Exception as exc:  # noqa: BLE001 - persist failed/unavailable attempts too
        failures.append("component qualification unavailable: " + type(exc).__name__ + ": " + str(exc))
    invocation_evidence, snapshot_digest = [], None
    try:
        snapshot_digest = C.exact_tree_record(clone)["sha256"]
        if snapshot_digest != before:
            raise C.StageGateError("private component compiler changed during domain qualification")
        invocation_evidence = E.invocation_members(evidence_root / "grade")
        E.require_member_invocations(report, evidence_root / "grade", invocation_evidence, preparation=preparation)
    except Exception as exc:  # noqa: BLE001 - keep real incomplete or drifted attempts unavailable
        failures.append("component actual invocation evidence unavailable: " + str(exc))
    if C.exact_tree_record(candidate)["sha256"] != before:
        failures.append("compiler source changed during domain qualification")
    if _verify_origin(compiler_origin, compiler_lineage, candidate) != origin_binding:
        failures.append("fresh compiler authoring provenance changed during domain qualification")
    if _verify_runtime(runtime_authority, compiler_origin, descriptor) != runtime_digest:
        failures.append("independent target runtime changed during domain qualification")
    reopened, actual_domain = D.reopen_domain(corpus_root, origin=compiler_origin)
    if reopened["sha256"] != report["sha256"] or not D.unchanged_domain(domain, actual_domain):
        failures.append("component domain changed during qualification")
    source_inputs.verify(implementation, repo=source_root, entrypoint=Path(__file__), descriptor=None)
    if component_sources != _component_sources():
        failures.append("component ordinary compiler/grader source membership changed during qualification")
    if evaluator_distribution != _evaluator_distribution():
        failures.append("installed component evaluator distribution changed during qualification")
    for grant in runtime:
        grant.verify()
    if (
        C.sha256_file(descriptor) != descriptor_digest
        or C.exact_tree_record(contract_root)["sha256"] != contract_digest
    ):
        failures.append("target or grading contract changed during qualification")
    if qualification_compile_roles(
        compile_role_evaluation,
        origin=compiler_origin,
        lineage=compiler_lineage,
        candidate=candidate,
        contract_root=contract_root,
    ) != (compile_roles, compile_failures):
        failures.append("original compilation/static evidence changed during qualification")
    status = "qualified" if not failures else "refused"
    receipt = evidence_root / "qualification.json"
    C.write_json(
        receipt,
        {
            **D.receipt_domain(domain),
            "status": status,
            "compiler_origin": origin_binding,
            "compile_roles": compile_roles,
            "runtime_authority_sha256": runtime_digest,
            "compiler_transport_sha256": container_transport.sha256 if container_transport is not None else None,
            "candidate_sha256": before,
            "compiler_snapshot_sha256": snapshot_digest,
            "invocation_evidence": invocation_evidence,
            "coverage_sha256": report["sha256"],
            "guard_link": build_guard_link(report),
            "target_descriptor_sha256": descriptor_digest,
            "contract_sha256": contract_digest,
            "implementation_sources": implementation,
            "component_sources": component_sources,
            "evaluator_distribution": evaluator_distribution,
            "runtime": [{"destination": row.destination, "sha256": row.sha256} for row in runtime],
            "public_view": {
                "root": str(view.root),
                "manifest_sha256": view.manifest_sha256,
                "generation_sha256": view.generation_sha256,
                "library_sha256": view.library_sha256,
            },
            "score": score,
            "failures": failures,
            "result_evidence": result_evidence,
            "stage_witnesses": stage_witnesses,
            "source_applicability": source_observations,
            "execution_tree_sha256": C.exact_tree_record(evidence_root / "grade")["sha256"]
            if (evidence_root / "grade").is_dir()
            else None,
            "scope": "mandatory finite declared domain; every output; no arbitrary future-model proof",
        },
    )
    receipt.chmod(0o400)
    qualification = ComponentQualification(
        candidate,
        before,
        corpus_root,
        report["sha256"],
        receipt,
        C.sha256_file(receipt),
        descriptor,
        descriptor_digest,
        contract_root,
        contract_digest,
        source_root,
        json.dumps(implementation, sort_keys=True),
        runtime,
        view,
        status,
        compiler_origin,
        compiler_lineage,
        runtime_authority,
        runtime_digest,
        object(),
        compile_role_evaluation,
        container_transport,
    )
    _ISSUED[qualification._issuer] = _authority_identity(qualification)
    return qualification
