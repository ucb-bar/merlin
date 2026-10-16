"""Ordinary component execution using independently qualified runtime services.

The compiler is the fresh Phase 1 baseline or its current candidate. Runtime
grading and semantic observation come from separately evaluated independent
support. No compiler-provider backend, renderer or reference compiler is resolved.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from merlin.benchharness import hash_tree
from merlin.perf.component_cost import ComponentCostScope
from merlin_experiments.phase1.component_package_execution import (
    qualified_package_execution,
    selected_compiler_transport,
)
from merlin_experiments.phase1.component_witness import REQUIRED_EXECUTION_EFFECTS, verify_component_stage_witness

from .component_baseline import ComponentBaselineAdmission, verify_baseline_admission
from .component_experiment import ComponentView, RuntimeGrant, verify_component_view
from .component_runtime import IndependentComponentRuntime, require_independent_runtime
from .contracts import StageGateError, document_sha256, exact_tree_record, mapping_file, sha256_file
from .edit_authority import FrozenEditAuthority

_ISSUER = object()
_DEFAULT_VERIFICATION = object()


def _candidate_authority(admission, candidate, edit_authority):
    if (
        type(edit_authority) is not FrozenEditAuthority
        or not edit_authority.configured
        or not isinstance(candidate, Path)
        or candidate.resolve() != candidate
        or candidate.is_symlink()
        or edit_authority.initial_source != candidate
    ):
        raise StageGateError("independent execution requires the exact controller-owned candidate edit authority")
    edit_authority.check_integrity()
    admission.qualification.verify(candidate=edit_authority.seed)
    edit_authority.validate_candidate(candidate)
    return document_sha256(edit_authority.binding)


def _member_binding(admission, member):
    from merlin_experiments.phase0.component_coverage import verify_report
    from merlin_experiments.phase1.source_inputs import fingerprint

    report = verify_report(admission.qualification.corpus_root)
    rows = [
        row
        for obligation in report["obligations"]
        if obligation["cohort"] == "development"
        for row in obligation["members"]
        if row["state"] == "generated" and row["name"] == member.capsule
    ]
    if len(rows) != 1 or fingerprint(member.source_dir) != rows[0]["sha256"]:
        raise StageGateError("component execution member is outside the exact independent development cohort")
    return rows[0]


@dataclass(frozen=True)
class ComponentIndependentExecution:
    baseline_admission: ComponentBaselineAdmission
    independent_runtime: IndependentComponentRuntime
    candidate: Path
    edit_authority: FrozenEditAuthority
    edit_authority_sha256: str
    view: ComponentView
    runtime: tuple[RuntimeGrant, ...]
    contract_root: Path
    contract_sha256: str
    scope: ComponentCostScope
    output: Path
    _issuer: object = field(repr=False, compare=False)
    container_transport: object = None

    @property
    def sha256(self):
        return document_sha256(
            {
                "schema": "merlin.component_independent_execution.v1",
                "baseline_admission_sha256": self.baseline_admission.sha256,
                "independent_runtime_sha256": self.independent_runtime.sha256,
                "candidate_path": str(self.candidate),
                "edit_authority_sha256": self.edit_authority_sha256,
                "view_sha256": self.view.manifest_sha256,
                "runtime": [(str(grant.source), grant.destination, grant.sha256) for grant in self.runtime],
                "contract_sha256": self.contract_sha256,
                "scope_sha256": self.scope.sha256,
                "compiler_transport_sha256": self.container_transport.sha256
                if self.container_transport is not None
                else None,
            }
        )

    def verify(self, *, retained_revision=_DEFAULT_VERIFICATION):
        if retained_revision is _DEFAULT_VERIFICATION:
            from .retained_revisions import executing_candidate_revision

            retained_revision = executing_candidate_revision(self.edit_authority)
        if self._issuer is not _ISSUER:
            raise StageGateError("component execution requires issued independent normal service authority")
        verify_baseline_admission(
            self.baseline_admission,
            baseline=self.baseline_admission.baseline,
            corpus=self.baseline_admission.corpus,
            target_descriptor=self.baseline_admission.target_descriptor,
        )
        require_independent_runtime(
            self.independent_runtime,
            required_roles=("grade", "stage_verifier"),
            target_descriptor=self.baseline_admission.target_descriptor,
        )
        transport = selected_compiler_transport(self.independent_runtime, view=self.view, runtime=self.runtime)
        if (
            transport is not self.container_transport
            or transport is not self.baseline_admission.qualification.container_transport
        ):
            raise StageGateError("independent execution compiler transport differs from its original qualification")
        if retained_revision is None:
            edit_sha = _candidate_authority(self.baseline_admission, self.candidate, self.edit_authority)
        else:
            retained_revision.verify(authority=self.edit_authority)
            self.baseline_admission.qualification.verify(candidate=self.edit_authority.seed)
            edit_sha = retained_revision.edit_authority_sha256
        if edit_sha != self.edit_authority_sha256:
            raise StageGateError("independent component execution candidate/edit selection changed")
        verify_component_view(self.view)
        if exact_tree_record(self.contract_root)["sha256"] != self.contract_sha256:
            raise StageGateError("independent component grading contract changed")
        for grant in self.runtime:
            grant.verify()
        return self.sha256

    def execute(self, *, compiler, member, corpus, workspace, timeout_s):
        """Compile, grade and verify one exact component through ordinary services."""
        from .retained_revisions import executing_revision

        retained = executing_revision(Path(compiler))
        self.verify(retained_revision=retained)
        self.baseline_admission.verify(corpus=corpus)
        if member not in corpus.capsules:
            raise StageGateError("independent component execution selected another source member")
        if type(timeout_s) not in (int, float):
            raise StageGateError("independent component execution requires a bounded remaining budget")
        timeout = min(float(timeout_s), 600.0)
        if not math.isfinite(timeout) or timeout < 1:
            raise StageGateError("independent component execution requires a bounded remaining budget")
        compiler, workspace = Path(compiler), Path(workspace)
        if (
            compiler.resolve() != compiler
            or compiler.is_symlink()
            or workspace.resolve() != workspace
            or workspace.exists()
            or workspace.is_symlink()
            or not workspace.is_relative_to(self.output)
            or compiler.is_relative_to(workspace)
            or workspace.is_relative_to(compiler)
        ):
            raise StageGateError("independent component execution requires exact source and fresh private output")
        before = str(hash_tree(compiler)["sha256"])
        selected_sha = exact_tree_record(compiler)["sha256"]
        current_sha = exact_tree_record(self.candidate)["sha256"] if retained is None else None
        seed_sha = exact_tree_record(self.baseline_admission.baseline)["sha256"]
        if retained is None and selected_sha not in {seed_sha, current_sha}:
            raise StageGateError("independent execution selected neither the fresh seed nor the authorized candidate")
        started = time.monotonic()
        member_binding = _member_binding(self.baseline_admission, member)
        workspace.mkdir(parents=True, mode=0o700)
        if retained is None:
            clone = workspace / "compiler"
            shutil.copytree(compiler, clone)
        else:
            retained.verify(authority=self.edit_authority)
            clone = retained.source
        candidate_sha = exact_tree_record(clone)["sha256"]
        grade_root = workspace / "grade"
        with qualified_package_execution(
            candidate=clone,
            view=self.view,
            runtime=self.runtime,
            evidence_root=grade_root,
            container_transport=self.container_transport,
        ):
            score = self.independent_runtime.grader(
                clone,
                capsules_root=[member.source_dir],
                runs_root=grade_root,
                model_snapshot_root=workspace / "sources",
                labels={"public", "hidden", "dev"},
                contract=self.contract_root,
                timeout=int(timeout),
                max_workers=1,
                target=self.independent_runtime.hardware_intake.target,
            )
        rows = score.get("per_capsule") if isinstance(score, dict) else None
        if (
            not isinstance(score, dict)
            or score.get("integrity_status") != "clean"
            or not isinstance(rows, list)
            or len(rows) != 1
            or not isinstance(rows[0], dict)
            or rows[0].get("capsule") != member.capsule
            or rows[0].get("status") != "pass"
            or rows[0].get("numeric") != "pass"
            or rows[0].get("cert_verdict")
        ):
            raise StageGateError("independent component failed complete ordinary numerical/executable gates")
        paths = tuple(grade_root.rglob("capsule_result.json"))
        if len(paths) != 1:
            raise StageGateError("independent component lacks its actual complete produced result")
        result_path = paths[0]
        actual = mapping_file(result_path)
        if (
            actual.get("capsule") != member.capsule
            or actual.get("status") != "pass"
            or actual.get("numeric") != "pass"
            or actual.get("failure")
            or actual.get("cert_verdict")
        ):
            raise StageGateError("independent component summary differs from its actual produced result")
        from merlin_experiments.phase1.component_source_applicability import evaluate_component_source_applicability

        capsule = mapping_file(member.source_dir / "capsule.yaml", yaml_file=True)
        coverage = capsule.get("component_coverage") or {}
        frontend = coverage.get("frontend", "mlir")
        effects = tuple(sorted(set(REQUIRED_EXECUTION_EFFECTS) | set(coverage.get("generated_effects", []))))
        source = member.source_dir / capsule.get("interface_mlir", "capsule.interface.mlir")
        applicability = evaluate_component_source_applicability(
            source=source,
            source_program_sha256=member_binding["program_sha256"],
            frontend=frontend,
        )
        remaining = int(timeout - (time.monotonic() - started))
        if remaining <= 0:
            raise StageGateError("independent component execution exhausted its remaining budget")
        witness = self.independent_runtime.stage_verifier(
            member=member_binding,
            result_path=result_path,
            candidate_root=compiler,
            compiler_snapshot=clone,
            candidate_sha256=candidate_sha,
            capsule_root=member.source_dir,
            evidence_root=grade_root,
            target_descriptor=self.baseline_admission.target_descriptor,
            frontend=frontend,
            required_effects=effects,
            timeout_s=remaining,
            source_applicability=applicability,
        )
        verified = verify_component_stage_witness(
            witness,
            member=member_binding,
            candidate_sha256=candidate_sha,
            target_descriptor_sha256=self.baseline_admission.target_sha256,
            frontend=frontend,
            capsule_root=member.source_dir,
            evidence_root=grade_root,
            required_effects=effects,
        )
        if (
            str(hash_tree(compiler)["sha256"]) != before
            or exact_tree_record(clone)["sha256"] != candidate_sha
            or retained is None
            and exact_tree_record(self.candidate)["sha256"] != current_sha
        ):
            raise StageGateError("independent component execution changed frozen compiler bytes")
        self.verify(retained_revision=retained)
        receipt = workspace / "execution.json"
        receipt.write_text(
            json.dumps(
                {
                    "schema": "merlin.independent_component_execution_observation.v1",
                    "service_sha256": self.sha256,
                    "compiler_sha256": before,
                    "member_sha256": member.source_sha256,
                    "result_sha256": sha256_file(result_path),
                    "stage_witness": verified,
                    "cost_status": "UNKNOWN",
                    "scope": "full functional execution only",
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "score": score,
            "result_path": result_path,
            "stage_witness": witness,
            "receipt_path": receipt,
            "receipt_sha256": sha256_file(receipt),
        }


def prepare_component_independent_execution(
    *,
    baseline_admission,
    independent_runtime,
    view,
    runtime,
    contract_root,
    scope,
    output,
    candidate=None,
    edit_authority=None,
):
    """Issue the ordinary route from fresh source and independent runtime authority."""
    if (
        type(baseline_admission) is not ComponentBaselineAdmission
        or type(scope) is not ComponentCostScope
        or type(view) is not ComponentView
        or not isinstance(runtime, tuple)
        or not runtime
        or any(type(grant) is not RuntimeGrant for grant in runtime)
    ):
        raise StageGateError("independent execution requires exact fresh baseline, scope, view and runtime grants")
    baseline_admission.verify()
    require_independent_runtime(
        independent_runtime,
        required_roles=("grade", "stage_verifier"),
        target_descriptor=baseline_admission.target_descriptor,
    )
    edit_sha = _candidate_authority(baseline_admission, candidate, edit_authority)
    contract_root, output = Path(contract_root), Path(output)
    if any(path.resolve() != path or path.is_symlink() for path in (contract_root, output)):
        raise StageGateError("independent execution inputs require canonical paths")
    if any(
        output.is_relative_to(path) or path.is_relative_to(output)
        for path in (
            baseline_admission.baseline,
            baseline_admission.corpus.root,
            candidate,
            view.root,
            contract_root,
        )
    ):
        raise StageGateError("independent execution private output overlaps frozen source/runtime inputs")
    qualification = baseline_admission.qualification
    if qualification.view != view or qualification.runtime != runtime or qualification.contract_root != contract_root:
        raise StageGateError("independent execution public/contract/runtime differs from its original qualification")
    container_transport = selected_compiler_transport(independent_runtime, view=view, runtime=runtime)
    if container_transport is not qualification.container_transport:
        raise StageGateError("independent execution command transport differs from its original qualification")
    binding = ComponentIndependentExecution(
        baseline_admission,
        independent_runtime,
        candidate,
        edit_authority,
        edit_sha,
        view,
        runtime,
        contract_root,
        exact_tree_record(contract_root)["sha256"],
        scope,
        output,
        _ISSUER,
        container_transport,
    )
    binding.verify()
    return binding
