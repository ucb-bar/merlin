"""Typed public diagnostic inputs for capsule-qualified source development.

These are live original input owners, not a performance corpus or qualification.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from merlin.targetgen.common_transforms import CommonTransformService
from merlin.targetgen.target_experiment import TargetExperiment

from . import public_diagnostic_corpus as PDC
from .contracts import StageGateError, canonical_json, mapping_file, sha256_file


def _pin(path):
    path = Path(path)
    if path.is_symlink() or path.resolve(strict=True) != path or not path.is_file():
        raise StageGateError("capsule development needs canonical original descriptor leaves")
    return {"path": str(path), "sha256": sha256_file(path)}


@dataclass(frozen=True)
class CapsuleSourceDevelopment:
    corpus: PDC.FrozenPublicDiagnosticCorpus
    service: CommonTransformService
    target: TargetExperiment
    native_exploration: object | None = None
    diagnostic_pair_check: object | None = None
    qualification_verification: dict | None = None

    def verify(self, qualification_owner):
        if (
            type(self) is not CapsuleSourceDevelopment
            or type(self.corpus) is not PDC.FrozenPublicDiagnosticCorpus
            or type(self.service) is not CommonTransformService
            or type(self.target) is not TargetExperiment
        ):
            raise StageGateError(
                "capsule source development requires exact original public corpus, profile and target owners"
            )
        from .capsule_iteration import FrozenQualifiedCapsulePhase1

        if type(qualification_owner) is not FrozenQualifiedCapsulePhase1:
            raise StageGateError("development descriptor needs its original qualified capsule owner")
        if self.qualification_verification is not None:
            config = self.qualification_verification
            if (
                type(config) is not dict
                or set(config) != {"snapshot", "seal_sha256", "qualification"}
                or config["qualification"]
                != {
                    "path": str(qualification_owner.qualification_path),
                    "sha256": qualification_owner.qualification_sha256,
                }
            ):
                raise StageGateError("worker qualification inputs differ from the original selected receipt")
        qualification = _pin(qualification_owner.descriptor)
        development = _pin(self.target.path)
        if development["sha256"] != self.target.descriptor_sha256:
            raise StageGateError("development descriptor differs from its original parsed target")
        original = mapping_file(qualification_owner.descriptor, yaml_file=True)
        current = mapping_file(self.target.path, yaml_file=True)
        if current.get("target") != self.target.target:
            raise StageGateError("development target differs from its selected descriptor")
        # A different public input view is explicit. Every other selected ISA,
        # hardware/software/numerical declaration stays exactly the same.
        original.pop("capsule_corpus", None)
        current.pop("capsule_corpus", None)
        if canonical_json(original) != canonical_json(current):
            raise StageGateError(
                "development descriptor changed original hardware, software, ISA or numerical declarations"
            )
        corpus = self.corpus.verify()
        if corpus["target"] != self.target.target or corpus["performance_cohort_admitted"] is not False:
            raise StageGateError("development corpus cannot become a performance objective")
        self.service._verify()
        return dict(
            schema="merlin.capsule_source_development_inputs.v1",
            qualification_verification=self.qualification_verification,
            qualification_descriptor=qualification,
            development_descriptor=development,
            corpus_manifest=_pin(self.corpus.manifest_path),
            source_members=corpus["members"],
            complete_pending_inventory=corpus["planning_inventory"],
            profile_sha256=self.service.profile["sha256"],
            snapshot_sha256=self.service.snapshot_digest,
            formal_admission=False,
            performance_cohort_admitted=False,
            promotion_authority="NONE",
        )


def prepare_policy(
    development,
    scope,
    *,
    candidate,
    baseline,
    corpus,
    target_experiment,
    edit_authority,
    compiler_library_selection,
    compiler_pass_selection,
    checked_source_diagnostic,
    contract_root,
    work_root,
    receipt_path=None,
):
    from . import capsule_ordinary_scope as C
    from .public_source_diagnostic import prepare_public_source_diagnostic

    if (
        type(development) is not CapsuleSourceDevelopment
        or development.corpus is not corpus
        or development.target is not target_experiment
    ):
        raise StageGateError("capsule source policy substituted its exact original development inputs")
    C.verify(scope)
    return prepare_public_source_diagnostic(
        candidate=candidate,
        baseline=baseline,
        corpus=corpus,
        target_experiment=target_experiment,
        edit_authority=edit_authority,
        service=development.service,
        compiler_library_selection=compiler_library_selection,
        compiler_pass_selection=compiler_pass_selection,
        checked_source_diagnostic=checked_source_diagnostic,
        contract_root=contract_root,
        work_root=work_root,
        native_exploration=development.native_exploration,
        diagnostic_pair_check=development.diagnostic_pair_check,
        capsule_development_scope=scope,
        receipt_path=receipt_path,
    )


def broker_execute(policy, request, action_name, rendered, call_index, timeout_s, started):
    from . import capsule_ordinary_scope as C
    from . import component_workflow as W
    from . import queued_search_actions as Q

    C.binding(policy)
    if action_name not in {a.name for a in C.actions()}:
        raise StageGateError("capsule development cannot dispatch RTL or measured feedback")
    if action_name == W.EVALUATION_SUBMIT_ACTION:
        C.require_kind(policy, rendered.get("kind"))
    result = Q.execute(policy, action_name, rendered, timeout_s, started)
    C.note_submission(policy, action_name, result[0])
    return result


def discover(development, target, families, capsules):
    from . import corpus as C

    if development is None:
        return C.discover_performance_corpus(target, families=families, capsules=capsules)
    if type(development) is not CapsuleSourceDevelopment or development.target is not target:
        raise StageGateError("development corpus differs from its exact target owner")
    development.corpus.verify()
    return development.corpus


def freeze(corpus, destination):
    from . import corpus as C

    if type(corpus) is PDC.FrozenPublicDiagnosticCorpus:
        corpus.verify()
        return corpus
    return C.freeze_performance_corpus(corpus, destination)


def formal_claim(corpus, replicates):
    from . import candidate_record as R

    if type(corpus) is PDC.FrozenPublicDiagnosticCorpus:
        return {"scope": "unscored_public_source_diagnostics", "formal_cohort_admitted": False}, 0
    claim = R.prepare_formal_claim(corpus.capsules, replicates)
    return claim, len(R.preflight_cohort(claim))


def prepare_feedback(development, checked_source, passes, **inputs):
    from . import development_feedback as DF

    if development is None:
        return DF.prepare_development_feedback(**inputs)
    from . import ordinary_checked_source as CHECKED
    from .public_source_diagnostic import PublicDiagnosticExecution

    record = CHECKED.record(checked_source, passes)
    return PublicDiagnosticExecution(development.service, inputs["work_root"], Path(record["rtl_facts"]["path"]))


def feedback_binding(feedback):
    from .public_source_diagnostic import PublicDiagnosticExecution

    if type(feedback) is PublicDiagnosticExecution:
        return {
            "certificate": None,
            "rtl_identity": {"path": str(feedback.rtl_facts_path), "sha256": sha256_file(feedback.rtl_facts_path)},
        }
    return {"certificate": feedback.certificate.to_dict(), "rtl_identity": dict(feedback.rtl_identity)}


def validate_preparation(development, selection, preparation, library, diagnostic, search, contract, checked):
    from . import compiler_pass_preparation as P

    if development is None or preparation is not None:
        return P.validate_stage_selection(selection, preparation, library, diagnostic, search, contract)
    from merlin.targetgen.checked_source_model_diagnostic import MECHANISM, CheckedSourceModelDiagnostic
    from merlin.targetgen.compiler_pass_selection import CompilerPassSelection

    if (
        type(development) is not CapsuleSourceDevelopment
        or type(selection) is not CompilerPassSelection
        or type(checked) is not CheckedSourceModelDiagnostic
        or checked._passes is not selection
        or selection.mechanisms != (MECHANISM,)
        or library is None
        or selection.binding.library != library.contract.sha256
        or search is None
        or contract is None
    ):
        raise StageGateError(
            "unprepared capsule sources require only their exact live source-diagnostic pass selection"
        )
    from . import ordinary_checked_source as CHECKED

    CHECKED.record(checked, selection)
