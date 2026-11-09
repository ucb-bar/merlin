"""Admit only the fresh functional compiler as an optimization baseline.

This capability is private to the experiment controller. Hashes and JSON reports
can identify bytes, but cannot substitute for an issued fresh-author origin and
the ordinary domain qualification. Handwritten final references have no role in
this owner or in component comparison feedback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from merlin.benchharness import hash_tree

from . import corpus as C
from .contracts import StageGateError, document_sha256, sha256_file

_ISSUER = object()


def _origin(qualification, baseline):
    try:
        from merlin_experiments.phase1.component_origin import FreshCompilerOrigin
    except ImportError as error:
        raise StageGateError("fresh Phase 1 compiler origin authority is unavailable") from error
    origin = getattr(qualification, "compiler_origin", None)
    if type(origin) is not FreshCompilerOrigin:
        raise StageGateError("component baseline must originate in the fresh Phase 1 author session")
    origin.verify(candidate=baseline)
    return origin.receipt_sha256


def _development_members(qualification, corpus):
    from merlin_experiments.phase0.component_coverage import verify_report
    from merlin_experiments.phase1.source_inputs import fingerprint

    report = verify_report(qualification.corpus_root)
    rows = {
        member["name"]: member
        for obligation in report["obligations"]
        if obligation["cohort"] == "development"
        for member in obligation["members"]
        if member["state"] == "generated"
    }
    if not rows or not corpus.capsules:
        raise StageGateError("component optimization requires independently generated development members")
    for member in corpus.capsules:
        row = rows.get(member.capsule)
        if row is None or fingerprint(member.source_dir) != row["sha256"]:
            raise StageGateError("component feedback member is outside the qualified development cohort")


@dataclass(frozen=True)
class ComponentBaselineAdmission:
    """One issued fresh compiler and one exact independently generated cohort."""

    baseline: Path
    baseline_sha256: str
    qualification: object
    qualification_sha256: str
    origin_sha256: str
    corpus: C.FrozenPerformanceCorpus
    target_descriptor: Path
    target_sha256: str
    _issuer: object = field(repr=False, compare=False)

    @property
    def sha256(self):
        return document_sha256({
            "schema": "merlin.component_baseline_admission.v1",
            "baseline_sha256": self.baseline_sha256,
            "qualification_sha256": self.qualification_sha256,
            "origin_sha256": self.origin_sha256,
            "corpus_sha256": self.corpus.capsules_sha256,
            "manifest_sha256": self.corpus.manifest_sha256,
            "target_sha256": self.target_sha256,
        })

    def verify(self, *, baseline=None, corpus=None, target_descriptor=None):
        from merlin_experiments.phase1.component_qualification import ComponentQualification

        if self._issuer is not _ISSUER or type(self.qualification) is not ComponentQualification:
            raise StageGateError("component comparison requires an issued fresh Phase 1 baseline admission")
        selected = self.baseline if baseline is None else Path(baseline)
        if selected != self.baseline or selected.is_symlink() or selected.resolve() != selected:
            raise StageGateError("component provider selected a different baseline compiler")
        self.qualification.verify(candidate=selected)
        if self.qualification.receipt_sha256 != self.qualification_sha256:
            raise StageGateError("component baseline qualification selection changed")
        if _origin(self.qualification, selected) != self.origin_sha256:
            raise StageGateError("component baseline fresh author origin changed")
        if str(hash_tree(selected)["sha256"]) != self.baseline_sha256:
            raise StageGateError("component baseline compiler changed after fresh Phase 1 admission")
        C.verify_frozen_performance_corpus(self.corpus)
        _development_members(self.qualification, self.corpus)
        if corpus is not None:
            C.verify_frozen_performance_corpus(corpus)
        if corpus is not None and (
            corpus.root, corpus.manifest_sha256, corpus.capsules_sha256
        ) != (self.corpus.root, self.corpus.manifest_sha256, self.corpus.capsules_sha256):
            raise StageGateError("component provider selected a different independent development cohort")
        descriptor = self.target_descriptor if target_descriptor is None else Path(target_descriptor)
        if descriptor != self.target_descriptor or sha256_file(descriptor) != self.target_sha256:
            raise StageGateError("component provider selected a different target descriptor")
        if self.qualification.target_descriptor != descriptor:
            raise StageGateError("component baseline qualification targets another descriptor")
        return self.sha256


def admit_component_baseline(*, qualification, baseline, corpus, target_descriptor):
    """Issue from live fresh-origin qualification, never from a serialized receipt."""
    from merlin_experiments.phase1.component_qualification import ComponentQualification

    if type(qualification) is not ComponentQualification or type(corpus) is not C.FrozenPerformanceCorpus:
        raise StageGateError("component baseline requires evaluated fresh compiler and exact independent corpus")
    baseline, target_descriptor = Path(baseline), Path(target_descriptor)
    if any(path.is_symlink() or path.resolve() != path for path in (baseline, target_descriptor)):
        raise StageGateError("component baseline authorities require canonical paths without indirection")
    qualification.verify(candidate=baseline)
    origin_sha256 = _origin(qualification, baseline)
    admission = ComponentBaselineAdmission(
        baseline, str(hash_tree(baseline)["sha256"]), qualification, qualification.receipt_sha256,
        origin_sha256, corpus, target_descriptor, sha256_file(target_descriptor), _ISSUER,
    )
    admission.verify()
    return admission


def verify_baseline_admission(admission, *, baseline, corpus, target_descriptor):
    if type(admission) is not ComponentBaselineAdmission:
        raise StageGateError("component feedback requires the fresh Phase 1 baseline admission")
    return admission.verify(baseline=baseline, corpus=corpus, target_descriptor=target_descriptor)


def verify_policy_baselines(policy):
    """Every active comparison service uses one fresh baseline and cohort."""
    providers = (policy.component_analytical, policy.component_rtl, policy.component_cca)
    if not any(provider is not None for provider in providers):
        if policy.services.command_buffer_analysis is not None:
            raise StageGateError("structural comparison requires observed fresh compiler artifacts")
        return
    admission = policy.baseline_admission
    verify_baseline_admission(admission, baseline=admission.baseline if admission is not None else None,
                              corpus=policy.component_corpus, target_descriptor=policy.target_experiment.path)
    analytical, rtl, cca = providers
    if analytical is not None:
        from .component_analytical import ComponentAnalyticalBinding

        binding = analytical.binding
        if type(binding) is not ComponentAnalyticalBinding or binding.baseline_admission is not admission:
            raise StageGateError("component analytical service lacks the admitted fresh baseline")
        admission.verify(baseline=binding.baseline, corpus=binding.corpus,
                         target_descriptor=binding.target_descriptor)
    if cca is not None and cca.baseline_admission is not admission:
        raise StageGateError("component CCA service lacks the admitted fresh baseline")
    for provider in (rtl, cca):
        if provider is not None:
            admission.verify(baseline=provider.baseline, corpus=policy.component_corpus,
                             target_descriptor=policy.target_experiment.path)
            if provider.baseline_sha256 != admission.baseline_sha256:
                raise StageGateError("component comparison provider uses a different compiler baseline")
    require_independent_measurement_service(policy)


def require_independent_measurement_service(policy):
    """Fresh source ownership and independently qualified observation are separate."""
    from .component_runtime import require_independent_runtime

    providers = ((policy.component_analytical, "feature_provider"),
                 (policy.component_cca, "cca_provider"), (policy.component_rtl, "rtl_executor"))
    roles = tuple(role for provider, role in providers if provider is not None)
    runtime = require_independent_runtime(policy.independent_runtime, required_roles=roles,
                                          target_descriptor=policy.target_experiment.path)
    verify_binding = getattr(runtime.qualification, "verify_component_binding", None)
    if not callable(verify_binding):
        raise StageGateError("component feedback lacks independent exact compiler/cohort measurement qualification")
    verify_binding(policy.baseline_admission)
    for provider, role in providers:
        if provider is None:
            continue
        callback = (provider.binding.feature_provider if role == "feature_provider"
                    else provider.evaluate if role == "cca_provider" else provider.executor)
        if callback is not getattr(runtime.services, role):
            raise StageGateError("component feedback callable binding differs from qualified independent runtime")
        if role == "feature_provider" and provider.binding.independent_runtime is not runtime:
            raise StageGateError("component analytical service selected another independent runtime")
        if role == "cca_provider" and provider.independent_runtime is not runtime:
            raise StageGateError("component CCA service selected another independent runtime")
