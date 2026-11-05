"""Construct feedback from independently qualified support and the fresh seed."""

from pathlib import Path

from .component_baseline import ComponentBaselineAdmission
from .component_runtime import require_independent_runtime
from .contracts import StageGateError


def _inputs(independent_runtime, baseline_admission, role):
    if type(baseline_admission) is not ComponentBaselineAdmission:
        raise StageGateError("independent feedback requires the fresh Phase 1 baseline admission")
    baseline_admission.verify()
    runtime = require_independent_runtime(
        independent_runtime, required_roles=(role,), target_descriptor=baseline_admission.target_descriptor
    )
    verify_binding = getattr(runtime.qualification, "verify_component_binding", None)
    if not callable(verify_binding):
        raise StageGateError("independent feedback requires exact compiler/cohort measurement qualification")
    verify_binding(baseline_admission)
    return runtime


def build_independent_component_analytical_provider(*, independent_runtime, baseline_admission, dependencies, **kwargs):
    """Use the evaluated independent observer; no backend/plugin factory lookup."""
    from .component_analytical import build_component_analytical_provider

    runtime = _inputs(independent_runtime, baseline_admission, "feature_provider")
    for key in ("feature_provider", "baseline", "corpus", "target_descriptor"):
        if key in kwargs:
            raise StageGateError("independent component provider inputs are fixed by fresh baseline admission")
    pins = {Path(path): digest for path, digest in runtime.source_pins}
    for path, digest in dependencies.items():
        path = Path(path)
        if path in pins and pins[path] != digest:
            raise StageGateError("independent component dependency conflicts with evaluated runtime source")
        pins[path] = digest
    return build_component_analytical_provider(
        baseline=baseline_admission.baseline,
        corpus=baseline_admission.corpus,
        target_descriptor=baseline_admission.target_descriptor,
        feature_provider=runtime.services.feature_provider,
        baseline_admission=baseline_admission,
        independent_runtime=runtime,
        dependencies=pins,
        **kwargs,
    )


def build_independent_component_cca_provider(*, independent_runtime, baseline_admission):
    """CCA compares the fresh seed with its candidate on identical components."""
    from .component_cca import ComponentCCAProvider
    from .component_workflow import _callable_source

    runtime = _inputs(independent_runtime, baseline_admission, "cca_provider")
    callback = runtime.services.cca_provider
    owner, digest = _callable_source(callback)
    provider = ComponentCCAProvider(
        callback,
        owner,
        digest,
        baseline_admission.baseline,
        baseline_admission.baseline_sha256,
        baseline_admission,
        runtime,
    )
    provider.validate()
    return provider


def build_independent_component_rtl_provider(
    *,
    independent_runtime,
    baseline_admission,
    target_experiment,
    certificate_path,
    certificate_sha256,
    rtl_facts_path,
    rtl_facts_sha256,
    work_root,
    tuning_call_budget=None,
):
    """Select the existing certified evaluator and its originally qualified executor.

    Source membership does not qualify timing. The live measurement issuer must
    already grant ``rtl_executor`` for this fresh compiler/cohort, and the strict
    existing certificate gate must admit each exact development workload.
    """
    from merlin.targetgen.target_experiment import TargetExperiment

    from .component_workflow import _callable_source
    from .contracts import sha256_file
    from .development_feedback import DevelopmentGsimFeedback, prepare_development_feedback

    runtime = _inputs(independent_runtime, baseline_admission, "rtl_executor")
    if (
        type(target_experiment) is not TargetExperiment
        or target_experiment.path != baseline_admission.target_descriptor
        or target_experiment.descriptor_sha256 != baseline_admission.target_sha256
        or sha256_file(target_experiment.path) != baseline_admission.target_sha256
    ):
        raise StageGateError("component RTL requires the exact fresh target descriptor")
    pins = dict(runtime.source_pins)

    def original_file(path, digest):
        path = Path(path)
        if (
            not path.is_absolute()
            or path.resolve() != path
            or any(member.is_symlink() for member in (path, *path.parents))
            or not path.is_file()
            or pins.get(path) != digest
            or sha256_file(path) != digest
        ):
            raise StageGateError("component RTL certificate/facts must be pinned original runtime sources")
        return path

    certificate_path = original_file(certificate_path, certificate_sha256)
    rtl_facts_path = original_file(rtl_facts_path, rtl_facts_sha256)
    callback = runtime.services.rtl_executor
    owner, digest = _callable_source(callback)
    original_file(owner, digest)
    work_root = Path(work_root)
    if (
        not work_root.is_absolute()
        or work_root.resolve() != work_root
        or any(member.is_symlink() for member in (work_root, *work_root.parents))
        or work_root.exists()
    ):
        raise StageGateError("component RTL requires a fresh canonical private work root")
    public_roots = (
        baseline_admission.baseline,
        baseline_admission.corpus.root,
        baseline_admission.qualification.contract_root,
    )
    if any(work_root.is_relative_to(root) or root.is_relative_to(work_root) for root in public_roots):
        raise StageGateError("component RTL work root overlaps its fresh input owners")
    if tuning_call_budget is not None and (type(tuning_call_budget) is not int or tuning_call_budget < 1):
        raise StageGateError("component RTL tuning call budget must be an explicit positive count")
    provider = prepare_development_feedback(
        contract_root=baseline_admission.qualification.contract_root,
        certificate_path=certificate_path,
        certificate_sha256=certificate_sha256,
        rtl_facts_path=rtl_facts_path,
        corpus=baseline_admission.corpus,
        baseline=baseline_admission.baseline,
        baseline_sha256=baseline_admission.baseline_sha256,
        target_experiment=target_experiment,
        work_root=work_root,
        tuning_call_budget=tuning_call_budget,
    )
    if type(provider) is not DevelopmentGsimFeedback:
        raise StageGateError("component RTL requires the original certified development evaluator")
    for identity in provider.certificate.pins.values():
        original_file(identity["path"], identity["sha256"])
    if set(provider.rtl_identity) != {"rtl_facts", "circt_core_hw"} or provider.rtl_identity["rtl_facts"] != {
        "path": str(rtl_facts_path),
        "sha256": rtl_facts_sha256,
    }:
        raise StageGateError("component RTL facts selection differs from original runtime sources")
    for identity in provider.rtl_identity.values():
        if not isinstance(identity, dict) or set(identity) != {"path", "sha256"}:
            raise StageGateError("component RTL lacks exact original elaborated source membership")
        original_file(identity["path"], identity["sha256"])
    provider.executor = callback
    baseline_admission.verify(
        corpus=provider.corpus, baseline=provider.baseline, target_descriptor=provider.target_experiment.path
    )
    runtime.verify(required_roles=("rtl_executor",))
    original_file(certificate_path, certificate_sha256)
    original_file(rtl_facts_path, rtl_facts_sha256)
    return provider
