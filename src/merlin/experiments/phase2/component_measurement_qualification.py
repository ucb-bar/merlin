"""Execute independent performance controls and held groups before tool admission.

Correctness authority does not grant timer, feature, CCA or RTL measurement roles.
This controller runs their pinned independent methods, joins complete ordinary
source/output witnesses, and validates both cold and warm estimates against actual
matched measurements. Unavailable physical support persists a private refusal.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from weakref import WeakKeyDictionary

from merlin.perf.component_applicability import ComponentApplicabilityCoordinates, ComponentApplicabilityDomain
from merlin.perf.component_cost import ComponentCostScope
from merlin.perf.component_screen import qualify_component_screen, validate_component_screen_report
from merlin.perf.fast_estimate_validation import Observation
from merlin.xdsl_dialects.lowering.global_plan import CycleInterval

from .component_applicability import (
    applicability_observation,
    qualify_required_applicability_strata,
    verify_domain_scope,
)
from .component_baseline import ComponentBaselineAdmission
from .component_measurement_evidence import calibration_evidence, matched_measurement
from .component_measurement_protocol import (
    MEASUREMENT_CASES,
    MEASUREMENT_ROLES,
    ComponentMeasurementControl,
    ComponentScreenSample,
    control_binding,
    measurement_binding,
    sample_binding,
)
from .component_runtime_authority import IndependentComponentRuntime, _callback_identity
from .component_runtime_qualification import (
    IndependentRuntimeQualification,
    RuntimeControlRefusal,
    _bounded_arguments,
    _evidence_file,
    _invocations,
)
from .component_variants import ComponentVariantSnapshot
from .contracts import StageGateError, document_sha256, mapping_file, sha256_file, write_json

_ISSUED = WeakKeyDictionary()
_METHODS = (
    "prepare_measurement_control", "verify_measurement_control", "prepare_screen_samples",
    "verify_screen_sample", "observe_measurement", "verify_measurement", "verify_measurement_calibration",
    "prepare_applicability_domain", "verify_applicability_domain", "observe_component_applicability",
    "verify_component_applicability",
)


def _controller_sources():
    owners = (qualify_independent_component_measurement, matched_measurement, calibration_evidence,
              control_binding, sample_binding, ComponentVariantSnapshot.verify, IndependentComponentRuntime.verify,
              IndependentRuntimeQualification.verify, qualify_component_screen, validate_component_screen_report)
    owners += (verify_domain_scope, applicability_observation, qualify_required_applicability_strata)
    owners += (ComponentApplicabilityDomain.verify, ComponentApplicabilityCoordinates.verify)
    return {_callback_identity(owner)[2]: _callback_identity(owner)[3] for owner in owners}


class _CalibratedPrediction:
    """Only already calibrated costs; held cycle labels cannot change coefficients."""

    def predict(self, features, *, domain_sha256):
        return CycleInterval(features["/calibrated_lo"], features["/calibrated_hi"],
                             provenance=("independent complete-cost calibration",))


def _identity_sources(runtime):
    context, pins = runtime.qualification.context, dict(runtime.source_pins)
    entries = []
    for role in MEASUREMENT_ROLES:
        callback = getattr(runtime.services, role)
        if not callable(callback):
            raise StageGateError("independent physical measurement role is unavailable: " + role)
        entries.append(("service", role, _callback_identity(callback)))
    for name in _METHODS:
        callback = getattr(context, name, None)
        if not callable(callback):
            raise StageGateError("independent physical measurement producer is unavailable: " + name)
        entries.append(("context", name, _callback_identity(callback)))
    for _owner, _name, identity in entries:
        if pins.get(identity[2]) != identity[3]:
            raise StageGateError("independent measurement method is outside evaluated source membership")
    return tuple(entries)


def _role_calls(runtime, fixture, *, started, timeout_s):
    if not isinstance(fixture.arguments, dict) or set(fixture.arguments) != set(MEASUREMENT_ROLES):
        raise StageGateError("independent measurement omitted its exact feature/CCA/RTL invoked roster")
    outputs = {}
    for role in MEASUREMENT_ROLES:
        outputs[role] = getattr(runtime.services, role)(**_bounded_arguments(
            fixture.arguments[role], "timeout_s", started=started, timeout_s=timeout_s,
        ))
    return outputs


def _private_fixture(fixture, root):
    if (fixture.evidence_root.resolve() != fixture.evidence_root or fixture.evidence_root.is_symlink()
        or not fixture.evidence_root.is_relative_to(root) or not fixture.required_invocation_stages):
        raise StageGateError("independent measurement lacks private normal invocation ownership")


def _samples(context, root, admission, variants, domain):
    samples = context.prepare_screen_samples(root)
    if (not isinstance(samples, tuple) or not samples
        or any(type(sample) is not ComponentScreenSample for sample in samples)):
        raise StageGateError("independent measurement lacks its predeclared actual held sample roster")
    expected = {(variant.sha256, member.source_sha256) for variant in variants for member in admission.corpus.capsules}
    observed = set()
    strata = {row.id: row for row in domain.strata}
    for sample in samples:
        sample.variant.verify()
        identity = sample.variant.sha256, sample.member.source_sha256
        if (identity in observed or identity not in expected or sample.variant not in variants
            or sample.member not in admission.corpus.capsules or sample.group != sample.member.family
            or sample.applicability_stratum not in strata
            or sample.group not in strata[sample.applicability_stratum].held_groups
            or sample.id != document_sha256(list(identity))):
            raise StageGateError("held measurements leak groups or select another fresh compiler/source cohort")
        observed.add(identity)
        _private_fixture(sample, root)
        context.verify_screen_sample(sample)
    if observed != expected:
        raise StageGateError("independent held measurement omitted a predeclared compiler/workload variant")
    return samples


@dataclass(frozen=True, eq=False)
class IndependentMeasurementQualification:
    functional_runtime: IndependentComponentRuntime
    baseline_admission: ComponentBaselineAdmission
    scope: ComponentCostScope
    calibration_adapter: Path
    calibration_sha256: str
    variants: tuple[ComponentVariantSnapshot, ...]
    context: object
    services: object
    hardware_intake: object
    target_descriptor: Path
    source_pins: tuple[tuple[Path, str], ...]
    qualified_roles: tuple[str, ...]
    receipt: Path
    receipt_sha256: str
    controls: tuple[ComponentMeasurementControl, ...]
    control_sha256s: tuple[str, ...]
    samples: tuple[ComponentScreenSample, ...]
    sample_sha256s: tuple[str, ...]
    measurements: tuple
    measurement_sha256s: tuple[str, ...]
    screen_reports: tuple[tuple[str, Path, str], ...]
    callbacks: tuple
    applicability_domain: object
    applicability_sha256: str
    stratum_reports: tuple[tuple[str, str, Path, str], ...]

    def verify(self):
        issued = _ISSUED.get(self)
        if issued is None or issued != self.receipt_sha256:
            raise StageGateError("measurement roles require live evaluated controls and matched held measurements")
        self.functional_runtime.verify(required_roles=("grade", "stage_verifier"))
        self.baseline_admission.verify()
        if (self.context is not self.functional_runtime.qualification.context
            or self.services is not self.functional_runtime.services
            or self.hardware_intake is not self.functional_runtime.hardware_intake
            or self.target_descriptor != self.baseline_admission.target_descriptor
            or self.callbacks != _identity_sources(self.functional_runtime)):
            raise StageGateError("independent measurement context/source selection changed")
        for variant in self.variants:
            variant.verify()
        for path, digest in (*self.source_pins, (self.receipt, self.receipt_sha256)):
            if path.resolve() != path or path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
                raise StageGateError("independent measurement source/control/sample evidence changed")
        for control, digest in zip(self.controls, self.control_sha256s, strict=True):
            self.context.verify_measurement_control(control)
            if control_binding(control) != digest:
                raise StageGateError("independent measurement control binding changed")
            _invocations(control)
        for sample, digest in zip(self.samples, self.sample_sha256s, strict=True):
            self.context.verify_screen_sample(sample)
            if sample_binding(sample) != digest:
                raise StageGateError("independent held measurement source/group binding changed")
            _invocations(sample)
        for measurement, digest in zip(self.measurements, self.measurement_sha256s, strict=True):
            if measurement_binding(measurement) != digest:
                raise StageGateError("independent measured feature/source/timer binding changed")
            self.context.verify_measurement(measurement)
            self.context.verify_component_applicability(measurement.applicability, measurement.features)
            if self.applicability_domain.lookup(measurement.applicability.coordinates)["status"] != "IN_DOMAIN":
                raise StageGateError("independent measurement escaped its qualified joint applicability")
        if (verify_domain_scope(self.applicability_domain, runtime=self.functional_runtime, scope=self.scope)
            != self.applicability_sha256):
            raise StageGateError("independent semantic applicability domain changed")
        expected_strata = {(row.id, regime) for row in self.applicability_domain.strata for regime in ("cold", "warm")}
        if (len(self.stratum_reports) != len(expected_strata)
            or {(ident, regime) for ident, regime, _path, _sha in self.stratum_reports} != expected_strata):
            raise StageGateError("independent applicability omitted required cold/warm held strata")
        for ident, regime, path, _digest in self.stratum_reports:
            report = validate_component_screen_report(mapping_file(path))
            row = next(row for row in self.applicability_domain.strata if row.id == ident)
            if (not report["exposable"] or report.get("stratum") != ident or report.get("regime") != regime
                or report.get("domain_sha256") != self.applicability_sha256
                or report.get("applicability_cells") != sorted(point.sha256 for point in row.cells)
                or report.get("calibration_sha256") != self.calibration_sha256):
                raise StageGateError("independent applicability stratum no longer meets held acceptance")
        calibration, _pins = calibration_evidence(runtime=self.functional_runtime,
                                                  adapter=self.calibration_adapter, samples=self.samples)
        if document_sha256(calibration) != self.calibration_sha256:
            raise StageGateError("independent measurement calibration changed after held qualification")
        for regime, path, _digest in self.screen_reports:
            report = validate_component_screen_report(mapping_file(path))
            if (report.get("exposable") is not True or report.get("regime") != regime
                or report.get("calibration_sha256") != self.calibration_sha256):
                raise StageGateError("independent cold/warm screen no longer meets held acceptance")
        document = mapping_file(self.receipt)
        expected_reports = [[ident, regime, sha] for ident, regime, _path, sha in self.stratum_reports]
        if (document.get("schema") != "merlin.independent_measurement_qualification.v2"
            or document.get("status") != "qualified" or document.get("cases") != list(MEASUREMENT_CASES)
            or document.get("applicability_sha256") != self.applicability_sha256
            or document.get("stratum_reports") != expected_reports
            or len(self.controls) != len(MEASUREMENT_CASES)
            or set(regime for regime, *_ in self.screen_reports) != {"cold", "warm"}
            or self.qualified_roles != ("grade", "stage_verifier", *MEASUREMENT_ROLES)):
            raise StageGateError("independent measurement complete role/control/scope roster changed")
        return self.receipt_sha256

    def verify_feedback_binding(self, *, baseline_admission, scope, calibration_adapter, qualification, objective):
        self.verify()
        if (baseline_admission is not self.baseline_admission or scope != self.scope
            or calibration_adapter != self.calibration_adapter
            or (objective, qualification) not in {(regime, path) for regime, path, _sha in self.screen_reports}):
            raise StageGateError("feedback selected different independent compiler/cohort/calibration/timer authority")

    def verify_component_binding(self, baseline_admission):
        self.verify()
        if baseline_admission is not self.baseline_admission:
            raise StageGateError("measurement service selected another fresh compiler/source cohort")

    def observe_applicability(self, features, workspace):
        self.verify()
        value = self.context.observe_component_applicability(features, workspace)
        lookup, _pins = applicability_observation(
            observation=value, features=features, domain=self.applicability_domain,
            runtime=self.functional_runtime, owner=workspace,
        )
        return value.coordinates, lookup


def qualify_independent_component_measurement(*, functional_runtime, baseline_admission, scope,
                                            calibration_adapter, variants, evidence_root, timeout_s=600):
    """Run physical controls and both preregistered held screens; never mint from JSON.

    All measured variants originate in the fresh compiler's allowed edit scope.
    Calibration provenance must be independently rederived and disjoint from the
    held sample labels. Every missing producer, wrong scope, crash, failed control,
    or deficient ranking/error/coverage result leaves all performance roles unissued.
    """
    if (type(functional_runtime) is not IndependentComponentRuntime
        or type(functional_runtime.qualification) is not IndependentRuntimeQualification
        or type(baseline_admission) is not ComponentBaselineAdmission or type(scope) is not ComponentCostScope
        or not isinstance(variants, tuple) or not variants
        or any(type(variant) is not ComponentVariantSnapshot for variant in variants)
        or type(timeout_s) is not int or not 0 < timeout_s <= 600):
        raise StageGateError("measurement qualification requires live functional/fresh source, scope and variants")
    functional_runtime.verify(required_roles=("grade", "stage_verifier"))
    baseline_admission.verify()
    if functional_runtime.target_descriptor != baseline_admission.target_descriptor:
        raise StageGateError("measurement qualification selected another hardware/runtime target")
    for variant in variants:
        variant.verify()
        if variant.execution.baseline_admission is not baseline_admission:
            raise StageGateError("measurement variant originated from another Phase 1 compiler")
    if len({variant.sha256 for variant in variants}) != len(variants):
        raise StageGateError("independent measurement compiler variants must be distinct")
    root, adapter = Path(evidence_root), Path(calibration_adapter)
    if (root.exists() or root.is_symlink() or root.resolve() != root
        or adapter.resolve() != adapter or adapter.is_symlink()):
        raise StageGateError("measurement qualification requires fresh canonical private output and calibration input")
    if any(root.is_relative_to(path) or path.is_relative_to(root) for path in (
        baseline_admission.baseline, baseline_admission.corpus.root, *(variant.compiler for variant in variants),
    )):
        raise StageGateError("independent measurement output overlaps compiler/source authority")
    root.mkdir(parents=True, mode=0o700)
    context = functional_runtime.qualification.context
    started, controls, samples, measurements, rows, callbacks, reports = time.monotonic(), [], (), [], [], (), []
    control_shas, sample_shas, measurement_shas, evidence = [], [], [], dict(functional_runtime.source_pins)
    calibration, calibration_sha = None, ""
    domain, domain_sha, stratum_reports, stratum_rows = None, "", [], []
    try:
        callbacks = _identity_sources(functional_runtime)
        domain = context.prepare_applicability_domain()
        domain_sha = verify_domain_scope(domain, runtime=functional_runtime, scope=scope)
        samples = _samples(context, root / "held", baseline_admission, variants, domain)
        sample_shas = [sample_binding(sample) for sample in samples]
        calibration, pins = calibration_evidence(runtime=functional_runtime, adapter=adapter, samples=samples)
        evidence.update(pins)
        calibration_sha = document_sha256(calibration)
        for name in MEASUREMENT_CASES:
            mechanism, _, direction = name.partition(".")
            try:
                control = context.prepare_measurement_control(name, root / "controls" / name)
                if type(control) is not ComponentMeasurementControl or control.case_id != name:
                    raise StageGateError("independent measurement prepared a different control")
                _private_fixture(control, root)
                context.verify_measurement_control(control)
                digest = control_binding(control)
                try:
                    outputs = _role_calls(functional_runtime, control, started=started, timeout_s=timeout_s)
                    observed = context.observe_measurement(control, outputs)
                    if direction == "negative":
                        raise StageGateError("independent measurement accepted an actual defective control")
                    if not isinstance(observed, tuple) or not observed:
                        raise StageGateError("independent positive control omitted complete measured execution")
                    for observation in observed:
                        _totals, pins = matched_measurement(
                            observation=observation, owner=control.evidence_root, runtime=functional_runtime,
                            admission=baseline_admission, scope=scope, calibration=calibration, variants=variants,
                            applicability_domain=domain,
                        )
                        measurements.append(observation)
                        measurement_shas.append(measurement_binding(observation))
                        evidence.update(pins)
                except RuntimeControlRefusal as refusal:
                    if (direction != "negative" or refusal.case_id != name or refusal.mechanism != mechanism
                        or not isinstance(refusal.evidence_files, tuple) or not refusal.evidence_files):
                        raise StageGateError("measurement refusal differs from actual predeclared control") from refusal
                    for path, sha in refusal.evidence_files:
                        evidence[_evidence_file(path, sha, control.evidence_root)] = sha
                context.verify_measurement_control(control)
                evidence.update(_invocations(control))
                if control_binding(control) != digest:
                    raise StageGateError("independent measurement control changed during execution")
                controls.append(control)
                control_shas.append(digest)
                rows.append({"case_id": name, "status": "pass"})
            except Exception as error:  # noqa: BLE001 - any incomplete physical control refuses
                rows.append({"case_id": name, "status": "refused", "reason": type(error).__name__,
                             "detail": str(error)[:512]})
        observed_rows = {"cold": [], "warm": []}
        for sample in samples:
            context.verify_screen_sample(sample)
            outputs = _role_calls(functional_runtime, sample, started=started, timeout_s=timeout_s)
            observed = context.observe_measurement(sample, outputs)
            if not isinstance(observed, tuple) or len(observed) != 1:
                raise StageGateError("held measurement requires one exact allowed compiler/workload execution")
            observation = observed[0]
            totals, pins = matched_measurement(
                observation=observation, owner=sample.evidence_root, runtime=functional_runtime,
                admission=baseline_admission, scope=scope, calibration=calibration, variants=variants, sample=sample,
                applicability_domain=domain,
            )
            measurements.append(observation)
            measurement_shas.append(measurement_binding(observation))
            evidence.update(pins)
            evidence.update(_invocations(sample))
            for regime in ("cold", "warm"):
                value = Observation(
                    observation.features.executable_sha256, sample.member.source_sha256, sample.group,
                    observation.features.domain_sha256,
                    {"/calibrated_lo": totals[regime].lo, "/calibrated_hi": totals[regime].hi},
                    getattr(observation, regime + "_cycles"), tuple(pins.values()),
                )
                observed_rows[regime].append(value)
                stratum_rows.append((sample.applicability_stratum, observation.applicability.coordinates.sha256,
                                     regime, value))
            context.verify_screen_sample(sample)
        for regime, observed in observed_rows.items():
            report = qualify_component_screen(observed, lambda _training: _CalibratedPrediction(),
                                               calibration_sha256=calibration_sha)
            report["regime"] = regime
            validate_component_screen_report(report)
            path = root / (regime + "_held_screen.json")
            write_json(path, report)
            path.chmod(0o400)
            reports.append((regime, path, sha256_file(path)))
            evidence[path] = sha256_file(path)
        strata = qualify_required_applicability_strata(domain, stratum_rows, calibration_sha256=calibration_sha)
        for (ident, regime), report in strata.items():
            path = root / (document_sha256([ident, regime]) + "_held_stratum.json")
            write_json(path, report)
            path.chmod(0o400)
            stratum_reports.append((ident, regime, path, sha256_file(path)))
            evidence[path] = sha256_file(path)
    except Exception as error:  # noqa: BLE001 - missing independent producers are private UNAVAILABLE
        rows.append({"case_id": "preparation_or_held_measurement", "status": "unavailable",
                     "reason": type(error).__name__, "detail": str(error)[:512]})
    context.verify()
    functional_runtime.verify(required_roles=("grade", "stage_verifier"))
    passed = (len(controls) == len(MEASUREMENT_CASES) and len(reports) == 2
              and domain is not None and len(stratum_reports) == 2 * len(domain.strata)
              and len(rows) == len(MEASUREMENT_CASES) and all(row["status"] == "pass" for row in rows)
              and time.monotonic() - started <= timeout_s
              and all(measurement_binding(value) == digest
                      for value, digest in zip(measurements, measurement_shas, strict=True))
              and all(mapping_file(path).get("exposable") is True for _regime, path, _sha in reports))
    roles = ("grade", "stage_verifier", *MEASUREMENT_ROLES) if passed else ()
    receipt = root / "qualification.json"
    write_json(receipt, {
        "schema": "merlin.independent_measurement_qualification.v2", "status": "qualified" if passed else "unavailable",
        "cases": list(MEASUREMENT_CASES), "controls": rows, "qualified_roles": list(roles),
        "baseline_admission_sha256": baseline_admission.sha256, "scope_sha256": scope.sha256,
        "calibration_sha256": calibration_sha, "screen_reports": [(regime, sha) for regime, _path, sha in reports],
        "applicability_sha256": domain_sha,
        "stratum_reports": [(ident, regime, sha) for ident, regime, _path, sha in stratum_reports],
        "elapsed_seconds": time.monotonic() - started, "scope": "screening only; no final validation authority",
    })
    receipt.chmod(0o400)
    evidence.update(_controller_sources())
    qualification = IndependentMeasurementQualification(
        functional_runtime, baseline_admission, scope, adapter, calibration_sha, variants, context,
        functional_runtime.services, functional_runtime.hardware_intake, functional_runtime.target_descriptor,
        tuple(sorted(evidence.items(), key=lambda row: str(row[0]))), roles, receipt, sha256_file(receipt),
        tuple(controls), tuple(control_shas), samples, tuple(sample_shas),
        tuple(measurements), tuple(measurement_shas),
        tuple(reports), callbacks, domain, domain_sha, tuple(stratum_reports),
    )
    if passed:
        _ISSUED[qualification] = qualification.receipt_sha256
        qualification.verify()
    return qualification
