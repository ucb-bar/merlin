"""Join semantic applicability to independent measured execution and held strata.

Arithmetic and these data objects cannot grant physical roles. The measurement
qualifier executes its fixed source/output/effect/timer controls and owns issuance.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from merlin.common import invocation_record
from merlin.perf.component_applicability import ComponentApplicabilityCoordinates, ComponentApplicabilityDomain
from merlin.perf.component_screen import qualify_component_screen, validate_component_screen_report

from .component_runtime_qualification import _evidence_file
from .contracts import StageGateError, mapping_file, sha256_file


@dataclass(frozen=True)
class ComponentApplicabilityObservation:
    coordinates: ComponentApplicabilityCoordinates
    product: Path
    product_sha256: str
    invocations: tuple[Path, ...]


def verify_domain_scope(domain, *, runtime, scope):
    if type(domain) is not ComponentApplicabilityDomain:
        raise StageGateError("component measurement requires an independently frozen applicability domain")
    try:
        domain.verify()
    except ValueError as error:
        raise StageGateError("component applicability declaration is incomplete") from error
    runtime.qualification.context.verify_applicability_domain(domain)
    point = domain.strata[0].cells[0]
    if (point.hardware_sha256, point.timer_sha256, point.accuracy_sha256, point.input_policy_sha256) != (
        runtime.hardware_intake.sha256, scope.timer_sha256, scope.accuracy_sha256, scope.input_policy_sha256,
    ):
        raise StageGateError("component applicability selected different HW/numeric/input/timer scope")
    return domain.sha256


def applicability_observation(*, observation, features, domain, runtime, owner):
    """Reopen an actual derived product; source/HW semantics remain context-owned."""
    if type(observation) is not ComponentApplicabilityObservation:
        raise StageGateError("component features omit actual independently derived applicability coordinates")
    try:
        lookup = domain.lookup(observation.coordinates)
    except ValueError as error:
        raise StageGateError("component applicability coordinates violate resource/shape semantics") from error
    runtime.qualification.context.verify_component_applicability(observation, features)
    if features.domain_sha256 != domain.sha256:
        raise StageGateError("component feature domain differs from its complete semantic applicability")
    product = _evidence_file(observation.product, observation.product_sha256, owner)
    expected = {
        "schema": "merlin.component_applicability_observation.v1", "domain_sha256": domain.sha256,
        "coordinates": observation.coordinates.to_dict(), "compiler_sha256": features.compiler_sha256,
        "member_sha256": features.member_sha256, "executable_sha256": features.executable_sha256,
        "inputs_sha256": features.inputs_sha256, "dependencies_sha256": features.dependencies_sha256,
    }
    # Normal JSON encoding converts coordinate tuples to lists.
    if mapping_file(product) != json.loads(json.dumps(expected)):
        raise StageGateError("component applicability product differs from actual source/ELF/input/unit coordinates")
    if not isinstance(observation.invocations, tuple) or not observation.invocations:
        raise StageGateError("component applicability has no actual independent derivation invocations")
    pins, records = {product: observation.product_sha256}, []
    for value in observation.invocations:
        path = _evidence_file(value, sha256_file(value), owner)
        records.append(invocation_record.verify(path))
        pins[path] = sha256_file(path)
    producers = [record for record in records if any(
        pin["path"] == str(product) and pin["sha256"] == observation.product_sha256 for pin in record["outputs"]
    )]
    if not any(pin["sha256"] == features.executable_sha256 for record in producers for pin in record["inputs"]):
        raise StageGateError("component applicability lacks actual exact executed-ELF derivation")
    return lookup, pins


def qualify_required_applicability_strata(domain, rows, *, calibration_sha256):
    """Statistical arithmetic on matched full-output/timer observations, not issuance."""
    domain.verify()
    strata = {row.id: row for row in domain.strata}
    grouped = {(ident, regime): [] for ident in strata for regime in ("cold", "warm")}
    seen_cells = {key: set() for key in grouped}
    cell_groups = {key: set() for key in grouped}
    for ident, cell, regime, observation in rows:
        if (ident not in strata or regime not in ("cold", "warm")
            or cell not in {point.sha256 for point in strata[ident].cells}
            or observation.group not in strata[ident].held_groups
            or observation.group in domain.calibration_groups or observation.domain != domain.sha256):
            raise StageGateError("applicability held observation leaks or selects another semantic transfer stratum")
        grouped[ident, regime].append(observation)
        seen_cells[ident, regime].add(cell)
        cell_groups[ident, regime].add((cell, observation.group))
    reports = {}
    from .component_measurement_qualification import _CalibratedPrediction

    for (ident, regime), observed in grouped.items():
        if {row.group for row in observed} != set(strata[ident].held_groups):
            raise StageGateError("applicability omitted a required held stratum/transfer group in " + regime)
        expected_cells = {point.sha256 for point in strata[ident].cells}
        if seen_cells[ident, regime] != expected_cells:
            raise StageGateError("applicability omitted a required joint cell in " + ident + "/" + regime)
        if cell_groups[ident, regime] != {(cell, group) for cell in expected_cells
                                         for group in strata[ident].held_groups}:
            raise StageGateError("applicability omitted held transfer coverage for a required joint cell")
        report = qualify_component_screen(observed, lambda _training: _CalibratedPrediction(),
                                           calibration_sha256=calibration_sha256)
        report.update(stratum=ident, regime=regime, applicability_cells=sorted(expected_cells))
        validate_component_screen_report(report)
        if not report["exposable"]:
            raise StageGateError("applicability stratum failed ranking/error/coverage: " + ident + "/" + regime)
        reports[ident, regime] = report
    return reports
