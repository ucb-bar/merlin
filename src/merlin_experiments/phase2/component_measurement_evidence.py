"""Reopen actual source, executable and complete timer observations."""
from __future__ import annotations

from pathlib import Path

from merlin.common import invocation_record
from merlin.perf.component_cost import COMPLETE_STAGES, ComponentFeatureObservation, complete_component_cost
from merlin_experiments.phase1.component_witness import REQUIRED_EXECUTION_EFFECTS, verify_component_stage_witness

from .component_applicability import applicability_observation
from .component_execution import _member_binding
from .component_measurement_protocol import MatchedComponentMeasurement
from .component_runtime_qualification import _evidence_file, _invocations
from .contracts import StageGateError, mapping_file, sha256_file


def actual_measurement_product(invocations, *, measurement, measurement_sha256, executable_sha256):
    producers = [invocation for invocation in invocations if any(
        pin["path"] == str(measurement) and pin["sha256"] == measurement_sha256
        for pin in invocation["outputs"]
    )]
    if not producers:
        raise StageGateError("independent timer observation was not produced at an actual invoked boundary")
    if not any(pin["sha256"] == executable_sha256
               for invocation in invocations if invocation["kind"] == "subprocess" for pin in invocation["inputs"]):
        raise StageGateError("independent measured executable was not consumed by an actual runtime process")
    if not any(pin["sha256"] == executable_sha256 for producer in producers for pin in producer["inputs"]):
        raise StageGateError("independent timer producer did not consume the exact executed ELF")


def matched_measurement(*, observation, owner, runtime, admission, scope, calibration, variants,
                        applicability_domain, sample=None):
    """Join observed timers to the same complete source/ELF/output execution.

    The independently prepared context must separately rederive the timer's
    physical semantics; these generic checks establish product correspondence.
    """
    if type(observation) is not MatchedComponentMeasurement:
        raise StageGateError("independent measurement requires a complete typed source/HW/timer observation")
    runtime.qualification.context.verify_measurement(observation)
    features = observation.features
    if type(features) is not ComponentFeatureObservation or not features.artifact_files:
        raise StageGateError("independent measurement omits actual emitted/executed feature artifacts")
    lookup, applicability_pins = applicability_observation(
        observation=observation.applicability, features=features, domain=applicability_domain,
        runtime=runtime, owner=owner,
    )
    if lookup["status"] != "IN_DOMAIN":
        raise StageGateError("independent measured execution is outside its frozen semantic applicability")
    if sample is not None and lookup["stratum"] != sample.applicability_stratum:
        raise StageGateError("held measurement selected another required semantic applicability stratum")
    members = [member for member in admission.corpus.capsules if member.source_sha256 == features.member_sha256]
    if len(members) != 1:
        raise StageGateError("independent measurement selected an unqualified source member")
    member = members[0]
    binding = _member_binding(admission, member)
    source_memberships = {variant.compiler_sha256: variant.membership_sha256 for variant in variants}
    from .contracts import exact_tree_record

    source_memberships[admission.baseline_sha256] = exact_tree_record(admission.baseline)["sha256"]
    if (observation.member_binding != binding
        or source_memberships.get(features.compiler_sha256) != observation.candidate_membership_sha256
        or observation.hardware_intake_sha256 != runtime.hardware_intake.sha256
        or observation.scope != scope or features.scope_sha256 != scope.sha256
        or features.target_sha256 != admission.target_sha256
        or features.corpus_sha256 != admission.corpus.capsules_sha256
        or features.functional_status != "PASS" or features.legality_status != "PASS"
        or not set(REQUIRED_EXECUTION_EFFECTS) <= set(observation.required_effects)):
        raise StageGateError("independent measurement differs from fresh source/HW/complete scope authority")
    if sample is not None and (member != sample.member or features.compiler_sha256 != sample.variant.compiler_sha256):
        raise StageGateError("held measurement selected another source/compiler variant")
    evidence = dict(applicability_pins)
    for path, digest in features.artifact_files:
        evidence[_evidence_file(path, digest, owner)] = digest
    witness = verify_component_stage_witness(
        observation.stage_witness, member=binding, candidate_sha256=observation.candidate_membership_sha256,
        target_descriptor_sha256=admission.target_sha256, frontend=observation.frontend,
        capsule_root=member.source_dir, evidence_root=owner, required_effects=observation.required_effects,
    )
    elf = [row for row in observation.stage_witness.stages if row.stage == "elf"]
    if len(elf) != 1 or elf[0].sha256 != features.executable_sha256:
        raise StageGateError("independent measured executable differs from complete ordinary verification")
    for row in observation.stage_witness.stages:
        if row.stage != "source":
            evidence[_evidence_file(row.path, row.sha256, owner)] = row.sha256
    records = _invocations(_InvocationOwner(owner))
    invocations = [invocation_record.verify(path) for path, _digest in records]
    measurement = _evidence_file(observation.measurement_record, observation.measurement_record_sha256, owner)
    expected = {
        "schema": "merlin.independent_component_measurement.v1",
        "hardware_intake_sha256": runtime.hardware_intake.sha256,
        "target_sha256": admission.target_sha256, "scope_sha256": scope.sha256,
        "compiler_sha256": features.compiler_sha256, "member_sha256": member.source_sha256,
        "executable_sha256": features.executable_sha256, "inputs_sha256": features.inputs_sha256,
        "dependencies_sha256": features.dependencies_sha256,
        "cold_cycles": observation.cold_cycles, "warm_cycles": observation.warm_cycles,
    }
    if mapping_file(measurement) != expected:
        raise StageGateError("independent cycle record differs from the actually observed full timer scope")
    if any(type(cycles) is not int or cycles <= 0 for cycles in (observation.cold_cycles, observation.warm_cycles)):
        raise StageGateError("independent complete cold/warm cycle measurements are unavailable")
    actual_measurement_product(invocations, measurement=measurement,
                               measurement_sha256=observation.measurement_record_sha256,
                               executable_sha256=features.executable_sha256)
    for regime in ("cold", "warm"):
        if {stage for region in getattr(features, regime) for stage in region.stages} != set(COMPLETE_STAGES):
            raise StageGateError("independent measurement omits a complete cold/warm execution stage")
    totals, _report = complete_component_cost(features, calibration, scope=scope,
                                             qualified_domains=(features.domain_sha256,),
                                             applicability_domain=applicability_domain,
                                             applicability_coordinates=observation.applicability.coordinates)
    if any(not totals[regime].resolved for regime in ("cold", "warm")):
        raise StageGateError("independent measurement has unpriced complete cold/warm work")
    evidence[measurement] = observation.measurement_record_sha256
    evidence.update(records)
    for row in witness["stages"]:
        if row["stage"] != "source":
            evidence[Path(row["path"])] = row["sha256"]
    return totals, evidence


class _InvocationOwner:
    def __init__(self, evidence_root):
        self.evidence_root = evidence_root
        self.required_invocation_stages = ()


def calibration_evidence(*, runtime, adapter, samples):
    """Require independent physical calibration provenance, not pinned history."""
    from merlin.perf.phase2_calibration_bundle import prepare_phase2_calibration

    pins = dict(runtime.source_pins)
    if adapter not in pins or sha256_file(adapter) != pins[adapter]:
        raise StageGateError("measurement calibration is outside independently prepared membership")
    runtime.qualification.context.verify_measurement_calibration(adapter, samples)
    prepared = prepare_phase2_calibration(adapter)
    if prepared.get("status") != "ready" or not isinstance(prepared.get("calibration"), dict):
        raise StageGateError("independent controlled measurement calibration is unavailable")
    evidence = {adapter: pins[adapter]}
    for row in prepared.get("evidence_files", []):
        path, digest = Path(row["path"]), row["sha256"]
        if pins.get(path) != digest or path.resolve() != path or path.is_symlink() or sha256_file(path) != digest:
            raise StageGateError("measurement calibration evidence is outside independent source/provenance authority")
        evidence[path] = digest
    return prepared["calibration"], evidence
