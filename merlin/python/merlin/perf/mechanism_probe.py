"""Admit separate calibration probes bound to a compiled whole-model plan.

Signatures describe one repeated mechanism, not a reduced model's operator name. The host's
target adapter extracts these facts from both emitted programs. Matching signatures are necessary
for calibration; this module cannot certify the truth of an adapter's extraction. Unknown facts
and stale artifact identities refuse admission. Timing never executes the full graph here.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence

from merlin.xdsl_dialects.lowering.global_plan import CycleInterval, ValueRepresentation

from .activity_schedule import ActivityEvent, schedule_activity
from .execution_policy import (
    ITERATION_MAX_SECONDS, SimulationAdmission, SimulationBudget, WarmComputeReceipt, WarmProfileContract,
    admit_reduced_witness, require_probe_execution,
)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _sha(value: str, name: str) -> None:
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(f"{name} must be a SHA-256 digest of current evidence bytes")


def _known(value: Any) -> None:
    if value is None or (isinstance(value, str) and
                         (not value.strip() or value.lower() in {"unknown", "unresolved"})):
        raise ValueError("mechanism equivalence contains UNKNOWN evidence")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _known(key)
            _known(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _known(item)


@dataclass(frozen=True)
class ProbeBinding:
    graph_digest: str
    plan_digest: str
    compiler_digest: str
    target_digest: str

    def __post_init__(self) -> None:
        for key, value in asdict(self).items():
            _sha(value, key)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProbeBinding:
        return cls(**dict(data))


@dataclass(frozen=True)
class MechanismSignature:
    """Canonical extracted facts; resource names refer to target-derived physical resources."""

    canonical_json: str

    def __post_init__(self) -> None:
        facts = json.loads(self.canonical_json)
        required = {"representations", "events", "capacity_regime", "tile_shape",
                    "edge_cases", "repetition_semantics", "instruction_semantics"}
        if not isinstance(facts, dict) or set(facts) != required:
            raise ValueError("incomplete mechanism signature")
        _known(facts)
        if any(not facts[name] for name in required):
            raise ValueError("empty mechanism evidence cannot establish equivalence")
        if not isinstance(facts["capacity_regime"], dict):
            raise ValueError("capacity regime must identify each live resource")
        for rep in facts["representations"]:
            ValueRepresentation(**{**rep, "attributes": tuple(sorted(rep["attributes"].items()))})
        for dimension in facts["tile_shape"]:
            if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
                raise ValueError("tile dimensions must be positive integers")
        events = []
        for index, event in enumerate(facts["events"]):
            group = event["serial_group"]
            if not isinstance(group, list) or len(group) > 1:
                raise ValueError("serial group must be empty or name one physical contention group")
            events.append(ActivityEvent(
                str(index), event["resource"], event["kind"], 0,
                tuple(str(dep) for dep in event["depends_on"]), group[0] if group else "",
                event["movement_bytes"], event["movement_commands"], event["encoding_transition"]))
        schedule_activity(events)
        # Canonicalize even when reconstructed from an on-disk host receipt.
        object.__setattr__(self, "canonical_json", json.dumps(
            facts, sort_keys=True, separators=(",", ":"), allow_nan=False))

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.canonical_json)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MechanismSignature:
        return cls(json.dumps(dict(data), allow_nan=False))


def derive_mechanism_signature(*, representations: Sequence[ValueRepresentation],
                               events: Sequence[ActivityEvent],
                               capacity_regime: Mapping[str, str],
                               tile_shape: Sequence[int], edge_cases: Sequence[str],
                               repetition_semantics: str,
                               instruction_semantics: Sequence[Mapping[str, Any]],
                               ) -> MechanismSignature:
    """Lift a repeated emitted motif, retaining dependency and physical contention structure.

    Event durations are excluded: those are the unknown being calibrated. Physical resource and
    serial-group names are retained; equal-looking stages sharing a port are not interchangeable
    with independent engines. ``edge_cases`` must explicitly include alignment/tail/halo cases or
    an adapter-derived ``none``; an empty list means no evidence. Capacity classification includes
    all shared live storage, not only the isolated probe's own tensors. Instruction semantics
    retain decoded identities, modes and operand relationships; a generic compute role alone
    cannot distinguish different operations, or accumulator overwrite from accumulation.
    """
    if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in tile_shape):
        raise ValueError("tile dimensions must be positive integers")
    schedule_activity(events)  # Validate topological dependencies and unique event identities.
    positions = {event.id: index for index, event in enumerate(events)}
    body = {
        "representations": [rep.to_dict() for rep in representations],
        "events": [{
            "resource": event.resource, "kind": event.kind,
            "depends_on": sorted(positions[dep] for dep in event.depends_on),
            "serial_group": [event.serial_group] if event.serial_group else [],
            "movement_bytes": event.movement_bytes,
            "movement_commands": event.movement_commands,
            "encoding_transition": event.encoding_transition,
        } for event in events],
        "capacity_regime": dict(capacity_regime), "tile_shape": list(tile_shape),
        "edge_cases": sorted(set(edge_cases)), "repetition_semantics": repetition_semantics,
        "instruction_semantics": [dict(item) for item in instruction_semantics],
    }
    return MechanismSignature.from_dict(body)


@dataclass(frozen=True)
class MechanismEvidence:
    binding: ProbeBinding
    signature: MechanismSignature
    artifact_digest: str
    repetitions: int
    extraction_provenance: str

    def __post_init__(self) -> None:
        _sha(self.artifact_digest, "artifact_digest")
        if isinstance(self.repetitions, bool) or not isinstance(self.repetitions, int) or self.repetitions <= 0:
            raise ValueError("mechanism repetition count must be a positive integer")
        _known(self.extraction_provenance)

    def to_dict(self) -> dict[str, Any]:
        return {"binding": self.binding.to_dict(), "signature": self.signature.to_dict(),
                "artifact_digest": self.artifact_digest, "repetitions": self.repetitions,
                "extraction_provenance": self.extraction_provenance}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MechanismEvidence:
        return cls(ProbeBinding.from_dict(data["binding"]),
                   MechanismSignature.from_dict(data["signature"]), data["artifact_digest"],
                   data["repetitions"], data["extraction_provenance"])


@dataclass(frozen=True)
class ProbeExtraction:
    evidence: MechanismEvidence | None
    missing: tuple[str, ...]
    inventory: Mapping[str, Any]


def extract_mechanism_evidence(*, artifact: bytes, command_buffer: Mapping[str, Any],
                               binding: ProbeBinding, repetitions: int,
                               artifact_analyzer: Callable[[bytes], Mapping[str, Any]],
                               motif_extractor: Callable[
                                   [bytes, Mapping[str, Any]], MechanismSignature] | None = None,
                               extraction_provenance: str) -> ProbeExtraction:
    """Join emitted-byte analysis with declarations, then require a host-owned motif extractor.

    ``artifact_analyzer`` is the target's decoder/activity analyzer, selected by the host. Neither
    callback is loaded from a candidate manifest. Target instruction roles alone cannot establish
    live capacity, physical representations, loop tails, or cross-iteration dependencies. The
    extractor must obtain these from actual lowering and target facts; absent extraction reports
    missing dimensions instead of using descriptor names as evidence. Passing the artifact bytes
    directly ensures the host analyzes the exact object being bound to the returned record.
    """
    from .command_buffer_diagnostics import representation_activity

    declared = representation_activity(command_buffer)
    activity = dict(artifact_analyzer(artifact))
    inventory = {"declared": declared, "issued": activity,
                 "artifact_digest": hashlib.sha256(artifact).hexdigest()}
    missing = []
    if not artifact:
        missing.append("emitted artifact is empty")
    if declared["lowering"]["status"] != "emitted":
        missing.append("whole-model lowering declined")
    if activity.get("status") == "UNKNOWN" or not activity.get("instruction_count"):
        missing.append("no recognized emitted instruction stream")
    encoding = activity.get("encoding_resolution", {})
    if encoding.get("status") not in {"resolved", "complete"}:
        missing.append("instruction encoding resolution is incomplete")
    if motif_extractor is None:
        missing.extend((
            "host extraction of physical dtype/layout/encoding/quantization per motif operand",
            "host extraction of resource topology, dependencies, and serial contention groups",
            "host extraction of live capacity regime including simultaneous neighboring activity",
            "host extraction of tile shape, repetition semantics, alignment/tail/halo cases",
        ))
    if missing:
        return ProbeExtraction(None, tuple(missing), inventory)
    signature = motif_extractor(artifact, command_buffer)
    if not isinstance(signature, MechanismSignature):
        raise TypeError("host motif extractor must produce a validated MechanismSignature")
    evidence = MechanismEvidence(binding, signature, inventory["artifact_digest"], repetitions,
                                 extraction_provenance)
    return ProbeExtraction(evidence, (), inventory)


def require_probe_admission(*, current_binding: ProbeBinding, model: MechanismEvidence,
                            probe: MechanismEvidence, descriptor: Mapping[str, Any],
                            budget: SimulationBudget, estimated_cycles: int,
                            measured_cycles_per_second: float | None,
                            startup_seconds: float = 0.0,
                            contract: WarmProfileContract = WarmProfileContract(),
                            ) -> SimulationAdmission:
    require_probe_execution(descriptor)
    if contract.warmup_runs != 1 or contract.measured_runs != 1:
        raise ValueError("probe calibration requires exactly one warm and one measured invocation")
    if model.binding != current_binding or probe.binding != current_binding:
        raise ValueError("stale graph, global plan, compiler, or target evidence")
    if model.artifact_digest == probe.artifact_digest:
        raise ValueError("the full-model artifact cannot be executed as its own calibration probe")
    if model.signature != probe.signature:
        raise ValueError("probe mechanism differs in representation, resource, capacity, or edge domain")
    if probe.repetitions >= model.repetitions:
        raise ValueError("a calibration probe must reduce the full-model repetition count")
    if isinstance(estimated_cycles, bool) or not isinstance(estimated_cycles, int) or estimated_cycles <= 0:
        raise ValueError("positive per-invocation cycle bound required for probe admission")
    for value in (startup_seconds, measured_cycles_per_second):
        if value is not None and (isinstance(value, bool) or not math.isfinite(value)):
            raise ValueError("runtime estimates must be finite")
    # The entire pair must fit, not each half separately. The runner must enforce the same timeout.
    return admit_reduced_witness(
        estimated_cycles=2 * estimated_cycles,
        measured_cycles_per_second=measured_cycles_per_second,
        startup_seconds=startup_seconds, budget=budget)


@dataclass(frozen=True)
class TimingMeasurementIdentity:
    """Identity observed by a host adapter, including runtime flags and timer/counter semantics."""

    engine_binary_sha256: str
    engine_configuration_sha256: str
    counter_semantics_sha256: str

    def __post_init__(self) -> None:
        for key, value in asdict(self).items():
            _sha(value, key)


_HOST_TIMING_VALIDATED = object()


@dataclass(frozen=True)
class HostTimingAuthority:
    """Host-validated target-cycle scope, not a JSON permission or an engine-name allowlist.

    Mint only through ``authorize_probe_timing``. A digest provides integrity, not accuracy:
    the required host validator must independently check the reference evidence and error bound.
    No production engine is automatically authorized by this type.
    """

    measurement_identity: TimingMeasurementIdentity
    target_digest: str
    mechanism_digest: str
    validation_record_sha256: str
    reference_evidence_sha256: str
    repetition_domain: tuple[int, int]
    systematic_error_cycles: float
    provenance: str
    _host_validation: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._host_validation is not _HOST_TIMING_VALIDATED:
            raise ValueError("timing authority requires independent host validation")
        if type(self.measurement_identity) is not TimingMeasurementIdentity:
            raise ValueError("timing authority requires typed measurement identity")
        for name in ("target_digest", "mechanism_digest", "validation_record_sha256", "reference_evidence_sha256"):
            _sha(getattr(self, name), name)
        if (len(self.repetition_domain) != 2 or
                any(type(n) is not int or n <= 0 for n in self.repetition_domain) or
                self.repetition_domain[0] > self.repetition_domain[1]):
            raise ValueError("timing authority repetition domain is invalid")
        if (isinstance(self.systematic_error_cycles, bool) or
                not math.isfinite(self.systematic_error_cycles) or self.systematic_error_cycles <= 0):
            raise ValueError("timing authority needs an explicit positive systematic error bound")
        if not self.provenance.strip():
            raise ValueError("timing authority requires host validation provenance")

    def to_evidence(self) -> dict[str, Any]:
        return {"schema": "host_probe_timing_authority_v1",
            "measurement_identity": asdict(self.measurement_identity),
            "target_digest": self.target_digest, "mechanism_digest": self.mechanism_digest,
            "validation_record_sha256": self.validation_record_sha256,
            "reference_evidence_sha256": self.reference_evidence_sha256,
            "repetition_domain": list(self.repetition_domain),
            "systematic_error_cycles": self.systematic_error_cycles, "provenance": self.provenance}

    @property
    def digest(self) -> str:
        return _digest(self.to_evidence())


def authorize_probe_timing(*, validation_record: Mapping[str, Any], reference_evidence: bytes,
                           host_validator: Callable[[Mapping[str, Any], bytes], None],
                           provenance: str) -> HostTimingAuthority:
    """Host-only trust boundary; never call on candidate requests or use a permissive validator.

    The supplied host validator must check independent reference evidence, applicability to this
    target/mechanism/engine/configuration/counter contract, and the claimed systematic error bound.
    It raises on missing/invalid evidence and returns None on success. This API checks immutable
    identity and shape; it cannot infer physical accuracy from hashes or four simulator samples.
    No particular reference substrate is mandated. Raw engine-relative diagnostics need no grant.
    """
    record_bytes = json.dumps(validation_record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    record = json.loads(record_bytes)
    if record.get("schema") != "host_probe_timing_validation_v1":
        raise ValueError("timing authority validation schema is invalid")
    if type(reference_evidence) is not bytes or not reference_evidence:
        raise ValueError("timing authority requires actual independent reference evidence")
    reference_sha = hashlib.sha256(reference_evidence).hexdigest()
    if reference_sha != record.get("reference_evidence_sha256"):
        raise ValueError("timing authority reference evidence changed")
    identity = TimingMeasurementIdentity(**record["measurement_identity"])
    if not callable(host_validator):
        raise ValueError("timing authority requires an independent host validator")
    # Pass a detached document: the checker cannot mutate the bytes later minted as authority.
    if host_validator(json.loads(record_bytes), reference_evidence) is not None:
        raise ValueError("host timing validator must raise on failure and return None on success")
    return HostTimingAuthority(identity, record["target_digest"], record["mechanism_digest"],
        hashlib.sha256(record_bytes).hexdigest(), reference_sha, tuple(record["repetition_domain"]),
        record["systematic_error_cycles"], provenance, _HOST_TIMING_VALIDATED)


@dataclass(frozen=True)
class ProbeObservation:
    evidence: MechanismEvidence
    receipt: WarmComputeReceipt
    elapsed_seconds: float
    counter_uncertainty_cycles: float
    receipt_artifact_digest: str
    # Optional host-adapter counter report; its scope stays that of this short
    # probe and never establishes the surrounding model's resource occupancy.
    resource_profile: Mapping[str, Any] | None = None
    # Optional so functional and explicitly engine-relative heuristic observations remain usable.
    # Adapter obligation: capture these identities from actual execution, not candidate metadata.
    timing_authority: HostTimingAuthority | None = None
    observed_timing_identity: TimingMeasurementIdentity | None = None

    def __post_init__(self) -> None:
        if (self.receipt.contract.warmup_runs != 1 or self.receipt.contract.measured_runs != 1):
            raise ValueError("calibration receipt must contain the exact warm 1 + measured 1 contract")
        if self.receipt_artifact_digest != self.evidence.artifact_digest:
            raise ValueError("measurement receipt does not bind the actual probe artifact")
        if not math.isfinite(self.elapsed_seconds) or not 0 < self.elapsed_seconds <= ITERATION_MAX_SECONDS:
            raise ValueError("complete probe execution must fit the 600-second iteration bound")
        if (not math.isfinite(self.counter_uncertainty_cycles)
                or self.counter_uncertainty_cycles <= 0):
            raise ValueError("positive counter/model uncertainty must be supplied, not assumed exact")
        if self.timing_authority is not None and (
                type(self.timing_authority) is not HostTimingAuthority or
                type(self.observed_timing_identity) is not TimingMeasurementIdentity or
                self.observed_timing_identity != self.timing_authority.measurement_identity):
            raise ValueError("timing authority does not match actual host-observed engine/config/counter identity")


@dataclass(frozen=True)
class ProbeCalibration:
    binding: ProbeBinding
    signature: MechanismSignature
    fixed_cycles: float
    cycles_per_repetition: float
    error_cycles: float
    repetition_domain: tuple[int, int]
    provenance: tuple[str, ...]
    timing_authority: HostTimingAuthority

    def interval(self, *, binding: ProbeBinding, signature: MechanismSignature,
                 repetitions: int, current_timing_authority: HostTimingAuthority | None = None) -> CycleInterval:
        if (type(current_timing_authority) is not HostTimingAuthority or
                current_timing_authority != self.timing_authority):
            return CycleInterval.unknown("target-cycle timing authority is absent or changed")
        if binding != self.binding or signature != self.signature:
            return CycleInterval.unknown("calibration identity or mechanism domain changed")
        if (isinstance(repetitions, bool) or not isinstance(repetitions, int)
                or not self.repetition_domain[0] <= repetitions <= self.repetition_domain[1]):
            return CycleInterval.unknown(
                "repetition count outside calibrated domain; a stationary pipeline proof is required")
        center = self.fixed_cycles + repetitions * self.cycles_per_repetition
        return CycleInterval(max(0.0, center - self.error_cycles), center + self.error_cycles,
                             provenance=self.provenance)


def fit_probe_calibration(observations: Sequence[ProbeObservation], *,
                          current_binding: ProbeBinding,
                          current_timing_authority: HostTimingAuthority | None = None) -> ProbeCalibration:
    """Fit fixed + repetition costs with two independent points per parameter.

    The residual envelope is an empirical interval, not a hardware cycle guarantee. It is valid
    only inside the observed repetition domain. Full-model expansion requires a separately
    justified stationary resource pipeline (``pipeline_projection``) or explicit activity schedule;
    this function never extrapolates a short run's cycle count to an entire model. Target-cycle
    fitting requires explicit host timing authority; raw engine-relative heuristics and structural
    search remain permitted without it. Wall-time admission is a separate, unchanged decision.
    """
    if len(observations) < 4:
        raise ValueError("fixed + rate calibration requires at least four independent points")
    first = observations[0].evidence
    xs = [row.evidence.repetitions for row in observations]
    if len(set(xs)) < 4:
        raise ValueError("at least four distinct repetition counts are required")
    if len({row.receipt.provenance for row in observations}) != len(observations):
        raise ValueError("duplicate measurement receipts are not independent observations")
    for row in observations:
        if row.evidence.binding != current_binding:
            raise ValueError("stale calibration evidence")
        if row.evidence.signature != first.signature:
            raise ValueError("cannot fit across different mechanism equivalence domains")
    authority = current_timing_authority
    if type(authority) is not HostTimingAuthority:
        raise ValueError("target-cycle fitting requires explicit host timing authority")
    if authority.target_digest != current_binding.target_digest or authority.mechanism_digest != first.signature.digest:
        raise ValueError("stale timing authority target or mechanism binding")
    for row in observations:
        if (type(row.timing_authority) is not HostTimingAuthority or row.timing_authority != authority
                or row.observed_timing_identity != authority.measurement_identity):
            raise ValueError("mixed, stale or missing observation timing authority")
        if not authority.repetition_domain[0] <= row.evidence.repetitions <= authority.repetition_domain[1]:
            raise ValueError("observation is outside timing authority validity domain")
    ys = [float(row.receipt.total_compute_cycles) for row in observations]
    xbar, ybar = sum(xs) / len(xs), sum(ys) / len(ys)
    rate = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sum(
        (x - xbar) ** 2 for x in xs)
    fixed = ybar - rate * xbar
    if fixed < 0 or rate < 0:
        raise ValueError("measurements do not support nonnegative fixed + repetition costs")
    error = max(abs(y - (fixed + rate * x)) + row.counter_uncertainty_cycles + authority.systematic_error_cycles
                for row, x, y in zip(observations, xs, ys))
    provenance = tuple(f"empirical calibration: {row.receipt.provenance}; "
                       f"artifact={row.evidence.artifact_digest}; "
                       f"timing_authority={authority.digest}; "
                       f"signature={row.evidence.signature.digest}" for row in observations)
    return ProbeCalibration(current_binding, first.signature, fixed, rate, error,
                            (min(xs), max(xs)), provenance, authority)
