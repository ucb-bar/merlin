"""Calibration may follow only the current graph/plan's actual execution mechanism."""
from dataclasses import replace
import hashlib
import json

import pytest

from merlin.perf.activity_schedule import ActivityEvent
from merlin.perf.execution_policy import SimulationBudget, WarmComputeReceipt, WarmProfileContract
from merlin.perf.mechanism_probe import (
    MechanismEvidence, MechanismSignature, ProbeBinding, ProbeObservation,
    TimingMeasurementIdentity, authorize_probe_timing,
    derive_mechanism_signature, extract_mechanism_evidence, fit_probe_calibration,
    require_probe_admission,
)
from merlin.xdsl_dialects.lowering.global_plan import ValueRepresentation


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def binding():
    return ProbeBinding(*(digest(name) for name in ("graph", "plan", "compiler", "target")))


def signature(*, dtype="i8", encoding="blocked", shared=False, regime="fits_double",
              edges=("aligned",), dep=True):
    return derive_mechanism_signature(
        representations=[ValueRepresentation("local", "row-major", dtype, encoding)],
        events=[ActivityEvent("load", "load-engine", "movement", 0,
                              movement_bytes=128, movement_commands=1,
                              serial_group="shared-port" if shared else ""),
                ActivityEvent("compute", "compute-engine", "compute", 0,
                              depends_on=("load",) if dep else (),
                              serial_group="shared-port" if shared else "")],
        capacity_regime={"local-store": regime}, tile_shape=(8, 8, 16),
        edge_cases=edges, repetition_semantics="independent output tiles with resident weights",
        instruction_semantics=[{"operation": "operand-load", "operand": "activation"},
                               {"operation": "multiply-accumulate", "accumulate": True}])


def evidence(repetitions, *, sig=None, identity=None):
    return MechanismEvidence(identity or binding(), sig or signature(),
                             digest(f"emitted-{repetitions}"), repetitions,
                             "host parsed emitted loop and allocation facts")


def admit(**kwargs):
    args = dict(current_binding=binding(), model=evidence(10000), probe=evidence(4),
                descriptor={"kind": "kernel", "operation": {"op": "matmul"}},
                budget=SimulationBudget(600, 600), estimated_cycles=200,
                measured_cycles_per_second=1)
    args.update(kwargs)
    return require_probe_admission(**args)


def test_short_probe_admits_full_pair_but_not_two_separately_admissible_runs():
    assert admit().admitted
    result = admit(estimated_cycles=301)
    assert not result.admitted
    assert result.estimated_seconds == 602
    assert not admit(measured_cycles_per_second=None).admitted


@pytest.mark.parametrize("field", ["graph_digest", "plan_digest", "compiler_digest", "target_digest"])
def test_every_current_identity_is_checked(field):
    stale = replace(binding(), **{field: digest("stale")})
    with pytest.raises(ValueError, match="stale"):
        admit(probe=evidence(4, identity=stale))


@pytest.mark.parametrize("change", [dict(dtype="i32"), dict(encoding="packed"),
                                     dict(shared=True), dict(regime="spills"),
                                     dict(edges=("tail",)), dict(dep=False)])
def test_mechanism_changes_cannot_reuse_calibration(change):
    with pytest.raises(ValueError, match="mechanism differs"):
        admit(probe=evidence(4, sig=signature(**change)))


def test_unknown_regime_or_missing_edge_case_is_not_equivalent():
    with pytest.raises(ValueError, match="UNKNOWN"):
        signature(regime="unknown")
    with pytest.raises(ValueError, match="empty mechanism"):
        signature(edges=())


@pytest.mark.parametrize("descriptor", [
    {"kind": "model"}, {"operation": {"op": "model"}},
    {"performance": {"measurement_scope": "full_layer"}},
    {"performance": {"global_objective": True}},
])
def test_full_model_or_layer_never_admitted_even_if_tiny(descriptor):
    with pytest.raises(ValueError, match="compile-only"):
        admit(descriptor=descriptor, estimated_cycles=1)


def test_same_artifact_and_unreduced_repetition_refused():
    with pytest.raises(ValueError, match="own calibration"):
        admit(probe=replace(evidence(4), artifact_digest=evidence(10000).artifact_digest))
    with pytest.raises(ValueError, match="reduce"):
        admit(probe=evidence(10001))


def test_roundtrip_preserves_binding_and_normalized_signature():
    original = evidence(4)
    assert MechanismEvidence.from_dict(original.to_dict()) == original
    body = original.signature.to_dict()
    body["events"][1]["depends_on"] = [7]
    with pytest.raises(ValueError, match="dependencies"):
        MechanismSignature.from_dict(body)


def test_empty_and_named_independent_serial_groups_are_distinct():
    body = signature().to_dict()
    for event in body["events"]:
        event["serial_group"] = ["independent"]
    assert MechanismSignature.from_dict(body) != signature()


def test_same_compute_role_does_not_hide_different_accumulator_semantics():
    body = signature().to_dict()
    body["instruction_semantics"][1]["accumulate"] = False
    with pytest.raises(ValueError, match="mechanism differs"):
        admit(probe=evidence(4, sig=MechanismSignature.from_dict(body)))


def timing_authority(**changes):
    """Independent analytical test fixture only; this authorizes no real simulator or target."""
    measured_identity = TimingMeasurementIdentity(*(digest(name) for name in (
        "synthetic-engine", "synthetic-configuration", "synthetic-timer-counter-semantics")))
    document = {"schema": "host_probe_timing_validation_v1",
        "measurement_identity": measured_identity.__dict__, "target_digest": binding().target_digest,
        "mechanism_digest": signature().digest, "repetition_domain": [1, 10],
        "systematic_error_cycles": 2}
    document.update(changes)
    reference = {"scope": "synthetic analytical fixture; not physical hardware qualification",
        "identity": document["measurement_identity"], "target_digest": document["target_digest"],
        "mechanism_digest": document["mechanism_digest"],
        "points": [{"repetitions": n, "reference_cycles": 16 + 3*n, "engine_cycles": 17 + 3*n}
                   for n in range(1, 11)]}
    payload = json.dumps(reference, sort_keys=True).encode()
    document["reference_evidence_sha256"] = hashlib.sha256(payload).hexdigest()
    def validate(record, raw):
        actual = json.loads(raw)
        assert actual["identity"] == record["measurement_identity"]
        assert actual["target_digest"] == record["target_digest"]
        assert actual["mechanism_digest"] == record["mechanism_digest"]
        assert record["repetition_domain"][0] >= 1 and record["repetition_domain"][1] <= 10
        for point in actual["points"]:
            assert point["reference_cycles"] == 16 + 3 * point["repetitions"]
            assert abs(point["engine_cycles"] - point["reference_cycles"]) <= record["systematic_error_cycles"]
    return authorize_probe_timing(validation_record=document, reference_evidence=payload,
        host_validator=validate, provenance="host-checked synthetic analytical test fixture")


def observation(repetitions, *, cycles=None, sig=None, identity=None, authority=None):
    ev = evidence(repetitions, sig=sig, identity=identity)
    receipt = WarmComputeReceipt(f"probe-{repetitions}", cycles or (17 + 3 * repetitions),
                                 WarmProfileContract(), f"receipt-{repetitions}")
    return ProbeObservation(ev, receipt, 10, 1, ev.artifact_digest,
        timing_authority=authority,
        observed_timing_identity=authority.measurement_identity if authority else None)


def test_four_authority_free_observations_cannot_license_target_cycle_fit():
    rows = [observation(n) for n in (2, 4, 6, 8)]
    with pytest.raises(ValueError, match="timing authority"):
        fit_probe_calibration(rows, current_binding=binding())


def test_fixed_and_rate_fit_returns_empirical_interval_only_within_domain():
    authority = timing_authority()
    fitted = fit_probe_calibration([observation(n, authority=authority) for n in (2, 4, 6, 8)],
                                   current_binding=binding(), current_timing_authority=authority)
    assert fitted.fixed_cycles == 17
    assert fitted.cycles_per_repetition == 3
    interval = fitted.interval(binding=binding(), signature=signature(), repetitions=5,
                               current_timing_authority=authority)
    assert (interval.lo, interval.hi) == (29, 35)  # resolution 1 + independently validated model error 2
    assert all("artifact=" in item for item in interval.provenance)
    assert not fitted.interval(binding=binding(), signature=signature(), repetitions=9,
                               current_timing_authority=authority).resolved  # authority wider than actual samples
    assert not fitted.interval(binding=binding(), signature=signature(shared=True), repetitions=4,
                               current_timing_authority=authority).resolved
    assert not fitted.interval(binding=replace(binding(), compiler_digest=digest("new")),
                               signature=signature(), repetitions=4, current_timing_authority=authority).resolved
    assert not fitted.interval(binding=binding(), signature=signature(), repetitions=4).resolved


@pytest.mark.parametrize("field", ["engine_binary_sha256", "engine_configuration_sha256", "counter_semantics_sha256"])
def test_observation_authority_must_match_actual_observed_identity(field):
    authority = timing_authority()
    row = observation(2, authority=authority)
    with pytest.raises(ValueError, match="actual host-observed"):
        replace(row, observed_timing_identity=replace(authority.measurement_identity, **{field: digest("changed")}))
    with pytest.raises(ValueError, match="actual host-observed"):
        replace(row, observed_timing_identity=None)
    with pytest.raises(ValueError, match="actual host-observed"):
        replace(row, timing_authority=authority.to_evidence())


def test_raw_same_engine_observations_still_store_but_cannot_fit_target_cycles():
    identity = timing_authority().measurement_identity
    rows = [replace(observation(n), observed_timing_identity=identity) for n in (2, 4, 6, 8)]
    assert rows[-1].receipt.total_compute_cycles - rows[0].receipt.total_compute_cycles == 18
    with pytest.raises(ValueError, match="timing authority"):
        fit_probe_calibration(rows, current_binding=binding())
    assert admit().admitted  # execution wall-time admission is independent


def test_mixed_stale_and_missing_authority_refused():
    authority = timing_authority()
    rows = [observation(n, authority=authority) for n in (2, 4, 6, 8)]
    for stale in (timing_authority(systematic_error_cycles=3),
                  timing_authority(measurement_identity=replace(authority.measurement_identity,
                       engine_configuration_sha256=digest("other configuration")).__dict__)):
        with pytest.raises(ValueError, match="mixed, stale"):
            fit_probe_calibration(rows, current_binding=binding(), current_timing_authority=stale)
        mixed = rows[:3] + [observation(8, authority=stale)]
        with pytest.raises(ValueError, match="mixed, stale"):
            fit_probe_calibration(mixed, current_binding=binding(), current_timing_authority=authority)
    with pytest.raises(ValueError, match="missing observation"):
        fit_probe_calibration(rows[:3] + [observation(8)], current_binding=binding(), current_timing_authority=authority)
    with pytest.raises(ValueError, match="stale timing authority"):
        fit_probe_calibration(rows, current_binding=binding(), current_timing_authority=timing_authority(target_digest=digest("other target")))
    narrow = timing_authority(repetition_domain=[1, 7])
    with pytest.raises(ValueError, match="validity domain"):
        fit_probe_calibration([observation(n, authority=narrow) for n in (2, 4, 6, 8)],
                              current_binding=binding(), current_timing_authority=narrow)


def test_reference_hash_alone_is_not_host_validation():
    with pytest.raises(AssertionError):
        timing_authority(systematic_error_cycles=0.5)  # independent reference disproves claimed error
    authority = timing_authority()
    record = {"schema": "host_probe_timing_validation_v1", **authority.to_evidence(),
              "reference_evidence_sha256": digest("different bytes")}
    record["schema"] = "host_probe_timing_validation_v1"
    with pytest.raises(ValueError, match="reference evidence changed"):
        authorize_probe_timing(validation_record=record, reference_evidence=b"actual reference",
                               host_validator=None, provenance="not validated")
    record["reference_evidence_sha256"] = digest("actual reference")
    with pytest.raises(ValueError, match="independent host validator"):
        authorize_probe_timing(validation_record=record, reference_evidence=b"actual reference",
                               host_validator=None, provenance="not validated")


def test_fit_requires_independent_points_and_receipts_not_replicated_sample():
    with pytest.raises(ValueError, match="four independent"):
        fit_probe_calibration([observation(n) for n in (2, 4, 6)], current_binding=binding())
    with pytest.raises(ValueError, match="distinct"):
        fit_probe_calibration([observation(2)] * 4, current_binding=binding())
    rows = [observation(n) for n in (2, 4, 6, 8)]
    rows[-1] = replace(rows[-1], receipt=replace(rows[-1].receipt, provenance="receipt-2"))
    with pytest.raises(ValueError, match="duplicate"):
        fit_probe_calibration(rows, current_binding=binding())


def test_fit_rejects_capacity_domain_or_stale_evidence():
    rows = [observation(n) for n in (2, 4, 6)]
    with pytest.raises(ValueError, match="equivalence domains"):
        fit_probe_calibration(rows + [observation(8, sig=signature(regime="spills"))],
                              current_binding=binding())
    with pytest.raises(ValueError, match="stale"):
        fit_probe_calibration(rows + [observation(8, identity=replace(
            binding(), target_digest=digest("different RTL")))], current_binding=binding())


def test_receipt_must_bind_artifact_and_exact_warm_policy():
    row = observation(2)
    with pytest.raises(ValueError, match="actual probe artifact"):
        replace(row, receipt_artifact_digest=digest("stale binary"))
    with pytest.raises(ValueError, match="warm 1"):
        replace(row, receipt=replace(row.receipt, contract=WarmProfileContract(warmup_runs=2)))
    with pytest.raises(ValueError, match="600-second"):
        replace(row, elapsed_seconds=601)


def test_host_extraction_does_not_invent_resource_topology_from_command_counts():
    cb = {"tensors": {"x": {"dtype": "i8", "shape": [8, 8]}},
          "commands": [{"opcode": "compute"}]}
    def analyzer(artifact):
        assert artifact == b"actual lowered artifact"
        return {"instruction_count": 2, "encoding_resolution": {"status": "complete"}}
    result = extract_mechanism_evidence(
        artifact=b"actual lowered artifact", command_buffer=cb, binding=binding(),
        repetitions=4, artifact_analyzer=analyzer, extraction_provenance="host analyzer")
    assert result.evidence is None
    assert any("resource topology" in missing for missing in result.missing)
    assert result.inventory["declared"]["tensors"]["x"]["dtype"] == "i8"


def test_host_extractor_bound_to_actual_bytes_and_decline_blocks_extraction():
    args = dict(artifact=b"artifact", command_buffer={"tensors": {}, "commands": []},
                binding=binding(), repetitions=4,
                artifact_analyzer=lambda _: {"instruction_count": 2,
                                             "encoding_resolution": {"status": "resolved"}},
                motif_extractor=lambda artifact, cb: signature(), extraction_provenance="host")
    result = extract_mechanism_evidence(**args)
    assert result.evidence.artifact_digest == digest("artifact")
    args["command_buffer"] = {"declined": {"reason": "host loop unsupported"}}
    result = extract_mechanism_evidence(**args)
    assert result.evidence is None
    assert "whole-model lowering declined" in result.missing
