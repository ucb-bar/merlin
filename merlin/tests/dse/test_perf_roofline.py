"""Public-contract tests for the empirical, fail-closed roofline report."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from merlin.perf.decompose import ResourceKind, Unavailable, is_unknown
from merlin.perf.envelope import Basis, FixedTerm, Peak, ResourceDemand
from merlin.perf.headroom import Composition
from merlin.perf.roofline import EvidenceReceipt, EmpiricalObservation, empirical_roofline


def _demand(resource: str, kind: ResourceKind, amount: float, unit: str,
            provenance: str) -> ResourceDemand:
    return ResourceDemand(resource, kind, amount, unit, basis=Basis.MOVED,
                          provenance=provenance)


def _peak(resource: str, value: float, unit: str) -> Peak:
    return Peak(resource, value, unit, evidence_kind="measured",
                provenance=f"measured {resource} saturation sweep", n_samples=4,
                is_ceiling=True)


def _observation() -> EmpiricalObservation:
    return EmpiricalObservation(
        workload="layer0",
        cycles=100,
        work=_demand("compute", ResourceKind.COMPUTE, 1000, "ops", "retired-op counter"),
        moved_bytes=(
            _demand("dram", ResourceKind.MOVEMENT, 400, "bytes", "DRAM bus counter"),
        ),
        provenance="cycle-accurate measurement receipt",
    )


def _peaks() -> dict[str, Peak]:
    return {"compute": _peak("compute", 20, "ops"),
            "dram": _peak("dram", 10, "bytes")}


def _fixed() -> dict[str, tuple[FixedTerm, ...]]:
    return {"layer0": (
        FixedTerm("launch", 10, provenance="measured empty-run intercept",
                  evidence_kind="measured"),
    )}


def _receipt(kind: str, *sample_ids: str) -> EvidenceReceipt:
    return EvidenceReceipt("a" * 64, kind, tuple(sample_ids))


def _receipts(observations, peaks, fixed_terms):
    receipts = {"composition": _receipt("rtl_counter_partition", "overlap-0")}
    for observation in observations:
        receipts[f"observation:{observation.workload}"] = _receipt(
            "rtl_cycle_measurement", f"cycles-{observation.workload}")
        if observation.work is not None:
            receipts[f"work:{observation.workload}:{observation.work.resource}"] = _receipt(
                "compiler_ir", f"work-{observation.workload}")
        for demand in observation.moved_bytes:
            if isinstance(demand, ResourceDemand):
                receipts[f"traffic:{observation.workload}:{demand.resource}"] = _receipt(
                    "physical_counter", f"traffic-{observation.workload}-{demand.resource}")
        if fixed_terms is not None and observation.workload in fixed_terms:
            receipts[f"fixed:{observation.workload}"] = _receipt(
                "calibration_fit", *(f"fixed-{observation.workload}-{index}" for index in range(4)))
    for resource in (peaks or {}):
        receipts[f"peak:{resource}"] = _receipt(
            "calibration_fit", *(f"peak-{resource}-{index}" for index in range(4)))
    return receipts


def _empirical_roofline(observations, **kwargs):
    kwargs.setdefault("evidence_receipts", _receipts(
        observations, kwargs.get("peaks"), kwargs.get("fixed_terms")))
    return empirical_roofline(observations, **kwargs)


def _report(observation: EmpiricalObservation, *, peaks=None, fixed_terms=None,
            expected_workloads=("layer0",)):
    return _empirical_roofline(
        [observation], peaks=_peaks() if peaks is None else peaks,
        fixed_terms=_fixed() if fixed_terms is None else fixed_terms,
        composition=Composition.SUM, composition_eta=0.0,
        composition_provenance="measured zero-overlap trace",
        expected_workloads=expected_workloads,
    )


def test_complete_measurements_produce_a_serializable_empirical_roofline_point() -> None:
    report = _empirical_roofline(
        [_observation()], peaks=_peaks(), fixed_terms=_fixed(),
        composition=Composition.SUM,
        composition_eta=0.0,
        composition_provenance="measured zero-overlap trace",
        expected_workloads=("layer0",),
    )

    point = report.points["layer0"]
    assert report.complete is True and report.refusals == ()
    assert point.bound_cycles == 100.0
    assert point.efficiency == 1.0
    assert point.measured_rate == 10.0
    assert point.intensity_by_level == {"dram": 2.5}
    assert point.limiter == "compute"
    assert point.margin_to_second == 10.0
    assert report.coverage.complete is True
    assert report.coverage.cycle_weighted_resolved_share == 1.0
    encoded = json.loads(json.dumps(report.to_dict()))
    assert encoded["status"] == "resolved"
    assert encoded["points"]["layer0"]["provenance"] == {
        "cycles": "cycle-accurate measurement receipt",
        "fixed_terms": {"launch": "measured empty-run intercept"},
        "moved_bytes": {"dram": "DRAM bus counter"},
        "peaks": {
            "compute": "measured compute saturation sweep",
            "dram": "measured dram saturation sweep",
        },
        "work": "retired-op counter",
    }


def test_missing_composition_is_a_named_refusal_and_never_defaults_to_max() -> None:
    report = _empirical_roofline(
        [_observation()], peaks=_peaks(), fixed_terms=_fixed(),
        expected_workloads=("layer0",),
    )

    assert report.complete is False
    assert is_unknown(report.points["layer0"].bound_cycles)
    assert {item.missing for item in report.refusals} >= {
        ("an explicitly measured Composition",),
        ("composition provenance",),
    }
    encoded = json.loads(json.dumps(report.to_dict()))
    assert encoded["composition"] is None
    assert encoded["points"]["layer0"]["bound_cycles"] == "UNKNOWN"

    bare_string = _empirical_roofline(
        [_observation()], peaks=_peaks(), fixed_terms=_fixed(),
        composition="max", composition_eta=1.0,
        composition_provenance="caller assertion", expected_workloads=("layer0",))
    assert json.loads(json.dumps(bare_string.to_dict()))["composition"] is None


def test_missing_actual_moved_bytes_refuses_the_point_instead_of_using_algorithmic_bytes() -> None:
    observation = EmpiricalObservation(
        workload="layer0", cycles=100,
        work=_demand("compute", ResourceKind.COMPUTE, 1000, "ops", "retired-op counter"),
        moved_bytes=(), provenance="cycle-accurate measurement receipt")
    report = _empirical_roofline(
        [observation], peaks={"compute": _peak("compute", 20, "ops")},
        fixed_terms=_fixed(), composition=Composition.SUM, composition_eta=0.0,
        composition_provenance="measured zero-overlap trace",
        expected_workloads=("layer0",),
    )

    point = report.points["layer0"]
    assert is_unknown(point.bound_cycles)
    assert any("actual moved bytes" in missing
               for item in point.refusals for missing in item.missing)
    assert point.intensity_by_level == {}


def test_a_missing_or_unknown_peak_propagates_unknown_without_inventing_bandwidth() -> None:
    missing = _empirical_roofline(
        [_observation()], peaks={"compute": _peak("compute", 20, "ops")},
        fixed_terms=_fixed(), composition=Composition.SUM, composition_eta=0.0,
        composition_provenance="measured zero-overlap trace",
        expected_workloads=("layer0",),
    )
    unknown_peak = Peak.unknown(
        "dram", "bytes", "bandwidth sweep did not run", provenance="measurement refusal")
    unknown = _empirical_roofline(
        [_observation()], peaks={"compute": _peak("compute", 20, "ops"),
                                 "dram": unknown_peak},
        fixed_terms=_fixed(), composition=Composition.SUM, composition_eta=0.0,
        composition_provenance="measured zero-overlap trace",
        expected_workloads=("layer0",),
    )

    assert is_unknown(missing.points["layer0"].bound_cycles)
    assert any("peak" in item.what for item in missing.points["layer0"].refusals)
    assert is_unknown(unknown.points["layer0"].bound_cycles)
    assert "bandwidth sweep did not run" in unknown.points["layer0"].envelope.to_dict()[
        "unresolved_reasons"]["dram"]


def test_peak_identity_and_units_must_match_the_demand_exactly() -> None:
    wrong_identity = replace(_peak("dram", 10, "bytes"), resource="other")
    wrong_unit = replace(_peak("dram", 10, "bytes"), unit="beats")

    for peak in (wrong_identity, wrong_unit):
        report = _report(
            _observation(), peaks={"compute": _peak("compute", 20, "ops"), "dram": peak})
        assert is_unknown(report.points["layer0"].bound_cycles)
        assert any("exact peak identity and unit" in missing
                   for item in report.points["layer0"].refusals for missing in item.missing)


def test_every_measurement_contributing_to_a_point_requires_provenance() -> None:
    observation = _observation()
    cases = (
        replace(observation, provenance=""),
        replace(observation, work=replace(observation.work, provenance="")),
        replace(observation, moved_bytes=(replace(observation.moved_bytes[0], provenance=""),)),
    )
    reports = [_report(case) for case in cases]
    reports.append(_report(
        observation,
        fixed_terms={"layer0": (FixedTerm("launch", 10, evidence_kind="measured"),)}))

    for report in reports:
        point = report.points["layer0"]
        assert is_unknown(point.bound_cycles)
        assert any("provenance" in missing
                   for item in point.refusals for missing in item.missing)


def test_coverage_requires_an_explicit_complete_workload_set() -> None:
    unscoped = _report(_observation(), expected_workloads=None)
    incomplete = _report(_observation(), expected_workloads=("layer0", "layer1"))

    assert isinstance(unscoped.coverage, Unavailable)
    assert is_unknown(unscoped.coverage)
    assert any("coverage" in item.what for item in unscoped.refusals)
    assert incomplete.coverage.missing == ("layer1",)
    assert incomplete.coverage.complete is False
    assert is_unknown(incomplete.coverage.cycle_weighted_resolved_share)
    assert incomplete.complete is False


def test_coverage_denominator_is_non_vacuous_and_has_unique_identities() -> None:
    empty = _empirical_roofline(
        [], peaks={}, fixed_terms={}, composition=Composition.SUM, composition_eta=0.0,
        composition_provenance="measured zero-overlap trace", expected_workloads=())
    repeated = _report(_observation(), expected_workloads=("layer0", "layer0"))

    for report in (empty, repeated):
        assert report.complete is False
        assert any("workload identities" in missing
                   for item in report.refusals for missing in item.missing)


def test_fixed_terms_must_be_explicit_even_when_the_measured_set_is_empty() -> None:
    absent = _report(_observation(), fixed_terms={})
    explicit_zero = _report(_observation(), fixed_terms={"layer0": ()})

    assert is_unknown(absent.points["layer0"].bound_cycles)
    assert any("fixed-term" in missing
               for item in absent.points["layer0"].refusals for missing in item.missing)
    assert explicit_zero.points["layer0"].bound_cycles == 90.0
    assert explicit_zero.complete is True


def test_missing_work_or_invalid_cycles_is_refused_not_coerced_to_a_default() -> None:
    missing_work = _report(replace(_observation(), work=None))
    invalid_cycles = _report(replace(_observation(), cycles=0))

    for report in (missing_work, invalid_cycles):
        point = report.points["layer0"]
        assert is_unknown(point.bound_cycles)
        assert is_unknown(point.efficiency)
        assert point.refusals
        assert json.loads(json.dumps(report.to_dict()))["status"] == "refused"


@pytest.mark.parametrize("traffic", [
    ResourceDemand("dram", ResourceKind.MOVEMENT, 400, "bytes",
                   basis=Basis.ALGORITHMIC, provenance="operand-size calculation"),
    _demand("dram", ResourceKind.MOVEMENT, 400, "beats", "transaction counter"),
    _demand("dram", ResourceKind.MOVEMENT, 0, "bytes", "DRAM bus counter"),
])
def test_byte_axis_requires_positive_actual_moved_bytes(traffic: ResourceDemand) -> None:
    observation = replace(_observation(), moved_bytes=(traffic,))
    peak = _peak("dram", 10, traffic.unit)
    report = _report(observation, peaks={"compute": _peak("compute", 20, "ops"),
                                         "dram": peak})

    assert is_unknown(report.points["layer0"].bound_cycles)
    assert any("actual moved bytes" in missing
               for item in report.points["layer0"].refusals for missing in item.missing)


def test_duplicate_observations_cannot_disappear_behind_a_workload_mapping() -> None:
    report = _empirical_roofline(
        [_observation(), _observation()], peaks=_peaks(), fixed_terms=_fixed(),
        composition=Composition.SUM, composition_eta=0.0,
        composition_provenance="measured zero-overlap trace",
        expected_workloads=("layer0",),
    )

    assert report.complete is False
    assert any("duplicate" in item.detail for item in report.refusals)
    assert is_unknown(report.points["layer0"].bound_cycles)


def test_each_measured_memory_level_is_a_separate_roofline_axis_and_term() -> None:
    observation = replace(
        _observation(), cycles=200,
        moved_bytes=(
            _demand("dram", ResourceKind.MOVEMENT, 400, "bytes", "DRAM bus counter"),
            _demand("l2", ResourceKind.MOVEMENT, 200, "bytes", "L2 bus counter"),
        ))
    report = _report(
        observation,
        peaks={"compute": _peak("compute", 20, "ops"),
               "dram": _peak("dram", 10, "bytes"),
               "l2": _peak("l2", 20, "bytes")},
        fixed_terms={"layer0": (
            FixedTerm("compute_fill", 5, resource="compute",
                      provenance="measured sweep intercept", evidence_kind="measured"),
            FixedTerm("launch", 5, provenance="measured empty-run intercept",
                      evidence_kind="measured"),
        )})

    point = report.points["layer0"]
    assert point.intensity_by_level == {"dram": 2.5, "l2": 5.0}
    assert point.envelope.terms == {"compute": 55.0, "dram": 40.0, "l2": 10.0}
    assert point.bound_cycles == 110.0
    assert point.efficiency == 0.55
    assert point.limiter == "compute" and point.margin_to_second == 15.0


def test_a_bound_above_measured_cycles_is_a_falsification_not_super_efficiency() -> None:
    report = _report(
        _observation(),
        peaks={"compute": _peak("compute", 5, "ops"),
               "dram": _peak("dram", 10, "bytes")})

    point = report.points["layer0"]
    assert point.bound_cycles == 250.0
    assert point.is_valid_lower_bound is False
    assert is_unknown(point.efficiency)
    assert any("exceeds" in item.detail for item in point.refusals)
    assert report.complete is False


def test_empirical_report_rejects_assumed_or_structural_peaks_and_fixed_terms() -> None:
    structural_peak = replace(_peak("dram", 10, "bytes"), evidence_kind="structural_bound")
    assumed_fixed = FixedTerm(
        "launch", 10, provenance="design estimate", evidence_kind="assumed")
    peak_report = _report(
        _observation(), peaks={"compute": _peak("compute", 20, "ops"),
                               "dram": structural_peak})
    fixed_report = _report(_observation(), fixed_terms={"layer0": (assumed_fixed,)})

    for report in (peak_report, fixed_report):
        point = report.points["layer0"]
        assert is_unknown(point.bound_cycles)
        assert any("measured evidence" in missing
                   for item in point.refusals for missing in item.missing)


def test_duplicate_resource_demands_and_unmatched_fixed_terms_are_refused() -> None:
    duplicate_traffic = replace(
        _observation(), moved_bytes=(_observation().moved_bytes[0],
                                     _observation().moved_bytes[0]))
    unmatched_fixed = {"layer0": (
        FixedTerm("other_fill", 3, resource="other",
                  provenance="measured intercept", evidence_kind="measured"),
    )}

    for report in (_report(duplicate_traffic),
                   _report(_observation(), fixed_terms=unmatched_fixed)):
        point = report.points["layer0"]
        assert is_unknown(point.bound_cycles)
        assert any("resource" in missing
                   for item in point.refusals for missing in item.missing)


def test_untyped_traffic_and_fixed_terms_are_refused_without_crashing_the_report() -> None:
    bad_traffic = replace(_observation(), moved_bytes=(400,))
    reports = (
        _report(bad_traffic),
        _report(_observation(), fixed_terms={"layer0": (10,)}),
    )

    for report in reports:
        assert is_unknown(report.points["layer0"].bound_cycles)
        assert json.loads(json.dumps(report.to_dict()))["status"] == "refused"


def test_coverage_is_weighted_by_measured_cycles_and_keeps_refusals_visible() -> None:
    second = EmpiricalObservation(
        workload="layer1", cycles=300,
        work=_demand("other_compute", ResourceKind.COMPUTE, 500, "ops", "work counter"),
        moved_bytes=(_demand("dram", ResourceKind.MOVEMENT, 400, "bytes",
                             "DRAM bus counter"),),
        provenance="cycle-accurate measurement receipt")
    report = _empirical_roofline(
        [_observation(), second], peaks=_peaks(),
        fixed_terms={**_fixed(), "layer1": ()},
        composition=Composition.SUM, composition_eta=0.0,
        composition_provenance="measured zero-overlap trace",
        expected_workloads=("layer0", "layer1"))

    assert report.coverage.resolved == ("layer0",)
    assert report.coverage.cycle_weighted_resolved_share == 0.25
    assert any("other_compute" in item.what for item in report.refusals)


def test_work_units_are_preserved_without_assuming_two_operations_per_mac() -> None:
    observation = replace(
        _observation(),
        work=_demand("compute", ResourceKind.COMPUTE, 7, "macs", "MAC retire counter"))
    report = _report(
        observation,
        peaks={"compute": _peak("compute", 1, "macs"),
               "dram": _peak("dram", 10, "bytes")},
        fixed_terms={"layer0": ()})

    point = report.points["layer0"]
    assert point.work == 7.0 and point.work_unit == "macs"
    assert point.measured_rate == 0.07
    assert point.intensity_by_level == {"dram": 0.0175}


def test_prose_only_trust_claims_cannot_resolve_an_empirical_roofline() -> None:
    invented_peaks = {
        "compute": Peak("compute", 20, "ops", "measured", "trust me"),
        "dram": Peak("dram", 10, "bytes", "measured", "trust me"),
    }
    report = empirical_roofline(
        [_observation()], peaks=invented_peaks, fixed_terms=_fixed(),
        composition=Composition.SUM, composition_eta=0.0,
        composition_provenance="trust me", expected_workloads=("layer0",))

    assert not report.complete and is_unknown(report.points["layer0"].bound_cycles)
    assert any("content-addressed" in missing or "receipt" in missing.lower()
               for refusal in report.refusals for missing in refusal.missing)


def test_composition_enum_must_agree_with_counter_derived_eta() -> None:
    report = _empirical_roofline(
        [_observation()], peaks=_peaks(), fixed_terms=_fixed(),
        composition=Composition.SUM, composition_eta=1.0,
        composition_provenance="counter-derived", expected_workloads=("layer0",))

    assert not report.complete and is_unknown(report.points["layer0"].bound_cycles)
    assert any("consistent" in missing for refusal in report.refusals for missing in refusal.missing)
