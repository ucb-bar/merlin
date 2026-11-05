"""Necessary service aggregation is not an overlap or qualification assertion."""

import math

import pytest

from merlin.perf.capsule_roofline import compose_service_floor


def resource(rate, unit="bytes"):
    return dict(unit=unit, capacity_per_cycle_upper=rate, provenance="selected explicit structural proof")


def event(name, demand, compulsory=True):
    return dict(id=name, demand=demand, compulsory=compulsory, provenance="original complete effects")


def test_shared_ports_sum_demands_and_distinct_resources_take_max_without_overlap_claim():
    independent = compose_service_floor(
        {"read": resource(16), "write": resource(8)}, [event("load", {"read": 64}), event("store", {"write": 64})]
    )
    shared = compose_service_floor({"port": resource(8)}, [event("load", {"port": 64}), event("store", {"port": 64})])
    assert independent["floor_cycles"] == 8
    assert shared["floor_cycles"] == 16
    assert independent["authority"].startswith("none")
    assert not independent["convergence_eligible"]


def test_only_explicit_complete_dependencies_add_independent_services():
    resources = {"dma": resource(16), "input": resource(1, "rows")}
    events = [event("load", {"dma": 64}), event("execute", {"input": 16})]
    assert compose_service_floor(resources, events)["floor_cycles"] == 16
    proof = [dict(events=["load", "execute"], provenance="checked last-load completion before first execute input")]
    assert compose_service_floor(resources, events, completion_chains=proof)["floor_cycles"] == 20
    with pytest.raises(ValueError, match="completion proof"):
        compose_service_floor(resources, events, completion_chains=[dict(events=["load", "load"], provenance="order")])


def test_unknown_service_cannot_be_zero_or_disappear_from_status():
    result = compose_service_floor(
        {"dma": resource(16), "readout": resource(None)},
        [event("a", {"dma": 65}), event("b", {"readout": 16}), event("c", {"dma": None})],
    )
    assert result["floor_cycles"] == 5
    assert result["status"] == "partial"
    assert len(result["unresolved"]) == 2
    absent = compose_service_floor({"dma": resource(None)}, [event("a", {"dma": None})])
    assert absent["floor_cycles"] is None and absent["status"] == "unknown"


def test_candidate_waste_is_not_a_workload_lower_bound():
    resources = {"input": resource(1, "rows")}
    with pytest.raises(ValueError, match="compulsory scope"):
        compose_service_floor(resources, [event("tile", {"input": 16}, False)])
    scoped = compose_service_floor(resources, [event("tile", {"input": 16}, False)], scope="emitted_schedule")
    assert scoped["floor_cycles"] == 16 and scoped["scope"] == "emitted_schedule"


@pytest.mark.parametrize("rate", [0, -1, True, math.inf, math.nan])
def test_bad_prices_or_nonfinite_rates_refuse(rate):
    with pytest.raises(ValueError, match="rate"):
        compose_service_floor({"dma": resource(rate)}, [event("a", {"dma": 1})])
