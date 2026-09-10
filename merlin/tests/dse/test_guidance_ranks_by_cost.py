"""The work list must open with the biggest measured cost, not with the alphabet.

Findings were ordered by a hardcoded priority class and then by ``str(kind)``. Measured on a real
brief: the host hotspot holding 45.7% of the host lane -- 549,695 of 1,202,745 dynamic operations,
on a lane that is >=93% of the measured window -- ranked SIXTH, beneath a synchronization finding
whose entire evidence was ``{"baseline": 3, "candidate": 23}``. Twenty units outranked half a
million because "a" precedes "h".

Every case is built from an analysis whose numbers make the correct answer unambiguous.
"""
from __future__ import annotations

from merlin.perf.agent_guidance import (
    PackageOptimizationInventory,
    guidance_for_emission_analysis,
)

EMPTY_INVENTORY = PackageOptimizationInventory(symbols=(), surfaces=())


def _task(index: int, *, integer: int = 0, floating: int = 0, regions: tuple[str, ...] = ()):
    return {"dynamic_operations": {"integer_arithmetic": integer, "floating_arithmetic": floating},
            "load_payload_bytes": 0, "source_regions": list(regions), "task": index}


def _analysis(tasks, *, sync_baseline: int = 3, sync_candidate: int = 23):
    """An emission analysis carrying one big host hotspot AND a small sync regression."""
    families: dict[str, int] = {}
    for task in tasks:
        for name, value in task["dynamic_operations"].items():
            families[name] = families.get(name, 0) + value
    return {
        "arms": {"baseline": {"status": "emitted", "macs": 1, "exact": True},
                 "candidate": {"status": "emitted", "macs": 1, "exact": True}},
        "target_artifact_activity": {
            "baseline": {"synchronization_operations": sync_baseline},
            "candidate": {"synchronization_operations": sync_candidate}},
        "verified_global_plan_emission": {
            "status": "verified",
            "host_activity": {
                "status": "derived",
                "dynamic_operations": families,
                "tasks": tasks,
                "top_tasks_by_scalar_memory_payload": list(reversed(tasks)),
                "top_allocations_by_static_payload": [{"buffer_root": "alloca:1"}],
                "top_buffers_by_scalar_memory_payload": [{"buffer_root": "alloca:1"}],
            },
        },
    }


def _brief(tasks, **kwargs):
    return guidance_for_emission_analysis(_analysis(tasks, **kwargs), EMPTY_INVENTORY)


def _rank_of(brief, kind: str) -> int:
    return next(row["rank"] for row in brief["ranked_actions"] if row["kind"] == kind)


def test_the_dominant_cost_location_ranks_first() -> None:
    brief = _brief([_task(0, integer=549_695, regions=("conv_0",)),
                    _task(1, integer=100_000), _task(2, integer=50_000)])
    assert brief["ranked_actions"][0]["kind"] == "host_memory_hotspot"
    assert _rank_of(brief, "host_memory_hotspot") == 1


def test_a_measured_hotspot_outranks_an_unsized_regression() -> None:
    """The regression is real; it is simply not known to be big, and the hotspot is known to be."""
    brief = _brief([_task(0, integer=549_695), _task(1, integer=653_050)])
    hotspot = _rank_of(brief, "host_memory_hotspot")
    others = [row["rank"] for row in brief["ranked_actions"]
              if row["kind"] != "host_memory_hotspot" and row["magnitude_share"] is None]
    assert others, "the fixture must produce at least one unsized finding to outrank"
    assert hotspot < min(others)


def test_the_reported_share_is_the_largest_task_over_the_whole_lane() -> None:
    brief = _brief([_task(0, integer=750), _task(1, integer=250)])
    hotspot = next(row for row in brief["ranked_actions"] if row["kind"] == "host_memory_hotspot")
    assert hotspot["magnitude_share"] == 0.75
    assert "share of host-lane dynamic operations" in hotspot["magnitude_basis"]


def test_tasks_are_ranked_by_operations_not_by_bytes() -> None:
    """The pre-existing list is ordered by payload bytes; cost order is a different order."""
    brief = _brief([_task(0, integer=10), _task(1, integer=1_000), _task(2, integer=100)])
    ranked = next(row for row in brief["ranked_actions"]
                  if row["kind"] == "host_memory_hotspot")["evidence"]["cost_ranked_tasks"]
    assert [row["task_index"] for row in ranked] == [1, 2, 0]
    assert [row["dynamic_operations"] for row in ranked] == [1_000, 100, 10]


def test_each_cost_location_names_its_dominant_family() -> None:
    """Removing integer operations while leaving floating ones untouched was the observed failure."""
    brief = _brief([_task(0, integer=10, floating=990)])
    ranked = next(row for row in brief["ranked_actions"]
                  if row["kind"] == "host_memory_hotspot")["evidence"]["cost_ranked_tasks"]
    assert ranked[0]["dominant_family"] == "floating_arithmetic"
    assert ranked[0]["dominant_family_operations"] == 990


def test_cost_locations_name_their_source_regions() -> None:
    brief = _brief([_task(0, integer=100, regions=("conv_0", "dtype_cast_17"))])
    ranked = next(row for row in brief["ranked_actions"]
                  if row["kind"] == "host_memory_hotspot")["evidence"]["cost_ranked_tasks"]
    assert ranked[0]["source_regions"] == ["conv_0", "dtype_cast_17"]
    assert ranked[0]["source_region_count"] == 2


def test_ties_still_fall_back_to_the_name_so_the_order_is_deterministic() -> None:
    first = _brief([_task(0, integer=100), _task(1, integer=100)])
    second = _brief([_task(0, integer=100), _task(1, integer=100)])
    assert [row["kind"] for row in first["ranked_actions"]] == \
           [row["kind"] for row in second["ranked_actions"]]


def test_an_unmeasurable_host_lane_yields_no_share_rather_than_a_zero() -> None:
    """A zero share would order the finding LAST as though it were known to be free."""
    analysis = _analysis([_task(0, integer=100)])
    analysis["verified_global_plan_emission"]["host_activity"]["tasks"] = "UNKNOWN"
    brief = guidance_for_emission_analysis(analysis, EMPTY_INVENTORY)
    hotspot = next((row for row in brief["ranked_actions"]
                    if row["kind"] == "host_memory_hotspot"), None)
    if hotspot is not None:
        assert hotspot["magnitude_share"] is None
        assert hotspot["evidence"]["cost_ranked_tasks"] == []


def test_a_declined_lowering_still_outranks_the_biggest_hotspot() -> None:
    """Nothing to optimize matters while the program does not compile."""
    analysis = _analysis([_task(0, integer=1_000_000)])
    analysis["arms"]["candidate"] = {"status": "declined",
                                     "declined": {"op": "conv2d", "reason": "over budget"}}
    brief = guidance_for_emission_analysis(analysis, EMPTY_INVENTORY)
    assert brief["ranked_actions"][0]["kind"] == "whole_model_lowering_declined"


# --------------------------------------------------------------------------------------------
# Block signatures: the emitter SHAPE behind a cost, not only its family. Three levers worth
# 10-45% of the host lane each hid under one `integer_arithmetic` bucket until the operation
# sequences were read; the guidance must carry them so an author can name the primitive.
# --------------------------------------------------------------------------------------------

def test_dominant_block_signatures_are_ranked_and_carry_their_share() -> None:
    analysis = _analysis([_task(0, integer=100)])
    host = analysis["verified_global_plan_emission"]["host_activity"]
    host["dynamic_operations"] = {"integer_arithmetic": 1000, "floating_arithmetic": 0}
    host["block_signatures"] = [
        {"signature": "fsub ashr and and xor or", "blocks": 50, "trips": 100, "operations_per_trip": 6,
         "dynamic_total": 600, "dynamic_operations": {"integer_arithmetic": 500, "floating_arithmetic": 100}},
        {"signature": "udiv urem mul add", "blocks": 2, "trips": 100, "operations_per_trip": 4,
         "dynamic_total": 400, "dynamic_operations": {"integer_arithmetic": 400}},
    ]
    brief = guidance_for_emission_analysis(analysis, EMPTY_INVENTORY)
    hot = next(r for r in brief["ranked_actions"] if r["kind"] == "host_memory_hotspot")
    sigs = hot["evidence"]["dominant_block_signatures"]
    assert [s["signature"] for s in sigs] == ["fsub ashr and and xor or", "udiv urem mul add"]
    assert sigs[0]["share_of_host_dynamic_operations"] == 0.6
    assert sigs[1]["share_of_host_dynamic_operations"] == 0.4


def test_block_signatures_absent_yields_an_empty_list_not_a_crash() -> None:
    brief = _brief([_task(0, integer=100)])
    hot = next(r for r in brief["ranked_actions"] if r["kind"] == "host_memory_hotspot")
    assert hot["evidence"]["dominant_block_signatures"] == []
