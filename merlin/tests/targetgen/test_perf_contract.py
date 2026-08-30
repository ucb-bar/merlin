"""The performance contract: what each resource costs, and -- mostly -- what is not known.

Per-resource peaks are not generally structurally derivable, and UNKNOWN is the COMMON case. So the
properties worth testing are negative ones: a refused walk must not become a zero, a partial answer
must not become a whole one, two disagreeing sources must not silently elect a winner, and a
resource the evidence never reached must not vanish from the contract as if the machine did not have
it.

The fixtures mirror ``test_perf_profile``: two synthetic machines of different archetypes (so the
anti-overfit property holds on any host), plus the two real targets as regression fixtures, which
SKIP rather than pass where their facts artifact is not on this host.
"""
from __future__ import annotations

import json

import pytest

from merlin.perf.contract import PerformanceContract, contract_table, derive_contract
from merlin.perf.decompose import ResourceKind
from merlin.perf.term import UNKNOWN, UnknownValueError

from test_perf_profile import (HOST_QUEUED_TARGET, SELF_HOSTED_TARGET, _host_queued_facts,
                            _host_queued_residual, _self_hosted_facts, _self_hosted_residual,
                            _timing)


def _synthetic_contract(kind: str, **over) -> PerformanceContract:
    if kind == "self_hosted":
        facts, residual = _self_hosted_facts(), _self_hosted_residual()
    else:
        facts, residual = _host_queued_facts(), _host_queued_residual()
    return derive_contract(f"a_{kind}_machine", facts=over.get("facts", facts),
                           residual=over.get("residual", residual))


def _real_or_skip(target: str) -> PerformanceContract:
    c = derive_contract(target)
    if not c.profile.sources.body:
        pytest.skip(f"no RTL facts artifact for {target!r} on this host: the fixture could not "
                    "run, which is not_run and never a pass")
    if c.profile.timing.status != "present":
        pytest.skip(f"{target}: the timing fact class is {c.profile.timing.status} on this host "
                    "(uncached, not absent) -- the fixture could not run")
    return c


# ---------------------------------------------------------------------------------------------
# THE ANTI-OVERFIT GATE
# ---------------------------------------------------------------------------------------------


def test_two_archetypes_get_different_contracts_from_one_code_path():
    a = _synthetic_contract("self_hosted")
    b = _synthetic_contract("host_queued")

    # Different geometry -> different peak, both structurally grounded.
    assert a.resource("mxu").term("peak_macs_per_cycle").value == 32 * 32
    assert b.resource("systolic_mesh").term("peak_macs_per_cycle").value == 16 * 16

    # A feed-forward array resolves its container's depth; a weight-stationary one refuses it.
    assert a.resource("mxu").term("container_depth_cycles").value == 31
    depth = b.resource("systolic_mesh").term("container_depth_cycles")
    assert depth.is_unknown and "feedback" in depth.unknown_reason

    # Only the machine whose facts evidence a movement engine gets one priced.
    assert b.resources_of(ResourceKind.MOVEMENT)
    assert not a.resources_of(ResourceKind.MOVEMENT)

    # And only the machine with discovered memories gets a grounded capacity.
    assert b.resource("scratchpad").term("capacity_bytes").value == 262144
    assert a.resource("operand_store").term("capacity_bytes").is_unknown


def test_real_targets_reproduce_their_measured_terms():
    a = _real_or_skip(SELF_HOSTED_TARGET)
    b = _real_or_skip(HOST_QUEUED_TARGET)

    a_unit = a.resources_of(ResourceKind.COMPUTE)[0]
    b_unit = b.resources_of(ResourceKind.COMPUTE)[0]
    a_array = a.profile.sources.arrays()[0]
    b_array = b.profile.sources.arrays()[0]

    assert a_unit.term("peak_macs_per_cycle").value == a_array["rows"] * a_array["cols"]
    assert b_unit.term("peak_macs_per_cycle").value == b_array["rows"] * b_array["cols"]
    assert a_unit.term("peak_macs_per_cycle").value != b_unit.term("peak_macs_per_cycle").value

    # The measured asymmetry: one array's container depth resolves, the other's refuses outright.
    assert a_unit.term("container_depth_cycles").value == a_array["rows"] - 1
    assert b_unit.term("container_depth_cycles").is_unknown
    # ... and the second machine's array ELEMENT is combinational: a real 0, not a hole.
    assert b_unit.term("element_latency_cycles").value == 0

    table = contract_table([a, b])
    assert "UNKNOWN" in table and str(a_array["rows"] * a_array["cols"]) in table


# ---------------------------------------------------------------------------------------------
# UNKNOWN is a distinct state, and it is the common one
# ---------------------------------------------------------------------------------------------


def test_a_refused_walk_never_becomes_a_zero():
    c = _synthetic_contract("host_queued")
    fill = c.resource("systolic_mesh").term("container_depth_cycles")
    assert fill.value is UNKNOWN
    with pytest.raises(UnknownValueError):
        float(fill.value)
    with pytest.raises(UnknownValueError):
        _ = fill.value or 0                 # the exact line this whole design exists to break
    assert fill.unknown_reason, "a refusal must say why"


def test_a_zero_depth_term_is_a_value_not_a_hole():
    c = _synthetic_contract("host_queued")
    element = c.resource("systolic_mesh").term("element_latency_cycles")
    assert element.value == 0
    assert not element.is_unknown
    assert element.unknown_reason == ""
    assert float(element.value) == 0.0      # a real number, and it may be read as one


def test_a_partial_depth_does_not_fill_a_refused_fill():
    facts = _host_queued_facts()
    facts["facts"]["timing"] = _timing([
        {"module": "Mesh", "pipeline_depth": None, "partial_depth": 12, "n_outputs": 36,
         "n_cyclic": 4, "evidence": "4 of 36 hw.output operands are reached through feedback"},
        {"module": "Tile", "pipeline_depth": 0, "n_outputs": 10},
    ])
    c = _synthetic_contract("host_queued", facts=facts)
    depth = c.resource("systolic_mesh").term("container_depth_cycles")
    assert depth.is_unknown
    assert "12" not in str(depth.to_dict()["value"])
    # ... and it does not leak into the fill's lower bound either.
    assert c.resource("systolic_mesh").term("pipeline_fill_cycles").bounds.lower is UNKNOWN


def test_deleting_the_array_fact_makes_the_peak_unknown_not_a_default():
    facts = _host_queued_facts()
    del facts["facts"]["arrays"]
    c = _synthetic_contract("host_queued", facts=facts)
    peak = c.resource("systolic_mesh").term("peak_macs_per_cycle")
    assert peak.is_unknown and "fabricated" in peak.unknown_reason
    assert c.resource("systolic_mesh").term("container_depth_cycles").is_unknown


def test_the_fill_is_never_the_container_depth_but_is_bounded_below_by_it():
    """The container holds the elements; the datapath an operand traverses may be deeper.

    On one measured elaboration the element grid resolves ``rows-1`` while the enclosing datapath
    is ``2*DIM-2`` -- so publishing the inner depth as the fill would understate every small tile,
    in the flattering direction. The resolved depth is a LOWER BOUND, recorded as one.
    """
    c = _synthetic_contract("self_hosted")
    depth = c.resource("mxu").term("container_depth_cycles")
    fill = c.resource("mxu").term("pipeline_fill_cycles")
    assert depth.value == 31
    assert fill.is_unknown, "a resolved container depth is not the datapath's fill"
    assert fill.bounds.lower == 31
    assert fill.bounds.upper is UNKNOWN, "an unknown upper end is not an infinite one"
    assert "LOWER BOUND" in fill.unknown_reason

    # Where the container itself refused, there is no bound to offer either.
    b = _synthetic_contract("host_queued")
    b_fill = b.resource("systolic_mesh").term("pipeline_fill_cycles")
    assert b_fill.is_unknown and b_fill.bounds.lower is UNKNOWN


def test_the_peak_counts_multipliers_when_the_facts_state_them():
    facts = _host_queued_facts()
    facts["facts"]["arrays"][0]["mac_idiom"] = {"muls": 2, "adds": 7, "regs": 3}
    c = _synthetic_contract("host_queued", facts=facts)
    peak = c.resource("systolic_mesh").term("peak_macs_per_cycle")
    assert peak.value == 16 * 16 * 2
    assert "mac_idiom" in peak.provenance.evidence[0]


def test_the_initiation_interval_is_never_the_completion_latency():
    """Two different numbers. A flat vendor latency table conflates them; this must not."""
    for kind in ("self_hosted", "host_queued"):
        c = _synthetic_contract(kind)
        unit = c.resources_of(ResourceKind.COMPUTE)[0]
        ii = unit.term("initiation_interval_cycles")
        assert ii.is_unknown
        assert "interval between successive issues" in ii.unknown_reason


# ---------------------------------------------------------------------------------------------
# Capacity: disagreeing sources elect no winner
# ---------------------------------------------------------------------------------------------


def test_a_capacity_two_sources_disagree_on_stays_unknown():
    residual = dict(_host_queued_residual())
    residual["memory_model"] = {**residual["memory_model"], "scratchpad_bytes": 1048576}
    c = _synthetic_contract("host_queued", residual=residual)
    cap = c.resource("scratchpad").term("capacity_bytes")
    assert cap.is_unknown
    assert "262144" in cap.unknown_reason and "1048576" in cap.unknown_reason
    assert "DISAGREE" in cap.unknown_reason
    # Both sources are still named, so the disagreement is actionable rather than merely absent.
    assert len(cap.provenance.evidence) == 2
    # The memory the sources AGREE about is unaffected.
    assert c.resource("accumulator").term("capacity_bytes").value == 65536


def test_agreeing_sources_produce_a_grounded_capacity():
    residual = dict(_host_queued_residual())
    residual["memory_model"] = {**residual["memory_model"], "scratchpad_bytes": 262144}
    c = _synthetic_contract("host_queued", residual=residual)
    assert c.resource("scratchpad").term("capacity_bytes").value == 262144


def test_a_declared_only_capacity_is_marked_as_declared():
    residual = dict(_self_hosted_residual())
    residual["memory_model"] = {**residual["memory_model"], "operand_store_bytes": 1572864}
    c = _synthetic_contract("self_hosted", residual=residual)
    cap = c.resource("operand_store").term("capacity_bytes")
    assert cap.value == 1572864
    assert cap.provenance.kind == "assumed"          # a declaration is intent, not evidence
    assert "DECLARED" in cap.validity.validated_regime


def test_a_target_with_no_memory_fact_still_carries_the_missing_capacity():
    c = _synthetic_contract("self_hosted")
    cap = c.resource("operand_store").term("capacity_bytes")
    assert cap.is_unknown
    assert "never a fact" in cap.unknown_reason


# ---------------------------------------------------------------------------------------------
# Gaps: what the contract cannot say, said out loud
# ---------------------------------------------------------------------------------------------


def test_an_unevidenced_movement_engine_is_a_named_gap_not_a_silence():
    c = _synthetic_contract("self_hosted")
    gaps = {g.what: g for g in c.gaps}
    assert "a data-movement resource" in gaps
    gap = gaps["a data-movement resource"]
    assert gap.missing
    assert "hole in the evidence rather than in the machine" in gap.detail


def test_the_composition_operator_is_never_defaulted():
    for kind in ("self_hosted", "host_queued"):
        c = _synthetic_contract(kind)
        whats = " ".join(g.what for g in c.gaps)
        assert "composition operator" in whats
        detail = " ".join(g.detail for g in c.gaps)
        assert "NEVER defaulted to max" in detail


def test_an_unevidenced_elaboration_is_a_gap():
    facts = _host_queued_facts()
    facts["inputs"] = {"hw_mlir": "some_soc.hw.mlir", "hw_sha": "missing"}
    c = _synthetic_contract("host_queued", facts=facts)
    assert any("elaboration" in g.what for g in c.gaps)


def test_an_uncached_timing_class_is_a_gap_that_says_so():
    facts = _host_queued_facts()
    del facts["facts"]["timing"]
    c = _synthetic_contract("host_queued", facts=facts)
    detail = " ".join(g.detail for g in c.gaps)
    assert "UNCACHED is not the same as absent" in detail


# ---------------------------------------------------------------------------------------------
# Every term carries provenance and a validity domain
# ---------------------------------------------------------------------------------------------


def test_every_term_names_its_elaboration_and_its_evidence():
    for kind in ("self_hosted", "host_queued"):
        c = _synthetic_contract(kind)
        assert c.terms(), "a contract with no terms is not a contract"
        for name, term in c.terms().items():
            assert term.provenance.evidence, f"{name} cites nothing"
            assert term.validity.validated_regime, f"{name} claims no regime"
            assert "elaboration" in term.validity.validated_regime, f"{name} names no elaboration"
            assert term.unit, f"{name} has no unit"
            if term.is_unknown:
                assert term.unknown_reason, f"{name} is UNKNOWN with no reason"


def test_the_peak_rate_says_it_cannot_price_a_small_tile_alone():
    c = _synthetic_contract("self_hosted")
    peak = c.resource("mxu").term("peak_macs_per_cycle")
    assert peak.provenance.kind == "structural_bound"
    assert "full" in peak.validity.validated_regime or "every one of the" in peak.validity.validated_regime
    assert "fill intercept" in peak.validity.weak_regime
    assert peak.bounds.upper == peak.value and peak.bounds.lower == 0


def test_the_fixed_intercept_is_first_class_and_unknown():
    c = _synthetic_contract("host_queued")
    fixed = c.resources_of(ResourceKind.FIXED)
    assert fixed, "the intercept is a resource, not noise"
    startup = fixed[0].term("startup_cycles")
    assert startup.is_unknown
    assert "isolation experiment" in startup.unknown_reason


def test_the_contract_serializes_and_lists_its_unknowns():
    c = _synthetic_contract("host_queued")
    d = c.to_dict()
    assert d["target"] == "a_host_queued_machine"
    assert {r["name"] for r in d["resources"]} >= {"systolic_mesh", "data_movement", "scratchpad"}
    assert d["unknown_terms"], "the backlog is the product; an empty one would be a lie here"
    assert json.loads(json.dumps(d))          # round-trips
    for key, why in c.unknown_terms().items():
        assert why, f"{key} is UNKNOWN with no reason"


def test_contract_terms_are_qualified_so_they_can_share_one_record():
    c = _synthetic_contract("host_queued")
    assert "systolic_mesh.peak_macs_per_cycle" in c.terms()
    assert "data_movement.base_latency_cycles" in c.terms()
