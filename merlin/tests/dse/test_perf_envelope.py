"""Regression fixtures for :mod:`merlin.perf.envelope`.

Three things are pinned here, each of which a plausible implementation gets wrong:

* a resolved pipeline depth of **0** is a real answer (a combinational module) and must not read as
  UNKNOWN, while an unresolved one must not read as 0 -- the same bug one level up from
  ``float(x or 0)``;
* ``partial_depth`` is **never** a latency, so a module that refuses with an acyclic maximum still
  refuses, and the refusal names why;
* the composed bound never falls below the busiest single resource, whichever operator composed it.

The RTL facts used are the ones the depth walk actually produced on this host; where they are not
cached the test skips, because an absent cache is "did not run", not a pass.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

import pytest

from merlin.common.paths import env, repo_root
from merlin.perf.decompose import (
    UNKNOWN,
    ResourceKind,
    Unavailable,
    UnknownValueError,
    is_unknown,
)
from merlin.perf.envelope import (
    Basis,
    FixedTerm,
    Peak,
    ResourceDemand,
    ResourceTime,
    compose,
    envelope,
    resource_time,
)
from merlin.perf.headroom import Composition

# --- fixtures on disk ----------------------------------------------------------------------------


@functools.cache
def _suite() -> dict:
    root = env("MERLIN_MLC_DIR")
    if not root:
        pytest.skip("MERLIN_MLC_DIR unset -- the measured cycle suite lives in the mlc checkout")
    path = Path(root) / "mlc" / "validate" / "npu_model_suite.json"
    if not path.is_file():
        pytest.skip(f"measured cycle suite not present at {path}")
    assert repo_root().is_dir()
    return json.loads(path.read_text(encoding="utf-8"))


@functools.cache
def _timing(target: str) -> list[dict]:
    facts = pytest.importorskip("merlin.targetgen.rtl.facts")
    try:
        body = facts.load_facts(target).get("facts") or {}
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"RTL facts for the {target} archetype unavailable: {exc}")
    records = body.get("timing")
    if not records:
        pytest.skip(f"no cached timing facts for {target}: UNCACHED is not absent -- run "
                    "circt_introspect.dump_facts to build them (~70 s/target)")
    return records


def _module(target: str, name: str) -> dict:
    for r in _timing(target):
        if r["module"] == name:
            return r
    pytest.skip(f"module {name!r} is not in the {target} timing walk on this host")


# --- (c) fixed terms are intercepts, read from the frozen timing contract ------------------------


def test_a_resolved_depth_of_zero_is_a_real_answer_not_unknown():
    # A combinational module. `if not depth:` would read this as UNKNOWN, which is the bug.
    record = _module("gemmini", "Tile")
    assert record["pipeline_depth"] == 0 and record["n_cyclic"] == 0
    term = FixedTerm.from_pipeline_depth(record, name="mesh_tile_fill", resource="mesh")
    assert isinstance(term, FixedTerm)
    assert term.cycles == 0
    assert term.law == "rtl_pipeline_depth"


def test_an_unresolved_depth_refuses_and_never_reads_as_zero():
    # The container the accumulation feeds back through: 36/36 outputs cyclic, so no finite wiring
    # depth is its latency.
    record = _module("gemmini", "Mesh")
    assert record["pipeline_depth"] is None and record["partial_depth"] is None
    got = FixedTerm.from_pipeline_depth(record, name="mesh_fill", resource="mesh")
    assert isinstance(got, Unavailable)
    assert "feedback" in got.detail
    assert got.value is UNKNOWN
    with pytest.raises(UnknownValueError):
        float(got.value)


def test_partial_depth_is_refused_by_name_and_is_never_a_latency():
    # partial_depth == 0 on a module whose real latency is unbounded by wiring: the value that would
    # be most tempting to substitute, and the most wrong.
    record = _module("atlas", "AtlasCore")
    assert record["pipeline_depth"] is None
    assert record["partial_depth"] == 0
    got = FixedTerm.from_pipeline_depth(record, name="core_fill", resource="core")
    assert isinstance(got, Unavailable)
    assert "partial_depth" in got.detail
    assert "NOT this module's latency" in got.detail


def test_the_dominant_resource_of_each_archetype_refuses():
    # This is the hard constraint, not an edge case: on one archetype the movement engine refuses
    # while movement is the majority of every cycle count; on the other the mesh itself refuses.
    assert isinstance(FixedTerm.from_pipeline_depth(_module("atlas", "DmaEngine"),
                                                    name="dma_fill"), Unavailable)
    assert isinstance(FixedTerm.from_pipeline_depth(_module("gemmini", "Mesh"),
                                                    name="mesh_fill"), Unavailable)


def test_unreachable_rtl_is_a_different_refusal_from_a_walked_module_that_declined():
    got = FixedTerm.from_pipeline_depth(None, name="anything")
    assert isinstance(got, Unavailable)
    assert "UNCACHED is not ABSENT" in got.detail


def test_a_malformed_timing_record_raises_rather_than_defaulting():
    with pytest.raises(ValueError, match="pipeline_depth"):
        FixedTerm.from_pipeline_depth({"module": "X"}, name="x")


def test_the_systolic_fill_law_and_the_rtl_walk_agree():
    # Two independent derivations of the same intercept: the structural law 2*DIM-2 over the
    # measured geometry, and the depth walk over the RTL.
    dim = _suite()["_meta"]["mxu_dim"]
    law = FixedTerm.from_fill_law("systolic_2d", dim, name="mxu_fill", resource="mxu")
    assert law.cycles == 2 * dim - 2 == 62
    walked = FixedTerm.from_pipeline_depth(_module("atlas", "SystolicArray"), name="mxu_fill",
                                           resource="mxu")
    assert isinstance(walked, FixedTerm)
    assert walked.cycles == law.cycles


# --- (a) peaks: derived, falsified, or UNKNOWN ---------------------------------------------------


def _dma_samples() -> list[tuple[float, float]]:
    return [(k["arc"]["reads"] + k["arc"]["writes"], k["arc"]["dma_busy"])
            for k in _suite()["kernels"].values()]


def test_the_movement_peak_is_an_observed_ceiling_and_bounds_every_workload():
    peak = Peak.observed_ceiling("dma", _dma_samples(), unit="beats",
                                 provenance="per-cycle activity decomposition")
    assert peak.known and peak.is_ceiling
    assert peak.n_samples == 21
    assert round(peak.value, 6) == 0.999024
    # By construction, not by luck: demand / max-observed-rate <= busy on every point.
    for demand, busy in _dma_samples():
        assert demand / peak.value <= busy + 1e-9


def test_a_demand_that_does_not_drive_occupancy_is_refuted_rather_than_priced():
    # Vector-op COUNT looks like a demand for the vector engine, and one workload issues seven of
    # them while the engine is charged zero cycles. No positive rate can bound that, so the peak is
    # UNKNOWN -- the alternative is a model that predicts more cycles than the hardware spent.
    kernels = _suite()["kernels"]
    samples = [(sum(1 for fam, _m, _i in k["op_stream"] if fam == "Vector"), k["arc"]["vpu"])
               for k in kernels.values()]
    assert any(d > 0 and b == 0 for d, b in samples), "the refuting point must be in the corpus"
    peak = Peak.observed_ceiling("vpu", samples, unit="ops", provenance="program op stream")
    assert not peak.known
    assert "refuted" in peak.reason


def test_one_observation_cannot_establish_a_rate():
    peak = Peak.observed_ceiling("x", [(10, 5), (0, 0)], unit="ops", provenance="p")
    assert not peak.known
    assert "at least two points" in peak.reason


def test_a_non_positive_peak_is_refused():
    with pytest.raises(ValueError, match="not a"):
        Peak(resource="x", value=0.0, unit="ops", evidence_kind="measured", provenance="p")


def test_an_unknown_peak_must_say_why():
    with pytest.raises(ValueError, match="no reason"):
        Peak(resource="x", value=UNKNOWN, unit="ops", evidence_kind="measured", provenance="p")


# --- (b) the bytes axis is MOVED bytes -----------------------------------------------------------


def _peak(value: float = 1.0) -> Peak:
    return Peak(resource="dma", value=value, unit="bytes", evidence_kind="measured",
                provenance="observed")


def test_an_algorithmic_byte_demand_without_amplification_is_unknown_not_optimistic():
    d = ResourceDemand("dma", ResourceKind.MOVEMENT, 4096, "bytes", basis=Basis.ALGORITHMIC)
    t = resource_time(d, _peak())
    assert not t.known
    assert "moved bytes" in t.reason
    with pytest.raises(UnknownValueError):
        float(t.cycles)


def test_an_algorithmic_demand_is_priced_only_at_its_measured_amplification():
    d = ResourceDemand("dma", ResourceKind.MOVEMENT, 4096, "bytes", basis=Basis.ALGORITHMIC,
                       amplification=16.0)
    assert resource_time(d, _peak()).cycles == 4096 * 16.0
    assert resource_time(ResourceDemand("dma", ResourceKind.MOVEMENT, 4096 * 16, "bytes"),
                         _peak()).cycles == 4096 * 16.0


def test_applying_an_amplification_to_a_moved_demand_is_refused_as_double_counting():
    with pytest.raises(ValueError, match="twice"):
        ResourceDemand("dma", ResourceKind.MOVEMENT, 4096, "bytes", basis=Basis.MOVED,
                       amplification=16.0)


def test_an_unknown_peak_propagates_into_the_time_with_its_own_reason():
    d = ResourceDemand("dma", ResourceKind.MOVEMENT, 4096, "bytes")
    t = resource_time(d, Peak.unknown("dma", "bytes", "the engine's module refuses: 21/21 cyclic",
                                      provenance="rtl walk"))
    assert not t.known
    assert "21/21 cyclic" in t.reason


# --- composition: never defaulted, never below the floor -----------------------------------------


def _times(*pairs: tuple[str, float]) -> list[ResourceTime]:
    return [ResourceTime(resource=n, kind=ResourceKind.COMPUTE, cycles=c, unit="cycles",
                         basis=Basis.MOVED) for n, c in pairs]


def test_the_three_operators_give_the_three_documented_answers():
    ts = _times(("a", 100.0), ("b", 40.0))
    assert compose(ts, operator=Composition.SUM, eta=0.0).cycles == 140.0
    assert compose(ts, operator=Composition.MAX, eta=1.0).cycles == 100.0
    assert compose(ts, operator=Composition.PARTIAL, eta=0.5).cycles == 120.0


def test_max_understates_sum_by_the_smaller_resource_which_is_the_flattering_direction():
    ts = _times(("movement", 2050.0), ("compute", 158.0))
    s = float(compose(ts, operator=Composition.SUM, eta=0.0).cycles)
    m = float(compose(ts, operator=Composition.MAX, eta=1.0).cycles)
    assert s - m == 158.0
    assert m < s, "the textbook operator is the optimistic one"


def test_an_operator_must_be_passed_and_cannot_be_a_bare_string():
    with pytest.raises(TypeError, match="never defaulted"):
        compose(_times(("a", 1.0)), operator="max", eta=1.0)


def test_eta_outside_zero_to_one_is_refused():
    with pytest.raises(ValueError, match="realised fraction"):
        compose(_times(("a", 1.0)), operator=Composition.PARTIAL, eta=1.5)


def test_no_composition_falls_below_the_busiest_single_resource():
    # Three resources: the pairwise overlap credit would take the total below the floor, so it is
    # clamped -- nothing finishes before its slowest resource does.
    ts = _times(("a", 100.0), ("b", 90.0), ("c", 80.0))
    c = compose(ts, operator=Composition.PARTIAL, eta=1.0)
    assert c.clamped_to_floor
    assert c.cycles == c.floor_cycles == 100.0


def test_workload_intercepts_sit_outside_the_overlap_and_resource_fills_inside_it():
    reset = FixedTerm(name="reset", cycles=12)
    ts = _times(("a", 100.0), ("b", 40.0))
    assert compose(ts, operator=Composition.MAX, eta=1.0,
                   workload_fixed=[reset]).cycles == 112.0
    with pytest.raises(ValueError, match="inside that engine"):
        compose(ts, operator=Composition.SUM, eta=0.0,
                workload_fixed=[FixedTerm(name="fill", cycles=62, resource="a")])


def test_a_fixed_term_of_zero_cycles_is_legal_and_a_negative_one_is_not():
    assert FixedTerm(name="combinational", cycles=0).cycles == 0
    with pytest.raises(ValueError, match=">= 0"):
        FixedTerm(name="bad", cycles=-1)


# --- (d) UNKNOWN propagates, with the partial bound under its own name ---------------------------


def test_an_unresolved_resource_makes_the_bound_unknown_but_keeps_a_partial_one():
    ts = _times(("a", 100.0)) + [ResourceTime(
        resource="b", kind=ResourceKind.MOVEMENT, cycles=UNKNOWN, unit="cycles",
        basis=Basis.MOVED, reason="its module refuses")]
    c = compose(ts, operator=Composition.SUM, eta=0.0)
    assert c.cycles is UNKNOWN
    assert c.partial_cycles == 100.0
    assert c.unresolved == ("b",)
    with pytest.raises(UnknownValueError):
        float(c.cycles)


def test_the_partial_bound_carries_a_different_name_from_the_full_one():
    # Same discipline as the RTL walk's pipeline_depth vs partial_depth: two differently derived
    # numbers must not share one field.
    env = envelope("w", _times(("a", 10.0)) + [ResourceTime(
        resource="b", kind=ResourceKind.COMPUTE, cycles=UNKNOWN, unit="cycles", basis=Basis.MOVED,
        reason="unresolved")], operator=Composition.SUM, eta=0.0)
    assert env.lower_bound_cycles is UNKNOWN
    assert env.partial_lower_bound_cycles == 10.0
    assert env.to_dict()["lower_bound_cycles"] == "UNKNOWN"
    assert env.to_dict()["partial_lower_bound_cycles"] == 10.0


# --- (a) the generalized ridge point --------------------------------------------------------------


def test_the_envelope_names_the_binding_resource_and_the_margin_to_second():
    env = envelope("w", _times(("movement", 2050.0), ("compute", 158.0), ("other", 40.0)),
                   operator=Composition.SUM, eta=0.0)
    assert env.limiter == "movement"
    assert env.limiter_cycles == 2050.0
    assert env.margin_to_second == 2050.0 - 158.0
    assert round(env.margin_share, 4) == round(1 - 158.0 / 2050.0, 4)
    assert not env.limiter_is_provisional
    assert list(env.terms) == ["movement", "compute", "other"], "ranked, like mlc's limiter terms"


def test_a_near_tie_reports_a_small_margin_rather_than_a_confident_binder():
    env = envelope("w", _times(("a", 1000.0), ("b", 999.0)), operator=Composition.SUM, eta=0.0)
    assert env.limiter == "a"
    assert env.margin_to_second == 1.0
    assert env.margin_share < 0.002


def test_the_limiter_is_provisional_while_any_resource_is_unresolved():
    env = envelope("w", _times(("a", 10.0)) + [ResourceTime(
        resource="dma", kind=ResourceKind.MOVEMENT, cycles=UNKNOWN, unit="cycles",
        basis=Basis.MOVED, reason="peak not derivable")], operator=Composition.SUM, eta=0.0)
    assert env.limiter == "a"
    assert env.limiter_is_provisional, "an unresolved resource could bind instead"
    assert env.unresolved == ("dma",)


def test_an_envelope_over_nothing_resolved_has_an_unknown_limiter():
    env = envelope("w", [ResourceTime(resource="dma", kind=ResourceKind.MOVEMENT, cycles=UNKNOWN,
                                      unit="cycles", basis=Basis.MOVED, reason="no peak")],
                   operator=Composition.SUM, eta=0.0)
    assert env.limiter is UNKNOWN
    assert is_unknown(env.margin_to_second)
