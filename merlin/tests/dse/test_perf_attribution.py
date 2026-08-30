"""Regression fixtures for :mod:`merlin.perf.attribution`.

The measured fixture: the activity buckets **partition** the timeline, and they over-count it by
exactly one cycle on all 21 workloads (``dma + compute + idle == total + 1``). Two consequences that
this file pins:

1. attributed components sum to the measured total **exactly**, which is only true because the
   residual is a first-class component that is emitted even when -- especially when -- it is a
   constant ``-1`` nobody would have gone looking for;
2. a partition **cannot** measure overlap. It returns zero concurrency by construction, so no
   overlap term is ever derived from it, and every result says so.

The bucket vocabulary is mlc's ``compute / dma / stall / control / host``, cross-checked against the
real thing when mlc is importable and reported as "did not run" when it is not.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

import pytest

from merlin.common.paths import env, repo_root
from merlin.perf.attribution import (
    BUCKETS,
    RESIDUAL,
    OptimizationFamily,
    attribute,
    attribute_corpus,
    buckets_from_kinds,
    buckets_match_reference,
)
from merlin.perf.decompose import (
    UNKNOWN,
    ResourceKind,
    Unavailable,
    activity_from_busy,
    is_unknown,
)
from merlin.perf.envelope import Basis, Peak, ResourceDemand, ResourceTime, envelope, resource_time
from merlin.perf.headroom import Composition, concurrency_traits, headroom

BUCKET_KINDS = {
    "dma": ResourceKind.MOVEMENT,
    "mxu": ResourceKind.COMPUTE,
    "vpu": ResourceKind.COMPUTE,
    "none": ResourceKind.FIXED,
}


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


def _sources():
    out = []
    for name, body in _suite()["kernels"].items():
        arc = body["arc"]
        out.append(activity_from_busy(
            name, arc["truth"],
            {"dma": arc["dma_busy"], "mxu": arc["mxu"], "vpu": arc["vpu"], "none": arc["none"]},
            BUCKET_KINDS, partitioned=True, completion_observable=True,
            provenance="per-cycle activity decomposition from the cycle-accurate model"))
    return out


def _buckets() -> dict[str, str]:
    # Cycles charged to no engine could be issue stalls or sequencer overhead and occupancy alone
    # cannot separate them, so the caller states which; the mapping is never read off a name.
    return buckets_from_kinds(BUCKET_KINDS, fixed_bucket="control")


# --- the vocabulary is mlc's, 1:1 -----------------------------------------------------------------


def test_the_five_buckets_are_exactly_mlcs():
    assert BUCKETS == ("compute", "dma", "stall", "control", "host")
    got = buckets_match_reference()
    if isinstance(got, Unavailable):
        pytest.skip(f"mlc not importable, so the cross-check did NOT RUN: {got.detail}")
    assert got is True


def test_kinds_that_decide_a_bucket_do_and_kinds_that_do_not_refuse():
    assert buckets_from_kinds({"a": ResourceKind.COMPUTE, "b": ResourceKind.MOVEMENT},
                              fixed_bucket="control") == {"a": "compute", "b": "dma"}
    with pytest.raises(ValueError, match="does not decide a bucket"):
        buckets_from_kinds({"x": ResourceKind.OTHER}, fixed_bucket="control")
    with pytest.raises(ValueError, match="not one of"):
        buckets_from_kinds({"x": ResourceKind.FIXED}, fixed_bucket="idle")


def test_an_unmapped_resource_raises_rather_than_being_folded_somewhere_plausible():
    src = _sources()[0]
    with pytest.raises(ValueError, match="no bucket declared"):
        attribute(src, buckets={"dma": "dma"})


# --- the buckets sum to the arc total EXACTLY -----------------------------------------------------


def test_attributed_buckets_sum_to_the_arc_total_exactly_on_all_21():
    corpus = attribute_corpus(_sources(), buckets=_buckets())
    assert len(corpus.workloads) == 21
    assert corpus.closes
    for name, attr in corpus.workloads.items():
        assert attr.attributed_cycles == attr.total_cycles, name


def test_the_residual_is_a_constant_minus_one_and_never_vanishes():
    # The buckets over-count by exactly one cycle everywhere: a fencepost, not noise. An
    # implementation that only reported a non-zero residual would still show this one -- but one
    # that dropped it "because it rounds to nothing" would hide a systematic accounting offset.
    corpus = attribute_corpus(_sources(), buckets=_buckets())
    assert set(corpus.residual_cycles.values()) == {-1}
    assert corpus.residual_is_constant == -1
    for attr in corpus.workloads.values():
        assert attr.residual.measured_cycles == -1
        assert attr.residual.evidence_kind == "assumed"
        assert is_unknown(attr.residual.family)


def test_every_bucket_is_present_even_at_zero_cycles():
    attr = attribute(_sources()[0], buckets=_buckets())
    assert [c.bucket for c in attr.components] == list(BUCKETS) + [RESIDUAL]
    # No host or stall bucket exists on this instrument; zero is a fact and stays visible.
    assert attr.component("host").measured_cycles == 0
    assert attr.component("stall").measured_cycles == 0


def test_a_component_list_missing_a_bucket_is_rejected_at_construction():
    from merlin.perf.attribution import Attribution, GapComponent

    one = GapComponent(bucket="compute", measured_cycles=0, structural_cycles=UNKNOWN,
                       gap_cycles=UNKNOWN, evidence_kind="assumed", family=UNKNOWN, rationale="")
    with pytest.raises(ValueError, match="every bucket plus the residual"):
        Attribution(workload="w", total_cycles=0, components=(one,), partitioned=True)


# --- a partition cannot measure overlap -----------------------------------------------------------


def test_a_partitioned_source_reports_that_overlap_is_not_derivable_from_it():
    for attr in attribute_corpus(_sources(), buckets=_buckets()).workloads.values():
        assert attr.partitioned is True
        assert attr.overlap_derivable is False


# --- the gap, and the family it implies -----------------------------------------------------------


def _dma_peak() -> Peak:
    samples = [(k["arc"]["reads"] + k["arc"]["writes"], k["arc"]["dma_busy"])
               for k in _suite()["kernels"].values()]
    return Peak.observed_ceiling("dma", samples, unit="beats",
                                 provenance="per-cycle activity decomposition")


def _envelope_for(name: str):
    """dma from a measured rate ceiling, mxu from the structural fill law + the program's schedule,
    vpu from neither (its peak is refuted and its module refuses)."""
    from merlin.perf.record import (
        compose_unit_busy,
        derive_delay_mnemonic,
        derive_unit_roles,
        fill_cycles,
    )

    suite = _suite()
    kernels = suite["kernels"]
    streams = [k["op_stream"] for k in kernels.values() if k.get("op_stream")]
    delay = derive_delay_mnemonic(streams)
    roles = derive_unit_roles(streams, "MatrixSystolic", delay)
    fill = fill_cycles("systolic_2d", suite["_meta"]["mxu_dim"])
    entry = kernels[name]
    arc = entry["arc"]

    beats = arc["reads"] + arc["writes"]
    times = [resource_time(
        ResourceDemand("dma", ResourceKind.MOVEMENT, beats, "beats", basis=Basis.MOVED,
                       provenance="measured read/write beats"), _dma_peak())]
    busy = compose_unit_busy(entry["op_stream"], roles, fill, delay)
    times.append(ResourceTime(
        resource="mxu", kind=ResourceKind.COMPUTE,
        cycles=UNKNOWN if busy.cycles is None else float(busy.cycles), unit="cycles",
        basis=Basis.MOVED, fixed_cycles=fill, evidence_kind="structural_bound",
        provenance=f"fill={fill} plus the delays the program schedules",
        reason="" if busy.cycles is not None else busy.reason))
    vops = sum(1 for fam, _m, _i in entry["op_stream"] if fam == "Vector")
    times.append(resource_time(
        ResourceDemand("vpu", ResourceKind.COMPUTE, vops, "ops", basis=Basis.MOVED,
                       provenance="program op stream"),
        Peak.unknown("vpu", "ops",
                     "its module refuses (31/31 outputs cyclic) and the op-count demand proxy is "
                     "refuted by a workload that issues seven and is charged zero cycles",
                     provenance="rtl timing walk + activity decomposition")))
    times.append(ResourceTime(
        resource="none", kind=ResourceKind.FIXED, cycles=float(suite["_meta"]["reset_cycles"]),
        unit="cycles", basis=Basis.MOVED, evidence_kind="measured",
        provenance="reset_cycles declared by the measurement source"))
    return envelope(name, times, operator=Composition.SUM, eta=0.0)


def test_a_bucket_at_its_structural_bound_implies_no_family_which_is_a_finding():
    # This workload issues no vector ops, so the compute bucket resolves entirely: 62 fill + 96
    # scheduled delay = 158, and the engine measured exactly 158.
    src = next(s for s in _sources() if s.workload == "matmul")
    attr = attribute(src, buckets=_buckets(), envelope=_envelope_for("matmul"))
    compute = attr.component("compute")
    assert compute.measured_cycles == 158
    assert compute.structural_cycles == 158.0
    assert compute.gap_cycles == 0.0
    assert compute.family is OptimizationFamily.NONE


def test_an_unresolved_resource_makes_its_bucket_gap_and_family_unknown():
    # A workload that issues vector ops: the vpu peak is UNKNOWN, so the whole compute bucket is.
    src = next(s for s in _sources() if s.workload == "gemma_attention")
    attr = attribute(src, buckets=_buckets(), envelope=_envelope_for("gemma_attention"))
    compute = attr.component("compute")
    assert compute.measured_cycles == 158 + 298
    assert is_unknown(compute.structural_cycles)
    assert is_unknown(compute.gap_cycles)
    assert is_unknown(compute.family), "no family follows from a distance that is not established"
    assert "not established" in compute.rationale


def test_the_movement_gap_names_a_family_only_when_the_amplification_split_resolves():
    src = next(s for s in _sources() if s.workload == "matmul")
    env = _envelope_for("matmul")
    bare = attribute(src, buckets=_buckets(), envelope=env).component("dma")
    # Measured 2054 movement cycles against a structural 2050.0 from the observed rate ceiling.
    assert bare.measured_cycles == 2054
    assert bare.gap_cycles == pytest.approx(4.0, abs=0.01)
    assert is_unknown(bare.family), "the ratio alone cannot say which half of it survives tiling"
    assert "granule" in bare.rationale

    # The split is what names the family. Operand sizes are not in this measurement artifact, so
    # the two observations below are SYNTHETIC and exercise only the routing: one where the excess
    # is mostly the fixed per-command block, one where it is mostly refetch.
    from merlin.perf.amplification import MovementObservation
    from merlin.perf.amplification import amplification as amplify
    from merlin.perf.decompose import Trait

    trait = Trait("explicit_data_movement", True,
                  evidence="the program issues movement commands under compiler control")
    # A tiny payload inside a large fixed command block: granularity 40.96x vs redundancy 2x.
    mostly_granule = amplify(MovementObservation(
        workload="matmul", moved_bytes=8192, useful_bytes=100, transfers=2, provenance="synthetic"),
        trait=trait)
    # Eight commands carrying a payload one command could hold: redundancy 8x vs granularity 1x.
    mostly_refetch = amplify(MovementObservation(
        workload="matmul", moved_bytes=8192, useful_bytes=4096, transfers=8, provenance="synthetic"),
        trait=trait)
    got_granule = attribute(src, buckets=_buckets(), envelope=env,
                            amplification=mostly_granule).component("dma")
    got_refetch = attribute(src, buckets=_buckets(), envelope=env,
                            amplification=mostly_refetch).component("dma")
    assert got_granule.family is OptimizationFamily.TRANSFER_GRANULARITY
    assert "amortizes away" in got_granule.rationale
    assert got_refetch.family is OptimizationFamily.TRANSFER_REDUNDANCY
    assert "survives amortization" in got_refetch.rationale


def test_a_stall_family_needs_a_headroom_result_and_a_positive_saving():
    sources = _sources()
    src = next(s for s in sources if s.workload == "matmul")
    # Route the idle bucket to `stall` for this check, so the stall component carries cycles.
    buckets = buckets_from_kinds(BUCKET_KINDS, fixed_bucket="stall")
    bare = attribute(src, buckets=buckets, envelope=_envelope_for("matmul")).component("stall")
    assert is_unknown(bare.family), "a partition cannot settle whether these cycles overlap"

    head = headroom(src, traits=concurrency_traits(sources))
    with_head = attribute(src, buckets=buckets, envelope=_envelope_for("matmul"),
                          headroom=head).component("stall")
    assert with_head.family is OptimizationFamily.OVERLAP
    assert "158" in with_head.rationale
    assert "upper bound" in with_head.rationale


def test_without_an_envelope_every_structural_side_is_unknown_not_zero():
    attr = attribute(_sources()[0], buckets=_buckets())
    for bucket in BUCKETS:
        c = attr.component(bucket)
        assert is_unknown(c.structural_cycles)
        assert is_unknown(c.gap_cycles) or c.measured_cycles == 0


# --- corpus roll-up --------------------------------------------------------------------------------


def test_the_corpus_bucket_shares_reproduce_the_measured_split():
    corpus = attribute_corpus(_sources(), buckets=_buckets())
    cycles = corpus.bucket_cycles()
    assert cycles["dma"] == 31204
    assert cycles["compute"] == 1548 + 4990
    assert cycles["control"] == 4646
    assert cycles[RESIDUAL] == -21
    assert sum(cycles.values()) == sum(a.total_cycles for a in corpus.workloads.values()) == 42367
    shares = corpus.bucket_shares()
    assert round(shares["dma"] * 100, 1) == 73.7, "movement dominates: it is the binding resource"
