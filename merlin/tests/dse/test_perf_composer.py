"""Regression fixtures for :mod:`merlin.perf.composer` -- the generalized composer end to end.

The fixtures, and what each one would catch:

* **The 6/7 prediction, with zero fitting.** ``mxu_busy = fill + completion`` with ``fill = 2*DIM-2``
  from the geometry and the completion delay read out of the program's own schedule gives **158**
  on four kernels and **316** on two, exactly. The seventh, a K-accumulating matmul, refutes the
  naive extension: it would predict ``62 + 2*96 = 254`` against a measured **284**. It is recorded
  UNKNOWN. Two points that show a law is wrong do not show what is right, and the fitted correction
  would be indistinguishable from the law having been right.
* **No prediction below the structural bound**, over all 21.
* **Coverage weighted by measured time, never by module count.** On this corpus a module-count share
  of 51.2% sits next to a structurally-derivable *time* share of 3.7%, because the depth walk
  resolves combinational leaves and refuses the sequenced engine that carries three quarters of the
  cycles.
* **A second target of a different archetype gets a different, correct answer** rather than this
  one's.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

import pytest

from merlin.common.paths import env, repo_root
from merlin.perf.attribution import RESIDUAL, buckets_from_kinds
from merlin.perf.decompose import (
    UNKNOWN,
    ResourceKind,
    Unavailable,
    activity_from_busy,
    is_unknown,
)
from merlin.perf.envelope import (
    Basis,
    FixedTerm,
    Peak,
    ResourceDemand,
    ResourceTime,
    resource_time,
)
from merlin.perf.headroom import Composition, composition_operator
from merlin.perf.composer import (
    Coverage,
    compose_corpus,
    coverage,
    fixed_terms_from_timing,
    operator_sensitivity,
    peaks_from_observations,
    structural_unit_time,
)

BUCKET_KINDS = {
    "dma": ResourceKind.MOVEMENT,
    "mxu": ResourceKind.COMPUTE,
    "vpu": ResourceKind.COMPUTE,
    "none": ResourceKind.FIXED,
}
#: Which RTL module implements which activity bucket. A target fact, supplied at the edge.
RESOURCE_MODULES = {"dma": "DmaEngine", "mxu": "SystolicArray", "vpu": "VectorEngine"}


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
        pytest.skip(f"RTL facts for {target} unavailable: {exc}")
    records = body.get("timing")
    if not records:
        pytest.skip(f"no cached timing facts for {target}: UNCACHED is not absent")
    return records


@functools.cache
def _sources() -> tuple:
    out = []
    for name, body in _suite()["kernels"].items():
        arc = body["arc"]
        out.append(activity_from_busy(
            name, arc["truth"],
            {"dma": arc["dma_busy"], "mxu": arc["mxu"], "vpu": arc["vpu"], "none": arc["none"]},
            BUCKET_KINDS, partitioned=True, completion_observable=True,
            provenance="per-cycle activity decomposition from the cycle-accurate model"))
    return tuple(out)


def _operator() -> tuple[Composition, float]:
    """The operator, DERIVED -- from an overlap observation independent of the activity buckets.

    The buckets themselves cannot settle it: they partition the timeline and report zero overlap by
    construction. Independent evidence (movement-command issue/wait pairing) put realised overlap at
    zero on every workload, which is what makes ``sum`` the derived answer here rather than the
    textbook ``max``.
    """
    got = composition_operator(list(_sources()),
                               observed_overlap_cycles={s.workload: 0 for s in _sources()})
    assert not isinstance(got, Unavailable)
    return got


@functools.cache
def _mxu_law():
    from merlin.perf.record import derive_delay_mnemonic, derive_unit_roles, fill_cycles

    suite = _suite()
    streams = [k["op_stream"] for k in suite["kernels"].values() if k.get("op_stream")]
    delay = derive_delay_mnemonic(streams)
    roles = derive_unit_roles(streams, "MatrixSystolic", delay)
    return delay, roles, fill_cycles("systolic_2d", suite["_meta"]["mxu_dim"])


def _demands() -> dict[str, dict[str, ResourceDemand]]:
    """Per-workload demands for the two resources that have a demand/peak form."""
    out: dict[str, dict[str, ResourceDemand]] = {}
    for name, body in _suite()["kernels"].items():
        arc = body["arc"]
        out[name] = {
            "dma": ResourceDemand("dma", ResourceKind.MOVEMENT,
                                  arc["reads"] + arc["writes"], "beats", basis=Basis.MOVED,
                                  provenance="measured read/write beats x the port width"),
            "vpu": ResourceDemand("vpu", ResourceKind.COMPUTE,
                                  sum(1 for fam, _m, _i in body["op_stream"] if fam == "Vector"),
                                  "ops", basis=Basis.MOVED, provenance="program op stream"),
        }
    return out


@functools.cache
def _times() -> dict[str, tuple]:
    """Per-resource times: movement from a measured rate ceiling, the systolic engine from its
    structural law, the vector engine from neither, reset as a serial intercept."""
    from merlin.perf.record import compose_unit_busy

    suite = _suite()
    delay, roles, fill = _mxu_law()
    demands = _demands()
    peaks = peaks_from_observations(demands, _sources(), units={"dma": "beats", "vpu": "ops"},
                                    provenance="per-cycle activity decomposition")
    out: dict[str, tuple] = {}
    for name, body in suite["kernels"].items():
        ts = [resource_time(demands[name]["dma"], peaks["dma"]),
              structural_unit_time("mxu", ResourceKind.COMPUTE,
                                   compose_unit_busy(body["op_stream"], roles, fill, delay),
                                   provenance=f"fill={fill} plus the program's scheduled delays"),
              resource_time(demands[name]["vpu"], peaks["vpu"]),
              ResourceTime(resource="none", kind=ResourceKind.FIXED,
                           cycles=float(suite["_meta"]["reset_cycles"]), unit="cycles",
                           basis=Basis.MOVED, evidence_kind="measured",
                           provenance="reset_cycles declared by the measurement source")]
        out[name] = tuple(ts)
    return out


@functools.cache
def _prediction():
    op, eta = _operator()
    structural, refused = fixed_terms_from_timing(_timing("atlas"), RESOURCE_MODULES)
    return compose_corpus(
        list(_sources()), times={k: list(v) for k, v in _times().items()}, operator=op, eta=eta,
        buckets=buckets_from_kinds(BUCKET_KINDS, fixed_bucket="control"),
        timing_records=_timing("atlas"), structural_resources=list(structural)), refused


# --- fixture 1: the 6/7 prediction, zero fitting ---------------------------------------------------


def test_the_structural_law_predicts_six_of_seven_matrix_kernels_exactly():
    times = _times()
    measured = {k: v["arc"]["mxu"] for k, v in _suite()["kernels"].items()}

    def mxu(name):
        return next(t for t in times[name] if t.resource == "mxu")

    fill = _mxu_law()[2]
    assert fill == 62 == 2 * _suite()["_meta"]["mxu_dim"] - 2

    for name in ("matmul", "smolvla_matmul", "smolvla_fused_matmul_bias", "gemma_attention"):
        assert mxu(name).cycles == 158.0 == measured[name], name
    for name in ("gemma_mlp", "smolvla_attention"):
        assert mxu(name).cycles == 316.0 == measured[name] == 2 * 158, name


def test_the_seventh_refutes_the_naive_extension_and_is_recorded_unknown_not_fitted():
    name = "smolvla_matmul_k_chain"
    t = next(t for t in _times()[name] if t.resource == "mxu")
    assert not t.known, "the K-accumulate path is outside the law's validated regime"
    assert "accumulate" in t.reason
    # What the naive extension WOULD have said, and what the hardware actually did.
    fill = _mxu_law()[2]
    delay = _mxu_law()[1].compute_delay
    assert fill + 2 * delay == 254
    assert _suite()["kernels"][name]["arc"]["mxu"] == 284
    assert 284 - 254 == 30, "the delta the accumulate path carries and the law does not express"
    # The lower bound the law DID establish survives, under its own name.
    assert t.fixed_cycles == fill


def test_a_unit_the_program_never_issues_to_is_zero_cycles_which_is_derived():
    t = next(t for t in _times()["gemma_rms_norm"] if t.resource == "mxu")
    assert t.cycles == 0.0
    assert t.known, "no matrix ops issued is a fact about the program, not a hole in the evidence"


# --- fixture 2: nothing falls below the structural bound --------------------------------------------


def test_no_prediction_falls_below_the_structural_bound_on_any_of_the_21():
    pred, _ = _prediction()
    assert len(pred.predictions) == 21
    assert pred.floor_violations == ()
    for name, p in pred.predictions.items():
        assert p.respects_floor, name
        c = p.envelope.composed
        value = c.partial_cycles if is_unknown(c.cycles) else float(c.cycles)
        assert value >= c.floor_cycles - 1e-9, name


def test_no_lower_bound_exceeds_the_measurement_it_bounds():
    pred, _ = _prediction()
    assert pred.bound_violations == ()
    for name, p in pred.resolved.items():
        assert float(p.predicted_cycles) <= p.measured_cycles, name


def test_the_fully_resolved_workloads_recover_most_of_the_measured_runtime():
    pred, _ = _prediction()
    # Only the workloads with no vector ops resolve end to end -- the vector engine's peak is
    # neither structurally derivable nor observable from the op stream.
    assert sorted(pred.resolved) == ["matmul", "smolvla_matmul"]
    matmul = pred.predictions["matmul"]
    # 2050.0 movement (2048 beats at the observed ceiling) + 158 systolic + 0 vector + 12 reset.
    assert matmul.predicted_cycles == pytest.approx(2220.0, abs=0.5)
    assert matmul.measured_cycles == 2383
    assert round(matmul.recovered_share, 3) == 0.932


def test_unresolved_workloads_keep_a_partial_bound_under_its_own_name():
    pred, _ = _prediction()
    p = pred.predictions["gemma_attention"]
    assert is_unknown(p.predicted_cycles)
    assert p.envelope.unresolved == ("vpu",)
    assert p.partial_recovered_share > 0.7
    assert p.envelope.to_dict()["lower_bound_cycles"] == "UNKNOWN"


# --- fixture 3: the buckets close, and the residual is the fencepost ---------------------------------


def test_the_corpus_attribution_closes_and_keeps_the_constant_residual():
    pred, _ = _prediction()
    assert pred.attribution.closes
    assert pred.attribution.residual_is_constant == -1
    assert pred.attribution.bucket_cycles()[RESIDUAL] == -21


def test_the_binding_resource_is_movement_on_almost_every_workload():
    pred, _ = _prediction()
    limiters = [v for v in pred.limiters.values() if not is_unknown(v)]
    assert limiters.count("dma") >= 18, "movement binds: 73.7% of all measured cycles"


# --- coverage: weighted by TIME, never by module count -----------------------------------------------


def test_the_module_count_share_and_the_structural_time_share_disagree_by_an_order_of_magnitude():
    pred, refused = _prediction()
    cov = pred.coverage
    # The walk resolved 43 of 84 modules -- and refused the movement engine, which carries three
    # quarters of the cycles. Reporting the module count as confidence would claim half the design
    # is understood while the unresolved half holds nearly all the time.
    assert (cov.resolved_module_count, cov.walked_module_count) == (43, 84)
    assert round(cov.module_count_share, 3) == 0.512
    assert round(cov.structurally_resolved_time_share, 4) == 0.0365
    assert cov.module_count_share > 10 * cov.structurally_resolved_time_share
    # Both archetypes refuse their own dominant resource; here it is the movement engine.
    assert "dma" in refused and "vpu" in refused
    assert "feedback" in refused["dma"].detail


def test_coverage_exposes_no_field_called_confidence():
    pred, _ = _prediction()
    assert not hasattr(pred.coverage, "confidence")
    body = pred.coverage.to_dict()
    assert "confidence" not in body
    assert "A COUNT OF MODULES" in body["module_count_share_note"]


def test_the_structural_time_share_is_unknown_when_nobody_said_which_resources_resolved():
    op, eta = _operator()
    pred = compose_corpus(list(_sources()), times={k: list(v) for k, v in _times().items()},
                          operator=op, eta=eta)
    assert is_unknown(pred.coverage.structurally_resolved_time_share), "not established, not zero"


def test_absent_timing_records_are_uncached_rather_than_a_design_with_no_sequenced_logic():
    cov = coverage(list(_sources()), {})
    assert cov.walked_module_count == 0
    assert "UNCACHED" in cov.note
    assert is_unknown(cov.module_count_share), "0/0 is not 0.0"


def test_unresolved_time_is_reported_per_resource_with_its_reason():
    pred, _ = _prediction()
    cov = pred.coverage
    assert round(cov.unresolved_time_share["vpu"], 4) == 0.1177
    assert "refuted" in cov.unresolved_reasons["vpu"]


# --- what defaulting the operator to max would have cost -----------------------------------------


def _measured_times() -> dict[str, list[ResourceTime]]:
    """Per-resource times taken straight from the measured occupancies.

    This asks the operator question directly -- "given what each resource actually did, what would
    each composition rule predict?" -- over all 21 workloads, rather than over the two whose
    structural bound resolves end to end.
    """
    out: dict[str, list[ResourceTime]] = {}
    for s in _sources():
        out[s.workload] = [
            ResourceTime(resource=r.name, kind=r.kind, cycles=float(r.busy_cycles), unit="cycles",
                         basis=Basis.MOVED, evidence_kind="measured",
                         provenance="per-cycle activity decomposition")
            for r in s.resources]
    return out


def test_the_textbook_max_understates_the_corpus_runtime_by_thirteen_percent():
    got = operator_sensitivity(list(_sources()), times=_measured_times())
    assert not isinstance(got, Unavailable)
    assert got.n_workloads == 21
    assert got.by_operator["sum"] == 42388.0
    assert got.by_operator["max"] == 36725.0
    assert round(got.understatement_vs_sum * 100, 2) == 13.36
    # The worst single workload is far worse than the corpus figure, which is why a corpus mean is
    # not a safe summary of what defaulting the operator costs.
    assert got.worst_workload == "gemma_rms_norm"
    assert round(got.worst_understatement * 100, 2) == 30.46


def test_operator_sensitivity_refuses_over_partially_resolved_terms():
    # The structural times leave the vector engine UNKNOWN on most workloads; comparing operators
    # there would compare two different sets of terms.
    got = operator_sensitivity([s for s in _sources() if s.workload == "gemma_attention"],
                               times={k: list(v) for k, v in _times().items()})
    assert isinstance(got, Unavailable)
    assert "two different sets of terms" in got.detail


def test_the_operator_is_derived_and_refuses_from_the_buckets_alone():
    # The composer never picks one. Asked without an independent observation, the derivation says so.
    got = composition_operator(list(_sources()))
    assert isinstance(got, Unavailable)
    assert "partition" in got.detail
    op, eta = _operator()
    assert op is Composition.SUM and eta == 0.0


# --- the anti-overfit gate: a second target of a different archetype -------------------------------


def test_the_same_code_gives_a_different_correct_answer_on_the_second_archetype():
    # A weight-stationary mesh with no per-unit decomposition: total cycles only, and the mesh
    # container refuses (36/36 outputs cyclic) so its peak is not structurally derivable either.
    records = _timing("gemmini")
    structural, refused = fixed_terms_from_timing(records, {"mesh": "Mesh"})
    assert structural == {}
    assert "feedback" in refused["mesh"].detail

    src = activity_from_busy("G01_multitile_sq", 7439, {"mesh": 7439},
                             {"mesh": ResourceKind.COMPUTE},
                             partitioned=None, completion_observable=None,
                             provenance="cycle-accurate RTL simulation, total cycles only")
    times = {src.workload: [resource_time(
        ResourceDemand("mesh", ResourceKind.COMPUTE, 4096, "macs", basis=Basis.MOVED,
                       provenance="tile geometry"),
        Peak.unknown("mesh", "macs", refused["mesh"].detail,
                     provenance="rtl timing walk"))]}
    pred = compose_corpus([src], times=times, operator=Composition.SUM, eta=0.0,
                          timing_records=records, structural_resources=list(structural))

    p = pred.predictions[src.workload]
    assert is_unknown(p.predicted_cycles), "no peak, so no bound -- and no fabricated one"
    assert p.envelope.limiter is UNKNOWN
    assert p.envelope.limiter_is_provisional
    cov = pred.coverage
    assert (cov.resolved_module_count, cov.walked_module_count) == (31, 116)
    assert round(cov.module_count_share, 3) == 0.267
    assert cov.structurally_resolved_time_share == 0.0, "none of the time is structurally bounded"
    assert cov.module_count_share > 0.25, "while a quarter of the MODULES resolved"


def test_on_the_second_archetype_the_operator_is_unobservable_rather_than_defaulted():
    # One engine group, so there is no pair whose overlap could be observed. The refusal names that
    # -- distinct both from "nobody looked" and from a quietly assumed `max`.
    src = activity_from_busy("G01_multitile_sq", 7439, {"mesh": 7439},
                             {"mesh": ResourceKind.COMPUTE},
                             provenance="cycle-accurate RTL simulation, total cycles only")
    bare = composition_operator([src])
    assert isinstance(bare, Unavailable)
    assert "independent of the activity buckets" in " ".join(bare.missing)
    observed = composition_operator([src], observed_overlap_cycles={src.workload: 0})
    assert isinstance(observed, Unavailable)
    assert "unobservable" in observed.detail


def test_a_combinational_module_on_the_second_target_still_yields_a_real_zero():
    ok, bad = fixed_terms_from_timing(_timing("gemmini"), {"tile": "Tile"})
    assert "tile" not in bad
    assert isinstance(ok["tile"], FixedTerm)
    assert ok["tile"].cycles == 0


def test_a_resource_whose_module_is_not_in_the_walk_is_named_rather_than_dropped():
    ok, bad = fixed_terms_from_timing(_timing("gemmini"), {"x": "NoSuchModule"})
    assert ok == {}
    assert "was not among them" in bad["x"].detail


def test_coverage_is_a_frozen_record_that_serializes_both_units_side_by_side():
    pred, _ = _prediction()
    assert isinstance(pred.coverage, Coverage)
    body = pred.to_dict()["coverage"]
    assert set(body) >= {"module_count_share", "time_weighted_resolved_share",
                         "structurally_resolved_time_share"}
    assert body["module_count_share"] != body["structurally_resolved_time_share"]
