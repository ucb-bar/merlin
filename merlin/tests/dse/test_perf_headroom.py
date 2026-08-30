"""Regression fixtures for :mod:`merlin.perf.headroom`.

The hand-derived result this reproduces: overlapping data movement with compute is worth
**4,457 cycles = 14.7%** over the workloads where it is the lever -- not the "~2x" that was
propagated unchecked before anyone did the arithmetic. Perfect overlap saves exactly ``min(a, b)``,
and on this target compute is far too small relative to movement for that to be worth a multiple.

The second target declares one compute unit and publishes no per-unit occupancy, so the concurrency
traits cannot be established and the answer is UNKNOWN with the missing traits named.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

import pytest

from merlin.common.paths import env, repo_root
from merlin.perf.decompose import ResourceKind, Unavailable, activity_from_busy
from merlin.perf.headroom import (
    Composition,
    composition_operator,
    concurrency_traits,
    corpus_headroom,
    headroom,
)
from merlin.perf.workload_roles import classify_workloads

BUCKET_KINDS = {
    "dma": ResourceKind.MOVEMENT,
    "mxu": ResourceKind.COMPUTE,
    "vpu": ResourceKind.COMPUTE,
    "none": ResourceKind.FIXED,
}


def _suite() -> dict:
    root = env("MERLIN_MLC_DIR")
    if not root:
        pytest.skip("MERLIN_MLC_DIR unset -- the per-unit activity fixture lives in the mlc checkout")
    path = Path(root) / "mlc" / "validate" / "npu_model_suite.json"
    if not path.is_file():
        pytest.skip(f"activity fixture not present at {path}")
    assert repo_root().is_dir()
    return json.loads(path.read_text(encoding="utf-8"))


def _sources():
    out = []
    for name, body in _suite()["kernels"].items():
        arc = body["arc"]
        out.append(activity_from_busy(
            name, arc["truth"],
            {"dma": arc["dma_busy"], "mxu": arc["mxu"], "vpu": arc["vpu"], "none": arc["none"]},
            BUCKET_KINDS,
            partitioned=True, completion_observable=True,
            provenance="per-cycle activity decomposition from the cycle-accurate model"))
    return out


# --- the hand-derived fixtures -------------------------------------------------------------------

def test_total_overlap_headroom_over_the_affected_set_is_4457_cycles_and_147_percent():
    sources = _sources()
    # The "affected set" is not the whole corpus: it is the workloads a lever can act on, which the
    # role classifier derives rather than a human picking them.
    split = classify_workloads(sources)
    result = corpus_headroom(sources, only=split.optimize)
    assert result.total_saving_cycles == 4457
    assert round(result.saving_share * 100, 1) == 14.7
    assert result.is_upper_bound, "min(a,b) is a ceiling until realised overlap is observed"


def test_per_workload_headroom_matches_the_hand_derived_values():
    result = corpus_headroom(_sources())
    best = result.workloads["smolvla_gelu_tanh"]
    assert best.saving_cycles == 1234
    assert round(best.saving_share * 100, 1) == 23.9

    mm = result.workloads["matmul"]
    assert mm.saving_cycles == 158
    assert round(mm.saving_share * 100, 1) == 6.6


def test_a_workload_with_no_compute_has_no_headroom_and_that_is_zero_not_unknown():
    result = corpus_headroom(_sources())
    w = result.workloads["smolvla_rms_norm"]
    assert w.saving_cycles == 0, "there is nothing to hide behind"
    assert w.best is None


def test_pairs_are_enumerated_over_groups_not_a_hardcoded_pair():
    result = corpus_headroom(_sources())
    w = result.workloads["matmul"]
    # Two kinds are present (movement, compute) -> exactly one pair; the grouping rationale is
    # recorded on the result rather than being an unstated assumption.
    assert len(w.pairs) == 1
    assert {w.pairs[0].a, w.pairs[0].b} == {"movement", "compute"}
    assert "kind" in w.grouping


def test_same_kind_engines_are_aggregated_so_the_pair_is_movement_vs_all_compute():
    # This workload runs both compute engines. Grouping by kind gives min(dma, mxu+vpu); pairing the
    # raw units instead would report min(dma, vpu) and understate the headroom.
    src = next(s for s in _sources() if s.workload == "gemma_attention")
    result = headroom(src, traits=concurrency_traits(_sources()))
    assert result.saving_cycles == src.busy("mxu") + src.busy("vpu") == 456


# --- the composition operator is never defaulted --------------------------------------------------

def test_composition_operator_refuses_to_answer_from_partitioned_buckets():
    result = composition_operator(_sources())
    assert isinstance(result, Unavailable)
    assert "independent of the activity buckets" in " ".join(result.missing)
    assert "partition" in result.detail


def test_composition_operator_derives_sum_only_from_an_independent_observation():
    sources = _sources()
    # Independent evidence (movement-command issue/wait pairing) put realised overlap at zero.
    op, eta = composition_operator(sources, observed_overlap_cycles={s.workload: 0 for s in sources})
    assert op is Composition.SUM
    assert eta == 0.0


def test_composition_operator_propagates_unknown_when_one_workload_is_unobserved():
    sources = _sources()
    partial = {s.workload: 0 for s in sources[1:]}
    result = composition_operator(sources, observed_overlap_cycles=partial)
    assert isinstance(result, Unavailable)


def test_realised_overlap_reduces_the_headroom_rather_than_being_assumed_zero():
    sources = _sources()
    traits = concurrency_traits(sources)
    src = next(s for s in sources if s.workload == "matmul")
    already = headroom(src, traits=traits, observed_overlap_cycles=100)
    assert already.saving_cycles == 58
    assert not already.is_upper_bound


# --- the anti-overfit gate: a second target of a different archetype ------------------------------

@functools.cache
def _second_target_manifest():
    cm = pytest.importorskip("merlin.targetgen.capability_manifests")
    try:
        return cm.manifest_for("gemmini")
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"second target unavailable: {exc}")


def test_second_target_cannot_establish_the_concurrency_traits():
    manifest = _second_target_manifest()
    traits = concurrency_traits([], manifest=manifest)
    assert not traits.satisfied
    assert traits.independent_ports is None, "not established is not the same as established false"
    assert traits.explicit_completion is None
    assert len(traits.missing) == 3


def test_second_target_headroom_is_unknown_with_the_missing_traits_named():
    manifest = _second_target_manifest()
    src = activity_from_busy(
        "G01_multitile_sq", 7439, {"mesh": 7439}, {"mesh": ResourceKind.COMPUTE},
        provenance="cycle-accurate RTL simulation, total cycles only")
    result = headroom(src, manifest=manifest)
    assert isinstance(result, Unavailable)
    joined = " ".join(result.missing)
    assert "engine groups" in joined
    assert "independent ports" in joined
    assert "completion" in joined
