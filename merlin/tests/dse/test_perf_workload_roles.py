"""Regression fixtures for :mod:`merlin.perf.workload_roles`.

The hand-picked list this replaces: "these 12 capsules are the optimize set, this one is the
memory-term calibration, and the ~688-cycle elementwise trio is the fixed-term calibration". The
classifier has to produce that 12 / 1 / 3 split from the cost decomposition alone, and it has to
produce a *different* answer on a target that does not expose one.

Functional eligibility is the wrong axis for a performance corpus: a workload with no headroom is
not a failed optimization target, it is the wrong instrument.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from merlin.common.paths import env, repo_root
from merlin.perf.decompose import ResourceKind, activity_from_busy
from merlin.perf.workload_roles import (
    FIXED_TERM,
    Role,
    RolePolicy,
    classify_workloads,
)

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

def test_the_split_is_twelve_optimize_one_movement_calibration_three_fixed_calibration():
    split = classify_workloads(_sources())
    counts = split.counts()
    assert counts["optimize"] == 12
    assert counts["calibration:movement"] == 1
    assert counts[f"calibration:{FIXED_TERM}"] == 3
    assert sum(counts.values()) == 21, "every workload is classified; none is silently dropped"


def test_the_zero_compute_workload_is_the_movement_term_calibration():
    split = classify_workloads(_sources())
    assert split.named(Role.CALIBRATION, "movement") == ["smolvla_rms_norm"]
    role = split.roles["smolvla_rms_norm"]
    assert "no confound" in role.rule


def test_the_688_cycle_elementwise_trio_is_the_fixed_term_calibration():
    split = classify_workloads(_sources())
    trio = split.named(Role.CALIBRATION, FIXED_TERM)
    assert trio == ["smolvla_elementwise_add", "smolvla_elementwise_mul", "smolvla_elementwise_sub"]
    for name in trio:
        assert split.roles[name].total_cycles == 688
        assert split.roles[name].fixed_share > 1 / 3


def test_the_optimize_set_is_the_one_the_headroom_total_was_measured_over():
    split = classify_workloads(_sources())
    assert sum(split.roles[n].headroom_cycles for n in split.optimize) == 4457


def test_the_regime_and_the_headroom_floor_are_corpus_derived_not_constants():
    split = classify_workloads(_sources())
    assert split.modal_binding_kind is ResourceKind.MOVEMENT
    # The floor is expressed in quanta of the smallest engine occupancy the corpus resolves, so it
    # travels to a target with a different clock or a different unit granularity.
    assert split.quantum_cycles == 38
    assert split.headroom_floor_cycles == 76


def test_workloads_bound_by_the_minority_term_calibrate_that_term():
    split = classify_workloads(_sources())
    compute_calib = split.named(Role.CALIBRATION, "compute")
    assert compute_calib == ["gemma_rms_norm", "smolvla_fused_silu_gate", "smolvla_silu"]
    for name in compute_calib:
        assert split.roles[name].binding_kind is ResourceKind.COMPUTE


def test_in_regime_workloads_below_the_floor_are_no_lever_not_optimize():
    split = classify_workloads(_sources())
    assert split.named(Role.NO_LEVER) == ["smolvla_reduction_sum", "smolvla_requant"]
    for name in split.named(Role.NO_LEVER):
        assert split.roles[name].headroom_cycles < split.headroom_floor_cycles


def test_every_role_carries_the_rule_that_produced_it():
    split = classify_workloads(_sources())
    assert all(r.rule for r in split.roles.values())


def test_the_policy_thresholds_are_declared_and_retunable():
    sources = _sources()
    loose = classify_workloads(sources, policy=RolePolicy(fixed_share_min=0.25))
    # Two more workloads sit between 25% and 33% fixed share, so relaxing the policy moves them.
    assert loose.counts()[f"calibration:{FIXED_TERM}"] > 3
    assert loose.counts()["optimize"] < 12


# --- the anti-overfit gate: a second target of a different archetype ------------------------------

def test_second_target_without_a_decomposition_classifies_nothing_and_says_why():
    src = activity_from_busy(
        "G01_multitile_sq", 7439, {"mesh": 7439}, {"mesh": ResourceKind.COMPUTE},
        provenance="cycle-accurate RTL simulation, total cycles only")
    split = classify_workloads([src])
    assert split.roles == {}, "a workload with no decomposition must never default to OPTIMIZE"
    assert "G01_multitile_sq" in split.unavailable
    assert "busy-cycle" in " ".join(split.unavailable["G01_multitile_sq"].missing)


def test_a_compute_bound_corpus_flips_the_regime_and_the_calibration_term():
    # The regime is corpus-relative, so the same code on a compute-bound corpus makes the
    # movement-bound outlier the calibration instrument -- the mirror image of the first target.
    kinds = {"dma": ResourceKind.MOVEMENT, "pe": ResourceKind.COMPUTE, "idle": ResourceKind.FIXED}
    corpus = [
        activity_from_busy(f"c{i}", 1200, {"dma": 400, "pe": 750, "idle": 50}, kinds)
        for i in range(4)
    ] + [activity_from_busy("mover", 1000, {"dma": 800, "pe": 150, "idle": 50}, kinds)]
    split = classify_workloads(corpus)
    assert split.modal_binding_kind is ResourceKind.COMPUTE
    assert split.named(Role.CALIBRATION, "movement") == ["mover"]
    assert len(split.optimize) == 4
