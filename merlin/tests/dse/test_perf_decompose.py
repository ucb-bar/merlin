"""Regression fixtures for :mod:`merlin.perf.decompose`.

The reference numbers were derived by hand, once, from one target's per-cycle activity model. They
are the contract: the tool is wrong if it does not reproduce them, and overfit if it cannot produce a
different, correct answer for a target of another archetype.

The second target here is a decoupled-queue systolic accelerator whose published performance data is
total cycles plus a derived array utilization -- one number, not a decomposition. The correct result
for it is UNKNOWN with the missing trait named. A number would be the failure.
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
    activity_from_busy,
    activity_trait,
    busy_by_kind,
    decompose,
    decompose_corpus,
    is_unknown,
)

# The activity model's buckets, mapped to kinds from the target's own unit roles (a DMA engine, a
# systolic matrix unit, a vector unit) rather than from how the bucket happens to be spelled.
BUCKET_KINDS = {
    "dma": ResourceKind.MOVEMENT,
    "mxu": ResourceKind.COMPUTE,
    "vpu": ResourceKind.COMPUTE,
    "none": ResourceKind.FIXED,
}


def _suite() -> dict:
    """The hand-derived per-cycle activity suite, resolved through the repo's own path config."""
    root = env("MERLIN_MLC_DIR")
    if not root:
        pytest.skip("MERLIN_MLC_DIR unset -- the per-unit activity fixture lives in the mlc checkout")
    path = Path(root) / "mlc" / "validate" / "npu_model_suite.json"
    if not path.is_file():
        pytest.skip(f"activity fixture not present at {path}")
    assert repo_root().is_dir()
    return json.loads(path.read_text(encoding="utf-8"))


def _sources():
    suite = _suite()
    out = []
    for name, body in suite["kernels"].items():
        arc = body["arc"]
        out.append(activity_from_busy(
            name, arc["truth"],
            {"dma": arc["dma_busy"], "mxu": arc["mxu"], "vpu": arc["vpu"], "none": arc["none"]},
            BUCKET_KINDS,
            partitioned=True, completion_observable=True,
            provenance="per-cycle activity decomposition from the cycle-accurate model"))
    return {s.workload: s for s in out}


@functools.cache
def _second_target():
    """Manifest + RTL facts for a target of a different archetype (decoupled queue / systolic)."""
    cm = pytest.importorskip("merlin.targetgen.capability_manifests")
    facts_mod = pytest.importorskip("merlin.targetgen.rtl.facts")
    try:
        return cm.manifest_for("gemmini"), facts_mod.load_facts("gemmini")
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"second target unavailable: {exc}")


# --- the hand-derived fixtures -------------------------------------------------------------------

def test_matmul_is_movement_bound_at_the_measured_shares():
    d = decompose(_sources()["matmul"])
    assert not isinstance(d, Unavailable)
    assert round(d.shares["dma"] * 100, 1) == 86.2
    assert round(d.shares["mxu"] * 100, 1) == 6.6
    assert d.binding == "dma"
    assert d.binding_kind is ResourceKind.MOVEMENT


def test_a_zero_compute_workload_reports_movement_937_and_compute_exactly_zero():
    d = decompose(_sources()["smolvla_rms_norm"])
    assert round(d.shares["dma"] * 100, 1) == 93.7
    assert d.shares_by_kind[ResourceKind.COMPUTE] == 0.0
    assert d.binding == "dma"


def test_the_corpus_regime_is_movement_bound():
    corpus = decompose_corpus(_sources().values())
    assert len(corpus.workloads) == 21
    assert corpus.modal_binding_kind() is ResourceKind.MOVEMENT
    assert corpus.binding_kind_counts[ResourceKind.MOVEMENT] == 18


def test_the_partition_residual_is_reported_not_absorbed():
    # The buckets partition the timeline with a one-cycle fencepost; the residual is surfaced so a
    # reader can see the accounting does not close exactly, rather than having it folded away.
    d = decompose(_sources()["matmul"])
    assert d.unattributed_cycles == -1
    assert d.partitioned is True


def test_busy_by_kind_adds_same_kind_engines():
    src = _sources()["gemma_attention"]
    by_kind = busy_by_kind(src)
    assert by_kind[ResourceKind.COMPUTE] == src.busy("mxu") + src.busy("vpu")


# --- the anti-overfit gate: a second target of a different archetype ------------------------------

def test_second_target_has_no_per_unit_decomposition_and_says_so():
    manifest, facts = _second_target()
    trait = activity_trait([], manifest=manifest, facts=facts)
    assert trait.satisfied is None, "declaring a unit is not observing it; this must not read True"
    assert trait.missing, "the missing trait must be NAMED, not merely absent"
    assert "busy-cycle" in trait.missing[0]


def test_second_target_total_cycles_only_is_unavailable_not_a_fabricated_share():
    # All this target publishes per workload is total cycles (plus a derived array utilization).
    # One bucket is a total, not a decomposition.
    src = activity_from_busy(
        "G00_single_tile", 308, {"mesh": 308}, {"mesh": ResourceKind.COMPUTE},
        provenance="cycle-accurate RTL simulation, total cycles only")
    result = decompose(src)
    assert isinstance(result, Unavailable)
    assert is_unknown(result)
    assert result.value is UNKNOWN
    assert "busy-cycle" in " ".join(result.missing)


def test_unknown_is_never_readable_as_zero():
    with pytest.raises(UnknownValueError):
        float(UNKNOWN)
    with pytest.raises(UnknownValueError):
        bool(UNKNOWN)
    assert UNKNOWN != 0.0
    assert UNKNOWN != None  # noqa: E711 - the point is that it is not None either


def test_a_unit_with_no_declared_kind_raises_instead_of_defaulting():
    with pytest.raises(ValueError, match="ResourceKind"):
        activity_from_busy("w", 10, {"mystery": 5}, {})
