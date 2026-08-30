"""Regression fixtures for :mod:`merlin.perf.amplification`.

The hand-derived result: the demo tiles move 9-28x more bytes than the computation needs. Moved
bytes are ``(reads + writes) * beat_bytes`` -- measured beats through the RTL-derived port width, not
a shape estimate. Useful bytes come from each workload's declared tensors (shape x dtype), with
broadcast splat constants excluded: a materialized ``full((32,16), 0.5)`` is a lowering artifact, not
data the computation needs.

The caveat that makes the ratio actionable rather than merely alarming: part of it is a 32x32 demo
tile paying a fixed per-command transfer block, which proper tiling amortizes. So the tool must
report how much of the ratio is that artifact, and these tests pin that split too.

The second target has the trait (explicit DMA + a software-managed scratchpad, derived from its own
manifest and RTL facts) but no byte evidence -- so the failure is named as *evidence*, not as
*trait*, and the two stay distinguishable.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

import pytest

from merlin.common.paths import env, repo_root
from merlin.perf.amplification import (
    UNKNOWN,
    MovementObservation,
    TensorOperand,
    Unavailable,
    amplification,
    corpus_amplification,
    moved_bytes_from_beats,
    movement_trait,
    useful_bytes,
)

# Each workload's declared tensors, read from the reference programs' own memory regions and golden
# result (shape x dtype). `broadcast=True` marks a splat constant.
#   * the matmul streams two 32x32 fp8 operands and writes a 32x32 bf16 result;
#   * the bf16 vector kernels stage a 32x32 tile as two stacked 32x16 halves.
OPERANDS = {
    "matmul": (
        TensorOperand("activation", 32 * 32, 1.0),
        TensorOperand("weight", 32 * 32, 1.0),
        TensorOperand("result", 32 * 32, 2.0, is_output=True),
    ),
    "smolvla_rms_norm": (
        TensorOperand("x_h0", 32 * 16, 2.0),
        TensorOperand("x_h1", 32 * 16, 2.0),
        TensorOperand("out", 64 * 16, 2.0, is_output=True),
        TensorOperand("inv_dim", 32 * 16, 2.0, broadcast=True),
        TensorOperand("eps", 32 * 16, 2.0, broadcast=True),
    ),
    "smolvla_gelu_tanh": (
        TensorOperand("x_h0", 32 * 16, 2.0),
        TensorOperand("x_h1", 32 * 16, 2.0),
        TensorOperand("out", 64 * 16, 2.0, is_output=True),
        TensorOperand("c044", 32 * 16, 2.0, broadcast=True),
        TensorOperand("csqrt", 32 * 16, 2.0, broadcast=True),
        TensorOperand("chalf", 32 * 16, 2.0, broadcast=True),
    ),
    "smolvla_elementwise_add": (
        TensorOperand("a", 32 * 32, 2.0),
        TensorOperand("b", 32 * 32, 2.0),
        TensorOperand("out", 32 * 32, 2.0, is_output=True),
    ),
}


def _suite() -> dict:
    root = env("MERLIN_MLC_DIR")
    if not root:
        pytest.skip("MERLIN_MLC_DIR unset -- the movement fixture lives in the mlc checkout")
    path = Path(root) / "mlc" / "validate" / "npu_model_suite.json"
    if not path.is_file():
        pytest.skip(f"movement fixture not present at {path}")
    assert repo_root().is_dir()
    return json.loads(path.read_text(encoding="utf-8"))


@functools.cache
def _first_target_trait():
    cm = pytest.importorskip("merlin.targetgen.capability_manifests")
    facts_mod = pytest.importorskip("merlin.targetgen.rtl.facts")
    try:
        return movement_trait(cm.manifest_for("atlas"), facts_mod.load_facts("atlas"))
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"first target manifest unavailable: {exc}")


def _observations():
    suite = _suite()
    beat_bytes = suite["_meta"]["beat_bytes"]
    kernels = suite["kernels"]

    def commands(name: str, direction: str) -> int:
        # Structural count over the op stream: a movement op's mnemonic is dotted
        # `<engine>.<action>.<channel>`, so the action is field 1. No pattern matching.
        return sum(1 for unit, mnemonic, _ in kernels[name]["op_stream"]
                   if unit == "Dma" and mnemonic.split(".")[1] == direction)

    out = []
    for name, operands in OPERANDS.items():
        arc = kernels[name]["arc"]
        useful, splat = useful_bytes(operands)
        # A command that carried no beats moved nothing; count only directions that moved data.
        transfers = ((commands(name, "load") if arc["reads"] else 0)
                     + (commands(name, "store") if arc["writes"] else 0))
        out.append(MovementObservation(
            workload=name,
            moved_bytes=moved_bytes_from_beats(arc["reads"] + arc["writes"], beat_bytes),
            useful_bytes=useful, transfers=transfers, broadcast_bytes=splat,
            provenance="measured bus beats x the RTL-derived data-port width"))
    return out


# --- the hand-derived fixtures -------------------------------------------------------------------

def test_moved_bytes_are_measured_beats_times_the_port_width():
    suite = _suite()
    arc = suite["kernels"]["matmul"]["arc"]
    moved = moved_bytes_from_beats(arc["reads"] + arc["writes"], suite["_meta"]["beat_bytes"])
    assert moved == 65536


def test_useful_bytes_exclude_broadcast_splats():
    useful, splat = useful_bytes(OPERANDS["smolvla_rms_norm"])
    assert useful == 4096
    assert splat == 2048, "the splats are reported, not silently dropped"


@pytest.mark.parametrize(("workload", "expected"), [
    ("matmul", 16.0),
    ("smolvla_rms_norm", 24.0),
    ("smolvla_gelu_tanh", 28.0),
    ("smolvla_elementwise_add", 2.0),
])
def test_amplification_ratios_match_the_hand_derived_values(workload, expected):
    result = corpus_amplification(_observations(), trait=_first_target_trait())
    assert round(result.workloads[workload].ratio, 2) == expected


def test_the_ratio_factors_exactly_into_granularity_times_redundancy():
    result = corpus_amplification(_observations(), trait=_first_target_trait())
    for w in result.workloads.values():
        assert float(w.granularity_factor) * float(w.redundancy_factor) == pytest.approx(w.ratio)


def test_the_sensitivity_names_how_much_of_the_ratio_is_the_small_tile_artifact():
    result = corpus_amplification(_observations(), trait=_first_target_trait())
    sens = result.sensitivity
    assert not isinstance(sens, Unavailable)
    assert sens.n_points == 4
    # The 16x on the matmul is a 4x fixed-block artifact on top of a 4x genuine refetch, so only
    # part of the headline survives proper tiling. Reporting the bare ratio would overstate the win.
    mm = result.workloads["matmul"]
    assert float(mm.block_bytes) == 16384.0
    assert float(mm.granularity_factor) == pytest.approx(4.0)
    assert float(mm.redundancy_factor) == pytest.approx(4.0)
    assert mm.artifact_share == pytest.approx(0.5)
    # The granule is a per-workload descriptor choice, not one corpus-wide constant.
    assert sens.block_bytes_consistent is False
    assert 1.0 < float(sens.amortized_ratio) < float(max(result.ratios.values()))


def test_a_bigger_tile_at_the_same_granule_has_a_smaller_artifact():
    # The elementwise workload fills its blocks best and shows the smallest ratio, which is the
    # amortization claim in miniature: the artifact shrinks as the payload approaches the block.
    result = corpus_amplification(_observations(), trait=_first_target_trait())
    small_tile = result.workloads["matmul"]
    filled = result.workloads["smolvla_elementwise_add"]
    assert float(filled.granularity_factor) < float(small_tile.granularity_factor)
    assert float(filled.fill_fraction) > float(small_tile.fill_fraction)


def test_the_sensitivity_refuses_to_fit_from_one_point():
    obs = _observations()[:1]
    result = corpus_amplification(obs, trait=_first_target_trait())
    assert isinstance(result.sensitivity, Unavailable)
    assert "two workloads" in " ".join(result.sensitivity.missing)


def test_a_single_command_cannot_derive_its_own_granule():
    trait = _first_target_trait()
    one = MovementObservation("one_shot", moved_bytes=4096, useful_bytes=1024, transfers=1)
    result = amplification(one, trait=trait)
    assert result.ratio == 4.0
    # The ratio still reports; the SPLIT does not, because one transfer is one point and a block
    # derived from it is indistinguishable from that command's payload.
    assert result.block_bytes is UNKNOWN
    assert result.granularity_factor is UNKNOWN
    assert result.redundancy_factor is UNKNOWN
    assert result.artifact_share is UNKNOWN


# --- the anti-overfit gate: a second target of a different archetype ------------------------------

@functools.cache
def _second_target():
    cm = pytest.importorskip("merlin.targetgen.capability_manifests")
    facts_mod = pytest.importorskip("merlin.targetgen.rtl.facts")
    try:
        return cm.manifest_for("gemmini"), facts_mod.load_facts("gemmini")
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"second target unavailable: {exc}")


def test_second_target_satisfies_the_movement_trait_from_its_own_facts():
    manifest, facts = _second_target()
    trait = movement_trait(manifest, facts)
    assert trait.satisfied is True
    # Derived, not assumed: the RTL facts discover the software-managed on-chip memories.
    assert "on-chip memories" in trait.evidence
    assert "scratchpad" in trait.evidence


def test_second_target_has_the_trait_but_no_byte_evidence_so_the_failure_names_evidence():
    manifest, facts = _second_target()
    trait = movement_trait(manifest, facts)
    # Its published performance data is cycles and MACs; the operand shapes are known but nobody
    # counted the bytes that crossed the bus.
    known_shapes = useful_bytes((
        TensorOperand("a", 16 * 16, 1.0),
        TensorOperand("b", 16 * 16, 1.0),
        TensorOperand("c", 16 * 16, 4.0, is_output=True),
    ))[0]
    assert known_shapes == 1536
    result = amplification(
        MovementObservation("G00_single_tile", moved_bytes=0, useful_bytes=known_shapes),
        trait=trait)
    assert isinstance(result, Unavailable)
    assert "moved bytes" in " ".join(result.missing)
    assert "explicit" not in " ".join(result.missing), "the trait holds; only the measurement is absent"


def test_a_target_with_no_managed_movement_fails_on_the_trait_not_the_evidence():
    trait = movement_trait({"compute_units": [{"name": "alu", "semantic_capabilities": []}]}, {})
    assert trait.satisfied is None
    result = amplification(MovementObservation("w", 4096, 1024, transfers=4), trait=trait)
    assert isinstance(result, Unavailable)
    assert "scratchpad" in " ".join(result.missing)
