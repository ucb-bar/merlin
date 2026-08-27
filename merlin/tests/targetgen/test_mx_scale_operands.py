"""A block-scaled operand format must DECLARE its scales, not hide them in the golden.

A microscaling operand is a pair: quantized elements plus one shared E8M0 exponent per fixed-length run
of them. The capsule contract declared only the elements. The scales were computed at corpus-generation
time from the capsule-name salt and written to ``golden.yaml`` as bare code lists, which
``canonical_input_raws`` skips by design as non-tensor provenance -- so the only thing that ever saw them
was the reference kernel, which bakes them in.

The consequence was structural, not incidental: a submitted backend received block-scaled element bytes
and no scales, which is half a number, and could not reproduce the golden for ANY input. Measured on one
target, 9 capsules failed this way every round for every arm -- read as compiler failures, when no
compiler could have passed them.

These tests pin the three properties that close it:

  * the scales are DECLARED operands, shaped from the target's OWN derived scale group;
  * the block length is never assumed -- a target that declares block scaling without a group, or a
    contraction depth that is not a whole number of blocks, declares no scale operands rather than
    guessing one (fail closed);
  * an unscaled target's capsules and interface are byte-identical to before.
"""
from __future__ import annotations

import pathlib
import tempfile

import pytest
import yaml

from merlin.targetgen import capsule_golden as CG
from merlin.targetgen import corpus_spec as CS
from merlin.targetgen import model_slice_export as MSE


def _binding(scaling, blk, dtype="mxfp8"):
    return CS.CorpusBinding(target="t", tile_dim=16, operand_dtype=dtype, accum_dtype="f32",
                            integer=False, tiers=["L2"], compare="tolerance_float",
                            scaling=scaling, scale_block=blk)


# --------------------------------------------------------------- deriving the block length

def test_the_block_length_comes_from_the_targets_own_manifest():
    """Never a baked 32: a different microscaling profile uses a different run length."""
    contract = {"compute_units": [{"name": "u", "scaling": "block_e8m0"}], "mx_iface": {"group": 64}}
    assert CS._scale_block_elems(contract) == 64


def test_a_target_with_no_block_scaled_unit_has_no_block_length():
    contract = {"compute_units": [{"name": "u", "scaling": "none"}], "mx_iface": {"group": 32}}
    assert CS._scale_block_elems(contract) is None


def test_block_scaling_declared_without_a_group_fails_closed():
    """An undeclared run length is unknown, not 32."""
    assert CS._scale_block_elems({"compute_units": [{"name": "u", "scaling": "block_e8m0"}]}) is None


def test_the_group_is_found_by_key_not_by_a_path_spelled_here():
    """Targets name their MX interface block differently; the group is located by key at any depth."""
    for wrapper in ("mx_mmio", "some_other_name", "deeply"):
        contract = {"compute_units": [{"scaling": "block_e8m0"}], wrapper: {"nested": {"group": 16}}}
        assert CS._scale_block_elems(contract) == 16


# --------------------------------------------------------------- the declared operands

def test_a_block_scaled_matmul_declares_a_scale_stream_per_operand():
    ins = CS._matmul_inputs("A0", "W", 16, 32, 16, "mxfp8", _binding("block_e8m0", 32))
    by = {i["name"]: i for i in ins}
    assert "A0_scale" in by and "W_scale" in by, "the scales must be operands the backend can be handed"
    assert by["A0_scale"]["shape"] == [1, 16], "one exponent per K-block, per lhs row"
    assert by["W_scale"]["shape"] == [1, 16], "one exponent per K-block, per weight column"
    assert by["A0_scale"]["scale_of"] == "A0" and by["W_scale"]["scale_of"] == "W"
    assert by["A0_scale"]["block"] == 32


def test_the_group_count_tracks_the_contraction_depth():
    ins = CS._matmul_inputs("A0", "W", 16, 128, 16, "mxfp8", _binding("block_e8m0", 32))
    by = {i["name"]: i for i in ins}
    assert by["A0_scale"]["shape"] == [4, 16], "K=128 over blocks of 32 is 4 scale groups"


def test_an_unscaled_target_declares_exactly_what_it_always_did():
    ins = CS._matmul_inputs("A0", "W", 16, 32, 16, "int8", _binding(None, None, "int8"))
    assert [i["name"] for i in ins] == ["W", "A0"], "an unscaled capsule must not grow operands"


def test_a_depth_that_is_not_whole_blocks_declares_no_scales():
    """Fail closed: a partial block has no well-defined shared exponent, so nothing is declared."""
    ins = CS._matmul_inputs("A0", "W", 16, 24, 16, "mxfp8", _binding("block_e8m0", 32))
    assert [i["name"] for i in ins] == ["W", "A0"]


# --------------------------------------------------------------- the interface the backend reads

def test_the_interface_declares_the_scale_tensors():
    m = MSE.emit_interface_mlir(lhs="A0", weight="W", out="Y0", M=16, K=32, N=16, epilogue=[],
                                output_dtype="bf16", target="t", operand_dtype="f8E4M3FN",
                                acc_dtype="bf16", scale_block=32)
    assert 'name = "A0_scale", role = "scale"' in m
    assert 'scale_of = "A0", block = 32 : i64' in m
    assert "tensor<1x16xi8>" in m, "the scale stream needs a shape the backend can allocate"


def test_the_unscaled_interface_is_unchanged():
    m = MSE.emit_interface_mlir(lhs="A0", weight="W", out="Y0", M=16, K=32, N=16, epilogue=[],
                                output_dtype="i32", target="t")
    assert 'role = "scale"' not in m


# --------------------------------------------------------------- the scales actually reach the harness

def test_the_scales_arrive_as_ordinary_canonical_operands():
    """The integration point: before, only the reference kernel could see them."""
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "golden.yaml").write_text(yaml.safe_dump({
        "oracle_provenance": {"inputs": {
            "A0": {"shape": [16, 32], "decoded": [0.0] * 512},
            "W": {"shape": [32, 16], "decoded": [0.0] * 512},
            "SA_e8m0_codes": [[125] * 16],          # the pre-existing non-tensor provenance
            "SB_e8m0_codes": [[130] * 16],
            "A0_scale": {"shape": [1, 16], "decoded": [125] * 16},
            "W_scale": {"shape": [1, 16], "decoded": [130] * 16},
        }}, "outputs": {"Y0": [[0.0] * 16] * 16}}), encoding="utf-8")
    vals = CG.canonical_input_values({"name": "X"}, d)
    assert "A0_scale" in vals and "W_scale" in vals, "the scales must reach a general backend"
    assert vals["A0_scale"]["values"] == [125] * 16
    assert vals["A0_scale"]["shape"] == [1, 16]


def test_the_oracles_own_scale_provenance_is_untouched():
    """Additive: the MX oracle keeps reading the code lists it always read."""
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "golden.yaml").write_text(yaml.safe_dump({
        "oracle_provenance": {"inputs": {
            "SA_e8m0_codes": [[125] * 16], "SB_e8m0_codes": [[130] * 16],
            "A0_scale": {"shape": [1, 16], "decoded": [125] * 16},
        }}, "outputs": {}}), encoding="utf-8")
    gy = yaml.safe_load((d / "golden.yaml").read_text())
    ins = gy["oracle_provenance"]["inputs"]
    assert ins["SA_e8m0_codes"] == [[125] * 16]
    # and the new per-tensor spec must not be mistaken for one of them
    assert isinstance(ins["A0_scale"], dict)


# --------------------------------------------------------------- the other two block-scaled shapes
# The single GEMM is not the only block-scaled capsule form. A batched GEMM carries one scale stream per
# batch, and a fused attention capsule has TWO MX stages contracting over different axes. Both were
# equally scale-blind, so both are pinned here.

def test_a_batched_matmul_carries_a_scale_stream_per_batch():
    ins = CS._batched_mx_inputs("A0", "W", 2, 16, 32, 16, "mxfp8", _binding("block_e8m0", 32))
    by = {i["name"]: i for i in ins}
    assert by["A0_scale"]["shape"] == [2, 1, 16], "batch leads: each batch is its own GEMM"
    assert by["W_scale"]["shape"] == [2, 1, 16]


def test_an_unscaled_batched_matmul_is_unchanged():
    ins = CS._batched_mx_inputs("A0", "W", 2, 16, 32, 16, "int8", _binding(None, None, "int8"))
    assert [i["name"] for i in ins] == ["W", "A0"]


def test_attention_scales_follow_each_stages_own_contraction_axis():
    """QK contracts over the head dim, PV over the key count -- different axes, different streams."""
    ins = CS._attention_mx_inputs("Q", "K", "V", 16, 32, 32, 16, "mxfp8", _binding("block_e8m0", 32))
    by = {i["name"]: i for i in ins}
    assert by["Q_scale"]["shape"] == [1, 16], "H/32 groups x M queries"
    assert by["K_scale"]["shape"] == [1, 32], "H/32 groups x Skv keys"
    assert by["V_scale"]["shape"] == [1, 16], "Skv/32 groups x Dv value columns"


def test_the_softmax_intermediates_requant_scale_is_an_operand_too():
    """P is produced by the kernel, but the exponent it is requantized against was chosen when the golden
    was built -- so the kernel cannot derive it, and a value it cannot derive is an input."""
    by = {i["name"]: i for i in
          CS._attention_mx_inputs("Q", "K", "V", 16, 32, 32, 16, "mxfp8", _binding("block_e8m0", 32))}
    assert "P_scale" in by
    assert by["P_scale"]["scale_of"] == "P"
    assert by["P_scale"]["shape"] == [1, 16], "Skv/32 groups x M rows"


def test_attention_declares_no_scales_when_either_axis_is_not_whole_blocks():
    b = _binding("block_e8m0", 32)
    assert [i["name"] for i in CS._attention_mx_inputs("Q", "K", "V", 16, 24, 32, 16, "mxfp8", b)] \
        == ["Q", "K", "V"]
    assert [i["name"] for i in CS._attention_mx_inputs("Q", "K", "V", 16, 32, 24, 16, "mxfp8", b)] \
        == ["Q", "K", "V"]
