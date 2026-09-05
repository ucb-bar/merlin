"""The corpus matmul builder must carry fused-pool semantics into its interface.

Pooling is not merely capsule metadata: its geometry configures the commit/store path and changes the
result tensor's row extent.  These tests exercise the real ``corpus_spec.build_matmul`` ->
``model_slice_export.emit_interface_mlir`` seam so either a dropped attribute or an unpooled result
type fails before corpus materialization.
"""
from __future__ import annotations

import pytest

from merlin.runtime.reference import reference_outputs
from merlin.targetgen import corpus_spec as CS
from merlin.targetgen import model_slice_export as MSE
from merlin.targetgen.contract import interface_emit as IE


def _binding() -> CS.CorpusBinding:
    return CS.CorpusBinding(
        target="gemmini", tile_dim=16, operand_dtype="int8", accum_dtype="i32",
        integer=True, tiers=["L0", "L1", "L2", "L3"], compare="exact_int",
        classes_for=lambda **_: [],
    )


def _entry(**over) -> dict:
    entry = {
        "name": "POOL", "kind": "layer", "op": "matmul",
        "M": 16, "K": 16, "N": 16, "lhs": "A0", "weight": "W", "out": "Y0",
        "epilogue": ["maxpool"], "pool_in_dims": [4, 4],
        "pool_size": [2, 2], "pool_stride": [2, 2],
        "source_role": "handauthored_compiler_test", "source_reference": "pool emission test",
    }
    entry.update(over)
    return entry


def test_matmul_builder_emits_pool_geometry_and_pooled_result_extent():
    """A 4x4 plane pooled 2x2/2 has four committed rows, not the 16-row accumulator extent."""
    capsule, mlir = CS.build_matmul(_entry(), _binding())

    attrs = capsule["operation"]["attributes"]
    assert attrs["pool_in_dims"] == [4, 4]
    assert attrs["pool_size"] == [2, 2]
    assert attrs["pool_stride"] == [2, 2]
    assert attrs["pool_padding"] == [0, 0, 0, 0]
    # i8, NOT the accumulator width: a fused max-pool runs in the STORE DMA, which reads the
    # target's operand width rather than the full-width accumulator container, so the commit is at
    # `binding.operand_dtype` (see `_resolve_output_dtype`, landed in c9a404d2). An i32 pool store
    # is one the backend refuses to compile -- a previous change that produced one turned 16 tests
    # red. The property this test is about is the pooled EXTENT; the dtype rides along.
    assert "tensor<4x16xi8>" in mlir

    parsed = IE.parse_interface_mlir(mlir)
    commit = next(c for c in parsed["commands"] if c["opcode"] == "COMMIT")
    assert commit["operands"]["dst"] == "Y0"
    assert commit["attributes"] == {
        "epilogue": ["maxpool"], "output_dtype": "i8",
        "pool_in_dims": [4, 4], "pool_size": [2, 2], "pool_stride": [2, 2],
        "pool_padding": [0, 0, 0, 0],
    }
    output = reference_outputs(parsed)["Y0"]
    assert (len(output), len(output[0])) == (4, 16)


def test_matmul_builder_emits_ragged_pool_extent_without_folding_the_tail():
    """A 5x5 plane under 2x2/2 drops its fifth row/column and commits a 2x2 plane."""
    _, mlir = CS.build_matmul(
        _entry(M=25, N=17, pool_in_dims=[5, 5]), _binding())
    assert "tensor<4x17xi8>" in mlir      # operand width, per the store-DMA rule above
    output = reference_outputs(IE.parse_interface_mlir(mlir))["Y0"]
    assert (len(output), len(output[0])) == (4, 17)


def test_emitter_refuses_pool_geometry_without_the_pool_stage():
    """Accepting unread geometry would make an interface look pooled while committing the raw matrix."""
    with pytest.raises(ValueError, match="maxpool"):
        MSE.emit_interface_mlir(
            lhs="A0", weight="W", out="Y0", M=16, K=16, N=16,
            epilogue=[], output_dtype="i32", commit_rows=4,
            pool_attrs={"pool_in_dims": [4, 4], "pool_size": [2, 2],
                        "pool_stride": [2, 2], "pool_padding": [0, 0, 0, 0]},
        )


def test_emitter_refuses_pool_stage_without_geometry_and_extent():
    """A pool stage with no window or pooled type is a silently skipped epilogue, never a valid module."""
    with pytest.raises(ValueError, match="pool_attrs"):
        MSE.emit_interface_mlir(
            lhs="A0", weight="W", out="Y0", M=16, K=16, N=16,
            epilogue=["maxpool"], output_dtype="i32",
        )
