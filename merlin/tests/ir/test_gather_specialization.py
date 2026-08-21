"""The value-indexed-gather analysis finds the specializable table and refuses the unsound cases.

The case that matters most is `test_a_second_consumer_of_the_indices_is_rejected`: specializing
renumbers the stored index values, so a second reader of that tensor would receive renumbered tokens
and be wrong with no numerical tell. That rejection is the analysis's reason for existing, not a
corner case.
"""
from __future__ import annotations

import pytest

from merlin.common import mlir_query as mq
from merlin.common.ir_lock import IR_LOCK
from merlin.xdsl_dialects.lowering.gather_specialization import (
    find_gather_specializations,
    kept_rows,
)

MAPS = ("indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, "
        "affine_map<(d0, d1, d2) -> (d0, d1, d2)>], "
        'iterator_types = ["parallel", "parallel", "parallel"]')

GATHER = """
module {
  func.func @forward(%%w: tensor<1000x4xf32>, %%ids: tensor<1x3xi64>) -> tensor<1x3x4xf32> {
    %%e = tensor.empty() : tensor<1x3x4xf32>
    %%g = linalg.generic {%s}
         ins(%%ids : tensor<1x3xi64>) outs(%%e : tensor<1x3x4xf32>) {
    ^bb0(%%id: i64, %%o: f32):
      %%r = arith.index_cast %%id : i64 to index
      %%c = linalg.index 2 : index
      %%v = tensor.extract %%w[%%r, %%c] : tensor<1000x4xf32>
      linalg.yield %%v : f32
    } -> tensor<1x3x4xf32>
    return %%g : tensor<1x3x4xf32>
  }
}
""" % MAPS

# Same gather, but the ids are ALSO read by a second generic -- the unsound shape.
TWO_CONSUMERS = """
module {
  func.func @forward(%%w: tensor<1000x4xf32>, %%ids: tensor<1x3xi64>)
      -> (tensor<1x3x4xf32>, tensor<1x3xi64>) {
    %%e = tensor.empty() : tensor<1x3x4xf32>
    %%g = linalg.generic {%s}
         ins(%%ids : tensor<1x3xi64>) outs(%%e : tensor<1x3x4xf32>) {
    ^bb0(%%id: i64, %%o: f32):
      %%r = arith.index_cast %%id : i64 to index
      %%c = linalg.index 2 : index
      %%v = tensor.extract %%w[%%r, %%c] : tensor<1000x4xf32>
      linalg.yield %%v : f32
    } -> tensor<1x3x4xf32>
    %%e2 = tensor.empty() : tensor<1x3xi64>
    %%m = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0, d1)>],
                         iterator_types = ["parallel", "parallel"]}
         ins(%%ids : tensor<1x3xi64>) outs(%%e2 : tensor<1x3xi64>) {
    ^bb0(%%a: i64, %%b: i64):
      linalg.yield %%a : i64
    } -> tensor<1x3xi64>
    return %%g, %%m : tensor<1x3x4xf32>, tensor<1x3xi64>
  }
}
""" % MAPS

# A big table read as a contraction, not a gather: nothing to specialize, and no near-miss either.
MATMUL = """
module {
  func.func @forward(%a: tensor<8x4xf32>, %b: tensor<4x1000xf32>) -> tensor<8x1000xf32> {
    %e = tensor.empty() : tensor<8x1000xf32>
    %m = linalg.matmul ins(%a, %b : tensor<8x4xf32>, tensor<4x1000xf32>)
                       outs(%e : tensor<8x1000xf32>) -> tensor<8x1000xf32>
    return %m : tensor<8x1000xf32>
  }
}
"""


def _analyze(src: str):
    with IR_LOCK:
        return find_gather_specializations(mq.parse(src))


def test_a_value_indexed_gather_is_found():
    specs, rejections = _analyze(GATHER)
    assert rejections == []
    assert len(specs) == 1
    s = specs[0]
    assert (s.table_arg, s.index_arg) == (0, 1)
    assert s.table_shape == [1000, 4]
    assert s.table_dtype == "f32"
    assert s.row_dim == 0
    assert s.rows == 1000


def test_a_second_consumer_of_the_indices_is_rejected():
    """The soundness condition. Renumbering the ids would corrupt the other reader."""
    specs, rejections = _analyze(TWO_CONSUMERS)
    assert specs == [], "a table whose indices have another reader must not be offered"
    assert len(rejections) == 1
    assert "consumers" in rejections[0].reason


def test_a_contraction_over_a_big_table_is_neither_found_nor_rejected():
    """No opportunity, and no near-miss noise: this shape is simply not a gather."""
    specs, rejections = _analyze(MATMUL)
    assert specs == []
    assert rejections == []


def test_kept_rows_dedupes_and_renumbers():
    specs, _ = _analyze(GATHER)
    kept, remapped = kept_rows(specs[0], [7, 3, 7])
    assert kept == [3, 7], "a repeated index must cost one row, not one row per occurrence"
    assert remapped == [1, 0, 1]
    # the renumbered values must address the kept rows and reproduce the originals
    assert [kept[i] for i in remapped] == [7, 3, 7]


def test_an_out_of_range_index_raises_instead_of_reading_the_wrong_row():
    specs, _ = _analyze(GATHER)
    with pytest.raises(ValueError, match="out of range"):
        kept_rows(specs[0], [7, 1000])


def test_a_missing_function_is_reported_not_silently_empty():
    specs, rejections = _analyze(GATHER)
    assert specs and not rejections
    with IR_LOCK:
        specs, rejections = find_gather_specializations(mq.parse(GATHER), func_name="nope")
    assert specs == []
    assert rejections and "no func.func" in rejections[0].reason
