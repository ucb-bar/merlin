"""The host-lane permutation probe reproduces a layout the captures carry, not a rank-2 transpose."""

from __future__ import annotations

import pytest

from merlin.targetgen.capsule_builtin_source import render_builtin_source
from merlin.targetgen.capsule_source import _OP_BODIES, _PREAMBLE
from merlin.targetgen.corpus_synth import DEFAULT_PERMUTE_LAYOUT, permute_probe_layout
from merlin.targetgen.host_lane_ops import observed_host_ops

_MODULE = """"builtin.module"() ({
  "func.func"() <{sym_name = "forward", function_type = (tensor<1x5x5x3xf32>) -> tensor<1x3x5x5xf32>}> ({
  ^bb0(%x: tensor<1x5x5x3xf32>):
    %e = "tensor.empty"() : () -> tensor<1x3x5x5xf32>
    %t = "linalg.transpose"(%x, %e) <{permutation = array<i64: 0, 3, 1, 2>}> ({
    ^bb0(%a: f32, %b: f32):
      "linalg.yield"(%a) : (f32) -> ()
    }) {prov.op = "permute", prov.aten = "aten.permute.default", prov.family = "movement"}
       : (tensor<1x5x5x3xf32>, tensor<1x3x5x5xf32>) -> tensor<1x3x5x5xf32>
    "func.return"(%t) : (tensor<1x3x5x5xf32>) -> ()
  }) : () -> ()
}) : () -> ()
"""


def _render(spec):
    return render_builtin_source(
        {"op": "permute", "dtype": "fp32", "seed": 1, **spec},
        preamble=_PREAMBLE,
        integer_matmul="",
        parametric_linear="",
        bodies=_OP_BODIES,
        input_names={},
    )


def test_the_captures_permutation_layouts_are_recorded(tmp_path):
    capture = tmp_path / "model.mlir"
    capture.write_text(_MODULE)
    (row,) = observed_host_ops({"app": capture}, [("movement", "f32")])["movement/f32"]
    assert row["op"] == "permute" and row["n_regions"] == 1
    assert row["layouts"] == [{"shape": [1, 5, 5, 3], "permutation": [0, 3, 1, 2], "n_regions": 1}]


def test_the_probe_takes_the_most_frequent_rank_3_or_4_layout_else_a_rank_4_default():
    observed = {
        "op": "permute",
        "layouts": [
            {"shape": [48, 80], "permutation": [1, 0], "n_regions": 9},  # a weight view, not host movement
            {"shape": [1, 2, 3, 4], "permutation": [0, 1, 2, 3], "n_regions": 5},  # identity moves nothing
            {"shape": [1, 12, 2, 8], "permutation": [0, 2, 1, 3], "n_regions": 2},
        ],
    }
    assert permute_probe_layout(observed) == {"shape": [1, 12, 2, 8], "permutation": [0, 2, 1, 3]}
    assert permute_probe_layout({"op": "permute"}) == DEFAULT_PERMUTE_LAYOUT
    assert permute_probe_layout(None) == DEFAULT_PERMUTE_LAYOUT
    assert len(DEFAULT_PERMUTE_LAYOUT["shape"]) == 4


def test_the_permute_program_is_written_at_the_selected_rank_and_axis_order():
    source = _render({"shape": [1, 5, 5, 3], "permutation": [0, 3, 1, 2]})
    assert "x.permute(0, 3, 1, 2)" in source and "_r(1, 5, 5, 3)" in source
    # Without a selected layout the historic [M, K] transpose is unchanged.
    legacy = _render({"M": 4, "K": 8})
    assert "x.permute(1, 0)" in legacy and "_r(4, 8)" in legacy
    with pytest.raises(ValueError, match="every axis"):
        _render({"shape": [1, 5, 5, 3], "permutation": [0, 3, 1, 1]})
