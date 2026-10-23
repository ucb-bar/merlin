# ruff: noqa: E501 -- the interface-grammar module text is one op per line, as the parser reads it
"""AFFINE_MAP and the movement opcodes have one exact definition, shared by every engine.

WHY THIS EXISTS. The command-buffer ABI had no affine elementwise opcode (only a RESIDUAL_ADD that rounds
per operand), MOVEMENT was an identity copy, and nothing could state a fill, a pad, a cat or a scatter,
so none of the non-contraction support set had an expressible command. ``runtime.integer_ops`` defines
AFFINE_MAP (integer weights and an int32 constant in the accumulator, ReLU on the accumulator, ONE
readout), the MOVEMENT forms, CONCAT, INSERT_SLICE and FILL; the reference engine and the simulator both
call it, so an L0 comparison cannot pass on two different arithmetics.
"""

from __future__ import annotations

import numpy as np
import pytest

from merlin.runtime import integer_ops as IO
from merlin.runtime.reference import reference_outputs
from merlin.runtime.simulator import simulate
from merlin.runtime.tensor import Tensor
from merlin.targetgen import recipe_numerics as RN
from merlin.targetgen.contract.interface_emit import parse_interface_mlir


def _t(values, dtype="i8") -> Tensor:
    array = np.asarray(values, dtype=np.int64)
    return Tensor(tuple(array.shape), [int(v) for v in array.reshape(-1)], dtype)


def _np(t: Tensor) -> np.ndarray:
    return np.asarray(t.data, dtype=np.int64).reshape(t.shape)


_X = np.arange(-128, 128).reshape(16, 16)
_Y = np.flip(_X, axis=1)


def test_affine_map_is_the_recipe_numerics_form():
    recipe = RN.linear_combination(0.5, 0.25, 1.0, sign_b=-1)
    attrs = {
        "weights": list(recipe.weights),
        "constant": 0,
        "activation": "none",
        "readout_scale": recipe.readout_scale,
        "output_dtype": "i8",
    }
    out = IO.affine_map(_t(_X), _t(_Y), attrs)
    assert np.array_equal(_np(out), recipe.evaluate(_X, _Y))


def test_affine_map_relu_acts_on_the_accumulator_and_i32_has_no_readout():
    attrs = {"weights": [3], "constant": -10, "activation": "relu", "readout_scale": 0.5, "output_dtype": "i8"}
    assert np.array_equal(
        _np(IO.affine_map(_t(_X), None, attrs)), np.clip(np.rint(np.maximum(3 * _X - 10, 0) * 0.5), -128, 127)
    )
    widen = {"weights": [1], "constant": 0, "activation": "none", "readout_scale": 1.0, "output_dtype": "i32"}
    widened = IO.affine_map(_t(_X), None, widen)
    assert widened.dtype == "i32" and np.array_equal(_np(widened), _X)


def test_affine_map_row_broadcast_only_along_leading_dims():
    attrs = {"weights": [1, 1], "constant": 0, "activation": "none", "readout_scale": 1.0, "output_dtype": "i8"}
    row = _t(_Y[:1])
    assert np.array_equal(_np(IO.affine_map(_t(_X), row, attrs)), np.clip(_X + _Y[:1], -128, 127))
    with pytest.raises(IO.IntegerOpError, match="innermost"):
        IO.affine_map(_t(_X), _t(_Y[:, :1]), attrs)


@pytest.mark.parametrize(
    "attrs, match",
    [
        ({"weights": [1], "constant": 0, "activation": "none", "output_dtype": "i8"}, "readout_scale"),
        (
            {"weights": [1], "constant": 0, "activation": "relu", "readout_scale": -1.0, "output_dtype": "i8"},
            "negative",
        ),
        (
            {"weights": [1], "constant": 0, "activation": "none", "readout_scale": 1.0, "output_dtype": "i8", "x": 1},
            "unknown",
        ),
        (
            {"weights": [1, 2], "constant": 0, "activation": "none", "readout_scale": 1.0, "output_dtype": "i8"},
            "weight",
        ),
        ({"weights": [1], "constant": 0, "activation": "none", "readout_scale": 0.5, "output_dtype": "i32"}, "readout"),
    ],
)
def test_affine_map_refuses_what_it_cannot_state(attrs, match):
    with pytest.raises(IO.IntegerOpError, match=match):
        IO.affine_map(_t(_X), None, attrs)


@pytest.mark.parametrize(
    "attrs, expected",
    [
        ({"form": "permute", "perm": [1, 0]}, _X.T),
        ({"form": "slice", "offsets": [2, 1], "sizes": [3, 4], "strides": [2, 3]}, _X[2:8:2, 1:13:3]),
        (
            {"form": "pad", "low": [1, 0], "high": [0, 2], "mode": "constant", "value_code": -3},
            np.pad(_X, [(1, 0), (0, 2)], constant_values=-3),
        ),
        (
            {"form": "pad", "low": [2, 1], "high": [1, 2], "mode": "reflect"},
            np.pad(_X, [(2, 1), (1, 2)], mode="reflect"),
        ),
        (
            {"form": "pad", "low": [2, 1], "high": [1, 2], "mode": "replicate"},
            np.pad(_X, [(2, 1), (1, 2)], mode="edge"),
        ),
        ({"form": "repeat", "repeats": [2, 1]}, np.tile(_X, [2, 1])),
        ({"form": "identity", "scale": 0.5}, np.clip(np.rint(_X * 0.5), -128, 127)),
    ],
)
def test_movement_forms(attrs, expected):
    out = IO.move(_t(_X), {**attrs, "output_dtype": "i8"}, list(expected.shape))
    assert np.array_equal(_np(out), expected)


def test_broadcast_replays_rows_only_and_forms_reject_foreign_attributes():
    out = IO.move(_t(_X[:1]), {"form": "broadcast", "output_dtype": "i8"}, [4, 16])
    assert np.array_equal(_np(out), np.repeat(_X[:1], 4, axis=0))
    with pytest.raises(IO.IntegerOpError, match="innermost"):
        IO.move(_t(_X[:, :1]), {"form": "broadcast"}, [16, 16])
    with pytest.raises(IO.IntegerOpError, match="belong"):
        IO.move(_t(_X), {"form": "permute", "perm": [1, 0], "repeats": [1, 1]}, None)
    with pytest.raises(IO.IntegerOpError, match="requires"):
        IO.move(_t(_X), {"form": "slice"}, None)


def test_concat_insert_slice_and_fill():
    both = IO.concat([_t(_X), _t(_Y)], {"axis": 1, "output_dtype": "i8"})
    assert np.array_equal(_np(both), np.concatenate([_X, _Y], axis=1))
    scaled = IO.concat([_t(_X), _t(_Y)], {"axis": 0, "output_dtype": "i8", "scales": [1.0, 0.5]})
    assert np.array_equal(_np(scaled), np.concatenate([_X, np.clip(np.rint(_Y * 0.5), -128, 127)]))
    base = np.zeros((6, 16), dtype=np.int64)
    out = IO.insert_slice(_t(base), _t(_X[:2]), {"offsets": [1, 0], "strides": [2, 1], "output_dtype": "i8"})
    expected = base.copy()
    expected[1:5:2] = _X[:2]
    assert np.array_equal(_np(out), expected)
    assert _np(IO.fill([2, 3], {"value_code": -7, "output_dtype": "i8"})).tolist() == [[-7] * 3] * 2
    with pytest.raises(IO.IntegerOpError, match="axis"):
        IO.concat([_t(_X), _t(_Y)], {"output_dtype": "i8"})
    with pytest.raises(IO.IntegerOpError, match="outside"):
        IO.insert_slice(_t(base), _t(_X[:2]), {"offsets": [5, 0], "strides": [1, 1]})


_MODULE = """module attributes {merlin_iface.version = "0.1", merlin_iface.target = "t", merlin_iface.abi_version = "0.1"} {
  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<4x8xi8>
  %Y = merlin_iface.tensor {name = "Y", role = "input"} : tensor<4x8xi8>
  %Z = merlin_iface.affine_map %X, %Y {name = "Z", weights = [304, -245], constant = 0 : i64, activation = "none", readout_scale = 0.0021 : f32, output_dtype = "i8"} : (tensor<4x8xi8>, tensor<4x8xi8>) -> tensor<4x8xi8>
  %N = merlin_iface.affine_map %Z {name = "N", weights = [1], constant = 0 : i64, activation = "none", readout_scale = -1.0 : f32, output_dtype = "i8"} : (tensor<4x8xi8>) -> tensor<4x8xi8>
  merlin_iface.fence : () -> ()
  %C = merlin_iface.concat %N, %X {name = "C", axis = 0 : i64, output_dtype = "i8"} : (tensor<4x8xi8>, tensor<4x8xi8>) -> tensor<8x8xi8>
  %F = merlin_iface.fill {name = "F", value_code = 0 : i64, output_dtype = "i8"} : () -> tensor<8x8xi8>
  %S = merlin_iface.insert_slice %F, %X {name = "S", offsets = [2, 0], strides = [1, 1], output_dtype = "i8"} : (tensor<8x8xi8>, tensor<4x8xi8>) -> tensor<8x8xi8>
  %P = merlin_iface.movement %S {name = "P", form = "pad", low = [0, 1], high = [0, 1], mode = "constant", value_code = 0 : i64, output_dtype = "i8"} : (tensor<8x8xi8>) -> tensor<8x10xi8>
  %T = merlin_iface.movement %C {name = "T", form = "permute", perm = [1, 0], output_dtype = "i8"} : (tensor<8x8xi8>) -> tensor<8x8xi8>
}
"""


def test_the_interface_grammar_reaches_every_new_opcode_and_the_engines_agree():
    cb = parse_interface_mlir(_MODULE)
    opcodes = [command["opcode"] for command in cb["commands"]]
    assert opcodes == ["AFFINE_MAP", "AFFINE_MAP", "FENCE", "CONCAT", "FILL", "INSERT_SLICE", "MOVEMENT", "MOVEMENT"]
    assert cb["commands"][1]["operands"] == {"x": "Z", "dst": "N"}
    reference = reference_outputs(cb)
    assert reference == simulate(cb)["outputs"]
    assert set(reference) >= {"Z", "N", "C", "F", "S", "P", "T"}
    assert np.array_equal(np.asarray(reference["N"]), -np.asarray(reference["Z"]))
