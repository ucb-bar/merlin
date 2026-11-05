"""The output path is one ordered, target-neutral sequence, and a malformed one is refused."""

from __future__ import annotations

import pytest

from merlin.targetgen import output_path as OP


def _path(readout, accumulator=(), exits=None):
    return {
        "schema": OP.SCHEMA,
        "accumulator": {"dtype": "i32"},
        "accumulator_stages": list(accumulator),
        "readout": list(readout),
        "exits": exits or {"accumulator": True, "requant": True},
    }


_CONVERT = {"kind": "convert", "to": "f32", "rounding": "half_even"}
_SCALE = {
    "kind": "scale",
    "dtype": "f32",
    "rounding": "half_even",
    "granularity": "per_command",
    "columns_per_scale": 4,
}
_REQUANT = {"kind": "requant", "to": "i8", "rounding": "half_even", "clamp": [-128, 127]}


def test_a_well_formed_path_is_accepted():
    path = _path([_CONVERT, _SCALE, {"kind": "activation", "function": "relu", "domain": "scaled"}, _REQUANT])
    assert OP.validate(path) == path


@pytest.mark.parametrize(
    "readout",
    [
        [_SCALE, _CONVERT, _REQUANT],  # a scale of an unconverted value
        [_CONVERT, _SCALE, _REQUANT, {"kind": "activation", "function": "relu", "domain": "scaled"}],
        [_CONVERT, {**_SCALE, "granularity": "sometimes"}, _REQUANT],
        [_CONVERT, _SCALE, {"kind": "activation", "function": "relu"}, _REQUANT],
        [{"kind": "softmax"}],
    ],
)
def test_a_path_out_of_order_or_out_of_vocabulary_is_refused(readout):
    with pytest.raises(ValueError):
        OP.validate(_path(readout))


def test_a_record_with_nothing_derived_has_an_empty_path():
    from merlin.targetgen.rtl import output_epilogue as OE

    path = OP.from_record(OE.underived("nothing"))
    assert path["accumulator_stages"] == [] and path["readout"] == []
    assert path["exits"] == {"accumulator": False, "requant": False}
