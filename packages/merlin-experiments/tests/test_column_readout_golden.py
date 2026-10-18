"""The capsule oracle consumes the same explicit column readout as native execution."""

import pytest

from merlin.runtime.tensor import Tensor
from merlin.targetgen.capsule_golden import _apply_epilogue


def test_column_readout_golden_rounds_and_saturates_each_column():
    value = Tensor((2, 3), [-3, -2, -1, 3, 6, 131], "i32")
    attrs = {"epilogue": ["acc_scale"], "acc_scale_by_column": [0.5, 0.25, 2.0], "output_dtype": "i8"}
    assert _apply_epilogue(value, attrs, {}).data == [-2, 0, -2, 2, 2, 127]


@pytest.mark.parametrize("change", [{"acc_scale": 1.0}, {"acc_scale_by_column": [0.5]}])
def test_column_readout_golden_refuses_ambiguous_or_incomplete_parameters(change):
    attrs = {"epilogue": ["acc_scale"], "acc_scale_by_column": [0.5, 0.25], "output_dtype": "i8", **change}
    with pytest.raises(ValueError, match="acc_scale"):
        _apply_epilogue(Tensor((1, 2), [3, 5], "i32"), attrs, {})
