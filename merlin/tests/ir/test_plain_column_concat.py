"""Plain column assembly is distinct from unknown convolution window gathers."""

import pytest

from merlin.common import mlir_query as mq
from merlin.llvmlower.group_host_call import _host_window_gather


def _concat(*, offset=1, row_offset=0, stride=1, computed=False):
    extra = "%twice = arith.addi %r, %r : tensor<3x4xi8>" if computed else ""
    source = "%twice" if computed else "%r"
    text = f"""builtin.module {{
      func.func @f(%x: tensor<3x5xi8>, %r: tensor<3x4xi8>) -> tensor<3x7xi8> {{
        {extra}
        %s = "tensor.extract_slice"({source}) <{{static_offsets = array<i64: {row_offset}, {offset}>,
            static_sizes = array<i64: 3, 2>, static_strides = array<i64: 1, {stride}>,
            operandSegmentSizes = array<i32: 1, 0, 0, 0>}}> : (tensor<3x4xi8>) -> tensor<3x2xi8>
        %c = tensor.concat dim(1) %x, %s : (tensor<3x5xi8>, tensor<3x2xi8>) -> tensor<3x7xi8>
        func.return %c : tensor<3x7xi8>
      }}
    }}"""
    return next(op for op in mq.parse(text).walk() if op.name == "tensor.concat").results[0]


def test_full_rows_and_static_column_slice_keep_an_ordinary_matmul():
    assert _host_window_gather(_concat(), 3, 7) is None


@pytest.mark.parametrize(
    "change", [{"offset": 3}, {"offset": -1}, {"row_offset": 1}, {"stride": 2}, {"computed": True}]
)
def test_out_of_bounds_rows_strides_and_computed_pieces_stay_refused(change):
    with pytest.raises(ValueError, match="concatenated gather"):
        _host_window_gather(_concat(**change), 3, 7)


def test_mismatched_consumer_geometry_does_not_admit_the_concat():
    with pytest.raises(ValueError, match="concatenated gather"):
        _host_window_gather(_concat(), 3, 8)
