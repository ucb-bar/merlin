"""Complete original windows through bounded horizontal multiblock panels."""
from dataclasses import replace
import operator

import numpy as np
import pytest

from merlin.compile.scheduling.horizontal_convolution import (
    emit_horizontal_convolution_panels, plan_horizontal_convolution_panels,
)
from merlin.compile.scheduling.window_transfer import CountedConvolutionPlan, Window2D


class Integers:
    def op(self, name, a, b, dtype="i64"):
        return {"add": operator.add, "sub": operator.sub, "mul": operator.mul,
                "udiv": operator.floordiv, "urem": operator.mod}[name](a, b)
    def cmp(self, name, a, b):
        return {"ult": operator.lt, "ule": operator.le, "eq": operator.eq}[name](a % (1 << 64), b % (1 << 64))
    def loop(self, start, stop, step, body):
        for i in range(start, stop, step):
            body(i)
    def branch(self, predicate, yes, no):
        (yes if predicate else no)()
    def select(self, predicate, yes, no):
        return yes if predicate else no


@pytest.mark.parametrize("shape,kernel,stride,padding,dilation", [
    ((1, 6, 11, 3), (3, 3), (2, 2), (1, 2, 0, 1), (1, 1)),
    ((2, 5, 8, 1), (2, 3), (1, 1), (1, 0, 1, 2), (2, 1)),
    ((1, 2, 5, 5), (3, 2), (1, 1), (2, 1, 2, 0), (1, 1)),
])
@pytest.mark.parametrize("biased", [False, True])
def test_complete_original_reduction_padding_and_tails(shape, kernel, stride, padding, dilation, biased):
    window = Window2D(shape, kernel, stride, dilation, padding)
    source = CountedConvolutionPlan(window, 7, 4, 4, 4, 4, 2048, 64, 0, 16, 0, biased)
    plan = plan_horizontal_convolution_panels(source, max_row_blocks=3, maximum_transfer_columns=64,
                                              private_rows_owned=True)
    oh, ow = window.output_spatial
    x = (np.arange(np.prod(shape)) % 19 - 9).astype(np.int8).reshape(shape)
    pt, pl, pb, pr = padding
    padded = np.pad(x, ((0, 0), (pt, pb), (pl, pr), (0, 0))).ravel()
    weight = (np.arange(np.prod(kernel) * shape[-1] * 7) % 11 - 5).astype(np.int8).reshape(-1, 7)
    bias = np.array([-(1 << 31), (1 << 31) - 1, -1, 0, 1, 19, -37], dtype=np.int32)
    sp = np.zeros((source.operand_rows, 4), dtype=np.int8)
    acc = np.zeros((source.accumulator_rows, 4), dtype=np.int32)
    result = np.zeros((shape[0] * oh * ow, 7), dtype=np.int32)
    written = np.zeros_like(result)
    array = None
    phase = "rhs"
    completed = []

    def complete(reason, fields):
        nonlocal phase
        completed.append(reason)
        phase = {"horizontal_rhs_ready": "stage", "horizontal_group_ready": "consume",
                 "horizontal_group_last_use": "stage"}[reason]

    def lower(kind, a):
        nonlocal array
        if kind == "load_weight":
            assert phase == "rhs"
            sp[a["local_row"]:a["local_row"] + a["rows"], :a["columns"]] = weight[
                a["reduction_row"]:a["reduction_row"] + a["rows"],
                a["output_column"]:a["output_column"] + a["columns"]]
        elif kind == "load_horizontal_input":
            assert phase == "stage"
            for r in range(a["rows"]):
                start = a["first_element"] + r * a["row_stride_elements"]
                for c in range(a["columns"]):
                    row = a["local_row"] + r + (c // 4) * a["block_stride_rows"]
                    assert row < plan.rhs_base
                    sp[row, c % 4] = padded[start + c]
        elif kind == "initialize_bias":
            assert phase == "stage"
            acc[a["accumulator_row"]:a["accumulator_row"] + a["rows"], :a["columns"]] = bias[
                a["output_column"]:a["output_column"] + a["columns"]]
        elif kind == "compute":
            assert phase == "consume"
            rhs = sp[a["rhs_row"]:a["rhs_row"] + a["depth"], :a["columns"]].copy()
            if a["fresh_weights"]:
                array = rhs
            np.testing.assert_array_equal(array, rhs)
            lhs = sp[a["lhs_row"]:a["lhs_row"] + a["rows"], :a["depth"]]
            dst = acc[a["accumulator_row"]:a["accumulator_row"] + a["rows"], :a["columns"]]
            product = lhs.astype(np.int32) @ array.astype(np.int32)
            dst[:] = product if a["first_reduction"] and not biased else dst + product
        elif kind == "store":
            assert phase == "consume"
            r, c = a["output_row"], a["output_column"]
            result[r:r + a["rows"], c:c + a["columns"]] = acc[
                a["accumulator_row"]:a["accumulator_row"] + a["rows"], :a["columns"]]
            written[r:r + a["rows"], c:c + a["columns"]] += 1
        else:
            raise AssertionError(kind)

    receipt = emit_horizontal_convolution_panels(plan, Integers(), lower, complete)
    expected = np.zeros_like(result)
    weights = weight.reshape(*kernel, shape[-1], 7)
    for b in range(shape[0]):
        for y in range(oh):
            for xx in range(ow):
                value = bias.astype(np.int64).copy() if biased else np.zeros(7, dtype=np.int64)
                for ky in range(kernel[0]):
                    for kx in range(kernel[1]):
                        iy = y * stride[0] + ky * dilation[0] - pt
                        ix = xx * stride[1] + kx - pl
                        if 0 <= iy < shape[1] and 0 <= ix < shape[2]:
                            value += x[b, iy, ix].astype(np.int64) @ weights[ky, kx].astype(np.int64)
                expected[(b * oh + y) * ow + xx] = value.astype(np.int32)
    np.testing.assert_array_equal(result, expected)
    assert np.all(written == 1)
    assert completed[0] == "horizontal_rhs_ready" and completed[-1] == "horizontal_group_last_use"
    assert receipt["reduction_tiles"] == kernel[0] * ((kernel[1] * shape[-1] + 3) // 4)


def test_capacity_and_unsupported_views():
    source = CountedConvolutionPlan(Window2D((1, 8, 9, 3), (3, 3)), 7, 4, 4, 4, 4, 2048, 64, 0, 16, 0)
    with pytest.raises(ValueError, match="ownership"):
        plan_horizontal_convolution_panels(source, max_row_blocks=4, maximum_transfer_columns=64)
    assert plan_horizontal_convolution_panels(source, max_row_blocks=4, maximum_transfer_columns=8,
                                               private_rows_owned=True) is None
    plan = plan_horizontal_convolution_panels(source, max_row_blocks=4, maximum_transfer_columns=64,
                                               private_rows_owned=True)
    with pytest.raises(ValueError, match="private storage"):
        replace(plan, row_blocks=100)
    with pytest.raises(ValueError, match="partition"):
        replace(plan, column_intervals=((1, 4), (5, 2)))
    with pytest.raises(ValueError, match="completion"):
        emit_horizontal_convolution_panels(plan, Integers(), lambda *_: None, None)
