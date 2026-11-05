"""Actual source effects preserve padded/tail arithmetic with finite issue queues."""

from dataclasses import replace

import numpy as np
import pytest
from test_counted_convolution import Integers

from merlin.compile.scheduling.window_transfer import (
    ConvolutionLoadIssueLaw,
    CountedConvolutionPlan,
    Window2D,
    emit_convolution_load_issue_tile,
    emit_convolution_tile,
)


def law():
    return ConvolutionLoadIssueLaw(
        "neutral-hardware",
        "source-ordered-i32",
        "original-window",
        "independent-neutral-scoreboard-and-full-drain",
        8,
        local_range_dependencies=True,
        ordered_load_configuration=True,
        paired_descriptor_compute=True,
        finite_queue_backpressure=True,
        complete_device_drain=True,
        immutable_mapped_disjoint_buffers=True,
    )


@pytest.mark.parametrize("biased", [False, True])
def test_issued_effects_match_original_windows_at_each_real_completion(biased):
    shape = (1, 3, 5, 3)
    kernel = (3, 2)
    padding = (1, 1, 1, 0)
    outputs = 5
    window = Window2D(shape, kernel, padding=padding)
    oh, ow = window.output_spatial
    plan = CountedConvolutionPlan(window, outputs, 4, min(4, ow), 4, 3, 32, 8, 0, 8, 0, biased)
    image = (np.arange(np.prod(shape)) % 17 - 8).astype(np.int8).reshape(shape)
    weight = (
        (np.arange(np.prod(kernel) * shape[-1] * outputs) % 13 - 6).astype(np.int8).reshape(*kernel, shape[-1], outputs)
    )
    bias = np.array([-(1 << 31), (1 << 31) - 1, -9, 0, 7], dtype=np.int32)
    actual = np.zeros((oh * ow, outputs), dtype=np.int32)
    written = np.zeros_like(actual)
    lhs = np.zeros((plan.row_tile, plan.channel_tile), dtype=np.int8)
    rhs = None
    acc = None
    queued = []
    source = []
    issued = []
    completed = []
    queue_capacity = 2

    def execute(kind, a):
        nonlocal rhs, acc
        if kind == "initialize_bias":
            acc = np.broadcast_to(
                bias[a["output_column"] : a["output_column"] + a["columns"]], (a["rows"], a["columns"])
            ).copy()
        elif kind == "load_input":
            for row in range(a["rows"]):
                start = a["first_element"] + row * a["row_stride_elements"]
                lhs[a["local_row"] + row, : a["columns"]] = (
                    0 if a["zero"] else image.ravel()[start : start + a["columns"]]
                )
        elif kind == "load_weight":
            rhs = weight.reshape(-1, outputs)[
                a["reduction_row"] : a["reduction_row"] + a["rows"],
                a["output_column"] : a["output_column"] + a["columns"],
            ]
        elif kind == "compute":
            product = lhs[: a["rows"], : a["depth"]].astype(np.int32) @ rhs.astype(np.int32)
            acc = product if a["first_reduction"] and not a["bias"] else acc + product
        elif kind == "store":
            row, col = a["output_row"], a["output_column"]
            actual[row : row + a["rows"], col : col + a["columns"]] = acc
            written[row : row + a["rows"], col : col + a["columns"]] += 1
        else:
            raise AssertionError(kind)

    def issue(kind, a):
        issued.append((kind, dict(a)))
        if len(queued) == queue_capacity:
            execute(*queued.pop(0))
        queued.append((kind, dict(a)))

    def complete(kind, a):
        assert queued and queued[-1][0] == kind
        while queued:
            execute(*queued.pop(0))
        completed.append(kind)

    for y in range(oh):
        for x in range(0, ow, plan.row_tile):
            for col in range(0, outputs, plan.column_tile):
                args = dict(
                    output_row=y * ow + x,
                    output_column=col,
                    rows=min(plan.row_tile, ow - x),
                    columns=min(plan.column_tile, outputs - col),
                    accumulator_row=0,
                )
                emit_convolution_tile(
                    plan, Integers(), lambda k, a: source.append((k, dict(a))), synchronous_completion=True, **args
                )
                result = emit_convolution_load_issue_tile(plan, Integers(), issue, complete, law=law(), **args)
                assert not queued and result["timing"] == "UNKNOWN"
    assert source == issued
    assert completed.count("compute") == sum(k == "compute" for k, a in source)
    assert completed.count("store") == sum(k == "store" for k, a in source)
    assert all(kind in ("compute", "store") for kind in completed)
    expected = np.zeros_like(actual)
    for y in range(oh):
        for x in range(ow):
            value = bias.astype(np.int64).copy() if biased else np.zeros(outputs, dtype=np.int64)
            for ky in range(kernel[0]):
                for kx in range(kernel[1]):
                    iy, ix = y + ky - padding[0], x + kx - padding[1]
                    if 0 <= iy < shape[1] and 0 <= ix < shape[2]:
                        value += image[0, iy, ix].astype(np.int64) @ weight[ky, kx].astype(np.int64)
            expected[y * ow + x] = value.astype(np.int32)
    assert np.array_equal(actual, expected) and np.all(written == 1)


@pytest.mark.parametrize("field", ["finite_queue_backpressure", "complete_device_drain"])
def test_missing_completion_law_refuses_before_issuing(field):
    plan = CountedConvolutionPlan(Window2D((1, 2, 2, 1), (1, 1)), 1, 2, 2, 1, 1, 8, 4, 0, 4, 0)
    seen = []
    with pytest.raises(ValueError, match="complete source dependencies"):
        emit_convolution_load_issue_tile(
            plan,
            Integers(),
            lambda *x: seen.append(x),
            lambda *x: None,
            output_row=0,
            output_column=0,
            rows=2,
            columns=1,
            accumulator_row=0,
            law=replace(law(), **{field: False}),
        )
    assert seen == []
    with pytest.raises(ValueError, match="logical epoch budget"):
        emit_convolution_load_issue_tile(
            plan,
            Integers(),
            lambda *x: seen.append(x),
            lambda *x: None,
            output_row=0,
            output_column=0,
            rows=2,
            columns=1,
            accumulator_row=0,
            law=replace(law(), load_issue_budget=2),
        )
    assert seen == []
