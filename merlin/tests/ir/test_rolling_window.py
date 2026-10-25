"""Actual counted row/frontier transport, original taps and completion custody."""

import operator
from dataclasses import replace

import pytest
from xdsl.dialects import arith

from merlin.compile.scheduling.program_limits import AddressWindow
from merlin.compile.scheduling.rolling_window import (
    emit_rolling_window,
    emit_rolling_window_lookup,
    materialize_rolling_window,
    propose_rolling_window,
    read_rolling_window,
    rolling_window_text,
    verify_rolling_window,
)
from merlin.compile.scheduling.window_transfer import Window2D
from merlin.xdsl_dialects.rolling_window import CompleteOp, StageRowOp


class Integers:
    def op(self, name, a, b, dtype="i64"):
        return {
            "add": operator.add,
            "sub": operator.sub,
            "mul": operator.mul,
            "urem": operator.mod,
            "and": operator.and_,
        }[name](a, b)

    def cmp(self, name, a, b):
        assert name == "ult"
        return (a % (1 << 64)) < (b % (1 << 64))

    def loop(self, start, stop, step, body):
        for i in range(start, stop, step):
            body(i)


@pytest.mark.parametrize(
    "window,options",
    [
        (Window2D((2, 7, 8, 3), (3, 3), stride=(1, 2), padding=(1, 1, 1, 1)), {}),
        (
            Window2D((2, 8, 9, 2), (3, 2), stride=(2, 1), dilation=(2, 2), padding=(2, 1, 1, 2)),
            {"slots": 7, "output_x_first": 1, "output_x_count": 4, "output_y_first": 1, "output_y_count": 2},
        ),
        (Window2D((1, 17, 7, 2), (3, 3), stride=(6, 2), padding=(1, 1, 1, 1)), {}),
        (Window2D((1, 3, 4, 1), (1, 1)), {}),
    ],
)
def test_actual_ir_transport_equals_original_dilated_taps_and_never_reuses_before_complete(window, options):
    plan = propose_rolling_window(
        window, AddressWindow("caller_cache", 16, 8192), source_identity="a" * 64, alignment=8, **options
    )
    module = materialize_rolling_window(plan)
    module = read_rolling_window(rolling_window_text(module), plan)
    batch, height, width, channels = window.input_shape
    data = bytes((i * 7 + 13) % 251 for i in range(batch * height * width * channels))
    cache = bytearray([255] * (16 + 8192 + 8))
    state, row_count, outputs = None, 0, []
    integers = Integers()

    def lower(kind, f):
        nonlocal state, row_count
        if kind == "begin_transport":
            assert state is None
            state = ("transport", object())
            return state[1]
        if kind == "stage_row":
            assert state == ("transport", f["transfer"])
            count, source, target = f["row_bytes"], f["source_byte"], f["cache_byte"]
            cache[target : target + count] = data[source : source + count] if f["valid"] else bytes(count)
            row_count += 1
        elif kind == "await_transport":
            assert state == ("transport", f["transfer"])
            state = ("resident", object())
            return state[1]
        elif kind == "consume":
            assert state == ("resident", f["resident"])
            b, y = f["image"], f["output_y"]
            outputs.append((b, y))
            for x in range(plan.output_x_first, plan.output_x_first + plan.output_x_count):
                for ky in range(window.kernel[0]):
                    for kx in range(window.kernel[1]):
                        iy = y * window.stride[0] - window.padding[0] + ky * window.dilation[0]
                        ix = x * window.stride[1] - window.padding[1] + kx * window.dilation[1]
                        for c in range(channels):
                            lookup = emit_rolling_window_lookup(plan, integers, y, x, ky, kx, c)
                            actual = cache[lookup["cache_byte"]] if lookup["predicate"] else 0
                            expected = (
                                data[((b * height + iy) * width + ix) * channels + c]
                                if (0 <= iy < height and 0 <= ix < width)
                                else 0
                            )
                            assert actual == expected, (b, y, x, ky, kx, c)
            state = ("consumer", object())
            return state[1]
        elif kind == "complete":
            assert state == ("consumer", f["consumer"])
            state = None

    receipt = emit_rolling_window(module, plan, integers, lower)
    assert state is None and row_count == receipt["dynamic_stage_rows"]
    assert outputs == [
        (b, y) for b in range(batch) for y in range(plan.output_y_first, plan.output_y_first + plan.output_y_count)
    ]
    assert row_count <= receipt["unstaged_reference_rows"]
    assert cache[:16] == bytes([255] * 16) and cache[16 + plan.required_bytes :] == bytes(
        [255] * (8192 + 8 - plan.required_bytes)
    )
    for coordinates in ((-1, 0, 0, 0, 0), (0, -1, 0, 0, 0), (0, 0, -1, 0, 0), (0, 0, 0, 0, channels)):
        assert not emit_rolling_window_lookup(plan, integers, *coordinates)["predicate"]


@pytest.mark.parametrize("mutation", ["source", "destination", "padding", "completion"])
def test_checker_reads_actual_transport_operands_and_completion_not_requested_metadata(mutation):
    plan = propose_rolling_window(
        Window2D((1, 5, 6, 2), (3, 3), padding=(1, 1, 1, 1)), AddressWindow("cache", 16, 256), source_identity="b" * 64
    )
    module = materialize_rolling_window(plan)
    stage = next(op for op in module.walk() if type(op) is StageRowOp)
    if mutation == "source":
        stage.operands[3] = stage.cache_byte
    elif mutation == "destination":
        stage.operands[4] = stage.source_byte
    elif mutation == "padding":
        original = stage.valid.owner
        original.properties["predicate"] = arith.CmpiOp(stage.input_row, stage.input_row, "eq").predicate
    else:
        complete = next(op for op in module.walk() if type(op) is CompleteOp)
        complete.parent.erase_op(complete)
    with pytest.raises(ValueError, match="rolling-window check"):
        verify_rolling_window(module, plan)


def test_proof_and_actual_ir_are_compact_and_capacity_is_not_a_new_hardware_limit():
    def case(height):
        return propose_rolling_window(
            Window2D((2, height, 11, 3), (3, 3), padding=(1, 1, 1, 1)),
            AddressWindow("cache", 0, 1024),
            source_identity="c" * 64,
        )

    small, large = case(9), case(70000)
    a, b = (verify_rolling_window(materialize_rolling_window(p), p) for p in (small, large))
    assert a["actual_counted_ir_operations"] == b["actual_counted_ir_operations"]
    assert a["symbolic_queries"] == b["symbolic_queries"]
    assert b["dynamic_stage_rows"] == 140004
    assert b["target_qualification"] is False
    with pytest.raises(ValueError, match="capacity"):
        replace(small, cache=AddressWindow("cache", 0, small.required_bytes - 1))
    with pytest.raises(ValueError, match="slots"):
        replace(small, slots=2)
    with pytest.raises(ValueError, match="binding handle"):
        emit_rolling_window(materialize_rolling_window(small), small, Integers(), lambda *_: None)
