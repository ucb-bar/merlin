"""Grouped transport proves actual SCF and retains every original source tap."""

import operator
from dataclasses import replace

import pytest
from xdsl.dialects import arith, builtin, scf

from merlin.compile.scheduling.program_limits import AddressWindow
from merlin.compile.scheduling.rolling_window import (
    emit_grouped_rolling_window,
    emit_rolling_window_lookup,
    materialize_grouped_rolling_window,
    propose_rolling_window,
    read_grouped_rolling_window,
    rolling_window_text,
    verify_grouped_rolling_window,
    verify_rolling_window,
)
from merlin.compile.scheduling.window_transfer import Window2D
from merlin.xdsl_dialects.rolling_window import AwaitOp, CompleteOp, ConsumeGroupOp, StageRowOp


class Integers:
    def op(self, name, a, b, dtype="i64"):
        return {"add": operator.add, "sub": operator.sub, "mul": operator.mul,
                "urem": operator.mod, "and": operator.and_}[name](a, b)

    def cmp(self, name, a, b):
        assert name == "ult"
        return a % (1 << 64) < b % (1 << 64)

    def loop(self, start, stop, step, body):
        for index in range(start, stop, step):
            body(index)


def plan_for(window, rows, **options):
    slots = (window.kernel[0] - 1) * window.dilation[0] + 1 + (rows - 1) * window.stride[0]
    return propose_rolling_window(window, AddressWindow("private", 16, 8192),
        source_identity="a" * 64, slots=slots, alignment=8, **options)


@pytest.mark.parametrize("window,rows,options", [
    (Window2D((2, 13, 8, 3), (3, 3), padding=(1, 1, 1, 1)), 4, {}),
    (Window2D((2, 17, 9, 2), (3, 2), stride=(2, 1), dilation=(2, 2), padding=(2, 1, 1, 2)),
        3, {"output_x_first": 1, "output_x_count": 4, "output_y_first": 1, "output_y_count": 7}),
    (Window2D((1, 5, 6, 2), (3, 3)), 4, {}),
    (Window2D((1, 9, 5, 1), (3, 3), stride=(3, 1)), 2, {}),
    (Window2D((1, 5, 6, 2), (3, 3)), 1, {}),
])
def test_group_union_stages_original_rows_once_and_keeps_all_taps_until_complete(window, rows, options):
    plan = plan_for(window, rows, **options)
    module = materialize_grouped_rolling_window(plan, output_rows=rows)
    module = read_grouped_rolling_window(rolling_window_text(module), plan, output_rows=rows)
    integers = Integers()
    batch, height, width, channels = window.input_shape
    source = bytes((i * 17 + 7) % 251 for i in range(batch * height * width * channels))
    cache = bytearray([255] * (plan.cache.start + plan.required_bytes + 16))
    state, pending, seen, stages, groups = None, [], [], 0, 0

    def lower(kind, fields):
        nonlocal state, stages, groups
        if kind == "begin_transport":
            assert state is None and not pending
            state = ("transport", object())
            return state[1]
        if kind == "stage_row":
            assert state == ("transport", fields["transfer"])
            address, count, offset = fields["cache_byte"], fields["row_bytes"], fields["source_byte"]
            assert all(address != previous[0] for previous in pending)
            # Issue is deliberately asynchronous: only await publishes bytes.
            pending.append((address, source[offset:offset + count] if fields["valid"] else bytes(count)))
            stages += 1
        elif kind == "await_transport":
            assert state == ("transport", fields["transfer"])
            for address, data in pending:
                cache[address:address + len(data)] = data
            pending.clear()
            state = ("resident", object())
            return state[1]
        elif kind == "consume_group":
            assert state == ("resident", fields["resident"]) and not pending
            groups += 1
            image = fields["image"]
            for y in range(fields["output_y"], fields["output_y"] + fields["output_rows"]):
                seen.append((image, y))
                for x in range(plan.output_x_first, plan.output_x_first + plan.output_x_count):
                    for ky in range(window.kernel[0]):
                        for kx in range(window.kernel[1]):
                            iy = y * window.stride[0] - window.padding[0] + ky * window.dilation[0]
                            ix = x * window.stride[1] - window.padding[1] + kx * window.dilation[1]
                            for channel in range(channels):
                                lookup = emit_rolling_window_lookup(plan, integers, y, x, ky, kx, channel)
                                expected = (
                                    source[((image * height + iy) * width + ix) * channels + channel]
                                    if 0 <= iy < height and 0 <= ix < width else 0
                                )
                                assert (cache[lookup["cache_byte"]] if lookup["predicate"] else 0) == expected
            state = ("consumer", object())
            return state[1]
        elif kind == "complete":
            assert state == ("consumer", fields["consumer"])
            state = None
        else:
            raise AssertionError(kind)

    receipt = emit_grouped_rolling_window(module, plan, integers, lower, output_rows=rows)
    assert state is None and not pending
    assert seen == [(b, y) for b in range(batch)
                    for y in range(plan.output_y_first, plan.output_y_first + plan.output_y_count)]
    assert groups == receipt["consumer_groups"] == batch * ((plan.output_y_count + rows - 1) // rows)
    assert stages == receipt["dynamic_stage_rows"] == batch * (plan.span_rows + (plan.output_y_count - 1) * window.stride[0])
    assert cache[:16] == bytes([255] * 16) and cache[-16:] == bytes([255] * 16)
    assert receipt["target_qualification"] is False
    with pytest.raises(ValueError, match="identity changed"):
        verify_rolling_window(module, plan)


@pytest.mark.parametrize("mutation", ["source", "destination", "padding", "rows", "group_origin", "frontier", "await", "completion"])
def test_group_checker_refuses_mutated_actual_source_union_consumer_and_boundaries(mutation):
    plan = plan_for(Window2D((2, 13, 8, 3), (3, 3), padding=(1, 1, 1, 1)), 4)
    module = materialize_grouped_rolling_window(plan, output_rows=4)
    stages = [op for op in module.walk() if type(op) is StageRowOp]
    if mutation == "source":
        stages[0].operands[3] = stages[0].cache_byte
    elif mutation == "destination":
        stages[-1].operands[4] = stages[-1].source_byte
    elif mutation == "padding":
        stages[0].valid.owner.properties["predicate"] = arith.CmpiOp(stages[0].input_row, stages[0].input_row, "eq").predicate
    elif mutation == "rows":
        next(op for op in module.walk() if type(op) is ConsumeGroupOp).properties["output_rows"] = builtin.IntegerAttr(3, 64)
    elif mutation == "group_origin":
        consumer = next(op for op in module.walk() if type(op) is ConsumeGroupOp)
        consumer.operands[2] = consumer.image
    elif mutation == "frontier":
        # An actual inner stage loop must cover every newly required row.
        loop = next(op for op in module.walk() if type(op) is scf.ForOp
                    and any(type(child) is StageRowOp for child in op.body.block.ops))
        loop.ub.owner.properties["value"] = builtin.IntegerAttr(5, 64)
    else:
        cls = AwaitOp if mutation == "await" else CompleteOp
        op = next(op for op in module.walk() if type(op) is cls)
        if cls is AwaitOp:
            # Preserve valid SSA while moving readiness after its consumer.
            parent = op.parent
            parent.detach_op(op)
            consumer = next(child for child in parent.ops if type(child) is ConsumeGroupOp)
            parent.insert_op_after(op, consumer)
        else:
            op.parent.erase_op(op)
    with pytest.raises(ValueError, match="rolling-window check"):
        verify_grouped_rolling_window(module, plan, output_rows=4)


def test_group_horizon_capacity_bounds_and_compact_counted_proof():
    def case(height):
        return plan_for(Window2D((2, height, 8, 3), (3, 3), padding=(1, 1, 1, 1)), 4)
    small, large = case(13), case(70001)
    receipts = [verify_grouped_rolling_window(materialize_grouped_rolling_window(plan, output_rows=4),
                plan, output_rows=4) for plan in (small, large)]
    assert receipts[0]["actual_counted_ir_operations"] == receipts[1]["actual_counted_ir_operations"]
    assert receipts[0]["symbolic_queries"] == receipts[1]["symbolic_queries"]
    with pytest.raises(ValueError, match="ring slots"):
        materialize_grouped_rolling_window(replace(small, slots=5), output_rows=4)
    with pytest.raises(ValueError, match="overlapping or adjacent"):
        materialize_grouped_rolling_window(plan_for(Window2D((1, 24, 8, 3), (3, 3), stride=(5, 1)), 2), output_rows=2)
    for bound in (0, 17, True):
        with pytest.raises(ValueError, match="bound"):
            materialize_grouped_rolling_window(small, output_rows=bound)
    with pytest.raises(ValueError, match="binding handle"):
        emit_grouped_rolling_window(materialize_grouped_rolling_window(small, output_rows=4),
                                   small, Integers(), lambda *_: None, output_rows=4)
