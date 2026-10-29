"""Counted member partitions retain original pointers, products and scalar exits."""

from dataclasses import replace

import numpy as np
import pytest
from test_compact_dense import ADAPTER
from test_dense_retained_reduction import chosen, inputs
from test_shared_input_schedule import source

from merlin.compile.scheduling.block_schedule import Compute, Load, Preload, Store
from merlin.compile.scheduling.projection_dense import bind_dense_projection
from merlin.compile.scheduling.projection_views import plan_shared_projection_views
from merlin.compile.scheduling.shared_input_groups import find_shared_input_groups
from merlin.sched.ir import Call, evaluate, instances
from merlin.transforms.dense_planning import dense_blocks


def setup(m=9, k=14, widths=(8, 4, 12)):
    module = source(m, k, widths)
    (group,) = find_shared_input_groups(module)
    views = plan_shared_projection_views(
        module, group, weight_element_bytes=1, output_element_bytes=(4,) * len(widths), tile_columns=4
    )
    p, r, s, _b, a = inputs((m, k, sum(widths)))
    b = next(b for b in dense_blocks(p, r) if (b.m, b.k, b.n) == (2, p.tiles[1], 1))
    original = chosen(p, r, s, b, a, role="lhs", order=("m", "n"), compute_order=("k", "n", "m"))
    return module, views, p, original


def bind(model, **kwargs):
    module, views, p, original = model
    return bind_dense_projection(
        original.kernel,
        views,
        ADAPTER,
        source_module=module,
        column_loop_paths=(("outer_m", "outer_n"), ("outer_n",)) if p.tiles[0] % 2 else (("outer_m", "outer_n"),),
        **{"panel_tiles": 1, **kwargs},
    )


@pytest.mark.parametrize("m,k,widths", [(9, 14, (8, 4, 12)), (12, 12, (4, 12, 8))])
def test_actual_counted_pointer_maps_execute_every_original_product_and_tail(m, k, widths):
    model = setup(m, k, widths)
    module, views, p, original = model
    result = bind(model)
    lhs = (np.arange(m * k) % 7 - 3).reshape(m, k)
    weights = [(np.arange(k * n) % 5 - 2).reshape(k, n) for n in widths]
    output = [np.zeros((m, n), dtype=np.int64) for n in widths]
    memory = {"lhs": {}, "weight": {}, "acc": {}}
    descriptor = None
    reads = 0
    for call, env in instances(result.kernel):
        values = {name: evaluate(v, env) for name, v in call.args}
        if call.instr == "merlin_dense_panel_drain":
            continue
        member = values.pop("projection_member", None)
        column = values.pop("projection_local_column", None)
        row_width = values.pop("projection_row_columns", None)
        operation = ADAPTER.project(
            replace(call, args=tuple((n, v) for n, v in call.args if not n.startswith("projection_")))
        )

        def value(x):
            return evaluate(x, env)

        if type(operation) is Load:
            row, col, rows, cols, local = map(
                value, (operation.dram_row, operation.dram_col, operation.rows, operation.cols, operation.row)
            )
            tensor = lhs if operation.role == "lhs" else weights[member]
            if member is not None:
                assert row_width == widths[member]
                col = column
            else:
                reads += rows * cols
            for i in range(rows):
                for j in range(cols):
                    memory[operation.role][local + i, j] = int(tensor[row + i, col + j])
        elif type(operation) is Preload:
            descriptor = operation
        elif type(operation) is Compute:
            row, rows, depth = map(value, (operation.input_row, operation.rows, operation.cols))
            ar, wr, cols = map(value, (descriptor.accumulator_row, descriptor.weight_row, descriptor.cols))
            for i in range(rows):
                for j in range(cols):
                    total = memory["acc"][ar + i, j] if value(descriptor.accumulate) else 0
                    total += sum(memory["lhs"][row + i, q] * memory["weight"][wr + q, j] for q in range(depth))
                    memory["acc"][ar + i, j] = total
        else:
            assert type(operation) is Store and row_width == widths[member]
            ar, row, rows, cols = map(
                value, (operation.accumulator_row, operation.dram_row, operation.rows, operation.cols)
            )
            for i in range(rows):
                for j in range(cols):
                    output[member][row + i, column + j] = memory["acc"][ar + i, j]
    for got, weight in zip(output, weights):
        np.testing.assert_array_equal(got, lhs @ weight)
    assert reads == m * k
    assert len(result.report["physical_sites"]) < 32 and not result.report["dynamic_command_expansion"]
    result.revalidate()
    with pytest.raises(ValueError, match="mapping changed"):
        replace(result, kernel=original.kernel).revalidate()


def test_crossing_or_unpartitioned_physical_rectangles_are_not_granted_member_identity():
    model = setup()
    with pytest.raises(ValueError, match="aligned"):
        bind(model, panel_tiles=2)
    module, views, p, original = model
    with pytest.raises(ValueError):
        bind_dense_projection(
            original.kernel, views, ADAPTER, source_module=module, column_loop_paths=(("outer_m",),), panel_tiles=1
        )


def test_original_member_coefficients_and_output_precision_cannot_be_replaced():
    from merlin.compile.scheduling.dense_contracts import DenseTiledIntegerReadout

    module, views, problem, original = setup()
    views = replace(views, output_element_bytes=(1, 1, 1))
    readouts = tuple(
        DenseTiledIntegerReadout.from_column_scales([scale] * width, tile_columns=4)
        for scale, width in zip((0.125, 0.25, 0.5), views.source.widths)
    )
    words = tuple(word for readout in readouts for word in readout.column_scale_bits)
    virtual = replace(readouts[0], column_scale_bits=words)
    options = dict(
        source_module=module, column_loop_paths=(("outer_m", "outer_n"), ("outer_n",)), panel_tiles=1, readouts=readouts
    )
    changed = replace(virtual, column_scale_bits=(readouts[1].column_scale_bits[0],) * 4 + words[4:])
    with pytest.raises(ValueError, match="original column coefficient"):
        bind_dense_projection(original.kernel, views, ADAPTER, virtual_readout=changed, **options)
    with pytest.raises(ValueError, match="precision differs"):
        bind_dense_projection(original.kernel, views, ADAPTER, virtual_readout=virtual, **options)
