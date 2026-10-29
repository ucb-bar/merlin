"""Projection views retain independent source products and mixed-width exits."""

from dataclasses import replace

import numpy as np
import pytest
from test_shared_input_schedule import source

from merlin.compile.scheduling.projection_views import SharedProjectionViews, plan_shared_projection_views
from merlin.compile.scheduling.shared_input_groups import SharedInputGroup, find_shared_input_groups


def views():
    return SharedProjectionViews(SharedInputGroup("source", (0, 1, 2), (0, 1, 2), 3, 7, (4, 6, 2)), 1, (1, 4, 1), 2)


def test_piecewise_original_weights_preserve_all_products_and_output_bytes():
    plan = views()
    lhs = np.arange(21, dtype=np.int32).reshape(3, 7) - 11
    weights = tuple((np.arange(7 * n, dtype=np.int32).reshape(7, n) % 13) - 6 for n in plan.source.widths)
    # Independent references do not construct a virtual concatenated weight.
    references = tuple(lhs @ weight for weight in weights)
    observed = np.zeros((3, plan.columns), dtype=np.int32)
    for r in range(3):
        for c in range(plan.columns):
            for k in range(7):
                mapping = plan.rectangle("weight", k, c, 1, 1)
                original = weights[mapping.member].reshape(-1)[mapping.byte_offset]
                observed[r, c] += lhs[r, k] * original
    for i, (first, stop) in enumerate(zip(plan.boundaries, plan.boundaries[1:])):
        np.testing.assert_array_equal(observed[:, first:stop], references[i])
        mapped = plan.rectangle("output", 2, first, 1, stop - first)
        assert mapped.member == i
        assert mapped.byte_offset == 2 * plan.source.widths[i] * plan.output_element_bytes[i]
        assert mapped.row_stride_bytes == plan.source.widths[i] * plan.output_element_bytes[i]
    assert plan.report()["host_payload_copy_bytes"] == 0


@pytest.mark.parametrize(
    "kind,row,col,rows,columns",
    [
        ("weight", 0, 3, 1, 2),
        ("output", 0, 9, 1, 2),
        ("weight", 6, 0, 2, 1),
        ("output", 2, 0, 2, 1),
        ("weight", 0, -1, 1, 1),
        ("output", 0, 12, 1, 1),
    ],
)
def test_crossings_and_out_of_domain_coordinates_refuse(kind, row, col, rows, columns):
    with pytest.raises(ValueError):
        views().rectangle(kind, row, col, rows, columns)


def test_alignment_depends_on_all_members_and_original_result_widths():
    plan = replace(views(), source=replace(views().source, widths=(24, 12, 36)))
    assert plan.aligned_panel_tiles(8) == (1, 2, 3, 6)
    assert plan.aligned_panel_tiles(4) == (1, 2, 3)
    with pytest.raises(ValueError):
        replace(plan, tile_columns=5)
    with pytest.raises(ValueError):
        replace(plan, output_element_bytes=(1, 4))
    with pytest.raises(ValueError):
        replace(plan, output_element_bytes=(1, 0, 4))
    with pytest.raises(ValueError):
        replace(plan, weight_element_bytes=2)


def test_virtual_index_space_must_fit_even_when_each_original_tensor_fits():
    plan = views()
    with pytest.raises(ValueError):
        replace(plan, source=replace(plan.source, k=1 << 61, widths=(2, 2, 2)))


def test_changed_source_group_does_not_become_current_from_a_receipt():
    from merlin.common import mlir_query

    module = mlir_query.parse("module {}")
    with pytest.raises(ValueError):
        views().verify_current(module)


def test_factory_checks_actual_immutable_input_identity_and_complete_current_source():
    module = source(3, 7, (4, 6, 2))
    (group,) = find_shared_input_groups(module)
    plan = plan_shared_projection_views(
        module, group, weight_element_bytes=1, output_element_bytes=(1, 4, 1), tile_columns=2
    )
    assert plan.boundaries == (0, 4, 10, 12)
    assert plan.report()["source"]["source_ir_sha256"] == group.source_ir_sha256
    changed = source(3, 7, (4, 6, 2), different_lhs=True)
    with pytest.raises(ValueError):
        plan.verify_current(changed)
