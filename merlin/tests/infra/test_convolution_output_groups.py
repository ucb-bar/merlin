"""Every live M/N tile has a private row window until complete readout."""

from dataclasses import replace

import pytest

from merlin.compile.scheduling.affine_gather_runs import plan_convolution_resident_panel
from merlin.compile.scheduling.block_schedule import Geometry
from merlin.compile.scheduling.convolution_output_groups import plan_convolution_output_group


@pytest.mark.parametrize("ci,co,pixels,expected", [(64, 64, 12544, 16), (128, 128, 3136, 8),
                                                (256, 256, 784, 9), (17, 19, 30, 2)])
def test_every_spatial_and_column_tail_retains_distinct_bounded_live_rows(ci, co, pixels, expected):
    geometry = Geometry(16, 16384, 4096, 1024, True, {})
    panel = plan_convolution_resident_panel((3, 3, ci, co), geometry, private_rows_owned=True)
    group = plan_convolution_output_group(panel, pixels)
    assert group.row_blocks == expected
    lhs = [group.lhs_row(row) for row in range(group.row_blocks)]
    assert len(set(lhs)) == group.row_blocks
    assert max(lhs) + panel.tile <= panel.rhs_base
    for active in range(1, panel.column_blocks + 1):
        accumulators = [group.accumulator_row(row, column, active_column_blocks=active)
                        for row in range(group.row_blocks) for column in range(active)]
        assert len(set(accumulators)) == group.row_blocks * active
        assert max(accumulators) + panel.tile <= panel.accumulator_capacity_rows
    assert plan_convolution_output_group(panel, pixels, max_row_blocks=1).row_blocks == 1
    for changes in ({"row_blocks": 0}, {"row_blocks": True},
                    {"row_blocks": panel.accumulator_capacity_rows // panel.accumulator_rows + 1}):
        with pytest.raises(ValueError, match="live spatial/output group"):
            replace(group, **changes)
    with pytest.raises(ValueError, match="spatial tile"):
        group.lhs_row(group.row_blocks)
    with pytest.raises(ValueError, match="column tile"):
        group.accumulator_row(0, panel.column_blocks)


@pytest.mark.parametrize("pixels,maximum", [(0, 4), (1, 0), (True, 4), (1 << 63, 4), (17, True)])
def test_unknown_or_overflowing_extent_and_expansion_bound_refuse(pixels, maximum):
    panel = plan_convolution_resident_panel((1, 1, 16, 16), Geometry(16, 128, 64, 64, True, {}),
                                             private_rows_owned=True)
    with pytest.raises(ValueError, match="bounded spatial"):
        plan_convolution_output_group(panel, pixels, max_row_blocks=maximum)
