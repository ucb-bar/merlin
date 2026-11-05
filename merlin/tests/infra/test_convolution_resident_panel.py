"""Complete-reduction residency has disjoint bounded rows, including panel tails."""

from dataclasses import replace

import pytest

from merlin.compile.scheduling.affine_gather_runs import plan_convolution_resident_panel
from merlin.compile.scheduling.block_schedule import Geometry

GEOMETRY = Geometry(16, 16384, 4096, 1024, True, {"facts": "selected-test-geometry"})


@pytest.mark.parametrize("ci,co,expected", [(64, 64, 4), (128, 128, 8), (256, 256, 7), (17, 19, 2)])
def test_complete_k_rows_are_injective_and_disjoint_for_every_active_tail(ci, co, expected):
    panel = plan_convolution_resident_panel((3, 3, ci, co), GEOMETRY, private_rows_owned=True)
    assert panel.column_blocks == expected
    assert panel.channel_tiles == (ci + 15) // 16
    for active in range(1, panel.column_blocks + 1):
        starts = [panel.rhs_row(tap, channel, column, active_column_blocks=active)
                  for tap in range(9) for channel in range(panel.channel_tiles) for column in range(active)]
        assert len(starts) == len(set(starts)) == 9 * panel.channel_tiles * active
        assert min(starts) >= panel.tile
        assert max(starts) + panel.tile <= GEOMETRY.operand_rows
        assert max(starts) + panel.tile <= panel.rhs_base + panel.rhs_rows
        assert active * panel.tile <= GEOMETRY.accumulator_rows
    for args in ((9, 0, 0), (0, panel.channel_tiles, 0), (0, 0, panel.column_blocks), (-1, 0, 0)):
        with pytest.raises(ValueError, match="leaves"):
            panel.rhs_row(*args)
    with pytest.raises(ValueError, match="leaves"):
        panel.rhs_row(0, 0, 0, active_column_blocks=True)


def test_explicit_capacity_and_private_epoch_control_admission():
    shape = (3, 3, 256, 256)
    with pytest.raises(ValueError, match="private resource epoch"):
        plan_convolution_resident_panel(shape, GEOMETRY)
    with pytest.raises(ValueError, match="separate accumulator"):
        plan_convolution_resident_panel(shape, replace(GEOMETRY, separate_accumulator_space=False),
                                        private_rows_owned=True)
    limited = plan_convolution_resident_panel(shape, replace(GEOMETRY, accumulator_rows=32),
                                             private_rows_owned=True)
    assert limited.column_blocks == 2
    assert plan_convolution_resident_panel(shape, replace(GEOMETRY, operand_rows=2304),
                                           private_rows_owned=True) is None
    assert plan_convolution_resident_panel(shape, GEOMETRY, private_rows_owned=True,
                                           max_column_blocks=3).column_blocks == 3
    for maximum in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="column bound"):
            plan_convolution_resident_panel(shape, GEOMETRY, private_rows_owned=True, max_column_blocks=maximum)


def test_forged_overlapping_or_out_of_range_layout_and_index_overflow_refuse():
    panel = plan_convolution_resident_panel((3, 3, 17, 19), GEOMETRY, private_rows_owned=True)
    for change in ({"rhs_base": 0}, {"rhs_base": 17}, {"rhs_base": GEOMETRY.operand_rows},
                   {"rhs_rows": panel.rhs_rows - 1}, {"accumulator_capacity_rows": 16},
                   {"reduction_tiles": panel.reduction_tiles + 1}):
        with pytest.raises(ValueError, match="bounded tile windows"):
            replace(panel, **change)
    for shape in ((1, 1, 1 << 62, 4), (1, 1, True, 16), (0, 1, 16, 16)):
        with pytest.raises(ValueError, match="positive geometry"):
            plan_convolution_resident_panel(shape, GEOMETRY, private_rows_owned=True)
