"""Bound live spatial/output-channel tiles around a complete resident RHS panel."""

from dataclasses import dataclass

from .affine_gather_runs import ConvolutionResidentPanel


@dataclass(frozen=True)
class ConvolutionOutputGroup:
    panel: ConvolutionResidentPanel
    row_blocks: int

    def __post_init__(self):
        if type(self.panel) is not ConvolutionResidentPanel:
            raise ValueError("original complete resident panel required")
        self.panel.__post_init__()
        if (type(self.row_blocks) is not int or self.row_blocks <= 0
                or self.row_blocks * self.panel.tile > self.panel.rhs_base
                or self.row_blocks * self.panel.accumulator_rows > self.panel.accumulator_capacity_rows):
            raise ValueError("live spatial/output group exceeds its disjoint operand or accumulator windows")

    def lhs_row(self, row_block):
        if type(row_block) is not int or not 0 <= row_block < self.row_blocks:
            raise ValueError("spatial tile exceeds its live output group")
        return row_block * self.panel.tile

    def accumulator_row(self, row_block, column_block, *, active_column_blocks=None):
        active = self.panel.column_blocks if active_column_blocks is None else active_column_blocks
        if (type(active) is not int or not 0 < active <= self.panel.column_blocks
                or type(column_block) is not int or not 0 <= column_block < active):
            raise ValueError("column tile exceeds its live output group")
        self.lhs_row(row_block)
        return (row_block * active + column_block) * self.panel.tile


def plan_convolution_output_group(panel, pixels, *, max_row_blocks=16):
    """Select a bounded M group without spilling full-K live accumulators.

    The static unroll bound is explicit and limits code expansion; capacity
    chooses the remaining extent. Profitability and target weight reuse remain
    separate choices. Each K epoch loads all LHS tiles before immutable reads;
    the caller must complete every read before the next epoch overwrites them.
    """
    if type(panel) is not ConvolutionResidentPanel:
        raise ValueError("original complete resident panel required")
    panel.__post_init__()
    if any(type(value) is not int or not 0 < value < 1 << 63 for value in (pixels, max_row_blocks)):
        raise ValueError("positive bounded spatial extent and static unroll limit required")
    blocks = min((pixels + panel.tile - 1) // panel.tile, panel.rhs_base // panel.tile,
                 panel.accumulator_capacity_rows // panel.accumulator_rows, max_row_blocks)
    return ConvolutionOutputGroup(panel, blocks)
