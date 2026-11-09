"""Explicit-budget geometry for the opt-in streaming window contraction.

VLEN describes architectural register capacity, not execution width.  No DLEN,
cache size, measured tile, or target name is inferred here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PanelTarget:
    vlen_bits: int
    elen_bits: int
    operand_bits: int
    accumulator_bits: int
    operand_lmul: int
    workspace_bytes: int
    register_budget: int = 32
    reserved_registers: int = 4
    max_mr: int = 4
    cache_budget_bytes: int | None = None

    def __post_init__(self):
        if self.vlen_bits < 128 or self.vlen_bits & (self.vlen_bits - 1):
            raise ValueError("VLEN must be a power of two >= 128 bits")
        if self.elen_bits not in (8, 16, 32, 64):
            raise ValueError("explicit supported ELEN required")
        if self.operand_bits not in (8, 16, 32, 64) or self.accumulator_bits not in (
            8,
            16,
            32,
            64,
        ):
            raise ValueError("legal RVV SEW must be 8, 16, 32, or 64 bits")
        if not (0 < self.operand_bits <= self.accumulator_bits <= self.elen_bits):
            raise ValueError("operand/accumulator SEW exceeds ELEN")
        if self.operand_bits % 8 or self.accumulator_bits % self.operand_bits:
            raise ValueError("whole-byte, integral widening required")
        if self.operand_lmul not in (1, 2, 4, 8):
            raise ValueError("integer architectural LMUL required")
        if self.accumulator_lmul not in (1, 2, 4, 8):
            raise ValueError("widened accumulator LMUL exceeds architectural limit")
        if not (0 <= self.reserved_registers < self.register_budget <= 32):
            raise ValueError("invalid architectural register budget")
        if self.workspace_bytes <= 0 or self.max_mr <= 0:
            raise ValueError("explicit positive workspace and MR budgets required")
        if self.cache_budget_bytes is not None and self.cache_budget_bytes <= 0:
            raise ValueError("cache budget must be explicit and positive")

    @property
    def accumulator_lmul(self):
        return self.operand_lmul * self.accumulator_bits // self.operand_bits


@dataclass(frozen=True)
class PanelGeometry:
    mr: int
    nr: int
    panel_width: int
    full_panels_per_row: int
    tail_width: int
    workspace_bytes: int
    source: PanelTarget

    def to_dict(self):
        return {
            **asdict(self),
            "dlen_bits": None,
            "physical_lane_utilization": None,
            "basis": "architectural_register_and_explicit_workspace_budget",
        }


def derive(target: PanelTarget, *, k: int, output_width: int) -> PanelGeometry:
    """Choose register blocks and a separately budgeted spatial panel.

    Two disjoint bounded workspaces (full panels and row tails) are reserved,
    because their lifetime after bufferization may overlap.  No global im2col
    allocation is included. MR accounts for one accumulator group per row plus
    expanded operands/broadcast scratch, rather than treating LMUL as lanes.
    """
    if k <= 0 or output_width <= 0:
        raise ValueError("positive static K/output width required")
    ratio = target.accumulator_bits // target.operand_bits
    if ratio & (ratio - 1):
        raise ValueError("power-of-two widening chain required")
    # The generic vector.contract may become full-width integer vmacc with
    # sign-extended operands, rather than a specific widening-MAC instruction.
    # Reserve an expanded B group, a broadcast/product group, and an input
    # group. This is a conservative architectural storage budget, not a
    # promise about allocator assignments or physical throughput.
    shared = 2 * target.accumulator_lmul + target.operand_lmul
    mr = min(
        target.max_mr,
        (target.register_budget - target.reserved_registers - shared)
        // target.accumulator_lmul,
    )
    if mr < 1:
        raise ValueError("register budget cannot hold one accumulator row")
    budget = min(
        target.workspace_bytes,
        target.cache_budget_bytes or target.workspace_bytes,
    )
    capacity = budget // (2 * k * (target.operand_bits // 8))
    if capacity < 1:
        raise ValueError("workspace cannot hold one full/tail activation column")
    lanes = target.vlen_bits * target.operand_lmul // target.operand_bits
    width = min(capacity, output_width)
    if width >= lanes:
        width = width // lanes * lanes
    nr = min(width, lanes)
    full, tail = divmod(output_width, width)
    actual = k * (width + tail) * (target.operand_bits // 8)
    assert actual <= budget
    return PanelGeometry(mr, nr, width, full, tail, actual, target)
