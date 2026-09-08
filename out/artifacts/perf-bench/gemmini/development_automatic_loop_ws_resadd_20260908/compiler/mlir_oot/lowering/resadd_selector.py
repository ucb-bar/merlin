"""Fail-closed comparison of PT2E residual semantics with Gemmini ``is_resadd``.

Gemmini's residual mode consumes two i8 tensors.  It rounds and clamps each
scaled input while moving it into the accumulator, adds the resulting integers,
then rounds/clamps the scaled sum on store.  Canonical PT2E ResNet instead keeps
each convolution accumulator in i32, applies ordered f32 scalar and per-channel
scales plus a per-channel bias, adds the f32 branches, and rounds once at the
final quantizer.  Those orders are not interchangeable.
"""
from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import TensorType
from xdsl.ir import Operation

from ..frontend.residual_epilogue import ResidualEpilogue


@dataclass(frozen=True)
class Decision:
    selected: bool
    reasons: tuple[str, ...]

    def receipt(self, formation: ResidualEpilogue,
                source_indices: dict[Operation, int]) -> dict:
        return {
            "source_op_index": source_indices[formation.residual_add],
            "selected": self.selected,
            "reasons": list(self.reasons),
            "source_shape": list(formation.shape),
            "ordered_i32_branch_count": len(formation.branches),
        }


def _rank(value) -> int | None:
    ty = value.type
    return len(ty.get_shape()) if isinstance(ty, TensorType) else None


def assess_float_residual(formation: ResidualEpilogue) -> Decision:
    """State every semantic blocker; never approximate an ineligible residual."""
    reasons: list[str] = []
    reasons.append(
        "source operands are f32 at the add; LOOP_WS is_resadd consumes i8 operands")
    if len(formation.branches) != 2:
        reasons.append(
            "one residual operand is a live f32 block value, not an exclusive i32 branch")
    if any(_rank(branch.per_channel_scale) != 0 for branch in formation.branches):
        reasons.append(
            "source uses per-channel branch scales; LOOP_WS exposes one scalar scale per input")
    if formation.branches:
        reasons.append(
            "source adds a per-channel f32 bias before the residual join; is_resadd has no bias input")
    reasons.append(
        "hardware rounds/clamps each scaled input before addition; source rounds once after f32 addition")
    reasons.append(
        "source qparams are runtime SSA tensors; LOOP_WS scale fields must be compile-time f32 constants")
    return Decision(False, tuple(reasons))
