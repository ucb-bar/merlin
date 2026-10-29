"""Actual-operation witnesses for conservative scalar reuse and invariant motion.

These records carry no proof authority. Their replay validator is part of the
explicit structural-checker trusted computing base, separately from Lean lemmas.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScalarRewriteWitness:
    operation_index: int
    operation: str
    rule: str
    width: int
    operation_sha256: str
    representative_index: int | None = None
    loop_index: int | None = None


SCALAR_SCOPES = {
    "scalar_redundancy_elimination": (
        "total_scalar_integer_operations",
        "preserved_unknown_effects",
        "ssa_use_def_and_dominance",
    ),
    "scalar_loop_invariant_motion": (
        "total_scalar_integer_operations",
        "total_speculation",
        "loop_external_operands",
        "unchanged_effect_order",
    ),
}


def validate_scalar_change(change) -> None:
    """Reopen and replay the admitted mechanism against the actual complete IR."""
    from .certificates import ir_text, parse_ir
    from .pure_scalar import eliminate_scalar_redundancy, hoist_scalar_invariants

    change.validate_identity()
    if change.kind not in SCALAR_SCOPES or change.certificate.scope != SCALAR_SCOPES[change.kind]:
        raise ValueError("scalar transformation coverage differs from the fixed semantic domain")
    if change.parameters or any(not isinstance(w, ScalarRewriteWitness) for w in change.certificate.witnesses):
        raise ValueError("scalar transformation has unsupported parameters or witnesses")
    before, after = parse_ir(change.before_ir), parse_ir(change.after_ir)
    transform = {
        "scalar_redundancy_elimination": eliminate_scalar_redundancy,
        "scalar_loop_invariant_motion": hoist_scalar_invariants,
    }[change.kind]
    witnesses = transform(before)
    if not witnesses or witnesses != change.certificate.witnesses or ir_text(before) != ir_text(after):
        raise ValueError("actual scalar IR or witnesses differ from the admitted rewrite")
