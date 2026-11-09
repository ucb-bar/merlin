"""Bounded translation validation for an exact signed split reduction.

This is an algebraic pilot, not a verifier for an emitted kernel. The caller must
separately establish that its source and target IR implement these two expressions,
and that the declared widths agree with the selected hardware facts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SplitReductionVerdict:
    status: str  # verified | refuted | unknown
    k: int
    operand_width: int
    partial_width: int
    output_width: int
    chunks: tuple[int, ...]
    counterexample: tuple[tuple[int, int], ...] | None = None
    reason: str | None = None


def maximum_safe_chunk(operand_width: int, partial_width: int) -> int:
    """Conservative full-domain K bound for a signed product sum.

    The maximum absolute product of two signed w-bit values is 2**(2*w-2).
    Bounding absolute sums by K times that product covers both signs.
    """
    if operand_width < 2 or partial_width < operand_width * 2:
        raise ValueError("partial width must hold a full signed operand product")
    return ((1 << (partial_width - 1)) - 1) // (1 << (2 * operand_width - 2))


def verify_split_reduction(
    *,
    k: int,
    operand_width: int,
    partial_width: int,
    output_width: int,
    chunks: tuple[int, ...],
    timeout_ms: int = 30_000,
) -> SplitReductionVerdict:
    """Prove or refute equality for all signed input pairs at this concrete K.

    The source multiplies and accumulates at ``output_width`` modulo 2**width.
    Each target chunk multiplies and accumulates at ``partial_width`` modulo
    2**width, then sign-extends its partial sum into the output accumulator.
    Missing/duplicated reduction terms are rejected before invoking the solver.
    ``unknown`` is never a proof.
    """
    if k < 1 or timeout_ms < 1 or operand_width < 2:
        raise ValueError("k, timeout and operand width must be positive and nontrivial")
    if partial_width < operand_width * 2 or output_width < partial_width:
        raise ValueError("widths must satisfy output >= partial >= twice operand width")
    if not chunks or any(size < 1 for size in chunks) or sum(chunks) != k:
        raise ValueError("positive chunks must cover every reduction term exactly once")

    import z3

    lhs = [z3.BitVec(f"lhs_{i}", operand_width) for i in range(k)]
    rhs = [z3.BitVec(f"rhs_{i}", operand_width) for i in range(k)]

    def product(i: int, width: int):
        return z3.SignExt(width - operand_width, lhs[i]) * z3.SignExt(width - operand_width, rhs[i])

    source = z3.BitVecVal(0, output_width)
    for i in range(k):
        source += product(i, output_width)

    target = z3.BitVecVal(0, output_width)
    offset = 0
    for count in chunks:
        partial = z3.BitVecVal(0, partial_width)
        for i in range(offset, offset + count):
            partial += product(i, partial_width)
        target += z3.SignExt(output_width - partial_width, partial)
        offset += count

    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    solver.add(source != target)
    result = solver.check()
    base = dict(
        k=k,
        operand_width=operand_width,
        partial_width=partial_width,
        output_width=output_width,
        chunks=chunks,
    )
    if result == z3.unsat:
        return SplitReductionVerdict(status="verified", **base)
    if result == z3.sat:
        model = solver.model()
        sign_bit = 1 << (operand_width - 1)
        modulus = 1 << operand_width

        def signed(value):
            raw = model.eval(value, model_completion=True).as_long()
            return raw - modulus if raw & sign_bit else raw

        return SplitReductionVerdict(
            status="refuted",
            counterexample=tuple((signed(lhs[i]), signed(rhs[i])) for i in range(k)),
            **base,
        )
    return SplitReductionVerdict(status="unknown", reason=solver.reason_unknown(), **base)


# Preserve published exception/record identities through the exact legacy alias.
SplitReductionVerdict.__module__ = "merlin.verify.split_reduction"
