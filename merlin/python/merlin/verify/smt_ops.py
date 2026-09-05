"""The ``smt`` ops xDSL 0.68 does not ship: the solver scope, and bitvector concatenation.

xDSL vendors upstream MLIR's ``smt`` dialect (bitvectors, quantifiers, ``declare_fun``, ``assert``,
``yield``) but not ``smt.solver`` / ``smt.check``. Upstream's ``mlir-translate --export-smtlib``
requires the solver scope — it is what delimits one query — so without it a module of ``smt`` ops
cannot be exported at all.

``smt.check`` is genuinely unnecessary here: the solver API supplies ``(check-sat)``. Only the scope
is missing, and it is a region-holding op with no operands or results, so this is twenty lines rather
than a fork.

Measured: a module built this way prints in generic form and is accepted verbatim by upstream
``mlir-translate --export-smtlib``.

``smt.bv.concat`` is the second gap, and it is the expensive one. Without it the encoder had to
declare every tensor element at the ACCUMULATOR width and constrain it down to its real element
range, which makes an i8 x i8 product a full 32x32 bit-blasted multiplier carrying eight meaningful
bits. Measured 2026-09-05: at 16x16x16 that query never refutes (``unknown`` at a 1829 s budget),
while the identical query at an 8-bit multiplier width refutes in 37 s -- partial-product area scales
as terms x width^2, so the wasted width was the whole cost. Upstream's exporter emits ``(concat a b)``
for this op, verified by running it.
"""
from __future__ import annotations

from . import HAS_XDSL

if HAS_XDSL:
    from xdsl.dialects.smt import BitVectorType
    from xdsl.ir import Dialect, Region
    from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, region_def, result_def

    @irdl_op_definition
    class SolverOp(IRDLOperation):
        """``smt.solver`` — the scope one SMT query lives in.

        Mirrors upstream MLIR's op of the same name. The body is terminated by the existing
        ``smt.yield``; assertions inside the region become the query's assertions.
        """
        name = "smt.solver"
        body = region_def()

        @staticmethod
        def from_region(region: Region) -> "SolverOp":
            return SolverOp(regions=[region])

    @irdl_op_definition
    class BVConcatOp(IRDLOperation):
        """``smt.bv.concat`` — bitvector concatenation, exported as SMT-LIB ``(concat lhs rhs)``.

        The result width is the sum of the operand widths. Mirrors upstream MLIR's op of the same
        name; xDSL 0.68 ships no extract/concat/extend, which is why this is defined here rather than
        imported. With it, an ``n``-bit value sign-extends to ``2n`` bits as
        ``concat(ashr(x, n-1), x)`` — the high half of an arithmetic right shift by ``n-1`` is all
        sign bits — so no separate extend op is needed.
        """
        name = "smt.bv.concat"
        lhs = operand_def(BitVectorType)
        rhs = operand_def(BitVectorType)
        result = result_def(BitVectorType)

        @staticmethod
        def get(lhs, rhs) -> "BVConcatOp":
            width = lhs.type.width.data + rhs.type.width.data
            return BVConcatOp(operands=[lhs, rhs], result_types=[BitVectorType(width)])

    #: Registered separately from xDSL's own ``smt`` Dialect object so nothing upstream is mutated.
    SMT_SOLVER = Dialect("smt", [SolverOp, BVConcatOp], [])
else:  # pragma: no cover - import guard
    SolverOp = None  # type: ignore[assignment]
    BVConcatOp = None  # type: ignore[assignment]
    SMT_SOLVER = None  # type: ignore[assignment]
