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
    from xdsl.dialects.builtin import IntegerAttr, i64
    from xdsl.dialects.smt import BitVectorType, BoolType
    from xdsl.ir import Dialect, Region
    from xdsl.irdl import (IRDLOperation, irdl_op_definition, operand_def, prop_def, region_def,
                           result_def)

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

    #: Predicate ordinals for :class:`BVCmpOp`, MEASURED rather than assumed.
    #:
    #: Upstream spells the predicate as a keyword in custom syntax but stores it as an ``i64``
    #: ordinal, which is what xDSL must emit since it prints generic form. The mapping below was
    #: derived by round-tripping every keyword through ``mlir-opt --mlir-print-op-generic`` and
    #: reading the ordinal back; :func:`bv_cmp_exports_as` and the tests assert the exported SMT-LIB
    #: still says ``bvslt``, so a change in upstream's enum order fails loudly instead of silently
    #: inverting a comparison.
    BV_CMP_PREDICATES = {"slt": 0, "sle": 1, "sgt": 2, "sge": 3,
                         "ult": 4, "ule": 5, "ugt": 6, "uge": 7}

    @irdl_op_definition
    class BVCmpOp(IRDLOperation):
        """``smt.bv.cmp`` — a bitvector comparison, exported as ``(bvslt a b)`` and friends.

        Needed for the readout epilogue: ``relu`` is ``ite(slt(x, 0), 0, x)`` and the saturating
        narrow to i8 is two clamps, neither expressible with the arithmetic ops xDSL ships.
        """
        name = "smt.bv.cmp"
        lhs = operand_def(BitVectorType)
        rhs = operand_def(BitVectorType)
        result = result_def(BoolType)
        pred = prop_def(IntegerAttr)

        @staticmethod
        def get(pred: str, lhs, rhs) -> "BVCmpOp":
            if pred not in BV_CMP_PREDICATES:
                raise ValueError(f"unknown bv compare predicate {pred!r}; "
                                 f"known: {sorted(BV_CMP_PREDICATES)}")
            return BVCmpOp(operands=[lhs, rhs], result_types=[BoolType()],
                           properties={"pred": IntegerAttr(BV_CMP_PREDICATES[pred], i64)})

    #: Registered separately from xDSL's own ``smt`` Dialect object so nothing upstream is mutated.
    SMT_SOLVER = Dialect("smt", [SolverOp, BVConcatOp, BVCmpOp], [])
else:  # pragma: no cover - import guard
    SolverOp = None  # type: ignore[assignment]
    BVConcatOp = None  # type: ignore[assignment]
    BVCmpOp = None  # type: ignore[assignment]
    BV_CMP_PREDICATES = {}  # type: ignore[assignment]
    SMT_SOLVER = None  # type: ignore[assignment]
