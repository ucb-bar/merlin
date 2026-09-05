"""The SMT chain: xDSL ``smt`` module -> upstream ``mlir-translate --export-smtlib`` -> z3.

The chain is deliberately made of upstream parts, so these tests are mostly guarding the two seams
where it can silently lie.
"""
from __future__ import annotations

import pytest

from merlin.verify import HAS_XDSL, HAS_Z3, smt_export as E
from merlin.verify.tools import find_mlir_tool

pytestmark = pytest.mark.skipif(
    not (HAS_XDSL and HAS_Z3 and find_mlir_tool("mlir-translate")),
    reason="needs the verify extra (xdsl + z3) and mlir-translate")


def _query(const: int, width: int = 8):
    """``x == const`` inside a solver scope."""
    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects import builtin, smt
    from xdsl.ir import Block, Region

    from merlin.verify.smt_ops import SolverOp

    blk = Block()
    with ImplicitBuilder(blk):
        x = smt.DeclareFunOp(smt.BitVectorType(width), "x").results[0]
        c = smt.BvConstantOp(smt.BitVectorAttr(const, width)).results[0]
        smt.AssertOp(smt.EqOp(x, c).results[0])
        smt.YieldOp()
    return builtin.ModuleOp([SolverOp.from_region(Region([blk]))])


def test_export_produces_smtlib():
    smt2 = E.to_smtlib(_query(3))
    assert "(declare-const x (_ BitVec 8))" in smt2
    assert "(assert" in smt2


def test_sat_query_yields_a_nonempty_model():
    """Regression for the exporter's trailing ``(reset)``.

    Handed to z3 verbatim, the query answers ``sat`` with an EMPTY model — the reset discards it, so
    a counterexample vanishes while the verdict still looks meaningful. If this ever returns an
    empty model again, every 'refuted' result has silently lost its witness.
    """
    v = E.check_module(_query(3))
    assert v.refuted, v.status
    assert v.model and "x" in v.model, f"counterexample was discarded: {v.model!r}"


def test_reset_is_actually_present_upstream():
    """Pin the upstream behaviour the strip exists for, so the workaround is removed when obsolete."""
    smt2 = E.to_smtlib(_query(3))
    assert "(reset)" in smt2, "upstream no longer emits (reset); strip_reset can be retired"
    assert "(reset)" not in E.strip_reset(smt2)


def test_unsat_is_the_verified_state():
    """A contradiction is unsat; unsat is what a refinement obligation must return to PASS."""
    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects import builtin, smt
    from xdsl.ir import Block, Region

    from merlin.verify.smt_ops import SolverOp

    blk = Block()
    with ImplicitBuilder(blk):
        x = smt.DeclareFunOp(smt.BitVectorType(8), "x").results[0]
        c3 = smt.BvConstantOp(smt.BitVectorAttr(3, 8)).results[0]
        c4 = smt.BvConstantOp(smt.BitVectorAttr(4, 8)).results[0]
        smt.AssertOp(smt.EqOp(x, c3).results[0])
        smt.AssertOp(smt.EqOp(x, c4).results[0])
        smt.YieldOp()
    v = E.check_module(builtin.ModuleOp([SolverOp.from_region(Region([blk]))]))
    assert v.status == "unsat" and v.verified


def test_unknown_is_not_verified():
    """A timeout must never read as a pass."""
    v = E.Verdict("unknown")
    assert not v.verified and not v.refuted
