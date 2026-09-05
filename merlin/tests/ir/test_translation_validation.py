"""Translation validation of ``merlin-materialize-interface``.

A validator that has never rejected anything has not been shown to work, so the negative controls
here are the load-bearing tests: each mutates the pass's real output the way a miscompiling backend
would, and requires a refutation with a counterexample.
"""
from __future__ import annotations

import pytest

from merlin.verify import HAS_XDSL, HAS_Z3
from merlin.verify.tools import find_mlir_tool

pytestmark = pytest.mark.skipif(
    not (HAS_XDSL and HAS_Z3 and find_mlir_tool("mlir-translate")),
    reason="needs the verify extra (xdsl + z3) and mlir-translate")


#: Refutation (finding a `sat` model) is far costlier than verification (proving `unsat`), and the
#: cost climbs steeply in M.K.N. These are pinned explicitly rather than left as defaults so a
#: failure message can name the shape it was measured at.
_SHAPE = (2, 2, 2)
_REFUTE_TIMEOUT_MS = 60_000


def _module(m=2, k=2, n=2, reuse=2):
    from merlin.xdsl_dialects.lowering import pipeline
    return pipeline.lower_repeated_rhs_matmul(reuse=reuse, m=m, k=k, n=n).interface_module


def _ops(module, name):
    func = next(o for o in module.walk() if o.name == "func.func")
    return [o for o in func.body.block.ops if o.name == name]


def test_the_real_pass_output_is_verified():
    """unsat = the interface program equals the declared contraction on EVERY input at this shape."""
    from merlin.verify.refine import validate_workload

    r = validate_workload(m=2, k=2, n=2, reuse=2)
    assert r.verified, f"expected unsat, got {r}"
    assert r.n_outputs == 2


@pytest.mark.parametrize("shape", [(2, 2, 2), (2, 3, 2), (3, 2, 2)])
def test_verified_across_shapes(shape):
    """The obligation is quantified over shapes, so it is checked at more than one."""
    from merlin.verify.refine import validate_workload

    m, k, n = shape
    r = validate_workload(m=m, k=k, n=n, reuse=2)
    assert r.verified, f"expected unsat, got {r}"


# --- negative controls ---------------------------------------------------------------------------

def _miswire_commit(module):
    """The second commit reads the FIRST accumulator: a duplicated / mis-wired commit."""
    commits, matmuls = _ops(module, "interface.commit"), _ops(module, "interface.matmul")
    commits[1].operands[0] = matmuls[0].results[0]
    return "second commit reads the first accumulator"


def _swap_matmul_operands(module):
    """A @ W becomes W @ A — legal to build at square shapes, wrong arithmetic."""
    mm = _ops(module, "interface.matmul")[0]
    a, b = mm.operands[0], mm.operands[1]
    mm.operands[0], mm.operands[1] = b, a
    return "first matmul operands swapped"


def _reuse_wrong_activation(module):
    """The second matmul consumes the FIRST activation: a dropped input."""
    mm = _ops(module, "interface.matmul")
    mm[1].operands[0] = mm[0].operands[0]
    return "second matmul reuses the first activation"


@pytest.mark.parametrize("mutate", [_miswire_commit, _swap_matmul_operands, _reuse_wrong_activation])
def test_mutations_are_refuted(mutate):
    from merlin.verify.refine import validate_interface_module

    module = _module()
    label = mutate(module)
    v = validate_interface_module(module, timeout_ms=_REFUTE_TIMEOUT_MS)
    # An `unknown` is NOT the validator accepting a miscompilation -- it is the solver running out of
    # budget, and saying otherwise falsely accuses the checker. Refutation cost grows steeply with
    # M.K.N (measured: sat in 3-5 s at 4^3, unknown after 73-88 s at 16^3), so the shape is pinned
    # small on purpose and the two failure modes get different messages.
    if v.status == "unknown":
        pytest.fail(
            f"solver ABSTAINED on {label} at {_SHAPE} within {_REFUTE_TIMEOUT_MS} ms — this is a "
            f"budget/tractability failure, NOT the validator accepting the miscompilation. Raise "
            f"the bound or shrink the shape; do not read it as a correctness result.")
    assert v.refuted, f"validator ACCEPTED a miscompilation ({label}) at {_SHAPE}: status={v.status}"
    assert v.model, f"refuted {label} but produced no counterexample"


def test_timeout_is_not_a_pass():
    """An unknown verdict must never be counted as verified."""
    from merlin.verify.smt_export import Verdict

    assert not Verdict("unknown").verified


def test_operands_are_declared_at_their_own_width_not_the_accumulator_width():
    """The multiplier must be as narrow as the data, or refutation is intractable at a real shape.

    Bit-blasted multiplier area scales as width^2, so declaring an i8 element at the 32-bit
    accumulator width and constraining it down spends 16x the partial-product area to compute the
    same product. Measured 2026-09-05 under the old encoding: `swapped_matmul_operands` at 16x16x16
    returned `unknown` after 1829 s (a solver wall, not a timeout -- 30x the budget did not reach
    `sat`), while the identical query at an 8-bit multiplier width refuted in 37 s.

    This test guards the encoding, not the timing: no declared constant may be wider than the widest
    element dtype in the program.
    """
    from merlin.verify.refine import validate_interface_module

    v = validate_interface_module(_module(), timeout_ms=_REFUTE_TIMEOUT_MS)
    assert v.status == "unsat"
    widths = sorted({int(line.split("(_ BitVec")[1].split(")")[0])
                     for line in v.smt2.splitlines() if "declare-const" in line})
    assert widths == [8], (
        f"symbolic elements declared at {widths}; an i8 program must declare i8 constants and widen "
        f"with smt.bv.concat, not declare wide constants and constrain them down")
    assert "(concat" in v.smt2, "no concat in the export — the widening path is not being used"


def test_a_product_too_wide_for_the_accumulator_is_refused_not_truncated():
    """Truncating a product silently would make the checker unsound in the direction that matters."""
    import pytest as _pytest
    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region

    from merlin.verify.smt_ops import SolverOp
    from merlin.verify.smt_semantics import Encoder, UnsupportedSemantics

    module = ModuleOp([])
    with ImplicitBuilder(module.body):
        solver = SolverOp.from_region(Region([Block()]))
    with ImplicitBuilder(solver.body):
        enc = Encoder()
        a = enc.symbolic_tensor("a", 1, 1, 16)
        b = enc.symbolic_tensor("b", 1, 1, 16)
        with _pytest.raises(UnsupportedSemantics, match="refusing rather than truncating"):
            enc.matmul(a, b, acc_width=16)
