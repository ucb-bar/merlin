"""Source-to-target translation validation of ``merlin-materialize-interface`` (VER-25).

``test_translation_validation.py`` checks the pass output against a specification RE-DERIVED from the
function signature. That is target-vs-respecification: if the re-derivation and the pass share a
wrong assumption, both are wrong together and the query still returns ``unsat``. The tests here close
that gap by encoding the ACTUAL ``linalg`` module the pass consumed, so the only artifacts in the
query are the two the compiler handled.

The negative controls are the load-bearing tests. A validator that has never rejected anything has
not been shown to work, and the zero-point control in particular exists because "assume the zero
points are zero" is the single easiest way to make this check silently vacuous.
"""
from __future__ import annotations

import pytest

from merlin.verify import HAS_XDSL, HAS_Z3
from merlin.verify.tools import find_mlir_tool

pytestmark = pytest.mark.skipif(
    not (HAS_XDSL and HAS_Z3 and find_mlir_tool("mlir-translate")),
    reason="needs the verify extra (xdsl + z3) and mlir-translate")


#: Pinned explicitly, not left to a default. Refutation (finding a `sat` model) costs far more than
#: verification (proving `unsat`) and the cost climbs steeply in M*K*N. Measured 2026-09-05 on THIS
#: query: `unsat` in 0.03 s at 2^3, 0.08 s at 4^3, 0.50 s at 8^3; `sat` on a mis-wired commit in
#: 0.12 s at 2^3 and 1.42 s at 4^3. The same steep curve on the signature-derived query reached no
#: verdict at all at 16^3 in 15 minutes, which is why the shape here is small and named: a failure
#: message is only actionable if it can say what it was measured at.
_SHAPE = (2, 2, 2)
_REUSE = 2
_TIMEOUT_MS = 60_000


def _lowered(m=2, k=2, n=2, reuse=_REUSE):
    """Run the REAL pass. The modules under test are what it consumed and what it emitted."""
    from merlin.xdsl_dialects.lowering import pipeline

    return pipeline.lower_repeated_rhs_matmul(reuse=reuse, m=m, k=k, n=n)


def _func_ops(module, name):
    func = next(o for o in module.walk() if o.name == "func.func")
    return [o for o in func.body.block.ops if o.name == name]


def _assert_refuted(verdict, label):
    """A refutation, distinguished from an abstention — the two mean opposite things.

    An ``unknown`` is the solver running out of budget, NOT the validator accepting a
    miscompilation, and reporting it as the latter falsely accuses the checker.
    """
    if verdict.status == "unknown":
        pytest.fail(
            f"solver ABSTAINED on {label} at {_SHAPE} within {_TIMEOUT_MS} ms — this is a "
            f"budget/tractability failure, NOT the validator accepting the miscompilation. Shrink "
            f"the shape or raise the bound; do not read it as a correctness result.")
    assert verdict.refuted, (
        f"validator ACCEPTED a miscompilation ({label}) at {_SHAPE}: status={verdict.status}")
    assert verdict.model_values, f"refuted {label} but produced no counterexample"


# --- positive control ----------------------------------------------------------------------------

def test_the_real_pass_is_semantics_preserving_on_its_own_source():
    """unsat = source and output agree on EVERY integer input at this shape.

    This is the per-compilation theorem VER-25 is about: not "the output computes the workload we
    think we asked for", but "the output computes what the input program says".
    """
    from merlin.verify.refine import validate_pass

    r = _lowered()
    v = validate_pass(r.input_module, r.interface_module, timeout_ms=_TIMEOUT_MS)
    assert v.status == "unsat", f"expected unsat, got {v.status}"
    assert v.verified


@pytest.mark.parametrize("shape", [(2, 2, 2), (2, 3, 2), (3, 2, 2)])
def test_verified_across_shapes(shape):
    """The obligation is quantified over shapes, so it is discharged at more than one."""
    from merlin.verify.refine import validate_pass

    m, k, n = shape
    r = _lowered(m=m, k=k, n=n)
    v = validate_pass(r.input_module, r.interface_module, timeout_ms=_TIMEOUT_MS)
    assert v.status == "unsat", f"expected unsat at {shape}, got {v.status}"


def test_the_source_side_is_the_linalg_ir_not_a_respecification():
    """The spec side must come from the source module's OPS, or the check is the old one renamed.

    Encoding the source module alone yields one tensor per ``func.return`` operand, over leaves taken
    from its block arguments. If that were still derived from the signature it could not tell a
    ``linalg.quantized_matmul`` from any other op, and the abstention tests below would not fire.
    """
    from xdsl.builder import ImplicitBuilder
    from xdsl.ir import Block

    from merlin.verify.linalg_semantics import encode_linalg
    from merlin.verify.smt_semantics import Encoder

    r = _lowered()
    blk = Block()
    with ImplicitBuilder(blk):
        spec = encode_linalg(Encoder(), r.input_module)
    assert len(spec.outputs) == _REUSE
    # reuse activations + one shared weight
    assert len(spec.inputs) == _REUSE + 1
    for t in spec.outputs.values():
        assert (t.rows, t.cols, t.width) == (2, 2, 32)


# --- negative controls: mutate the TARGET side ----------------------------------------------------
# Reusing the corpus operators rather than re-writing them, so the fault a test exercises is the same
# object the detection matrix reports on.

@pytest.mark.parametrize("fault_name", [
    "miswired_commit", "swapped_matmul_operands", "dropped_activation"])
def test_interface_mutations_are_refuted(fault_name):
    from merlin.verify import faults
    from merlin.verify.refine import validate_pass

    fault = next(f for f in faults.CORPUS if f.name == fault_name)
    r = _lowered()
    fault.mutate(r.interface_module)
    v = validate_pass(r.input_module, r.interface_module, timeout_ms=_TIMEOUT_MS)
    _assert_refuted(v, f"{fault.name}: {fault.summary}")


# --- negative control: mutate the SOURCE side -----------------------------------------------------

def test_a_non_zero_zero_point_is_read_from_the_source_not_assumed():
    """Change the source's zero point and the (unchanged, previously verified) output must REFUTE.

    This is the test that makes the zero-point handling non-vacuous. If ``encode_linalg`` assumed
    ``zp = 0`` — the value the reference workload happens to use — this query would return ``unsat``
    for a source program the emitted interface module demonstrably does not implement, and every
    other test here would still pass. The pass is not wrong; the SOURCE has been changed out from
    under it, and the validator must notice.
    """
    from xdsl.dialects.builtin import IntegerAttr, i32

    from merlin.verify.refine import validate_pass

    r = _lowered()
    const = _func_ops(r.input_module, "arith.constant")[0]
    const.properties["value"] = IntegerAttr(7, i32)
    v = validate_pass(r.input_module, r.interface_module, timeout_ms=_TIMEOUT_MS)
    _assert_refuted(v, "source zero point changed to 7 while the output implements zp=0")


# --- abstentions ----------------------------------------------------------------------------------

def _module_with_symbolic_zero_point():
    """``func @f(%a, %w, %zp: i32)`` — a runtime zero point, which is legal quantized ``linalg``."""
    from xdsl.dialects import tensor as tensor_d
    from xdsl.dialects.builtin import FunctionType, ModuleOp, TensorType, i8, i32
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.dialects.linalg import ops as linalg_ops
    from xdsl.ir import Block, Region

    at, wt, ot = TensorType(i8, [2, 2]), TensorType(i8, [2, 2]), TensorType(i32, [2, 2])
    blk = Block(arg_types=[at, wt, i32])
    a, w, zp = blk.args
    init = tensor_d.EmptyOp((), ot)
    mm = linalg_ops.QuantizedMatmulOp(inputs=(a, w, zp, zp), outputs=(init.tensor,), res=(ot,))
    blk.add_ops([init, mm, ReturnOp(mm.results[0])])
    fn = FuncOp("f", FunctionType.from_lists([at, wt, i32], [ot]), Region([blk]))
    return ModuleOp([fn])


def test_a_symbolic_zero_point_abstains_rather_than_assuming_zero():
    """A zero point that is not a resolvable constant is an ABSTENTION, never a silent zero.

    Assuming zero here would be the worst possible failure mode: it would report ``unsat`` — i.e.
    "verified" — for a program whose arithmetic the encoder never actually modelled.
    """
    from merlin.verify.refine import validate_pass
    from merlin.verify.smt_semantics import UnsupportedSemantics

    r = _lowered(reuse=1)
    with pytest.raises(UnsupportedSemantics) as excinfo:
        validate_pass(_module_with_symbolic_zero_point(), r.interface_module,
                      timeout_ms=_TIMEOUT_MS)
    message = str(excinfo.value)
    assert "zero point" in message
    assert "assuming it is zero" in message


def test_an_unencoded_source_op_is_named_not_skipped():
    """A construct with no semantics raises naming itself; a skipped op weakens the theorem."""
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerAttr, i32

    from merlin.verify.refine import validate_pass
    from merlin.verify.smt_semantics import UnsupportedSemantics

    r = _lowered()
    func = next(o for o in r.input_module.walk() if o.name == "func.func")
    const = _func_ops(r.input_module, "arith.constant")[0]
    extra = arith.AddiOp(const.result, arith.ConstantOp(IntegerAttr(1, i32)).result)
    func.body.block.insert_op_after(extra, const)
    with pytest.raises(UnsupportedSemantics) as excinfo:
        validate_pass(r.input_module, r.interface_module, timeout_ms=_TIMEOUT_MS)
    assert "arith.addi" in str(excinfo.value)


def test_leaves_must_bind_or_the_query_is_refused():
    """Shared leaves are bound BY POSITION; a mismatch is refused, not coerced.

    Two sides encoded over independent symbols make the query trivially satisfiable, so this guard is
    what keeps a ``sat`` meaningful. Handing the validator a source and an output of different arity
    must abstain rather than compare whatever happens to line up.
    """
    from merlin.verify.refine import validate_pass
    from merlin.verify.smt_semantics import UnsupportedSemantics

    source = _lowered(reuse=3).input_module
    interface = _lowered(reuse=2).interface_module
    with pytest.raises(UnsupportedSemantics):
        validate_pass(source, interface, timeout_ms=_TIMEOUT_MS)


def test_timeout_is_not_a_pass():
    """An unknown verdict must never be counted as verified."""
    from merlin.verify.smt_export import Verdict

    assert not Verdict("unknown").verified
    assert not Verdict("unknown").refuted
