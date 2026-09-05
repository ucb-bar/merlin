"""A contraction's accumulator INIT is part of the computation — carry it or refuse it.

``linalg.matmul ins(%a, %b) outs(%c)`` computes ``C + A@B``: the ``outs`` operand seeds the
reduction, which is why the op's own body is ``yield add(mul(a, b), out)``. Interface materialization
rebuilds it as ``interface.matmul`` -> ``interface.commit``, which accumulates from ZERO and never
reads the source ``outs``. When ``%c`` was a function argument that was a silent miscompile, and every
guard in the stage said the payload was complete:

    outs(%c) with a random C:   max |got - (A@W + C)| = 3.55        <- the init was gone
                                max |got -  A@W     | = 1.8e-15     <- the un-biased program

``unaccounted_ops`` could not see it because it enumerates OPS and a block argument is not an op —
the same blind spot for any init that arrives as a value rather than as a droppable op.

The fix is a value-level guard beside the op-level ones. It is a PROOF that the init contributes
nothing (``tensor.empty``; a ``linalg.fill``/``tensor.splat`` of a zero constant; an all-zero dense
constant), not a pattern: a function argument, a computed value, a non-zero constant and an
unresolvable fill all fail it, and the lowering is REFUSED with a message naming what would have been
lost. Refusing is the honest outcome here — ``interface.commit``'s epilogue vocabulary is a
per-column ``bias_add``, so there is no correct lowering of a general full-tensor init to emit.

The load-bearing test in this file is :func:`test_lower_or_refuse_never_computes_a_different_program`:
it EXECUTES every accepted lowering on the host engine and compares against numpy including the init.
The defect was invisible at the IR level — the emitted module verified at all six stages — so an
IR-inspection test could not have caught it and cannot protect against its return.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("xdsl")

from merlin.frontends.linalg_mlir import parse_mlir_text  # noqa: E402
from merlin.xdsl_dialects.lowering.input_workload import find_matmuls  # noqa: E402
from merlin.xdsl_dialects.lowering.interface_lowering import (  # noqa: E402
    LoweringError, accumulated_inits, init_contributes_nothing, lower_to_interface,
    nonzero_accumulator_inits, payload_ops, unaccounted_ops)
from merlin.xdsl_dialects.lowering.pipeline import execute, lower_module  # noqa: E402

N = 8
TT = f"tensor<{N}x{N}xf32>"

#: (label, ops that define the init, the init SSA name, the init's VALUE as numpy, is it acceptable)
#: Every module is the same contraction; only how its accumulator is initialized differs.
INITS = [
    # -- provably contributes nothing -------------------------------------------------------
    ("tensor.empty", f"    %e = tensor.empty() : {TT}", "%e", np.zeros((N, N)), True),
    ("zero fill",
     f"    %e = tensor.empty() : {TT}\n"
     "    %z = arith.constant 0.0 : f32\n"
     f"    %f = linalg.fill ins(%z : f32) outs(%e : {TT}) -> {TT}", "%f", np.zeros((N, N)), True),
    ("all-zero dense constant", f"    %k = arith.constant dense<0.0> : {TT}", "%k",
     np.zeros((N, N)), True),
    ("zero splat",
     "    %z = arith.constant 0.0 : f32\n"
     f"    %s = tensor.splat %z : {TT}", "%s", np.zeros((N, N)), True),
    # -- NOT provably zero: each of these is a real value the rebuild would drop ---------------
    ("function argument", "", "%c", None, False),          # None -> the C input array
    ("non-zero fill",
     f"    %e = tensor.empty() : {TT}\n"
     "    %z = arith.constant 1.5 : f32\n"
     f"    %f = linalg.fill ins(%z : f32) outs(%e : {TT}) -> {TT}", "%f",
     np.full((N, N), 1.5), False),
    ("non-zero dense constant", f"    %k = arith.constant dense<2.0> : {TT}", "%k",
     np.full((N, N), 2.0), False),
    ("non-zero splat",
     "    %z = arith.constant 3.0 : f32\n"
     f"    %s = tensor.splat %z : {TT}", "%s", np.full((N, N), 3.0), False),
    # A computed init. `tensor.pad` is separately unsupported by the rebuild, and stays refused.
    ("computed (tensor.pad)",
     f"    %p = tensor.pad %d low[0, 0] high[{N - 5}, 0] {{\n"
     "      ^bb0(%i: index, %j: index):\n"
     "        tensor.yield %zp : f32\n"
     f"    }} : tensor<5x{N}xf32> to {TT}", "%p", None, False),
]


def _module_text(init_ops: str, init: str) -> str:
    """One contraction over (%a, %b), accumulating onto whatever ``init`` names.

    Every extra argument is present in EVERY case (used or not) so the command buffer's tensor
    names do not depend on which init the case uses — the numeric comparison feeds them by name.
    """
    return f"""
module {{
  func.func @forward(%a: {TT}, %b: {TT}, %c: {TT}, %d: tensor<5x{N}xf32>, %zp: f32) -> {TT} {{
{init_ops}
    %r = linalg.matmul ins(%a, %b : {TT}, {TT})
                       outs({init} : {TT}) -> {TT}
    return %r : {TT}
  }}
}}
"""


def _block(module):
    return [op for op in module.walk() if op.name == "func.func"][0].body.blocks[0]


def _init_value(module):
    """The accumulator init SSA value of the module's single contraction."""
    return find_matmuls(module)[0].outputs[0]


# --------------------------------------------------------------------------------------------
# The measurement that matters: run it, and compare against numpy
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("label, init_ops, init, init_value, acceptable", INITS,
                         ids=[c[0] for c in INITS])
def test_lower_or_refuse_never_computes_a_different_program(label, init_ops, init, init_value,
                                                            acceptable):
    """For EVERY init spelling: either the pipeline refuses, or what it emits equals init + A@W.

    Executed on the host engine, not inspected. This is the shape of the original defect — a module
    that lowered, verified, emitted a command buffer, and computed the wrong thing — so the assertion
    has to be numeric. Reverting the guard turns the four "not provably zero" cases red: they lower
    again, and what comes back is A@W with the init missing.
    """
    rng = np.random.default_rng(20260905)
    a = rng.standard_normal((N, N))
    w = rng.standard_normal((N, N))
    c = rng.standard_normal((N, N))
    d = rng.standard_normal((5, N))

    module = parse_mlir_text(_module_text(init_ops, init))
    try:
        res = lower_module(module)
    except LoweringError:
        assert not acceptable, f"{label}: a provably-zero init must still lower"
        return
    assert acceptable, f"{label}: an init that is not provably zero must not lower silently"

    cb = res.command_buffer
    supplied = {"A0": a, "W": w, "A1": c, "A2": d}
    inputs = {n: v.tolist() for n, v in supplied.items() if n in cb["tensors"]}
    got = np.array(execute(res, inputs)["outputs"][cb["outputs"][0]], dtype=float)

    expected = init_value + a @ w
    assert np.allclose(got, expected, atol=1e-9), (label, float(np.abs(got - expected).max()))


def test_the_argument_init_repro_is_refused_by_name():
    """The reported defect, verbatim: outs(%c) where %c is a function argument.

    The refusal must NAME what would have been lost — an error that only said "cannot lower" would
    leave the next reader to rediscover that the init was the thing.
    """
    module = parse_mlir_text(_module_text("", "%c"))
    # The op-level guards see nothing wrong — that is why the value-level one had to exist.
    assert unaccounted_ops(_block(module), payload_ops(_block(module), find_matmuls(module))) == []
    with pytest.raises(LoweringError) as excinfo:
        lower_to_interface(module)
    msg = str(excinfo.value)
    assert "linalg.matmul" in msg
    assert "block argument" in msg          # where the dropped value came from
    assert "not provably zero" in msg
    assert "A@B" in msg                     # what would have been computed instead


def test_a_fused_bias_epilogue_does_not_excuse_a_non_zero_init():
    """The guard runs on the whole payload, not on the ops that had no epilogue.

    A contraction with a bias-add consumer is REBUILT (as a `bias_add` commit stage). That rebuild
    still accumulates from zero, so an init underneath a fused bias would be dropped exactly as
    before — a shape a guard placed only on the un-fused path would miss.
    """
    text = f"""
module {{
  func.func @forward(%a: {TT}, %b: {TT}, %c: {TT}, %bias: tensor<{N}xf32>) -> {TT} {{
    %r = linalg.matmul ins(%a, %b : {TT}, {TT}) outs(%c : {TT}) -> {TT}
    %o = tensor.empty() : {TT}
    %s = linalg.generic {{
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d1)>,
                       affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel","parallel"]
    }} ins(%r, %bias : {TT}, tensor<{N}xf32>) outs(%o : {TT}) {{
      ^bb0(%x: f32, %bb: f32, %oo: f32):
        %y = arith.addf %x, %bb : f32
        linalg.yield %y : f32
    }} -> {TT}
    return %s : {TT}
  }}
}}
"""
    with pytest.raises(LoweringError, match="not provably zero"):
        lower_to_interface(parse_mlir_text(text))


# --------------------------------------------------------------------------------------------
# The proof itself, and the derivation it rests on
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("label, init_ops, init, _v, expected", INITS, ids=[c[0] for c in INITS])
def test_the_zero_proof_answers_each_init_spelling(label, init_ops, init, _v, expected):
    """``init_contributes_nothing`` decides each spelling directly, one case per row above."""
    module = parse_mlir_text(_module_text(init_ops, init))
    assert init_contributes_nothing(_init_value(module)) is expected


def test_an_unresolvable_fill_is_not_provably_zero():
    """A fill of a value the compiler cannot read is undecidable — so it answers NO, not yes.

    Fail-closed is the whole point: a proof that returns True when it cannot see the value is not a
    proof, and this is the exact shape ("it looked like the zeroing idiom") that would let the
    original defect back in through a different door.
    """
    text = f"""
module {{
  func.func @forward(%a: {TT}, %b: {TT}, %v: f32) -> {TT} {{
    %e = tensor.empty() : {TT}
    %f = linalg.fill ins(%v : f32) outs(%e : {TT}) -> {TT}
    %r = linalg.matmul ins(%a, %b : {TT}, {TT}) outs(%f : {TT}) -> {TT}
    return %r : {TT}
  }}
}}
"""
    module = parse_mlir_text(text)
    assert init_contributes_nothing(_init_value(module)) is False
    with pytest.raises(LoweringError, match="not provably zero"):
        lower_to_interface(module)


def test_which_inits_accumulate_is_read_out_of_the_op_body():
    """A contraction READS its ``outs``; an elementwise op does not — derived, not listed by name.

    ``linalg.matmul``'s body is ``yield add(mul(a, b), out)`` and ``linalg.add``'s is
    ``yield add(a, b)``. So the init of an add is a destination and may be anything, while the init
    of a matmul is an operand of the computation. Keeping this as a structural read of the region
    (rather than a set of op names) is what makes the guard cover the quantized contraction and any
    future named op without an edit.
    """
    text = f"""
module {{
  func.func @forward(%a: {TT}, %b: {TT}, %c: {TT}) -> {TT} {{
    %e = tensor.empty() : {TT}
    %mm = linalg.matmul ins(%a, %b : {TT}, {TT}) outs(%e : {TT}) -> {TT}
    %s = linalg.add ins(%mm, %b : {TT}, {TT}) outs(%c : {TT}) -> {TT}
    return %s : {TT}
  }}
}}
"""
    module = parse_mlir_text(text)
    ops = {op.name: op for op in _block(module).ops if op.name.startswith("linalg.")}
    assert len(accumulated_inits(ops["linalg.matmul"])) == 1
    assert accumulated_inits(ops["linalg.add"]) == []
    # ... and so the non-zero init of the ADD is not held against the payload.
    assert nonzero_accumulator_inits(list(ops.values())) == []
    lower_to_interface(module).verify()


def test_an_op_with_no_inspectable_body_fails_closed():
    """No body to read -> EVERY init is reported as accumulated (never "no evidence, so fine").

    A stub without a region stands in for any future op whose ``outs`` semantics this cannot read.
    """
    class _Bodyless:
        name = "fake.contraction"
        outputs = ("an init value",)
        regions = ()

    assert accumulated_inits(_Bodyless()) == ["an init value"]


# --------------------------------------------------------------------------------------------
# The fail-closed half: what was refused before is still refused
# --------------------------------------------------------------------------------------------

MASKED_STORE = f"""
module {{
  func.func @forward(%a: {TT}, %b: {TT}) -> {TT} {{
    %e = tensor.empty() : {TT}
    %r = linalg.matmul ins(%a, %b : {TT}, {TT}) outs(%e : {TT}) -> {TT}
    %o = tensor.empty() : {TT}
    %s = "tensor.insert_slice"(%r, %o) <{{static_offsets = array<i64: 0, 0>,
          static_sizes = array<i64: {N}, {N}>, static_strides = array<i64: 1, 1>,
          operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}}>
        : ({TT}, {TT}) -> {TT}
    return %s : {TT}
  }}
}}
"""

ROW_BIAS = f"""
module {{
  func.func @forward(%a: {TT}, %b: {TT}, %bias: tensor<{N}xf32>) -> {TT} {{
    %e = tensor.empty() : {TT}
    %r = linalg.matmul ins(%a, %b : {TT}, {TT}) outs(%e : {TT}) -> {TT}
    %o = tensor.empty() : {TT}
    %s = linalg.generic {{
      indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0)>,
                       affine_map<(d0,d1)->(d0,d1)>],
      iterator_types = ["parallel","parallel"]
    }} ins(%r, %bias : {TT}, tensor<{N}xf32>) outs(%o : {TT}) {{
      ^bb0(%x: f32, %bb: f32, %oo: f32):
        %y = arith.addf %x, %bb : f32
        linalg.yield %y : f32
    }} -> {TT}
    return %s : {TT}
  }}
}}
"""

#: A pad AFTER the contraction — the shape that made `support_ops` misclassify a pad as support.
PAD_CONSUMER = f"""
module {{
  func.func @forward(%a: {TT}, %b: {TT}, %zp: f32) -> tensor<{N + 2}x{N}xf32> {{
    %e = tensor.empty() : {TT}
    %r = linalg.matmul ins(%a, %b : {TT}, {TT}) outs(%e : {TT}) -> {TT}
    %p = tensor.pad %r low[0, 0] high[2, 0] {{
      ^bb0(%i: index, %j: index):
        tensor.yield %zp : f32
    }} : {TT} to tensor<{N + 2}x{N}xf32>
    return %p : tensor<{N + 2}x{N}xf32>
  }}
}}
"""


@pytest.mark.parametrize("label, text", [
    ("masked store", MASKED_STORE),
    ("row bias (a non-bias elementwise epilogue)", ROW_BIAS),
    ("tensor.pad", PAD_CONSUMER),
])
def test_what_was_refused_before_is_still_refused(label, text):
    """The danger of teaching a fail-closed guard a new case is turning it fail-OPEN.

    ``tensor.pad`` is here because it is a known-misclassified op (``support_ops`` calls it support,
    which its docstring's "the rebuilt body re-creates their effect" does not cover): it is refused
    today by the rebuild's backstop, and this pins that it still is.
    """
    with pytest.raises(LoweringError):
        lower_to_interface(parse_mlir_text(text))


def test_the_no_epilogue_payload_is_unchanged():
    """Nothing else moves: a zero-init matmul still lowers to one matmul + one bare commit."""
    module = parse_mlir_text(_module_text(f"    %e = tensor.empty() : {TT}", "%e"))
    out = lower_to_interface(module)
    commits = [op for op in out.walk() if op.name == "interface.commit"]
    assert len(commits) == 1
    assert [e.data for e in commits[0].epilogue] == []
    assert commits[0].bias is None
    out.verify()


def test_a_vector_lane_operand_that_was_folded_away_is_a_named_refusal():
    """The same blind spot one lane over: a VALUE the rebuild does not produce.

    ``support_ops`` absorbs a dense constant because its only consumer is payload — sound for a
    contraction's zeroing scaffold, wrong for an elementwise operand, which the vector lane has to
    read from a materialized tensor. It was already fail-closed, but as a bare ``KeyError`` naming an
    SSA value: a crash no caller can catch by type. It is a ``LoweringError`` naming the operand now.
    """
    text = f"""
module {{
  func.func @forward(%a: {TT}, %b: {TT}) -> {TT} {{
    %e = tensor.empty() : {TT}
    %r = linalg.matmul ins(%a, %b : {TT}, {TT}) outs(%e : {TT}) -> {TT}
    %k = arith.constant dense<1.0> : {TT}
    %o = tensor.empty() : {TT}
    %s = linalg.add ins(%r, %k : {TT}, {TT}) outs(%o : {TT}) -> {TT}
    return %s : {TT}
  }}
}}
"""
    with pytest.raises(LoweringError, match="does not produce"):
        lower_to_interface(parse_mlir_text(text))
