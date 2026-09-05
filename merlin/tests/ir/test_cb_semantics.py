"""The command-buffer encoder must agree with the engine it claims to mirror.

This is the load-bearing test file for the whole command-buffer layer. The validator's value comes
entirely from refutations being trustworthy, and a refutation is only trustworthy if the encoder and
:func:`merlin.runtime.simulator.simulate` compute the same function. Where they diverge, a CORRECT
backend gets refuted — penalising good work, which is worse than not checking at all.

So the differential test below is not a nicety: it pins concrete inputs into the symbolic encoding,
asserts the encoded output differs from what the reference actually produced, and requires ``unsat``.
Any disagreement is an encoder bug until proven otherwise.
"""
from __future__ import annotations

import copy

import pytest

from merlin.verify import HAS_XDSL, HAS_Z3
from merlin.verify.tools import find_mlir_tool

pytestmark = pytest.mark.skipif(
    not (HAS_XDSL and HAS_Z3 and find_mlir_tool("mlir-translate")),
    reason="needs the verify extra (xdsl + z3) and mlir-translate")


def _pair(m=2, k=2, n=2, reuse=2):
    """The interface program and the command buffer the in-tree pipeline produced from it."""
    from merlin.verify.evaluate import _finish_lowering, _lower_to_interface

    iface, tc = _lower_to_interface(m, k, n, reuse)
    return iface, _finish_lowering(iface, tc), tc


# -- the differential check -------------------------------------------------------------------

def test_the_encoder_agrees_with_the_reference_simulator_on_concrete_inputs():
    """Pin the symbolic leaves to real values; the encoded outputs must equal the simulator's.

    Formulated as an SMT query rather than an evaluation: constrain each leaf element to the concrete
    input the simulator was given, assert some output element differs from the simulator's answer, and
    require `unsat`. `sat` would hand back the exact element where the two engines disagree.
    """
    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects import builtin, smt
    from xdsl.ir import Block, Region

    from merlin.runtime import simulate
    from merlin.runtime.commandbuffer import materialize_inputs
    from merlin.verify.cb_semantics import encode_command_buffer
    from merlin.verify.refine import check_module
    from merlin.verify.smt_ops import SolverOp
    from merlin.verify.smt_semantics import Encoder

    _, cb, _ = _pair()
    concrete = materialize_inputs(cb, None)          # the same values the simulator will run on
    golden = simulate(cb)["outputs"]

    blk = Block()
    with ImplicitBuilder(blk):
        enc = Encoder()
        outs, leaves = encode_command_buffer(enc, cb)

        pins = []
        for name, tensor in leaves.items():
            ref = concrete.get(name)
            if ref is None:
                continue
            rows, cols = ref.shape
            for r in range(rows):
                for c in range(cols):
                    lit = enc.const(int(ref.data[r * cols + c]), tensor.width)
                    pins.append(smt.EqOp(tensor.at(r, c), lit).results[0])
        assert pins, "no leaf could be pinned; the differential test would be vacuous"

        diffs = []
        for name, tensor in outs.items():
            want = golden[name]
            for r in range(tensor.rows):
                for c in range(tensor.cols):
                    lit = enc.const(int(want[r][c]), tensor.width)
                    eq = smt.EqOp(tensor.at(r, c), lit).results[0]
                    diffs.append(smt.NotOp(eq).results[0])
        assert diffs, "no output was compared; the differential test would be vacuous"

        term = diffs[0]
        for d in diffs[1:]:
            term = smt.OrOp(term, d).results[0]
        for p in pins:
            smt.AssertOp(p)
        smt.AssertOp(term)
        smt.YieldOp()

    verdict = check_module(builtin.ModuleOp([SolverOp.from_region(Region([blk]))]),
                           timeout_ms=120_000)
    assert verdict.status == "unsat", (
        f"the SMT encoding disagrees with merlin.runtime.simulate on concrete inputs "
        f"(status={verdict.status}, model={verdict.model_values}). Treat this as an ENCODER bug: a "
        f"validator that disagrees with its own oracle refutes correct backends.")


# -- fail-closed behaviour --------------------------------------------------------------------

@pytest.mark.parametrize("opcode,expect", [
    ("SOFTMAX", "computes in float"),          # the reference itself is float here
    ("LAYERNORM", "no branch in the reference"),  # in the schema enum, unimplemented upstream
    ("CONV2D", "gap in THIS encoder"),         # encodable in principle, not built
    ("NOT_AN_OPCODE", "no definition for"),    # genuinely unknown
])
def test_an_unencodable_opcode_raises_and_says_WHICH_class(opcode, expect):
    """Silently skipping a command changes what the query is about, without saying so.

    And "unknown" is not an actionable diagnostic: a float opcode, an opcode the reference simulator
    does not implement either, and a genuinely unrecognised mnemonic call for three different
    responses from whoever reads the abstention.
    """
    from merlin.verify.cb_semantics import CommandBufferEncoder
    from merlin.verify.smt_semantics import UnsupportedSemantics

    _, cb, _ = _pair()
    cb = copy.deepcopy(cb)
    cb["commands"][1]["opcode"] = opcode
    e = CommandBufferEncoder(_null_encoder(), cb)
    e.declare_leaves()
    with pytest.raises(UnsupportedSemantics, match=expect):
        e.run()


def test_the_opcode_classes_are_disjoint_and_cover_the_schema_enum():
    """A new opcode in the schema must land in a named class, not silently in "unknown"."""
    import json

    from merlin.common.paths import merlin_dir
    from merlin.verify.cb_semantics import (DEFERRED_OPCODES, ENCODABLE_OPCODES,
                                            FLOAT_ONLY_OPCODES, NO_NUMERIC_EFFECT,
                                            UNIMPLEMENTED_OPCODES)

    classes = [ENCODABLE_OPCODES, FLOAT_ONLY_OPCODES, UNIMPLEMENTED_OPCODES,
               frozenset(DEFERRED_OPCODES)]
    for i, a in enumerate(classes):
        for b in classes[i + 1:]:
            assert not (a & b), f"opcode in two classes at once: {sorted(a & b)}"
    assert NO_NUMERIC_EFFECT <= ENCODABLE_OPCODES, "a no-effect opcode must still be encodable"

    schema = json.loads(
        (merlin_dir() / "contract" / "schemas" / "command_buffer.schema.json").read_text())

    def _find_enum(node):
        if isinstance(node, dict):
            if "enum" in node and any(str(v).isupper() for v in node["enum"]):
                return set(node["enum"])
            for v in node.values():
                found = _find_enum(v)
                if found:
                    return found
        elif isinstance(node, list):
            for v in node:
                found = _find_enum(v)
                if found:
                    return found
        return set()

    enum = _find_enum(schema)
    assert enum, "could not locate the opcode enum; this test would be vacuous"
    unclassified = (enum - ENCODABLE_OPCODES - FLOAT_ONLY_OPCODES - UNIMPLEMENTED_OPCODES
                    - set(DEFERRED_OPCODES))
    assert not unclassified, (
        f"schema opcodes in no named class: {sorted(unclassified)} — they would abstain with an "
        f"unhelpful 'unknown' instead of saying why")


def test_a_float_epilogue_stage_abstains_and_never_passes():
    """`acc_scale` is an IEEE-754 f32 round-trip; approximating it would reject correct backends."""
    from merlin.verify.cb_semantics import ENCODABLE_EPILOGUE

    assert "acc_scale" not in ENCODABLE_EPILOGUE
    assert {"bias_add", "requant", "relu"} <= ENCODABLE_EPILOGUE


def test_res_pack_with_a_scale_operand_abstains():
    """A scale turns the pack into a per-channel dequantize to f32."""
    from merlin.verify.cb_semantics import CommandBufferEncoder
    from merlin.verify.smt_semantics import UnsupportedSemantics

    _, cb, _ = _pair()
    cb = copy.deepcopy(cb)
    pack = next(c for c in cb["commands"] if c["opcode"] == "RES_PACK")
    pack["operands"]["scale"] = "W"
    e = CommandBufferEncoder(_null_encoder(), cb)
    e.declare_leaves()
    with pytest.raises(UnsupportedSemantics, match="float is refused"):
        e.run()


# -- the overflow side condition --------------------------------------------------------------

def test_the_overflow_bound_is_derived_not_assumed():
    """The reference accumulates in unbounded ints; this encoder wraps. The bound is where they agree.

    `Tensor.matmul` documents "accumulated in i32" but never enforces it, so beyond the bound the two
    engines answer different questions and the honest verdict is an abstention.
    """
    from merlin.verify.cb_semantics import safe_k_bound

    assert safe_k_bound(8, 32) == 131071          # (2**31 - 1) // 2**14
    assert safe_k_bound(16, 32) == 1
    # narrower accumulators leave less headroom, and the bound must fall, never rise
    assert safe_k_bound(8, 16) < safe_k_bound(8, 32)


def test_a_contraction_past_the_bound_abstains_rather_than_wrapping_quietly():
    from merlin.verify.cb_semantics import CommandBufferEncoder, safe_k_bound
    from merlin.verify.smt_semantics import UnsupportedSemantics

    _, cb, _ = _pair()
    cb = copy.deepcopy(cb)
    bound = safe_k_bound(8, 32)
    # declare a K past the bound; the guard must fire before any encoding happens
    for spec in cb["tensors"].values():
        if spec["dtype"] == "i8":
            spec["shape"] = [2, bound + 1] if spec["role"] == "input" else [bound + 1, 2]
    with pytest.raises(UnsupportedSemantics, match="overflow-free bound"):
        enc = _null_encoder()
        e = CommandBufferEncoder(enc, cb)
        e.declare_leaves()
        e.run()


def _null_encoder():
    """An Encoder built outside a builder context — enough for the paths that raise before emitting."""
    from xdsl.builder import ImplicitBuilder
    from xdsl.ir import Block

    from merlin.verify.smt_semantics import Encoder

    blk = Block()
    with ImplicitBuilder(blk):
        return Encoder()
