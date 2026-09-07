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


@pytest.mark.parametrize("mutate,label", [
    (lambda cb: [c["attributes"].__setitem__("output_dtype", "i16")
                 for c in cb["commands"] if c["opcode"] == "COMMIT"], "output_dtype i16"),
    (lambda cb: [c["attributes"].__setitem__("output_dtype", "u8")
                 for c in cb["commands"] if c["opcode"] == "COMMIT"], "output_dtype u8"),
])
def test_the_encoder_tracks_the_engine_on_readout_variants(mutate, label):
    """The differential test above only ever saw a DECLARED i32, so it could not see this.

    Measured 2026-09-05: a defect replay found `_COMMIT_DEFAULT_DTYPE` still set to "i8" AFTER both
    runtime engines had been changed to default i32 — and every existing test passed, because the
    in-tree pipeline always declares the attribute. An encoder silently holding an older opinion than
    the engine it mirrors refutes correct backends, which is the one outcome that makes this tool
    worse than nothing. These cases exercise the readout paths the pipeline never produces.

    The ABSENT case used to be here and has moved to
    `test_the_encoders_default_matches_the_engines_default` below: `validate_command_buffer` now
    refuses a narrowing command that declares no container, so an absent-dtype buffer cannot reach
    `simulate` and the end-to-end differential has nothing to compare against. The property still needs
    checking, so it is checked where it now lives — on the constant itself.
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

    _, cb, _ = _pair(m=4, k=64, n=4)          # K large enough that the accumulator leaves i8 range
    cb = copy.deepcopy(cb)
    mutate(cb)
    concrete = materialize_inputs(cb, None)
    golden = simulate(cb)["outputs"]

    blk = Block()
    with ImplicitBuilder(blk):
        enc = Encoder()
        outs, leaves = encode_command_buffer(enc, cb)
        pins, diffs = [], []
        for name, tensor in leaves.items():
            ref = concrete.get(name)
            if ref is None:
                continue
            rows, cols = ref.shape
            for r in range(rows):
                for c in range(cols):
                    lit = enc.const(int(ref.data[r * cols + c]), tensor.width)
                    pins.append(smt.EqOp(tensor.at(r, c), lit).results[0])
        for name, tensor in outs.items():
            want = golden[name]
            for r in range(tensor.rows):
                for c in range(tensor.cols):
                    lit = enc.const(int(want[r][c]), tensor.width)
                    diffs.append(smt.NotOp(smt.EqOp(tensor.at(r, c), lit).results[0]).results[0])
        assert pins and diffs, "vacuous"
        term = diffs[0]
        for d in diffs[1:]:
            term = smt.OrOp(term, d).results[0]
        for pin in pins:
            smt.AssertOp(pin)
        smt.AssertOp(term)
        smt.YieldOp()

    verdict = check_module(builtin.ModuleOp([SolverOp.from_region(Region([blk]))]),
                           timeout_ms=180_000)
    assert verdict.status == "unsat", (
        f"with {label} the encoder disagrees with merlin.runtime.simulate "
        f"(status={verdict.status}). The encoder must MIRROR the engine, whatever it says.")


def test_the_encoders_default_matches_the_engines_default():
    """The absent-dtype half of the differential above, which can no longer run end to end.

    `validate_command_buffer` refuses a narrowing command with no declared container, so a buffer that
    omits `output_dtype` raises before either engine reads it. That is the stronger contract, but it
    also removes the only path that used to compare the encoder's default against the engines' — and
    the defect this guards against (the encoder holding "i8" for weeks after the engines moved to i32)
    lived in exactly that constant. Asserted directly instead, against the engines' own source rather
    than a number repeated here.
    """
    import inspect

    from merlin.runtime import reference, simulator
    from merlin.verify.cb_semantics import _COMMIT_DEFAULT_DTYPE

    for module in (simulator, reference):
        src = inspect.getsource(module)
        assert f'attrs.get("output_dtype", "{_COMMIT_DEFAULT_DTYPE}")' in src, (
            f"the encoder defaults an absent output_dtype to {_COMMIT_DEFAULT_DTYPE!r} but "
            f"{module.__name__} does not; an encoder that disagrees with the engine it mirrors "
            f"refutes correct backends")


# -- spellings the buffer is allowed to use, and spellings nobody defined ----------------------

def test_an_op_spelled_combine_abstains_instead_of_silently_meaning_add():
    """`op: "identity"` must not be read as an ADDITION, which is what the default used to do.

    The crash this was found through (`KeyError: 'rhs'`) was the lucky outcome. `attrs.get("combine",
    "add")` answers "add" for a buffer that spells the combine as `op`, so a buffer carrying BOTH
    `op: "identity"` and an `rhs` would have encoded an addition, been agreed with by a reference that
    defaults the same way, and returned VERIFIED while the hardware moved data. Nothing in the schema
    defines `op` for VECTOR_MAP, so the only safe reading of an unexplained `op` is to refuse.
    """
    from merlin.verify.cb_semantics import CommandBufferEncoder
    from merlin.verify.smt_semantics import UnsupportedSemantics

    _, cb, _ = _pair()
    cb = copy.deepcopy(cb)
    src = next(t for t, s in cb["tensors"].items() if (s or {}).get("role") == "input")
    cb["commands"] = [{"opcode": "VECTOR_MAP",
                       # both operands present: the silent-add reading is REACHABLE here, so this
                       # test fails loudly if the default is ever restored
                       "operands": {"lhs": src, "rhs": src, "dst": "vm_out"},
                       "attributes": {"op": "identity"}}]
    e = CommandBufferEncoder(_null_encoder(), cb)
    e.declare_leaves()
    with pytest.raises(UnsupportedSemantics, match="op='identity'"):
        e.run()


def test_the_contract_sanctioned_src_spelling_is_accepted_for_movement():
    """`mlir_oot_backend_contract.yaml` defines the movement source as `src` OR `lhs`.

    Neither engine implemented the alternative, so a buffer written to the published contract crashed
    the encoder with a bare KeyError and was counted as OUR defect. Both spellings must reach the same
    encoding; the contract text is read here rather than restated, so this test fails if the contract
    stops sanctioning the alias.
    """
    from merlin.common.paths import merlin_dir
    from merlin.verify.cb_semantics import CommandBufferEncoder

    contract = (merlin_dir() / "contract" / "mlir_oot_backend_contract.yaml").read_text()
    assert "`src` (or `lhs`)" in contract, (
        "the contract no longer sanctions the src/lhs alias; this test is now asserting invention")

    _, cb, _ = _pair()
    base = copy.deepcopy(cb)
    src = next(t for t, s in base["tensors"].items() if (s or {}).get("role") == "input")

    outs = []
    for key in ("src", "lhs"):
        b = copy.deepcopy(base)
        b["commands"] = [{"opcode": "MOVEMENT",
                          "operands": {key: src, "dst": "mv_out"},
                          "attributes": {}}]
        e = CommandBufferEncoder(_null_encoder(), b)
        e.declare_leaves()
        outs.append(e.run())

    assert outs[0].keys() == outs[1].keys(), (
        "the two contract-sanctioned spellings of the movement source produced different outputs")


def test_comparing_a_narrow_output_against_a_wide_one_does_not_kill_the_query():
    """A `bv<8>` leaf vs a `bv<32>` clamped accumulator must be reconciled, not exported ill-formed.

    `saturate` clamps at the accumulator width and KEEPS it, so an identity path and a contraction
    path reach the same declared output at different bitvector widths. Upstream `smt.eq` carries
    `SameTypeOperands`, so emitting that comparison produced a module `mlir-translate` rejects — and
    the failure took the WHOLE query with it, returning no verdict at all rather than an abstention.
    Sign-extending the narrower side is sound because the wider one was already clamped into range.
    """
    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects import builtin, smt
    from xdsl.ir import Block, Region

    from merlin.verify.smt_export import to_smtlib
    from merlin.verify.smt_ops import SolverOp
    from merlin.verify.smt_semantics import Encoder

    blk = Block()
    with ImplicitBuilder(blk):
        enc = Encoder()
        narrow = enc.symbolic_tensor("narrow", 2, 2, 8)
        wide = enc.saturate(enc.symbolic_tensor("wide", 2, 2, 32), -128, 127, 32)
        assert (narrow.width, wide.width) == (8, 32), "the widths under test must actually differ"
        smt.AssertOp(enc.any_differs(narrow, wide))
        smt.YieldOp()

    # exporting is the assertion: a width mismatch used to fail here, taking the verdict with it
    text = to_smtlib(builtin.ModuleOp([SolverOp.from_region(Region([blk]))]))
    assert text.strip(), "export produced nothing"


def test_an_instruction_level_buffer_abstains_instead_of_verifying_falsely():
    """The worst verdict in the vocabulary, reproduced: a false VERIFIED.

    Found by adjudicating a contradiction, not by a seeded fault. A submission wrote its buffer at RoCC
    instruction level -- the semantic opcode reused across commands, the real step carried in an `op`
    attribute (`config_ex`/`config_ld`/`mvin`/`preload`/`compute_preloaded`/`config_st`/`mvout`). The
    encoder read none of them, so three RES_PACKs collapsed to one and two COMMITs to one, and the
    collapsed program happened to be exactly what the specification asked for. It reported VERIFIED,
    while on hardware every output saturated at 127 (INT8_MAX) against a golden expecting i32.

    Ignoring an attribute is not neutral: it makes a DIFFERENT program, and here the different program
    was the correct one. Anything the encoder cannot model must abstain.
    """
    from merlin.verify.cb_semantics import CommandBufferEncoder
    from merlin.verify.smt_semantics import UnsupportedSemantics

    _, cb, _ = _pair()
    cb = copy.deepcopy(cb)
    pack = next(c for c in cb["commands"] if c["opcode"] == "RES_PACK")
    # one duplicated command carrying a config step -- the shape the real submission had
    pack["attributes"]["op"] = "config_ex"

    e = CommandBufferEncoder(_null_encoder(), cb)
    e.declare_leaves()
    with pytest.raises(UnsupportedSemantics, match="config_ex"):
        e.run()


def test_the_opcodes_that_legitimately_carry_op_are_not_refused():
    """VREDUCE selects its reduction with `op`; refusing it would abstain on everything it can encode.

    The guard above must discriminate, not blanket-refuse -- an over-broad abstention is a coverage
    loss disguised as caution, and would have quietly emptied the reduction family.
    """
    from merlin.verify.cb_semantics import CommandBufferEncoder

    e = CommandBufferEncoder(_null_encoder(), {"tensors": {}, "commands": []})
    for opcode in ("VREDUCE", "VECTOR_MAP"):
        e._refuse_unmodelled_step(0, opcode, {"op": "sum"})      # must not raise
