"""What the hand-written matmul recipe emits today, pinned byte-exactly, so it can be replaced safely.

WHY THIS EXISTS. `sched/AGENT.md` directs that the hand-written LOOP_WS recipe and its legality rules be
replaced by recipes composed from `merlin.sched.primitives`. That replacement is only safe if the
current output is pinned first: a rewrite of a schedule generator has no natural falsifier, and "the
tests still pass" is not one when the tests exercise the generator through a simulator that costs
minutes per shape and has its own error band. Here the falsifier is exact, free and immediate --
`Kernel.digest()` is the sha256 of the kernel's canonical text, so two schedules are the same schedule
exactly when they digest the same. A composed recipe that reproduces these digests is not merely
"close to" the hand-written one, it IS it, and the 42 golden cells and the measured 0.950x carry over
without re-running anything.

WHAT THE MATRIX COVERS, which is the part that matters. The recipe's structure is chosen by the number
of k-tiles, and all three regimes are represented -- checked below rather than asserted here, so this
comment cannot drift from the shapes:

  1 call   k_tiles == 1   the whole reduction in one macro-instruction; no peeling
  2 calls  k_tiles == 2   first and last peeled, no middle loop between them
  3 calls  k_tiles >= 3   first peeled, a middle loop, last peeled

plus partial tiles in every dimension (pad_I/pad_J/pad_K non-zero), a k of 1, an m of 1, bias and no
bias, and relu and no relu. A pin that only covered the common regime would let a rewrite break the
peeling and still look green, and the peeling is the part a composition has to reconstruct.

THIS FILE IS NOT A SPECIFICATION. It records what the current generator does, including anything about
it that is wrong. If a digest changes because the recipe was deliberately improved, update the pin in
the same commit as the change and say why in the message -- do not relax the assertion.
"""

from __future__ import annotations

import pytest

from merlin.runtime.backends import base as _bk
from merlin.sched.ir import TensorArg

pytestmark = pytest.mark.target("gemmini")

#: ``(m, n, k, relu, bias) -> digest``. Generated from the tree at the commit that added this file.
#:
#: RE-PINNED when the two branches were reconciled. `matmul_reference` gained the full-width
#: accumulator readout on the other line of development, and with it two annotations the emitter does
#: not read: the recipe name now distinguishes the full-width readout from the narrow one, and a
#: `readout` attr says which port a schedule leaves through. `Kernel.digest` is the digest of the
#: canonical text, annotations included, so all twelve moved.
#:
#: This is the case `test_an_annotation_the_emitter_does_not_read_moves_only_the_ir_pin` describes,
#: and it was CHECKED rather than assumed: `PINNED_C` below did not move for any of the twelve, so
#: the program the machine runs is byte-for-byte what it was and only the IR's description changed.
#:
#: RE-PINNED 2026-09-21 because the recipe's `readout` ATTRIBUTE changed spelling, and for no other
#: reason. It read "narrow, requantized"; the attribute separator in the canonical text is ", ", so
#: that value cut its own attribute in half and the kernel could not be parsed back -- which matters
#: because a kernel's identity IS its text. The value is now spelled without the comma and `text()`
#: refuses any key or value carrying the separator, so this cannot recur.
#:
#: This is again the case `test_an_annotation_the_emitter_does_not_read_moves_only_the_ir_pin`
#: describes, and again it was CHECKED rather than assumed: all twelve entries of `PINNED_C` below are
#: unmoved, so the program the machine runs is byte-for-byte what it was and only the IR's description
#: of it changed. The 42 golden cells and the measured 0.950x carry over untouched.
PINNED: dict[tuple[int, int, int, bool, bool], str] = {
    (16, 16, 16, False, False): "bdc72c577cc8f2b545f735353842c29e3c6c9d1d678ca4d7771340cb4461d8c5",
    (64, 64, 64, False, False): "bc385af995d08bc17d8243c8acb0f01b386527a5a49663c4aaafeafdceb53b52",
    (100, 100, 100, False, False): "538ef72256a0d129a98f2a87a481a78d2af2189e93d5f1c3cf9da0ffe4867f2d",
    (128, 512, 512, True, False): "08ef74cca674c101fc725c820706731d1adb3f558485454774a9980738f3d5d2",
    (512, 512, 4096, True, True): "e6fde56d30b85de1ad37ad72b69cb7042e02b53314643225a20ed464d1ad9a68",
    (1000, 1024, 1, False, True): "6bd03b3f0c334481d748fa48fbdf9a29a4bcfa9c5f692d9545b02469d3bbaf3b",
    (1, 1000, 2048, False, True): "02cd360220252c374ff59535812c43b180d76f5d6e52f5a247ad4128bcc9096c",
    (3136, 64, 64, False, True): "090a216d7c5e95b507d2d011789701447fbc982c771e0d0f0a4f704f07ef9d97",
    (3136, 64, 576, False, True): "309413c9579bcfd87105e37680c718cd4fd809b88f088fb798febfbf7b330a19",
    (196, 256, 1152, False, True): "30c7b0352d277301c6ef7a671184f2a998c560d8c1d46e24adcfb1e04d6ed8c7",
    (196, 512, 256, True, True): "f8c5e37bc2c30609a3434d20f1a76d06964053a8b333468b9ff7e4ddbe9f7064",
    (49, 512, 2304, False, True): "d81a8aaeaaddbd847c5b470ec8301776cf56f9d406c4ccc972faedd3328d2092",
}


#: ``(m, n, k, relu, bias) -> sha256 of the emitted C``. A SECOND pin, and the one that is about
#: behaviour rather than identity.
#:
#: `PINNED` above is the digest of the kernel's canonical text, which is the right falsifier for a
#: rewrite that must change nothing at all. It is the wrong one for a change that is deliberately
#: visible in the IR: the IR carries annotations the emitter does not read -- `sets`, `assumes`,
#: `stages`, unit and token names -- so declaring a schedule's configuration dependences moves every
#: digest here while the machine runs byte-identical code. Without this pin the only way to make such
#: a change is to re-take `PINNED` on trust, which is exactly the move that turns a pin into a record
#: of whatever the generator last did.
#:
#: These bytes are what the 42 golden cells and the measured 0.950x belong to. A change that keeps them
#: keeps the measurement; a change that moves them does not, whatever the IR says.
PINNED_C: dict[tuple[int, int, int, bool, bool], str] = {
    (16, 16, 16, False, False): "e555ce40fe5293ce27a86f63616bb7a0207124f7c1d13f9184e33081cf24a67b",
    (64, 64, 64, False, False): "b761aea1d7151d0eec862fdc9c0327842829195539c8cf44d0c7df8572d7c26b",
    (100, 100, 100, False, False): "58a057c3c130a5f5b78932235f11976272e378e86b21499dca2cde2c7c39debf",
    (128, 512, 512, True, False): "1cf2430e3b670389c009a347e21fe9a7c174021edb5f1fe0b7d4ed0231b5685f",
    (512, 512, 4096, True, True): "9065fb0fd0a3e6798a2969e77d1f003cc6e6fb68f78792ff0abc78105e249877",
    (1000, 1024, 1, False, True): "1ae3c3761327f831510d77241a8ce744ed5aaac43f4086124cac5c67266bd1ce",
    (1, 1000, 2048, False, True): "e1ad8e1c9c3e80dcc0435a5a36be848ba0f9751b1b4b898d520d54088ebcf2aa",
    (3136, 64, 64, False, True): "ecdd32bbef26f2cbe08c26746b1c739413be718644e9dbbe62a263c874d75231",
    (3136, 64, 576, False, True): "b26ba31086c770c65beac4f79d70f3e1aa851900d9c5dafa9ad546f7cc9a57b2",
    (196, 256, 1152, False, True): "804cfee34012137919917dcc8a5725e532cfb212bcc7d2d1d5ead94684fa4a96",
    (196, 512, 256, True, True): "8d224636faa90c8f7ce201602d50b06488f69166b4cbf4b433b7a93a8d846dd9",
    (49, 512, 2304, False, True): "c621f2857e0a1cbf96dfa769f6f793308a671cf9c19ef8911c47a2d5497ba456",
}


def _kernel(m: int, n: int, k: int, relu: bool, bias: bool):
    operands = {
        "a": TensorArg("A", (m, k), "i8", "read"),
        "b": TensorArg("B", (k, n), "i8", "read"),
        "c": TensorArg("C", (m, n), "i8", "write"),
    }
    if bias:
        operands["d"] = TensorArg("D", (n,), "i32", "read")
    backend = _bk.get_backend("gemmini")
    return backend.sched_matmul_reference(name="mm", m=m, n=n, k=k, operands=operands, relu=relu, scale=1.0)


@pytest.mark.parametrize("shape", sorted(PINNED), ids=lambda s: "x".join(map(str, s[:3])))
def test_the_recipe_emits_exactly_what_it_emitted(shape):
    """The pin. A composed recipe replaces the hand-written one only by reproducing these."""
    assert _kernel(*shape).digest() == PINNED[shape], (
        f"the schedule for {shape[:3]} changed. If that was deliberate, re-pin it in the same commit "
        "as the change and say why; if it was not, the rewrite is not behaviour-preserving."
    )


@pytest.mark.parametrize("shape", sorted(PINNED_C), ids=lambda s: "x".join(map(str, s[:3])))
def test_the_recipe_emits_exactly_the_c_it_emitted(shape):
    """What the machine runs, pinned separately from what the IR says.

    This is the assertion that survives a deliberate change to the schedule's IR. If a commit moves
    `PINNED` and leaves this alone, the change was to the IR's description of the schedule; if it moves
    this too, the change was to the program, and the golden cells and the measured ratio have to be
    re-established rather than inherited.
    """
    import hashlib

    from merlin.sched.codegen import emit_c_function

    c = emit_c_function(_kernel(*shape), _bk.get_backend("gemmini").sched_instruction_set(), symbol="mm")
    assert hashlib.sha256(c.encode()).hexdigest() == PINNED_C[shape], (
        f"the C emitted for {shape[:3]} changed. This is the program, not its description: the 42 golden "
        "cells and the measured 0.950x were taken on the old bytes and do not carry over."
    )


def test_the_two_pins_cover_the_same_shapes():
    """A shape pinned in one and not the other is a shape where half the falsifier is missing."""
    assert set(PINNED) == set(PINNED_C)


def test_the_c_pin_is_sensitive_to_the_program():
    """The mutation. Without it the C pin would pass for an emitter that returns a constant."""
    import hashlib

    from merlin.sched.codegen import emit_c_function

    iset = _bk.get_backend("gemmini").sched_instruction_set()
    a = emit_c_function(_kernel(64, 64, 64, False, False), iset, symbol="mm")
    b = emit_c_function(_kernel(64, 64, 64, True, False), iset, symbol="mm")
    assert hashlib.sha256(a.encode()).hexdigest() != hashlib.sha256(b.encode()).hexdigest()


def test_an_annotation_the_emitter_does_not_read_moves_only_the_ir_pin():
    """Why the two pins are separate, demonstrated rather than asserted.

    Declaring a call's configuration dependence changes the kernel's text -- and therefore its digest --
    while the emitted C is byte-identical, because `emit_c_function` renders operands through the
    target's macro and never reads `sets` or `assumes`. A single pin cannot tell that change apart from
    one that alters the program.
    """
    import hashlib

    from merlin.sched.codegen import emit_c_function
    from merlin.sched.ir import Kernel, call

    iset = _bk.get_backend("gemmini").sched_instruction_set()
    plain = _kernel(64, 64, 64, False, False)
    old = plain.body[0]
    head = call(old.instr, sets={"ex_A_transpose": 0}, **dict(old.args))
    annotated = Kernel(name=plain.name, args=plain.args, body=(head,) + plain.body[1:], attrs=plain.attrs)
    assert annotated.digest() != plain.digest(), "the annotation did not reach the IR's identity"
    assert (
        hashlib.sha256(emit_c_function(annotated, iset, symbol="mm").encode()).hexdigest()
        == hashlib.sha256(emit_c_function(plain, iset, symbol="mm").encode()).hexdigest()
    ), "an annotation the emitter does not read changed the emitted C"


def test_all_three_peeling_regimes_are_represented():
    """The pin is only worth as much as its coverage, and the peeling is what a composition must
    reconstruct: the k-loop's first iteration loads the bias and its last stores the result, so a
    rewrite that got the peel wrong would still pass a matrix of single-call shapes."""
    counts = {_kernel(*shape).text().count("loop_ws(") for shape in PINNED}
    assert {1, 2, 3} <= counts, f"the matrix misses a peeling regime; call counts seen: {sorted(counts)}"


def test_partial_tiles_are_represented():
    """A shape whose dimensions are all multiples of the block has every pad at zero, which is the case
    an off-by-one in the padding cannot fail on."""
    padded = [
        s
        for s in PINNED
        if any("pad_I=0, pad_J=0, pad_K=0" not in ln for ln in _kernel(*s).text().splitlines() if "loop_ws(" in ln)
    ]
    assert padded, "no shape in the matrix produces a non-zero pad"


def test_the_digest_is_sensitive_to_the_schedule():
    """The mutation. Without it every assertion above would pass for a digest that ignores the body."""
    a = _kernel(64, 64, 64, False, False)
    b = _kernel(64, 64, 64, True, False)
    assert a.digest() != b.digest(), "the digest does not distinguish two different schedules"


def test_every_pinned_shape_is_legal_on_the_target_isa():
    """The pin proves the composed recipe emits the same TEXT; this proves the text is legal.

    Byte-identity already implies it, which is exactly why it is worth asserting separately: if the
    composition is ever changed deliberately and the pin re-taken, this is the check that still has an
    opinion. It runs the target's own declared semantics over every dynamic instance -- the operand kind
    rules, the per-instruction checks, the configuration cross-checks, the partial-sum liveness rule,
    and the end-of-kernel `finish` hook that catches an accumulator never stored.
    """
    from merlin.sched.check.static import check_kernel

    iset = _bk.get_backend("gemmini").sched_instruction_set()
    for shape in sorted(PINNED):
        errors = check_kernel(_kernel(*shape), iset)
        assert not errors, f"{shape[:3]} is not legal on the target ISA: {errors[:3]}"
