"""Which scheduling primitives can change one byte of emitted code -- MEASURED, one primitive at a time.

WHY THIS FILE EXISTS. A vocabulary whose moves reach no emitter gives a search nothing to search: it can
apply a transformation, pass every legality check, and emit the identical program. Nothing else in the
suite can see that. The IR digest moves (the annotation is in the text), the placement and cost checks
read the annotation and have opinions about it, and the C pin passes because the C did not change -- and
"the C did not change" is the defect, not the evidence against it. So each primitive is applied here and
the emitted C is compared BEFORE against AFTER. A move classified as reaching emission must differ; one
classified annotation-only must not, and must say so where a reader will find it rather than in a
docstring nobody diffs.

The classification is data (:data:`REACH`) and is checked against the live export list, so a primitive
added to the vocabulary without a verdict fails here rather than joining it unmeasured. ``UNKNOWN`` is a
real value: a move nobody has exercised is not thereby annotation-only, and collapsing the two is how a
map like this becomes decoration.
"""

from __future__ import annotations

import dataclasses

import pytest

from merlin.sched.codegen import UNRENDERABLE, emit_c_function
from merlin.sched.ir import Call, Kernel, Ptr, Stage, TensorArg, call, loop
from merlin.sched.ir.expr import Var, add, as_expr, evaluate, mul
from merlin.sched.isa import InstrDef, InstructionSet, IsaError, Operand
from merlin.sched.mach import Latency, Machine, Memory, Unit
from merlin.sched.primitives import (
    Cursor,
    NotApplicable,
    PrimitiveError,
    bind_bank,
    divide_loop,
    fuse,
    hoist_config,
    multi_buffer,
    peel,
    pipeline,
    place,
    reorder,
    replace,
    specialize,
    stage_mem,
    unroll,
)
from merlin.sched.primitives import __all__ as PRIMITIVE_EXPORTS

# -- the verdict map ---------------------------------------------------------------------------------

#: Whether each primitive's effect can reach emitted code, and the line that carries or fails to carry
#: it. ``reaches`` means a test below shows the C differing; ``annotation-only`` means a test below
#: shows it byte-identical; ``unknown`` means nobody has exercised it and the entry says what it would
#: take. Every entry is exercised by a test in this file, so the map cannot drift from the code.
REACH: dict[str, tuple[str, str]] = {
    # Rewrites of the loop nest itself. `codegen/__init__.py` walks `Loop` and emits a counted `for`,
    # so any change to the nest is a change to the program text.
    "divide_loop": ("reaches", "sched/codegen/__init__.py:128 emits one `for` per Loop node"),
    "unroll": ("reaches", "sched/codegen/__init__.py:128 -- the Loop is gone, so the `for` is gone"),
    "reorder": ("reaches", "sched/codegen/__init__.py:128 -- the two `for` headers swap"),
    "fuse": ("reaches", "sched/codegen/__init__.py:128 -- two `for` headers become one"),
    "peel": ("reaches", "sched/codegen/__init__.py:128 -- a copy outside a shortened `for`"),
    # Rewrites of the statement list or of an operand. `render_c` is handed the operand texts, so an
    # operand rewrite reaches C; a statement inserted, moved or substituted reaches it structurally.
    "specialize": ("reaches", "sched/codegen/__init__.py:144 renders every operand through render_c"),
    "replace": ("reaches", "sched/codegen/__init__.py:144 -- the nest's statements become one call"),
    "hoist_config": ("reaches", "sched/codegen/__init__.py:128 -- the call moves outside the `for`"),
    "stage_mem": ("reaches", "sched/codegen/__init__.py:144 -- but only via the move it inserts"),
    # Annotations. `Call.unit`, `.produces`, `.awaits` and `.stages` are not operands, and the operand
    # list is the emitter's only channel to a target's C. The first two are now REFUSED rather than
    # dropped; with a speller they reach C, which is what `_Spells` below shows.
    "place": ("annotation-only", "sched/codegen/__init__.py:144 renders s.args; Call.unit is not one"),
    "pipeline": ("annotation-only", "sched/codegen/__init__.py:144; Call.produces/.awaits are not operands"),
    "bind_bank": ("annotation-only", "sched/codegen/__init__.py:144; Call.stages is not an operand"),
    "multi_buffer": ("annotation-only", "sched/ir/kernel.py:93 Stage.depth; reaches C only with rotate="),
}


def test_the_map_has_a_verdict_for_every_exported_primitive():
    """A move added to the vocabulary without a verdict fails here rather than joining it unmeasured.

    The failure this guards is not hypothetical: thirteen primitives were exported and five of them
    reached no emitter, and the only reason anyone found out was an audit. A map that has to be
    updated by hand would have the same property.
    """
    import merlin.sched.primitives as prims

    exported = {n for n in PRIMITIVE_EXPORTS if getattr(getattr(prims, n, None), "PROOF", None)}
    assert set(REACH) == exported, (
        f"unclassified: {sorted(exported - set(REACH))}; classified but not exported: {sorted(set(REACH) - exported)}"
    )
    assert set(v for v, _ in REACH.values()) <= {"reaches", "annotation-only", "unknown"}


def test_no_verdict_is_a_bare_assertion():
    """Every entry cites a line of a real module -- the one that carries, or fails to carry, the effect.

    The cited file is resolved; the line number is not pinned, because pinning it would make an
    unrelated edit above it fail this suite and teach the next reader to delete the citation.
    """
    from merlin.common.paths import merlin_dir

    for name, (_, where) in REACH.items():
        path, _, rest = where.partition(":")
        assert rest[:1].isdigit(), f"{name}: verdict cites no line"
        assert (merlin_dir() / "python" / "merlin" / path).is_file(), f"{name}: cites missing {path}"


# -- a toy target ------------------------------------------------------------------------------------

ARG = (TensorArg("a", (4096,), "i8", "read"),)


def _iset() -> InstructionSet:
    """Four instructions and nothing target-specific: a move, a multiply, a macro and a config.

    ``mvin`` takes its on-chip destination as an ordinary ``int`` operand, which is the fact the
    rotation argument exists for: nothing in the IR distinguishes an on-chip address from a stride or
    a tile count, so the caller has to name it.
    """
    return InstructionSet(
        target="toy",
        instrs={
            "mvin": InstrDef(
                "mvin",
                (Operand("dram", "ptr"), Operand("spad", "int"), Operand("rows", "int")),
                render_c=lambda a: f"toy_mvin({', '.join(a)});",
            ),
            "mm": InstrDef(
                "mm",
                (Operand("a", "int"), Operand("b", "int"), Operand("c", "int")),
                render_c=lambda a: f"toy_mm({', '.join(a)});",
            ),
            "macro": InstrDef("macro", (Operand("n", "int"),), render_c=lambda a: f"toy_macro({a[0]});"),
            "cfg": InstrDef("cfg", (), render_c=lambda a: "toy_cfg();"),
        },
        c_types={"i8": "int8_t"},
        drain_c="toy_drain();",
    )


ISET = _iset()

MACHINE = Machine(
    target="a_machine",
    hazard_resolution="explicit",
    units=(Unit(name="dma", kind="dma", queue="q"), Unit(name="mesh", kind="systolic", queue="q")),
    memories=(Memory(name="spad", rows=64, row_bytes=16, banks=2),),
    latencies=(
        Latency(instr="mvin", unit="dma", issue=1, result=33, completion="polled", source="test"),
        Latency(instr="mm", unit="mesh", issue=1, result=None, completion="immediate", source="test"),
    ),
)


class _DeclaredFree:
    """A speller that says, for every annotation, that it costs no code on this target.

    Not the same thing as an emitter that ignores it: the target SAID so, and a reader can find the
    line that says it. That is the whole difference this file is about.
    """

    def issue_on(self, c: Call) -> str:
        return ""

    def on_await(self, c: Call) -> str:
        return ""

    def on_produce(self, c: Call) -> str:
        return ""


class _Spells:
    """A speller that renders the annotations as real statements."""

    def issue_on(self, c: Call) -> str:
        return f"toy_select_unit({c.unit});"

    def on_await(self, c: Call) -> str:
        return "".join(f"toy_wait({t});" for t in c.awaits)

    def on_produce(self, c: Call) -> str:
        return f"toy_started({c.produces});"


FREE = _DeclaredFree()


def _c(kernel: Kernel, spell=FREE) -> str:
    return emit_c_function(kernel, ISET, symbol="k", spell=spell)


# -- the moves, one per verdict ----------------------------------------------------------------------


def _mvin_loop(extent: int = 8) -> Kernel:
    return Kernel(
        "k",
        ARG,
        (
            loop(
                "k",
                extent,
                call("mvin", dram=Ptr("a", mul(Var("k"), 16)), spad=as_expr(0), rows=as_expr(16)),
            ),
        ),
    )


def _nest() -> Kernel:
    inner = call("mm", a=add(mul(Var("i"), 2), Var("j")), b=as_expr(0), c=as_expr(0))
    return Kernel("k", ARG, (loop("i", 4, loop("j", 2, inner)),))


def _two_loops() -> Kernel:
    return Kernel(
        "k",
        ARG,
        (
            loop("i", 4, call("mm", a=Var("i"), b=as_expr(0), c=as_expr(0))),
            loop("j", 4, call("mm", a=Var("j"), b=as_expr(1), c=as_expr(0))),
        ),
    )


def _one_call() -> Kernel:
    return Kernel("k", ARG, (call("mm", a=as_expr(0), b=as_expr(0), c=as_expr(0)),))


def _two_calls() -> Kernel:
    return Kernel(
        "k",
        ARG,
        (
            call("mvin", dram=Ptr("a", as_expr(0)), spad=as_expr(0), rows=as_expr(8), unit="dma"),
            call("mm", a=as_expr(0), b=as_expr(0), c=as_expr(0), unit="mesh"),
        ),
    )


def _configured_loop() -> Kernel:
    return Kernel(
        "k",
        ARG,
        (loop("i", 4, call("cfg", sets={"mode": 1}), call("mm", a=Var("i"), b=as_expr(0), c=as_expr(0))),),
    )


def _staged_in_a_rotation(depth_loop: int = 2) -> Kernel:
    """``for ko { for ki { mvin(..., spad=0, rows=8) } }``, the shape a rotation is written over.

    The inner loop's extent IS the depth, which is how this IR writes ``iteration mod depth``: splitting
    the rotation loop by the depth makes the inner variable the phase, exactly.
    """
    inner = call(
        "mvin",
        dram=Ptr("a", add(mul(Var("ko"), 16), mul(Var("ki"), 8))),
        spad=as_expr(0),
        rows=as_expr(8),
        stages=(Stage("spad", 0, 8),),
    )
    return Kernel("k", ARG, (loop("ko", 4, loop("ki", depth_loop, inner)),))


ROTATION_CURSOR = Cursor(loops=("ko", "ki"), index=0)

#: ``name -> (before, apply)``. One move each, applied to the smallest kernel that admits it.
MOVES = {
    "divide_loop": (_mvin_loop, lambda k: divide_loop(k, Cursor(loops=("k",)), 2)),
    "unroll": (lambda: _mvin_loop(3), lambda k: unroll(k, Cursor(loops=("k",)))),
    "reorder": (_nest, lambda k: reorder(k, Cursor(loops=("i",)))),
    "fuse": (_two_loops, lambda k: fuse(k, Cursor(index=0))),
    "peel": (lambda: _mvin_loop(4), lambda k: peel(k, Cursor(loops=("k",)))),
    "specialize": (_one_call, lambda k: specialize(k, Cursor(index=0), b=as_expr(7))),
    "replace": (
        _nest,
        lambda k: replace(k, Cursor(index=0), expansion=_nest().body, call=call("macro", n=as_expr(8))),
    ),
    "hoist_config": (_configured_loop, lambda k: hoist_config(k, Cursor(loops=("i",), index=0))),
    "stage_mem": (
        _one_call,
        lambda k: stage_mem(
            k,
            Cursor(index=0),
            move=call("mvin", dram=Ptr("a", as_expr(0)), spad=as_expr(0), rows=as_expr(8)),
            memory="spad",
            row=0,
            rows=8,
            machine=MACHINE,
        ),
    ),
    "place": (_one_call, lambda k: place(k, Cursor(index=0), unit="mesh", machine=MACHINE)),
    "pipeline": (
        _two_calls,
        lambda k: pipeline(k, Cursor(index=0), Cursor(index=1), token="t0", machine=MACHINE),
    ),
    "bind_bank": (
        _one_call,
        lambda k: bind_bank(k, Cursor(index=0), memory="spad", row=0, rows=8, bank=1, machine=MACHINE),
    ),
    "multi_buffer": (
        _staged_in_a_rotation,
        lambda k: multi_buffer(k, ROTATION_CURSOR, memory="spad", depth=2, machine=MACHINE),
    ),
}


def test_every_classified_primitive_is_actually_exercised():
    """A verdict with no move behind it is an opinion. This is what makes the map evidence."""
    assert set(MOVES) == set(REACH)


@pytest.mark.parametrize("name", sorted(n for n, (v, _) in REACH.items() if v == "reaches"))
def test_a_reaching_primitive_changes_the_emitted_c(name):
    """THE MUTATION, for the nine moves that do reach the program.

    Each is applied and the emitted C compared. Without this the whole layer could be replaced by a
    function that returns its argument and every other test in the suite would still pass.
    """
    before, apply = MOVES[name]
    k = before()
    out = apply(k)
    assert out.digest() != k.digest(), f"{name} did not change the schedule at all"
    assert _c(out) != _c(k), f"{name} is classified as reaching emission, and the emitted C is identical"


@pytest.mark.parametrize("name", sorted(n for n, (v, _) in REACH.items() if v == "annotation-only"))
def test_an_annotation_only_primitive_leaves_the_emitted_c_byte_identical(name):
    """THE SEAM, pinned rather than described.

    These four change the schedule's identity and cannot change its program. That is a fact about the
    emitter's interface -- ``InstrDef.render_c`` takes the operand texts and nothing else -- so it is
    pinned here as the baseline the fix has to move. When a primitive stops being annotation-only, this
    test fails and its row moves to ``reaches``: that is the intended way to find out.
    """
    before, apply = MOVES[name]
    k = before()
    out = apply(k)
    assert out.digest() != k.digest(), f"{name} did not change the schedule at all"
    assert _c(out) == _c(k), f"{name} is classified annotation-only and the emitted C moved"


def test_stage_mem_reaches_the_c_only_through_the_statement_it_inserts():
    """The half-and-half case, stated rather than rounded off.

    `stage_mem` inserts the caller's move ahead of the consumer AND records a staging on both. The move
    is a statement, so it emits; the staging is an annotation, so it does not. Classifying the whole
    primitive as reaching would overstate it, so this says exactly which half does.
    """
    before, apply = MOVES["stage_mem"]
    k = before()
    out = apply(k)
    assert "toy_mvin(" in _c(out) and "toy_mvin(" not in _c(k)
    stripped = dataclasses.replace(out, body=tuple(dataclasses.replace(s, stages=()) for s in out.body))
    assert _c(stripped) == _c(out), "removing the stagings changed the C, so a staging does reach it"
    assert stripped.digest() != out.digest(), "removing the stagings did not change the schedule"


# -- what the emitter refuses to drop ----------------------------------------------------------------


def test_the_emitter_refuses_an_annotation_nothing_can_render():
    """Fail closed. Without a speller, a placed or pipelined kernel is not emitted as if it were
    neither: the emitted bytes would be identical to the unannotated kernel's, and no measurement
    downstream could tell that apart from a transformation that did not help."""
    placed = MOVES["place"][1](_one_call())
    with pytest.raises(IsaError, match="byte-identical"):
        emit_c_function(placed, ISET, symbol="k")
    piped = MOVES["pipeline"][1](_two_calls())
    with pytest.raises(IsaError, match="is placed on unit"):
        emit_c_function(piped, ISET, symbol="k")


def test_an_unannotated_kernel_still_needs_no_speller():
    """The compatibility property every recorded C digest in the repo rests on: a kernel that uses none
    of these emits exactly what it always did, with no speller and no bump to EMITTER_VERSION."""
    assert emit_c_function(_mvin_loop(), ISET, symbol="k") == _c(_mvin_loop())
    assert "toy_mvin(((uint8_t *)a + (k * 16)), 0, 16);" in emit_c_function(_mvin_loop(), ISET, symbol="k")


def test_a_speller_makes_placement_and_synchronisation_reach_the_c():
    """The other half of the refusal: the annotation is renderable, and when a target says how, it
    lands in the program -- before the call for a wait, after it for a completion token."""
    piped = MOVES["pipeline"][1](_two_calls())
    src = emit_c_function(piped, ISET, symbol="k", spell=_Spells())
    assert "toy_select_unit(dma);" in src and "toy_started(t0);" in src
    assert src.index("toy_wait(t0);") < src.index("toy_mm("), "the wait must precede the call that waits"
    assert src.index("toy_mvin(") < src.index("toy_started(t0);"), "a token is produced by the call"
    assert src != emit_c_function(_two_calls(), ISET, symbol="k", spell=_Spells())


def test_a_speller_that_returns_a_non_statement_is_refused():
    placed = MOVES["place"][1](_one_call())

    class _Bad:
        def issue_on(self, c):
            return 7

    with pytest.raises(IsaError, match="not a C statement"):
        emit_c_function(placed, ISET, symbol="k", spell=_Bad())


def test_every_non_operand_field_of_a_call_is_either_spellable_or_declared_not_to_be():
    """A field added to ``Call`` later cannot join it unclassified.

    ``instr`` and ``args`` are what ``render_c`` receives. Everything else is an annotation, and each
    one is either in ``UNRENDERABLE`` (the emitter asks a speller for it) or named here with the reason
    it is not. A new field belongs to one list or the other, and this is what says so.
    """
    carried_by_render_c = {"instr", "args"}
    not_unrenderable = {
        # A staging says WHERE an operand sits while a call runs; the address itself is already an
        # operand, which is the channel that works. What makes a staging reach C is a rewrite of that
        # operand (`multi_buffer(rotate=...)`), not a hook in the emitter.
        "stages",
        # Configuration is established and consumed by CALLS -- the call that sets it is itself emitted
        # -- so the key/value pair needs no separate statement. `check/config.py` is what reads it.
        "sets",
        "assumes",
    }
    fields = {f.name for f in dataclasses.fields(Call)}
    assert fields - carried_by_render_c == {f for f, _, _ in UNRENDERABLE} | not_unrenderable, (
        f"a Call field is neither an operand, nor spellable, nor declared unspellable: "
        f"{sorted(fields - carried_by_render_c - {f for f, _, _ in UNRENDERABLE} - not_unrenderable)}"
    )


# -- multi_buffer: the reservation, and the rotation that reaches the program ------------------------


def test_the_reservation_alone_emits_the_same_program():
    """The defect, pinned at its narrowest. Reserving two copies changes the schedule's identity, the
    capacity it needs and its price, and does not change one byte of what runs -- so a search that
    'applied multi-buffering' and measured no difference measured the truth."""
    k = _staged_in_a_rotation()
    out = multi_buffer(k, ROTATION_CURSOR, memory="spad", depth=2, machine=MACHINE)
    assert out.body[0].body[0].body[0].stages[0].depth == 2
    assert "spad[0:8]x2" in out.text()
    assert _c(out) == _c(k)


def test_the_rotation_changes_the_emitted_c():
    """THE MUTATION this whole exercise is for. Same reservation, plus the operand rewrite that says
    which copy an iteration addresses -- and now the emitted C differs."""
    k = _staged_in_a_rotation()
    reserved = multi_buffer(k, ROTATION_CURSOR, memory="spad", depth=2, machine=MACHINE)
    rotated = multi_buffer(
        k, ROTATION_CURSOR, memory="spad", depth=2, machine=MACHINE, rotate="spad", over="ki", advance=8
    )
    assert _c(rotated) != _c(reserved), "the rotation reached no emitted code"
    assert "toy_mvin(((uint8_t *)a + ((ko * 16) + (ki * 8))), (ki * 8), 8);" in _c(rotated)
    # The reservation is not lost to the rewrite: the copies still have to fit, and the check that
    # prices them still reads the same field.
    assert rotated.body[0].body[0].body[0].stages[0].depth == 2


def test_the_rotation_addresses_a_different_copy_each_iteration():
    """Not merely 'the text moved'. The operand is evaluated at each phase and lands on a distinct copy,
    inside the reservation and nowhere else -- which is the property a reader would otherwise have to
    take from the rendered string."""
    rotated = multi_buffer(
        _staged_in_a_rotation(),
        ROTATION_CURSOR,
        memory="spad",
        depth=2,
        machine=MACHINE,
        rotate="spad",
        over="ki",
        advance=8,
    )
    inner = rotated.body[0].body[0].body[0]
    seen = [evaluate(inner.arg("spad"), {"ko": 1, "ki": phase}) for phase in (0, 1)]
    assert seen == [0, 8], seen
    stage = inner.stages[0]
    assert all(stage.row <= addr < stage.row + stage.rows * stage.depth for addr in seen)


@pytest.mark.parametrize(
    "kwargs,exc,why",
    [
        (dict(rotate="spad"), PrimitiveError, "together"),
        (dict(rotate="spad", over="ki"), PrimitiveError, "together"),
        (dict(rotate="nope", over="ki", advance=8), PrimitiveError, "has no operand"),
        (dict(rotate="dram", over="ki", advance=8), PrimitiveError, "not an index expression"),
        (dict(rotate="spad", over="ki", advance=0), PrimitiveError, "single buffer written as a rotation"),
        (dict(rotate="spad", over="ko", advance=8), NotApplicable, "not the phase"),
        (dict(rotate="spad", over="zz", advance=8), NotApplicable, "does not enclose this call"),
    ],
    ids=["no over", "no advance", "unknown operand", "a dram pointer", "zero advance", "wrong extent", "out of scope"],
)
def test_the_rotation_refuses_what_would_emit_a_plausible_wrong_address(kwargs, exc, why):
    """THE REFUSAL HALF. Every one of these leaves a schedule that is well formed, fits the store, and
    reads the same copy every iteration -- the defect one level down, wearing a rotation's clothes."""
    with pytest.raises(exc, match=why):
        multi_buffer(_staged_in_a_rotation(), ROTATION_CURSOR, memory="spad", depth=2, machine=MACHINE, **kwargs)


def test_a_rotation_over_a_loop_the_caller_split_is_the_modulus():
    """How a rotation is reached from a plain loop, end to end, with no modulus in the IR.

    `divide_loop(k, depth)` gives ``k = ko*depth + ki``, so ``ki`` IS ``k mod depth`` -- exactly, not
    approximately. A depth of two is the parity toggle one target spells by hand and the slot ping-pong
    another does; neither idiom is in the vocabulary, both are instances of this.
    """
    flat = Kernel(
        "k",
        ARG,
        (
            loop(
                "k",
                8,
                call(
                    "mvin",
                    dram=Ptr("a", mul(Var("k"), 8)),
                    spad=as_expr(0),
                    rows=as_expr(8),
                    stages=(Stage("spad", 0, 8),),
                ),
            ),
        ),
    )
    split = divide_loop(flat, Cursor(loops=("k",)), 2, names=("ko", "ki"))
    rotated = multi_buffer(
        split,
        Cursor(loops=("ko", "ki"), index=0),
        memory="spad",
        depth=2,
        machine=MACHINE,
        rotate="spad",
        over="ki",
        advance=8,
    )
    assert _c(rotated) != _c(split)
    inner = rotated.body[0].body[0].body[0]
    # Every original iteration k maps to the copy k mod 2, which is what a modulus would have said.
    for k in range(8):
        assert evaluate(inner.arg("spad"), {"ko": k // 2, "ki": k % 2}) == (k % 2) * 8
