"""Each scheduling primitive does what it claims, and refuses what it cannot justify.

Two tests per primitive, not one. The legality half is cheap and would pass for a primitive that never
refuses anything; the refusal half is where a proof obligation either exists or is decoration. A
primitive declaring ``bounded_check`` whose check returns nothing unconditionally is the failure mode
this file is written against.
"""

from __future__ import annotations

import pytest

from merlin.sched.ir import Kernel, Ptr, Stage, TensorArg, call, loop
from merlin.sched.ir.expr import Var, add, as_expr, mul, select
from merlin.sched.ir.kernel import check_structure
from merlin.sched.primitives import (
    PROOFS,
    Cursor,
    NotApplicable,
    PrimitiveError,
    _instance_keys,
    divide_loop,
    fuse,
    multi_buffer,
    peel,
    proof_of,
    reorder,
    replace,
    specialize,
    unroll,
)
from merlin.sched.primitives import __all__ as PRIMITIVE_EXPORTS

PRIMITIVES = (divide_loop, reorder, unroll, fuse, peel)

#: The tensor argument the peeling tests below share; the older tests in this file spell it inline.
ARG = (TensorArg("a", (64,), "i8", "read"),)


def _kernel() -> Kernel:
    """A perfect 8x4 nest whose operand is a linear function of both variables, so a mis-indexed
    rewrite changes WHICH bytes are touched rather than merely when."""
    return Kernel(
        "k",
        (TensorArg("a", (64,), "i8", "read"),),
        (loop("i", 8, loop("j", 4, call("op", src=Ptr("a", add(mul(as_expr(Var("i")), 4), Var("j"))), n=as_expr(1)))),),
    )


def _touched(kernel: Kernel) -> list[tuple]:
    """Every dynamic instance as (instruction, concrete operand values) -- what a loop transformation
    must preserve. Order is deliberately excluded: changing it is the point of a schedule."""
    from merlin.sched.primitives import _instance_keys

    return sorted(_instance_keys(kernel))


# -- the structural contract -------------------------------------------------------------------------


def test_the_primitive_set_is_reflected_not_listed():
    """Anti-vacuity. Every parametrized test below iterates a set; an empty one passes them all."""
    exported = {n for n in PRIMITIVE_EXPORTS if n[0].islower() and n not in {"proof_of"}}
    assert len(exported) >= 4, f"only {exported} are exported; the suite would assert almost nothing"


@pytest.mark.parametrize("primitive", PRIMITIVES, ids=lambda p: p.__name__)
def test_every_primitive_declares_an_obligation_in_the_closed_vocabulary(primitive):
    assert proof_of(primitive) in PROOFS


def test_a_primitive_declaring_no_obligation_is_refused():
    """An unclassified primitive must not reach the register as though it had been reasoned about."""

    def undeclared(kernel):  # pragma: no cover - never called
        return kernel

    with pytest.raises(PrimitiveError, match="declares no proof obligation"):
        proof_of(undeclared)


# -- divide_loop -------------------------------------------------------------------------------------


def test_divide_loop_preserves_every_instance_and_moves_the_digest():
    before = _kernel()
    after = divide_loop(before, Cursor(loops=("i",)), 4)
    assert _touched(after) == _touched(before), "splitting a loop changed which bytes are touched"
    assert after.digest() != before.digest(), "the rewrite did not change the schedule's identity"
    assert "for io in 0..2:" in after.text() and "for ii in 0..4:" in after.text()


@pytest.mark.parametrize(
    "factor,expect",
    [(3, "does not divide"), (1, "splits nothing"), (0, "splits nothing")],
)
def test_divide_loop_refuses_a_factor_that_would_need_a_tail(factor, expect):
    """A tail invented here would change what the kernel computes, and would do it silently."""
    with pytest.raises(NotApplicable, match=expect):
        divide_loop(_kernel(), Cursor(loops=("i",)), factor)


def test_divide_loop_refuses_to_shadow_an_existing_variable():
    with pytest.raises(PrimitiveError, match="already a loop variable"):
        divide_loop(_kernel(), Cursor(loops=("i",)), 4, names=("j", "ii"))


# -- unroll ------------------------------------------------------------------------------------------


def test_unroll_preserves_every_instance():
    before = _kernel()
    after = unroll(before, Cursor(loops=("i", "j")))
    assert _touched(after) == _touched(before)
    assert "for j" not in after.text(), "the unrolled loop is still there"
    assert after.text().count("op(") == 4, "the inner loop's 4 iterations were not all emitted"


def test_unroll_refuses_an_extent_that_is_a_tiling_decision():
    big = Kernel("k", (TensorArg("a", (8,), "i8", "read"),), (loop("i", 99999, call("op", n=as_expr(1))),))
    with pytest.raises(NotApplicable, match="tiling decision"):
        unroll(big, Cursor(loops=("i",)))


# -- reorder -----------------------------------------------------------------------------------------


def test_reorder_swaps_the_nest_and_preserves_every_instance():
    before = _kernel()
    after = reorder(before, Cursor(loops=("i",)))
    assert _touched(after) == _touched(before), "swapping loops changed which bytes are touched"
    first_loops = [line.strip() for line in after.text().splitlines() if "for " in line]
    assert first_loops[0].startswith("for j"), f"the nest was not swapped: {first_loops}"


def test_reorder_refuses_an_imperfect_nest():
    """A statement beside the inner loop has no defined position after a swap; fission it first."""
    imperfect = Kernel(
        "k",
        (TensorArg("a", (64,), "i8", "read"),),
        (loop("i", 8, loop("j", 4, call("op", n=as_expr(1))), call("other", n=as_expr(1))),),
    )
    with pytest.raises(NotApplicable, match="not perfect"):
        reorder(imperfect, Cursor(loops=("i",)))


def test_the_instance_check_can_actually_fail():
    """The mutation that proves ``bounded_check`` is not decoration.

    Every refusal above is a precondition the primitive tests BEFORE rewriting. This exercises the
    obligation itself: hand it two kernels that differ in which instances run and require it to refuse.
    A check that returned nothing unconditionally passes every other test in this file.
    """
    from merlin.sched.primitives import _require_same_instances

    before = _kernel()
    dropped = Kernel(before.name, before.args, (loop("i", 4, loop("j", 4, call("op", n=as_expr(1)))),))
    with pytest.raises(PrimitiveError, match="changed which instances run"):
        _require_same_instances(before, dropped, what="a rewrite that dropped half the work")


# -- fuse --------------------------------------------------------------------------------------------


def test_fuse_joins_two_loops_of_one_extent_and_preserves_every_instance():
    before = Kernel(
        "k",
        (TensorArg("a", (64,), "i8", "read"),),
        (
            loop("i", 4, call("op", src=Ptr("a", Var("i")), n=as_expr(1))),
            loop("j", 4, call("op2", src=Ptr("a", Var("j")), n=as_expr(1))),
        ),
    )
    after = fuse(before, Cursor(index=0))
    assert _touched(after) == _touched(before)
    assert len([line for line in after.text().splitlines() if "for " in line]) == 1


def test_fuse_refuses_differing_extents():
    before = Kernel(
        "k",
        (TensorArg("a", (64,), "i8", "read"),),
        (loop("i", 4, call("op", n=as_expr(1))), loop("j", 8, call("op2", n=as_expr(1)))),
    )
    with pytest.raises(NotApplicable, match="extents differ"):
        fuse(before, Cursor(index=0))


# -- cursors -----------------------------------------------------------------------------------------


def test_a_cursor_naming_a_position_that_does_not_exist_is_not_applicable():
    """A search may try the next move."""
    with pytest.raises(NotApplicable):
        divide_loop(_kernel(), Cursor(loops=("nope",)), 2)


def test_a_cursor_asserting_the_wrong_instruction_is_a_definite_failure():
    """Not a miss: the caller reasoned about a different statement, so continuing the search with the
    next move would carry that mistaken reasoning forward."""
    with pytest.raises(PrimitiveError, match="reasoned about a different statement"):
        fuse(_kernel(), Cursor(index=0, instr="not_the_instruction_here"))


def test_primitives_compose_and_each_step_preserves_the_instances():
    """The property that makes a search possible at all: after any sequence of moves, the kernel still
    computes what it computed, and its identity has moved."""
    k0 = _kernel()
    k1 = divide_loop(k0, Cursor(loops=("i",)), 4)
    k2 = reorder(k1, Cursor(loops=("io",)))
    assert _touched(k2) == _touched(k0)
    assert len({k0.digest(), k1.digest(), k2.digest()}) == 3


# -- peeling, and the field-preservation bug it surfaced ---------------------------------------------


def test_peel_front_shifts_the_remaining_loop():
    """`for k in 0..N` becomes `B(0)` then `for k in 0..N-1` of `B(k + 1)` -- the shape a reduction
    needs when its first iteration initialises an accumulator."""
    k = Kernel("k", ARG, (loop("k0", 7, call("ws", K=Var("k0"))),))
    out = peel(k, Cursor(loops=("k0",)), at="front")
    lines = [line.strip() for line in out.text().splitlines()[1:]]
    assert lines == ["ws(K=0)", "for k0 in 0..6:", "ws(K=(k0 + 1))"], lines


def test_peel_back_takes_the_last_iteration():
    k = Kernel("k", ARG, (loop("k0", 7, call("ws", K=Var("k0"))),))
    out = peel(k, Cursor(loops=("k0",)), at="back")
    lines = [line.strip() for line in out.text().splitlines()[1:]]
    assert lines == ["for k0 in 0..6:", "ws(K=k0)", "ws(K=6)"], lines


def test_peeling_both_ends_leaves_the_middle():
    """The composition the matmul recipe needs: first, a middle loop, last."""
    k = Kernel("k", ARG, (loop("k0", 7, call("ws", K=Var("k0"))),))
    out = peel(peel(k, Cursor(loops=("k0",)), at="front"), Cursor(loops=("k0",)), at="back")
    lines = [line.strip() for line in out.text().splitlines()[1:]]
    assert lines == ["ws(K=0)", "for k0 in 0..5:", "ws(K=(k0 + 1))", "ws(K=6)"], lines


def test_a_substitution_that_makes_a_subtree_constant_folds_it():
    """THE digest test. `(5 + 1)` and `6` are the same number and a different canonical text, so a
    peel that left the sum unfolded would produce a kernel that is equal in meaning and unequal in
    identity -- which is the only kind of equality this IR has."""
    k = Kernel("k", ARG, (loop("k0", 7, call("ws", K=Var("k0") + 1)),))
    out = peel(k, Cursor(loops=("k0",)), at="back")
    assert out.text().rstrip().endswith("ws(K=7)"), out.text()


def test_peeling_collapses_a_select_on_the_peeled_variable():
    """A last-tile remainder is written as a select on the loop variable, and peeling that iteration
    must DECIDE it rather than carry a select whose subject no longer varies.

    The remaining loop loses the select too, and that is correct rather than over-eager: it ran 0..3,
    the peel took iteration 3, so no iteration left can satisfy `i == 3`. Leaving the dead select would
    cost nothing at run time and change the kernel's canonical text, which is its identity."""
    k = Kernel("k", ARG, (loop("i", 4, call("ws", I=select(Var("i"), 3, 2, 8))),))
    out = peel(k, Cursor(loops=("i",)), at="back")
    lines = [line.strip() for line in out.text().splitlines()[1:]]
    assert lines == ["for i in 0..3:", "ws(I=8)", "ws(I=2)"], lines


def test_a_select_the_remaining_loop_can_still_reach_is_kept():
    """The control for the pruning above. Peeling the FRONT off the same loop leaves iterations that do
    reach the tested value, so the select must survive -- shifted, since the loop was renumbered."""
    k = Kernel("k", ARG, (loop("i", 4, call("ws", I=select(Var("i"), 3, 2, 8))),))
    out = peel(k, Cursor(loops=("i",)), at="front")
    lines = [line.strip() for line in out.text().splitlines()[1:]]
    assert lines == ["ws(I=8)", "for i in 0..3:", "ws(I=select(i == 2, 2, 8))"], lines


def test_peel_refuses_a_loop_with_nothing_left_behind():
    k = Kernel("k", ARG, (loop("i", 1, call("ws")),))
    with pytest.raises(NotApplicable, match="nothing behind"):
        peel(k, Cursor(loops=("i",)))


def test_peel_refuses_a_direction_it_does_not_have():
    k = Kernel("k", ARG, (loop("i", 4, call("ws")),))
    with pytest.raises(PrimitiveError, match="'front' or 'back'"):
        peel(k, Cursor(loops=("i",)), at="middle")


@pytest.mark.parametrize("at", ["front", "back"])
def test_peel_preserves_the_instances(at):
    """Correct by construction only if it is: the peeled copy plus the remaining loop must run exactly
    the iterations the original did."""
    k = Kernel("k", ARG, (loop("i", 5, call("ws", I=Var("i") * 3)),))
    before = sorted(_instance_keys(k))
    assert sorted(_instance_keys(peel(k, Cursor(loops=("i",)), at=at))) == before


@pytest.mark.parametrize("prim", ["divide_loop", "unroll", "peel"])
def test_a_loop_transformation_keeps_the_unit_token_staging_and_configuration(prim):
    """Measured bug: three primitives rebuilt a call positionally and dropped every field but the
    operands, so tiling a placed, synchronised, configured kernel returned an unplaced, unsynchronised,
    unconfigured one. `_require_same_instances` cannot see it -- it compares instruction and operand
    values only, by design, since order is what a transformation changes."""
    # No completion token here: `unroll` and `peel` DUPLICATE the body, so a produced token would
    # legitimately be refused as two static producers of one token. That refusal has its own test
    # below; this one is about fields surviving a rewrite, and the token is covered by divide_loop,
    # which nests rather than duplicates.
    k = Kernel(
        "k",
        ARG,
        (loop("i", 4, call("mm", unit="mesh", stages=(Stage("spad", 0, 8),), sets={"s": 1})),),
    )
    cursor = Cursor(loops=("i",))
    out = {
        "divide_loop": lambda: divide_loop(k, cursor, 2),
        "unroll": lambda: unroll(k, cursor),
        "peel": lambda: peel(k, cursor),
    }[prim]()
    text = out.text()
    for kept in ("on mesh", "in spad[0:8]", "sets s=1"):
        assert kept in text, f"{prim} dropped {kept!r}:\n{text}"


def test_divide_loop_keeps_a_completion_token_because_it_does_not_duplicate():
    """The token half of the check above, on the one primitive that can legitimately carry it."""
    k = Kernel("k", ARG, (loop("i", 4, call("mm", unit="mesh", produces="t0")),))
    assert "t0 = " in divide_loop(k, Cursor(loops=("i",)), 2).text()


@pytest.mark.parametrize("prim", ["unroll", "peel"])
def test_duplicating_a_body_that_produces_a_token_is_refused(prim):
    """Measured: both returned a kernel their own structural check rejects, without a word.

    `_require_same_instances` cannot catch it -- it compares instruction and operand values only, by
    design, because order is exactly what a transformation changes. So a defect confined to the token
    passed straight through."""
    k = Kernel("k", ARG, (loop("i", 3, call("mm", x=Var("i"), produces="t0")),))
    cursor = Cursor(loops=("i",))
    with pytest.raises(NotApplicable, match="already a live token"):
        {"unroll": lambda: unroll(k, cursor), "peel": lambda: peel(k, cursor)}[prim]()


@pytest.mark.parametrize("prim", ["specialize", "peel"])
def test_two_sibling_loops_of_the_same_name_make_a_cursor_ambiguous(prim):
    """Measured, and silent both times. Two siblings may legally share a name -- `check_structure` only
    rejects shadowing of an ENCLOSING name -- and the read path took the first match while the write
    path rewrote EVERY match. `specialize` on the first of two `for i` loops rewrote both, destroying
    the second's operands; `peel` replaced the second loop with a copy of the first."""
    k = Kernel(
        "k",
        ARG,
        (loop("i", 4, call("mm", x=Var("i"), acc=0)), loop("i", 4, call("mm", x=Var("i"), acc=1))),
    )
    assert check_structure(k) == [], "the premise: two sibling loops of one name are well formed"
    with pytest.raises(PrimitiveError, match="sibling loops at this level are named"):
        if prim == "specialize":
            specialize(k, Cursor(loops=("i",), index=0, instr="mm"), acc=99)
        else:
            peel(k, Cursor(loops=("i",)))


# -- replace: a macro instruction is a loop nest ------------------------------------------------------


def _nest(outer: str = "i", inner: str = "j"):
    """The nest a macro is declared equal to: a 4x2 tile loop moving one operand and computing on it."""
    return loop(
        outer,
        4,
        loop(
            inner,
            2,
            call("mvin", dst=add(mul(Var(outer), 2), Var(inner))),
            call("compute", k=Var(inner)),
        ),
    )


def _host(nest) -> Kernel:
    return Kernel(name="m", args=ARG, body=(call("config", mode=1), nest, call("fence")))


MACRO = call("loop_ws", rows=4, cols=2)


def test_replace_substitutes_the_declared_nest():
    out = replace(_host(_nest()), Cursor(index=1), expansion=_nest(), call=MACRO)
    assert [s.instr for s in out.body] == ["config", "loop_ws", "fence"]
    assert not check_structure(out)


def test_replace_matches_up_to_loop_variable_renaming():
    """The nest a schedule arrives at was named by whatever produced it -- `divide_loop` mints its own
    names -- while a macro's declared expansion is written once. Requiring the two to coincide would
    make the substitution depend on a spelling neither side controls."""
    out = replace(_host(_nest("p", "q")), Cursor(index=1), expansion=_nest("i", "j"), call=MACRO)
    assert [s.instr for s in out.body] == ["config", "loop_ws", "fence"]


@pytest.mark.parametrize(
    "actual,why",
    [
        (_nest_wrong_extent := loop("i", 8, loop("j", 2, call("mvin", dst=Var("j")))), "extent"),
        (loop("i", 4, loop("j", 2, call("mvin", dst=Var("j")))), "statement"),
        (
            loop("i", 4, loop("j", 2, call("mvin", dst=Var("i")), call("compute", k=Var("j")))),
            "where the expansion has",
        ),
    ],
    ids=["extent differs", "body is shorter", "an operand differs"],
)
def test_replace_refuses_a_nest_that_is_not_the_declared_one(actual, why):
    """THE FALSIFIER, and the failure this primitive actually guards against. Whether the hardware macro
    computes what the nest computes is a claim about silicon that nothing here can discharge; matching
    the WRONG nest is a compiler bug at this level, it is the likelier of the two, and it is silent --
    the kernel stays well-formed and the digest simply becomes that of a different schedule.
    """
    with pytest.raises(NotApplicable, match=why):
        replace(_host(actual), Cursor(index=1), expansion=_nest(), call=MACRO)


def test_replace_refuses_a_renaming_that_is_not_a_bijection():
    """Two of the expansion's loops landing on one of the kernel's would accept a nest that collapses
    loops the macro keeps apart -- a different computation, matched as if it were the same."""
    collapsed = loop("i", 4, loop("i2", 2, call("mvin", dst=Var("i")), call("compute", k=Var("i"))))
    with pytest.raises(NotApplicable):
        replace(_host(collapsed), Cursor(index=1), expansion=_nest(), call=MACRO)


def test_replace_refuses_an_empty_expansion():
    """An expansion of nothing matches anywhere, which would make the check vacuous rather than lenient."""
    with pytest.raises(PrimitiveError, match="empty one matches anything"):
        replace(_host(_nest()), Cursor(index=1), expansion=(), call=MACRO)


def test_replace_refuses_when_the_expansion_runs_off_the_end():
    kernel = Kernel(name="m", args=ARG, body=(_nest(),))
    with pytest.raises(NotApplicable, match="remain at this position"):
        replace(kernel, Cursor(index=0), expansion=(_nest(), call("fence")), call=MACRO)


def test_replace_consumes_exactly_the_statements_the_expansion_names():
    """A multi-statement expansion substitutes all of them and nothing else -- an off-by-one here would
    silently swallow a neighbouring fence, which is a correctness instruction on these targets."""
    kernel = Kernel(name="m", args=ARG, body=(call("config", mode=1), _nest(), call("fence"), call("drain")))
    out = replace(kernel, Cursor(index=1), expansion=(_nest(), call("fence")), call=MACRO)
    assert [s.instr for s in out.body] == ["config", "loop_ws", "drain"]


# -- multi_buffer: a staging has a depth --------------------------------------------------------------


def _Machine(rows: int = 64):
    """A real Machine, not a stub.

    A hand-rolled double is what a check reads when nobody looked: the first version of this omitted
    `arbiter` and `shared_by`, the bank-starvation pass reached for them, and three tests failed on the
    double rather than on the subject. The model refuses to be built wrong, which is the point of it.
    """
    from merlin.sched.mach.model import Machine, Memory

    return Machine(
        target="a_machine",
        hazard_resolution="interlocked",
        memories=(Memory(name="spad", rows=rows, row_bytes=16, banks=1),),
    )


def _staged(*stages) -> Kernel:
    return Kernel(name="m", args=ARG, body=(call("mvin", dst=as_expr(0), stages=stages),))


def test_multi_buffer_reserves_the_copies():
    out = multi_buffer(_staged(Stage("spad", 0, 8)), Cursor(index=0), memory="spad", depth=3)
    assert out.body[0].stages[0].depth == 3
    assert "spad[0:8]x3" in out.text(), out.text()


def test_a_single_buffer_still_prints_as_it_always_did():
    """The compatibility property the whole field rests on. `depth` prints only when it is more than
    one, so every schedule written before it existed keeps its text and therefore its digest -- which is
    what let this land without re-pinning twelve recipe digests and the C they emit."""
    assert Stage("spad", 0, 8).depth == 1
    assert _staged(Stage("spad", 0, 8)).text().rstrip().endswith("in spad[0:8]")


def test_multi_buffer_refuses_copies_that_do_not_fit():
    """THE FALSIFIER. Enlarging a reservation is exactly the operation that runs off the end of a
    store, and the failure is silent -- the schedule stays well-formed and the device overwrites."""
    with pytest.raises(NotApplicable, match="rows of spad, which has 64"):
        multi_buffer(_staged(Stage("spad", 0, 8)), Cursor(index=0), memory="spad", depth=9, machine=_Machine())


def test_multi_buffer_refuses_copies_that_land_on_a_neighbour():
    """A second staging on the same call sits where the later copies would go. Without depth in the
    overlap arithmetic this reads as two disjoint eight-row buffers, which is how a result gets written
    over an operand still being read."""
    kernel = _staged(Stage("spad", 0, 8), Stage("spad", 8, 8, writes=True))
    with pytest.raises(NotApplicable, match="over a staging this call already holds"):
        multi_buffer(kernel, Cursor(index=0), memory="spad", row=0, depth=2)


@pytest.mark.parametrize(
    "kwargs,exc,why",
    [
        (dict(memory="spad", depth=1), PrimitiveError, "single buffer"),
        (dict(memory="acc", depth=2), NotApplicable, "stages nothing in"),
    ],
    ids=["depth of one", "no staging there"],
)
def test_multi_buffer_refuses_a_request_it_cannot_honour(kwargs, exc, why):
    with pytest.raises(exc, match=why):
        multi_buffer(_staged(Stage("spad", 0, 8)), Cursor(index=0), **kwargs)


def test_multi_buffer_refuses_an_ambiguous_staging():
    """Two stagings in one memory on one call is the COMMON case -- both operands of a matmul live in
    the scratchpad -- so this refuses the guess and not the situation: name the row and it proceeds."""
    kernel = _staged(Stage("spad", 0, 4), Stage("spad", 32, 4))
    with pytest.raises(PrimitiveError, match=r"at rows \[0, 32\]. Pass row="):
        multi_buffer(kernel, Cursor(index=0), memory="spad", depth=2)
    # ...and naming the row resolves it, which is the common case: both operands of a matmul live in
    # one scratchpad, so refusing outright would make the primitive unusable on the shape it is for.
    assert multi_buffer(kernel, Cursor(index=0), memory="spad", row=32, depth=2).body[0].stages[1].depth == 2


def test_the_capacity_check_prices_the_copies():
    """The half that matters: a reservation nothing prices is a comment. A double-buffered eight-row
    staging occupies sixteen, and a store of twelve cannot hold it."""
    from merlin.sched.check.placement import check_placement

    kernel = _staged(Stage("spad", 0, 8, depth=2))
    report = check_placement(kernel, _Machine(rows=12))
    assert not report.ok
    assert any("x2 copies" in p and "runs past" in p for p in report.problems), report.problems
    assert check_placement(kernel, _Machine(rows=16)).ok
