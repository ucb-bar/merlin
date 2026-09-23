"""Reading a kernel back from its canonical text, and the guard that makes that safe.

WHY THIS FILE EXISTS. A kernel's identity IS its text -- the digest is the sha256 of what
``Kernel.text`` prints -- so a parser that is merely mostly right is worse than no parser at all. It
would return a kernel that looks like the one written down and carries a DIFFERENT identity, and
everything keyed on that digest downstream (the layer-bench key, a numerics contract, a recorded
certificate) would be attributed to a schedule nobody wrote. Nothing would raise.

So ``parse_kernel`` re-renders what it built and refuses on any difference, and the tests below are
organised around that guard rather than around the grammar: the round-trip is asserted on real
schedules and on a kernel that uses every construct, and then the guard is shown to FIRE -- once on
input a lenient parser would accept, and once against a planted bug in the parser itself. A guard that
cannot fire would make every other test here vacuous, because they would all pass on a parser that
returned the input's own kernel by magic.

The last test is about where this is going rather than what it does: the module has to stay free of
merlin imports, because the reason to have a parser at all is that a composed schedule can travel --
into a minted package as data, or beside a vendored copy of the rewrite vocabulary, neither of which
may import the harness they are graded against.
"""

from __future__ import annotations

import pytest

from merlin.sched.ir import (
    NULL,
    Kernel,
    ParseError,
    Ptr,
    Stage,
    TensorArg,
    call,
    loop,
    parse_expr,
    parse_kernel,
)
from merlin.sched.ir.expr import Const, Scaled, Select, Sum, Var, add, mul, render, select

I, J = Var("i"), Var("j")


def _everything() -> Kernel:
    """One kernel exercising every construct the text form can carry.

    Deliberately hand-built rather than taken from a target: the real recipe below is the evidence
    that this works on schedules we actually emit, but it uses none of the asynchrony, none of the
    staging, no float operand and no attribute -- so on its own it would leave most of the grammar
    unexercised while reading as full coverage.
    """
    return Kernel(
        name="everything",
        args=(TensorArg("A", (64, 32), "i8", "read"), TensorArg("C", (64,), "i32", "write")),
        attrs=(("origin", "handmade"), ("note", "a=b")),
        body=(
            loop(
                "i",
                4,
                loop(
                    "j",
                    3,
                    call(
                        "dma",
                        dst=Ptr("A", add(mul(I, 32), J)),
                        n=Const(-7),
                        produces="t0",
                        unit="mover",
                        stages=(Stage("spad", 0, 16, bank=2, writes=True), Stage("acc", 4, 8)),
                        sets={"stride": 64, "mode": NULL},
                        assumes={"dataflow": 1},
                    ),
                    call(
                        "mm",
                        x=select(I, 3, add(J, 1), mul(J, 4)),
                        y=1.5,
                        z=NULL,
                        w=float("-inf"),
                        awaits=("t0",),
                        unit="mesh",
                    ),
                ),
            ),
            call("drain"),
        ),
    )


# -- the round trip ------------------------------------------------------------------------------


def test_a_kernel_using_every_construct_reads_back_identical():
    """Structural equality, not just digest equality.

    The digest is what callers key on, but two kernels can agree on it only because both printed the
    same -- asserting the dataclasses are equal is the stronger claim, and it is the one that catches
    a value recovered as the right TEXT but the wrong TYPE (an int where a float was written).
    """
    kernel = _everything()
    back = parse_kernel(kernel.text())
    assert back == kernel
    assert back.digest() == kernel.digest()


@pytest.mark.parametrize(
    "value",
    [
        "null",
        "0x1.8000000000000p+0",
        "-0x1.8000000000000p+0",
        "inf",
        "-inf",
        "nan",
        "&A[0]",
        "&A[((i * 32) + j)]",
        "-7",
        "0",
        "i",
        "(i + 1)",
        "(i * 4)",
        "((i * 32) + j)",
        "select(i == 3, (j + 1), (j * 4))",
        "select(i == 0, select(j == 1, 2, 3), 4)",
    ],
    ids=lambda v: v,
)
def test_every_operand_form_survives_a_round_trip(value):
    """Each value form printed into a kernel and read back out.

    Parametrised over the FORMS rather than asserted in one kernel so a form that stops round-tripping
    names itself, instead of one opaque failure on a kernel carrying sixteen of them.
    """
    text = f"kernel k(A: i8[64,32] read)\n  op(v={value})\n"
    assert parse_kernel(text).text() == text


@pytest.mark.parametrize(
    "expr",
    [
        Const(0),
        Const(-7),
        Var("i"),
        Sum((Var("i"), Const(1))),
        Scaled(Var("i"), 4),
        Sum((Scaled(Var("i"), 32), Var("j"))),
        Scaled(Sum((Var("i"), Var("j"))), 3),
        Select("i", 3, Sum((Var("j"), Const(1))), Scaled(Var("j"), 4)),
    ],
    ids=lambda e: render(e),
)
def test_every_expression_form_survives_a_round_trip(expr):
    assert parse_expr(render(expr)) == expr


def test_a_kernel_with_no_body_and_no_arguments_is_still_a_kernel():
    """The degenerate end. A parser that assumes at least one statement or one argument would take
    these as malformed, and an empty schedule is a legal thing for a search to produce."""
    text = "kernel empty()\n"
    assert parse_kernel(text).text() == text


# -- the guard fires -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,because",
    [
        ("kernel k(A: i8[4] read)\n  op(x=1)\n", None),
        ("kernel k(A: i8[4] read)\n  op(x=1)", "a missing trailing newline is tolerated"),
    ],
)
def test_canonical_text_is_accepted(text, because):
    assert parse_kernel(text).text() == "kernel k(A: i8[4] read)\n  op(x=1)\n"


@pytest.mark.parametrize(
    "text",
    [
        "kernel k(A: i8[4]  read)\n  op(x=1)\n",
        "kernel k( A: i8[4] read)\n  op(x=1)\n",
        "kernel k(A: i8[4] read)\n  op(x = 1)\n",
        "kernel k(A: i8[4] read)\n  op(x=1) \n",
        "kernel k(A: i8[4] read)\n\n  op(x=1)\n",
    ],
    ids=["double space", "leading space", "spaced operand", "trailing space", "blank line"],
)
def test_text_that_would_not_print_back_the_same_is_refused(text):
    """The guard, on input a lenient parser would happily accept.

    Every one of these denotes exactly the kernel the canonical form denotes, so a parser without the
    self-check returns something perfectly usable -- with a digest that does not match the text it came
    from. That is the silent mis-attribution this refuses.
    """
    with pytest.raises(ParseError, match="different digest"):
        parse_kernel(text)


def test_the_self_check_catches_a_planted_parser_bug(monkeypatch):
    """The mutation. Without this, every round-trip test above would also pass on a parser that
    cheated, and the guard itself would be untested.

    A dropped bank is the right thing to plant: it is invisible in the structure a caller inspects,
    it changes where a buffer lands on a banked memory, and it is exactly the class of detail a
    hand-written parser silently loses.
    """
    from merlin.sched.ir import parse as parse_module

    kernel = Kernel(
        name="k",
        args=(TensorArg("A", (4,), "i8", "read"),),
        body=(call("op", x=1, stages=(Stage("spad", 0, 8, bank=2),)),),
    )
    text = kernel.text()
    assert parse_kernel(text) == kernel, "the unpatched parser must handle this, or the mutation proves nothing"

    real = parse_module._parse_stage
    monkeypatch.setattr(parse_module, "_parse_stage", lambda s: _drop_bank(real(s)))
    with pytest.raises(ParseError, match="different digest"):
        parse_kernel(text)


def _drop_bank(stage: Stage) -> Stage:
    return Stage(stage.memory, stage.row, stage.rows, None, stage.writes)


# -- malformed input is named, not guessed at ------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("", "empty"),
        ("op(x=1)\n", "starts with"),
        ("kernel k(A: i8[4] read\n", "unclosed"),
        ("kernel k(A: i8[4] read)\n   op(x=1)\n", "multiple of two"),
        ("kernel k(A: i8[4] read)\n      op(x=1)\n", "unexpected indentation"),
        ("kernel k(A: i8[4] read)\n  for i in 0..4\n", "ends with ':'"),
        ("kernel k(A: i8[4] read)\n  for i in 4:\n", "for <var> in"),
        ("kernel k(A: i8[4] read)\n  op(x=1) wat\n", "unexpected text"),
        ("kernel k(A: i8[4] read)\n  op(x=(1))\n", "sum or a scaled"),
        ("kernel k(A: i8[4] read)\n  op(x=1))\n", "unbalanced"),
        ("kernel k(A: i8[4] read)\n  op(x=&A[0))\n", "mismatched"),
        ("kernel k(A: i8[4] read)\n  op(x=@)\n", "not an integer"),
        ("kernel k(A: i8[4] read)\n  op(1)\n", "key=value"),
        ("kernel k(A read)\n  op(x=1)\n", "dtype"),
        ("kernel k(A: i8[4] read)\n  a = b = op(x=1)\n", "at most one token"),
        ("kernel k(A: i8[4] read)\n  op(x=1) in spad[8:0]\n", "ends before it starts"),
    ],
    ids=[
        "empty",
        "no header",
        "unclosed args",
        "odd indent",
        "over indent",
        "loop without colon",
        "loop without range",
        "trailing junk",
        "empty parens",
        "extra close bracket",
        "mismatched bracket",
        "bad int",
        "bare operand",
        "no dtype",
        "two tokens",
        "backwards span",
    ],
)
def test_malformed_text_is_refused_with_a_reason(text, expected):
    """FAIL CLOSED, and say which part. A parser that raised a bare ValueError would make every one of
    these look like the same problem to a caller trying to fix a generated schedule."""
    with pytest.raises(ParseError, match=expected):
        parse_kernel(text)


# -- the real thing ------------------------------------------------------------------------------


SHAPES = [
    (1, 1, 1),
    (16, 16, 16),
    (64, 64, 64),
    (100, 100, 100),
    (128, 512, 512),
    (512, 512, 4096),
    (1000, 1024, 1),
    (1, 1000, 2048),
    (3136, 64, 64),
    (3136, 64, 576),
    (196, 256, 1152),
    (196, 512, 256),
    (49, 512, 2304),
]


@pytest.mark.target("gemmini")
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_a_real_schedule_reads_back_identical(shape):
    """The evidence that this works on schedules the tree actually emits, over the same ResNet-50
    shapes the layer bench measures -- which is where the remainder `select`s, the multi-term pointer
    arithmetic and the configuration chains come from."""
    from merlin.runtime.backends import base as backends

    m, n, k = shape
    operands = {
        "a": TensorArg("A", (m, k), "i8", "read"),
        "b": TensorArg("B", (k, n), "i8", "read"),
        "c": TensorArg("C", (m, n), "i8", "write"),
        "d": TensorArg("D", (n,), "i32", "read"),
    }
    kernel = backends.get_backend("gemmini").sched_matmul_reference(
        name="mm", m=m, n=n, k=k, operands=operands, relu=False, scale=1.0, tiles=None
    )
    back = parse_kernel(kernel.text())
    assert back == kernel
    assert back.digest() == kernel.digest()


# -- it has to be able to travel -------------------------------------------------------------------


def test_the_parser_imports_nothing_from_merlin_so_it_can_travel_with_the_vocabulary():
    """The point of a parser is that a schedule becomes an artifact, and an artifact has to reach the
    places a schedule is used. Two of those may not import this tree: a minted package (the integrity
    scan rejects a merlin import outright) and any vendored copy of the rewrite vocabulary.

    Checked with the scan's own AST predicate rather than a substring search, for the reason that
    predicate exists: a name is not an import, and an import is not always spelled the same way.
    """
    from merlin.sched.ir import parse as parse_module
    from merlin.targetgen.oot_runner import _py_imports_merlin

    with open(parse_module.__file__, encoding="utf-8") as handle:
        assert _py_imports_merlin(handle.read()) is None
