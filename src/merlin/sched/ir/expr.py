"""Integer index expressions of the kernel IR.

A schedule's instruction operands are integers that depend on loop variables: a tile's DRAM offset, its
row count, a remainder on the last tile. They are small expression trees rather than strings so that
one operand can be evaluated (the static checker visits every dynamic instance), printed (the kernel's
text form, which is what gets digested and diffed) and rendered to C (codegen) without three hand-kept
copies drifting apart.

The forms are deliberately few: constants, loop variables, sums, products by an integer constant, and a
``select`` on one loop variable equalling one constant, which is how a last-tile remainder is written.
Anything richer belongs in a loop transformation (split a loop, peel an iteration), not in an operand.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


class ExprError(ValueError):
    pass


class Expr:
    """Base of the expression forms; ``+`` and ``*`` build simplified trees."""

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(other, self)


@dataclass(frozen=True)
class Const(Expr):
    value: int


@dataclass(frozen=True)
class Var(Expr):
    name: str


@dataclass(frozen=True)
class Sum(Expr):
    terms: tuple[Expr, ...]


@dataclass(frozen=True)
class Scaled(Expr):
    expr: Expr
    factor: int


@dataclass(frozen=True)
class Select(Expr):
    """``then`` when loop variable ``var`` equals ``equals``, else ``other``."""

    var: str
    equals: int
    then: Expr
    other: Expr


def as_expr(x) -> Expr:
    if isinstance(x, Expr):
        return x
    if isinstance(x, bool) or not isinstance(x, int):
        raise ExprError(f"not an integer index expression: {x!r}")
    return Const(x)


def add(*xs) -> Expr:
    terms: list[Expr] = []
    const = 0
    for x in xs:
        e = as_expr(x)
        parts = e.terms if isinstance(e, Sum) else (e,)
        for t in parts:
            if isinstance(t, Const):
                const += t.value
            else:
                terms.append(t)
    if const:
        terms.append(Const(const))
    if not terms:
        return Const(0)
    return terms[0] if len(terms) == 1 else Sum(tuple(terms))


def mul(a, b) -> Expr:
    ea, eb = as_expr(a), as_expr(b)
    if isinstance(eb, Const):
        e, k = ea, eb.value
    elif isinstance(ea, Const):
        e, k = eb, ea.value
    else:
        raise ExprError("a product of two non-constant expressions is not an index expression")
    if k == 0:
        return Const(0)
    if k == 1:
        return e
    if isinstance(e, Const):
        return Const(e.value * k)
    if isinstance(e, Scaled):
        return mul(e.expr, e.factor * k)
    if isinstance(e, Sum):
        return add(*(mul(t, k) for t in e.terms))
    return Scaled(e, k)


def select(var, equals: int, then, other) -> Expr:
    t, o = as_expr(then), as_expr(other)
    if t == o:
        return t
    return Select(var.name if isinstance(var, Var) else str(var), int(equals), t, o)


def evaluate(e: Expr, env: Mapping[str, int]) -> int:
    if isinstance(e, Const):
        return e.value
    if isinstance(e, Var):
        if e.name not in env:
            raise ExprError(f"unbound variable {e.name!r}")
        return env[e.name]
    if isinstance(e, Sum):
        return sum(evaluate(t, env) for t in e.terms)
    if isinstance(e, Scaled):
        return evaluate(e.expr, env) * e.factor
    if isinstance(e, Select):
        if e.var not in env:
            raise ExprError(f"unbound variable {e.var!r}")
        return evaluate(e.then if env[e.var] == e.equals else e.other, env)
    raise ExprError(f"unknown expression {e!r}")


def free_vars(e: Expr) -> frozenset[str]:
    if isinstance(e, Var):
        return frozenset((e.name,))
    if isinstance(e, Sum):
        return frozenset().union(*(free_vars(t) for t in e.terms))
    if isinstance(e, Scaled):
        return free_vars(e.expr)
    if isinstance(e, Select):
        return frozenset((e.var,)) | free_vars(e.then) | free_vars(e.other)
    return frozenset()


def render(e: Expr, *, c: bool = False, index_bits: int | None = None) -> str:
    """The expression as text: the IR's own form, or a C expression when ``c``."""

    def integer(value):
        if index_bits is None:
            return str(value)
        if not c or type(index_bits) is not int or index_bits not in (32, 64):
            raise ExprError("typed integer rendering requires C and an explicit 32 or 64-bit index width")
        if type(value) is not int or not -(1 << (index_bits - 1)) <= value < (1 << (index_bits - 1)):
            raise ExprError("integer literal exceeds the selected C index width")
        if value == -(1 << (index_bits - 1)):
            return f"INT{index_bits}_MIN"
        literal = f"INT{index_bits}_C({abs(value)})"
        return f"(-{literal})" if value < 0 else literal

    if isinstance(e, Const):
        return integer(e.value)
    if isinstance(e, Var):
        return e.name
    if isinstance(e, Sum):
        return "(" + " + ".join(render(t, c=c, index_bits=index_bits) for t in e.terms) + ")"
    if isinstance(e, Scaled):
        return f"({render(e.expr, c=c, index_bits=index_bits)} * {integer(e.factor)})"
    if isinstance(e, Select):
        a = render(e.then, c=c, index_bits=index_bits)
        b = render(e.other, c=c, index_bits=index_bits)
        return f"(({e.var} == {integer(e.equals)}) ? {a} : {b})" if c else f"select({e.var} == {e.equals}, {a}, {b})"
    raise ExprError(f"unknown expression {e!r}")
