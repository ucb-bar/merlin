"""The kernel IR ``mk``: loop nests over one target's declared instructions.

Integer index expressions, kernels and their stages, lexical completion events, the integer-domain
check for counted schedules, and the instruction declarations a kernel is written against. Nothing
here knows which units or memories a target has; those are names resolved against a machine.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# ---- expr --------------------------------------------------------------------------------------


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


# ---- kernel ------------------------------------------------------------------------------------
# The kernel IR ``mk``: a loop nest over ONE target's instructions.
#
# A kernel is what a schedule produces for one fused group: its tensor arguments (DRAM buffers with shape,
# dtype and access), and a body of counted loops and instruction calls. Every call names an instruction of
# the target's schedule instruction set (``merlin.compiler.kernel_ir.ir``) and gives each of that instruction's
# operands a value: an index expression over the enclosing loop variables, a pointer into one tensor
# argument (a byte offset), ``NULL``, or a float.
#
# The kernel is data, not code. It has one canonical text form (``Kernel.text``); its digest is the digest
# of that text, so two schedules are the same schedule exactly when they print the same. The static
# checker (``merlin.compiler.kernel_ir.checks``) and the C emitter (``merlin.compiler.kernel_ir.codegen``) both walk this
# structure; neither parses text.

DTYPE_BYTES = {"i8": 1, "i16": 2, "i32": 4, "f32": 4, "f64": 8}
ACCESS = ("read", "write", "readwrite")


class KernelError(ValueError):
    pass


@dataclass(frozen=True)
class TensorArg:
    name: str
    shape: tuple[int, ...]
    dtype: str
    access: str

    @property
    def nbytes(self) -> int:
        n = DTYPE_BYTES[self.dtype]
        for d in self.shape:
            n *= d
        return n


@dataclass(frozen=True)
class Ptr:
    """A byte offset into one tensor argument."""

    tensor: str
    offset: Expr


@dataclass(frozen=True)
class _Null:
    def __repr__(self) -> str:
        return "NULL"


NULL = _Null()


@dataclass(frozen=True)
class Stage:
    """Where one of a call's operands lives on chip, while the call runs.

    Recorded on the call rather than inferred from a pointer because the two failures this exists to
    catch are both invisible from a pointer alone. A destination written on top of a live operand
    produced zero correct elements out of 16384 on a shipped kernel, with no error anywhere -- the
    offset was a constant that happened to be right for one tile size. And a bank shared between two
    engines starves the one the arbiter deprioritises, which is a fact about the BANK, not the address.
    """

    #: Name of the on-chip memory, resolved against ``merlin.compiler.kernel_ir.machine.Machine.memories``.
    memory: str
    #: First row occupied.
    row: int
    #: Rows occupied.
    rows: int
    #: Which bank, where the caller chose one. ``None`` means the placement does not name a bank, which
    #: is honest on a memory whose bank count was never derived.
    bank: int | None = None
    #: Whether the call WRITES these rows. Two readers may share rows; a writer may not share with
    #: anyone.
    writes: bool = False
    #: How many rotating COPIES this staging occupies, starting at ``row``. ``1`` is a single buffer.
    #:
    #: A depth of n is what lets a producer fill one copy while a consumer reads another, and it is the
    #: shape two targets here independently arrive at -- a K-tile-parity toggle between two scratchpad
    #: halves on one, a weight-slot ping-pong on the other. Recorded on the staging rather than left to
    #: the row arithmetic because the copies are the thing that must FIT: n copies of r rows occupy
    #: n*r, and a capacity check that priced only r would approve a schedule the store cannot hold.
    depth: int = 1


@dataclass(frozen=True)
class Call:
    """One instruction, its operands, and -- where the machine has more than one agent -- WHERE it runs
    and WHAT it waits for.

    ``unit``, ``produces`` and ``awaits`` are the whole of the asynchrony this IR can express, and they
    are here rather than in a side table for one reason: a dependence that must hold has to survive in
    the PAYLOAD. Recorded only as statement order it does not survive a reordering, and a reordering is
    the thing a schedule does. One target states the cost of getting this wrong plainly -- program order
    at its memory-mapped command port does not order a register update against the work that reads it.

    All three are ``None``/empty on a machine with one agent and no asynchrony, which is the common case
    and costs such a kernel nothing: its text form is unchanged, so its digest is unchanged.
    """

    instr: str
    args: tuple[tuple[str, Any], ...]
    #: The unit this issues on, where the machine names more than one. Resolved against
    #: ``merlin.compiler.kernel_ir.machine.Machine.units``; nothing here knows what units a target has.
    unit: str | None = None
    #: Name of the completion token this produces, for a consumer to await. A call that produces one is
    #: asynchronous: it has started when the next statement issues, not finished.
    produces: str | None = None
    #: Tokens that must have completed before this issues.
    awaits: tuple[str, ...] = ()
    #: Where this call's operands sit on chip while it runs.
    stages: tuple[Stage, ...] = ()
    #: Configuration this call ESTABLISHES, as ``key -> value``. The value is live for every later call
    #: until another call sets the same key.
    sets: tuple[tuple[str, Any], ...] = ()
    #: Configuration this call REQUIRES to be live, as ``key -> value``. A call that assumes a key no
    #: earlier call set, or set to something else, is misconfigured -- and on these targets that returns
    #: wrong data rather than an error, because the device is configured, just not for this.
    assumes: tuple[tuple[str, Any], ...] = ()

    def arg(self, name: str):
        for k, v in self.args:
            if k == name:
                return v
        raise KeyError(name)


@dataclass(frozen=True)
class Loop:
    var: str
    extent: int
    body: tuple
    #: Explicit completion-token ownership: (local block argument, incoming token).
    iter_args: tuple[tuple[str, str], ...] = ()
    #: Body tokens passed to the next iteration, in block-argument order.
    yields: tuple[str, ...] = ()
    #: Final tokens bound in the enclosing lexical scope.
    results: tuple[str, ...] = ()


#: What separates one printed attribute from the next. Named because the READER splits on it and the
#: WRITER joins with it, and the two drifting apart is exactly how a kernel stops reading back.
_ATTR_SEP = ", "


@dataclass(frozen=True)
class Kernel:
    name: str
    args: tuple[TensorArg, ...]
    body: tuple
    attrs: tuple[tuple[str, str], ...] = ()

    def tensor(self, name: str) -> TensorArg:
        for t in self.args:
            if t.name == name:
                return t
        raise KernelError(f"kernel {self.name!r} has no tensor argument {name!r}")

    def text(self) -> str:
        params = ", ".join(f"{t.name}: {t.dtype}[{','.join(map(str, t.shape))}] {t.access}" for t in self.args)
        lines = [f"kernel {self.name}({params})"]
        if self.attrs:
            for key, value in self.attrs:
                # An attribute is separated from the next by ``", "`` and from its own value by ``"="``.
                # The reader splits on those, and only protects a separator that sits inside brackets --
                # so a BARE one inside a key or value cuts the attribute in half and the text no longer
                # reads back. That is not a cosmetic defect here: a kernel's identity IS its text (see
                # :meth:`digest`), and a schedule that cannot round-trip has no stable identity to
                # compare. Refuse to print it rather than emit something unreadable, because the failure
                # otherwise surfaces far away, as a parse error on a file nobody thinks they broke.
                for separator in (_ATTR_SEP, "="):
                    if separator in key:
                        raise KernelError(
                            f"kernel {self.name!r} attribute key {key!r} contains {separator!r}, which "
                            f"separates attributes in the printed form, so the kernel would not read back"
                        )
                if _ATTR_SEP in value:
                    raise KernelError(
                        f"kernel {self.name!r} attribute {key!r} has value {value!r} containing "
                        f"{_ATTR_SEP!r}, which separates attributes in the printed form, so the kernel "
                        f"would not read back. Spell the value without it."
                    )
            lines.append("  attrs {" + _ATTR_SEP.join(f"{k}={v}" for k, v in self.attrs) + "}")
        _print_body(self.body, 1, lines)
        return "\n".join(lines) + "\n"

    def digest(self) -> str:
        return hashlib.sha256(self.text().encode()).hexdigest()


def _value_text(v) -> str:
    if isinstance(v, Expr):
        return render(v)
    if isinstance(v, Ptr):
        return f"&{v.tensor}[{render(v.offset)}]"
    if v is NULL:
        return "null"
    if isinstance(v, float):
        return v.hex()
    raise KernelError(f"not an operand value: {v!r}")


def _print_body(body, depth: int, lines: list[str]) -> None:
    pad = "  " * depth
    for s in body:
        if isinstance(s, Loop):
            carried = (
                " carry("
                + ",".join(f"{name}={value}" for name, value in s.iter_args)
                + ") yield("
                + ",".join(s.yields)
                + ") result("
                + ",".join(s.results)
                + ")"
                if s.iter_args or s.yields or s.results
                else ""
            )
            lines.append(f"{pad}for {s.var} in 0..{s.extent}{carried}:")
            _print_body(s.body, depth + 1, lines)
        else:
            # The asynchrony decorates the call rather than replacing it, so a synchronous kernel prints
            # exactly as it always did and keeps its digest.
            prefix = f"{s.produces} = " if s.produces else ""
            suffix = f" on {s.unit}" if s.unit else ""
            waits = f" after {','.join(s.awaits)}" if s.awaits else ""
            if s.stages:
                where = " in " + ",".join(
                    f"{g.memory}[{g.row}:{g.row + g.rows}]"
                    # Printed only when it is more than one, so a single-buffered staging -- every one
                    # written before this field existed -- prints exactly as it did and keeps its digest.
                    + (f"x{g.depth}" if g.depth != 1 else "")
                    + (f"@{g.bank}" if g.bank is not None else "")
                    + ("w" if g.writes else "")
                    for g in s.stages
                )
            else:
                where = ""
            conf = "".join(
                f" {word} " + ",".join(f"{k}={_value_text(v)}" for k, v in pairs)
                for word, pairs in (("sets", s.sets), ("needs", s.assumes))
                if pairs
            )
            body = ", ".join(f"{k}={_value_text(v)}" for k, v in s.args)
            lines.append(f"{pad}{prefix}{s.instr}({body}){suffix}{waits}{where}{conf}")


def _normalize(v):
    if isinstance(v, Ptr):
        return v if isinstance(v.offset, Expr) else Ptr(v.tensor, as_expr(v.offset))
    if isinstance(v, Expr) or v is NULL:
        return v
    if isinstance(v, bool):
        return as_expr(int(v))
    if isinstance(v, int):
        return as_expr(v)
    if isinstance(v, float):
        return v
    raise KernelError(f"not an operand value: {v!r}")


def call(
    instr: str,
    *,
    unit: str | None = None,
    produces: str | None = None,
    awaits: tuple[str, ...] | str = (),
    stages: tuple[Stage, ...] = (),
    sets: Mapping[str, Any] | None = None,
    assumes: Mapping[str, Any] | None = None,
    **args,
) -> Call:
    """A call with operands in the order given (which must be the instruction's operand order)."""
    waits = (awaits,) if isinstance(awaits, str) else tuple(awaits)
    # Sorted, so two callers that pass the same configuration in a different order produce the same
    # text and therefore the same digest. The operand list is NOT sorted -- there the order is the
    # instruction's own and carries meaning.
    conf = tuple(tuple(sorted((k, _normalize(v)) for k, v in (m or {}).items())) for m in (sets, assumes))
    return Call(
        instr,
        tuple((k, _normalize(v)) for k, v in args.items()),
        unit,
        produces,
        waits,
        tuple(stages),
        *conf,
    )


def loop(var: str, extent: int, *body) -> Loop:
    return Loop(var, int(extent), tuple(body))


def check_structure(kernel: Kernel) -> list[str]:
    """Target-free well-formedness: unique tensor names, positive extents, no shadowed loop variable,
    every variable bound by an enclosing loop, every pointer naming a tensor argument."""
    errors: list[str] = []
    names = [t.name for t in kernel.args]
    if len(set(names)) != len(names):
        errors.append("duplicate tensor argument names")
    for t in kernel.args:
        if t.dtype not in DTYPE_BYTES or t.access not in ACCESS or not t.shape or min(t.shape) < 1:
            errors.append(f"tensor {t.name!r}: bad dtype/access/shape")

    def visit(body, bound: tuple[str, ...]):
        for s in body:
            if isinstance(s, Loop):
                if s.var in bound or s.var in names:
                    errors.append(f"loop variable {s.var!r} shadows an enclosing name")
                if s.extent < 1:
                    errors.append(f"loop {s.var!r} has extent {s.extent}")
                visit(s.body, bound + (s.var,))
                continue
            for k, v in s.args:
                e = v.offset if isinstance(v, Ptr) else v
                if isinstance(v, Ptr) and v.tensor not in names:
                    errors.append(f"{s.instr}.{k}: pointer into unknown tensor {v.tensor!r}")
                if isinstance(e, Expr):
                    unbound = free_vars(e) - set(bound)
                    if unbound:
                        errors.append(f"{s.instr}.{k}: unbound variables {sorted(unbound)}")
            for word, pairs in (("sets", s.sets), ("needs", s.assumes)):
                for k, v in pairs:
                    if isinstance(v, Expr):
                        unbound = free_vars(v) - set(bound)
                        if unbound:
                            errors.append(f"{s.instr} {word} {k}: unbound variables {sorted(unbound)}")

    visit(kernel.body, ())
    errors.extend(_token_errors(kernel))
    return errors


def _token_errors(kernel: Kernel) -> list[str]:
    """Target-free well-formedness of the completion tokens.

    Only what holds on ANY machine: a token is produced once, and is awaited after it is produced.
    whether an un-awaited token is a BUG is a property of the hardware -- on a machine whose hardware
    tracks dependencies it is fine, on one where the compiler must separate them it is a wrong answer --
    so that question belongs to a machine-aware check and deliberately is not asked here.
    """
    if has_carried_events(kernel.body):
        return lexical_event_errors(kernel)
    errors: list[str] = []
    produced: set[str] = set()

    def visit(body):
        for s in body:
            if isinstance(s, Loop):
                visit(s.body)
                continue
            for token in s.awaits:
                if token not in produced:
                    errors.append(f"{s.instr}: awaits {token!r}, which no earlier call produces")
            if s.produces:
                if s.produces in produced:
                    errors.append(f"{s.instr}: produces {s.produces!r}, which is already a live token")
                produced.add(s.produces)

    visit(kernel.body)
    return errors


def instances(kernel: Kernel) -> Iterator[tuple[Call, dict[str, int]]]:
    """Every dynamic instance of every call, in program order, with its loop environment."""

    def visit(body, env):
        for s in body:
            if isinstance(s, Loop):
                for i in range(s.extent):
                    env[s.var] = i
                    yield from visit(s.body, env)
                del env[s.var]
            else:
                yield s, dict(env)

    yield from visit(kernel.body, {})


@dataclass(frozen=True)
class ConcretePtr:
    tensor: str
    offset: int


def concretize(c: Call, env: Mapping[str, int]) -> dict[str, Any]:
    """One instance's operand values: ints, ``ConcretePtr``, ``None`` for NULL, floats."""
    out: dict[str, Any] = {}
    for k, v in c.args:
        try:
            if isinstance(v, Ptr):
                out[k] = ConcretePtr(v.tensor, evaluate(v.offset, env))
            elif v is NULL:
                out[k] = None
            elif isinstance(v, Expr):
                out[k] = evaluate(v, env)
            else:
                out[k] = v
        except ExprError as exc:
            raise KernelError(f"{c.instr}.{k}: {exc}") from exc
    return out


# ---- events ------------------------------------------------------------------------------------
# Lexical completion-token binding; no target timing or completion inference.


def has_carried_events(body):
    return any(
        isinstance(op, Loop) and (op.iter_args or op.yields or op.results or has_carried_events(op.body)) for op in body
    )


def lexical_event_errors(kernel):
    errors = []

    def name(value):
        return (
            type(value) is str
            and value
            and (value[0].isalpha() or value[0] == "_")
            and all(c.isalnum() or c == "_" for c in value)
        )

    def body(statements, incoming):
        available = set(incoming)
        for op in statements:
            if isinstance(op, Loop):
                if (
                    type(op.iter_args) is not tuple
                    or type(op.yields) is not tuple
                    or type(op.results) is not tuple
                    or len(op.iter_args) != len(op.yields)
                    or len(op.yields) != len(op.results)
                    or any(type(pair) is not tuple or len(pair) != 2 for pair in op.iter_args)
                ):
                    errors.append(f"loop {op.var!r}: completion input/yield/result arities differ")
                    continue
                inputs = tuple(pair[0] for pair in op.iter_args)
                initial = tuple(pair[1] for pair in op.iter_args)
                if any(not name(n) for n in (*inputs, *initial, *op.yields, *op.results)) or any(
                    len(set(v)) != len(v) for v in (inputs, initial, op.yields, op.results)
                ):
                    errors.append(f"loop {op.var!r}: malformed or aliased completion bindings")
                    continue
                if not set(initial) <= available:
                    errors.append(f"loop {op.var!r}: incoming completion has no dominating producer")
                child = body(op.body, inputs)
                if not set(op.yields) <= child:
                    errors.append(f"loop {op.var!r}: yielded completion is outside its lexical scope")
                # A carry moves ownership. The old name cannot accidentally await a
                # stale handle instead of the result of the final iteration.
                available.difference_update(initial)
                if set(op.results) & available:
                    errors.append(f"loop {op.var!r}: completion result shadows a live lexical name")
                available.update(op.results)
            else:
                if any(not name(n) for n in op.awaits) or not set(op.awaits) <= available:
                    errors.append(f"{op.instr}: awaits an unknown or out-of-scope completion")
                if op.produces:
                    if not name(op.produces) or op.produces in available:
                        errors.append(f"{op.instr}: completion producer shadows a live lexical name")
                    available.add(op.produces)
        return available

    body(kernel.body, ())
    return errors


# ---- integer_domain ----------------------------------------------------------------------------
# Check counted-schedule integer intermediates without target loading or emission.
#
# The same conservative interval check serves shared verification and C codegen.
# It never expands dynamic iterations and grants no target address/window or
# completion semantics. Unsupported expressions remain a barrier.


class IntegerDomainError(ValueError):
    """A selected signed index domain cannot represent every intermediate."""


def check_integer_domain(kernel, bits):
    """Conservative bounds for every emitted integer intermediate, without expansion."""
    if type(bits) is not int or bits not in (32, 64):
        raise IntegerDomainError("explicit signed 32-bit or 64-bit index domain required")
    limit = (1 << (bits - 1)) - 1

    def bounds(expression, domains):
        if isinstance(expression, Const):
            result = expression.value, expression.value
        elif isinstance(expression, Var):
            if expression.name not in domains:
                raise IntegerDomainError(f"unbound integer index variable: {expression.name}")
            result = domains[expression.name]
        elif isinstance(expression, Sum):
            lo = hi = 0
            for term in expression.terms:
                term_lo, term_hi = bounds(term, domains)
                lo, hi = lo + term_lo, hi + term_hi
                if lo < -limit - 1 or hi > limit:
                    raise IntegerDomainError(
                        f"integer sum intermediate may overflow the selected signed {bits}-bit C index domain"
                    )
            result = lo, hi
        elif isinstance(expression, Scaled):
            lo, hi = bounds(expression.expr, domains)
            result = tuple(sorted((lo * expression.factor, hi * expression.factor)))
        elif isinstance(expression, Select):
            if expression.var not in domains:
                raise IntegerDomainError(f"unbound integer index variable: {expression.var}")
            lo, hi = domains[expression.var]
            parts = []
            if lo <= expression.equals <= hi:
                parts.append(
                    bounds(expression.then, {**domains, expression.var: (expression.equals, expression.equals)})
                )
            if lo != hi or lo != expression.equals:
                parts.append(bounds(expression.other, domains))
            result = min(a for a, _b in parts), max(b for _a, b in parts)
        else:
            raise IntegerDomainError("unknown integer expression in selected C index domain")
        if result[0] < -limit - 1 or result[1] > limit:
            raise IntegerDomainError(f"integer expression may overflow the selected signed {bits}-bit C index domain")
        return result

    def walk(body, domains):
        for statement in body:
            if isinstance(statement, Loop):
                if not 0 < statement.extent <= limit:
                    raise IntegerDomainError(f"loop extent exceeds the selected signed {bits}-bit C index domain")
                walk(statement.body, {**domains, statement.var: (0, statement.extent - 1)})
            elif isinstance(statement, Call):
                for _name, value in statement.args:
                    if isinstance(value, Expr):
                        bounds(value, domains)
                    elif isinstance(value, Ptr):
                        bounds(value.offset, domains)
            else:
                raise IntegerDomainError("unknown statement in selected integer index domain")

    walk(kernel.body, {})


# ---- types -------------------------------------------------------------------------------------
# Pure instruction declarations; no backend discovery or target implementation.
#
# Targets supply these typed call interfaces and their independently qualified
# rendering/checking capabilities. The declarations contain no target schedule.

OPERAND_KINDS = ("int", "flag", "ptr", "float")


class IsaError(ValueError):
    pass


@dataclass(frozen=True)
class Operand:
    name: str
    kind: str

    def __post_init__(self):
        if self.kind not in OPERAND_KINDS:
            raise IsaError(f"operand {self.name!r}: unknown kind {self.kind!r}")


@dataclass(frozen=True)
class InstrDef:
    name: str
    operands: tuple[Operand, ...]
    #: C operand texts in operand order -> one C statement (no trailing newline).
    render_c: Callable[[Sequence[str]], str]
    check: Callable[[Mapping[str, Any], dict], list[str]] | None = None
    footprint: Callable[[Mapping[str, Any]], list[tuple[str, int, str]]] | None = None
    doc: str = ""

    def operand_names(self) -> tuple[str, ...]:
        return tuple(o.name for o in self.operands)


@dataclass(frozen=True)
class InstructionSet:
    target: str
    instrs: Mapping[str, InstrDef]
    #: IR dtype -> the target's C element type for a tensor argument of that dtype.
    c_types: Mapping[str, str]
    #: One C statement that waits until every issued instruction has completed and its writes are
    #: visible to the host; the emitter ends every kernel with it.
    drain_c: str
    facts: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, str] = field(default_factory=dict)
    #: Rules on the state left after a kernel's LAST instance (e.g. an unfinished accumulation).
    finish: Callable[[dict], list[str]] | None = None

    def instr(self, name: str) -> InstrDef:
        if name not in self.instrs:
            raise IsaError(f"{self.target}: no schedule instruction {name!r} (known: {sorted(self.instrs)})")
        return self.instrs[name]

    def digest(self) -> str:
        """Identity of everything that decides what a call means: operand lists, facts, provenance."""
        doc = {
            "target": self.target,
            "drain": self.drain_c,
            "c_types": dict(self.c_types),
            "instrs": {n: [(o.name, o.kind) for o in d.operands] for n, d in sorted(self.instrs.items())},
            "facts": self.facts,
            "provenance": dict(self.provenance),
        }
        return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()


# ---- isa ---------------------------------------------------------------------------------------
# Instruction declarations and the legacy explicit backend lookup bridge.


def instruction_set(target: str) -> InstructionSet:
    from merlin.runtime.backends import base

    backend = base.get_backend(target)
    hook = getattr(backend, "sched_instruction_set", None)
    if hook is None:
        raise IsaError(f"target {target!r} declares no schedule instruction set")
    iset = hook()
    if not isinstance(iset, InstructionSet):
        raise IsaError(f"target {target!r}: sched_instruction_set() returned {type(iset).__name__}")
    return iset


__all__ = [
    "Call",
    "ConcretePtr",
    "Const",
    "Expr",
    "ExprError",
    "InstrDef",
    "InstructionSet",
    "IsaError",
    "Kernel",
    "KernelError",
    "Loop",
    "NULL",
    "OPERAND_KINDS",
    "Operand",
    "Ptr",
    "Scaled",
    "Select",
    "Stage",
    "Sum",
    "TensorArg",
    "Var",
    "add",
    "as_expr",
    "call",
    "check_structure",
    "concretize",
    "evaluate",
    "instances",
    "instruction_set",
    "loop",
    "mul",
    "render",
    "select",
]
