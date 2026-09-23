"""The kernel IR ``mk``: a loop nest over ONE target's instructions.

A kernel is what a schedule produces for one fused group: its tensor arguments (DRAM buffers with shape,
dtype and access), and a body of counted loops and instruction calls. Every call names an instruction of
the target's schedule instruction set (``merlin.sched.isa``) and gives each of that instruction's
operands a value: an index expression over the enclosing loop variables, a pointer into one tensor
argument (a byte offset), ``NULL``, or a float.

The kernel is data, not code. It has one canonical text form (``Kernel.text``); its digest is the digest
of that text, so two schedules are the same schedule exactly when they print the same. The static
checker (``merlin.sched.check.static``) and the C emitter (``merlin.sched.codegen``) both walk this
structure; neither parses text.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

from .expr import Expr, ExprError, as_expr, evaluate, free_vars, render

DTYPE_BYTES = {"i8": 1, "i16": 2, "i32": 4, "f32": 4}
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

    #: Name of the on-chip memory, resolved against ``merlin.sched.mach.Machine.memories``.
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
    #: ``merlin.sched.mach.Machine.units``; nothing here knows what units a target has.
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
            lines.append(f"{pad}for {s.var} in 0..{s.extent}:")
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
