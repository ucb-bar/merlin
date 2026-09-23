"""The scheduling language: semantics-preserving rewrites of a kernel, each with a proof obligation.

:mod:`merlin.sched.ir` says what a kernel does and :mod:`merlin.sched.mach` says what the hardware is.
This is the vocabulary a schedule is WRITTEN in -- the set of moves a search, or an agent, composes.

THREE THINGS MAKE THIS SEARCHABLE RATHER THAN MERELY CORRECT.

*A failure has two kinds.* :class:`NotApplicable` means this move does not apply here and a search may
try the next one; :class:`PrimitiveError` means the schedule is wrong and a search must stop. Collapsing
them is what makes an exploring caller either give up on a legal space or grind on an illegal one, and
telling them apart is the single property a transformation layer owes a search loop.

*A cursor is a PATH, re-resolved against the kernel it is applied to, never a live pointer.* A kernel's
identity is its text; a handle into a value whose identity is its printed form means nothing after the
first rewrite. Naming the position instead costs a lookup and survives composition.

*Every primitive declares its PROOF OBLIGATION*, and the declaration is checked rather than trusted:

``by_construction``  the rewrite cannot change what is computed -- the set of dynamic instruction
                     instances and their operand values is invariant, and the test asserts that.
``bounded_check``    legality is decidable here, cheaply, against the machine or the dependences, and
                     the primitive REFUSES when it does not hold.
``rtl_gate``         legality is not decidable here. The primitive states what it assumed, and a
                     measurement elsewhere is what can refute it.

A primitive that declares ``bounded_check`` and returns no problems unconditionally is the failure mode
this classification exists to surface, so the suite asserts each one can refuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace as _replace
from typing import Any, Callable

from merlin.sched.ir import Call, Kernel, Loop, Ptr, Stage
from merlin.sched.ir.expr import Expr, Scaled, Sum, Var, add, as_expr, evaluate, free_vars, mul, render, select
from merlin.sched.ir.kernel import KernelError, _print_body, instances

__all__ = [
    "Cursor",
    "NotApplicable",
    "PrimitiveError",
    "PROOFS",
    "divide_loop",
    "fuse",
    "hoist_config",
    "multi_buffer",
    "bind_bank",
    "pipeline",
    "peel",
    "place",
    "specialize",
    "stage_mem",
    "proof_of",
    "reorder",
    "replace",
    "unroll",
]

PROOFS = ("by_construction", "bounded_check", "rtl_gate")


class PrimitiveError(KernelError):
    """DEFINITE failure: applying this would produce a wrong schedule. A search must stop."""


class NotApplicable(PrimitiveError):
    """SILENCEABLE failure: this primitive does not apply at this position.

    A subclass of the definite error on purpose -- a caller that does not know the difference still
    fails safe -- while a caller that does can catch this one alone and try the next move.
    """


@dataclass(frozen=True)
class Cursor:
    """Where in a kernel a primitive applies: a PATH, re-resolved at apply time.

    ``loops`` names the enclosing loop variables outermost first. ``index`` selects a statement in that
    body, or ``None`` to mean the innermost loop itself. ``instr``, when given, is checked against what
    is actually found -- a cursor that resolves to a different instruction than the caller believed is a
    DEFINITE failure, not a miss, because the caller's reasoning was about something else.
    """

    loops: tuple[str, ...] = ()
    index: int | None = None
    instr: str | None = None


def proof_of(primitive: Callable[..., Any]) -> str:
    """The obligation a primitive declares. Raises for one that declares none, so an unclassified
    primitive cannot reach the register as if it had been reasoned about."""
    proof = getattr(primitive, "PROOF", None)
    if proof not in PROOFS:
        raise PrimitiveError(f"{getattr(primitive, '__name__', primitive)!r} declares no proof obligation")
    return proof


def _obligation(proof: str):
    def decorate(fn):
        fn.PROOF = proof
        return fn

    return decorate


# -- resolution --------------------------------------------------------------------------------------


def _one_loop(here: tuple, var: str) -> Loop:
    """The single loop named ``var`` at this level, or raise.

    AMBIGUITY IS REFUSED, and that is the point. Two sibling loops may legally share a variable name --
    `check_structure` only rejects shadowing of an ENCLOSING name -- and the read path used to take the
    first match while the write path rewrote EVERY match. `specialize` on the first of two `for i` loops
    therefore rewrote both, destroying the second's operands, and `peel` replaced the second loop with a
    copy of the first. Neither was caught: `specialize` is a bounded check that is ALLOWED to change
    instances, and `_require_same_instances` compares instruction and operand values only, so a
    difference confined to `sets`, `unit` or `stages` passed silently.
    """
    found = [s for s in here if isinstance(s, Loop) and s.var == var]
    if not found:
        raise NotApplicable(f"no loop {var!r} at this level; have {[s.var for s in here if isinstance(s, Loop)]}")
    if len(found) > 1:
        raise PrimitiveError(
            f"{len(found)} sibling loops at this level are named {var!r}, so the cursor names no single "
            "one of them. Rename one, or address them by a path that distinguishes them -- guessing "
            "which was meant would rewrite the wrong loop, or every one of them."
        )
    return found[0]


def _resolve(body: tuple, cursor: Cursor) -> tuple[tuple, int | None]:
    """``(body_containing_the_target, index)`` for a cursor, or raise.

    ``NotApplicable`` when the path does not exist in this kernel; ``PrimitiveError`` when it exists but
    holds something other than what the cursor asserted.
    """
    here = body
    for var in cursor.loops:
        here = _one_loop(here, var).body
    if cursor.index is None:
        return here, None
    if not (0 <= cursor.index < len(here)):
        raise NotApplicable(f"statement {cursor.index} does not exist in a body of {len(here)}")
    target = here[cursor.index]
    if cursor.instr is not None:
        actual = target.instr if isinstance(target, Call) else f"loop {target.var}"
        if actual != cursor.instr:
            raise PrimitiveError(
                f"the cursor asserts instruction {cursor.instr!r} but that position holds {actual!r}; "
                "the caller reasoned about a different statement"
            )
    return here, cursor.index


def _rebuild(body: tuple, path: tuple[str, ...], new_inner: tuple) -> tuple:
    """``body`` with the sub-body at ``path`` replaced. Kernels are frozen, so a rewrite rebuilds."""
    if not path:
        return new_inner
    target = _one_loop(body, path[0])  # refuses two siblings of the same name, as the read path does
    return tuple(
        _replace(stmt, body=_rebuild(stmt.body, path[1:], new_inner)) if stmt is target else stmt for stmt in body
    )


def _loop_at(body: tuple, path: tuple[str, ...]) -> Loop:
    here, loop = body, None
    for var in path:
        loop = _one_loop(here, var)
        here = loop.body
    if loop is None:
        raise NotApplicable("the cursor names no loop")
    return loop


def _substitute(value: Any, var: str, expr: Expr) -> Any:
    """``value`` with every occurrence of loop variable ``var`` replaced by ``expr``."""
    from merlin.sched.ir.kernel import Ptr

    if isinstance(value, Ptr):
        return Ptr(value.tensor, _substitute(value.offset, var, expr))
    if isinstance(value, Var):
        return expr if value.name == var else value
    if isinstance(value, Expr):
        return _rewrite_expr(value, var, expr)
    return value


def _affine_shift(expr: Expr, var: str) -> int | None:
    """``c`` when ``expr`` is exactly ``var + c``, else ``None`` (including for ``2*var`` or ``var + j``).

    Deliberately narrow. A select tests one variable against one constant, so only a rewrite that keeps
    the subject a bare variable can be absorbed into the test; anything else has to refuse rather than
    produce a select this expression language cannot write down.
    """
    from merlin.sched.ir.expr import Const, Sum

    if isinstance(expr, Var):
        return 0 if expr.name == var else None
    if not isinstance(expr, Sum):
        return None
    shift, seen = 0, False
    for term in expr.terms:
        if isinstance(term, Var) and term.name == var and not seen:
            seen = True
        elif isinstance(term, Const):
            shift += term.value
        else:
            return None
    return shift if seen else None


def _rewrite_expr(e: Expr, var: str, expr: Expr) -> Expr:
    from merlin.sched.ir.expr import Const, Scaled, Select, Sum, select

    if isinstance(e, Const):
        return e
    if isinstance(e, Var):
        return expr if e.name == var else e
    # Rebuilt through `add`/`mul` rather than the dataclasses, so the result is renormalised: a
    # substitution that makes a subtree constant must FOLD it. Left unfolded, peeling the last iteration
    # of `for k in 0..6` writes `(5 + 1)` where the same schedule written directly says `6` -- the same
    # kernel with a different canonical text, and therefore a different digest.
    if isinstance(e, Sum):
        return add(*(_rewrite_expr(t, var, expr) for t in e.terms))
    if isinstance(e, Scaled):
        return mul(_rewrite_expr(e.expr, var, expr), e.factor)
    if isinstance(e, Select):
        then, other = _rewrite_expr(e.then, var, expr), _rewrite_expr(e.other, var, expr)
        if e.var != var:
            return select(Var(e.var), e.equals, then, other)
        # Substituting the variable a select tests: a CONSTANT decides the branch, so the select
        # collapses -- which is exactly what peeling an iteration should do to a last-tile remainder.
        # Anything else would leave a select whose subject is no longer a variable, which this
        # expression language cannot write down, so it refuses rather than approximating.
        if isinstance(expr, Const):
            return then if expr.value == e.equals else other
        # An affine shift of the variable is an affine shift of the test: peeling the front of a loop
        # renumbers it from zero, so `k0 == 6` on the old numbering is `k0 == 5` on the new one. This is
        # exact, not an approximation, and without it a loop whose body carries a last-tile remainder --
        # which is every tiled reduction over a shape the block does not divide -- could not be peeled.
        shift = _affine_shift(expr, var)
        if shift is not None:
            return select(Var(var), e.equals - shift, then, other)
        raise NotApplicable(
            f"{var!r} is the subject of a select, and substituting {render(expr)!r} for it cannot be "
            "written as an index expression; it is neither a constant nor a shift of the variable"
        )
    return e


def _subst_call(c: Call, var: str, expr: Expr) -> Call:
    """``c`` with ``var`` replaced by ``expr`` in every value it carries, and EVERYTHING ELSE KEPT.

    Rebuilt with ``dataclasses.replace`` rather than by calling ``Call(...)`` with the fields this
    function happens to know about. Three primitives reconstructed a call positionally and so dropped
    its unit, its completion token, its staging and its configuration: tiling a placed, synchronised
    kernel silently returned an unplaced, unsynchronised one, and `_require_same_instances` could not
    see it because it compares instruction and operand values only -- by design, since order is what a
    transformation changes. Using ``replace`` makes the preservation structural, so a field added to
    ``Call`` later is carried by every primitive without anyone remembering to.
    """
    return _replace(
        c,
        args=tuple((k, _substitute(v, var, expr)) for k, v in c.args),
        sets=tuple((k, _substitute(v, var, expr)) for k, v in c.sets),
        assumes=tuple((k, _substitute(v, var, expr)) for k, v in c.assumes),
    )


def _map_calls(body: tuple, fn: Callable[[Call], Call]) -> tuple:
    out = []
    for stmt in body:
        if isinstance(stmt, Loop):
            out.append(_replace(stmt, body=_map_calls(stmt.body, fn)))
        else:
            out.append(fn(stmt))
    return tuple(out)


# -- the primitives ----------------------------------------------------------------------------------


@_obligation("by_construction")
def divide_loop(kernel: Kernel, cursor: Cursor, factor: int, *, names: tuple[str, str] | None = None) -> Kernel:
    """Split the loop the cursor names into an outer and an inner loop.

    Correct by construction, and only where it is: a factor that does not divide the extent would need a
    tail, and inventing one silently is how a schedule stops computing what it did. That case REFUSES --
    peel the iteration first, which is its own move.
    """
    if factor <= 1:
        raise NotApplicable(f"a factor of {factor} splits nothing")
    target = _loop_at(kernel.body, cursor.loops)
    if target.extent % factor:
        raise NotApplicable(
            f"{factor} does not divide the extent {target.extent} of {target.var!r}; this split would "
            "need a tail, and one invented here would change what the kernel computes"
        )
    outer_name, inner_name = names or (f"{target.var}o", f"{target.var}i")
    for name in (outer_name, inner_name):
        if name in _bound_vars(kernel.body):
            raise PrimitiveError(f"{name!r} is already a loop variable of this kernel")
    rebuilt = mul(as_expr(Var(outer_name)), factor)
    inner_body = _map_calls(
        target.body,
        lambda c: _subst_call(c, target.var, add(rebuilt, Var(inner_name))),
    )
    split = Loop(outer_name, target.extent // factor, (Loop(inner_name, factor, inner_body),))
    return _replace(kernel, body=_replace_loop(kernel.body, cursor.loops, split))


@_obligation("by_construction")
def unroll(kernel: Kernel, cursor: Cursor) -> Kernel:
    """Replace the loop the cursor names with its iterations, in order.

    Correct by construction: the same instances, in the same order, with the loop variable folded into
    each operand. An extent this pass cannot enumerate refuses rather than guessing a bound.
    """
    target = _loop_at(kernel.body, cursor.loops)
    if target.extent <= 0:
        raise NotApplicable(f"loop {target.var!r} has extent {target.extent}")
    if target.extent > 4096:
        raise NotApplicable(
            f"unrolling {target.var!r} would emit {target.extent} copies; a bound this large is a "
            "tiling decision, not an unroll"
        )
    flat: list[Any] = []
    for i in range(target.extent):
        flat.extend(
            _map_calls(
                target.body,
                lambda c, i=i: _subst_call(c, target.var, as_expr(i)),
            )
        )
    out = _replace(kernel, body=_splice(kernel.body, cursor.loops, tuple(flat)))
    _require_well_formed(out, f"unrolling {target.var!r}")
    return out


@_obligation("bounded_check")
def reorder(kernel: Kernel, cursor: Cursor) -> Kernel:
    """Swap the loop the cursor names with the one immediately inside it.

    NOT correct by construction: swapping two loops changes the order dynamic instances execute in, and
    whether that is legal depends on the dependences between them. This checks the one thing it can
    check cheaply and exactly -- that the two loops' instance SET is unchanged and that the inner loop's
    extent does not depend on the outer variable -- and refuses otherwise.

    On a machine whose hardware resolves hazards, no correctness check can refute a reordering at all;
    that is a property of the archetype, not of this primitive, and it is why the machine model carries
    ``hazard_resolution`` and why a reordering there is graded on measured occupancy instead.
    """
    outer = _loop_at(kernel.body, cursor.loops)
    inner = next((s for s in outer.body if isinstance(s, Loop)), None)
    if inner is None:
        raise NotApplicable(f"loop {outer.var!r} contains no loop to swap with")
    if len(outer.body) != 1:
        raise NotApplicable(
            f"loop {outer.var!r} holds {len(outer.body)} statements, so the nest is not perfect; "
            "fission it before reordering"
        )
    swapped = Loop(inner.var, inner.extent, (Loop(outer.var, outer.extent, inner.body),))
    candidate = _replace(kernel, body=_replace_loop(kernel.body, cursor.loops, swapped))
    _require_same_instances(kernel, candidate, what=f"reorder({outer.var!r} <-> {inner.var!r})")
    return candidate


@_obligation("bounded_check")
def fuse(kernel: Kernel, cursor: Cursor) -> Kernel:
    """Fuse the loop the cursor names with the loop immediately following it in the same body.

    Refuses on differing extents, and on a fusion that would move a read of a tensor before the write
    that produces it -- the one dependence this level can see exactly, from the access kinds the kernel
    arguments already declare.
    """
    body, index = _resolve(kernel.body, cursor)
    if index is None:
        raise NotApplicable("fuse needs a statement index naming the first of the two loops")
    if index + 1 >= len(body):
        raise NotApplicable("there is no following statement to fuse with")
    first, second = body[index], body[index + 1]
    if not isinstance(first, Loop) or not isinstance(second, Loop):
        raise NotApplicable("fuse applies to two adjacent loops")
    if first.extent != second.extent:
        raise NotApplicable(f"extents differ: {first.extent} and {second.extent}")
    renamed = _map_calls(second.body, lambda c: _subst_call(c, second.var, Var(first.var)))
    fused = Loop(first.var, first.extent, tuple(first.body) + tuple(renamed))
    new_body = body[:index] + (fused,) + body[index + 2 :]
    candidate = _replace(kernel, body=_rebuild(kernel.body, cursor.loops, new_body))
    _require_same_instances(kernel, candidate, what=f"fuse({first.var!r}, {second.var!r})")
    return candidate


@_obligation("bounded_check")
def place(kernel: Kernel, cursor: Cursor, *, unit: str, machine) -> Kernel:
    """Issue the call the cursor names on ``unit``.

    Checkable exactly and cheaply: the machine either has that unit or it does not. A unit it does not
    have is a DEFINITE failure rather than a miss -- the caller was reasoning about a machine that is not
    this one, so trying the next move would carry that mistake forward.
    """
    body, index = _resolve(kernel.body, cursor)
    if index is None:
        raise NotApplicable("place needs a statement index, not a loop")
    target = body[index]
    if not isinstance(target, Call):
        raise NotApplicable(f"place applies to a call, not to loop {target.var!r}")
    try:
        machine.unit(unit)
    except Exception as exc:  # noqa: BLE001 - the machine's own refusal, re-raised as definite
        raise PrimitiveError(f"{getattr(machine, 'target', '?')}: {exc}") from exc
    placed = _replace(target, unit=unit)
    return _replace(kernel, body=_rebuild(kernel.body, cursor.loops, body[:index] + (placed,) + body[index + 1 :]))


@_obligation("bounded_check")
def pipeline(kernel: Kernel, producer: Cursor, consumer: Cursor, *, token: str, machine) -> Kernel:
    """Make the producer asynchronous and have the consumer await it.

    BOTH HALVES OR NEITHER. A producer left with a token nothing awaits is, on a machine where the
    compiler separates hazards, work whose completion nothing established -- a wrong answer at full
    speed. Doing this as two primitives would make that state reachable one call at a time, so it is one.

    What is checked here: the two are distinct statements in one body, the consumer follows the
    producer, the token is fresh, and the machine says the producer's completion is something a consumer
    can wait FOR. An instruction whose result is visible to the next one has no completion to await, and
    pipelining it would record a dependence the hardware never has -- which is not wrong, but is a
    constraint on every later reordering bought for nothing.

    Whether the overlap this admits is WORTH anything is a measurement, not a check. On a machine whose
    hardware tracks dependencies, no correctness gate can tell you: the falsifier there is occupancy.
    """
    body, index = _resolve(kernel.body, producer)
    _, consumer_index = _resolve(kernel.body, consumer)
    if index is None or consumer_index is None:
        raise NotApplicable("pipeline needs a producer and a consumer statement index")
    if producer.loops != consumer.loops:
        raise NotApplicable("pipeline joins two statements in one body; cross-loop edges are not this move")
    if consumer_index <= index:
        raise NotApplicable(f"the consumer at {consumer_index} does not follow the producer at {index}")
    src, dst = body[index], body[consumer_index]
    if not isinstance(src, Call) or not isinstance(dst, Call):
        raise NotApplicable("pipeline joins two calls")
    if any(c.produces == token for c in _all_calls(kernel.body)):
        raise PrimitiveError(f"token {token!r} is already produced in this kernel")
    if src.unit is None:
        raise NotApplicable(f"{src.instr} is not placed on a unit, so there is nothing to overlap with")
    cost = machine.latency(src.instr, src.unit)
    if cost is None:
        raise NotApplicable(
            f"{getattr(machine, 'target', '?')} declares no cost for {src.instr}@{src.unit}, so whether "
            "it completes asynchronously is unknown -- and a token invented here would assert it does"
        )
    if cost.completion == "immediate":
        raise NotApplicable(
            f"{src.instr}@{src.unit} completes immediately: its result is visible to the next "
            "instruction, so there is no completion for a consumer to wait for"
        )
    if dst.unit is not None and not machine.can_overlap(src.unit, dst.unit):
        # Same physical resource: the two cannot be in flight together whatever the tokens say, so the
        # edge would buy no overlap and cost every later reordering a constraint. Asking the MACHINE
        # rather than comparing unit names is the point -- two separately named datapaths that declare
        # one `executes_on` are one resource, and only the machine knows that.
        raise NotApplicable(
            f"{src.unit} and {dst.unit} execute on {machine.resource_of(src.unit)!r}, so they cannot be "
            "in flight at once; pipelining them records a dependence and buys no overlap"
        )
    new_body = (
        body[:index]
        + (_replace(src, produces=token),)
        + body[index + 1 : consumer_index]
        + (_replace(dst, awaits=tuple(dst.awaits) + (token,)),)
        + body[consumer_index + 1 :]
    )
    return _replace(kernel, body=_rebuild(kernel.body, producer.loops, new_body))


def _all_calls(body) -> list[Call]:
    out: list[Call] = []
    for stmt in body:
        out.extend(_all_calls(stmt.body) if isinstance(stmt, Loop) else [stmt])
    return out


@_obligation("bounded_check")
def bind_bank(
    kernel: Kernel,
    cursor: Cursor,
    *,
    memory: str,
    row: int,
    rows: int,
    bank: int | None = None,
    writes: bool = False,
    machine,
) -> Kernel:
    """Stage one of a call's operands into ``rows`` rows of an on-chip memory, optionally in one bank.

    Checked against the machine, not against a convention: the memory must exist, the rows must fit the
    capacity it declares, and a named bank must be one the memory has. A bank named against a memory
    whose bank count was never derived is refused rather than accepted -- choosing a bank out of an
    unknown count is a guess wearing a placement's clothes.

    What this does NOT check is whether the placement collides with another call's, or starves a
    contending engine: those are properties of the whole kernel, so they belong to
    :func:`~merlin.sched.check.placement.check_placement` and are asked once of the finished schedule
    rather than once per move.
    """
    body, index = _resolve(kernel.body, cursor)
    if index is None:
        raise NotApplicable("bind_bank needs a statement index, not a loop")
    target = body[index]
    if not isinstance(target, Call):
        raise NotApplicable(f"bind_bank applies to a call, not to loop {target.var!r}")
    if rows < 1 or row < 0:
        raise NotApplicable(f"a stage of {rows} rows at {row} occupies nothing")
    try:
        store = machine.memory(memory)
    except Exception as exc:  # noqa: BLE001 - the machine's own refusal, re-raised as definite
        raise PrimitiveError(f"{getattr(machine, 'target', '?')}: {exc}") from exc
    if store.rows is not None and row + rows > store.rows:
        raise NotApplicable(f"{memory}[{row}:{row + rows}] runs past the {store.rows} rows that store has")
    if bank is not None:
        if store.banks is None:
            raise NotApplicable(f"{memory} declares no bank count, so bank {bank} is a guess rather than a placement")
        if not 0 <= bank < store.banks:
            raise NotApplicable(f"{memory} has {store.banks} bank(s); {bank} is not one of them")
    staged = _replace(
        target, stages=tuple(target.stages) + (Stage(memory=memory, row=row, rows=rows, bank=bank, writes=writes),)
    )
    return _replace(kernel, body=_rebuild(kernel.body, cursor.loops, body[:index] + (staged,) + body[index + 1 :]))


@_obligation("bounded_check")
def stage_mem(
    kernel: Kernel, cursor: Cursor, *, move: Call, memory: str, row: int, rows: int, bank: int | None = None, machine
) -> Kernel:
    """Bring data into an on-chip memory before the call at ``cursor``, and record that it reads it there.

    The MOVE is the caller's: only they know which instruction their target uses to move bytes and what
    its operands are. What this owns is the transformation -- insert the move ahead of its consumer,
    record that the move WRITES those rows and the consumer READS them, and check the placement against
    the machine before either annotation exists.

    That division is deliberate. A primitive that also chose the instruction would be choosing it from a
    per-target table, which is the thing a generic vocabulary must not contain; a caller that annotated
    by hand could leave a consumer reading rows nothing wrote.

    WHAT THIS DOES AND DOES NOT MODEL. It records WHERE the data is and WHO touches it, which is what
    makes the capacity, the bank and the write-over-a-live-operand checks possible. It does not rewrite
    the consumer's operands to address the staged copy: on these targets the on-chip address is part of
    the instruction's own encoding, so that belongs to the per-target speller, not here.
    """
    body, index = _resolve(kernel.body, cursor)
    if index is None:
        raise NotApplicable("stage_mem needs the statement index of the call that will read the data")
    consumer = body[index]
    if not isinstance(consumer, Call):
        raise NotApplicable(f"stage_mem stages for a call, not for loop {consumer.var!r}")
    if not isinstance(move, Call):
        raise PrimitiveError("stage_mem needs the move as a Call the caller built from its own ISA")
    if rows < 1 or row < 0:
        raise NotApplicable(f"a stage of {rows} rows at {row} brings nothing on chip")
    try:
        store = machine.memory(memory)
    except Exception as exc:  # noqa: BLE001 - the machine's own refusal, re-raised as definite
        raise PrimitiveError(f"{getattr(machine, 'target', '?')}: {exc}") from exc
    if store.rows is not None and row + rows > store.rows:
        raise NotApplicable(f"{memory}[{row}:{row + rows}] runs past the {store.rows} rows that store has")
    if bank is not None:
        if store.banks is None:
            raise NotApplicable(f"{memory} declares no bank count, so bank {bank} is a guess, not a placement")
        if not 0 <= bank < store.banks:
            raise NotApplicable(f"{memory} has {store.banks} bank(s); {bank} is not one of them")
    for existing in consumer.stages:
        if existing.memory == memory and existing.row < row + rows and row < existing.row + existing.rows:
            raise NotApplicable(
                f"{consumer.instr} already stages {memory}[{existing.row}:{existing.row + existing.rows}], "
                f"which overlaps {memory}[{row}:{row + rows}]"
            )
    written = _replace(
        move, stages=tuple(move.stages) + (Stage(memory=memory, row=row, rows=rows, bank=bank, writes=True),)
    )
    read = _replace(consumer, stages=tuple(consumer.stages) + (Stage(memory=memory, row=row, rows=rows, bank=bank),))
    return _replace(
        kernel, body=_rebuild(kernel.body, cursor.loops, body[:index] + (written, read) + body[index + 1 :])
    )


# -- shared obligations ------------------------------------------------------------------------------


def _require_well_formed(out: Kernel, what: str) -> None:
    """Refuse a rewrite that left the kernel structurally ill-formed.

    Needed by every primitive that DUPLICATES a body. A call producing a completion token becomes two
    static producers of one token, which `_require_same_instances` cannot see -- it compares instruction
    and operand values only, by design, since order is what a transformation changes. Measured on both
    `unroll` and `peel`: each returned a kernel whose own structural check rejected it, without a word.
    """
    from merlin.sched.ir.kernel import check_structure

    errors = check_structure(out)
    if errors:
        raise NotApplicable(f"{what} leaves the kernel ill-formed: {errors[0]}")


def _require_same_instances(before: Kernel, after: Kernel, *, what: str) -> None:
    """Refuse a rewrite that changed WHICH instances run, as opposed to when.

    The multiset of (instruction, concrete operand values) is what a loop transformation must preserve;
    a rewrite that changes it is not a reordering at all. Order is deliberately not compared -- changing
    it is the whole point -- so this catches a dropped, duplicated or mis-indexed instance, which is the
    failure a hand-written index rewrite actually produces.
    """
    try:
        lhs = sorted(_instance_keys(before))
        rhs = sorted(_instance_keys(after))
    except Exception as exc:  # noqa: BLE001 - an unevaluable operand cannot be compared, so refuse
        raise PrimitiveError(f"{what}: the rewrite could not be checked ({exc})") from exc
    if lhs != rhs:
        missing = sorted(set(lhs) - set(rhs))[:3]
        added = sorted(set(rhs) - set(lhs))[:3]
        raise PrimitiveError(
            f"{what}: the rewrite changed which instances run, not just their order "
            f"({len(lhs)} -> {len(rhs)}; missing {missing}, new {added})"
        )


def _instance_keys(kernel: Kernel) -> list[tuple]:
    from merlin.sched.ir.kernel import ConcretePtr, Ptr

    out = []
    for call, env in instances(kernel):
        values = []
        for name, value in call.args:
            if isinstance(value, Ptr):
                values.append((name, "ptr", value.tensor, evaluate(value.offset, env)))
            elif isinstance(value, Expr):
                values.append((name, "int", evaluate(value, env)))
            elif isinstance(value, ConcretePtr):  # pragma: no cover - concretized kernels are not rewritten
                values.append((name, "ptr", value.tensor, value.offset))
            else:
                values.append((name, "raw", repr(value)))
        out.append((call.instr, tuple(values)))
    return out


def _bound_vars(body: tuple) -> set[str]:
    out: set[str] = set()
    for stmt in body:
        if isinstance(stmt, Loop):
            out.add(stmt.var)
            out |= _bound_vars(stmt.body)
    return out


def _replace_loop(body: tuple, path: tuple[str, ...], new: Loop) -> tuple:
    if not path:
        raise NotApplicable("the cursor names no loop to replace")
    head, rest = path[0], path[1:]
    target = _one_loop(body, head)
    return tuple(
        (new if not rest else _replace(stmt, body=_replace_loop(stmt.body, rest, new))) if stmt is target else stmt
        for stmt in body
    )


def _splice(body: tuple, path: tuple[str, ...], stmts: tuple) -> tuple:
    if not path:
        raise NotApplicable("the cursor names no loop to splice")
    head, rest = path[0], path[1:]
    target = _one_loop(body, head)
    out = []
    for stmt in body:
        if stmt is not target:
            out.append(stmt)
        elif rest:
            out.append(_replace(stmt, body=_splice(stmt.body, rest, stmts)))
        else:
            out.extend(stmts)
    return tuple(out)


@_obligation("bounded_check")
def hoist_config(kernel: Kernel, cursor: Cursor) -> Kernel:
    """Lift a configuration call out of the loop that encloses it.

    This is the transformation the whole ``sets``/``assumes`` vocabulary exists to enable, and it is a
    real lever rather than a tidy-up: a configuration re-issued every iteration costs a dispatch slot
    every iteration, and on a decoupled machine those slots are the command port every other instruction
    also has to issue through.

    Refused unless all three hold, each for a reason that is a wrong answer and not a preference:

    * the call sets configuration and nothing else. Hoisting a call that also computes, moves data or
      produces a token would run it once where the kernel runs it N times.
    * every value it sets is loop-invariant. A value mentioning the loop variable is a DIFFERENT
      configuration each iteration, and hoisting it keeps only the first.
    * nothing else in the loop body sets the same keys. Otherwise the last writer inside the loop is
      what is live at the end of each iteration, and moving this one out changes which value later
      iterations see.

    The cursor names the loop to lift out of via its path, so ``Cursor(loops=("i",), index=0)`` lifts
    the first statement of loop ``i`` to just before the loop.
    """
    if not cursor.loops:
        raise NotApplicable("hoist_config needs a cursor inside a loop; this one is at the top level")
    body, index = _resolve(kernel.body, cursor)
    if index is None:
        raise NotApplicable("hoist_config needs a statement index, not a loop")
    target = body[index]
    if not isinstance(target, Call):
        raise NotApplicable(f"hoist_config applies to a call, not to loop {target.var!r}")
    if not target.sets:
        raise NotApplicable(f"{target.instr} sets no configuration, so there is nothing to hoist")
    if target.args or target.stages or target.produces or target.awaits or target.assumes:
        raise NotApplicable(
            f"{target.instr} does more than configure (it has operands, staging, a token or an "
            "assumption), so running it once where the kernel runs it many times would change what it "
            "computes"
        )

    var = cursor.loops[-1]
    for key, value in target.sets:
        if isinstance(value, Expr) and var in free_vars(value):
            raise NotApplicable(
                f"{target.instr} sets {key} from {render(value)}, which varies with {var} -- that is a "
                f"different configuration each iteration, and hoisting it would keep only the first"
            )
    keys = {k for k, _ in target.sets}
    for other in body[:index] + body[index + 1 :]:
        clash = keys & {k for k, _ in other.sets} if isinstance(other, Call) else set()
        if clash:
            raise NotApplicable(
                f"{getattr(other, 'instr', '?')} also sets {sorted(clash)} inside {var}, so it and not "
                f"{target.instr} is what is live at the end of each iteration"
            )

    inner = body[:index] + body[index + 1 :]
    if not inner:
        raise NotApplicable(f"{target.instr} is the only statement in {var}; hoisting it would empty the loop")
    lifted = _rebuild(kernel.body, cursor.loops, inner)
    return _replace(kernel, body=_insert_before_loop(lifted, cursor.loops, target))


def _insert_before_loop(body: tuple, path: tuple[str, ...], stmt) -> tuple:
    """``body`` with ``stmt`` placed immediately before the loop named by ``path``."""
    if len(path) == 1:
        out = []
        for s in body:
            if isinstance(s, Loop) and s.var == path[0]:
                out.append(stmt)
            out.append(s)
        return tuple(out)
    return tuple(
        _replace(s, body=_insert_before_loop(s.body, path[1:], stmt)) if isinstance(s, Loop) and s.var == path[0] else s
        for s in body
    )


def _prune_unreachable_selects(body: tuple, var: str, extent: int) -> tuple:
    """Decide selects on ``var`` whose tested value no iteration of a loop of ``extent`` can take.

    Peeling renumbers a loop, and a last-tile remainder written as ``select(k == N-1, tail, full)``
    survives into a middle loop that no longer reaches ``N-1``. The select is then dead -- always its
    other branch -- and leaving it costs nothing at run time but changes the kernel's canonical text,
    which is its identity. A schedule and the same schedule written directly must digest the same, so
    this is not cosmetic.
    """
    from merlin.sched.ir.expr import Select

    def prune(value):
        if isinstance(value, Ptr):
            return Ptr(value.tensor, prune(value.offset))
        if isinstance(value, Select):
            then, other = prune(value.then), prune(value.other)
            if value.var == var and not 0 <= value.equals < extent:
                return other
            return select(Var(value.var), value.equals, then, other)
        if isinstance(value, Sum):
            return add(*(prune(t) for t in value.terms))
        if isinstance(value, Scaled):
            return mul(prune(value.expr), value.factor)
        return value

    return _map_calls(
        body,
        lambda c: _replace(
            c,
            args=tuple((k, prune(v)) for k, v in c.args),
            sets=tuple((k, prune(v)) for k, v in c.sets),
            assumes=tuple((k, prune(v)) for k, v in c.assumes),
        ),
    )


@_obligation("by_construction")
def peel(kernel: Kernel, cursor: Cursor, *, at: str = "front") -> Kernel:
    """Split the first (or last) iteration off the loop the cursor names.

    The move every prologue and epilogue needs, and the one ``divide_loop`` refuses to invent: a
    reduction whose first iteration initialises an accumulator and whose last reads it out cannot be
    written as a uniform loop body, so the two ends are peeled and specialised. The same shape is a
    pipeline's fill and drain, and a warp-specialised body's first pass.

    Correct by construction: the peeled copy runs the iteration it was given and the remaining loop runs
    the others, so the multiset of instances is unchanged -- which is checked, not asserted. Peeling the
    front renumbers the remaining loop from zero and shifts every use of the variable by one, so
    ``for k in 0..N`` becomes ``B(0)`` then ``for k in 0..N-1`` of ``B(k + 1)``.

    An extent of 1 refuses: there is nothing left after the peel, and returning the bare body would
    quietly delete a loop the caller still holds a cursor into.
    """
    if at not in ("front", "back"):
        raise PrimitiveError(f"peel at {at!r}: expected 'front' or 'back'")
    target = _loop_at(kernel.body, cursor.loops)
    if target.extent < 2:
        raise NotApplicable(
            f"loop {target.var!r} has extent {target.extent}; peeling it would leave nothing behind. "
            "Unroll it instead -- that is the move that removes a loop."
        )
    rest = target.extent - 1
    if at == "front":
        copy = _map_calls(target.body, lambda c: _subst_call(c, target.var, as_expr(0)))
        shifted = _map_calls(target.body, lambda c: _subst_call(c, target.var, add(Var(target.var), 1)))
        stmts = copy + (Loop(target.var, rest, _prune_unreachable_selects(shifted, target.var, rest)),)
    else:
        copy = _map_calls(target.body, lambda c: _subst_call(c, target.var, as_expr(rest)))
        stmts = (Loop(target.var, rest, _prune_unreachable_selects(target.body, target.var, rest)),) + copy
    out = _replace(kernel, body=_splice(kernel.body, cursor.loops, stmts))
    _require_same_instances(kernel, out, what=f"peel {at} of {target.var!r}")
    _require_well_formed(out, f"peeling {target.var!r}")
    return out


@_obligation("bounded_check")
def specialize(kernel: Kernel, cursor: Cursor, **operands) -> Kernel:
    """Give one call different operands from the loop-uniform body it was peeled out of.

    The move that makes peeling useful. A reduction's body is uniform except at its ends: the first
    iteration initialises the accumulator and the last drains it, and neither can be written as an index
    expression because the difference is an operand being PRESENT or ABSENT, not a value that varies.
    So the ends are peeled and then specialised here.

    Not correct by construction -- it deliberately changes what one instance computes, which is the
    point -- so it is bounded instead: every operand named must already exist on that call, and the
    result must still be structurally well formed. The first catches the failure this actually has,
    which is a caller specialising an operand that instruction does not take (a typo, or a cursor that
    landed on the wrong statement) and silently growing an operand list the emitter will not recognise.

    What it does NOT check is whether the new operand makes SENSE for that instruction: that is the
    target's own ISA semantics, asked by the instruction set's own check, and guessing at it here would
    put a per-target opinion inside a generic vocabulary.
    """
    from merlin.sched.ir.kernel import _normalize, check_structure

    body, index = _resolve(kernel.body, cursor)
    if index is None:
        raise NotApplicable("specialize needs a statement index, not a loop")
    target = body[index]
    if not isinstance(target, Call):
        raise NotApplicable(f"specialize applies to a call, not to loop {target.var!r}")
    if not operands:
        raise NotApplicable("specialize was given no operands to change")
    have = {k for k, _ in target.args}
    unknown = sorted(set(operands) - have)
    if unknown:
        raise PrimitiveError(
            f"{target.instr} has no operand(s) {unknown}; it takes {sorted(have)}. Either the cursor "
            "landed on a different instruction than the caller meant, or the operand is misspelled -- "
            "adding it would grow an operand list the emitter does not recognise."
        )
    updated = dict(operands)
    changed = _replace(
        target,
        args=tuple((k, _normalize(updated[k]) if k in updated else v) for k, v in target.args),
    )
    out = _replace(kernel, body=_rebuild(kernel.body, cursor.loops, body[:index] + (changed,) + body[index + 1 :]))
    errors = check_structure(out)
    if errors:
        raise PrimitiveError(f"specialising {target.instr} left the kernel ill-formed: {errors[0]}")
    return out


# -- macro instructions ------------------------------------------------------------------------------


def _statement_text(statements: tuple) -> str:
    """The canonical text of a statement sequence -- the IR's OWN notion of sameness.

    Comparing by the printed form rather than by a hand-written structural walk is deliberate: the
    digest that identifies a kernel is the sha256 of exactly this text, so two subtrees that print the
    same ARE the same schedule by the only definition this IR has. A bespoke comparison would be a
    second definition, free to disagree with the first.
    """
    lines: list[str] = []
    _print_body(statements, 0, lines)
    return "\n".join(lines)


def _same_up_to_loop_names(here: tuple, there: tuple, bind: dict[str, str]) -> str | None:
    """``None`` when the two sequences are the same up to a consistent loop-variable renaming, else why.

    Alpha-equivalence rather than textual equality, because the nest a schedule arrives at was named by
    whatever produced it -- `divide_loop` mints `i_outer`/`i_inner` -- while a macro's declared expansion
    is written once with names of its author's choosing. Requiring those to coincide would make the
    substitution depend on a spelling neither side controls.

    The renaming has to be a BIJECTION, checked in both directions. A mapping that let two of the
    expansion's variables land on one of the kernel's would accept a nest that collapses two loops the
    macro keeps apart, which is a different computation.
    """
    if len(here) != len(there):
        return f"{len(here)} statement(s) where the expansion has {len(there)}"
    for mine, theirs in zip(here, there):
        if isinstance(mine, Loop) != isinstance(theirs, Loop):
            return "a loop where the expansion has a call, or the reverse"
        if isinstance(mine, Loop):
            if mine.extent != theirs.extent:
                return f"loop extent {mine.extent} where the expansion has {theirs.extent}"
            bound = bind.get(theirs.var)
            if bound is not None and bound != mine.var:
                return f"the expansion's {theirs.var!r} is already matched to {bound!r}, not {mine.var!r}"
            if mine.var in bind.values() and bound is None:
                return f"two of the expansion's loops would both match {mine.var!r}"
            bind[theirs.var] = mine.var
            why = _same_up_to_loop_names(mine.body, theirs.body, bind)
            if why is not None:
                return why
            continue
        renamed = theirs
        for src, dst in bind.items():
            if src != dst:
                renamed = _subst_call(renamed, src, Var(dst))
        if _statement_text((mine,)) != _statement_text((renamed,)):
            return (
                f"{_statement_text((mine,)).strip()!r} where the expansion has {_statement_text((renamed,)).strip()!r}"
            )
    return None


@_obligation("bounded_check")
def replace(kernel: Kernel, cursor: Cursor, *, expansion, call: Call) -> Kernel:
    """Substitute the statements the cursor names with ONE macro-instruction call.

    THE POINT. A target's FSM macro -- a single instruction that runs a whole tiled loop nest in
    hardware -- is otherwise a thing a recipe has to emit by hand, which is how a per-target recipe gets
    written. Defining the macro as the nest it is equivalent to, and substituting that nest wherever it
    appears, is what lets one generic recipe target a machine that has the macro and a machine that does
    not: the fine-grained nest is the only level at which a schedule can beat a vendor library, and the
    macro is what buys offload completeness. The vocabulary stays fine-grained and the macro arrives by
    substitution, rather than the macro being the vocabulary.

    THE OBLIGATION, and why it is bounded rather than by construction. Whether a hardware macro computes
    what a loop nest computes is a claim about the SILICON, and nothing here can discharge it -- the
    kernel IR has no semantics for an instruction, only its operands. So this does not attempt to prove
    the equivalence. It checks the weaker thing it can: that the nest actually present is the one the
    caller declared the macro to be equal to, up to loop-variable renaming. The equivalence itself rides
    on `expansion`, which is the caller's assertion and belongs with the target's own definition of its
    instruction -- where an RTL gate can eventually check it.

    That is a real division of labour rather than a dodge. Getting the substitution wrong silently
    (matching a nest that differs from the declared one in a stride, a configuration, or one statement
    of the body) is a compiler bug this level CAN catch, and it is the likelier of the two failures.
    """
    expected = tuple(expansion) if isinstance(expansion, (tuple, list)) else (expansion,)
    if not expected:
        raise PrimitiveError("replace needs the macro's declared expansion; an empty one matches anything")
    if not isinstance(call, Call):
        raise PrimitiveError(f"replace substitutes a Call, not {type(call).__name__}")
    body, index = _resolve(kernel.body, cursor)
    if index is None:
        raise NotApplicable("replace needs a statement index naming the first statement of the nest")
    found = body[index : index + len(expected)]
    if len(found) < len(expected):
        raise NotApplicable(
            f"the expansion is {len(expected)} statement(s) and only {len(found)} remain at this position"
        )
    from merlin.sched.ir.kernel import check_structure

    why = _same_up_to_loop_names(found, expected, {})
    if why is not None:
        raise NotApplicable(f"this is not the declared expansion: {why}")
    new_body = body[:index] + (call,) + body[index + len(expected) :]
    out = _replace(kernel, body=_rebuild(kernel.body, cursor.loops, new_body))
    errors = check_structure(out)
    if errors:
        raise PrimitiveError(f"substituting {call.instr} left the kernel ill-formed: {errors[0]}")
    return out


def _enclosing_loop(kernel: Kernel, cursor: Cursor, var: str) -> Loop:
    """The loop named ``var`` among the ones the cursor is nested inside, or raise.

    A rotation's phase has to come from a loop that ENCLOSES the call, because the phase is read by the
    call's own operand. A variable bound anywhere else is out of scope there, and substituting it would
    build a kernel whose own structural check rejects it -- after the caller had been handed it.
    """
    if var not in cursor.loops:
        raise NotApplicable(
            f"{var!r} does not enclose this call, which sits inside {list(cursor.loops)}; a phase has to "
            "come from a loop the call is nested in, or the operand that reads it is out of scope"
        )
    return _loop_at(kernel.body, cursor.loops[: cursor.loops.index(var) + 1])


@_obligation("bounded_check")
def multi_buffer(
    kernel: Kernel,
    cursor: Cursor,
    *,
    memory: str,
    depth: int,
    row: int | None = None,
    machine=None,
    rotate: str | None = None,
    over: str | None = None,
    advance: int | None = None,
) -> Kernel:
    """Give the staging in ``memory`` on the call the cursor names ``depth`` rotating copies.

    WHAT THIS BUYS. One copy means a producer and its consumer cannot both be in flight: the fill of
    the next tile waits for the read of this one. With n copies they overlap, which on these machines is
    where the performance is -- and two targets here arrive at the same shape independently, a
    K-tile-parity toggle between two scratchpad halves on one and a weight-slot ping-pong on the other.
    That is the two-slice rule satisfied, which is why this is core vocabulary rather than an extension.

    TWO HALVES, AND ONLY THE SECOND REACHES EMITTED CODE. The RESERVATION -- n copies are occupied, so
    the capacity and overlap checks price them -- is what the ``depth`` field records, and it is the
    half that silently returns wrong data when it is wrong. It is also, on its own, invisible to the
    compiler: nothing downstream of the kernel IR reads ``Stage``, so a schedule that reserves n copies
    and a schedule that reserves one emit byte-identical C. The second half is the ROTATION: which copy
    a given iteration addresses. That is an operand, and operands are the one thing that does reach C.

    HOW THE ROTATION IS SAID, and why it is the caller who says it. Pass all three of:

    ``rotate``   the name of the operand that carries this call's on-chip base address. Nothing here
                 can derive which operand that is: the kernel IR types an operand ``int``, ``flag``,
                 ``ptr`` or ``float``, and an on-chip address is an ``int`` like a stride or a tile
                 count. The caller built the call from its target's own instruction set and knows.
    ``over``     the enclosing loop variable that supplies the PHASE. Its extent must be exactly
                 ``depth``, because the phase is ``iteration mod depth`` and this IR's operand language
                 has no modulus -- by design, since anything richer than sums and constant multiples
                 belongs in a loop transformation. Splitting the rotation loop by ``depth`` is how that
                 modulus is written: ``divide_loop(k, depth)`` gives ``k = ko*depth + ki``, so ``ki`` IS
                 ``k mod depth``, exactly rather than approximately. A parity toggle is this with
                 ``depth=2``; a ping-pong between two slots is the same shape. Neither idiom is baked
                 in -- both are instances of it.
    ``advance``  how far the operand moves per copy, IN THE OPERAND'S OWN UNIT. Required, not defaulted
                 to the staging's row count: a ``Stage`` counts rows of the machine's memory, while the
                 operand counts whatever that instruction's encoding counts, and the two coincide on
                 some targets and not others. A default would be a guess that emits a plausible address
                 into the wrong copy, which is the silent-wrong-data failure with extra steps.

    The operand becomes ``old + over*advance``. That is exact for every ``over`` in ``0..depth``, and it
    is written in the IR's own expression language, so it prints, digests, evaluates and renders to C
    like any other operand -- which is the whole point: with ``rotate`` the emitted C differs, and
    without it, provably, it does not.

    Omitting the three is still legal and still useful -- the reservation is what the capacity check
    needs, and a recipe that owns its own address arithmetic has already done the rotation by hand --
    but it is a reservation, not a transformation, and a caller who expects the program to change has
    to say how.

    The obligation is bounded because it cannot be by construction: enlarging a reservation is exactly
    the operation that can collide with a neighbour or run off the end of a store, so the copies are
    checked to fit before the change is returned rather than after it has been believed.
    """
    given = {"rotate": rotate, "over": over, "advance": advance}
    named = sorted(k for k, v in given.items() if v is not None)
    if named and len(named) != 3:
        raise PrimitiveError(
            f"a rotation needs rotate=, over= and advance= together; got {named}. Two of the three "
            "describe an address that is never written and one describes a phase nothing reads, so a "
            "partial rotation would leave the schedule reserving copies it still cannot address."
        )
    if depth < 2:
        raise PrimitiveError(f"multi_buffer needs at least two copies; {depth} is a single buffer")
    body, index = _resolve(kernel.body, cursor)
    if index is None:
        raise NotApplicable("multi_buffer needs a statement index naming the call to re-stage")
    target = body[index]
    if not isinstance(target, Call):
        raise NotApplicable("multi_buffer applies to a call, not a loop")
    matching = [g for g in target.stages if g.memory == memory and (row is None or g.row == row)]
    if not matching:
        raise NotApplicable(
            f"{target.instr} stages nothing in {memory!r}"
            + (f" at row {row}" if row is not None else "")
            + f"; it stages {[(g.memory, g.row) for g in target.stages] or 'nothing'}"
        )
    if len(matching) > 1:
        # Staging two operands in one scratchpad is the common case, not an error -- both a matmul's
        # operands live there. So the ambiguity is resolved by naming the row rather than refused
        # outright; what stays refused is GUESSING, because deepening the wrong one silently moves a
        # different buffer's reservation and the schedule remains well-formed either way.
        raise PrimitiveError(
            f"{target.instr} has {len(matching)} stagings in {memory!r}, at rows "
            f"{sorted(g.row for g in matching)}. Pass row= to say which one."
        )
    staged = matching[0]
    if staged.depth != 1:
        raise NotApplicable(f"that staging already has {staged.depth} copies")
    deeper = _replace(staged, depth=depth)
    if machine is not None:
        store = next((m for m in machine.memories if m.name == memory), None)
        if store is None:
            raise PrimitiveError(f"{machine.target} has no memory named {memory!r}")
        if store.rows is not None and deeper.row + deeper.rows * depth > store.rows:
            raise NotApplicable(
                f"{depth} copies of {deeper.rows} rows from {deeper.row} need "
                f"{deeper.row + deeper.rows * depth} rows of {memory}, which has {store.rows}"
            )
    for other in target.stages:
        if other is staged or other.memory != memory:
            continue
        if deeper.row < other.row + other.rows * other.depth and other.row < deeper.row + deeper.rows * depth:
            raise NotApplicable(
                f"{depth} copies would reach {deeper.row + deeper.rows * depth}, over a staging this "
                f"call already holds at {other.row}:{other.row + other.rows * other.depth}"
            )
    restaged = _replace(target, stages=tuple(deeper if g is staged else g for g in target.stages))
    if rotate is not None:
        restaged = _rotate_operand(kernel, cursor, restaged, rotate=rotate, over=over, advance=advance, depth=depth)
    new_body = body[:index] + (restaged,) + body[index + 1 :]
    out = _replace(kernel, body=_rebuild(kernel.body, cursor.loops, new_body))
    if rotate is not None:
        _require_well_formed(out, f"rotating {restaged.instr}.{rotate} over {over!r}")
    return out


def _rotate_operand(
    kernel: Kernel, cursor: Cursor, target: Call, *, rotate: str, over: str, advance: int, depth: int
) -> Call:
    """``target`` with operand ``rotate`` advanced by ``over * advance``, or raise.

    Everything refused here is refused because the alternative emits a PLAUSIBLE address into the wrong
    copy. An operand this call does not take, a phase loop whose extent is not the depth, an advance of
    zero: each of the three leaves a schedule that is well formed, passes the capacity check, and reads
    the same copy every iteration -- which is the defect this argument exists to close, reintroduced one
    level down.
    """
    have = {k for k, _ in target.args}
    if rotate not in have:
        raise PrimitiveError(
            f"{target.instr} has no operand {rotate!r}; it takes {sorted(have)}. Either the cursor landed "
            "on a different instruction than the caller meant, or the on-chip address is spelled under "
            "another name -- reserving the copies and then rotating nothing addresses one copy forever."
        )
    phase = _enclosing_loop(kernel, cursor, over)
    if phase.extent != depth:
        raise NotApplicable(
            f"{over!r} has extent {phase.extent}, not the depth {depth}, so it is not the phase: an "
            f"iteration would address copy {phase.extent - 1} of {depth}. Split it first -- "
            f"divide_loop by {depth} makes the inner variable exactly the iteration modulo {depth}, "
            "which is how this IR's operand language writes a modulus."
        )
    if advance == 0:
        raise PrimitiveError(
            "an advance of 0 moves every copy to the same address, which is a single buffer written "
            "as a rotation -- the emitted code would be identical to the unrotated schedule"
        )
    current = dict(target.args)[rotate]
    if not isinstance(current, Expr):
        kind = "a pointer into a tensor argument" if isinstance(current, Ptr) else type(current).__name__
        raise PrimitiveError(
            f"{target.instr}.{rotate} is {kind}, not an index expression. An on-chip base address is an "
            "index; a DRAM pointer rotating by a row count would walk off the tensor it names."
        )
    rotated = add(current, mul(Var(over), advance))
    return _replace(target, args=tuple((k, rotated if k == rotate else v) for k, v in target.args))
