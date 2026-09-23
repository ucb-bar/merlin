"""Emit a kernel as one C function: counted loops around the target's own instruction statements.

The function takes the kernel's tensor arguments as pointers, in declaration order, typed with the
target's C element types. A pointer operand renders as a byte offset from its tensor argument; ``NULL``
as ``NULL``; an index expression as a C expression; a float as the exact float32 hex literal (so the
value compiled is the value the kernel text names). The body ends with the target's drain statement.

WHAT REACHES C, AND WHAT CANNOT REACH IT HERE. A call is rendered by the target's own
``InstrDef.render_c``, which takes ONE thing: the operand texts, in operand order. That is the emitter's
entire channel to the target, and it has a consequence worth stating rather than discovering: a
scheduling annotation that is not an operand -- the ``unit`` a call issues on, the completion token it
``produces``, the tokens it ``awaits`` -- has no way through. It is not that this emitter forgot to read
them; there is nowhere to put them.

So it does not quietly emit the same C as if they were absent. Emitting identical bytes for a schedule
that declares asynchrony is the failure this module is written against: a search loop can then apply a
transformation, pass every check, and change nothing, and no measurement can tell that apart from a
transformation that did not help. Instead the emitter REFUSES and names what is missing, unless the
caller hands it a ``spell`` -- a per-target speller that renders each annotation as a C statement, or
returns an empty string to DECLARE that on this target the annotation costs no code. A declaration is
not the same as silence: the target said it, and the emitter can be pointed at the statement that says
so.

The staging annotations (``Call.stages``) are deliberately not in that list. A staging says WHERE an
operand sits on chip while a call runs; the address itself is already one of the call's operands, which
is the channel that works. What makes a staging reach C is therefore a rewrite of that operand --
:func:`merlin.sched.primitives.multi_buffer` with ``rotate=`` -- not a hook here.
"""

from __future__ import annotations

import numpy as np

from merlin.sched.ir.expr import Expr, render
from merlin.sched.ir.kernel import NULL, Call, Kernel, Loop, Ptr
from merlin.sched.isa import InstructionSet, IsaError

#: Bumped whenever the emitted text for an unchanged kernel changes.
EMITTER_VERSION = "mk_c_v1"

#: Per-call annotations that ``InstrDef.render_c`` cannot carry, and the speller hook for each.
#:
#: ``(field, hook, how to describe it)``. Kept as data rather than as three branches so that a field
#: added to ``Call`` later shows up as a missing row here instead of as a silent omission: the
#: completeness test walks ``Call``'s own fields against this table.
UNRENDERABLE = (
    ("unit", "issue_on", lambda c: f"is placed on unit {c.unit!r}"),
    ("produces", "on_produce", lambda c: f"produces completion token {c.produces!r}"),
    ("awaits", "on_await", lambda c: f"awaits {list(c.awaits)}"),
)


def _c_value(v) -> str:
    if isinstance(v, Ptr):
        return f"((uint8_t *){v.tensor} + {render(v.offset, c=True)})"
    if v is NULL:
        return "NULL"
    if isinstance(v, Expr):
        return render(v, c=True)
    if isinstance(v, float):
        return float(np.float32(v)).hex() + "f"
    raise IsaError(f"cannot render operand value {v!r}")


def _carried(call: Call) -> list[tuple[str, str, str]]:
    """The annotations this call carries that are not operands, as ``(field, hook, description)``."""
    out = []
    for field, hook, describe in UNRENDERABLE:
        value = getattr(call, field)
        if value:  # None, and the empty token tuple, both mean "not annotated"
            out.append((field, hook, describe(call)))
    return out


def _spelled(call: Call, field: str, hook: str, description: str, spell, target: str) -> str | None:
    """The C statement the target spells for one annotation, or ``None`` when it costs no code.

    Raises when there is no speller for it. The message says what the silence would have looked like,
    because that is the thing a reader has to be able to rule out: without a speller the emitted bytes
    for this kernel are identical to the bytes for the same kernel with the annotation removed, and
    nothing downstream can tell those apart.
    """
    fn = getattr(spell, hook, None) if spell is not None else None
    if fn is None:
        raise IsaError(
            f"{target}: {call.instr} {description}, and nothing can render that. The operand list is the "
            f"emitter's only channel to a target's C, so this annotation would reach no emitted code -- "
            f"the C for this kernel would be byte-identical to the C for the same kernel without it. "
            f"Pass spell= with a {hook}() that renders it, or that returns '' to declare that on this "
            f"target the annotation costs no code."
        )
    text = fn(call)
    if text is None or text == "":
        return None
    if not isinstance(text, str):
        raise IsaError(f"{target}: {hook}() returned {type(text).__name__}, not a C statement")
    return text


def emit_c_function(kernel: Kernel, iset: InstructionSet, *, symbol: str, spell=None) -> str:
    """The kernel as one C function.

    ``spell``, when given, is the target's speller for the annotations the operand list cannot carry.
    It is duck-typed on purpose -- ``merlin.sched.isa`` describes an instruction, not an agent, and
    putting these hooks there would make every target declare three methods to say "not applicable".
    Each hook takes the :class:`~merlin.sched.ir.kernel.Call` and returns a C statement, or ``''``/
    ``None`` to say the annotation needs none on this target:

    ``issue_on(call)``   the call is placed on ``call.unit``.
    ``on_await(call)``   the call waits for ``call.awaits`` before it issues; emitted BEFORE the call.
    ``on_produce(call)`` the call starts asynchronously under ``call.produces``; emitted AFTER it.

    A kernel carrying an annotation with no hook for it is refused rather than emitted without it.
    """
    if not symbol.isidentifier():
        raise IsaError(f"symbol {symbol!r} is not a C identifier")
    params = []
    for t in kernel.args:
        if t.dtype not in iset.c_types:
            raise IsaError(f"{iset.target}: no C type for {t.dtype}")
        params.append(f"{iset.c_types[t.dtype]} *{t.name}")
    lines = [f"static void {symbol}({', '.join(params)}) {{"]

    def body(stmts, depth):
        pad = "    " * depth
        for s in stmts:
            if isinstance(s, Loop):
                lines.append(f"{pad}for (int {s.var} = 0; {s.var} < {s.extent}; {s.var}++) {{")
                body(s.body, depth + 1)
                lines.append(f"{pad}}}")
                continue
            d = iset.instr(s.instr)
            names = tuple(k for k, _ in s.args)
            if names != d.operand_names():
                raise IsaError(f"{s.instr}: operands {names} != {d.operand_names()}")
            # Resolved BEFORE anything is appended, so a kernel the target cannot spell produces no
            # half-written function: the refusal is about the schedule, not about how far we got.
            spelled = {field: _spelled(s, field, hook, why, spell, iset.target) for field, hook, why in _carried(s)}
            for field in ("awaits", "unit"):
                if spelled.get(field):
                    lines.append(pad + spelled[field])
            lines.append(pad + d.render_c([_c_value(v) for _, v in s.args]))
            if spelled.get("produces"):
                lines.append(pad + spelled["produces"])

    body(kernel.body, 1)
    lines.append("    " + iset.drain_c)
    lines.append("}")
    return "\n".join(lines) + "\n"
