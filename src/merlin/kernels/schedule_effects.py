"""Top-down stream facts: a ``merlin.compiler.kernel_ir`` schedule replayed through its own lowering.

ONE VOCABULARY, TWO PRODUCERS. Bottom-up, the lifter steps an emitted binary and records every
accelerator command it issues (``finite_control_counts(record_commands=True)``); the selected decoder
turns that stream into classes, declared roles and fields (:func:`decoded_effects.decode_stream`) and
:func:`decoded_effects.stream_facts` derives the stream facts. Top-down, this module walks a schedule
kernel's dynamic instances (:func:`merlin.compiler.kernel_ir.ir.instances`) and hands each one to the
lowering's OWN target callback -- the function that turns a schedule call into accelerator commands
on the real emission path -- with a builder that evaluates integer arithmetic instead of building IR.
The commands it submits are recorded in the lifter's stream shape and go through the SAME decoder and
the SAME fact derivation. Nothing here knows an instruction, a field or an encoding: roles come from
the machine's declared roster, fields from the selected decoder, and the machine is a parameter.

THREE-STATE. A value the schedule does not determine (a runtime extent, a value read from memory, an
IR operation the replay does not evaluate) stays unknown: a register keeps only the bits that are
known, so a field the decoder needs from unknown bits decodes as ``None`` and its fact stays unknown
with the decoder's reason. Host instruction counts between commands are not a property of a schedule
and are never invented; facts that need them stay unknown. A lowering the replay cannot follow is a
refusal with its reason, never a partial stream presented as complete.

ANNOTATIONS. ``unit``/``produces``/``awaits`` do not reach a target's operand list, so -- exactly as
:func:`merlin.compiler.kernel_ir.codegen.emit_c_function` -- a call carrying one needs a speller hook
(``issue_on``/``on_await``/``on_produce``, each called with the call and the replay context, which may
submit commands or a fence, or nothing to declare the annotation costs no command). Without one the
replay refuses. ``stages``/``sets``/``assumes`` are declarations about operands the call already
carries; they are summarized, never turned into commands, so a declared rotation that no operand
realizes shows up as a stream fact that disagrees with the declaration rather than as agreement.

Comparisons (:func:`attribute`): (a) top-down ours vs bottom-up expert -- what our schedule does
differently; (b) top-down ours vs bottom-up ours -- whether emission realized the schedule; (c)
bottom-up ours vs expert. A difference in (c) also present in (a) is a design choice of the
lowering; one in (c) that (a) does not show was introduced by emission.
"""

from __future__ import annotations

import hashlib
import json

SCHEMA = "merlin.schedule_command_stream.v1"
EFFECTS_SCHEMA = "merlin.schedule_effects.v1"
ATTRIBUTION_SCHEMA = "merlin.cca_attribution.v1"
BASIS = "schedule_instances_through_target_lowering"
LIMIT = 1_000_000
_WIDTH = 64
_MASK = (1 << _WIDTH) - 1


class ReplayRefusal(ValueError):
    """The lowering did something the replay cannot evaluate; the stream is not established."""


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


# ----------------------------------------------------------------------------- replay values
class Word:
    """A 64-bit integer whose bits are partly known (``mask`` marks the known bits)."""

    __slots__ = ("mask", "value", "reason")

    def __init__(self, mask=0, value=0, reason=None):
        self.mask = mask & _MASK
        self.value = value & self.mask
        self.reason = reason

    @property
    def exact(self):
        return self.value if self.mask == _MASK else None

    @property
    def type(self):
        """The builder's index type, so lowering code that widens narrower values leaves this alone."""
        from xdsl.dialects.builtin import i64

        return i64

    def __repr__(self):
        return f"Word(mask={self.mask:#x}, value={self.value:#x})"


class Pointer:
    """A tensor argument's base address plus a byte offset (an int, or a :class:`Word`)."""

    __slots__ = ("tensor", "index", "offset")

    def __init__(self, tensor, index, offset=0):
        self.tensor, self.index, self.offset = tensor, index, offset

    @property
    def type(self):
        return None

    def __repr__(self):
        return f"Pointer({self.tensor!r}, arg{self.index}, +{self.offset!r})"


def unknown(reason):
    """A value the schedule does not determine."""
    return Word(0, 0, reason)


def _bits(value):
    if type(value) is int or isinstance(value, bool):
        return Word(_MASK, int(value))
    if isinstance(value, Word):
        return value
    return Word(0, 0, "pointer used in non-address arithmetic")


def _result(word):
    return word.exact if word.exact is not None else word


def _exact(op, a, b):
    if type(a) is int and type(b) is int:
        return op(a, b)
    return unknown(
        next((v.reason for v in (a, b) if isinstance(v, Word) and v.reason), None)
        or "arithmetic on a value the schedule does not determine"
    )


def _shift(value):
    if type(value) is not int or not 0 <= value < _WIDTH:
        raise ReplayRefusal("shift amount is not a bounded constant")
    return value


class ReplayBuilder:
    """The integer subset of the lowering builder, evaluated instead of emitted.

    Exact values stay Python ints, so a lowering's constant fast paths behave exactly as they do when
    it emits code; partly known values are :class:`Word`; tensor addresses are :class:`Pointer`.
    ``for_loop``/``for_range`` run their bodies for every index. Any IR the replay does not evaluate
    (``add`` of an arbitrary operation, memory loads) refuses with the operation named.
    """

    def const(self, value, ty=None):
        if type(value) is not int:
            raise ReplayRefusal("builder constant is not an integer")
        return value

    def add(self, op):
        raise ReplayRefusal(f"lowering built an IR operation the replay does not evaluate: {type(op).__name__}")

    def load_i64(self, *args, **kwargs):
        return unknown("value loaded from memory at run time")

    def store_i64(self, *args, **kwargs):
        raise ReplayRefusal("lowering stores host memory; the replay records accelerator commands only")

    def add_i(self, a, b):
        if isinstance(a, Pointer) or isinstance(b, Pointer):
            if isinstance(a, Pointer) and isinstance(b, Pointer):
                return unknown("sum of two addresses")
            pointer, other = (a, b) if isinstance(a, Pointer) else (b, a)
            return Pointer(pointer.tensor, pointer.index, self.add_i(pointer.offset, other))
        return _exact(lambda x, y: x + y, a, b)

    def sub_i(self, a, b):
        if isinstance(a, Pointer) and not isinstance(b, Pointer):
            return Pointer(a.tensor, a.index, self.sub_i(a.offset, b))
        return _exact(lambda x, y: x - y, a, b)

    def mul_i(self, a, b):
        if 0 in (a, b) and all(type(v) is int or isinstance(v, Word) for v in (a, b)):
            return 0
        return _exact(lambda x, y: x * y, a, b)

    def sdiv_i(self, a, b):
        return _exact(lambda x, y: int(x / y) if y else None, a, b)

    def udiv_i(self, a, b):
        return _exact(lambda x, y: (x & _MASK) // (y & _MASK) if y else None, a, b)

    def srem_i(self, a, b):
        return _exact(lambda x, y: x - y * int(x / y) if y else None, a, b)

    def urem_i(self, a, b):
        return _exact(lambda x, y: (x & _MASK) % (y & _MASK) if y else None, a, b)

    def and_i(self, a, b):
        x, y = _bits(a), _bits(b)
        zeros = (x.mask & ~x.value) | (y.mask & ~y.value)
        return _result(Word((x.mask & y.mask) | zeros, x.value & y.value))

    def or_i(self, a, b):
        x, y = _bits(a), _bits(b)
        ones = x.value | y.value
        return _result(Word((x.mask & y.mask) | ones, ones))

    def xor_i(self, a, b):
        x, y = _bits(a), _bits(b)
        return _result(Word(x.mask & y.mask, x.value ^ y.value))

    def shl_i(self, a, bits):
        x, n = _bits(a), _shift(bits)
        return _result(Word((x.mask << n) | ((1 << n) - 1), x.value << n))

    def lshr_v(self, a, n):
        x, n = _bits(a), _shift(n)
        return _result(Word((x.mask >> n) | (_MASK & ~(_MASK >> n)), x.value >> n))

    def shl_v(self, a, n):
        return self.shl_i(a, n)

    def ashr_i(self, a, bits):
        x, n = _bits(a), _shift(bits)
        if not x.mask >> (_WIDTH - 1):
            return unknown("arithmetic shift of a value whose sign is unknown")
        sign = (x.value >> (_WIDTH - 1)) * (_MASK & ~(_MASK >> n))
        return _result(Word((x.mask >> n) | (_MASK & ~(_MASK >> n)), (x.value >> n) | sign))

    def ashr_v(self, a, n):
        return self.ashr_i(a, n)

    def gep(self, base, index, elem_ty):
        width = getattr(getattr(elem_ty, "width", None), "data", None)
        if not isinstance(base, Pointer) or type(width) is not int or width % 8:
            raise ReplayRefusal("address computation over a base or element type the replay does not track")
        return self.add_i(base, self.mul_i(index, width // 8))

    def for_loop(self, start, stop, step, body, **_):
        if any(type(v) is not int for v in (start, stop, step)) or step <= 0:
            raise ReplayRefusal("loop bounds are not determined by the schedule")
        for index in range(start, stop, step):
            body(index)

    def for_range(self, count, body):
        self.for_loop(0, count, 1, body)


def operand_record(value):
    """A replayed register in the lifter's issue-time shape (``riscv_machine._issue_operand``)."""
    if isinstance(value, bool):
        value = int(value)
    if type(value) is int:
        return value & _MASK
    if isinstance(value, Pointer):
        offset = value.offset if type(value.offset) is int else None
        return dict(kind="unknown", raw=None, pointer_base=f"arg{value.index}", byte_offset=offset)
    if isinstance(value, Word):
        if value.exact is not None:
            return value.exact
        record = dict(kind="unknown", raw=None)
        if value.mask:
            record["known_bits"] = dict(mask=value.mask, value=value.value, width=_WIDTH)
        return record
    if value is None:
        return 0
    raise ReplayRefusal(f"command operand is not an integer, address or partly known word: {type(value).__name__}")


class ReplayContext:
    """The lowering-context surface a target callback uses, recording commands instead of emitting them.

    ``emit(funct, rs1, rs2)`` submits one accelerator command, ``fence()`` the completion fence,
    ``ptr(name, elem_offset)`` a tensor argument address (argument positions follow ``arguments``),
    ``spec(name)`` a tensor declaration, ``fb`` the :class:`ReplayBuilder`.
    """

    def __init__(self, arguments, *, specs=None, limit=LIMIT):
        self.arguments = list(arguments)
        self.tensors = {name: dict(spec) for name, spec in (specs or {}).items()}
        self.fb = ReplayBuilder()
        self.metadata = {}
        self.stream = []
        self.truncated = False
        self.limit = limit
        self.emitted = []

    def spec(self, name):
        if name not in self.tensors:
            raise ReplayRefusal(f"no tensor declaration for {name!r}")
        return self.tensors[name]

    def buffer(self, name):
        if name not in self.arguments:
            raise ReplayRefusal(f"{name!r} is not a kernel argument; its address is not a schedule fact")
        return Pointer(name, self.arguments.index(name), 0)

    def ptr(self, name, elem_offset=0):
        base = self.buffer(name)
        dtype = str((self.tensors.get(name) or {}).get("dtype") or "")
        digits = dtype.lstrip("abcdefghijklmnopqrstuvwxyz")
        if not digits.isdigit() or int(digits) % 8:
            if elem_offset == 0:
                return base
            raise ReplayRefusal(f"element width of {name!r} is not declared")
        return self.fb.add_i(base, self.fb.mul_i(elem_offset, int(digits) // 8))

    def _append(self, entry):
        if len(self.stream) >= self.limit:
            self.truncated = True
            return
        entry.update(seq=len(self.stream), pc=None, host_instructions_since_prev=None)
        self.stream.append(entry)

    def emit(self, funct, rs1=0, rs2=0):
        if type(funct) is not int:
            raise ReplayRefusal("command function code is not determined by the schedule")
        self._append(dict(funct=funct, rs1=operand_record(rs1), rs2=operand_record(rs2)))
        self.emitted.append(funct)

    def fence(self):
        self._append(dict(kind="fence"))


# ----------------------------------------------------------------------------- schedule walk
_HOOKS = (("awaits", "on_await"), ("unit", "issue_on"))


def _operand(value, env, context):
    from merlin.compiler.kernel_ir.ir import NULL, Expr, Ptr, evaluate

    if isinstance(value, Ptr):  # a schedule pointer is a byte offset into one tensor argument
        return context.fb.add_i(context.buffer(value.tensor), evaluate(value.offset, env))
    if value is NULL:
        return None
    if isinstance(value, Expr):
        return evaluate(value, env)
    return value


def _annotations(kernel):
    from merlin.compiler.kernel_ir.ir import instances

    stages, rotations, sets = 0, [], 0
    seen = set()
    for call, _ in instances(kernel):
        if id(call) in seen:
            continue
        seen.add(id(call))
        stages += len(call.stages)
        sets += len(call.sets)
        rotations += [
            dict(instr=call.instr, memory=s.memory, row=s.row, rows=s.rows, depth=s.depth)
            for s in call.stages
            if s.depth > 1
        ]
    return dict(
        stagings=stages,
        multi_buffered_stagings=rotations,
        configuration_sets=sets,
        basis="declarations on the schedule's calls; summarized, never turned into commands",
    )


def replay_kernel(kernel, lower_call, context, *, spell=None):
    """Hand every dynamic instance of ``kernel`` to ``lower_call(instr, operands)``, in program order.

    ``operands`` maps each operand name to its value for that instance: an int (index expressions
    are evaluated over the instance's loop environment), a :class:`Pointer` for a tensor address,
    ``None`` for ``NULL``. ``lower_call`` submits commands through ``context``. Calls annotated with
    ``unit``/``produces``/``awaits`` need the matching ``spell`` hook (see the module docstring).
    """
    from merlin.compiler.kernel_ir.ir import check_structure, instances

    problems = check_structure(kernel)
    if problems:
        raise ReplayRefusal("schedule kernel is not well formed: " + "; ".join(problems))
    for call, env in instances(kernel):
        if context.truncated:
            break
        for field, hook in (*_HOOKS, ("produces", "on_produce")):
            if getattr(call, field) and getattr(spell, hook, None) is None:
                raise ReplayRefusal(f"{call.instr} carries {field} and no {hook}() speller says what it costs")
        for field, hook in _HOOKS:
            if getattr(call, field):
                getattr(spell, hook)(call, context)
        lower_call(call.instr, {name: _operand(value, env, context) for name, value in call.args})
        if call.produces:
            spell.on_produce(call, context)
    return _annotations(kernel)


def schedule_command_stream(lower, *, arguments, specs=None, limit=LIMIT):
    """Run a lowering entry ``lower(context)`` against a :class:`ReplayContext` and serialize its stream.

    ``lower`` is the author's own lowering of one occurrence (the function that submits its commands
    and hands its schedule kernels to :func:`replay_kernel`, or whose schedule result emits through
    the builder it is given). Returns the ``merlin.schedule_command_stream.v1`` document; a lowering
    the replay cannot follow is ``status: unknown`` with the reason and no stream.
    """
    context = ReplayContext(arguments, specs=specs, limit=limit)
    try:
        receipt = lower(context)
    except (ReplayRefusal, ArithmeticError, LookupError, TypeError, ValueError, AttributeError) as error:
        return dict(
            schema=SCHEMA,
            status="unknown",
            basis=BASIS,
            arguments=list(arguments),
            reason=f"lowering not replayable: {type(error).__name__}: {error}",
            stream=None,
        )
    return dict(
        schema=SCHEMA,
        status="derived",
        basis=BASIS,
        arguments=list(arguments),
        stream=context.stream,
        entries=len(context.stream),
        truncated=context.truncated,
        stream_sha256=_digest(context.stream),
        annotations=receipt if isinstance(receipt, dict) and "stagings" in receipt else None,
    )


def validate_stream_document(document, *, limit=LIMIT):
    """Refuse a serialized stream that is not this schema's shape or exceeds the bound."""
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError("schedule_stream_schema_mismatch")
    if document.get("status") != "derived":
        return document
    stream = document.get("stream")
    if not isinstance(stream, list) or len(stream) > limit:
        raise ValueError("schedule_stream_unbounded_or_malformed")
    for entry in stream:
        registers = [entry.get(slot) for slot in ("rs1", "rs2")] if isinstance(entry, dict) else []
        if not isinstance(entry, dict) or (
            entry.get("kind") != "fence"
            and (
                type(entry.get("funct")) is not int
                or any(not (type(r) is int or isinstance(r, dict)) for r in registers)
            )
        ):
            raise ValueError("schedule_stream_entry_malformed")
    if document.get("stream_sha256") != _digest(stream):
        raise ValueError("schedule_stream_digest_mismatch")
    return document


# ----------------------------------------------------------------------------- facts and records
def schedule_effects(document, *, decoder, structure, evidence_id="schedule"):
    """The stream facts of a serialized schedule stream, through the selected decoder and deriver.

    Returns a ``derived_effects``-shaped block whose ``facts`` are exactly what
    :func:`decoded_effects.stream_facts` derives; an unknown document makes every fact unknown with
    its reason.
    """
    from merlin.kernels.decoded_effects import _STREAM_FACTS, STREAM_SCOPE, _unknown, decode_stream, stream_facts

    document = validate_stream_document(document)
    if document.get("status") != "derived":
        reason = document.get("reason") or "schedule stream not established"
        facts = {name: _unknown(reason, scope=STREAM_SCOPE) for name in _STREAM_FACTS}
        decoded = None
    else:
        decoded = decode_stream(document["stream"], decoder=decoder)
        if decoded is None:
            facts = {
                name: _unknown("no selected decoder for the schedule stream", scope=STREAM_SCOPE)
                for name in _STREAM_FACTS
            }
        else:
            facts = stream_facts(decoded, structure, evidence_id=evidence_id, truncated=bool(document.get("truncated")))
    unknowns = [
        dict(fact=name, reason=fact.get("reason")) for name, fact in facts.items() if fact.get("status") == "unknown"
    ]
    return dict(
        schema=EFFECTS_SCHEMA,
        evidence_id=evidence_id,
        basis=BASIS,
        ordering="schedule program order of dynamic instances",
        facts=facts,
        unknowns=unknowns,
        inputs=dict(
            stream_sha256=document.get("stream_sha256"),
            stream_entries=document.get("entries"),
            decoded_entries=None if decoded is None else len(decoded),
        ),
        annotations=document.get("annotations"),
        ordered_stream=ordered_summary(decoded),
        not_established=["host_instruction_counts", "dynamic_timing", "numerical_equivalence", "performance"],
    )


def effects_record(effects, *, evidence_id=None):
    """An occurrence-shaped record (``derived_effects`` only) for the stream comparator."""
    evidence_id = evidence_id or effects.get("evidence_id") or "occurrence"
    return dict(derived_effects=dict(effects, evidence_id=evidence_id), evidence=dict(evidence_id=evidence_id))


def stream_differences(reference, candidate):
    """The comparator's stream-scoped rows only: static-site rows have no top-down counterpart."""
    from merlin.kernels.mechanism_applicability import STREAM_EFFECTS, mechanism_differences

    return [row for row in mechanism_differences(reference, candidate) if row.get("effect") in STREAM_EFFECTS]


#: Order facts lead every comparison: matching transfer counts with no load/compute overlap still loses.
ORDER_FACTS = (
    "issue_overlap",
    "prefetch_interleave",
    "prefetch_distance",
    "prefetch_compute_ahead",
    "operand_buffering",
    "output_group_extent",
)


def _order_first(rows):
    rank = {name: i for i, name in enumerate(ORDER_FACTS)}
    return sorted(rows, key=lambda row: rank.get(row.get("effect"), len(rank)))


def _kind(effect):
    return "order" if effect in ORDER_FACTS else "count_or_width"


def _keyed(rows):
    out, seen = {}, {}
    for row in rows:
        base = (row.get("effect"), row.get("role"))
        n = seen.get(base, 0)
        seen[base] = n + 1
        out[(*base, n)] = row
    return out


def _realized(rows):
    """``False`` when an established row differs, ``True`` when every established row matches, ``None``
    when nothing is established; rows either side leaves unknown are listed, never counted as matches."""
    states = [row.get("difference") for row in rows]
    value = False if "differs" in states else True if "same" in states else None
    return dict(
        value=value,
        compared=sum(s in ("same", "differs") for s in states),
        not_established=[
            dict(effect=r.get("effect"), role=r.get("role"), reason=r.get("reason"))
            for r in rows
            if r.get("difference") == "unknown"
        ],
    )


def attribute(*, schedule, emitted, expert):
    """Three comparisons over one vocabulary, and each (c) difference's attribution.

    ``schedule``: the top-down record of our schedule; ``emitted``: the bottom-up record of the
    binary we emitted from it; ``expert``: the bottom-up record of the expert unit. Records are
    occurrence-shaped (``derived_effects`` with stream facts). ``attribution`` is
    ``lowering_design_choice`` when the difference from the expert is already in the schedule,
    ``introduced_by_emission`` when the schedule agrees with the expert and the binary does not,
    and ``unknown`` with the reason otherwise.
    """
    a, b, c = (
        _order_first(stream_differences(ref, cand))
        for ref, cand in ((expert, schedule), (schedule, emitted), (expert, emitted))
    )
    ka, kb = _keyed(a), _keyed(b)
    rows = []
    for key, row in _keyed(c).items():
        if row.get("difference") != "differs":
            continue
        top, emission = ka.get(key), kb.get(key)
        top_state = (top or {}).get("difference")
        emission_state = (emission or {}).get("difference")
        if top_state == "differs":
            verdict, why = "lowering_design_choice", "the schedule already differs from the expert here"
        elif top_state == "same" and emission_state != "same":
            verdict, why = "introduced_by_emission", "the schedule matches the expert; the emitted binary does not"
        elif top_state == "same":
            verdict = "unknown"
            why = "schedule and binary each compare as equal to their counterpart; the rows disagree"
        else:
            verdict = "unknown"
            why = (top or {}).get("reason") or "the schedule side of this effect is not established"
        rows.append(
            dict(
                effect=key[0],
                role=key[1],
                kind=_kind(key[0]),
                attribution=verdict,
                reason=why,
                expert_vs_emitted=dict(
                    reference=row.get("reference"), candidate=row.get("candidate"), direction=row.get("direction")
                ),
                expert_vs_schedule=None
                if top is None
                else dict(difference=top_state, candidate=top.get("candidate"), direction=top.get("direction")),
                schedule_vs_emitted=None
                if emission is None
                else dict(
                    difference=emission_state, reference=emission.get("reference"), candidate=emission.get("candidate")
                ),
            )
        )
    return dict(
        schema=ATTRIBUTION_SCHEMA,
        comparisons=dict(schedule_vs_expert=a, emitted_vs_schedule=b, emitted_vs_expert=c),
        emission_realized_schedule=_realized(b),
        emission_changes=[
            dict(effect=r.get("effect"), role=r.get("role"), schedule=r.get("reference"), emitted=r.get("candidate"))
            for r in b
            if r.get("difference") == "differs"
        ],
        attribution=rows,
        basis="one stream-fact vocabulary: top-down from schedule instances through the target lowering, "
        "bottom-up from the lifted binary, both decoded and derived by the same functions",
        performance="unknown",
        semantic_equivalence="unknown",
    )


# ----------------------------------------------------------------------------- ordered summary, bounds, sweep
_RUNS = 256


def _role(entry):
    roles = entry.get("roles") or ()
    return "+".join(roles) if roles else "unroled"


def ordered_summary(decoded, *, runs=_RUNS):
    """Bounded issue-order summary of a decoded stream, for consumers that need order but not cycles.

    ``runs``: the first ``runs`` maximal runs of one class, ``[class, role, count]``; ``blocks``:
    per barrier-delimited block, how many commands of each role it holds (``max``/``median`` per role
    over all blocks). Order only -- no latency, occupancy or cycle is claimed.
    """
    if decoded is None:
        return dict(status="unknown", reason="no decoded stream")
    out, blocks, current = [], [], {}
    for entry in decoded:
        klass, role = entry.get("class"), _role(entry)
        if out and out[-1][0] == klass:
            out[-1][2] += 1
        elif len(out) < runs:
            out.append([klass, role, 1])
        if "sync" in (entry.get("roles") or ()):
            blocks.append(current)
            current = {}
        else:
            current[role] = current.get(role, 0) + 1
    blocks.append(current)
    per_role = {}
    for block in blocks:
        for role, count in block.items():
            per_role.setdefault(role, []).append(count)
    return dict(
        status="observed",
        entries=len(decoded),
        barriers=len(blocks) - 1,
        runs_prefix=out,
        runs_truncated=len(out) >= runs,
        commands_per_block={
            role: dict(max=max(v), median=sorted(v)[len(v) // 2], blocks=len(v)) for role, v in sorted(per_role.items())
        },
        timing="not_claimed: issue order only; no latency, occupancy or cycle is derived",
    )


def command_class_depths(facts):
    """Per command-class dispatch entries the selected RTL facts derive, or a reason when absent."""
    body = (facts or {}).get("facts", facts or {})
    classes = (((body.get("timing_structure") or {}).get("command_classes") or {}).get("classes")) or {}
    depths = {
        name: (row.get("entries") or {}).get("value")
        for name, row in classes.items()
        if (row.get("entries") or {}).get("status") == "derived"
    }
    if not depths:
        return dict(
            status="unknown", depths={}, reason="the selected RTL facts derive no per-command-class dispatch entries"
        )
    return dict(status="derived", depths=depths, source="facts.timing_structure.command_classes.classes.*.entries")


def resource_bounds(summary, depths, *, routing=None):
    """Where a block's commands of one role exceed the dispatch entries of the class that role enters.

    ``routing`` maps a decoded instruction class to an RTL command class; the selected facts carry the
    depths but not which instruction classes enter which class, so without ``routing`` every row stays
    unknown with that reason. A crossing is a structural fact about issue order, not a stall claim.
    """
    if summary.get("status") != "observed" or depths.get("status") != "derived":
        return [dict(status="unknown", reason=summary.get("reason") or depths.get("reason"))]
    rows = []
    roles = {}
    for klass, role, _ in summary.get("runs_prefix") or ():
        roles.setdefault(role, set()).add(klass)
    for role, use in (summary.get("commands_per_block") or {}).items():
        if role == "sync":
            continue
        targets = {(routing or {}).get(k) for k in roles.get(role, ())}
        if not routing or None in targets or len(targets) != 1:
            rows.append(
                dict(
                    role=role,
                    status="unknown",
                    block_max=use["max"],
                    bound=None,
                    reason="the selected facts do not map this role's instruction classes to one RTL "
                    "command class; supply routing= from a declared source",
                )
            )
            continue
        target = next(iter(targets))
        bound = depths["depths"].get(target)
        rows.append(
            dict(
                role=role,
                command_class=target,
                status="observed" if bound else "unknown",
                block_max=use["max"],
                bound=bound,
                exceeds=None if not bound else use["max"] > bound,
                **({} if bound else dict(reason=f"no derived entries for command class {target}")),
            )
        )
    return rows


def shape_sweep(instantiate, shapes, *, decoder, structure, facts=None, routing=None, limit=LIMIT):
    """Top-down facts per declared shape: the schedule is instantiated once per shape.

    ``instantiate(shape)`` returns ``(lower, arguments, specs)`` for :func:`schedule_command_stream`.
    Each row carries the order facts first, the bounded ordered summary and the per-role block
    occupancy against the selected RTL command-class depths (unknown where the facts say nothing).
    """
    from merlin.kernels.decoded_effects import decode_stream

    depths = command_class_depths(facts)
    rows = []
    for shape in shapes:
        lower, arguments, specs = instantiate(shape)
        document = schedule_command_stream(lower, arguments=arguments, specs=specs, limit=limit)
        effects = schedule_effects(document, decoder=decoder, structure=structure, evidence_id=f"schedule:{shape}")
        decoded = decode_stream(document["stream"], decoder=decoder) if document.get("stream") is not None else None
        summary = ordered_summary(decoded)
        facts_in_order = {
            name: effects["facts"][name] for name in (*ORDER_FACTS, *effects["facts"]) if name in effects["facts"]
        }
        rows.append(
            dict(
                shape=shape,
                status=document["status"],
                reason=document.get("reason"),
                facts=facts_in_order,
                ordered_stream=summary,
                resource_bounds=resource_bounds(summary, depths, routing=routing),
            )
        )
    return dict(
        schema="merlin.schedule_shape_sweep.v1",
        command_class_depths=depths,
        shapes=rows,
        performance="unknown",
        timing="not_claimed",
    )


# ----------------------------------------------------------------------------- author tool
def main(argv=None):
    """``python -m merlin.kernels.schedule_effects MODULE:FUNCTION --arguments a,b,c [--specs F] --out F``.

    Runs the author's lowering entry ``FUNCTION(context)`` in the author's own process and writes the
    ``merlin.schedule_command_stream.v1`` document; a consumer decodes it with its own selected decoder
    (:func:`schedule_effects`) after :func:`validate_stream_document`, so no candidate code runs there.
    """
    import argparse
    import importlib
    from pathlib import Path

    parser = argparse.ArgumentParser(prog="python -m merlin.kernels.schedule_effects")
    parser.add_argument("entry", help="MODULE:FUNCTION taking a ReplayContext")
    parser.add_argument("--arguments", required=True, help="tensor arguments in kernel argument order")
    parser.add_argument("--specs", help="JSON file: {tensor: {dtype, shape}}")
    parser.add_argument("--limit", type=int, default=LIMIT)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    module, _, function = args.entry.partition(":")
    if not module or not function.isidentifier() or not 1 <= args.limit <= LIMIT:
        parser.error("entry must be MODULE:FUNCTION and the limit within the bound")
    lower = getattr(importlib.import_module(module), function)
    specs = json.loads(Path(args.specs).read_text()) if args.specs else None
    document = schedule_command_stream(
        lower, arguments=[a for a in args.arguments.split(",") if a], specs=specs, limit=args.limit
    )
    Path(args.out).write_text(json.dumps(document, sort_keys=True, default=str) + "\n")
    print(json.dumps({k: document.get(k) for k in ("schema", "status", "reason", "entries", "truncated")}))
    return 0 if document["status"] == "derived" else 2


if __name__ == "__main__":
    raise SystemExit(main())
