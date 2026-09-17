"""Predict a program's bulk-movement byte volume from the program itself.

The term this replaces is a workload INPUT. A cost model that is handed the byte volume answers
"how long does this take GIVEN the traffic", which is a weaker claim than it looks: on a
movement-bound target the handed-in number carries most of the answer. Deriving it from the
program's own descriptors is what makes the prediction a prediction.

Instruction identity and operand placement come from the decoded program and ISA model. Whether that
identity reads, writes, or synchronises comes from a caller-supplied semantic fact grounded in the
RTL. Keeping those inputs separate is deliberate: an encoding says *which* instruction this is, not
what traffic it causes. Nothing here names a target, an opcode, a mnemonic spelling, or a channel
count.

TWO RULES THAT MAKE THE ANSWER HONEST, both enforced below and pinned by tests:

* The size operand is read from the ISA model's own field layout, never from a position. Taking
  "operand 2" because it was operand 2 in an example fails silently on the first form whose layout
  differs -- it returns a number, and the number is wrong.
* A descriptor whose size cannot be resolved makes the WHOLE KERNEL report a lower bound, not merely
  that descriptor report UNKNOWN. Summing only the descriptors that happened to resolve understates
  the footprint, and it understates it in the flattering direction: a smaller predicted footprint
  makes the compiler look better and the model look more accurate at the same time.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import string
from typing import Any

#: Field names a size may travel in, strongest first. Consulted against the ISA model's OWN layout;
#: a name absent from the layout is never assumed to exist.
_SIZE_FIELD_ORDER = ("len", "bytes", "size", "nbytes", "rs2")


class DmaVolumeError(RuntimeError):
    """A movement volume was asked for where the evidence cannot support one."""


@dataclass(frozen=True)
class Descriptor:
    """One bulk-movement command, as recovered from the program text."""

    index: int
    form: str                      # the movement form, from the encoding (e.g. a load/store/wait family)
    channel: int | None
    direction: str                 # "read" | "write" | "sync" | "unknown"
    size_bytes: int | None         # None = unresolved; NEVER 0 as a stand-in
    size_field: str | None         # which declared field the size was read from
    unresolved_reason: str | None = None
    #: Every field used in a product such as rows*columns. ``size_field`` remains for API compatibility.
    size_fields: tuple[str, ...] = ()
    semantic_provenance: str | None = None
    field_provenance: str | None = None

    @property
    def resolved(self) -> bool:
        return self.size_bytes is not None

    @property
    def traffic_resolved(self) -> bool:
        """Whether this descriptor contributes to one exact directional byte total."""
        return self.direction == "sync" or (self.direction in ("read", "write") and self.resolved)

    def to_dict(self) -> dict[str, Any]:
        """A stable JSON-ready descriptor record."""
        return {
            "index": self.index, "form": self.form, "channel": self.channel,
            "direction": self.direction, "size_bytes": self.size_bytes,
            "size_field": self.size_field, "size_fields": list(self.size_fields),
            "resolved": self.resolved, "traffic_resolved": self.traffic_resolved,
            "unresolved_reason": self.unresolved_reason,
            "provenance": {"semantic": self.semantic_provenance, "fields": self.field_provenance},
        }


@dataclass(frozen=True)
class KernelVolume:
    """A kernel's predicted movement volume, and how much of it is actually evidenced."""

    kernel: str
    descriptors: tuple[Descriptor, ...]
    read_bytes: int
    write_bytes: int
    #: True when ANY descriptor is unresolved -- the total is then a floor, not a prediction.
    is_lower_bound: bool
    unresolved: tuple[str, ...] = ()
    read_provenance: tuple[str, ...] = ()
    write_provenance: tuple[str, ...] = ()
    basis: str = "scheduled_descriptors"

    @property
    def total_bytes(self) -> int:
        """Evidenced bytes. When ``is_lower_bound`` this is a floor, not an exact zero/total."""
        return self.read_bytes + self.write_bytes

    @property
    def scheduled_read_bytes(self) -> int:
        return self.read_bytes

    @property
    def scheduled_write_bytes(self) -> int:
        return self.write_bytes

    @property
    def scheduled_total_bytes(self) -> int:
        return self.total_bytes

    @property
    def exact_total_bytes(self) -> int | None:
        """Exact total, or UNKNOWN. Existing integer totals remain available as evidenced floors."""
        return None if self.is_lower_bound else self.total_bytes

    @property
    def exact_read_bytes(self) -> int | None:
        return None if self.is_lower_bound else self.read_bytes

    @property
    def exact_write_bytes(self) -> int | None:
        return None if self.is_lower_bound else self.write_bytes

    def claim(self) -> str:
        """The strongest sentence this evidence supports. Never a bare number."""
        if self.is_lower_bound:
            return (f"AT LEAST {self.total_bytes} bytes scheduled; {len(self.unresolved)} of "
                    f"{len(self.descriptors)} descriptors did not resolve, so the true volume is "
                    f"higher by an unmeasured amount")
        return f"{self.total_bytes} bytes scheduled across {len(self.descriptors)} descriptors"

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready scheduled-volume record; exact and lower-bound values remain distinct."""
        return {
            "kernel": self.kernel, "basis": self.basis,
            "descriptors": [descriptor.to_dict() for descriptor in self.descriptors],
            "read_bytes": self.read_bytes, "write_bytes": self.write_bytes,
            "total_bytes": self.exact_total_bytes,
            "scheduled_read_bytes": self.read_bytes, "scheduled_write_bytes": self.write_bytes,
            "known_lower_bound_bytes": self.total_bytes, "exact_total_bytes": self.exact_total_bytes,
            "is_lower_bound": self.is_lower_bound, "unresolved": list(self.unresolved),
            "provenance": {"read": list(self.read_provenance), "write": list(self.write_provenance)},
        }


@dataclass(frozen=True)
class PhysicalVolume:
    """Bus traffic derived from explicitly bound target counters, never from scheduled descriptors."""

    read_bytes: int | None
    write_bytes: int | None
    read_lower_bound: int
    write_lower_bound: int
    unattributed_bytes: int
    is_lower_bound: bool
    unresolved: tuple[str, ...] = ()
    read_provenance: tuple[str, ...] = ()
    write_provenance: tuple[str, ...] = ()
    unattributed_provenance: tuple[str, ...] = ()
    basis: str = "physical_counters"

    @property
    def total_bytes(self) -> int | None:
        if self.read_bytes is None or self.write_bytes is None or self.is_lower_bound:
            return None
        return self.read_bytes + self.write_bytes

    @property
    def known_lower_bound_bytes(self) -> int:
        return self.read_lower_bound + self.write_lower_bound + self.unattributed_bytes

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready physical-counter record with UNKNOWN represented by ``None``."""
        return {
            "basis": self.basis, "read_bytes": self.read_bytes, "write_bytes": self.write_bytes,
            "read_lower_bound": self.read_lower_bound, "write_lower_bound": self.write_lower_bound,
            "unattributed_bytes": self.unattributed_bytes,
            "known_lower_bound_bytes": self.known_lower_bound_bytes,
            "total_bytes": self.total_bytes, "exact_total_bytes": self.total_bytes,
            "is_lower_bound": self.is_lower_bound,
            "unresolved": list(self.unresolved),
            "provenance": {
                "read": list(self.read_provenance), "write": list(self.write_provenance),
                "unattributed": list(self.unattributed_provenance),
            },
        }


def size_field_for(isa: Any, mnemonic: str) -> str | None:
    """Which DECLARED field of ``mnemonic`` carries a transfer size, or None if none does.

    Read from the ISA model's own layout. Returning None is a real answer -- it means this form does
    not name a size operand, and a caller must record UNKNOWN rather than pick a position."""
    try:
        fields = isa.fields_of(mnemonic)
    except Exception:  # noqa: BLE001 - a form the model cannot lay out tells us nothing
        return None
    if not fields:
        return None
    for name in _SIZE_FIELD_ORDER:
        if name in fields:
            return name
    return None


def _loop_positions(instructions: Sequence[Mapping[str, Any]]) -> set[int]:
    """Program positions whose dynamic execution is controlled by a backward edge.

    Looking only at branches already visited is unsound: the transfer normally precedes the backedge.
    If a decoded branch says it is backward but does not expose its target, every position is possibly
    loop-carried and therefore UNKNOWN. That conservative result is preferable to a confident one-pass
    underestimate.
    """
    by_index = {inst.get("index", pos): pos for pos, inst in enumerate(instructions)}
    positions: set[int] = set()
    unknown_target = False
    for pos, inst in enumerate(instructions):
        target = inst.get("branch_target")
        target_pos = by_index.get(target) if isinstance(target, int) and not isinstance(target, bool) else None
        is_backedge = target_pos is not None and target_pos <= pos
        if is_backedge:
            positions.update(range(target_pos, pos + 1))
        elif inst.get("branches_backward"):
            unknown_target = True
    return set(range(len(instructions))) if unknown_target else positions


def _immediate_fields(spec: Any) -> tuple[str, str] | None:
    """Return (destination, value) fields from a declared immediate-form fact.

    A string is the legacy API and keeps its canonical ``rd`` destination. New fact records may name
    both fields, avoiding any dependence on a target's operand ordering or naming.
    """
    if isinstance(spec, str):
        return "rd", spec
    if isinstance(spec, Mapping):
        destination = spec.get("destination_field")
        value = spec.get("value_field")
        if isinstance(destination, str) and isinstance(value, str):
            return destination, value
    return None


def propagate_constants(instructions: Sequence[Mapping[str, Any]], *,
                        immediate_forms: Mapping[str, Any]) -> list[dict[int, int | None]]:
    """Per-instruction snapshots of which scalar registers hold a known constant.

    Forward propagation with KILL semantics: a register written by anything other than a declared
    immediate form becomes UNKNOWN rather than keeping a stale value. That distinction is the whole
    point -- a program that loads a length once and rewrites the register later must not have the old
    length attributed to the later transfer.

    ``immediate_forms`` maps a form name either to its legacy value-field string or to explicit
    ``{destination_field, value_field}`` facts, so no operand position is assumed. A backward branch
    invalidates its entire loop, including transfers that textually precede the branch: a value or
    execution count that differs per iteration is not a constant."""
    state: dict[int, int | None] = {}
    out: list[dict[int, int | None]] = []
    loops = _loop_positions(instructions)
    for pos, inst in enumerate(instructions):
        form = str(inst.get("form") or "")
        ops = inst.get("operands") or {}
        immediate = _immediate_fields(immediate_forms.get(form))
        dest_field = immediate[0] if immediate else "rd"
        dest = ops.get(dest_field)
        if pos in loops:
            state = {}
        elif immediate is not None and isinstance(dest, int) and dest != 0:
            imm = ops.get(immediate[1])
            state[dest] = int(imm) if isinstance(imm, int) and not isinstance(imm, bool) else None
        elif isinstance(dest, int) and dest != 0:
            state[dest] = None          # written by something we cannot evaluate -> UNKNOWN, not stale
        out.append(dict(state))
    return out


def _integer(value: Any) -> int | None:
    """A decoded integer value, including the runner's explicit ``{kind: const, raw: ...}`` form."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, Mapping) and value.get("kind") == "const":
        raw = value.get("raw")
        return raw if isinstance(raw, int) and not isinstance(raw, bool) else None
    return None


def _provenance(fact: Any) -> str | None:
    if not isinstance(fact, Mapping):
        return None
    value = fact.get("provenance")
    return value if isinstance(value, str) and value else None


def _receipt_problem(fact: Any, *, kind: str, allow_tool: bool = False) -> str | None:
    """Validate one generated, content-addressed semantic binding.

    A prose provenance label is not evidence.  The binding must identify its schema role, carry the
    SHA-256 of the artifact it was extracted from, and state whether that extraction came from RTL or
    (only where admitted) an ISA/compiler tool.  This remains a pure validator: artifact pinning and
    digest recomputation belong to the I/O adapter that constructs the bundle.
    """
    if not isinstance(fact, Mapping):
        return "binding is not a mapping"
    if fact.get("fact_kind") != kind:
        return f"fact_kind must be {kind!r}"
    digest = fact.get("artifact_sha256")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(char not in string.hexdigits for char in digest)):
        return "artifact_sha256 is missing or malformed"
    if not _provenance(fact):
        return "provenance is missing"
    if fact.get("derived_from_rtl") is not True \
            and not (allow_tool and fact.get("derived_from_tool") is True):
        return "positive generated-fact standing is missing"
    return None


def _declared_fields(isa: Any, form: Any) -> set[str] | None:
    try:
        fields = isa.fields_of(form)
    except Exception:  # noqa: BLE001 - an unmodelled layout supports no field claim
        return None
    return set(fields) if isinstance(fields, Mapping) else None


def _size_field_names(fact: Mapping[str, Any]) -> tuple[str, ...]:
    many = fact.get("size_fields")
    if isinstance(many, Sequence) and not isinstance(many, (str, bytes)) \
            and all(isinstance(v, str) for v in many):
        return tuple(many)
    one = fact.get("size_field")
    return (one,) if isinstance(one, str) else ()


def _source_for(fact: Mapping[str, Any], field_name: str) -> str | None:
    sources = fact.get("size_sources")
    if isinstance(sources, Mapping):
        source = sources.get(field_name)
        return source if isinstance(source, str) else None
    source = fact.get("size_source")
    return source if isinstance(source, str) else None


def _resolve_value(value: Any, source: str | None, constants: Mapping[int, int | None]) -> int | None:
    decoded = _integer(value)
    if source == "value":
        return decoded
    if source == "register" and decoded is not None:
        return 0 if decoded == 0 else constants.get(decoded)
    return None


def _channel_for(fact: Mapping[str, Any], operands: Mapping[str, Any],
                 declared: set[str] | None, constants: Mapping[int, int | None]) -> int | None:
    channel_field = fact.get("channel_field")
    if not isinstance(channel_field, str) or declared is None or channel_field not in declared:
        return None
    return _resolve_value(operands.get(channel_field), fact.get("channel_source"), constants)


def descriptors_from_program(
        instructions: Sequence[Mapping[str, Any]], isa: Any, *,
        semantic_facts: Mapping[Any, Mapping[str, Any]],
        field_facts: Mapping[Any, Mapping[str, Any]],
        immediate_forms: Mapping[str, Any] | None = None,
        identity_field: str = "form", operands_field: str = "operands") -> tuple[Descriptor, ...]:
    """Derive traffic descriptors from decoded instructions and independently grounded facts.

    ``semantic_facts`` is keyed by the decoder's instruction identity and supplies ``direction`` as
    ``read``, ``write`` or ``sync``. ``field_facts`` supplies named fields and their interpretation:
    ``size_field`` or ``size_fields``; ``size_source`` (``value`` or ``register``); the mandatory
    ``unit_bytes``; and optional ``channel_field``/``channel_source``. Every size/channel field is
    checked against ``isa.fields_of(identity)`` before its decoded value is used.

    Identity and operand-container keys are explicit parameters so this works equally with a decoder
    exposing ``form``/``operands`` or ``class``/``decoded``. A semantic fact with an unknown direction
    still creates an ``unknown`` descriptor; an absent semantic fact is not guessed to be traffic.
    """
    normalized: list[dict[str, Any]] = []
    identities: list[Any] = []
    for pos, inst in enumerate(instructions):
        identity = inst.get(identity_field)
        identities.append(identity)
        instruction_index = inst.get("index", pos)
        if not isinstance(instruction_index, int) or isinstance(instruction_index, bool):
            instruction_index = pos
        normalized.append({
            "index": instruction_index, "form": str(identity) if identity is not None else "",
            "operands": inst.get(operands_field) if isinstance(inst.get(operands_field), Mapping) else {},
            "branches_backward": inst.get("branches_backward", False),
            "branch_target": inst.get("branch_target"),
        })
    constants = propagate_constants(normalized, immediate_forms=immediate_forms or {})
    loops = _loop_positions(normalized)
    out: list[Descriptor] = []
    for pos, (inst, identity) in enumerate(zip(normalized, identities)):
        if identity is None:
            out.append(Descriptor(
                index=pos, form="", channel=None, direction="unknown", size_bytes=None,
                size_field=None, unresolved_reason="decoded instruction identity is UNKNOWN"))
            continue
        semantic = semantic_facts.get(identity)
        if not isinstance(semantic, Mapping):
            out.append(Descriptor(
                index=pos, form=str(identity), channel=None, direction="unknown", size_bytes=None,
                size_field=None, unresolved_reason="decoded instruction has no semantic binding"))
            continue
        semantic_problem = _receipt_problem(
            semantic, kind="instruction_effect", allow_tool=False)
        if semantic_problem:
            out.append(Descriptor(
                index=pos, form=str(identity), channel=None, direction="unknown", size_bytes=None,
                size_field=None, unresolved_reason=f"unproven semantic binding: {semantic_problem}",
                semantic_provenance=_provenance(semantic)))
            continue
        if semantic.get("traffic") is False:
            continue
        direction_value = semantic.get("direction")
        direction = direction_value if direction_value in ("read", "write", "sync") else "unknown"
        form = str(identity)
        index_value = instructions[pos].get("index", pos)
        index = index_value if isinstance(index_value, int) and not isinstance(index_value, bool) else pos
        field_fact = field_facts.get(identity)
        fact = field_fact if isinstance(field_fact, Mapping) else {}
        declared = _declared_fields(isa, identity)
        channel = _channel_for(fact, inst["operands"], declared, constants[pos])
        if direction == "sync":
            out.append(Descriptor(index=index, form=form, channel=channel, direction="sync",
                                  size_bytes=None, size_field=None,
                                  semantic_provenance=_provenance(semantic),
                                  field_provenance=_provenance(fact)))
            continue

        size_fields = _size_field_names(fact)
        reasons: list[str] = []
        if direction == "unknown":
            reasons.append("traffic direction is UNKNOWN in the RTL-derived semantic facts")
        if not fact:
            reasons.append("no RTL/ISA-derived field fact identifies a byte size")
        else:
            field_problem = _receipt_problem(fact, kind="descriptor_layout", allow_tool=True)
            if field_problem:
                reasons.append(f"unproven field binding: {field_problem}")
            if fact.get("size_semantics") != "static_product":
                reasons.append(
                    "descriptor size is not established as a static product; stateful/dynamic "
                    "semantics require a trace-resolved transfer record")
            if not size_fields:
                reasons.append("field fact identifies no size field")
            elif declared is None or any(name not in declared for name in size_fields):
                reasons.append("size field is not present in the ISA layout")

        unit = _integer(fact.get("unit_bytes")) if fact else None
        if fact and (unit is None or unit <= 0):
            reasons.append("byte unit is UNKNOWN in the field facts")

        size: int | None = None
        if pos in loops:
            reasons.append("dynamic size or execution count is loop-carried")
        elif not reasons or (direction == "unknown" and len(reasons) == 1):
            factors: list[int] = []
            for name in size_fields:
                value = _resolve_value(inst["operands"].get(name), _source_for(fact, name), constants[pos])
                if value is None or value <= 0:
                    reasons.append(f"size field {name!r} is UNKNOWN")
                    break
                factors.append(value)
            if not any("size field" in reason and "UNKNOWN" in reason for reason in reasons):
                size = unit
                for value in factors:
                    size *= value

        out.append(Descriptor(
            index=index, form=form, channel=channel, direction=direction, size_bytes=size,
            size_field=size_fields[0] if len(size_fields) == 1 else None,
            unresolved_reason="; ".join(dict.fromkeys(reasons)) or None,
            size_fields=size_fields, semantic_provenance=_provenance(semantic),
            field_provenance=_provenance(fact)))
    return tuple(out)


def volume_from_program(kernel: str, instructions: Sequence[Mapping[str, Any]], isa: Any, **facts: Any
                        ) -> KernelVolume:
    """Convenience composition of :func:`descriptors_from_program` and :func:`kernel_volume`."""
    return kernel_volume(kernel, descriptors_from_program(instructions, isa, **facts))


def _unknown_trace_report(kernel: str, provenance: Any, unresolved: Sequence[str]) -> dict[str, Any]:
    return {
        "kernel": kernel, "status": "unknown", "provenance": provenance,
        "scheduled": None, "physical": None, "comparison": None,
        "unresolved": list(unresolved),
    }


def traffic_report_from_trace(
        kernel: str, trace: Mapping[str, Any], isa: Any, *,
        fact_bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Build a JSON-ready traffic report from a decoded trace and its complete adapter facts.

    The fact bundle must carry ``provenance``, an ``adapter`` naming ``instructions_field``,
    ``identity_field`` and ``operands_field``, plus ``semantic_facts`` and ``field_facts``. Optional
    counter integration requires both ``adapter.counter_readings_field`` and ``counter_facts``.
    Nothing is inferred from target, operation, mnemonic, or counter spelling.

    Missing adapter/fact-bundle structure returns ``status=unknown``. Missing semantic bindings for
    individual decoded instructions become unknown descriptors, preserving resolved work only as a
    lower bound instead of presenting an empty exact trace.
    """
    provenance = fact_bundle.get("provenance")
    fatal: list[str] = []
    if not isinstance(provenance, (str, Mapping)) or not provenance:
        fatal.append("fact bundle provenance is missing")
    adapter_value = fact_bundle.get("adapter")
    adapter = adapter_value if isinstance(adapter_value, Mapping) else {}
    required_adapter = ("instructions_field", "identity_field", "operands_field")
    for name in required_adapter:
        if not isinstance(adapter.get(name), str) or not adapter.get(name):
            fatal.append(f"missing adapter fact: {name}")
    if "semantic_facts" not in fact_bundle or not isinstance(fact_bundle.get("semantic_facts"), Mapping):
        fatal.append("semantic_facts mapping is missing from the fact bundle")
    if "field_facts" not in fact_bundle or not isinstance(fact_bundle.get("field_facts"), Mapping):
        fatal.append("field_facts mapping is missing from the fact bundle")
    if fatal:
        return _unknown_trace_report(kernel, provenance, fatal)

    instructions_value = trace.get(adapter["instructions_field"])
    if not isinstance(instructions_value, Sequence) or isinstance(instructions_value, (str, bytes)) \
            or not all(isinstance(inst, Mapping) for inst in instructions_value):
        return _unknown_trace_report(
            kernel, provenance,
            [f"trace field {adapter['instructions_field']!r} is missing or is not an instruction sequence"])
    instructions = list(instructions_value)
    semantic_facts = fact_bundle["semantic_facts"]
    field_facts = fact_bundle["field_facts"]
    identity_field = adapter["identity_field"]
    operands_field = adapter["operands_field"]
    descriptors = list(descriptors_from_program(
        instructions, isa, semantic_facts=semantic_facts, field_facts=field_facts,
        immediate_forms=fact_bundle.get("immediate_forms")
        if isinstance(fact_bundle.get("immediate_forms"), Mapping) else {},
        identity_field=identity_field, operands_field=operands_field))

    scheduled = kernel_volume(kernel, sorted(descriptors, key=lambda descriptor: descriptor.index))

    physical: PhysicalVolume | None = None
    comparison: dict[str, Any] | None = None
    integration_errors: list[str] = []
    has_counter_facts = "counter_facts" in fact_bundle
    counter_field = adapter.get("counter_readings_field")
    if has_counter_facts and (not isinstance(counter_field, str) or not counter_field):
        integration_errors.append("missing adapter fact: counter_readings_field")
    elif isinstance(counter_field, str):
        readings = trace.get(counter_field)
        counter_facts = fact_bundle.get("counter_facts")
        if not isinstance(readings, Mapping):
            integration_errors.append(f"trace counter field {counter_field!r} is missing or not a mapping")
        elif not isinstance(counter_facts, Sequence) or isinstance(counter_facts, (str, bytes)) \
                or not all(isinstance(fact, Mapping) for fact in counter_facts):
            integration_errors.append("counter_facts is missing or not a sequence of bindings")
        else:
            physical = physical_volume_from_counters(readings, counter_facts=counter_facts)
            comparison = compare_to_counters(scheduled, readings, counter_facts=counter_facts)

    unresolved = list(scheduled.unresolved) + integration_errors
    if physical is not None:
        unresolved.extend(physical.unresolved)
    status = "exact"
    if integration_errors:
        status = "unknown"
    elif scheduled.is_lower_bound or (physical is not None and physical.is_lower_bound):
        status = "lower_bound"
    return {
        "kernel": kernel, "status": status, "provenance": provenance,
        "scheduled": scheduled.to_dict(), "physical": physical.to_dict() if physical else None,
        "comparison": comparison, "unresolved": unresolved,
    }


def _descriptor_provenance(descriptors: Sequence[Descriptor], direction: str) -> tuple[str, ...]:
    sources: list[str] = []
    for descriptor in descriptors:
        if descriptor.direction != direction:
            continue
        joined = "; ".join(source for source in (
            descriptor.semantic_provenance, descriptor.field_provenance) if source)
        if joined and joined not in sources:
            sources.append(joined)
    return tuple(sources)


def kernel_volume(kernel: str, descriptors: Sequence[Descriptor]) -> KernelVolume:
    """Fold descriptors into scheduled volume, degrading to a LOWER BOUND on any unresolved one."""
    read = sum(d.size_bytes or 0 for d in descriptors if d.direction == "read" and d.resolved)
    write = sum(d.size_bytes or 0 for d in descriptors if d.direction == "write" and d.resolved)
    unresolved = tuple(f"[{d.index}] {d.form}: {d.unresolved_reason or 'traffic unresolved'}"
                       for d in descriptors if not d.traffic_resolved)
    return KernelVolume(kernel=kernel, descriptors=tuple(descriptors), read_bytes=read,
                        write_bytes=write, is_lower_bound=bool(unresolved), unresolved=unresolved,
                        read_provenance=_descriptor_provenance(descriptors, "read"),
                        write_provenance=_descriptor_provenance(descriptors, "write"))


def physical_volume_from_counters(
        readings: Mapping[str, Any], *, counter_facts: Sequence[Mapping[str, Any]]) -> PhysicalVolume:
    """Convert target counter readings to physical bus bytes through explicit semantic bindings.

    Each binding names ``counter_field``, ``direction``, ``unit_bytes`` and optional ``provenance``.
    There are deliberately no default counter names or beat sizes. A missing/unknown direction, value,
    unit, or directional binding keeps exact totals UNKNOWN while retaining every evidenced byte as a
    lower bound.
    """
    sums = {"read": 0, "write": 0}
    seen = {"read": False, "write": False}
    unknown = {"read": False, "write": False}
    provenance: dict[str, list[str]] = {"read": [], "write": []}
    unattributed = 0
    unattributed_provenance: list[str] = []
    unresolved: list[str] = []
    ambiguous_direction = False
    bound_fields: set[str] = set()
    for position, fact in enumerate(counter_facts):
        field_name = fact.get("counter_field")
        direction_value = fact.get("direction")
        direction = direction_value if direction_value in ("read", "write") else None
        source = _provenance(fact)
        receipt_problem = _receipt_problem(
            fact, kind="counter_byte_binding", allow_tool=False)
        count = _integer(readings.get(field_name)) if isinstance(field_name, str) else None
        unit = _integer(fact.get("unit_bytes"))
        byte_count = count * unit if (count is not None and count >= 0 and
                                      unit is not None and unit > 0 and receipt_problem is None) else None
        label = field_name if isinstance(field_name, str) else f"binding[{position}]"
        duplicate = isinstance(field_name, str) and field_name in bound_fields
        if not isinstance(field_name, str) or not field_name:
            unresolved.append(f"{label}: counter field is missing")
        elif duplicate:
            unresolved.append(f"{label}: duplicate counter binding")
            ambiguous_direction = True
        else:
            bound_fields.add(field_name)
        if receipt_problem:
            unresolved.append(f"{label}: unproven counter binding: {receipt_problem}")
        if duplicate:
            byte_count = None
        if direction is None:
            ambiguous_direction = True
            if byte_count is not None:
                unattributed += byte_count
            if source and source not in unattributed_provenance:
                unattributed_provenance.append(source)
            unresolved.append(f"{label}: counter direction is UNKNOWN")
            continue
        seen[direction] = True
        if source and source not in provenance[direction]:
            provenance[direction].append(source)
        if byte_count is None:
            unknown[direction] = True
            unresolved.append(f"{label}: counter value or byte unit is UNKNOWN")
        else:
            sums[direction] += byte_count
    extra = sorted(str(name) for name in readings if name not in bound_fields)
    if extra:
        ambiguous_direction = True
        unresolved.append(
            f"{len(extra)} counter reading(s) have no exhaustive byte binding: {extra[:4]}")
    absent = sorted(name for name in bound_fields if name not in readings)
    if absent:
        ambiguous_direction = True
        unresolved.append(f"{len(absent)} bound counter reading(s) are absent: {absent[:4]}")
    for direction in ("read", "write"):
        if not seen[direction]:
            unknown[direction] = True
            unresolved.append(f"no {direction} counter binding was supplied")
    read = None if unknown["read"] or ambiguous_direction else sums["read"]
    write = None if unknown["write"] or ambiguous_direction else sums["write"]
    return PhysicalVolume(
        read_bytes=read, write_bytes=write, read_lower_bound=sums["read"],
        write_lower_bound=sums["write"], unattributed_bytes=unattributed,
        is_lower_bound=bool(unresolved), unresolved=tuple(unresolved),
        read_provenance=tuple(provenance["read"]), write_provenance=tuple(provenance["write"]),
        unattributed_provenance=tuple(unattributed_provenance))


def compare_to_counters(
        volume: KernelVolume, readings: Mapping[str, Any], *,
        counter_facts: Sequence[Mapping[str, Any]], tolerance: float = 0.05) -> dict[str, Any]:
    """Compare scheduled descriptor bytes with physical counters without conflating their bases."""
    physical = physical_volume_from_counters(readings, counter_facts=counter_facts)
    scheduled_record = {
        "basis": volume.basis, "read_bytes": volume.read_bytes, "write_bytes": volume.write_bytes,
        "total_bytes": volume.exact_total_bytes, "known_lower_bound_bytes": volume.total_bytes,
        "is_lower_bound": volume.is_lower_bound, "read_provenance": volume.read_provenance,
        "write_provenance": volume.write_provenance,
    }
    physical_record = {
        "basis": physical.basis, "read_bytes": physical.read_bytes, "write_bytes": physical.write_bytes,
        "total_bytes": physical.total_bytes, "known_lower_bound_bytes": physical.known_lower_bound_bytes,
        "is_lower_bound": physical.is_lower_bound, "read_provenance": physical.read_provenance,
        "write_provenance": physical.write_provenance,
        "unattributed_provenance": physical.unattributed_provenance,
        "unresolved": physical.unresolved,
    }
    if physical.total_bytes is None:
        return {"kernel": volume.kernel, "verdict": "unknown_measurement",
                "scheduled": scheduled_record, "physical": physical_record}
    comparison = compare_to_measured(volume, physical.total_bytes, tolerance=tolerance)
    directional: dict[str, dict[str, Any]] = {}
    for direction in ("read", "write"):
        scheduled = getattr(volume, f"{direction}_bytes")
        measured = getattr(physical, f"{direction}_bytes")
        if volume.is_lower_bound:
            verdict = "consistent_lower_bound" if scheduled <= measured else "bound_violated"
        else:
            relative_error = (abs(scheduled - measured) / measured if measured
                              else (0.0 if scheduled == 0 else None))
            verdict = "match" if relative_error is not None and relative_error <= tolerance else "mismatch"
        directional[direction] = {
            "scheduled_bytes": scheduled, "physical_bytes": measured, "verdict": verdict,
            "scheduled_provenance": getattr(volume, f"{direction}_provenance"),
            "physical_provenance": getattr(physical, f"{direction}_provenance"),
        }
    if comparison["verdict"] == "match" and any(
            record["verdict"] != "match" for record in directional.values()):
        comparison["verdict"] = "directional_mismatch"
    comparison.update({"scheduled": scheduled_record, "physical": physical_record,
                       "directional": directional})
    return comparison


def compare_to_measured(volume: KernelVolume, measured_bytes: int, *, tolerance: float = 0.05
                        ) -> dict[str, Any]:
    """Predicted against measured, refusing a verdict a lower bound cannot support.

    A floor below the measurement is CONSISTENT, never a match: every unresolved descriptor could
    account for the gap. Only a fully resolved prediction can agree or disagree."""
    predicted = volume.total_bytes
    if volume.is_lower_bound:
        verdict = "consistent_lower_bound" if predicted <= measured_bytes else "bound_violated"
        return {"kernel": volume.kernel, "predicted": predicted, "measured": measured_bytes,
                "verdict": verdict, "is_lower_bound": True, "unresolved": volume.unresolved,
                "note": ("a floor cannot match; it can only fail to be exceeded"
                         if verdict == "consistent_lower_bound"
                         else "a LOWER bound above its measurement falsifies an input")}
    err = abs(predicted - measured_bytes) / measured_bytes if measured_bytes else None
    return {"kernel": volume.kernel, "predicted": predicted, "measured": measured_bytes,
            "verdict": "match" if (err is not None and err <= tolerance) else "mismatch",
            "relative_error": err, "is_lower_bound": False, "unresolved": ()}
