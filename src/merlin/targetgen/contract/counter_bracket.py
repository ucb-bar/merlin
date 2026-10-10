"""Header-free hardware-counter bracket for the logical (v2) runner-owned harness.

The logical harness includes no target header, so this module emits the counter commands as raw
RoCC instructions, built from DATA only:

* the counter command's funct CODE is the RTL-decoded code of the funct NAME the target contract
  declares (``facts.interfaces[funct_decode_table].names``), range-checked against the decoded field
  width; the custom opcode is the facts' ``custom_opcode``. Neither is a literal here;
* the bit layout of the command's configuration word (which bit resets, snapshots, configures, which
  field carries the slot and the event) is the contract's reviewed ``logical_harness.counter_bracket``
  block, with its RTL source cited there;
* the counter set and the event codes come from :func:`merlin.perf.hw_counters.counter_source_for_target`:
  the target's shipped counter header READ AS DATA on the host, or, when it declares none, the
  contract's reviewed counter-event table. Nothing is included in the generated C;
* the slot capacity is derived from the elaborated CIRCT (:func:`hw_counters.counter_slots_from_circt`)
  over the module and state families the contract names.

The RoCC transport itself (``.insn r opcode, funct3, funct7, rd, rs1, rs2`` with ``funct3`` carrying
the xd/xs1/xs2 bits) is the RISC-V RoCC convention, not a property of any accelerator. A counter read
returns its value in ``rd``, so every counter command is issued with all three bits set.

The emitted epilogue prints ``MERLIN_HWCOUNTER <name> <value>`` lines and the prologue prints the
``MERLIN_COUNTER_SCHEMA <header sha256>`` line, exactly the console protocol
:func:`hw_counters.parse_counter_output` / :func:`hw_counters.parse_counter_schema` read, so the existing
occupancy and physical-byte consumers apply unchanged. Anything underivable REFUSES the bracket: a
partial or mis-encoded bracket would be reported as a measurement.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The RoCC transport's xd | xs1 | xs2 funct3 bits (RISC-V RoCC convention): the command returns a
#: value (rd) and reads both source registers. Not an accelerator fact.
_ROCC_XD_XS1_XS2 = 0b111
_ROCC_FUNCT7_WIDTH = 7

_FIELD_NAMES = ("reset", "snapshot_take", "configure", "slot", "event")
_OPTIONAL_FIELDS = ("snapshot_reset", "external")

COUNTERS_ENV = "MERLIN_HW_COUNTERS"
UNIT_ENV = "MERLIN_HW_COUNTER_UNIT"


class CounterBracketError(ValueError):
    """The counter bracket cannot be derived; the run must not pretend it was instrumented."""


def counters_requested() -> bool:
    return str(os.environ.get(COUNTERS_ENV, "")).strip().lower() in ("1", "true", "yes", "on")


def unit_requested() -> str | None:
    value = str(os.environ.get(UNIT_ENV, "")).strip().upper()
    return value or None


@dataclass(frozen=True)
class Field:
    lsb: int
    width: int

    def place(self, value: int, *, name: str) -> int:
        if value < 0 or value >= (1 << self.width):
            raise CounterBracketError(f"counter field {name} cannot carry {value} in {self.width} bit(s)")
        return value << self.lsb


@dataclass(frozen=True)
class CounterCommand:
    """How to issue the target's counter command, with every value resolved from data."""

    opcode: int
    funct: int
    fields: Mapping[str, Field]
    external_event_base: str | None
    disabled_event: str | None
    capacity_module: str
    capacity_families: tuple[str, ...]
    evidence: str

    def word(self, **values: int) -> int:
        word = 0
        for name, value in values.items():
            field = self.fields.get(name)
            if field is None:
                raise CounterBracketError(f"the declared counter command has no {name!r} field")
            word |= field.place(int(value), name=name)
        return word


def _int(value: Any, *, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CounterBracketError(f"{what} must be a non-negative integer, got {value!r}")
    return value


def command_from_data(spec: Mapping[str, Any] | None, facts: Mapping[str, Any] | None) -> CounterCommand:
    """Resolve the contract's counter-bracket declaration against the RTL facts."""
    if not isinstance(spec, Mapping):
        raise CounterBracketError(
            "the target contract declares no logical_harness.counter_bracket, so no counter command is "
            "derivable for the logical harness"
        )
    from merlin.common.facts_view import interface

    body = (facts or {}).get("facts") if isinstance(facts, Mapping) and "facts" in facts else facts
    table = interface(body if isinstance(body, Mapping) else {}, "funct_decode_table")
    if not isinstance(table, Mapping):
        raise CounterBracketError("the RTL facts carry no funct_decode_table to resolve the counter command")
    name = spec.get("command")
    names = table.get("names") if isinstance(table.get("names"), Mapping) else {}
    codes = [int(code) for code, label in names.items() if str(label) == str(name) and str(code).isdigit()]
    if len(codes) != 1:
        raise CounterBracketError(
            f"the counter command {name!r} is not exactly one RTL-decoded funct in the facts "
            f"({len(codes)} match); refusing to emit an undecoded command"
        )
    funct = codes[0]
    legal = table.get("legal_funct")
    if isinstance(legal, list) and funct not in legal:
        raise CounterBracketError(f"funct {funct} is outside the RTL-decoded legal set")
    width = table.get("width") if isinstance(table.get("width"), int) else _ROCC_FUNCT7_WIDTH
    if funct >= (1 << int(width)):
        raise CounterBracketError(f"funct {funct} does not fit the decoded {width}-bit field")
    opcode = _int(table.get("custom_opcode"), what="facts funct_decode_table.custom_opcode")
    declared = spec.get("config_fields")
    if not isinstance(declared, Mapping):
        raise CounterBracketError("logical_harness.counter_bracket.config_fields is not declared")
    fields: dict[str, Field] = {}
    for key in _FIELD_NAMES + _OPTIONAL_FIELDS:
        raw = declared.get(key)
        if raw is None:
            if key in _FIELD_NAMES:
                raise CounterBracketError(f"counter_bracket.config_fields.{key} is not declared")
            continue
        if not isinstance(raw, Mapping):
            raise CounterBracketError(f"counter_bracket.config_fields.{key} must map lsb/width")
        field = Field(_int(raw.get("lsb"), what=f"{key}.lsb"), _int(raw.get("width"), what=f"{key}.width"))
        if field.width == 0 or field.lsb + field.width > 64:
            raise CounterBracketError(f"counter_bracket.config_fields.{key} does not fit a 64-bit operand")
        fields[key] = field
    occupied: dict[int, str] = {}
    for key, field in fields.items():
        for bit in range(field.lsb, field.lsb + field.width):
            if bit in occupied:
                raise CounterBracketError(f"counter fields {occupied[bit]} and {key} overlap at bit {bit}")
            occupied[bit] = key
    capacity = spec.get("slot_capacity") if isinstance(spec.get("slot_capacity"), Mapping) else {}
    families = capacity.get("state_families")
    if not isinstance(capacity.get("module"), str) or not isinstance(families, list) or not families:
        raise CounterBracketError("counter_bracket.slot_capacity must name a CIRCT module and state families")
    return CounterCommand(
        opcode=opcode,
        funct=funct,
        fields=fields,
        external_event_base=(str(spec["external_event_base"]) if spec.get("external_event_base") else None),
        disabled_event=(str(spec["disabled_event"]) if spec.get("disabled_event") else None),
        capacity_module=str(capacity["module"]),
        capacity_families=tuple(str(f) for f in families),
        evidence=str(spec.get("evidence") or ""),
    )


def _asm_helper(command: CounterCommand) -> str:
    return (
        "static inline uint64_t merlin_counter_op(uint64_t word) {\n"
        "  uint64_t value;\n"
        f'  __asm__ __volatile__(".insn r {command.opcode:#x}, {_ROCC_XD_XS1_XS2}, {command.funct:#x}, %0, %1, x0"'
        ' : "=r"(value) : "r"(word) : "memory");\n'
        "  return value;\n"
        "}\n"
    )


def render(
    command: CounterCommand,
    *,
    names: list[str] | tuple[str, ...],
    codes: Mapping[str, int],
    slots: int,
    schema_sha256: str,
    external: frozenset[str] | set[str] = frozenset(),
) -> dict[str, Any]:
    """C for configuring the named counters before the window and reading them after it.

    ``external`` names events of the separate external event space (selected with the command's
    external flag at their own code); a header-sourced set instead marks them by code above
    ``external_event_base``."""
    from merlin.perf.hw_counters import COUNTER_MARKER, COUNTER_SCHEMA_MARKER

    ordered = list(dict.fromkeys(str(n) for n in names))
    if not ordered:
        raise CounterBracketError("no counters were selected")
    if len(ordered) > slots:
        raise CounterBracketError(
            f"{len(ordered)} counter(s) need slots but the circuit exposes {slots}; a partial bracket would "
            "turn an exact measurement into an unlabelled lower bound"
        )
    missing = [n for n in ordered if n not in codes]
    if missing:
        raise CounterBracketError(f"no event code for {missing}")
    if len({codes[n] for n in ordered}) != len(ordered):
        raise CounterBracketError("selected counters share an event code")
    base = codes.get(command.external_event_base) if command.external_event_base else None
    if command.external_event_base and base is None:
        raise CounterBracketError(f"the counter header defines no {command.external_event_base!r}")

    def configure(slot: int, code: int, label: str) -> str:
        is_external = 0
        if label in external:
            if "external" not in command.fields:
                raise CounterBracketError(f"{label} is an external event but no external flag is declared")
            is_external = 1
        elif base is not None and code > base:
            if "external" not in command.fields:
                raise CounterBracketError(f"{label} is an external event but no external flag is declared")
            code, is_external = code - base, 1
        values = {"configure": 1, "slot": slot, "event": code}
        if is_external:
            values["external"] = 1
        return f"  (void)merlin_counter_op({command.word(**values):#x}ULL);  // slot {slot}: {label}"

    pro = [
        f'  printf("{COUNTER_SCHEMA_MARKER} {schema_sha256}\\n");',
        f"  (void)merlin_counter_op({command.word(reset=1):#x}ULL);  // reset the counter file",
    ]
    if "snapshot_reset" in command.fields:
        pro.append(f"  (void)merlin_counter_op({command.word(snapshot_reset=1):#x}ULL);  // clear any snapshot")
    for slot, name in enumerate(ordered):
        pro.append(configure(slot, int(codes[name]), name))
    if command.disabled_event:
        disabled = codes.get(command.disabled_event)
        if disabled is None:
            raise CounterBracketError(f"the counter header defines no {command.disabled_event!r}")
        for slot in range(len(ordered), slots):
            pro.append(configure(slot, int(disabled), "padding: disabled event"))
    epi = [f"  (void)merlin_counter_op({command.word(snapshot_take=1):#x}ULL);  // freeze the readings"]
    for slot, name in enumerate(ordered):
        word = command.word(slot=slot)
        epi.append(
            f'  printf("{COUNTER_MARKER} {name} %lu\\n", (unsigned long)(uint32_t)merlin_counter_op({word:#x}ULL));'
        )
    return {"helper": _asm_helper(command), "prologue": pro, "epilogue": epi, "names": tuple(ordered)}


def bracket_for_target(target: str, spec: Mapping[str, Any] | None, *, unit: str | None = None) -> dict[str, Any]:
    """The full bracket for ``target``: occupancy partition (``unit`` None) or the counters declaring
    ``unit`` (e.g. ``BYTES``), from the target's facts, contract, counter header and CIRCT."""
    from merlin.perf import hw_counters as hc
    from merlin.targetgen.rtl import facts as rtl_facts
    from merlin.targetgen.rtl import mlc_bridge

    command = command_from_data(spec, rtl_facts.load_facts(target))
    discovered = hc.counter_source_for_target(target)
    if discovered.get("status") != "derived":
        raise CounterBracketError(str(discovered.get("why") or "no counter set derived for this target"))
    try:
        occupancy, codes = hc.occupancy_for_discovery(discovered)
    except (OSError, ValueError) as exc:
        raise CounterBracketError(f"the counter event source is unreadable or changed: {exc}") from exc
    external = frozenset(discovered.get("external") or ())
    circt = mlc_bridge.core_hw_mlir(target)
    if circt is None or not Path(circt).is_file():
        raise CounterBracketError("the elaborated CIRCT is unavailable, so the counter slot capacity is unknown")
    capacity = hc.counter_slots_from_circt(
        Path(circt).read_text(encoding="utf-8", errors="replace"),
        module=command.capacity_module,
        state_families=command.capacity_families,
        source=str(circt),
    )
    if capacity.get("status") != "derived":
        raise CounterBracketError(str(capacity.get("why") or "counter slot capacity is not derivable"))
    if unit is None:
        if not occupancy.complete():
            raise CounterBracketError("the counter header does not derive a complete occupancy partition")
        names = [occupancy.by_combination[k] for k in sorted(occupancy.by_combination, key=lambda c: sorted(c))]
    else:
        names = sorted(n for n in codes if unit in n.upper().split("_"))
        if not names:
            raise CounterBracketError(f"the counter header declares no {unit!r} counters")
    out = render(
        command,
        names=names,
        codes=codes,
        slots=int(capacity["slots"]),
        schema_sha256=str(discovered["header_sha256"]),
        external=external,
    )
    out["slots"] = int(capacity["slots"])
    out["unit"] = unit
    return out
