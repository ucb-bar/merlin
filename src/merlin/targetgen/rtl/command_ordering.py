"""RTL-derived command ORDERING facts: which queue a command enters, what releases its dependencies,
which local rows it covers, and whether a queue's writes can land out of order.

WHY. A program is correct on a functional model that runs each command to completion in order, and
wrong on the hardware, when two commands write the same local rows and nothing makes the first one
COMPLETE before the second one ISSUES. Whether something does is a property of the dispatch unit
(does a younger command wait for an older one's issue or its completion?) and of the memory path
(can two transfers' responses land in either order?). Measured on one target: a zero-fill transfer
and the data transfers gathered into the same rows sat back to back in one queue; the dispatch table
released them at issue, the read client tracked 16 transactions, and the board kept the zeros in 8 of
12 runs while the functional model and the in-order RTL model passed.

HOW. Every value below is read from the selected elaborated HW, never assumed:

* The target contract names the dispatch module and its ports (``rtl_extraction.timing_structure
  .dispatch`` with an ``ordering`` block). :class:`Dispatch` drives that module alone, cycle by
  cycle (:mod:`.hw_cycle_eval`), with commands encoded from the target's own RoCC operand layouts.
* ``funct_queue``: one command per decoded funct (and per configuration selector code); the queue
  is the dispatch class whose issue port presents it.
* ``same_queue``: two identical commands; does the younger issue before the older is completed?
* ``cross_queue``: an older and a younger command of different queues at the same local row, then
  at a disjoint row; the younger waits for the older's completion only when they overlap.
* ``extents``: how many rows an access covers, found by moving a one-row partner until the
  dependency disappears. The formula ``(blocks - 1) * block_stride + rows * scale`` is searched
  over the address's sibling bundle fields and the declared configuration fields, then verified
  at fresh points; any mismatch leaves that extent UNKNOWN.
* ``completion_order``: a queue's writes may land out of order when its unit reaches a memory
  client whose response ID space and transaction slots both exceed one and the dispatch lets a
  second command proceed before the first completes. The client facts come from
  ``timing_structure``; a missing one leaves the order UNKNOWN, which consumers treat as
  out of order.

Premises recorded with the result: registers without a reset start at zero; ports the ordering
block does not name are held at zero; a probe observes the dispatch unit alone, not the memory
system behind it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "merlin.rtl_command_ordering.v1"
#: Cycles a probe waits for an issue before calling the command blocked.
_WAIT = 8
#: Spaces a local address can name; the accumulator space is selected by the declared flag.
SPACES = ("scratchpad", "accumulator")
#: A row far from any probe access, inside every space a probe uses.
_FAR_ROW = 96
#: A start row away from 0 for the condition scan (a moved start saturates at row 0).
_ORIGIN = 8


# ---------------------------------------------------------------------- declarations
def declarations(target: str) -> dict[str, Any] | None:
    """The dispatch block with its ``ordering`` declarations, or None when the target has none."""
    from .timing_structure import declarations as structure

    block = structure(target)
    dispatch = (block or {}).get("dispatch")
    if not isinstance(dispatch, dict) or not isinstance(dispatch.get("ordering"), dict):
        return None
    return dispatch


def _ports(dispatch: dict[str, Any]) -> dict[str, Any]:
    order = dispatch["ordering"]
    alloc, issue, done = order.get("allocate") or {}, order.get("issue") or {}, order.get("completion") or {}
    need = {
        "allocate": ("valid", "ready", "funct", "rs1", "rs2"),
        "issue": ("valid", "ready", "id", "funct", "rs1", "rs2"),
        "completion": ("valid", "id"),
    }
    for name, block in (("allocate", alloc), ("issue", issue), ("completion", done)):
        missing = [key for key in need[name] if not isinstance(block.get(key), str) or not block[key]]
        if missing:
            raise ValueError(f"dispatch.ordering.{name} declares no {', '.join(missing)}")
    classes = dispatch.get("classes")
    if not isinstance(classes, dict) or not classes:
        raise ValueError("dispatch declares no class issue prefixes")
    return {
        "reset": order.get("reset"),
        "allocate": alloc,
        "completion": done,
        "classes": {
            str(cls): {key: str(prefix) + issue[key] for key in need["issue"]} for cls, prefix in classes.items()
        },
        "config_fields": list(order.get("config_fields") or ()),
    }


# ---------------------------------------------------------------------- the ISA view a probe encodes with
@dataclass
class Access:
    """One local-address operand of a decoded class, located in the raw operands."""

    cls: str
    field: str
    operand: str
    offset: int
    width: int
    direction: str  # write | read
    siblings: dict[str, tuple[str, int, int]] = field(default_factory=dict)  # name -> (operand, offset, width)

    @property
    def key(self) -> str:
        return f"{self.cls}.{self.field}"


@dataclass
class IsaView:
    functs: list[int]
    funct_class: dict[int, str]
    config_class: str | None
    config_funct: int | None
    selector: tuple[str, int, int] | None
    subtypes: dict[int, str]
    accesses: dict[str, list[Access]]
    select_flag: int | None
    sentinel: int | None
    config_slots: list[dict[str, Any]]
    unknown: list[str] = field(default_factory=list)

    def funct_of(self, cls: str) -> int | None:
        for funct, name in self.funct_class.items():
            if name == cls:
                return funct
        if self.config_funct is not None and cls in self.subtypes.values():
            return self.config_funct
        return None

    def code_of(self, cls: str) -> int | None:
        for code, name in self.subtypes.items():
            if name == cls:
                return code
        return None


def isa_view(contract: dict[str, Any], facts_body: dict[str, Any], dispatch: dict[str, Any]) -> IsaView:
    """Read the funct table, operand layouts and local-address operands the probes encode with."""
    from merlin.common.facts_view import interface
    from merlin.perf.deps.rocc import DEFINES
    from merlin.targetgen import rtl_checks_generic as G
    from merlin.targetgen.rocc import semantics as S

    table = interface(facts_body, "funct_decode_table") or {}
    functs = [f for f in table.get("legal_funct") or () if type(f) is int]
    width = table.get("width") if type(table.get("width")) is int else S.ROCC_FUNCT7_WIDTH
    encoding = contract.get("encoding") or {}
    funct_class = S._numeric_code_map(encoding.get("semantic_class"), field="semantic_class", width=width)
    roles = S.operand_roles(contract)
    layouts = S.bundle_layouts({"facts": facts_body})
    config = roles.get("config") or {}
    selector_spec = config.get("selector") or {}
    slot = S.field_slot(layouts.get(selector_spec.get("bundle")), str(selector_spec.get("field")))
    selector = (str(selector_spec.get("operand", "rs1")), slot[0], slot[1]) if slot else None
    subtypes = (
        S._numeric_code_map(encoding.get("config_subtype"), field="config_subtype", width=slot[1])
        if slot and encoding.get("config_subtype") is not None
        else {}
    )
    config_class = config.get("class")
    config_funct = next((f for f, name in funct_class.items() if name == config_class), None)
    flags = S.local_address_flags(roles, encoding.get("addr_len"))
    protocol = G.protocol_from_contract(contract)
    unknown: list[str] = []
    accesses: dict[str, list[Access]] = {}
    specs = {str(cls): spec for cls, spec in (roles.get("classes") or {}).items() if isinstance(spec, dict)}
    for entry in protocol.local_addresses if protocol else ():
        for cls in sorted(protocol.classes(str(entry["role"]))):
            spec = next(
                (
                    f
                    for f in (specs.get(cls) or {}).get("fields") or ()
                    if isinstance(f, dict) and f.get("name") == entry["address"]
                ),
                None,
            )
            if spec is None or spec.get("operand") not in ("rs1", "rs2"):
                unknown.append(f"{cls}.{entry['address']}: no operand role locates this local address")
                continue
            layout = layouts.get(spec.get("bundle"))
            where = S.field_slot(layout, str(spec.get("field")))
            if where is None:
                unknown.append(
                    f"{cls}.{entry['address']}: the RTL layout does not carry {spec.get('bundle')}.{spec.get('field')}"
                )
                continue
            siblings = {}
            for name in (layout or {}).get("fields") or {}:
                sib = S.field_slot(layout, name)
                if name != spec.get("field") and sib is not None:
                    siblings[name] = (spec["operand"], sib[0], sib[1])
            direction = "write" if entry["address"] in DEFINES else "read"
            accesses.setdefault(cls, []).append(
                Access(cls, str(entry["address"]), spec["operand"], where[0], where[1], direction, siblings)
            )
    config_slots: list[dict[str, Any]] = []
    for ref in _ports(dispatch)["config_fields"]:
        layout = layouts.get((ref or {}).get("bundle"))
        operand = (ref or {}).get("operand")
        if not isinstance(layout, dict) or operand not in ("rs1", "rs2"):
            unknown.append(f"config field bundle {ref!r} is not in the RTL layouts")
            continue
        for name in layout.get("fields") or {}:
            where = S.field_slot(layout, name)
            if where is None:
                continue
            if (
                selector
                and operand == selector[0]
                and where[0] < selector[1] + selector[2]
                and selector[1] < where[0] + where[1]
            ):
                continue  # the selector bits name the configuration; they are not a value it latches
            config_slots.append(
                {"bundle": ref["bundle"], "field": name, "operand": operand, "offset": where[0], "width": where[1]}
            )
    retain = roles.get("retain_sentinel") or {}
    sentinel = None
    if isinstance(retain, dict) and retain.get("value") == "all_ones":
        where = S.field_slot(layouts.get(retain.get("bundle")), str(retain.get("field")))
        sentinel = (1 << where[1]) - 1 if where else None
    # a class no funct code names cannot be encoded or recognised in a stream: its accesses stay UNKNOWN
    config_codes = {name for name in subtypes.values()}
    for cls in sorted(accesses):
        if cls not in funct_class.values() and not (cls in config_codes and config_funct is not None):
            unknown.append(f"{cls}: no funct code names this class, so its accesses cannot be probed")
            del accesses[cls]
    return IsaView(
        functs=functs,
        funct_class=dict(funct_class),
        config_class=str(config_class) if config_class else None,
        config_funct=config_funct,
        selector=selector,
        subtypes=dict(subtypes),
        accesses=accesses,
        select_flag=flags.get("accumulator_select"),
        sentinel=sentinel,
        config_slots=config_slots,
        unknown=unknown,
    )


def _put(word: int, offset: int, width: int, value: int) -> int:
    mask = ((1 << width) - 1) << offset
    return (word & ~mask) | ((value << offset) & mask)


@dataclass(frozen=True)
class Command:
    funct: int
    rs1: int
    rs2: int


def _address(view: IsaView, space: str, row: int) -> int:
    if space == "accumulator":
        if view.select_flag is None:
            raise ValueError("no accumulator select flag is declared")
        return view.select_flag | row
    return row


def encode_access(
    view: IsaView, access: Access, *, space: str, row: int, values: dict[str, int] | None = None
) -> Command:
    """A command of ``access.cls`` whose ``access`` operand names ``row`` of ``space``; every sibling
    count field is 1 unless ``values`` names it, every other operand bit 0."""
    funct = view.funct_of(access.cls)
    if funct is None:
        raise ValueError(f"{access.cls} has no funct code")
    ops = {"rs1": 0, "rs2": 0}
    code = view.code_of(access.cls)
    if code is not None and view.selector:
        ops[view.selector[0]] = _put(ops[view.selector[0]], view.selector[1], view.selector[2], code)
    for name, (operand, offset, width) in access.siblings.items():
        ops[operand] = _put(ops[operand], offset, width, (values or {}).get(name, 1))
    ops[access.operand] = _put(ops[access.operand], access.offset, access.width, _address(view, space, row))
    # the other local operands of the same command name the OTHER space (or, with one space, a far
    # row), so only ``access`` can overlap a partner probed in ``space``
    elsewhere = [s for s in _spaces(view) if s != space]
    for other in view.accesses.get(access.cls, ()):
        if other.field != access.field and (other.operand, other.offset) != (access.operand, access.offset):
            far = _address(view, elsewhere[0], 0) if elsewhere else _address(view, space, _FAR_ROW * 8)
            ops[other.operand] = _put(ops[other.operand], other.offset, other.width, far)
    return Command(funct, ops["rs1"], ops["rs2"])


def encode_config(view: IsaView, code: int, slots: dict[tuple[str, int, int], int]) -> Command:
    ops = {"rs1": 0, "rs2": 0}
    if view.selector:
        ops[view.selector[0]] = _put(ops[view.selector[0]], view.selector[1], view.selector[2], code)
    for (operand, offset, width), value in slots.items():
        ops[operand] = _put(ops[operand], offset, width, value)
    return Command(int(view.config_funct), ops["rs1"], ops["rs2"])


# ---------------------------------------------------------------------- the dispatch harness
class Dispatch:
    """Drive the declared dispatch module alone: allocate, let classes issue, report completions."""

    def __init__(self, model, ports: dict[str, Any]):
        self.model, self.ports = model, ports
        self.cycles = 0
        self._snapshots: dict[tuple, list] = {}
        names = set(model.inputs) | set(model.outputs)
        wanted = [ports["allocate"][k] for k in ("valid", "ready", "funct", "rs1", "rs2")]
        wanted += [ports["completion"][k] for k in ("valid", "id")]
        wanted += [p for row in ports["classes"].values() for p in row.values()]
        missing = sorted(p for p in wanted if p not in names)
        if missing:
            raise ValueError(f"the dispatch module has no port(s) {missing}")

    def _tick(self, inputs: dict[str, int]) -> dict[str, int]:
        self.cycles += 1
        return self.model.step(inputs)

    def _fires(self, outputs: dict[str, int], ready: bool) -> list[tuple[str, int, Command]]:
        if not ready:
            return []
        fired = []
        for cls, row in self.ports["classes"].items():
            if outputs[row["valid"]]:
                fired.append(
                    (cls, outputs[row["id"]], Command(outputs[row["funct"]], outputs[row["rs1"]], outputs[row["rs2"]]))
                )
        return fired

    def _ready(self, on: bool) -> dict[str, int]:
        return {row["ready"]: int(on) for row in self.ports["classes"].values()}

    def _allocate(self, cmd: Command) -> bool:
        alloc = self.ports["allocate"]
        for _ in range(_WAIT):
            out = self._tick(
                {
                    alloc["valid"]: 1,
                    alloc["funct"]: cmd.funct,
                    alloc["rs1"]: cmd.rs1,
                    alloc["rs2"]: cmd.rs2,
                    **self._ready(False),
                }
            )
            if out[alloc["ready"]]:
                return True
        return False

    def _complete(self, ident: int) -> None:
        done = self.ports["completion"]
        self._tick({done["valid"]: 1, done["id"]: ident, **self._ready(False)})

    def start(self, prelude: tuple[Command, ...] = ()) -> bool:
        """Reset, then run ``prelude`` to completion. Snapshotted: a repeated prelude costs nothing."""
        key = tuple(prelude)
        if key in self._snapshots:
            self.model.state = list(self._snapshots[key])
            return True
        self.model.reset_state()
        if self.ports.get("reset"):
            for _ in range(2):
                self._tick({self.ports["reset"]: 1})
        for cmd in prelude:
            if not self._allocate(cmd):
                return False
            fired = self._run_until(lambda f, c=cmd: f[2] == c)
            if fired is None:
                return False
            self._complete(fired[1])
        self._snapshots[key] = list(self.model.state)
        return True

    def _run_until(self, match) -> tuple[str, int, Command] | None:
        for _ in range(_WAIT):
            out = self._tick(self._ready(True))
            for fire in self._fires(out, True):
                if match(fire):
                    return fire
        return None

    def queue_of(self, cmd: Command, prelude: tuple[Command, ...] = ()) -> str | None:
        if not self.start(prelude) or not self._allocate(cmd):
            return None
        fired = self._run_until(lambda f: f[2] == cmd)
        return fired[0] if fired else None

    def order(self, older: Command, younger: Command, prelude: tuple[Command, ...] = ()) -> str:
        """``issue`` (the younger issued before the older completed, after the older issued),
        ``none`` (the younger issued first), ``completion`` (it issued only once the older was
        reported complete) or ``unknown`` (a command never issued)."""
        if not self.start(prelude) or not self._allocate(older) or not self._allocate(younger):
            return "unknown"
        seen_older = None
        same = older == younger
        for _ in range(_WAIT):
            out = self._tick(self._ready(True))
            for cls, ident, cmd in self._fires(out, True):
                if seen_older is None and cmd == older:
                    seen_older = (cls, ident)
                    continue
                if cmd == younger and (not same or seen_older is not None):
                    return "issue" if seen_older is not None else "none"
        if seen_older is None:
            return "unknown"
        self._complete(seen_older[1])
        for _ in range(_WAIT):
            out = self._tick(self._ready(True))
            for _cls, _ident, cmd in self._fires(out, True):
                if cmd == younger:
                    return "completion"
        return "unknown"


# ---------------------------------------------------------------------- derivation
class Deriver:
    def __init__(self, dispatch: Dispatch, view: IsaView):
        self.d, self.view = dispatch, view
        self.unknown: list[str] = list(view.unknown)
        self.queues: dict[tuple[int, int | None], str | None] = {}

    # -- queues ---------------------------------------------------------------------------
    def funct_queue(self) -> list[dict[str, Any]]:
        rows = []
        view = self.view
        for funct in view.functs:
            if funct == view.config_funct and view.selector:
                operand, offset, width = view.selector
                codes = {}
                for code in range(1 << width):
                    ops = {"rs1": 0, "rs2": 0}
                    ops[operand] = _put(0, offset, width, code)
                    queue = self.d.queue_of(Command(funct, ops["rs1"], ops["rs2"]))
                    self.queues[(funct, code)] = queue
                    codes[str(code)] = queue
                rows.append(
                    {
                        "funct": funct,
                        "selector": {"operand": operand, "offset": offset, "width": width},
                        "by_code": codes,
                    }
                )
                continue
            queue = self.d.queue_of(Command(funct, 0, 0))
            stable = all(self.d.queue_of(Command(funct, rs1, rs2)) == queue for rs1, rs2 in _PATTERNS)
            self.queues[(funct, None)] = queue if stable else None
            row: dict[str, Any] = {"funct": funct, "queue": queue if stable else None}
            if not stable:
                row["reason"] = "the queue changed with operand bits no declared selector names"
            elif queue is None:
                row["reason"] = "the dispatch unit never presented this funct at an issue port"
            rows.append(row)
        return rows

    def queue_for(self, cls: str) -> str | None:
        funct = self.view.funct_of(cls)
        code = self.view.code_of(cls)
        return (
            self.queues.get((funct, code if funct == self.view.config_funct else None)) if funct is not None else None
        )

    # -- preludes -------------------------------------------------------------------------
    def config_codes(self) -> list[int]:
        return sorted(
            code
            for (funct, code), q in self.queues.items()
            if funct == self.view.config_funct and code is not None and q
        )

    # -- same queue -----------------------------------------------------------------------
    def same_queue(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        prelude: tuple[Command, ...] = ()  # reset configuration: dependency tracking is not configured
        for queue in sorted({q for q in self.queues.values() if q}):
            accesses = [a for cls, rows in self.view.accesses.items() if self.queue_for(cls) == queue for a in rows]
            if not accesses:
                out[queue] = {"release": "unknown", "reason": "no decoded class with a local operand enters this queue"}
                continue
            seen = {}
            for access in accesses:
                cmd = encode_access(self.view, access, space=SPACES[0], row=0)
                seen[access.cls] = self.d.order(cmd, cmd, prelude)
            kinds = set(seen.values())
            release = kinds.pop() if len(kinds) == 1 else "unknown"
            out[queue] = {"release": release, "by_class": seen}
            if release == "unknown":
                self.unknown.append(f"queue {queue}: same-queue release could not be established {seen}")
        return out

    # -- cross queue ----------------------------------------------------------------------
    def cross_queue(self) -> list[dict[str, Any]]:
        rows = []
        # At reset. A span the configuration scales to zero still names its start row: the dispatch
        # compares ranges as start-inclusive, so an empty range overlaps any range containing its start.
        prelude: tuple[Command, ...] = ()
        flat = [a for rows_ in self.view.accesses.values() for a in rows_]
        for older in flat:
            for younger in flat:
                qo, qy = self.queue_for(older.cls), self.queue_for(younger.cls)
                if not qo or not qy or qo == qy:
                    continue
                for space in _spaces(self.view):
                    a = encode_access(self.view, older, space=space, row=0)
                    near = encode_access(self.view, younger, space=space, row=0)
                    far = encode_access(self.view, younger, space=space, row=_FAR_ROW)
                    overlap, disjoint = self.d.order(a, near, prelude), self.d.order(a, far, prelude)
                    tracked = overlap == "completion" and disjoint in ("issue", "none")
                    rows.append(
                        {
                            "older": older.key,
                            "younger": younger.key,
                            "space": space,
                            "overlap": overlap,
                            "disjoint": disjoint,
                            "tracked": True if tracked else (False if overlap in ("issue", "none") else None),
                        }
                    )
        return rows

    # -- extents --------------------------------------------------------------------------
    def extents(self, cross: list[dict[str, Any]]) -> dict[str, Any]:
        out = {}
        flat = {a.key: a for rows_ in self.view.accesses.values() for a in rows_}
        for key, access in sorted(flat.items()):
            partners = [
                (row["younger"], row["space"]) for row in cross if row["older"] == key and row["tracked"] is True
            ]
            if not partners:
                out[key] = {"status": "unknown", "reason": "no partner in another queue waits on this access"}
                self.unknown.append(f"{key}: extent unknown (no usable partner)")
                continue
            partner_key, space = partners[0]
            try:
                out[key] = _ExtentSearch(self, access, flat[partner_key], space).run()
            except _Undecided as exc:
                out[key] = {"status": "unknown", "reason": str(exc)}
                self.unknown.append(f"{key}: extent unknown ({exc})")
        return out

    def _config(self, assignment: dict[int, dict[tuple[str, int, int], int]]) -> tuple[Command, ...]:
        return tuple(encode_config(self.view, code, slots) for code, slots in sorted(assignment.items()) if slots)


def _merge(*assignments: dict) -> dict:
    out: dict = {}
    for assign in assignments:
        for code, slots in assign.items():
            out.setdefault(code, {}).update(slots)
    return out


class _ExtentSearch:
    """``(blocks - 1) * stride + rows * scale`` for one access, the configuration fields under which
    that formula was seen to hold, and its verification at fresh points.

    Every question is "does a one-row partner at row k wait for this access's completion?", which is
    true exactly when k lies inside the covered span, so a span is measured by bisection and a
    candidate formula is checked with two probes (covers k-1, not k)."""

    def __init__(self, deriver: Deriver, access: Access, partner: Access, space: str):
        self.r, self.a, self.p, self.space = deriver, access, partner, space
        self.view = deriver.view
        self.codes = deriver.config_codes()
        seen: dict[tuple[str, int, int], dict] = {}
        for spec in self.view.config_slots:
            seen.setdefault((spec["operand"], spec["offset"], spec["width"]), spec)
        self.slots = seen
        self.base = {name: 3 for name in access.siblings}

    def covers(self, values: dict[str, int], assign: dict, row: int, *, origin: int = 0) -> bool:
        older = encode_access(self.view, self.a, space=self.space, row=origin, values={**self.base, **values})
        younger = encode_access(self.view, self.p, space=self.space, row=row)
        return self.r.d.order(older, younger, self.r._config(assign)) == "completion"

    def measure(self, values: dict[str, int], assign: dict) -> int:
        if not self.covers(values, assign, 0):
            return 0
        lo, hi = 0, 1
        while self.covers(values, assign, hi):
            lo, hi = hi, hi * 2
            if hi > 1 << 12:
                raise _Undecided("the covered span exceeds the probed bound")
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.covers(values, assign, mid):
                lo = mid
            else:
                hi = mid
        return hi

    def spans(self, values: dict[str, int], assign: dict, expect: int, *, origin: int = 0) -> bool:
        # an empty span still names its start row (start-inclusive comparison), so it reads as one
        expect = max(1, expect)
        if origin and self.covers(values, assign, origin - 1, origin=origin):
            return False  # the span starts below the addressed row
        return self.covers(values, assign, origin + expect - 1, origin=origin) and not self.covers(
            values, assign, origin + expect, origin=origin
        )

    def run(self) -> dict[str, Any]:
        reset = self.measure({}, {})
        scale = None
        for code in self.codes:
            for key in self.slots:
                if key[2] >= 2 and self.spans({}, {code: {key: 2}}, 6) and self.spans({}, {code: {key: 3}}, 9):
                    scale = (code, key)
                    break
            if scale:
                break
        factor = 2 if scale else 1
        assign = {scale[0]: {scale[1]: 2}} if scale else {}
        if not self.spans({}, assign, 3 * factor):
            raise _Undecided(f"no rows-times-scale form explains the covered span ({self.measure({}, assign)} rows)")
        rows = next((n for n in self.a.siblings if self.spans({n: 5}, assign, 5 * factor)), None)
        if rows is None:
            raise _Undecided("no sibling field scales the covered span")
        if scale is None and reset != 3:
            raise _Undecided(f"unscaled span {reset} at reset differs from the 3 encoded rows")
        # 3 rows at the reset scale: a multiple of 3, or the one start row an empty span still names
        reset_scale = reset // 3 if reset % 3 == 0 else (0 if reset == 1 else None)
        if scale is not None and reset_scale is None:
            raise _Undecided(f"the span at reset ({reset}) is not the rows times any scale")
        blocks = self.blocks(rows, assign, factor)
        record: dict[str, Any] = {
            "status": "derived",
            "rows": _slot(self.a.siblings[rows], rows),
            "scale": None if scale is None else {"code": scale[0], **_slot(scale[1]), "reset": reset_scale},
            "blocks": None if blocks is None else blocks["record"],
            "partner": self.p.key,
            "space": self.space,
        }
        used = {(scale[0], scale[1])} if scale else set()
        if blocks is not None:
            used |= blocks["used"]
        record["conditions"] = self.conditions(rows, assign, factor, used)
        checks = []
        for r, s in ((1, 3), (4, 2), (7, 1)):
            point = {scale[0]: {scale[1]: s}} if scale else {}
            expect = r * (s if scale else 1)
            if not self.spans({rows: r}, point, expect):
                raise _Undecided(f"the formula predicts {expect} rows at rows={r}, scale={s}; the RTL disagrees")
            checks.append([r, s if scale else None, expect])
        if blocks is not None:
            b = blocks
            cols = b["record"]["cols"]["name"]
            for n in (2, 3):
                expect = (n - 1) * b["per"] + 3 * factor
                if not self.spans({rows: 3, cols: (n - 1) * b["record"]["block"] + 1}, b["assign"], expect):
                    raise _Undecided(f"the block formula predicts {expect} rows for {n} blocks; the RTL disagrees")
                checks.append(["blocks", n, expect])
        record["verified"] = checks
        return record

    def blocks(self, rows: str, assign: dict, factor: int) -> dict[str, Any] | None:
        single = 3 * factor
        for name, (_op, _off, width) in self.a.siblings.items():
            if name == rows:
                continue
            big = min((1 << width) - 1, 64)
            for source in self._stride_sources(assign, name, rows, big, single):
                try:
                    found = self._block_formula(source, assign, rows, name, big, single)
                except _Undecided:
                    continue  # this candidate does not add whole blocks; try the next
                if found is not None:
                    return found
        return None

    def _block_formula(self, source, assign, rows, name, big, single):
        test = _merge(assign, source["assign"])
        lo, hi = 1, big  # smallest value of ``name`` that adds a block: covers row ``single``
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.covers({rows: 3, name: mid}, test, single):
                hi = mid
            else:
                lo = mid
        block = lo
        if block < 1 or block * 2 + 1 > big:
            return None
        per = self.measure({rows: 3, name: block + 1}, test) - single
        if per <= 0 or (source["value"] is not None and per != source["value"]):
            return None
        if not self.spans({rows: 3, name: 2 * block + 1}, test, 2 * per + single):
            return None
        record = {"cols": _slot(self.a.siblings[name], name), "block": block}
        record["stride"] = source["slot"] if source["slot"] else {"constant": per}
        return {"record": record, "assign": test, "per": per, "used": source["used"]}

    def _stride_sources(self, assign, name, rows, big, single):
        if self.covers({rows: 3, name: big}, assign, single):
            yield {"assign": {}, "slot": None, "value": None, "used": set()}
            return
        wide = [k for k in self.slots if k[2] >= 4]
        small = [k for k in self.slots if k[2] <= 3]
        for code in self.codes:
            for state_value in range(4):
                group = {k: 7 for k in wide}
                group.update({k: state_value & ((1 << k[2]) - 1) for k in small})
                if not self.covers({rows: 3, name: big}, _merge(assign, {code: group}), single):
                    continue
                for slot in wide:
                    states = [k for k in small if self.slots[k].get("bundle") == self.slots[slot].get("bundle")]
                    for state in states:
                        for value in range(1, 1 << state[2]):
                            trial = {slot: 7, state: value}
                            # a state field selects this stride only if its default value does not
                            if self.covers(
                                {rows: 3, name: big}, _merge(assign, {code: trial}), single
                            ) and not self.covers({rows: 3, name: big}, _merge(assign, {code: {slot: 7}}), single):
                                out = {
                                    "code": code,
                                    **_slot(slot),
                                    "reset": 0,
                                    "state": {**_slot(state), "value": value},
                                }
                                yield {
                                    "assign": {code: trial},
                                    "slot": out,
                                    "value": 7,
                                    "used": {(code, slot), (code, state)},
                                }
                # a state field at its default value selects this access's stride: the stride is
                # keyed by that field when setting it elsewhere makes the stride stop applying
                for slot in wide:
                    if not self.covers({rows: 3, name: big}, _merge(assign, {code: {slot: 7}}), single):
                        continue
                    out = {"code": code, **_slot(slot), "reset": 0}
                    used = {(code, slot)}
                    for state in [k for k in small if self.slots[k].get("bundle") == self.slots[slot].get("bundle")]:
                        moved = [
                            v
                            for v in range(1, 1 << state[2])
                            if not self.covers(
                                {rows: 3, name: big}, _merge(assign, {code: {slot: 7, state: v}}), single
                            )
                        ]
                        if moved:
                            out["state"] = {**_slot(state), "value": 0}
                            used.add((code, state))
                            break
                    yield {"assign": {code: {slot: 7}}, "slot": out, "value": 7, "used": used}

    def conditions(self, rows: str, assign: dict, factor: int, used: set) -> list[dict[str, Any]]:
        """Configuration fields that change the span when set: the formula holds only while each
        keeps its reset value, so a consumer must treat a program that sets one as unmodeled."""
        found = []
        for code in self.codes:
            # a slot sharing bits with a modeled field is that field, not another condition
            taken = [k for c, k in used if c == code]
            free = [k for k in self.slots if not any(_bits_overlap(k, t) for t in taken)]
            # distinct sibling values, so a field that swaps which sibling counts rows is visible
            values = {name: (3 if name == rows else 5) for name in self.a.siblings}
            group = {k: 1 for k in free}
            # away from row 0 too, so a field that moves the start (it can only saturate at 0) shows
            if self.spans(values, _merge(assign, {code: group}), 3 * factor, origin=_ORIGIN):
                continue
            for key in free:
                if not self.spans(values, _merge(assign, {code: {key: 1}}), 3 * factor, origin=_ORIGIN):
                    found.append(
                        {
                            "code": code,
                            **_slot(key),
                            "reset": 0,
                            "field": f"{self.slots[key].get('bundle')}.{self.slots[key].get('field')}",
                        }
                    )
        return found


class _Undecided(Exception):
    pass


def _bits_overlap(a: tuple[str, int, int], b: tuple[str, int, int]) -> bool:
    return a[0] == b[0] and a[1] < b[1] + b[2] and b[1] < a[1] + a[2]


#: Operand patterns used to confirm a funct's queue does not depend on bits no selector names.
_PATTERNS = ((0xFFFF_FFFF_FFFF_FFFF, 0), (0, 0xFFFF_FFFF_FFFF_FFFF), (0x5555_5555_5555_5555, 0xAAAA_AAAA_AAAA_AAAA))


def _slot(where: tuple[str, int, int], name: str | None = None) -> dict[str, Any]:
    out = {"operand": where[0], "offset": where[1], "width": where[2]}
    if name is not None:
        out["name"] = name
    return out


def _spaces(view: IsaView) -> tuple[str, ...]:
    return SPACES if view.select_flag is not None else SPACES[:1]


# ---------------------------------------------------------------------- completion order
def completion_order(structure: dict[str, Any] | None, same_queue: dict[str, Any]) -> dict[str, Any]:
    """Per queue: can two of its writes land in either order? ``out_of_order`` when its unit reaches
    a memory client with more than one response ID and transaction slot and the dispatch lets a
    second command start before the first completes; ``in_order`` when no memory client is on its
    path; UNKNOWN (consumers assume out of order) when the structure facts are missing."""
    out: dict[str, Any] = {}
    classes = ((structure or {}).get("command_classes") or {}).get("classes") or {}
    clients = ((structure or {}).get("memory_clients") or {}).get("clients") or {}
    for queue, release in sorted(same_queue.items()):
        row = classes.get(queue)
        if not isinstance(row, dict) or not clients:
            out[queue] = {
                "order": "unknown",
                "reason": "timing_structure carries no command class or memory client facts",
            }
            continue
        hops = [hop.get("instance") for hop in row.get("downstream_buffers") or () if isinstance(hop, dict)]
        reached = []
        for name, client in sorted(clients.items()):
            inst = client.get("instance")
            if isinstance(inst, str) and any(
                isinstance(h, str) and (h == inst or h.startswith(inst + "/")) for h in hops
            ):
                reached.append((name, client))
        if not reached:
            out[queue] = {
                "order": "in_order",
                "basis": "no memory client is on this queue's path; its unit writes in command order",
                "downstream": hops,
            }
            continue
        reasons = []
        reorder = False
        for name, client in reached:
            ids = client.get("id_space")
            inflight = (client.get("in_flight") or {}).get("value")
            if type(ids) is not int or type(inflight) is not int:
                reorder = None if reorder is False else reorder
                reasons.append(f"{name}: response ID space or transaction slots UNKNOWN")
                continue
            if ids > 1 and inflight > 1:
                reorder = True
            reasons.append(f"{name}: {ids} response IDs, {inflight} transaction slots")
        if release.get("release") == "completion":
            out[queue] = {
                "order": "in_order",
                "basis": "a younger command waits for the older one's completion",
                "clients": reasons,
            }
        elif reorder is True and release.get("release") == "issue":
            out[queue] = {
                "order": "out_of_order",
                "clients": reasons,
                "basis": "the dispatch releases a younger command at the older one's issue, and the "
                "memory client tracks several transactions whose responses may return in any order",
            }
        elif reorder is False:
            out[queue] = {"order": "in_order", "clients": reasons, "basis": "a single response ID or transaction slot"}
        else:
            out[queue] = {
                "order": "unknown",
                "clients": reasons,
                "reason": "the release or client facts are incomplete",
            }
    return out


# ---------------------------------------------------------------------- top level
def derive(
    design, *, dispatch: dict[str, Any], contract: dict[str, Any], facts_body: dict[str, Any], root: str | None = None
) -> dict[str, Any]:
    """Every ordering fact for one selected elaboration."""
    from .hw_cycle_eval import CycleModel

    ports = _ports(dispatch)
    module = dispatch.get("module")
    if not isinstance(module, str):
        raise ValueError("dispatch declares no module")
    instance = None
    if root:
        found = design.occurrences(root, module)
        if len(found) != 1:
            raise ValueError(f"dispatch module {module} does not occur exactly once under {root}")
        instance = found[0]
    view = isa_view(contract, facts_body, dispatch)
    model = CycleModel(design, module)
    harness = Dispatch(model, ports)
    deriver = Deriver(harness, view)
    funct_queue = deriver.funct_queue()
    same = deriver.same_queue()
    cross = deriver.cross_queue()
    extents = deriver.extents(cross)
    order = completion_order(facts_body.get("timing_structure"), same)
    accesses = {
        a.key: {
            "class": a.cls,
            "field": a.field,
            "operand": a.operand,
            "offset": a.offset,
            "width": a.width,
            "direction": a.direction,
            "queue": deriver.queue_for(a.cls),
            "funct": view.funct_of(a.cls),
            "code": view.code_of(a.cls) if view.funct_of(a.cls) == view.config_funct else None,
        }
        for rows in view.accesses.values()
        for a in rows
    }
    status = "derived" if not deriver.unknown else "partial"
    return {
        "schema": SCHEMA,
        "status": status,
        "method": "directed cycle evaluation of the declared dispatch module's elaborated HW",
        "dispatch": {"module": module, "instance": instance},
        "premises": [
            f"{model.unreset_registers} registers without a reset start at zero",
            "dispatch ports the ordering declaration does not name are held at zero",
            "a probe observes the dispatch unit alone; memory-path reordering comes from completion_order",
        ],
        "funct_queue": funct_queue,
        "accesses": accesses,
        "same_queue": same,
        "cross_queue": cross,
        "extents": extents,
        "completion_order": order,
        "classes": {
            "by_funct": {str(f): c for f, c in sorted(view.funct_class.items())},
            "config": {
                "funct": view.config_funct,
                "selector": _slot(view.selector) if view.selector else None,
                "by_code": {str(k): v for k, v in sorted(view.subtypes.items())},
            },
        },
        "local_address": {"accumulator_select": view.select_flag, "sentinel": view.sentinel},
        "probe_cycles": harness.cycles,
        "unknown": sorted(set(deriver.unknown)),
    }


def add_to_facts(target: str, facts: dict[str, Any]) -> list[str]:
    """Attach ``command_ordering`` for the active source selection; UNKNOWN when it cannot run."""
    from .source_selection import active_selection, digest

    dispatch = declarations(target)
    selected = active_selection(target)
    if dispatch is None or selected is None:
        return []
    receipt = selected.get("_genericization") or {}
    generic = (receipt.get("output") or {}).get("path")
    if not generic or digest(generic) != receipt["output"]["sha256"]:
        facts["command_ordering"] = {
            "schema": SCHEMA,
            "status": "unknown",
            "reason": "no receipted generic HW serialization of the selected core",
        }
        return []
    record = derive_from_paths(target, Path(generic), facts)
    facts["command_ordering"] = record
    return [f"command_ordering({record['status']})"]


def derive_from_paths(
    target: str, generic_hw: Path, facts_body: dict[str, Any], *, contract: dict | None = None
) -> dict:
    """Derive from a generic HW file; failures are recorded UNKNOWN with their reason, never raised."""
    import yaml

    from .facts import target_contract_path
    from .hw_design import load_design

    if contract is None:
        contract = yaml.safe_load(Path(target_contract_path(target)).read_text(encoding="utf-8")) or {}
    dispatch = ((contract.get("rtl_extraction") or {}).get("timing_structure") or {}).get("dispatch")
    if not isinstance(dispatch, dict) or not isinstance(dispatch.get("ordering"), dict):
        return {"schema": SCHEMA, "status": "unknown", "reason": "the target declares no dispatch ordering ports"}
    text = Path(generic_hw).read_text()
    try:
        root = (contract.get("rtl_extraction") or {}).get("timing_structure", {}).get("unit_root")
        root = root or ((facts_body.get("timing_structure") or {}).get("unit_root"))
        record = derive(load_design(text), dispatch=dispatch, contract=contract, facts_body=facts_body, root=root)
    except (ValueError, KeyError) as exc:
        record = {"schema": SCHEMA, "status": "unknown", "reason": f"{type(exc).__name__}: {exc}"}
    record["provenance"] = {
        "core_hw_generic": {"path": str(generic_hw), "sha256": hashlib.sha256(text.encode()).hexdigest()},
        "declarations_sha256": hashlib.sha256(json.dumps(dispatch, sort_keys=True).encode()).hexdigest(),
        "readers": [
            {"path": str(Path(m).resolve()), "sha256": hashlib.sha256(Path(m).read_bytes()).hexdigest()}
            for m in (__file__, Path(__file__).with_name("hw_cycle_eval.py"))
        ],
    }
    return record


def main(argv: list[str] | None = None) -> int:
    """``merlin-command-ordering --target T --hw core.hw.generic.mlir --facts facts.json [--write]``."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", required=True)
    parser.add_argument("--hw", required=True, help="generic CIRCT HW of the selected core")
    parser.add_argument("--facts", required=True, help="the facts artifact to read (and, with --write, update)")
    parser.add_argument("--write", action="store_true", help="store the section into --facts")
    args = parser.parse_args(argv)
    path = Path(args.facts)
    doc = json.loads(path.read_text(encoding="utf-8"))
    body = doc.get("facts", doc)
    record = derive_from_paths(args.target, Path(args.hw), body)
    if args.write:
        body["command_ordering"] = record
        path.write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: record.get(k) for k in ("status", "same_queue", "completion_order", "unknown", "probe_cycles")},
            indent=1,
        )
    )
    return 0 if record.get("status") in ("derived", "partial") else 1


if __name__ == "__main__":
    raise SystemExit(main())
