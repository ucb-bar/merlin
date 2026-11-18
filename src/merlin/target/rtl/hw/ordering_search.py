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

from merlin.target.rtl.hw.command_ordering import _ORIGIN, Access, Deriver, encode_access

# ---- command_ordering --------------------------------------------------------------------------


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


_PATTERNS = ((0xFFFF_FFFF_FFFF_FFFF, 0), (0, 0xFFFF_FFFF_FFFF_FFFF), (0x5555_5555_5555_5555, 0xAAAA_AAAA_AAAA_AAAA))


def _slot(where: tuple[str, int, int], name: str | None = None) -> dict[str, Any]:
    out = {"operand": where[0], "offset": where[1], "width": where[2]}
    if name is not None:
        out["name"] = name
    return out


def _spaces(view: IsaView) -> tuple[str, ...]:
    from merlin.target.rtl.hw.command_ordering import SPACES, IsaView

    return SPACES if view.select_flag is not None else SPACES[:1]


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


def derive(
    design, *, dispatch: dict[str, Any], contract: dict[str, Any], facts_body: dict[str, Any], root: str | None = None
) -> dict[str, Any]:
    """Every ordering fact for one selected elaboration."""
    from merlin.target.rtl.hw.command_ordering import SCHEMA, Deriver, Dispatch, _ports, isa_view
    from merlin.target.rtl.hw.evaluation import CycleModel

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
    from merlin.target.rtl.extraction.source_selection import active_selection, digest
    from merlin.target.rtl.hw.command_ordering import SCHEMA, declarations

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

    from merlin.target.rtl.facts import target_contract_path
    from merlin.target.rtl.hw.command_ordering import SCHEMA
    from merlin.target.rtl.hw.design import load_design

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
