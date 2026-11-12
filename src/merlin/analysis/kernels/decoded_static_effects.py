"""Recover static operand geometry and arithmetic effects from selected decoding contracts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

_DOMAIN = 4096

SCHEMA = "merlin.derived_effects.v1"

ORDERING = "lexical_listing_not_dynamic_issue_trace"

_MOVEMENT_ROLES = ("operand_load", "readout", "commit")

_WITNESSES = 4

_VALUES = 16


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _fact(value, status, *, scope=None, evidence=(), reason=None, **extra):
    fact = dict(value=value, status=status, scope=scope or {}, evidence=list(evidence))
    if reason is not None:
        fact["reason"] = reason
    fact.update(extra)
    return fact


def _unknown(reason, **extra):
    return _fact(None, "unknown", reason=reason, **extra)


def read_pinned_sidecar(selection):
    """Read the contract's pinned operand-protocol sidecar, refusing changed bytes."""
    sidecar = (selection or {}).get("sidecar") or {}
    path, expected = sidecar.get("path"), sidecar.get("sha256")
    if not path or not expected:
        return None
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("selected operand-protocol sidecar differs from its pin")
    return json.loads(raw)


def read_pinned_facts(selection):
    """Read the contract's pinned original RTL facts, refusing changed bytes; ``None`` when unpinned."""
    pin = (selection or {}).get("original_facts") or {}
    path, expected = pin.get("path"), pin.get("sha256")
    if not path or not expected:
        return None
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("selected original RTL facts differ from their pin")
    return json.loads(raw)


def local_row_geometry(target, facts):
    """Elements one local-memory row holds, per derived on-chip store, from the selected RTL facts.

    ``uniform_row_elements`` is set only when every store with a derived row width agrees, so a
    transfer's row count per source row needs no store-to-flag mapping; otherwise it stays ``None``
    and wide-transfer footprints stay unknown."""
    if not target or facts is None:
        return dict(
            status="unknown", reason="selected target or RTL facts unavailable", stores=[], uniform_row_elements=None
        )
    from merlin.development.targetgen.address_space import derive_address_space

    space = derive_address_space(target, facts=facts)
    stores = [dict(name=store.name, row_elements=store.elems_per_row()) for store in space.stores]
    widths = {row["row_elements"] for row in stores}
    uniform = next(iter(widths)) if len(widths) == 1 and type(next(iter(widths))) is int else None
    return dict(
        status="derived" if uniform else "unknown",
        stores=stores,
        uniform_row_elements=uniform,
        **({} if uniform else dict(reason="local stores do not share one derived row width")),
        source="targetgen.address_space.derive_address_space(selected RTL facts)",
    )


def selected_structure(decoder, contract_document):
    """Assemble the declared inputs ``derive_effects`` reads from the builder's selections.

    ``decoder`` is ``reference_analysis._operand_decoder(..., include_structure=True)``;
    ``contract_document`` is the parsed selected target contract.
    """
    contract_document = contract_document or {}
    base = dict((decoder or {}).get("structure") or {})
    selection = contract_document.get("rocc_operand_selection")
    sidecar = read_pinned_sidecar(selection)
    protocol = (sidecar or {}).get("protocol") or contract_document.get("rocc_operand_protocol")
    observer = (decoder or {}).get("semantics")
    states = getattr(observer, "load_states", None)
    ports = dict(states.ports) if states is not None else None
    facts = read_pinned_facts(selection)
    structure = dict(
        instruction_roles=base.get("instruction_roles") or {},
        compute_units=base.get("compute_units") or contract_document.get("compute_units") or [],
        output_epilogue=base.get("output_epilogue") or contract_document.get("output_epilogue") or {},
        operand_protocol=protocol or {},
        field_layouts=((sidecar or {}).get("layouts") or {}).get("fields") or [],
        load_state_selection=(sidecar or {}).get("load_state_selection") or {},
        load_state_ports=ports,
        operand_field_kind_sources=list((sidecar or {}).get("operand_field_kind_sources") or []),
        local_geometry=local_row_geometry((decoder or {}).get("target"), facts),
        encoding=contract_document.get("encoding") or {},
        input_pins=list((decoder or {}).get("decoder_inputs") or []) + list((decoder or {}).get("role_inputs") or []),
    )
    from merlin.analysis.kernels.command_routing import command_queue_routing
    from merlin.analysis.kernels.stream_resources import stream_bounds

    # Queue routing and RTL bounds of the stream resource facts, from the same pinned facts.
    structure["command_queue_routing"] = command_queue_routing(facts, structure) if facts is not None else None
    structure["stream_bounds"] = stream_bounds(facts)
    return structure


def decoded_commands(rows, *, decoder, entry_state=None, call_summaries=None):
    """Every selected accelerator site of ``rows`` in listing order, via the selected decoder.

    The same primitives as ``machine_projection`` (register known-bits, RoCC word
    fields and the bound operand observer); nothing is retained beyond the record.
    ``entry_state``/``call_summaries`` reach the lifter only when it accepts them.
    """
    from merlin.analysis.kernels.machine_evidence import lifter_call
    from merlin.analysis.riscv_machine import register_facts
    from merlin.analysis.rocc_commands import fields_of

    semantics, target = (decoder or {}).get("semantics"), (decoder or {}).get("target")
    if semantics is None:
        return None
    isa = semantics.isa_constants(target)
    states, _ = lifter_call(register_facts, rows, entry_state=entry_state, call_summaries=call_summaries)
    out = []
    for index, row in enumerate(rows):
        if not row.hexcode or len(row.hexcode) != 8:
            continue
        word = fields_of(int(row.hexcode, 16))
        if word["opcode"] != isa["CUSTOM_OPCODE"]:
            continue
        state = states.get(row.addr)
        if state is None:
            klass = isa["FUNCT_CLASS"].get(word["funct"], "UNKNOWN") if not row.relocations else "UNKNOWN"
            out.append(dict(index=index, pc=row.addr, instruction=klass, fields={}, status="unknown"))
            continue
        operands = [state[word[name]].record() for name in ("rs1", "rs2")]
        klass, fields = semantics.decode_instruction(word["funct"], *operands, isa)
        out.append(
            dict(
                index=index,
                pc=row.addr,
                instruction=klass,
                fields=dict(fields),
                status="decoded",
                operands=dict(zip(("rs1", "rs2"), operands, strict=True)),
            )
        )
    return out


class _Protocol:
    """Declared field bindings: (class, field name) -> bundle, RTL field, interpretation, slot."""

    def __init__(self, structure):
        protocol = structure.get("operand_protocol") or {}
        roles = protocol.get("roles") or {}
        self.bindings = {}
        for klass, spec in (roles.get("classes") or {}).items():
            for field in spec.get("fields") or ():
                if "names" in field:
                    for name in field["names"] or ():
                        self.bindings[(klass, name)] = dict(
                            bundle=field.get("bundle"), field=name, interp="int", operand=field.get("operand")
                        )
                elif "bundle" in field and "field" in field and field.get("name"):
                    self.bindings[(klass, field["name"])] = dict(
                        bundle=field["bundle"],
                        field=field["field"],
                        interp=field.get("as", "int"),
                        operand=field.get("operand"),
                        flag=field.get("flag"),
                    )
                elif field.get("name") and field.get("operand"):
                    self.bindings[(klass, field["name"])] = dict(
                        bundle=None, field=None, interp=field.get("as", "int"), operand=field["operand"]
                    )
        self.address_field = (protocol.get("address_field") or {}).get("field")
        selection = structure.get("load_state_selection") or {}
        bound = selection.get("transfer_bound") or {}
        self.extent = {axis: (bound.get(axis) or {}).get("name") for axis in ("rows", "cols")}
        self.widths = {
            (row.get("bundle"), row.get("name")): row.get("width")
            for row in structure.get("field_layouts") or ()
            if type(row.get("width")) is int
        }
        self.slots = {
            (row.get("bundle"), row.get("name")): (row.get("offset"), row.get("width"))
            for row in structure.get("field_layouts") or ()
            if type(row.get("offset")) is int and type(row.get("width")) is int
        }
        self.sentinel_fields = {b["field"] for b in self.bindings.values() if b["interp"] == "all_ones"}
        addr_len = (structure.get("encoding") or {}).get("addr_len")
        self.flags = {}
        if type(addr_len) is int and addr_len > 0:
            for name, spec in (roles.get("local_address_flags") or {}).items():
                offset = (spec or {}).get("msb_offset")
                if type(offset) is int and 0 <= offset < addr_len:
                    self.flags[name] = 1 << (addr_len - 1 - offset)
        self.flag_slots = {
            (k, b["operand"]) for (k, _), b in self.bindings.items() if b["interp"] in ("flag", "flag_choice")
        }

    def pointer(self, klass, name):
        return f"operand_protocol#roles.classes.{klass}.fields.{name}"

    def fields(self, klass, predicate):
        return [name for (k, name), b in self.bindings.items() if k == klass and predicate(b)]

    def is_address(self, b):
        return self.address_field is not None and b["field"] == self.address_field and b["interp"] == "int"

    def extent_field(self, klass, axis):
        names = self.fields(klass, lambda b: b["field"] == self.extent[axis] and b["interp"] == "int")
        return names[0] if len(names) == 1 else None

    def address_extent(self, klass, name, record):
        """Row extent of one address field: a declared protocol field, else the RTL layout's
        transfer-bound field in the same bundle and register slot. ``(value, pointer)``."""
        rows = self.extent_field(klass, "rows")
        if rows is not None:
            return record["fields"].get(rows), self.pointer(klass, rows)
        b = self.bindings.get((klass, name)) or {}
        slot = self.slots.get((b.get("bundle"), self.extent["rows"]))
        operand = (record.get("operands") or {}).get(b.get("operand"))
        if slot is None or not isinstance(operand, dict):
            return None, None
        return _bits(operand, *slot), f"field_layouts#{b['bundle']}.{self.extent['rows']}"

    def sentinel(self, klass, name, value):
        b = self.bindings.get((klass, name))
        if b is None or b["field"] not in self.sentinel_fields:
            return None
        width = self.widths.get((b["bundle"], b["field"]))
        return None if width is None else value == (1 << width) - 1


def _bits(operand, offset, width):
    """Exact bits of a recovered register operand; ``None`` unless every bit is known."""
    need = ((1 << width) - 1) << offset
    if operand.get("kind") == "const" and type(operand.get("raw")) is int:
        return (operand["raw"] & need) >> offset
    known = operand.get("known_bits") or {}
    if type(known.get("mask")) is int and known["mask"] & need == need:
        return (known["value"] & need) >> offset
    return None


def _symbolic_bits(operand, offset, width):
    """An allocation-relative field: the register's symbolic window starts at the field's bit offset.

    The field equals ``high << kept | (symbol + offset) mod 2**kept``; ``kept`` is the window part
    inside the field and every field bit above it must be known. ``None`` otherwise."""
    window = operand.get("window") if isinstance(operand, dict) else None
    if not isinstance(window, dict) or window.get("at") != offset:
        return None
    kept = min(window["width"], width)
    need = ((1 << (width - kept)) - 1) << (offset + kept)
    known = operand.get("known_bits") or {}
    if need and (known.get("mask", 0) & need) != need:
        return None
    return dict(
        symbol=window["symbol"],
        offset=window["offset"] % (1 << kept),
        width=kept,
        high=(known.get("value", 0) & need) >> (offset + kept),
    )


def _register_domain(operand):
    """Exact register words an operand takes across its loop domain, from the lifter's
    value-set or affine observation; ``None`` when neither is present or bounded."""
    if not isinstance(operand, dict):
        return None, None
    values = operand.get("value_set")
    if isinstance(values, dict):
        values = values.get("values")
    if isinstance(values, (list, tuple)) and values and len(values) <= _DOMAIN and all(type(v) is int for v in values):
        return sorted(set(values)), "value_set"
    affine = operand.get("affine")
    if not isinstance(affine, dict) or type(affine.get("base")) is not int:
        return None, None
    # One (step, count) term per enclosing loop with a proved induction and trip count.
    terms = affine.get("terms")
    if terms is None and "step" in affine:
        terms = [dict(step=affine["step"], count=affine.get("count"))]
    values = {affine["base"]}
    for term in terms if isinstance(terms, list) else ():
        step, count = (term or {}).get("step"), (term or {}).get("count")
        if type(step) is not int or type(count) is not int or not 0 < count or len(values) * count > _DOMAIN:
            return None, None
        values = {value + step * index for value in values for index in range(count)}
    return (sorted(values), "affine") if isinstance(terms, list) else (None, None)


def _field_domain(proto, klass, name, record):
    """Field values over the operand's observed loop domain, with the domain kind."""
    b = proto.bindings.get((klass, name)) or {}
    slot = proto.slots.get((b.get("bundle"), b.get("field")))
    words, kind = _register_domain((record.get("operands") or {}).get(b.get("operand")))
    if slot is None or words is None:
        return None, None
    offset, width = slot
    return sorted({(word >> offset) & ((1 << width) - 1) for word in words}), kind


def _site(evidence_id, record):
    return f"{evidence_id}#instruction:{record['index']}@0x{record['pc']:x}"


def _roles(structure, klass):
    return tuple((structure.get("instruction_roles") or {}).get(klass) or ())


def _retention(commands, structure, proto, evidence_id):
    weight_classes = {k for k, roles in (structure.get("instruction_roles") or {}).items() if "weight_load" in roles}
    if not weight_classes or not any("accumulate" in r for r in (structure.get("instruction_roles") or {}).values()):
        return _unknown("selected role roster declares no weight_load and accumulate roles")
    operands = {}
    for klass in weight_classes:
        operands[klass] = [
            n
            for n in proto.fields(klass, proto.is_address)
            if (klass, proto.bindings[(klass, n)]["operand"]) not in proto.flag_slots
        ]
    if not all(operands.values()):
        return _unknown(
            "weight_load class has no declared flag-free local-address operand field",
            evidence=[f"operand_protocol#roles.classes.{k}" for k in sorted(weight_classes)],
        )
    if not any(proto.sentinel(k, n, 0) is not None for k in operands for n in operands[k]):
        sentinel_note = "no declared all_ones interpretation binds the weight operand field"
    else:
        sentinel_note = None
    counts, witnesses, loads = Counter(), defaultdict(list), Counter()
    intervening, seen_compute = [], False
    for record in commands:
        roles = _roles(structure, record["instruction"])
        if not roles:
            intervening.append("unknown")
            continue
        if "weight_load" in roles:
            values = [record["fields"].get(n) for n in operands[record["instruction"]]]
            if record["status"] != "decoded" or any(type(v) is not int for v in values):
                status = "unresolved"
            elif all(
                proto.sentinel(record["instruction"], n, v)
                for n, v in zip(operands[record["instruction"]], values, strict=True)
            ):
                status = "no_operand"
            else:
                status = "loads"
            loads[status] += 1
            intervening.append(status)
        if "accumulate" in roles:
            if "loads" in intervening:
                outcome = "reloaded"
            elif "unresolved" in intervening or "unknown" in intervening:
                outcome = "unknown_intervening_site"
            elif not seen_compute:
                outcome = "unknown_inherited_state"
            else:
                outcome = "retained"
            counts[outcome] += 1
            if len(witnesses[outcome]) < _WITNESSES:
                witnesses[outcome].append(_site(evidence_id, record))
            seen_compute, intervening = True, []
    computes = sum(counts.values())
    if not computes:
        return _unknown("no accumulate-role sites in the occurrence")
    known = counts["retained"] + counts["reloaded"]
    return _fact(
        dict(
            compute_sites=computes,
            retained=counts["retained"],
            reloaded=counts["reloaded"],
            unknown=computes - known,
            unknown_breakdown={k: v for k, v in counts.items() if k.startswith("unknown")},
            weight_load_sites=dict(loads),
        ),
        "observed" if known == computes else "partial" if known else "unknown",
        scope=dict(static_sites=computes, ordering=ORDERING, weight_operand_fields=operands),
        evidence=[proto.pointer(k, n) for k in sorted(operands) for n in operands[k]]
        + [w for k in sorted(witnesses) for w in witnesses[k]],
        definition="accumulate-role site with no intervening weight_load-role site whose flag-free "
        "local-address operand differs from the declared all-ones sentinel since the previous "
        "accumulate-role site",
        limits=[note for note in (sentinel_note,) if note]
        + [
            "lexical order across loop back-edges is not dynamic issue order",
            "retention legality and mesh-state lifetime are not established",
        ],
    )


def _broadcast(commands, structure, proto, evidence_id):
    selection = structure.get("load_state_selection") or {}
    config, stride_slot = selection.get("config_class"), str(selection.get("stride_input") or "").rpartition(".")[2]
    state_ref = selection.get("config_field") or {}
    if not config or stride_slot not in ("rs1", "rs2"):
        return _unknown("selected load-state selection declares no stride configuration input")
    # The declared stride input is a whole register slot; its raw field is the stride.
    strides = [
        n
        for (k, n), b in proto.bindings.items()
        if k == config and b["operand"] == stride_slot and b["interp"] == "raw"
    ]
    states = proto.fields(
        config, lambda b: (b["bundle"], b["field"]) == (state_ref.get("bundle"), state_ref.get("name"))
    )
    if len(set(strides)) != 1 or len(states) != 1:
        return _unknown("stride or state field of the load-configuration class is not uniquely declared")
    stride, state = strides[0], states[0]
    zero, nonzero, unresolved, witnesses = Counter(), Counter(), 0, []
    ports = structure.get("load_state_ports")
    current, moves, inherited, move_unresolved = {}, Counter(), Counter(), Counter()
    for record in commands:
        klass = record["instruction"]
        if klass == config:
            value, sid = record["fields"].get(stride), record["fields"].get(state)
            if type(value) is not int or type(sid) is not int:
                unresolved += 1
                current = {key: None for key in current} if type(sid) is not int else {**current, sid: None}
                continue
            (zero if value == 0 else nonzero)[sid] += 1
            current[sid] = value
            if value == 0 and len(witnesses) < _WITNESSES:
                witnesses.append(_site(evidence_id, record))
        elif ports and klass in ports and set(_roles(structure, klass)) & {"operand_load"}:
            sid = ports[klass]
            if sid not in current:
                inherited[klass] += 1
            elif current[sid] is None:
                move_unresolved[klass] += 1
            elif current[sid] == 0:
                moves[klass] += 1
    sites = sum(zero.values()) + sum(nonzero.values()) + unresolved
    if not sites:
        return _unknown("no load-configuration sites in the occurrence")
    value = dict(
        config_sites=sites,
        stride_zero_config_sites=sum(zero.values()),
        stride_zero_by_state={str(k): v for k, v in sorted(zero.items())},
        nonzero_by_state={str(k): v for k, v in sorted(nonzero.items())},
        unresolved_config_sites=unresolved,
    )
    if ports:
        value.update(
            broadcast_movement_sites=dict(moves),
            movement_sites_inherited_state=dict(inherited),
            movement_sites_unresolved_stride=dict(move_unresolved),
        )
    return _fact(
        value,
        "observed" if not unresolved else "partial",
        scope=dict(static_sites=sites, ordering=ORDERING, load_state_ports=ports),
        evidence=[proto.pointer(config, stride), proto.pointer(config, state)] + witnesses,
        limits=[
            "movement binding requires the derived load-state ports"
            if not ports
            else "a stride inherited from before the occurrence stays unknown"
        ],
    )


def _tiles(commands, structure, proto, evidence_id):
    shapes, per_role, unresolved, sites, evidence = defaultdict(Counter), defaultdict(Counter), Counter(), Counter(), []
    for record in commands:
        klass = record["instruction"]
        roles = [r for r in _roles(structure, klass) if r in _MOVEMENT_ROLES]
        if not roles:
            continue
        rows, cols = proto.extent_field(klass, "rows"), proto.extent_field(klass, "cols")
        if rows is None or cols is None:
            unresolved[klass] += 1
            continue
        if proto.pointer(klass, rows) not in evidence:
            evidence += [proto.pointer(klass, rows), proto.pointer(klass, cols)]
        sites[klass] += 1
        r, c = record["fields"].get(rows), record["fields"].get(cols)
        if type(r) is not int or type(c) is not int:
            unresolved[klass] += 1
            continue
        shapes[klass][(r, c)] += 1
        for role in roles:
            per_role[role][(r, c)] += 1

    def table(counter):
        return [dict(rows=r, cols=c, sites=n) for (r, c), n in counter.most_common(_VALUES)]

    total = sum(sites.values()) + sum(v for k, v in unresolved.items() if k not in sites)
    if not total:
        return _unknown("no movement-role sites with declared transfer-bound extent fields")
    resolved = sum(sum(c.values()) for c in shapes.values())
    return _fact(
        dict(
            by_class={k: table(v) for k, v in sorted(shapes.items())},
            by_role={k: table(v) for k, v in sorted(per_role.items())},
            unresolved_sites=dict(unresolved),
            distinct_shapes_by_class={k: len(v) for k, v in shapes.items()},
        ),
        "observed" if resolved == total else "partial" if resolved else "unknown",
        scope=dict(static_sites=total, resolved_sites=resolved),
        evidence=evidence + [f"field_layouts#transfer_bound.{a}={proto.extent[a]}" for a in ("rows", "cols")],
    )


def _merge(agg, window):
    """Union one window into an aggregate; extents stay absent unless observed."""
    agg["sites"] = agg.get("sites", 0) + window["sites"]
    agg["extent_sites"] = agg.get("extent_sites", 0) + window["extent_sites"]
    agg["min"] = min(agg.get("min", window["min"]), window["min"])
    agg["max"] = max(agg.get("max", window["max"]), window["max"])
    ends = [e for e in (agg.get("max_end"), window["max_end"]) if e is not None]
    agg["max_end"] = max(ends) if ends else None


def _close(window):
    window["span"] = window["max"] - window["min"] + 1
    if window["max_end"] is not None:
        # The last row any observed site touches, measured from the lowest observed base.
        window["span_with_extent"] = max(window["max_end"], window["max"] + 1) - window["min"]


def _windows(commands, structure, proto, evidence_id):
    windows, sentinels, unresolved, domains = {}, Counter(), Counter(), Counter()
    mask = sum(proto.flags.values())
    for record in commands:
        klass = record["instruction"]
        roles = _roles(structure, klass)
        for name in proto.fields(klass, proto.is_address):
            value = record["fields"].get(name)
            label = f"{klass}.{name}"
            values, domain = ([value], None) if type(value) is int else _field_domain(proto, klass, name, record)
            if values is None:
                unresolved[label] += 1
                continue
            domains[domain or "exact_site_value"] += 1
            counted = extent_seen = False
            for value in values:
                if proto.sentinel(klass, name, value):
                    if not counted:
                        sentinels[label] += 1
                        counted = True
                    continue
                flags = tuple(sorted(f for f, bit in proto.flags.items() if value & bit))
                base = value & ~mask
                key = (klass, name, flags)
                w = windows.setdefault(
                    key,
                    dict(
                        instruction=klass,
                        field=name,
                        roles=list(roles),
                        flags=list(flags),
                        sites=0,
                        min=base,
                        max=base,
                        max_end=None,
                        extent_sites=0,
                        evidence=[proto.pointer(klass, name)],
                    ),
                )
                if not counted:
                    w["sites"] += 1
                    counted = True
                if domain:
                    w["loop_domain_values"] = w.get("loop_domain_values", 0) + 1
                    w["domain_basis"] = sorted(set(w.get("domain_basis", [])) | {domain})
                w["min"], w["max"] = min(w["min"], base), max(w["max"], base)
                extent, pointer = proto.address_extent(klass, name, record)
                if pointer and pointer not in w["evidence"]:
                    w["evidence"].insert(1, pointer)
                if type(extent) is int:
                    w["extent_sites"] += not extent_seen
                    extent_seen = True
                    w["max_end"] = base + extent if w["max_end"] is None else max(w["max_end"], base + extent)
                if len(w["evidence"]) < 2 + _WITNESSES and base in (w["min"], w["max"]):
                    w["evidence"].append(_site(evidence_id, record))
    for w in windows.values():
        _close(w)
    by_flags, by_flag = {}, {}
    for w in windows.values():
        _merge(by_flags.setdefault(",".join(w["flags"]) or "none", dict(flags=w["flags"])), w)
        for flag in w["flags"] or ("none",):
            _merge(by_flag.setdefault(flag, dict(flag=flag)), w)
    for agg in (*by_flags.values(), *by_flag.values()):
        _close(agg)
    resolved = sum(w["sites"] for w in windows.values()) + sum(sentinels.values())
    total = resolved + sum(unresolved.values())
    if not total:
        return _unknown("no declared local-address fields in the occurrence")
    return _fact(
        dict(
            windows=sorted(windows.values(), key=lambda w: (w["instruction"], w["field"], w["flags"])),
            by_flag_set=by_flags,
            by_declared_flag=by_flag,
            sentinel_sites=dict(sentinels),
            unresolved_sites=dict(unresolved),
        ),
        "observed" if resolved == total else "partial" if resolved else "unknown",
        scope=dict(
            field_sites=total,
            resolved_field_sites=resolved,
            coverage=round(resolved / total, 4),
            declared_flags=sorted(proto.flags),
            value_basis=dict(sorted(domains.items())),
        ),
        evidence=[
            f"operand_protocol#address_field={proto.address_field}",
            "operand_protocol#roles.local_address_flags",
        ],
        limits=[
            "addresses are field values with declared flag bits cleared; bank geometry and capacity are not interpreted"
        ],
    )


def _epilogue(commands, structure, proto, evidence_id, analysis):
    epilogue = structure.get("output_epilogue") or {}
    carriers = {}
    for role, spec in epilogue.items():
        if isinstance(spec, dict) and isinstance(spec.get("carriers"), list):
            for carrier in spec["carriers"]:
                carriers[(carrier.get("bundle"), carrier.get("field"))] = role
    if not carriers:
        return _unknown("selected output epilogue declares no carrier fields")
    bound = {
        (k, n): carriers[(b["bundle"], b["field"])]
        for (k, n), b in proto.bindings.items()
        if (b["bundle"], b["field"]) in carriers
    }
    if not bound:
        return _unknown("no protocol field binds a declared epilogue carrier")
    observed = defaultdict(lambda: dict(values=Counter(), unresolved=0, role=None))
    pooled, pool_state, witnesses = Counter(), None, []
    pool_fields = {key for key, role in bound.items() if role == "pool"}
    pool_classes = {k for k, _ in pool_fields}
    source = commands
    for record in source or ():
        klass = record["instruction"]
        for (k, name), role in bound.items():
            if k != klass:
                continue
            row = observed[f"{k}.{name}"]
            row["role"] = role
            value = record["fields"].get(name)
            if value is None:
                row["unresolved"] += 1
            else:
                row["values"][value] += 1
        if klass in pool_classes:
            values = [record["fields"].get(n) for k, n in pool_fields if k == klass]
            pool_state = None if any(type(v) is not int for v in values) else any(values)
            if pool_state and len(witnesses) < _WITNESSES:
                witnesses.append(_site(evidence_id, record))
        if set(_roles(structure, klass)) & {"readout"} and pool_classes:
            pooled["pooled" if pool_state else "unknown" if pool_state is None else "not_pooled"] += 1
    if source is None:
        profiles = (analysis.get("mechanism_projection") or {}).get("field_profiles") or {}
        for (k, name), role in bound.items():
            rows = profiles.get(f"{k}.{name}")
            if rows:
                observed[f"{k}.{name}"] = dict(
                    role=role,
                    values=Counter({r["value"]: r["sites"] for r in rows}),
                    unresolved=((analysis.get("mechanism_projection") or {}).get("unresolved_fields") or {}).get(
                        f"{k}.{name}", 0
                    ),
                )
    functions = {}
    for role, spec in epilogue.items():
        for fname, fspec in ((spec or {}).get("functions") or {}).items() if isinstance(spec, dict) else ():
            for code in fspec.get("codes") or ():
                functions[(role, code)] = fname
    fields = {}
    for label, row in sorted(observed.items()):
        entry = dict(
            role=row["role"],
            values=[dict(value=v, sites=n) for v, n in row["values"].most_common(_VALUES)],
            unresolved_sites=row["unresolved"],
        )
        named = sorted({functions[(row["role"], v)] for v in row["values"] if (row["role"], v) in functions})
        if named:
            entry["declared_functions_observed"] = named
        fields[label] = entry
    if not fields:
        return _unknown("no site of a carrier-bound protocol field in the occurrence")
    value = dict(fields=fields)
    if source is not None and pool_classes:
        value["readout_sites_after_pool_configuration"] = dict(pooled)
    complete = all(row["unresolved_sites"] == 0 for row in fields.values())
    return _fact(
        value,
        "observed" if complete else "partial",
        scope=dict(
            ordering=ORDERING if source is not None else "summary_profiles",
            carrier_bound_fields=sorted(f"{k}.{n}" for k, n in bound),
        ),
        evidence=[proto.pointer(k, n) for k, n in sorted(bound)] + witnesses,
        limits=[
            "a configuration inherited from before the occurrence stays unknown",
            "which readout instructions a configuration governs is lexical, not proven",
        ],
    )


def _summary_windows(analysis, structure, proto):
    projection = analysis.get("mechanism_projection") or {}
    profiles, omitted = projection.get("field_profiles") or {}, projection.get("omitted_profile_sites") or {}
    rows = []
    for (klass, name), b in sorted(proto.bindings.items()):
        if not proto.is_address(b) or f"{klass}.{name}" not in profiles:
            continue
        values = [r["value"] for r in profiles[f"{klass}.{name}"] if type(r["value"]) is int]
        rows.append(
            dict(
                instruction=klass,
                field=name,
                profiled_values=len(values),
                omitted_sites=omitted.get(f"{klass}.{name}", 0),
            )
        )
    return _unknown(
        "ordered decoded commands not supplied; summary value profiles are truncated", scope=dict(profiled_fields=rows)
    )


def _summary_broadcast(analysis, structure, proto):
    selection = structure.get("load_state_selection") or {}
    config = selection.get("config_class")
    slot = str(selection.get("stride_input") or "").rpartition(".")[2]
    names = [n for (k, n), b in proto.bindings.items() if k == config and b["operand"] == slot and b["interp"] == "raw"]
    projection = analysis.get("mechanism_projection") or {}
    if len(names) != 1:
        return _unknown("stride field of the load-configuration class is not uniquely declared")
    label = f"{config}.{names[0]}"
    rows = (projection.get("field_profiles") or {}).get(label)
    if not rows:
        return _unknown("no profiled load-configuration stride field")
    omitted = (projection.get("omitted_profile_sites") or {}).get(label, 0)
    unresolved = (projection.get("unresolved_fields") or {}).get(label, 0)
    zero = sum(r["sites"] for r in rows if r["value"] == 0)
    return _fact(
        dict(
            stride_zero_config_sites=zero,
            profiled_sites=sum(r["sites"] for r in rows),
            unresolved_config_sites=unresolved,
        ),
        "observed" if not omitted and not unresolved else "partial",
        scope=dict(static_sites=sum(r["sites"] for r in rows) + omitted + unresolved, basis="summary_profiles"),
        evidence=[proto.pointer(config, names[0]), f"analysis#mechanism_projection.field_profiles.{label}"],
    )


def _loop_nest(projection, analysis, evidence_id):
    """Project the lifter's loop nest: depth, per-level trip counts, induction steps and
    which accelerator roles sit at each level. Absent lifter loops stay unknown."""
    nest = projection.get("loop_nest") or {}
    value = nest.get("value")
    if nest.get("status") in (None, "unknown") or not isinstance(value, dict):
        return _unknown(nest.get("reason") or "occurrence analysis carries no lifter loop nest")
    levels = [
        dict(
            header_pc=level.get("header_pc"),
            nest_depth=level.get("nest_depth"),
            parent_header_pc=level.get("parent_header_pc"),
            trip_count=level.get("trip_count"),
            induction_steps=[
                dict(reg=row.get("reg"), step=row.get("step"), status=row.get("status"))
                for row in level.get("induction") or ()
            ],
            roles=level.get("role_sites") or {},
            instruction_classes=level.get("instruction_classes") or {},
            evidence=list(level.get("evidence") or ()),
        )
        for level in value.get("levels") or ()
    ]
    entry = projection.get("entry_state")
    return _fact(
        dict(
            depth=value.get("depth"),
            levels=levels,
            per_level_trip_counts=[level["trip_count"] for level in levels],
            top_level_command_sites=value.get("top_level_command_sites"),
            dynamic_command_count=value.get("dynamic_command_count"),
        ),
        nest["status"],
        scope=dict(
            loops=value.get("loops"),
            proved_trip_counts=value.get("proved_trip_counts"),
            entry_state="caller_agreed" if entry else "unconstrained_arguments",
        ),
        evidence=[f"{evidence_id}#mechanism_projection.loop_nest", *nest.get("evidence", ())],
        limits=["trip counts are proved only under the analyzed entry state; execution order is not issue order"],
    )
