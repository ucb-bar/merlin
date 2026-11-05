"""Derived static effects over decoded accelerator commands, from declared structure only.

Every fact is computed from the selected instruction-role roster, the selected
operand protocol (field names, register bundles, RTL field bindings and
interpretations), its RTL field layouts and the contract's declared epilogue
carriers. No instruction or field spelling, model table or target constant
selects a value. A fact the declared data cannot support stays ``unknown`` with
a reason; observed values are lexical static-listing facts, not dynamic issue
order, execution multiplicity, legality or performance.

Builder call (one per occurrence, after ``machine_projection``)::

    structure = selected_structure(decoder, contract_document)
    commands = decoded_commands(body, decoder=decoder)  # optional
    report["derived_effects"] = derive_effects(report, structure, commands=commands, evidence_id=evidence_id)

``report`` is the occurrence analysis record (``analysis.json``). Without
``commands`` only summary-supported facts are derived; ordered facts are unknown.
"""

from __future__ import annotations

from collections import Counter

from .decoded_static_effects import (
    SCHEMA as SCHEMA,
    ORDERING as ORDERING,
    _MOVEMENT_ROLES as _MOVEMENT_ROLES,
    _WITNESSES as _WITNESSES,
    _VALUES as _VALUES,
    _digest as _digest,
    _fact as _fact,
    _unknown as _unknown,
    read_pinned_sidecar as read_pinned_sidecar,
    read_pinned_facts as read_pinned_facts,
    local_row_geometry as local_row_geometry,
    selected_structure as selected_structure,
    decoded_commands as decoded_commands,
    _Protocol as _Protocol,
    _bits as _bits,
    _symbolic_bits as _symbolic_bits,
    _register_domain as _register_domain,
    _field_domain as _field_domain,
    _site as _site,
    _roles as _roles,
    _retention as _retention,
    _broadcast as _broadcast,
    _tiles as _tiles,
    _merge as _merge,
    _close as _close,
    _windows as _windows,
    _epilogue as _epilogue,
    _summary_windows as _summary_windows,
    _summary_broadcast as _summary_broadcast,
    _loop_nest as _loop_nest,
)


from .decoded_static_effects import _DOMAIN as _DOMAIN


# ----------------------------------------------------------------------------- proved dynamic stream
STREAM_SCOPE = "proved_dynamic_command_stream"
_NO_STREAM = "no_proved_command_stream"
_STREAM_FACTS = ("role_transfer_width", "operand_buffering", "issue_overlap", "prefetch_distance",
                 "output_group_extent")
_INBOUND_ROLES = ("operand_load", "weight_load", "dma")
_RUNS_PREFIX = 64
_BOUND = 32


def _stream_operand(value):
    """A recorded issue-time register as the observer's operand record."""
    if type(value) is int:
        return dict(kind="const", raw=value)
    return dict(value) if isinstance(value, dict) else dict(kind="unknown", raw=None)


def decode_stream(stream, *, decoder, limit=None):
    """Decode a ``finite_control_counts(record_commands=True)`` stream in issue order.

    ``stream`` is its ``command_stream`` or its ``command_stream_runs`` (expanded here);
    ``limit`` bounds the decoded entries.

    The same selected observer (``semantics.decode_instruction``) and declared
    instruction roles as ``machine_projection``; each entry becomes
    ``{seq, pc, class, roles, fields, host_instructions_since_prev}`` and a fence
    ``{seq, pc, class: "FENCE", roles: ["sync"], fields: {}, host_instructions_since_prev}``.
    """
    semantics, target = (decoder or {}).get("semantics"), (decoder or {}).get("target")
    if semantics is None or stream is None:
        return None
    stream = list(stream)
    if stream and "count" in stream[0]:
        from merlin.compile.analysis.riscv_machine import expand_command_stream

        stream = expand_command_stream(stream, limit=limit)
    elif limit is not None:
        stream = stream[:limit]
    isa = semantics.isa_constants(target)
    roles = ((decoder or {}).get("structure") or {}).get("instruction_roles") or {}
    out = []
    for entry in stream:
        host = entry.get("host_instructions_since_prev")
        if entry.get("kind") == "fence":
            out.append(
                {
                    "seq": entry["seq"],
                    "pc": entry.get("pc"),
                    "class": "FENCE",
                    "roles": ["sync"],
                    "fields": {},
                    "host_instructions_since_prev": host,
                }
            )
            continue
        operands = [_stream_operand(entry.get(name)) for name in ("rs1", "rs2")]
        klass, fields = semantics.decode_instruction(entry["funct"], *operands, isa)
        out.append(
            {
                "seq": entry["seq"],
                "pc": entry.get("pc"),
                "class": klass,
                "roles": list(roles.get(klass, ())),
                "fields": dict(fields),
                "host_instructions_since_prev": host,
            }
        )
        symbolic = {slot: record for slot, record in zip(("rs1", "rs2"), operands) if "window" in record}
        if symbolic:
            out[-1]["symbolic_operands"] = symbolic
    return out


def _median(values):
    values = sorted(values)
    if not values:
        return None
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def _spread(values):
    return dict(count=len(values), min=min(values), median=_median(values), max=max(values)) if values else None


class _StreamView:
    """Issue-order categories of a decoded stream from declared roles and fields only.

    ``load``: an inbound movement role whose class declares both transfer-bound extent
    fields (a memory-to-local transfer); ``local_staging``: a movement role without them
    (a preload of local data into the compute unit), reported apart and transparent
    to run structure; ``compute``: the ``accumulate`` role; ``barrier``: the ``sync`` role;
    ``readout``: ``readout``/``commit``; any other class is ``other``.
    """

    def __init__(self, stream, structure, proto, arguments):
        self.proto, self.entries, self.operand_fields, self.kinds = proto, [], {}, {}
        self._slots, self._consumption, self.structure = {}, None, structure
        self.arguments = {
            name: value
            for name, value in (arguments or {}).items()
            if type(value) is int and isinstance(name, str) and name.startswith("a")
        }
        for entry in stream:
            klass = entry["class"]
            roles = entry.get("roles") or list(_roles(structure, klass))
            key = (klass, tuple(roles))
            if key not in self.kinds:
                self.kinds[key] = self.kind(klass, roles)
            self.entries.append((entry, klass, roles, self.kinds[key]))

    def kind(self, klass, roles):
        if "sync" in roles:
            return "barrier"
        if "accumulate" in roles:
            return "compute"
        movement = [r for r in roles if r in _INBOUND_ROLES]
        extent = self.proto.extent_field(klass, "rows") and self.proto.extent_field(klass, "cols")
        if movement:
            return "load" if extent else "local_staging"
        if any(r in ("readout", "commit") for r in roles):
            return "readout"
        return "other"

    @staticmethod
    def movement_role(roles):
        return next((r for r in roles if r in _INBOUND_ROLES), None)

    def dram(self, klass, fields):
        """The declared memory-operand field of a transfer and the tensor argument it addresses."""
        if klass not in self.operand_fields:
            names = self.proto.fields(klass, lambda b: b["interp"] == "operand")
            self.operand_fields[klass] = names[0] if len(names) == 1 else None
        name = self.operand_fields[klass]
        record = fields.get(name) if name else None
        if not isinstance(record, dict):
            return None, "no_declared_memory_operand_field"
        base = record.get("pointer_base")
        if isinstance(base, str) and base.startswith("arg") and base[3:].isdigit():
            return "a" + base[3:], "symbolic_argument_pointer"
        raw = record.get("raw") if record.get("kind") == "const" else None
        if type(raw) is int and self.arguments:
            below = [(value, reg) for reg, value in self.arguments.items() if value <= raw]
            if below:
                return max(below)[1], "nearest_entry_argument_pointer_at_or_below"
        return None, "address_not_tied_to_an_argument_pointer"

    def local(self, klass, fields, operands=None):
        """``(address, flags, space)``: ``space`` is ``None`` for an exact address, else the
        allocation-relative ``(symbol, width, high bits)`` whose ``address`` is an offset mod 2**width."""
        names = self.proto.fields(klass, self.proto.is_address)
        values = [v for v in (fields.get(n) for n in names) if type(v) is int]
        mask = sum(self.proto.flags.values())
        if values:
            value = values[0]
            return value & ~mask, tuple(sorted(f for f, bit in self.proto.flags.items() if value & bit)), None
        for name in names:
            relative = self.symbolic(klass, name, operands)
            if relative is not None:
                high = relative["high"] << relative["width"]
                flags = tuple(sorted(f for f, bit in self.proto.flags.items() if high & bit))
                return relative["offset"], flags, (relative["symbol"], relative["width"], high & ~mask)
        return None, (), None

    def symbolic(self, klass, name, operands):
        """The allocation-relative value of one declared field from its register's symbolic window."""
        binding = self.proto.bindings.get((klass, name)) or {}
        slot = self.proto.slots.get((binding.get("bundle"), binding.get("field")))
        record = (operands or {}).get(binding.get("operand"))
        return None if slot is None or record is None else _symbolic_bits(record, *slot)

    def extent(self, klass, fields):
        rows, cols = self.proto.extent_field(klass, "rows"), self.proto.extent_field(klass, "cols")
        r, c = (fields.get(rows) if rows else None), (fields.get(cols) if cols else None)
        return (r if type(r) is int else None), (c if type(c) is int else None)

    def address_slots(self, klass):
        """``[(slot, field names, row-extent field)]``: the class's declared local-address fields grouped
        by the register slot that carries them, with the same slot's declared row-extent field."""
        cached = self._slots.get(klass)
        if cached is not None:
            return cached
        slots = {}
        for name in self.proto.fields(klass, self.proto.is_address):
            slot = self.proto.bindings[(klass, name)].get("operand")
            slots.setdefault(slot, []).append(name)
        out = []
        for slot, names in sorted(slots.items(), key=lambda item: str(item[0])):
            rows = self.proto.fields(klass, lambda b, slot=slot: b.get("operand") == slot and b["interp"] == "int"
                                     and b["field"] == self.proto.extent["rows"])
            out.append((slot, tuple(sorted(names)), rows[0] if len(rows) == 1 else None))
        self._slots[klass] = out
        return out

    def reads(self, klass, fields, operands=None):
        """Local rows one non-load issue reads: ``[(port field label, flags, base, rows or None, space)]``
        (``space`` as in :meth:`local`)."""
        mask = sum(self.proto.flags.values())
        out = []
        for _, names, rows_field in self.address_slots(klass):
            value = next((fields.get(n) for n in names if type(fields.get(n)) is int), None)
            rows = fields.get(rows_field) if rows_field else None
            rows = rows if type(rows) is int and rows > 0 else None
            if value is None:
                relative = next((r for r in (self.symbolic(klass, n, operands) for n in names) if r), None)
                if relative is not None:
                    high = relative["high"] << relative["width"]
                    flags = tuple(sorted(f for f, bit in self.proto.flags.items() if high & bit))
                    out.append(("/".join(names), flags, relative["offset"], rows,
                                (relative["symbol"], relative["width"], high & ~mask)))
                continue
            if self.proto.sentinel(klass, names[0], value):
                continue
            flags = tuple(sorted(f for f, bit in self.proto.flags.items() if value & bit))
            out.append(("/".join(names), flags, value & ~mask, rows, None))
        return out


def _port_role(kind, klass, roles):
    """The declared role a consuming issue reads its local operand under (never an instruction name)."""
    if kind == "compute":
        return "accumulate"
    if kind == "local_staging":
        return _StreamView.movement_role(roles)
    if kind == "readout":
        return next(r for r in roles if r in ("readout", "commit"))
    return "+".join(roles) if roles else "unroled"


def _block_step_field(structure, proto):
    """The load-configuration field that places a wide transfer's further row blocks, or a reason.

    Derived from the selected load-state selection and the sidecar's RTL field-kind sources: the
    configuration class's unique ``stride``-kind field whose RTL sink is indexed by the declared
    load-state field, other than the declared memory-stride input. Returns ``(config class, field,
    state field)`` or ``(None, reason)``."""
    selection = structure.get("load_state_selection") or {}
    config, state = selection.get("config_class"), selection.get("state")
    state_ref = selection.get("config_field") or {}
    if not config or not state:
        return None, "selected load-state selection declares no configuration class and state"
    memory_slot = str(selection.get("stride_input") or "").rpartition(".")[2]
    found = []
    for row in structure.get("operand_field_kind_sources") or ():
        klass, _, name = str(row.get("label") or "").partition(".")
        sink = str(row.get("sink") or "")
        index = sink.partition("[")[2]
        if row.get("kind") != "stride" or klass != config or not index.endswith("]") or index[:-1] != state:
            continue
        binding = proto.bindings.get((klass, name)) or {}
        if binding.get("interp") != "int" or binding.get("operand") == memory_slot:
            continue
        found.append(name)
    wanted = (state_ref.get("bundle"), state_ref.get("name"))
    states = proto.fields(config, lambda b: (b["bundle"], b["field"]) == wanted)
    if len(found) != 1 or len(states) != 1:
        return None, ("no unique per-load-state stride field of the load-configuration class other than the "
                      "memory stride (RTL field-kind sources)")
    return (config, found[0], states[0]), None


class _Consumption:
    """Where each load's rows land and which later issue first reads them, in proved issue order.

    A load writes ``ceil(cols / row elements)`` row blocks of ``rows`` rows (row elements from the
    selected RTL facts' local stores); block ``j`` starts ``j`` x the per-load-state block step after
    the base (the step a load-configuration issue set for that load's state). A consumer is any later
    non-load issue whose declared local-address field reads a row whose latest writer is that load.
    A load whose footprint cannot be established stays unresolved with the reason.
    """

    def __init__(self, view, structure):
        self.view, self.loads, self.unresolved = view, [], Counter()
        self.spaces, self.reads = set(), []  # allocation-relative bases; (position, base) of each read
        self._last_read = None
        geometry = structure.get("local_geometry") or {}
        self.row_elements = geometry.get("uniform_row_elements")
        self.geometry_reason = None if type(self.row_elements) is int else (
            geometry.get("reason") or "local row width not derived from the selected RTL facts")
        step, self.step_reason = _block_step_field(structure, view.proto)
        ports = structure.get("load_state_ports") or {}
        steps, writer = {}, {}
        current = Counter()
        loads_seen = computes_seen = 0
        for position, (entry, klass, roles, kind) in enumerate(view.entries):
            fields = entry["fields"]
            if step is not None and klass == step[0]:
                sid, value = fields.get(step[2]), fields.get(step[1])
                if type(sid) is int:
                    steps[sid] = value if type(value) is int else None
                else:
                    steps = {key: None for key in steps}
            symbolic = entry.get("symbolic_operands")
            if kind == "load":
                operand, _ = view.dram(klass, fields)
                base, flags, space = view.local(klass, fields, symbolic)
                rows, cols = view.extent(klass, fields)
                load = dict(position=position, ordinal=loads_seen, computes_before=computes_seen,
                            role=view.movement_role(roles), operand=operand or "unattributed", flags=flags,
                            first=None, port=None, ports=set(), footprint=None, space=space)
                loads_seen += 1
                group = (load["role"], load["operand"], flags)
                load["round"] = current[group]
                reason = None
                missing = [name for name, value in (("local address", base), ("transfer rows", rows),
                                                    ("transfer columns", cols)) if value is None]
                if missing:
                    reason = "load " + " and ".join(missing) + " unresolved"
                elif self.row_elements is None:
                    reason = self.geometry_reason
                else:
                    blocks = -(-cols // self.row_elements)
                    stride = 0
                    if blocks > 1:
                        stride = steps.get(ports.get(klass)) if step is not None and klass in ports else None
                        if step is None:
                            reason = self.step_reason
                        elif type(stride) is not int:
                            reason = "block step of this load's state not set by an earlier configuration issue"
                    if reason is None:
                        load["footprint"] = [(base + j * stride, base + j * stride + rows) for j in range(blocks)]
                        load["blocks"], load["block_step"] = blocks, stride if blocks > 1 else None
                if reason is not None:
                    load["reason"] = reason
                    self.unresolved[reason] += 1
                index = len(self.loads)
                self.loads.append(load)
                if space is not None:
                    self.spaces.add(space)
                for lo, hi in load["footprint"] or ():
                    for row in range(lo, hi):
                        writer[(flags, _row_key(space, row))] = index
                continue
            if kind == "compute":
                computes_seen += 1
            if kind == "barrier":
                continue
            for label, flags, base, rows, space in view.reads(klass, fields, symbolic):
                port = f"{_port_role(kind, klass, roles)}.{label}"
                self.reads.append((position, space))
                if space is not None:
                    self.spaces.add(space)
                for row in range(base, base + (rows or 1)):
                    index = writer.get((flags, _row_key(space, row)))
                    if index is None:
                        continue
                    load = self.loads[index]
                    load["ports"].add(port)
                    if load["first"] is None:
                        load.update(first=position, port=port, loads_between=loads_seen - load["ordinal"] - 1,
                                    computes_between=computes_seen - load["computes_before"]
                                    - (kind == "compute"))
                    group = (load["role"], load["operand"], load["flags"])
                    if load["round"] == current[group]:
                        current[group] += 1  # data of the open fill round is now being consumed

    def unestablished(self, load):
        """True for an unread load when a later issue read rows through another base: the rows of
        two bases are not comparable, so whether that issue read this load is not established."""
        if load["first"] is not None or not self.spaces:
            return False
        if self._last_read is None:
            self._last_read = {}
            for position, space in self.reads:
                self._last_read[space] = position
        return any(position > load["position"] for space, position in self._last_read.items()
                   if space != load["space"])

    def relative_limits(self):
        if not self.spaces:
            return []
        return [f"{len(self.spaces)} allocation-relative local-address bases (unproved call returns): rows "
                "compare only within one base; disjointness or overlap across allocations is not established, "
                "so a writer of another base is never matched to a reader"]

    def groups(self):
        """``{(role, operand label): [load]}``; the label carries flag sets only when they split an operand."""
        split = {}
        for load in self.loads:
            split.setdefault((load["role"], load["operand"]), set()).add(load["flags"])
        out = {}
        for load in self.loads:
            key = (load["role"], load["operand"])
            label = load["operand"] if len(split[key]) == 1 else f"{load['operand']}[{','.join(load['flags'])}]"
            out.setdefault((load["role"], label), []).append(load)
        return out

    @staticmethod
    def port(loads):
        """The consuming compute ports of one operand's loads (first consumers), or ``None``."""
        ports = sorted({load["port"] for load in loads if load["port"] is not None})
        return "+".join(ports) if ports else None

    def operand_ports(self):
        return {f"{role}:{operand}": self.port(loads) for (role, operand), loads in self.groups().items()}

    def by_tensor_port(self):
        """``{operand: port}`` across movement roles; an operand split across ports keeps the joined label."""
        ports = {}
        for (_, operand), loads in self.groups().items():
            ports.setdefault(operand, []).extend(loads)
        return {operand: self.port(loads) for operand, loads in ports.items()}


def _row_key(space, row):
    """A local row: exact, or an offset modulo its allocation-relative base's width."""
    return row if space is None else (space, row % (1 << space[1]))


def _consumption(view):
    if view._consumption is None:
        view._consumption = _Consumption(view, view.structure)
    return view._consumption


def _stream_fact(value, evidence, stream, truncated, *, partial=False, **extra):
    return _fact(
        value,
        "partial" if truncated or partial else "observed",
        scope=STREAM_SCOPE,
        evidence=evidence,
        stream=dict(entries=len(stream), truncated=bool(truncated)),
        **extra,
    )


def _role_transfer_width(view, stream, truncated, evidence_id):
    roles, staging, basis, extents = {}, {}, Counter(), set()
    for entry, klass, entry_roles, kind in view.entries:
        if kind == "local_staging":
            row = staging.setdefault(view.movement_role(entry_roles), dict(issues=0, classes=Counter()))
            row["issues"] += 1
            row["classes"][klass] += 1
            continue
        if kind != "load":
            continue
        operand, how = view.dram(klass, entry["fields"])
        basis[how] += 1
        shape = view.extent(klass, entry["fields"])
        extents.add(klass)
        row = roles.setdefault(
            view.movement_role(entry_roles), dict(transfers=0, shapes=Counter(), classes=Counter(), by_operand={})
        )
        tensor = row["by_operand"].setdefault(
            operand or "unattributed", dict(transfers=0, shapes=Counter(), classes=Counter())
        )
        for target in (row, tensor):
            target["transfers"] += 1
            target["shapes"][shape] += 1
            target["classes"][klass] += 1
    if not roles and not staging:
        return _unknown("proved stream has no movement-role issues", scope=STREAM_SCOPE)

    def shapes(counter):
        return [
            dict(rows=r, cols=c, count=n, elements=r * c if r is not None and c is not None else None)
            for (r, c), n in counter.most_common(_BOUND)
        ]

    def close(row):
        return dict(
            transfers=row["transfers"],
            shapes=shapes(row["shapes"]),
            distinct_shapes=len(row["shapes"]),
            classes=dict(row["classes"]),
        )

    ports = _consumption(view).operand_ports()
    value = dict(
        data_movement={
            role: dict(close(row), by_operand={k: dict(close(v), consumer_port=ports.get(f"{role}:{k}"))
                                               for k, v in sorted(row["by_operand"].items())})
            for role, row in sorted(roles.items())
        },
        local_staging={
            role: dict(issues=row["issues"], classes=dict(row["classes"])) for role, row in sorted(staging.items())
        },
        operand_basis=dict(basis),
    )
    evidence = [
        view.proto.pointer(k, view.proto.extent_field(k, axis)) for k in sorted(extents) for axis in ("rows", "cols")
    ]
    evidence += [f"field_layouts#transfer_bound.{a}={view.proto.extent[a]}" for a in ("rows", "cols")]
    return _stream_fact(value, evidence + [f"{evidence_id}#command_stream"], stream, truncated)


BUFFERING_DEFINITION = "consumption_rounds_v2"


def _rounds(loads):
    """Destination regions of one operand's loads and the order fill rounds visit them.

    A fill round is the run of the operand's loads issued until a later issue first reads a row one of
    them wrote; the next load opens a new round. A round whose footprint shares no row with an earlier
    region opens a region; one that refills part or all of exactly one region, or covers it, joins it.
    A round sharing rows with several regions, or only partly with one (neither containing nor
    contained), straddles regions: the regions merge and ``straddling_rounds`` counts it. Returns
    ``(regions, sequence, straddling_rounds)``; ``sequence`` is the region of each round in issue order."""
    footprints = {}
    for load in loads:
        footprints.setdefault(load["round"], set()).update(
            row for lo, hi in load["footprint"] for row in range(lo, hi))
    regions, owner, assigned, straddling = [], {}, [], 0
    for r in sorted(footprints):
        rows = footprints[r]
        touched = sorted({owner[row] for row in rows if row in owner})
        if not touched:
            index = len(regions)
            regions.append(dict(rows=set(rows), rounds=0))
        else:
            index = touched[0]
            region = regions[index]
            if len(touched) > 1 or not (rows <= region["rows"] or region["rows"] <= rows):
                straddling += 1
            for other in touched[1:]:
                region["rows"] |= regions[other]["rows"]
                regions[other]["merged_into"] = index
                for row in regions[other]["rows"]:
                    owner[row] = index
            region["rows"] |= rows
        assigned.append(index)
        for row in rows:
            owner[row] = index
    live = [i for i, region in enumerate(regions) if "merged_into" not in region]

    def final(i):
        while "merged_into" in regions[i]:
            i = regions[i]["merged_into"]
        return live.index(i)

    sequence = [final(i) for i in assigned]
    out = []
    for i in live:
        rows = sorted(regions[i]["rows"])
        spans = sum(1 for k, row in enumerate(rows) if k == 0 or row != rows[k - 1] + 1)
        out.append(dict(lo=rows[0], hi=rows[-1], rows=len(rows), intervals=spans,
                        rounds=sequence.count(live.index(i))))
    return out, sequence, straddling


def _split_by_base(groups):
    """One buffering entry per local-address base when an operand's resolved loads use several:
    rows of different bases are not comparable, so each base's regions are found on their own."""
    out = {}
    for (role, operand), loads in groups.items():
        spaces = {load["space"] for load in loads}
        if len(spaces) < 2 or any(load["footprint"] is None for load in loads):
            out[(role, operand)] = loads
            continue
        for load in loads:
            label = operand + "@" + ("exact" if load["space"] is None else load["space"][0])
            out.setdefault((role, label), []).append(load)
    return out


def _operand_buffering(view, stream, truncated, evidence_id):
    consumption = _consumption(view)
    if not consumption.loads:
        return _unknown("proved stream has no load with a resolved declared local-address field", scope=STREAM_SCOPE)
    value = {}
    for (role, operand), loads in sorted(_split_by_base(consumption.groups()).items()):
        bases = sorted({load["footprint"][0][0] for load in loads if load["footprint"]})
        entry = dict(definition=BUFFERING_DEFINITION, flags=list(loads[0]["flags"]), loads=len(loads),
                     consumer_port=consumption.port(loads), distinct_base_addresses=len(bases),
                     base_range=[bases[0], bases[-1]] if bases else None,
                     block_steps=sorted({load["block_step"] for load in loads if load.get("block_step")}))
        unresolved = Counter(load["reason"] for load in loads if load["footprint"] is None)
        if unresolved:
            entry.update(status="unknown", disjoint_windows=None, alternating=None,
                         reason="; ".join(f"{n} loads: {why}" for why, n in unresolved.most_common()))
            value.setdefault(role, {})[operand] = entry
            continue
        spaces = {load["space"] for load in loads}
        if spaces != {None}:
            space = next(iter(spaces))
            entry.update(windows_relative_to="allocation", allocation=space[0] if len(spaces) == 1 else None)
            if len(spaces) > 1:
                entry.update(status="unknown", disjoint_windows=None, alternating=None, allocations=len(spaces),
                             reason=f"loads address {len(spaces)} bases (exact or allocation-relative); "
                                    "disjointness across allocations not established")
                value.setdefault(role, {})[operand] = entry
                continue
            if any(hi > 1 << space[1] for load in loads for _, hi in load["footprint"]):
                entry.update(status="unknown", disjoint_windows=None, alternating=None,
                             reason="a footprint wraps the allocation-relative offset width")
                value.setdefault(role, {})[operand] = entry
                continue
        if any(not load["footprint"] or any(hi <= lo for lo, hi in load["footprint"]) for load in loads):
            entry.update(status="unknown", disjoint_windows=None, alternating=None,
                         reason="a load declares zero transfer rows or columns; its round writes no row")
            value.setdefault(role, {})[operand] = entry
            continue
        regions, sequence, straddling = _rounds(loads)
        visits = [w for i, w in enumerate(sequence) if i == 0 or w != sequence[i - 1]]
        if straddling:
            entry.update(status="unknown", disjoint_windows=None, alternating=None, fill_rounds=len(sequence),
                         straddling_rounds=straddling,
                         reason=f"{straddling} of {len(sequence)} fill rounds share rows with several regions or only "
                                "part of one (loads issued into another region before the round was first read); "
                                "the regions are not established")
            value.setdefault(role, {})[operand] = entry
            continue
        entry.update(
            status="observed",
            fill_rounds=len(sequence),
            disjoint_windows=len(regions),
            windows=regions[:_BOUND],
            window_sequence_prefix=visits[:_RUNS_PREFIX],
            window_switches=len(visits) - 1,
            # More than one region, and fill rounds return to a region after leaving it.
            alternating=len(regions) > 1 and len(visits) > len(regions),
            unconsumed_loads=sum(1 for load in loads if load["first"] is None),
        )
        value.setdefault(role, {})[operand] = entry
    return _stream_fact(
        value,
        [
            f"operand_protocol#address_field={view.proto.address_field}",
            "operand_protocol#roles.local_address_flags",
            "structure#local_geometry",
            "structure#operand_field_kind_sources",
            f"{evidence_id}#command_stream",
        ],
        stream,
        truncated,
        definition="a fill round is the run of one operand's loads issued before any of their rows is first read "
                   "by a later issue; rounds whose block footprints share a local row form one region; "
                   "alternating: several regions and the rounds return to a region after leaving it",
        limits=["footprints are declared rows x ceil(cols / derived row elements) blocks at the configured "
                "per-state block step; banks are not interpreted"] + (
            [f"{n} loads unresolved: {why}" for why, n in consumption.unresolved.most_common()]
            if consumption.unresolved else []) + consumption.relative_limits(),
    )


def _distance(loads, consumption=None):
    consumed = [load for load in loads if load["first"] is not None]
    ahead = [load for load in consumed if load["computes_between"] > 0]
    loads_between = [load["loads_between"] for load in consumed]
    relative = {}
    if consumption is not None and consumption.spaces:
        # An unread load later read through another base may have been consumed: not established.
        relative = dict(unestablished_consumer_loads=sum(1 for load in loads if consumption.unestablished(load)))
    return dict(
        **relative,
        loads=len(loads),
        consumed_loads=len(consumed),
        unconsumed_loads=len(loads) - len(consumed),
        unresolved_footprint_loads=sum(1 for load in loads if load["footprint"] is None),
        consumer_port=_Consumption.port(loads),
        loads_between=_spread(loads_between),
        computes_between=_spread([load["computes_between"] for load in consumed]),
        median_loads_between=_median(loads_between),
        min_loads_between=min(loads_between) if loads_between else None,
        fraction_with_compute_ahead=round(len(ahead) / len(consumed), 4) if consumed else None,
    )


def _prefetch_distance(view, stream, truncated, evidence_id):
    """Per operand: loads (and computes) issued between each load and the first issue reading its rows."""
    consumption = _consumption(view)
    if not consumption.loads:
        return _unknown("proved stream has no load with a resolved declared local-address field", scope=STREAM_SCOPE)
    tensors = {}
    for (_, operand), loads in consumption.groups().items():
        tensors.setdefault(operand, []).extend(loads)
    if not any(load["first"] is not None for load in consumption.loads):
        return _unknown("no later issue reads a row any resolved load wrote", scope=STREAM_SCOPE,
                        unresolved=dict(consumption.unresolved))
    return _stream_fact(
        dict(by_operand={k: _distance(v, consumption) for k, v in sorted(tensors.items())},
             all=_distance(consumption.loads, consumption)),
        ["structure#instruction_roles", "structure#local_geometry", f"{evidence_id}#command_stream"],
        stream,
        truncated,
        partial=bool(consumption.unresolved),
        definition="for each load, the loads and the compute issues issued after it and before the first later "
                   "issue whose declared local-address field reads a row the load wrote (latest writer); "
                   "fraction_with_compute_ahead: consumed loads with at least one compute in between",
        limits=["issue order of a proved scalar execution; a measured distance, not a timing or speed claim"]
        + [f"{n} loads unresolved: {why}" for why, n in consumption.unresolved.most_common()]
        + consumption.relative_limits(),
    )


def _issue_overlap(view, stream, truncated, evidence_id):
    runs, pending, loads = [], [], []
    computes = fences = barriers = 0
    for entry, klass, _, kind in view.entries:
        if kind == "local_staging":
            continue
        if runs and runs[-1][0] == kind:
            runs[-1][1] += 1
        else:
            runs.append([kind, 1])
        if kind == "barrier":
            barriers += 1
            fences += klass == "FENCE"
            pending = []
        elif kind == "compute":
            computes += 1
            for load in pending:
                load[1] += 1
        elif kind == "load":
            operand, _ = view.dram(klass, entry["fields"])
            load = [operand or "unattributed", 0]
            loads.append(load)
            pending.append(load)
    if not computes and not loads:
        return _unknown("proved stream has no compute or load issues", scope=STREAM_SCOPE)

    def overlap(rows):
        if not rows:
            return {}
        hit = sum(1 for _, n in rows if n) / len(rows)
        return dict(
            loads=len(rows),
            loads_with_compute_before_barrier=round(hit, 4),
            loads_without_compute_before_barrier=round(1 - hit, 4),
            computes_after_load_before_barrier=_spread([n for _, n in rows]),
        )

    compute_runs = [n for k, n in runs if k == "compute"]
    load_runs = [n for k, n in runs if k == "load"]
    # Interleave over the compute/load order alone (barriers and other issues between them are
    # ignored): a compute run followed by a load band, and compute runs between two load bands.
    bands = []
    for kind, n in runs:
        if kind not in ("compute", "load"):
            continue
        if bands and bands[-1][0] == kind:
            bands[-1][1] += n
        else:
            bands.append([kind, n])
    ratios = [m / n for (k, n), (j, m) in zip(bands, bands[1:], strict=False) if k == "compute" and j == "load"]
    between = [
        n
        for (p, _), (k, n), (q, _) in zip(bands, bands[1:], bands[2:], strict=False)
        if k == "compute" and p == q == "load"
    ]
    whole = overlap(loads)
    ports = _consumption(view).by_tensor_port()
    value = dict(
        runs_prefix=[list(run) for run in runs[:_RUNS_PREFIX]],
        run_count=len(runs),
        compute_run=_spread(compute_runs),
        load_run=_spread(load_runs),
        median_compute_run=_median(compute_runs),
        median_load_run=_median(load_runs),
        loads_per_compute_run=_median(ratios),
        compute_runs_followed_by_load_band=len(ratios),
        median_compute_run_between_load_bands=_median(between),
        computes=computes,
        loads=len(loads),
        loads_with_compute_before_barrier=whole.get("loads_with_compute_before_barrier"),
        loads_without_compute_before_barrier=whole.get("loads_without_compute_before_barrier"),
        by_operand={k: dict(overlap([row for row in loads if row[0] == k]), consumer_port=ports.get(k))
                    for k in sorted({row[0] for row in loads})},
        fences=fences,
        barriers=barriers,
        computes_per_fence=round(computes / fences, 4) if fences else None,
        computes_per_barrier=round(computes / barriers, 4) if barriers else None,
    )
    return _stream_fact(
        value,
        ["structure#instruction_roles", f"{evidence_id}#command_stream"],
        stream,
        truncated,
        limits=["issue order of a proved scalar execution; hardware overlap and timing are not observed"],
    )


def _readout_pitch_fields(view, structure):
    """``{config class: field}`` of the stride-kind configuration fields whose RTL sink lives in the same
    module as a readout class's declared transfer fields (the memory row pitch of readouts)."""
    readout_bundles = set()
    for (klass, name), binding in view.proto.bindings.items():
        roles = _roles(structure, klass)
        if any(r in ("readout", "commit") for r in roles) and binding.get("field") == view.proto.extent["cols"]:
            readout_bundles.add(binding.get("bundle"))
    modules = {row.get("module") for row in structure.get("field_layouts") or ()
               if row.get("bundle") in readout_bundles and row.get("module")}
    found = {}
    for row in structure.get("operand_field_kind_sources") or ():
        klass, _, name = str(row.get("label") or "").partition(".")
        if row.get("kind") == "stride" and row.get("module") in modules and (klass, name) in view.proto.bindings:
            found.setdefault(klass, []).append(name)
    return {klass: names[0] for klass, names in found.items() if len(names) == 1}


def _output_group_extent(view, stream, truncated, evidence_id):
    """Per barrier-delimited block: computes and the readouts' declared extents and memory span.

    ``column_groups`` counts the distinct column positions (byte offset modulo the configured readout row
    pitch) a block's readouts start at; ``output_columns`` is that count times the readouts' common
    column extent, when the pitch and a uniform extent are established."""
    pitch_fields = _readout_pitch_fields(view, view.structure)
    pitch = None
    blocks, block = [], None

    def fresh():
        return dict(computes=0, readouts=0, shapes=Counter(), elements=0, offsets=set(), rows=set(),
                    starts=set(), row_starts=set(), cols=set(), pitch_unknown=0)

    block = fresh()
    for entry, klass, _, kind in view.entries:
        if klass in pitch_fields:
            value = entry["fields"].get(pitch_fields[klass])
            pitch = value if type(value) is int and value > 0 else None
        if kind == "barrier":
            if block["computes"] or block["readouts"]:
                blocks.append(block)
            block = fresh()
        elif kind == "compute":
            block["computes"] += 1
        elif kind == "readout":
            r, c = view.extent(klass, entry["fields"])
            block["readouts"] += 1
            block["shapes"][(r, c)] += 1
            block["rows"].add(r)
            block["cols"].add(c)
            if r is not None and c is not None:
                block["elements"] += r * c
            operand, _ = view.dram(klass, entry["fields"])
            record = entry["fields"].get(view.operand_fields.get(klass) or "")
            offset = None
            if isinstance(record, dict):
                offset = record.get("byte_offset") if record.get("pointer_base") else record.get("raw")
                if type(offset) is int:
                    block["offsets"].add((operand, offset))
            if type(offset) is int and pitch:
                block["starts"].add((operand, offset % pitch))
                block["row_starts"].add((operand, offset // pitch))
            else:
                block["pitch_unknown"] += 1
    if block["computes"] or block["readouts"]:
        blocks.append(block)
    out = [b for b in blocks if b["readouts"]]
    if not out:
        return _unknown("proved stream has no readout-role issue with declared extents", scope=STREAM_SCOPE)

    def row(b):
        offsets = sorted(o for _, o in b["offsets"])
        single = len(b["rows"]) == 1 and None not in b["rows"]
        rows = next(iter(b["rows"])) if single else None
        uniform = len(b["cols"]) == 1 and None not in b["cols"]
        groups = len(b["starts"]) if not b["pitch_unknown"] else None
        return dict(
            computes=b["computes"],
            readouts=b["readouts"],
            shapes=[dict(rows=r, cols=c, count=n) for (r, c), n in b["shapes"].most_common(_BOUND)],
            readout_elements=b["elements"],
            row_extent=rows,
            column_extent_if_one_row_band=b["elements"] // rows if rows else None,
            column_groups=groups,
            row_bands=len(b["row_starts"]) if not b["pitch_unknown"] else None,
            output_columns=groups * next(iter(b["cols"])) if groups is not None and uniform else None,
            distinct_memory_offsets=len(offsets),
            memory_byte_span=offsets[-1] - offsets[0] if offsets else None,
        )

    rows = [row(b) for b in out]
    columns = [r["output_columns"] for r in rows]
    value = dict(
        blocks=len(rows),
        blocks_prefix=rows[:_BOUND],
        median_readouts_per_block=_median([r["readouts"] for r in rows]),
        median_readout_elements_per_block=_median([r["readout_elements"] for r in rows]),
        column_extents=dict(Counter(r["column_extent_if_one_row_band"] for r in rows).most_common(_BOUND)),
        row_extents=dict(Counter(r["row_extent"] for r in rows).most_common(_BOUND)),
        median_output_columns=_median(columns) if None not in columns else None,
        output_columns=dict(Counter(columns).most_common(_BOUND)),
        readout_pitch_fields=pitch_fields,
        median_computes_per_readout_block=_median([r["computes"] for r in rows]),
        compute_only_blocks=sum(1 for b in blocks if not b["readouts"]),
    )
    return _stream_fact(
        value,
        ["structure#instruction_roles", "field_layouts#transfer_bound", "structure#operand_field_kind_sources",
         f"{evidence_id}#command_stream"],
        stream,
        truncated,
        limits=[
            "column_extent_if_one_row_band assumes a block's readouts share one row band; "
            "memory_byte_span lets a consumer check that against the declared row stride",
            "output_columns counts distinct column start positions modulo the configured readout row pitch "
            "times the common column extent; overlapping column tiles are not excluded",
        ]
        + ([] if pitch_fields else ["no stride-kind configuration field shares the readout's RTL module; "
                                    "output_columns stays unknown"]),
    )


def _extension_facts(stream, structure):
    """Facts of the optional ``stream_effects`` module over the same decoded stream."""
    try:
        from merlin.kernels import stream_effects
    except ImportError:
        return {}
    derive = getattr(stream_effects, "derive", None)
    if derive is None:
        return {}
    try:
        facts = derive(stream, structure) or {}
    except (ArithmeticError, LookupError, TypeError, ValueError) as error:
        return {"stream_effects": _unknown(f"stream_effects.derive failed: {type(error).__name__}: {error}")}
    return {name: fact for name, fact in facts.items() if name not in _STREAM_FACTS}


def stream_facts(stream, structure, *, evidence_id="occurrence", arguments=None, truncated=False):
    """The issue-order facts of one decoded proved stream; unknown without one."""
    if stream is None:
        return {name: _unknown(_NO_STREAM, scope=STREAM_SCOPE) for name in _STREAM_FACTS}
    view = _StreamView(stream, structure or {}, _Protocol(structure or {}), arguments)
    facts = dict(
        role_transfer_width=_role_transfer_width(view, stream, truncated, evidence_id),
        operand_buffering=_operand_buffering(view, stream, truncated, evidence_id),
        issue_overlap=_issue_overlap(view, stream, truncated, evidence_id),
        prefetch_distance=_prefetch_distance(view, stream, truncated, evidence_id),
        output_group_extent=_output_group_extent(view, stream, truncated, evidence_id),
    )
    facts.update(_extension_facts(stream, structure or {}))
    return facts


# ----------------------------------------------------------------------------- entry point
def derive_effects(
    analysis, structure, *, commands=None, evidence_id=None, stream=None, arguments=None, stream_truncated=False
):
    """Return the ``derived_effects`` block for one occurrence analysis record.

    ``analysis``: the occurrence's ``analysis.json`` dict. ``structure``: from
    :func:`selected_structure` (role roster, operand protocol, RTL field layouts,
    load-state selection/ports, encoding, epilogue carriers). ``commands``: the
    optional complete ordered site list from :func:`decoded_commands`.
    ``stream``: the optional :func:`decode_stream` of a proved execution; it adds
    ``role_transfer_width``, ``operand_buffering``, ``issue_overlap``, ``prefetch_distance`` and
    ``output_group_extent`` (each unknown
    with ``no_proved_command_stream`` without it). ``arguments``: exact entry argument
    registers (``a0``..) that tie exact memory addresses to tensor arguments.
    """
    analysis, structure = analysis or {}, structure or {}
    evidence_id = (
        evidence_id or (analysis.get("evidence") or {}).get("evidence_id") or analysis.get("region_id") or "occurrence"
    )
    projection = analysis.get("mechanism_projection") or {}
    proto = _Protocol(structure)
    unknowns = []
    consistency = None
    if commands is not None:
        replay = Counter(r["instruction"] for r in commands)
        declared = Counter({k: v for k, v in (projection.get("instruction_classes") or {}).items() if v})
        consistency = dict(
            status="consistent" if not declared or replay == declared else "differs",
            replayed_sites=sum(replay.values()),
            analysis_classified_sites=sum(declared.values()),
        )
        if consistency["status"] == "differs":
            unknowns.append(dict(fact="*", reason="replayed commands differ from the analysis record"))
    if not projection:
        facts = {
            name: _unknown("occurrence has no selected accelerator projection")
            for name in (
                "operand_retention",
                "broadcast_loads",
                "transfer_tiles",
                "local_address_windows",
                "epilogue_configuration",
            )
        }
    elif commands is None:
        reason = "ordered decoded commands not supplied"
        facts = dict(
            operand_retention=_unknown(reason),
            broadcast_loads=_summary_broadcast(analysis, structure, proto),
            transfer_tiles=_unknown(reason + "; marginal profiles cannot pair rows with cols"),
            local_address_windows=_summary_windows(analysis, structure, proto),
            epilogue_configuration=_epilogue(None, structure, proto, evidence_id, analysis),
        )
    else:
        facts = dict(
            operand_retention=_retention(commands, structure, proto, evidence_id),
            broadcast_loads=_broadcast(commands, structure, proto, evidence_id),
            transfer_tiles=_tiles(commands, structure, proto, evidence_id),
            local_address_windows=_windows(commands, structure, proto, evidence_id),
            epilogue_configuration=_epilogue(commands, structure, proto, evidence_id, analysis),
        )
    roles = structure.get("instruction_roles") or {}
    observed = [k for k, v in (projection.get("instruction_classes") or {}).items() if v]
    if observed and all(roles.get(k) for k in observed):
        loops = sum(v for k, v in projection["instruction_classes"].items() if "loop_descriptor" in roles[k])
        facts["hardware_loop"] = _fact(
            dict(loop_descriptor_sites=loops, uses_hardware_loop=bool(loops)),
            "observed",
            scope=dict(static_sites=sum(projection["instruction_classes"].values())),
            evidence=["structure#instruction_roles", "analysis#mechanism_projection.instruction_classes"],
        )
    else:
        facts["hardware_loop"] = _unknown("role roster does not cover every observed instruction class")
    dataflow = [unit.get("dataflow") for unit in structure.get("compute_units") or () if unit.get("dataflow")]
    facts["dataflow"] = (
        _fact(sorted(set(map(str, dataflow))), "declared", evidence=["structure#compute_units"])
        if dataflow
        else _unknown("no selected protocol field or compute-unit declaration binds a dataflow selector")
    )
    facts["loop_nest"] = _loop_nest(projection, analysis, evidence_id)
    trips = (analysis.get("cfg") or {}).get("trip_counts", "unknown")
    facts["operation"] = _unknown(
        "operation kind and logical dimensions need a generic contract or loop-bound "
        "analysis; static trip counts are " + (trips if isinstance(trips, str) else str(trips.get("status"))),
        kind=None,
        logical_dims=None,
        dtypes=None,
    )
    facts["layer_identity"] = _unknown("no source or build correspondence; names are never inferred")
    facts.update(
        stream_facts(stream, structure, evidence_id=evidence_id, arguments=arguments, truncated=stream_truncated)
    )
    for name, fact in facts.items():
        if fact["status"] == "unknown":
            unknowns.append(dict(fact=name, reason=fact.get("reason")))
    return dict(
        schema=SCHEMA,
        evidence_id=evidence_id,
        basis="ordered_decoded_commands" if commands is not None else "analysis_summary",
        ordering=ORDERING,
        replay=consistency,
        inputs=dict(
            structure_sha256=_digest({k: v for k, v in structure.items() if k != "input_pins"}),
            input_pins=structure.get("input_pins") or [],
            commands_sha256=None if commands is None else _digest(commands),
            **({} if stream is None else dict(stream_sha256=_digest(stream), stream_entries=len(stream))),
        ),
        facts=facts,
        unknowns=unknowns,
        not_established=[
            "dynamic_execution_multiplicity",
            "optimization_legality",
            "numerical_equivalence",
            "performance",
        ],
    )
