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

from merlin.analysis.kernels.effects.static_geometry import (
    _DOMAIN,
    _MOVEMENT_ROLES,
    _VALUES,
    _WITNESSES,
    ORDERING,
    SCHEMA,
    _bits,
    _broadcast,
    _close,
    _digest,
    _epilogue,
    _fact,
    _field_domain,
    _loop_nest,
    _merge,
    _Protocol,
    _register_domain,
    _retention,
    _roles,
    _site,
    _summary_broadcast,
    _summary_windows,
    _symbolic_bits,
    _tiles,
    _unknown,
    _windows,
    decoded_commands,
    local_row_geometry,
    read_pinned_facts,
    read_pinned_sidecar,
    selected_structure,
)

# ---- decoded_effects ---------------------------------------------------------------------------

STREAM_SCOPE = "proved_dynamic_command_stream"


_NO_STREAM = "no_proved_command_stream"


_STREAM_FACTS = (
    "role_transfer_width",
    "operand_buffering",
    "issue_overlap",
    "prefetch_distance",
    "output_group_extent",
)


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
        from merlin.analysis.observation.riscv.machine import expand_command_stream

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
            rows = self.proto.fields(
                klass,
                lambda b, slot=slot: (
                    b.get("operand") == slot and b["interp"] == "int" and b["field"] == self.proto.extent["rows"]
                ),
            )
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
                    out.append(
                        (
                            "/".join(names),
                            flags,
                            relative["offset"],
                            rows,
                            (relative["symbol"], relative["width"], high & ~mask),
                        )
                    )
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
        return None, (
            "no unique per-load-state stride field of the load-configuration class other than the "
            "memory stride (RTL field-kind sources)"
        )
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
        self.geometry_reason = (
            None
            if type(self.row_elements) is int
            else (geometry.get("reason") or "local row width not derived from the selected RTL facts")
        )
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
                load = dict(
                    position=position,
                    ordinal=loads_seen,
                    computes_before=computes_seen,
                    role=view.movement_role(roles),
                    operand=operand or "unattributed",
                    flags=flags,
                    first=None,
                    port=None,
                    ports=set(),
                    footprint=None,
                    space=space,
                )
                loads_seen += 1
                group = (load["role"], load["operand"], flags)
                load["round"] = current[group]
                reason = None
                missing = [
                    name
                    for name, value in (("local address", base), ("transfer rows", rows), ("transfer columns", cols))
                    if value is None
                ]
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
                        load.update(
                            first=position,
                            port=port,
                            loads_between=loads_seen - load["ordinal"] - 1,
                            computes_between=computes_seen - load["computes_before"] - (kind == "compute"),
                        )
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
        return any(position > load["position"] for space, position in self._last_read.items() if space != load["space"])

    def relative_limits(self):
        if not self.spaces:
            return []
        return [
            f"{len(self.spaces)} allocation-relative local-address bases (unproved call returns): rows "
            "compare only within one base; disjointness or overlap across allocations is not established, "
            "so a writer of another base is never matched to a reader"
        ]

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
            role: dict(
                close(row),
                by_operand={
                    k: dict(close(v), consumer_port=ports.get(f"{role}:{k}"))
                    for k, v in sorted(row["by_operand"].items())
                },
            )
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
        footprints.setdefault(load["round"], set()).update(row for lo, hi in load["footprint"] for row in range(lo, hi))
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
        out.append(dict(lo=rows[0], hi=rows[-1], rows=len(rows), intervals=spans, rounds=sequence.count(live.index(i))))
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
