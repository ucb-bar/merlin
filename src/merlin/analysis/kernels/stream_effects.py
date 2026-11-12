"""Facts over a proved dynamic accelerator command stream, from declared roles and field kinds.

``derive(stream, structure, rtl_facts=...)`` reads an ORDERED list of decoded accelerator commands
(the dynamic issue order, fences included) and returns five facts, plus the resource, transfer-bound
and post-store facts of :mod:`merlin.analysis.kernels.stream_resources`. Every value comes from a
declared role, a declared field kind or an RTL fact used as data; nothing selects on an instruction
class or field spelling. A quantity the declarations cannot support is an explicit ``unknown`` with
a reason.

Stream entry::

    {seq, pc, class, roles: [role, ...], fields: {name: int | bool | float | [symbol, offset] | None},
     host_instructions_since_prev: int | None}

``roles`` fall back to ``structure["instruction_roles"][class]`` when an entry carries none.

Structure keys read (all optional; a missing one makes the facts that need it unknown):

``instruction_roles``
    ``{class: [role, ...]}``. Roles read: ``operand_load``/``weight_load`` (loads), ``accumulate``
    (compute), ``readout``/``commit`` (readout), ``config`` and ``sync`` (fence).
``operand_field_kinds``
    ``{"CLASS.field": kind}`` or ``{"CLASS.field": {"kind": kind, "axis": "rows"|"cols"}}``. Kinds
    read: ``local_address``, ``accumulator_select`` (a flag field whose truth selects the accumulator
    space), ``dram_address``, ``stride``, ``extent``, ``load_state``, ``scale``, ``activation``.
``operand_protocol``
    the operand protocol layout roster. ``roles.local_address_flags`` (``{flag: {msb_offset}}``) and
    ``encoding.addr_len`` locate flag bits that are cleared from a local address; a field declared
    ``as: flag`` whose kind is ``accumulator_select`` names the flag whose bit marks an accumulator
    address on any local-address field. ``address_field.field`` with ``roles.classes`` bindings is
    an alternative declaration of local-address fields.
``load_state_selection.transfer_bound.{rows,cols}.name``
    the RTL field names bound to the transfer extents, giving an ``extent`` field its axis.
``load_state_ports``
    ``{class: state index}`` from the RTL load-state facts; pairs a load with the config command
    whose ``load_state`` field equals that index.
``field_identity``
    ``{"CLASS.field": value}`` declared identity values for ``activation`` fields. A ``scale``
    field's identity is the multiplicative identity 1.

``rtl_facts`` (optional) is an RTL-facts artifact or bare body, read through
``merlin.development.targetgen.address_space`` for store rows, per-bank depth, bank count, row elements and
element width.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

SCOPE = "proved_dynamic_command_stream"
_LOAD_ROLES = ("operand_load", "weight_load")
_COMPUTE_ROLES = ("accumulate",)
_READOUT_ROLES = ("readout", "commit")
_CONFIG_ROLE = "config"
_SYNC_ROLE = "sync"
_THRESHOLDS = (2, 4, 8)
_SPAD, _ACC = "scratchpad", "accumulator"


def _fact(value, evidence, *, status="observed", reason=None):
    out = {"status": status, "value": value, "scope": SCOPE, "evidence": evidence}
    if reason is not None:
        out["reason"] = reason
    return out


def _unknown(reason, evidence=None, value=None):
    return _fact(value, evidence or {}, status="unknown", reason=reason)


def _is_int(v):
    return type(v) is int


# ----------------------------------------------------------------------------- declared structure
class _Declared:
    """The declared roles, field kinds, address flags and load-state ports, normalised."""

    def __init__(self, structure: dict[str, Any] | None):
        structure = structure or {}
        self.instruction_roles = structure.get("instruction_roles") or {}
        protocol = structure.get("operand_protocol") or {}
        bound = (structure.get("load_state_selection") or {}).get("transfer_bound") or {}
        axis_of_field = {(bound.get(axis) or {}).get("name"): axis for axis in ("rows", "cols")}
        axis_of_field.pop(None, None)
        self.kinds: dict[tuple[str, str], str] = {}
        self.axes: dict[tuple[str, str], str] = {}
        for key, spec in (structure.get("operand_field_kinds") or {}).items():
            klass, sep, name = str(key).partition(".")
            if not sep or not name:
                continue
            kind = spec.get("kind") if isinstance(spec, dict) else spec
            if not isinstance(kind, str):
                continue
            self.kinds[(klass, name)] = kind
            axis = spec.get("axis") if isinstance(spec, dict) else None
            if axis in ("rows", "cols"):
                self.axes[(klass, name)] = axis
        roles = protocol.get("roles") or {}
        bindings = {}
        for klass, spec in (roles.get("classes") or {}).items():
            for field in (spec or {}).get("fields") or ():
                if isinstance(field, dict) and field.get("name"):
                    bindings[(klass, field["name"])] = field
        # An extent's axis: declared on the kind, else the RTL transfer-bound field it binds.
        for key, kind in self.kinds.items():
            if kind == "extent" and key not in self.axes:
                bound_field = (bindings.get(key) or {}).get("field")
                if bound_field in axis_of_field:
                    self.axes[key] = axis_of_field[bound_field]
        address_field = (protocol.get("address_field") or {}).get("field")
        if address_field is not None:
            for key, field in bindings.items():
                if field.get("field") == address_field and field.get("as", "int") == "int":
                    self.kinds.setdefault(key, "local_address")
        addr_len = (structure.get("encoding") or protocol.get("encoding") or {}).get("addr_len")
        self.flag_bits: dict[str, int] = {}
        if _is_int(addr_len) and addr_len > 0:
            for name, spec in (roles.get("local_address_flags") or {}).items():
                offset = (spec or {}).get("msb_offset")
                if _is_int(offset) and 0 <= offset < addr_len:
                    self.flag_bits[name] = 1 << (addr_len - 1 - offset)
        self.flag_mask = sum(self.flag_bits.values())
        select_flags = {
            field.get("flag")
            for key, field in bindings.items()
            if self.kinds.get(key) == "accumulator_select" and field.get("as") == "flag"
        }
        self.accumulator_bit = self.flag_bits.get(next(iter(select_flags))) if len(select_flags) == 1 else None
        self.ports = {k: v for k, v in (structure.get("load_state_ports") or {}).items() if _is_int(v)}
        self.identity = dict(structure.get("field_identity") or {})

    def roles(self, entry) -> tuple[str, ...]:
        declared = entry.get("roles")
        if declared:
            return tuple(declared)
        return tuple(self.instruction_roles.get(entry.get("class")) or ())

    def fields_of_kind(self, klass, kind) -> list[str]:
        return [name for (k, name), v in self.kinds.items() if k == klass and v == kind]

    def kind_declared(self, kind) -> bool:
        return kind in self.kinds.values()


def _entries(stream, declared):
    out = []
    for index, entry in enumerate(stream or ()):
        if not isinstance(entry, dict):
            continue
        out.append(
            dict(
                seq=entry.get("seq", index),
                pc=entry.get("pc"),
                klass=entry.get("class"),
                roles=declared.roles(entry),
                fields=entry.get("fields") or {},
                host=entry.get("host_instructions_since_prev"),
            )
        )
    return out


def _blocks(entries):
    """Fence-delimited block index of every entry; a sync command closes its block."""
    block, out = 0, []
    for entry in entries:
        out.append(block)
        if _SYNC_ROLE in entry["roles"]:
            block += 1
    return out


# ----------------------------------------------------------------------------- 1. host issue rate
def _quantiles(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    median = ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    p90 = ordered[min(n - 1, max(0, -(-9 * n // 10) - 1))]  # nearest rank
    return {"n": n, "median": median, "p90": p90, "max": ordered[-1]}


def _host_issue_rate(entries):
    values = [e["host"] for e in entries if _is_int(e["host"]) and e["host"] >= 0]
    missing = sum(1 for e in entries if not (_is_int(e["host"]) and e["host"] >= 0))
    evidence = {"field": "host_instructions_since_prev", "commands": len(entries), "unresolved": missing}
    if not values:
        return _unknown("no command carries a resolved host_instructions_since_prev", evidence)
    by_role = defaultdict(list)
    for e in entries:
        if _is_int(e["host"]) and e["host"] >= 0:
            for role in e["roles"] or ("undeclared",):
                by_role[role].append(e["host"])
    value = dict(
        overall=_quantiles(values),
        by_role={role: _quantiles(v) for role, v in sorted(by_role.items())},
        share_below={str(n): round(sum(1 for v in values if v < n) / len(values), 6) for n in _THRESHOLDS},
    )
    return _fact(value, evidence)


# ----------------------------------------------------------------------------- 2. reconfiguration
def _reconfiguration(entries, blocks):
    evidence = {"roles": [_CONFIG_ROLE, *_COMPUTE_ROLES, _SYNC_ROLE], "ordering": "stream order"}
    if not any(e["roles"] for e in entries):
        return _unknown("no command carries a declared role", evidence)
    computes = [i for i, e in enumerate(entries) if any(r in _COMPUTE_ROLES for r in e["roles"])]
    configs = [i for i, e in enumerate(entries) if _CONFIG_ROLE in e["roles"]]
    per_block = Counter(blocks[i] for i in configs)
    value = dict(
        config_commands=len(configs),
        config_by_class=dict(sorted(Counter(str(entries[i]["klass"]) for i in configs).items())),
        blocks=(blocks[-1] + 1) if blocks else 0,
        config_per_block={str(b): n for b, n in sorted(per_block.items())},
    )
    if not computes:
        value.update(hoisted=None, inside_compute_run=None, after_last_compute=None)
        return _fact(value, evidence, status="unknown", reason="no accumulate-role command in the stream")
    first, last = computes[0], computes[-1]
    value.update(
        hoisted=sum(1 for i in configs if i < first),
        inside_compute_run=sum(1 for i in configs if first < i < last),
        after_last_compute=sum(1 for i in configs if i > last),
    )
    return _fact(value, evidence)


# ----------------------------------------------------------------------------- RTL store geometry
def _stores(rtl_facts):
    """``({space: Store}, reasons)`` from the RTL facts' address space; empty when not given."""
    if rtl_facts is None:
        return {}, {_SPAD: "no rtl_facts given", _ACC: "no rtl_facts given"}
    from merlin.development.targetgen.address_space import accumulator_store, derive_address_space, operand_store

    label = str(rtl_facts.get("target") or "rtl_facts") if isinstance(rtl_facts, dict) else "rtl_facts"
    space = derive_address_space(label, facts=rtl_facts)
    out, reasons = {}, {}
    for name, resolution in ((_SPAD, operand_store(space)), (_ACC, accumulator_store(space))):
        if resolution.store is not None:
            out[name] = resolution.store
        else:
            reasons[name] = resolution.reason
    return out, reasons


def _geometry(store):
    if store is None:
        return None
    return dict(
        name=store.name,
        capacity_rows=store.capacity_rows(),
        banks=store.banks,
        rows_per_bank=store.depth,
        elems_per_row=store.elems_per_row(),
        element_bits=store.element_bits,
    )


# ----------------------------------------------------------------------------- 3. local capacity
def _local_ref(declared, entry, name):
    """``(space, row)`` of one local-address field, flag bits cleared, or ``None``."""
    value = entry["fields"].get(name)
    if not _is_int(value) or value < 0:
        return None
    space = _SPAD
    if declared.accumulator_bit is not None and value & declared.accumulator_bit:
        space = _ACC
    for flag_field in declared.fields_of_kind(entry["klass"], "accumulator_select"):
        if entry["fields"].get(flag_field) is True:
            space = _ACC
    if declared.accumulator_bit is None and not declared.fields_of_kind(entry["klass"], "accumulator_select"):
        space = None  # no declared space selector: the store the row belongs to is unknown
    return space, value & ~declared.flag_mask


def _extent(declared, entry, axis):
    names = [
        n for n in declared.fields_of_kind(entry["klass"], "extent") if declared.axes.get((entry["klass"], n)) == axis
    ]
    if len(names) != 1:
        return None
    value = entry["fields"].get(names[0])
    return value if _is_int(value) and value >= 0 else None


def _footprint(declared, entry, store, row):
    """Rows ``[row, row + n)`` one addressed transfer occupies, or ``None``.

    Row-major residency: ``rows`` rows each spanning ``ceil(cols / elems_per_row)`` store rows
    (``address_space.working_set_rows``). Without a cols extent the transfer is one row per row.
    """
    rows = _extent(declared, entry, "rows")
    if rows is None:
        return None
    cols = _extent(declared, entry, "cols")
    if cols is None:
        return range(row, row + rows)
    if store is None:
        return None
    n = store.working_set_rows([rows, cols])
    return None if n is None else range(row, row + n)


def _local_capacity_use(entries, blocks, declared, stores, store_reasons):
    evidence = {
        "kinds": ["local_address", "accumulator_select", "extent"],
        "address_flags": sorted(declared.flag_bits),
        "stores": {k: _geometry(v) for k, v in stores.items()},
    }
    if not declared.kind_declared("local_address"):
        return _unknown("no field is declared of kind local_address", evidence)
    touched = defaultdict(lambda: defaultdict(set))  # block -> space -> rows
    addresses = defaultdict(lambda: defaultdict(set))
    role_windows = defaultdict(list)  # load role -> [(block, space, rows)]
    unresolved, no_footprint = 0, 0
    for entry, block in zip(entries, blocks):
        for name in declared.fields_of_kind(entry["klass"], "local_address"):
            ref = _local_ref(declared, entry, name)
            if ref is None:
                unresolved += 1
                continue
            space, row = ref
            key = space or "undetermined"
            addresses[block][key].add(row)
            rows = _footprint(declared, entry, stores.get(space), row)
            if rows is None:
                no_footprint += 1
                continue
            touched[block][key].update(rows)
            for role in entry["roles"]:
                if role in _LOAD_ROLES:
                    role_windows[role].append((block, key, rows))
    per_block = {}
    peak = Counter()
    for block in sorted(set(touched) | set(addresses)):
        row = {}
        for space in sorted(set(touched[block]) | set(addresses[block])):
            n = len(touched[block][space])
            peak[space] = max(peak[space], n)
            row[space] = {"distinct_rows": n, "distinct_addresses": len(addresses[block][space])}
        per_block[str(block)] = row
    capacity = {}
    for space, n in sorted(peak.items()):
        store = stores.get(space)
        cap = store.capacity_rows() if store is not None else None
        capacity[space] = dict(
            peak_distinct_rows_per_block=n,
            capacity_rows=cap,
            fraction=round(n / cap, 6) if cap else None,
            reason=None if cap else (store_reasons.get(space) or "no capacity for this space in the RTL facts"),
        )
    value = dict(
        per_block=per_block,
        capacity=capacity,
        unresolved_addresses=unresolved,
        addresses_without_footprint=no_footprint,
        bank_spread=_bank_spread(role_windows, stores, store_reasons),
    )
    return _fact(value, evidence)


def _bank_spread(role_windows, stores, store_reasons):
    """Bank index per load role, and whether a role's two alternating buffers share a bank."""
    out = {}
    for role, windows in sorted(role_windows.items()):
        spaces = {space for _, space, _ in windows}
        store = stores.get(next(iter(spaces))) if len(spaces) == 1 else None
        if store is None or not store.banks or not store.depth or store.total_rows != store.banks * store.depth:
            out[role] = {
                "status": "unknown",
                "reason": (
                    "role addresses span several spaces"
                    if len(spaces) != 1
                    else store_reasons.get(next(iter(spaces))) or "bank count or rows per bank unknown"
                ),
            }
            continue
        depth = store.depth
        banks = Counter()
        buffers = {}  # block -> rows
        for block, _, rows in windows:
            for b in range(rows.start // depth, (rows.stop - 1) // depth + 1):
                banks[b] += 1
            buffers.setdefault(block, set()).update(rows)
        sequence = []
        for block in sorted(buffers):
            key = (min(buffers[block]), max(buffers[block]))
            if not sequence or sequence[-1] != key:
                sequence.append(key)
        distinct = sorted(set(sequence))
        alternating = (
            len(distinct) == 2
            and len(sequence) >= 3
            and all(sequence[i] != sequence[i + 1] for i in range(len(sequence) - 1))
        )
        row = {
            "status": "observed",
            "banks": {str(b): n for b, n in sorted(banks.items()) if n},
            "buffers": len(distinct),
        }
        if alternating:
            sets = [set(range(lo // depth, hi // depth + 1)) for lo, hi in distinct]
            row["alternating"] = {
                "windows": [list(w) for w in distinct],
                "banks": [sorted(s) for s in sets],
                "same_bank": bool(sets[0] & sets[1]),
            }
        else:
            row["alternating"] = None
        out[role] = row
    return out


# ----------------------------------------------------------------------------- 4. DRAM contiguity
def _alignment(value):
    """Largest power of two dividing ``value``; ``0`` divides by every power and is kept apart."""
    return value & -value if value else 0


def _histogram(counter):
    return {("zero" if k == 0 else str(k)): n for k, n in sorted(counter.items())}


def _dram_contiguity(entries, declared, stores, store_reasons):
    evidence = {"kinds": ["stride", "load_state", "extent", "dram_address"], "load_state_ports": dict(declared.ports)}
    if not declared.kind_declared("stride"):
        return _unknown("no field is declared of kind stride", evidence)
    config_classes_with_stride = {k for (k, _), v in declared.kinds.items() if v == "stride"}
    stateful = {k for (k, _), v in declared.kinds.items() if v == "load_state"} & config_classes_with_stride
    current: dict[Any, int | None] = {}  # load state (or None when unstated) -> stride
    per_role = defaultdict(lambda: Counter())
    reasons = Counter()
    abs_align, rel_align = Counter(), Counter()
    for entry in entries:
        klass, fields = entry["klass"], entry["fields"]
        if _CONFIG_ROLE in entry["roles"] and klass in config_classes_with_stride:
            strides = [fields.get(n) for n in declared.fields_of_kind(klass, "stride")]
            stride = strides[0] if len(strides) == 1 and _is_int(strides[0]) else None
            if klass in stateful:
                states = [fields.get(n) for n in declared.fields_of_kind(klass, "load_state")]
                if len(states) == 1 and _is_int(states[0]):
                    current[states[0]] = stride
            elif not stateful:
                current[None] = stride
            continue
        load_roles = [r for r in entry["roles"] if r in _LOAD_ROLES]
        if not load_roles:
            continue
        for name in declared.fields_of_kind(klass, "dram_address"):
            address = fields.get(name)
            if _is_int(address):
                abs_align[_alignment(address)] += 1
            elif isinstance(address, (list, tuple)) and len(address) == 2 and _is_int(address[1]):
                rel_align[_alignment(address[1])] += 1
        if stateful:
            port = declared.ports.get(klass)
            stride = current.get(port) if port is not None else None
            why = None if port is not None else "load class has no declared load_state port"
        else:
            stride = current.get(None)
            why = None
        if why is None and stride is None:
            why = "no preceding config command with a resolved stride for this load"
        cols = _extent(declared, entry, "cols")
        spaces = [_local_ref(declared, entry, n) for n in declared.fields_of_kind(klass, "local_address")]
        space = spaces[0][0] if len(spaces) == 1 and spaces[0] is not None else None
        store = stores.get(space)
        bits = store.element_bits if store is not None else None
        if why is None and cols is None:
            why = "no extent field of axis cols"
        if why is None and not bits:
            why = store_reasons.get(space) or "no declared element width for the destination store"
        for role in load_roles:
            if why is not None:
                per_role[role]["unknown"] += 1
                reasons[why] += 1
                continue
            row_bytes = cols * bits // 8 if (cols * bits) % 8 == 0 else None
            if row_bytes is None:
                per_role[role]["unknown"] += 1
                reasons["row is not a whole number of bytes"] += 1
            else:
                per_role[role]["contiguous" if stride == row_bytes else "strided"] += 1
                per_role[role][f"stride={stride},row_bytes={row_bytes}"] += 1
    value = dict(
        per_role={
            role: {
                "loads": sum(n for k, n in c.items() if k in ("contiguous", "strided", "unknown")),
                "contiguous": c["contiguous"],
                "strided": c["strided"],
                "unknown": c["unknown"],
                "stride_vs_row_bytes": {k: n for k, n in sorted(c.items()) if k.startswith("stride=")},
            }
            for role, c in sorted(per_role.items())
        },
        dram_alignment_histogram=_histogram(abs_align),
        dram_offset_alignment_relative_to_symbol=_histogram(rel_align),
        unknown_reasons=dict(reasons),
    )
    if not per_role:
        return _unknown("no load-role command in the stream", evidence, value)
    known = any(c["contiguous"] or c["strided"] for c in per_role.values())
    if not known:
        return _fact(value, evidence, status="unknown", reason="; ".join(sorted(reasons)))
    return _fact(value, evidence)


# ----------------------------------------------------------------------------- 5. epilogue placement
def _is_identity(declared, klass, name, kind, value):
    key = f"{klass}.{name}"
    if key in declared.identity:
        return value == declared.identity[key]
    if kind == "scale" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return value == 1
    return None


def _epilogue_placement(entries, declared):
    evidence = {"kinds": ["activation", "scale"], "roles": [*_READOUT_ROLES, _CONFIG_ROLE]}
    if not (declared.kind_declared("activation") or declared.kind_declared("scale")):
        return _unknown("no field is declared of kind activation or scale", evidence)
    load_configs = {k for (k, _), v in declared.kinds.items() if v == "load_state"}
    fields = defaultdict(lambda: {"values": Counter(), "non_identity": 0, "identity_unknown": 0})
    for entry in entries:
        roles = entry["roles"]
        on_readout = any(r in _READOUT_ROLES for r in roles)
        on_config = _CONFIG_ROLE in roles and entry["klass"] not in load_configs
        if not (on_readout or on_config):
            continue
        for kind in ("activation", "scale"):
            for name in declared.fields_of_kind(entry["klass"], kind):
                value = entry["fields"].get(name)
                if value is None:
                    continue
                row = fields[f"{entry['klass']}.{name}"]
                row["kind"] = kind
                row["values"][repr(value)] += 1
                identity = _is_identity(declared, entry["klass"], name, kind, value)
                if identity is None:
                    row["identity_unknown"] += 1
                elif not identity:
                    row["non_identity"] += 1
    report = {
        key: {
            "kind": r["kind"],
            "values": dict(r["values"]),
            "non_identity": r["non_identity"],
            "identity_unknown": r["identity_unknown"],
        }
        for key, r in sorted(fields.items())
    }
    evidence["excluded_load_state_config_classes"] = sorted(load_configs)
    if any(r["non_identity"] for r in report.values()):
        return _fact({"placement": "on_accelerator", "fields": report}, evidence)
    if any(r["identity_unknown"] for r in report.values()):
        return _unknown(
            "an activation field has no declared identity value",
            evidence,
            {"placement": None, "fields": report},
        )
    return _fact({"placement": "not_on_accelerator", "fields": report}, evidence)


# ----------------------------------------------------------------------------- entry point
def derive(stream, structure, *, rtl_facts=None) -> dict[str, dict]:
    """Five facts over an ordered decoded command stream, plus ``stream_resources.derive``'s three
    (``resource_occupancy``, ``transfer_legality``, ``post_store_reuse``); see the module docstrings."""
    declared = _Declared(structure)
    entries = _entries(stream, declared)
    blocks = _blocks(entries)
    stores, store_reasons = _stores(rtl_facts)
    from merlin.analysis.kernels import stream_resources

    resources = stream_resources.derive(
        stream, structure, bounds=stream_resources.stream_bounds(rtl_facts) if rtl_facts is not None else None
    )
    return {
        "host_issue_rate": _host_issue_rate(entries),
        "reconfiguration": _reconfiguration(entries, blocks),
        "local_capacity_use": _local_capacity_use(entries, blocks, declared, stores, store_reasons),
        "dram_contiguity": _dram_contiguity(entries, declared, stores, store_reasons),
        "epilogue_placement": _epilogue_placement(entries, declared),
        **resources,
    }
