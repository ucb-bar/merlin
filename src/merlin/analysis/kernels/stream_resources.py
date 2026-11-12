"""Resource, transfer-bound and post-store facts over a proved dynamic accelerator command stream.

``derive(stream, structure, bounds=None)`` returns three observations of one ordered decoded stream
(``{seq, class, roles, fields}`` entries, fences included as ``sync``-role entries):

``resource_occupancy``
    Per dispatch queue the selected RTL facts declare (``timing_structure.command_classes``), the
    largest number of commands routed to that queue issued since the last ``sync`` command, and the
    issue points where that count passes the queue's dispatch-table entries (and dispatch plus
    command-queue entries). Routing comes from :mod:`merlin.analysis.kernels.command_routing`. It is an
    issue-order upper bound that assumes no routed command retires before the next sync; it is not
    a timing measurement.
``transfer_legality``
    Per movement or readout role, the decoded transfer extents compared with bounds taken from the
    RTL facts: one transfer row's bytes against the largest single request of the DMA client of that
    direction (``timing_structure.memory_clients``), and the transfer's local footprint against the
    addressed store's rows (``targetgen.address_space``). A comparison the facts cannot settle is
    ``unknown`` with the reason.
``post_store_reuse``
    After each readout-role command, whether the local rows it read are next read again, accumulated
    into, or rewritten in the stream, and which sync commands sit between the store and that touch.

Every bound is data with its source; roles, flags and extents come from the declared operand
protocol (``structure["operand_protocol"]``, the RTL transfer-bound field names in
``structure["load_state_selection"]``). Local-address flag names read are the protocol vocabulary
``accumulator_select`` (marks the accumulator space) and ``accumulate`` (read-modify-write).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

SCOPE = "proved_dynamic_command_stream"
_SYNC = "sync"
_INBOUND = ("operand_load", "weight_load", "dma")
_READOUT = ("readout", "commit")
_ACC_FLAG, _ACCUMULATE_FLAG = "accumulator_select", "accumulate"
_SPAD, _ACC = "scratchpad", "accumulator"
_EXAMPLES = 8


def _fact(value, evidence, *, status="observed", reason=None, limits=None):
    out = {"status": status, "value": value, "scope": SCOPE, "evidence": evidence}
    if reason is not None:
        out["reason"] = reason
    if limits:
        out["limits"] = limits
    return out


def _unknown(reason, evidence=None, value=None):
    return _fact(value, evidence or {}, status="unknown", reason=reason)


def _is_int(v):
    return type(v) is int


# ----------------------------------------------------------------------------- selected bounds
def stream_bounds(rtl_facts) -> dict[str, Any]:
    """The RTL-fact bounds these facts read, compact and with sources; ``{}`` without facts."""
    if rtl_facts is None:
        return {}
    from merlin.analysis.kernels.command_routing import _body, queue_depths
    from merlin.analysis.kernels.stream_effects import _stores

    timing = _body(rtl_facts).get("timing_structure") or {}
    clients = {}
    for name, row in sorted(((timing.get("memory_clients") or {}).get("clients") or {}).items()):
        row = row or {}
        response, request = row.get("response_data_bits"), row.get("request_data_bits")
        direction = "inbound" if _is_int(response) else "outbound" if _is_int(request) else None
        size = row.get("max_request_bytes")
        clients[name] = dict(
            direction=direction,
            max_request_bytes=size if _is_int(size) and size > 0 else None,
            beat_bytes=row.get("beat_bytes") if _is_int(row.get("beat_bytes")) else None,
            source=f"timing_structure.memory_clients.clients.{name}.max_request_bytes",
            direction_basis="response data bits present (inbound) or absent (outbound)",
        )
    stores, reasons = _stores(rtl_facts)
    return dict(
        queues={k: {f: v for f, v in row.items() if f != "chain"} for k, row in queue_depths(rtl_facts).items()},
        memory_clients=clients,
        stores={
            space: dict(
                capacity_rows=s.capacity_rows(),
                elems_per_row=s.elems_per_row(),
                element_bits=s.element_bits,
                name=s.name,
                source="targetgen.address_space.derive_address_space(selected RTL facts)",
            )
            for space, s in stores.items()
        },
        store_reasons=reasons,
    )


# ----------------------------------------------------------------------------- declared protocol
class _View:
    def __init__(self, structure):
        from merlin.analysis.kernels.decoded_effects import _Protocol

        self.structure = structure
        self.proto = _Protocol(structure)
        self.roles = structure.get("instruction_roles") or {}

    def entries(self, stream):
        out = []
        for index, entry in enumerate(stream or ()):
            if isinstance(entry, dict):
                roles = tuple(entry.get("roles") or self.roles.get(entry.get("class")) or ())
                out.append(
                    dict(
                        seq=entry.get("seq", index),
                        klass=entry.get("class"),
                        roles=roles,
                        fields=entry.get("fields") or {},
                    )
                )
        return out

    def _same_operand(self, klass, operand, rtl_field):
        names = self.proto.fields(
            klass, lambda b: b.get("operand") == operand and b["field"] == rtl_field and b["interp"] == "int"
        )
        return names[0] if len(names) == 1 else None

    def refs(self, entry):
        """``[(space, row, rows, cols, flags)]`` per distinct local address the command names."""
        proto, klass, fields = self.proto, entry["klass"], entry["fields"]
        if _ACC_FLAG not in proto.flags:
            return None
        mask = sum(proto.flags.values())
        sentinels = {
            proto.bindings[(klass, n)].get("operand")
            for n in proto.fields(klass, lambda b: b["interp"] == "all_ones")
            if fields.get(n) is True
        }
        seen, out = set(), []
        for name in proto.fields(klass, proto.is_address):
            value = fields.get(name)
            operand = proto.bindings[(klass, name)].get("operand")
            if not _is_int(value) or value < 0 or operand in sentinels or (operand, value) in seen:
                continue
            seen.add((operand, value))
            extent = {}
            for axis in ("rows", "cols"):
                n = proto.extent_field(klass, axis) or self._same_operand(klass, operand, proto.extent[axis])
                v = fields.get(n) if n else None
                extent[axis] = v if _is_int(v) and v >= 0 else None
            flags = frozenset(f for f, bit in proto.flags.items() if value & bit)
            out.append((_ACC if _ACC_FLAG in flags else _SPAD, value & ~mask, extent["rows"], extent["cols"], flags))
        return out


def _footprint(store, row, rows, cols):
    """Local rows ``[row, row + n)`` of a row-major transfer, or ``None``."""
    if rows is None:
        return None
    per = 1
    if cols is not None and store and _is_int(store.get("elems_per_row")) and store["elems_per_row"] > 0:
        per = -(-cols // store["elems_per_row"]) if cols else 1
    elif cols is not None:
        return None
    return range(row, row + rows * per)


# ----------------------------------------------------------------------------- 1. resource occupancy
def _row_effects(view, entry, stores):
    """``(reads, writes)``: sets of ``(space, row)`` the command's declared local addresses touch.

    A readout role reads its rows; an inbound role writes them; any other role reads operand-store rows
    and writes accumulator rows. Rows past the addressed store (sentinel addresses) are left out."""
    reads, writes = set(), set()
    readout = any(r in _READOUT for r in entry["roles"])
    inbound = any(r in _INBOUND for r in entry["roles"])
    for space, row, rows, cols, _ in view.refs(entry) or ():
        store = stores.get(space)
        cap = store.get("capacity_rows") if store else None
        span = _footprint(store, row, rows, cols)
        if span is None or (_is_int(cap) and span.start >= cap):
            continue
        target = reads if readout or (not inbound and space == _SPAD) else writes
        target.update((space, r) for r in span)
    return reads, writes


def _resource_occupancy(entries, routing, bounds, view):
    from merlin.analysis.kernels.command_routing import routed_queue

    queues = (bounds or {}).get("queues") or ((routing or {}).get("queues")) or {}
    stores = (bounds or {}).get("stores") or {}
    evidence = {
        "routing": "structure#command_queue_routing",
        "depths": "structure#stream_bounds.queues",
        "ordering": "stream order; a window closes at a sync command or when a command of another "
        "queue reads local rows a command of the window wrote",
    }
    if not queues:
        return _unknown(
            "the selected RTL facts derive no timing_structure.command_classes (dispatch queue depths)", evidence
        )
    if not ((routing or {}).get("classes")):
        return _unknown(
            "no instruction-class to dispatch-queue routing derived from the selected facts: "
            + "; ".join((routing or {}).get("missing") or ["structure carries no command_queue_routing"]),
            evidence,
        )
    since, window, written = Counter(), Counter(), defaultdict(set)
    peak_sync, peak, closed = Counter(), Counter(), Counter()
    run, longest, last = Counter(), Counter(), None
    over_dispatch, over_total, routed = Counter(), Counter(), Counter()
    first_over, unrouted = {}, Counter()
    for entry in entries:
        if _SYNC in entry["roles"]:
            since.clear()
            window.clear()
            written.clear()
            last = None
            continue
        queue = routed_queue(routing, entry["klass"])
        if queue is None:
            unrouted[str(entry["klass"])] += 1
            continue
        reads, writes = _row_effects(view, entry, stores)
        for other in [q for q in written if q != queue and written[q] & reads]:
            closed[other] += 1  # a consumer on another queue depends on this window's results
            window[other] = 0
            written[other] = set()
        routed[queue] += 1
        since[queue] += 1
        window[queue] += 1
        written[queue] |= writes
        peak_sync[queue] = max(peak_sync[queue], since[queue])
        peak[queue] = max(peak[queue], window[queue])
        run[queue] = run[queue] + 1 if last == queue else 1
        longest[queue] = max(longest[queue], run[queue])
        last = queue
        depth = (queues.get(queue) or {}).get("dispatch_entries")
        extra = (queues.get(queue) or {}).get("command_queue_entries")
        if _is_int(depth) and window[queue] > depth:
            over_dispatch[queue] += 1
            first_over.setdefault(queue, entry["seq"])
        if _is_int(depth) and _is_int(extra) and window[queue] > depth + extra:
            over_total[queue] += 1
    per_queue = {}
    for queue, row in sorted(queues.items()):
        depth = row.get("dispatch_entries")
        per_queue[queue] = dict(
            commands=routed[queue],
            max_outstanding_before_dependency=peak[queue],
            max_outstanding_since_sync=peak_sync[queue],
            dependency_closures=closed[queue],
            longest_consecutive_run=longest[queue],
            dispatch_entries=depth,
            dispatch_entries_source=row.get("dispatch_entries_source"),
            command_queue_entries=row.get("command_queue_entries"),
            command_queue_source=row.get("command_queue_source"),
            points_exceeding_dispatch_entries=over_dispatch[queue] if _is_int(depth) else None,
            points_exceeding_dispatch_and_queue_entries=over_total[queue]
            if _is_int(depth) and _is_int(row.get("command_queue_entries"))
            else None,
            first_exceeding_seq=first_over.get(queue),
            **({} if _is_int(depth) else {"reason": f"{row.get('dispatch_entries_source')} not derived"}),
        )
    value = dict(per_queue=per_queue, unrouted_commands=dict(sorted(unrouted.items())))
    if not routed:
        return _fact(value, evidence, status="unknown", reason="no stream command routes to a derived queue")
    return _fact(
        value,
        evidence,
        limits=[
            "issue-order upper bound: assumes no routed command retires before its window closes (a sync, or a "
            "command of another queue reading rows the window wrote); exceeding points are where allocation would "
            "stall if nothing had retired, not measured stalls",
            "dependencies are read from declared local-address fields only; rows the array touches without naming "
            "them are not modelled",
        ],
    )


# ----------------------------------------------------------------------------- 2. transfer legality
def _interval_bytes(cols, bits_low, bits_high):
    return (cols * bits_low + 7) // 8, (cols * bits_high + 7) // 8


def _transfer_legality(entries, view, bounds):
    stores = (bounds or {}).get("stores") or {}
    clients = (bounds or {}).get("memory_clients") or {}
    evidence = {
        "bounds": "structure#stream_bounds",
        "extents": "load_state_selection.transfer_bound",
        "roles": [*_INBOUND, *_READOUT],
    }
    if not bounds:
        return _unknown("structure carries no stream_bounds from the selected RTL facts", evidence)
    request = {}
    for direction in ("inbound", "outbound"):
        sizes = {
            (c["max_request_bytes"], c["source"])
            for c in clients.values()
            if c.get("direction") == direction and _is_int(c.get("max_request_bytes"))
        }
        request[direction] = sorted(sizes)[0] if len(sizes) == 1 else None
    widths = sorted({s["element_bits"] for s in stores.values() if _is_int(s.get("element_bits"))})
    per_role = defaultdict(
        lambda: dict(
            transfers=0,
            exceeds_bound=0,
            within=0,
            unknown=0,
            max_rows=None,
            max_cols=None,
            bounds={},
            exceeding=[],
            unknown_reasons=Counter(),
        )
    )
    for entry in entries:
        roles = [r for r in entry["roles"] if r in _INBOUND or r in _READOUT]
        if not roles:
            continue
        proto = view.proto
        if not (proto.extent_field(entry["klass"], "rows") and proto.extent_field(entry["klass"], "cols")):
            continue  # no single transfer-bound extent pair: local staging, not a memory transfer
        refs = view.refs(entry)
        if not refs:
            continue
        direction = "inbound" if any(r in _INBOUND for r in roles) else "outbound"
        space, row, rows, cols, _ = refs[0]
        if rows is None or cols is None:
            continue  # a local staging command without transfer extents is not a memory transfer
        store = stores.get(space)
        verdicts, reasons = [], []
        checks = {}
        bound = request.get(direction)
        if bound is None:
            reasons.append(f"no single {direction} DMA client with a derived max_request_bytes")
        elif not (store and _is_int(store.get("element_bits")) and widths):
            reasons.append((bounds.get("store_reasons") or {}).get(space) or f"no element width for {space}")
        else:
            low, high = _interval_bytes(cols, widths[0], store["element_bits"])
            name = "row_bytes_vs_largest_dma_request"
            checks[name] = dict(
                bound=bound[0],
                source=bound[1],
                basis="one transfer row's bytes, element width between the narrowest store element "
                "and the addressed store's element",
            )
            verdicts.append(True if low > bound[0] else False if high <= bound[0] else None)
            if verdicts[-1] is None:
                reasons.append("row bytes straddle the request bound: element width not fixed by the fields")
        footprint = _footprint(store, row, rows, cols)
        if store and _is_int(store.get("capacity_rows")) and footprint is not None:
            checks["local_rows_vs_store_rows"] = dict(bound=store["capacity_rows"], source=store["source"])
            verdicts.append(footprint.stop > store["capacity_rows"])
        elif store is None:
            reasons.append((bounds.get("store_reasons") or {}).get(space) or f"no derived {space} store")
        for role in roles:
            out = per_role[role]
            out["transfers"] += 1
            out["max_rows"] = max(out["max_rows"] or 0, rows)
            out["max_cols"] = max(out["max_cols"] or 0, cols)
            out["bounds"].update(checks)
            if True in verdicts:
                out["exceeds_bound"] += 1
                if len(out["exceeding"]) < _EXAMPLES:
                    out["exceeding"].append(
                        dict(seq=entry["seq"], klass=entry["klass"], space=space, row=row, rows=rows, cols=cols)
                    )
            elif verdicts and None not in verdicts and not reasons:
                out["within"] += 1
            else:
                out["unknown"] += 1
                for why in reasons or ["no bound applies"]:
                    out["unknown_reasons"][why] += 1
    value = {
        "per_role": {
            role: dict(row, unknown_reasons=dict(row["unknown_reasons"])) for role, row in sorted(per_role.items())
        }
    }
    if not per_role:
        return _unknown("no movement or readout command with decoded rows and cols extents", evidence, value)
    if all(r["exceeds_bound"] == 0 and r["within"] == 0 for r in per_role.values()):
        return _fact(
            value,
            evidence,
            status="unknown",
            reason="; ".join(sorted({w for r in per_role.values() for w in r["unknown_reasons"]})),
        )
    return _fact(
        value,
        evidence,
        limits=[
            "exceeds_bound marks a row larger than the largest single DMA request or a footprint past the store; "
            "an observation against an RTL bound, not a simulated failure"
        ],
    )


# ----------------------------------------------------------------------------- 3. post-store reuse
def _post_store_reuse(entries, view, bounds):
    stores = (bounds or {}).get("stores") or {}
    evidence = {
        "roles": [*_READOUT, *_INBOUND],
        "flags": [_ACC_FLAG, _ACCUMULATE_FLAG],
        "extents": "load_state_selection.transfer_bound",
    }
    if _ACC_FLAG not in view.proto.flags:
        return _unknown(f"the operand protocol declares no {_ACC_FLAG} local-address flag", evidence)
    touches = []  # (index, kind, space, rows) in stream order
    unresolved = 0
    syncs = []
    for index, entry in enumerate(entries):
        if _SYNC in entry["roles"]:
            syncs.append((index, str(entry["klass"])))
            continue
        refs = view.refs(entry) or ()
        readout = any(r in _READOUT for r in entry["roles"])
        inbound = any(r in _INBOUND for r in entry["roles"])
        for space, row, rows, cols, flags in refs:
            store = stores.get(space)
            if store and _is_int(store.get("capacity_rows")) and row >= store["capacity_rows"]:
                unresolved += 1
                continue
            span = _footprint(store, row, rows, cols)
            if span is None:
                unresolved += 1
                continue
            if readout:
                kind = "read_again"
            elif space == _ACC:
                kind = "accumulated_into" if _ACCUMULATE_FLAG in flags else "rewritten"
            else:
                kind = "rewritten" if inbound else "read_again"
            touches.append((index, kind, space, span, readout))
    per_role = defaultdict(
        lambda: dict(
            stores=0,
            read_again=0,
            accumulated_into=0,
            rewritten=0,
            not_touched_again=0,
            syncs_between=Counter(),
            sync_classes=Counter(),
            examples=[],
        )
    )
    for position, (index, _, space, span, readout) in enumerate(touches):
        if not readout:
            continue
        rows = set(span)
        outcome, at = "not_touched_again", None
        for later_index, kind, later_space, later_span, _ in touches[position + 1 :]:
            if later_space == space and rows & set(later_span):
                outcome, at = kind, later_index
                break
        between = [k for i, k in syncs if index < i < (at if at is not None else len(entries))]
        for role in (r for r in entries[index]["roles"] if r in _READOUT):
            out = per_role[role]
            out["stores"] += 1
            out[outcome] += 1
            out["syncs_between"][str(len(between))] += 1
            out["sync_classes"].update(between)
            if len(out["examples"]) < _EXAMPLES and outcome != "not_touched_again":
                out["examples"].append(
                    dict(
                        store_seq=entries[index]["seq"],
                        outcome=outcome,
                        touch_seq=entries[at]["seq"],
                        space=space,
                        rows=[span.start, span.stop],
                        syncs_between=len(between),
                    )
                )
    value = dict(
        per_role={
            role: dict(row, syncs_between=dict(row["syncs_between"]), sync_classes=dict(row["sync_classes"]))
            for role, row in sorted(per_role.items())
        },
        unresolved_local_references=unresolved,
    )
    if not per_role:
        return _unknown("no readout-role command with a resolved local footprint", evidence, value)
    return _fact(
        value,
        evidence,
        limits=[
            "touches are the declared local-address fields of later commands; rows the array reads or writes "
            "without naming them in a field are not modelled",
            "accumulated_into reads the stored rows (read-modify-write); rewritten overwrites them; host waits "
            "outside the command stream are not visible",
        ],
    )


# ----------------------------------------------------------------------------- entry point
def derive(stream, structure, *, bounds=None) -> dict[str, dict]:
    """Three facts over an ordered decoded command stream; see the module docstring."""
    structure = structure or {}
    bounds = bounds if bounds is not None else structure.get("stream_bounds") or {}
    view = _View(structure)
    entries = view.entries(stream)
    return {
        "resource_occupancy": _resource_occupancy(entries, structure.get("command_queue_routing"), bounds, view),
        "transfer_legality": _transfer_legality(entries, view, bounds),
        "post_store_reuse": _post_store_reuse(entries, view, bounds),
    }
