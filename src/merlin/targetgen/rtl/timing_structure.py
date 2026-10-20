"""RTL-derived timing STRUCTURE: capacities, in-flight limits, widths and fixed delays.

:mod:`.timing` reports the feed-forward pipeline depth of each module and leaves sequenced units
(queues, FSMs, reservation tables) UNKNOWN. This reader fills the structural half of that gap
from the same elaborated HW, without assigning prices:

* ``command_classes`` -- for each declared dispatch class, the dispatch-table entries readable at
  its issue port and the command buffers its intact payload crosses on the way to its unit;
* ``memory_clients`` -- per declared memory client, request/response beat widths, the request-size
  values the size field can carry, the request-ID space and the transaction slots its response ID
  selects (the in-flight limit);
* ``fixed_latencies`` / ``array_paths`` -- register stages between declared ports, with the
  conditions (enable, storage retention, feedback) that keep a count from being a fixed latency;
* ``dataflow`` -- which modes a declared cell's select chooses between, classified structurally;
* ``buffers`` -- every multi-location buffer in the unit.

Every value is DERIVED from the selected RTL. Module, port and protocol-field names are the
target's own declarations (``rtl_extraction.timing_structure`` in its contract); a value the
method cannot reach is recorded UNKNOWN with the reason, never defaulted. Capacities are
capacities: no count here is a latency, and no latency here is a price. Variable-latency paths
(memory-system round trips, arbitration, occupancy) are measurement fits, listed as such.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .hw_design import Design, width

SCHEMA = "merlin.rtl_timing_structure.v1"
#: Timing quantities that are never structural RTL facts: they depend on the memory system,
#: contention or occupancy, so they are fitted from measurement (e.g. gSIM), not read from RTL.
FIT_FROM_MEASUREMENT = (
    "memory_round_trip_latency_cycles",
    "dma_bytes_per_cycle_sustained",
    "command_issue_and_completion_overhead_cycles",
    "address_translation_miss_latency_cycles",
    "bus_arbitration_and_contention_cycles",
    "per_command_occupancy_cycles",
)


def _sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def declarations(target: str) -> dict[str, Any] | None:
    """The target's ``rtl_extraction.timing_structure`` block, shape-checked, or None."""
    from .extraction_contract import _rtl_extraction

    block = _rtl_extraction(target).get("timing_structure")
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ValueError("timing_structure must be a mapping")
    for key in ("dispatch", "memory_clients", "memory_protocol", "latencies", "array_paths", "dataflow"):
        if key in block and not isinstance(block[key], dict):
            raise ValueError(f"timing_structure.{key} must be a mapping")
    return block


def _unknown(reason: str, **extra) -> dict[str, Any]:
    return {"status": "unknown", "value": None, "reason": reason, **extra}


def _unique(design: Design, root: str, module: str) -> str | None:
    paths = design.occurrences(root, module)
    return paths[0] if len(paths) == 1 else None


# ---------------------------------------------------------------------- buffers
def buffer_census(design: Design, root: str) -> list[dict[str, Any]]:
    """Every module under ``root`` whose outputs read multi-location storage written from its inputs."""
    rows: dict[str, dict[str, Any]] = {}
    for path, module in sorted(design.instance_paths(root).items()):
        if design.module(module).extern:
            continue
        buffered = design.buffers(module)
        locations = sorted({row["locations"] for row in buffered.values()})
        if not locations or max(locations) <= 1:
            continue
        row = rows.setdefault(
            module,
            {
                "module": module,
                "locations": locations,
                "kind": sorted({entry["kind"] for entry in buffered.values()}),
                "buffered_outputs": len(buffered),
                "occurrences": 0,
                "instances": [],
            },
        )
        row["occurrences"] += 1
        if len(row["instances"]) < 4:
            row["instances"].append(path)
    return sorted(rows.values(), key=lambda row: row["module"])


# ---------------------------------------------------------------------- command classes
def command_classes(design: Design, root: str, decl: dict[str, Any]) -> dict[str, Any]:
    module = decl.get("module")
    classes = decl.get("classes") or {}
    if not isinstance(module, str) or not isinstance(classes, dict) or not classes:
        return {"status": "unknown", "reason": "dispatch declares no module and class issue prefixes"}
    path = _unique(design, root, module)
    if path is None:
        return {"status": "unknown", "reason": f"dispatch module {module} does not occur exactly once under {root}"}
    view = design.module(module)
    buffered = design.buffers(module)
    out: dict[str, Any] = {"status": "derived", "dispatch_module": module, "dispatch_instance": path, "classes": {}}
    for name, prefix in sorted(classes.items()):
        ports = [p for p in view.outputs if p.startswith(prefix) and (width(view.output_value[p]) or 0) > 1]
        if not ports:
            out["classes"][name] = {"issue_prefix": prefix, "entries": _unknown("no multi-bit issue output")}
            continue
        counts = sorted({buffered[p]["locations"] for p in ports if p in buffered})
        entries = (
            {
                "status": "derived",
                "value": counts[0],
                "method": "distinct dispatch-table registers readable at the class issue payload outputs",
                "evidence": {
                    "instance": path,
                    "ports_counted": sum(1 for p in ports if p in buffered),
                    "written_from": sorted({w for p in ports if p in buffered for w in buffered[p]["written_from"]})[
                        :6
                    ],
                    "storage_sample": buffered[next(p for p in ports if p in buffered)]["storage_sample"],
                },
            }
            if len(counts) == 1
            else _unknown(f"issue payload outputs read differing slot counts {counts}")
        )
        walk = design.forward_buffers(root, path, ports)
        widest = max(width(view.output_value[p]) or 0 for p in ports)
        payload = sorted(p for p in ports if width(view.output_value[p]) == widest)
        queue = [hop for hop in walk["buffers"].values() if set(payload) <= set(hop["intact_from"])]
        chain = _ordered(queue, path)
        if chain and all(len(hop["locations"]) == 1 for hop in chain):
            queue_record = {
                "status": "derived",
                "entries_total": sum(hop["locations"][0] for hop in chain),
                "chain": [
                    {"instance": hop["instance"], "module": hop["module"], "entries": hop["locations"][0]}
                    for hop in chain
                ],
                "method": "buffers the intact widest issue payload crosses from the dispatch port",
                "payload_ports": payload,
            }
        else:
            queue_record = _unknown("no buffer receives the intact issue payload with one location count")
        downstream = [
            {"instance": hop["instance"], "entries": hop["locations"]}
            for hop in walk["buffers"].values()
            if hop not in queue
        ]
        out["classes"][name] = {
            "issue_prefix": prefix,
            "entries": entries,
            "command_queue": queue_record,
            "downstream_buffers": downstream,
        }
    totals = [row["entries"].get("value") for row in out["classes"].values()]
    out["total_entries"] = sum(totals) if all(isinstance(v, int) for v in totals) else None
    return out


def _ordered(hops: list[dict], start: str) -> list[dict]:
    """Hops in payload order: each reached from the dispatcher or from the previous hop."""
    remaining, order, previous = list(hops), [], {start}
    while remaining:
        step = [hop for hop in remaining if set(hop["reached_from"]) & previous]
        if not step:
            return []
        step.sort(key=lambda hop: hop["instance"])
        order.extend(step)
        previous = {hop["instance"] for hop in step}
        remaining = [hop for hop in remaining if hop not in step]
    return order


# ---------------------------------------------------------------------- memory clients
def memory_clients(design: Design, root: str, clients: dict, protocol: dict) -> dict[str, Any]:
    fields = {key: protocol.get(key) for key in ("request", "response", "id", "data", "size", "size_encoding")}
    if any(not isinstance(fields[key], str) for key in ("request", "response", "id", "data", "size")):
        return {"status": "unknown", "reason": "memory_protocol declares no complete request/response field vocabulary"}
    out: dict[str, Any] = {"status": "derived", "protocol": fields, "clients": {}}
    for name, client in sorted(clients.items()):
        module, prefix = client.get("module"), client.get("port_prefix")
        path = _unique(design, root, module) if isinstance(module, str) else None
        if path is None or not isinstance(prefix, str):
            out["clients"][name] = _unknown(f"client module {module} does not occur exactly once under {root}")
            continue
        view = design.module(module)
        request, response = prefix + fields["request"], prefix + fields["response"]

        def port_width(port: str, outputs: bool) -> int | None:
            value = (view.output_value if outputs else view.input_value).get(port)
            return width(value) if value is not None else None

        data_bits = port_width(request + fields["data"], True)
        id_bits = port_width(request + fields["id"], True)
        size_port = request + fields["size"]
        sizes = design.value_set(module, view.output_value[size_port]) if size_port in view.output_value else None
        record: dict[str, Any] = {"instance": path, "module": module}
        record["request_data_bits"] = data_bits
        # One request channel per declared client port family; it issues at most one beat per cycle.
        record["request_channels"] = sum(
            1 for port in view.outputs if port.startswith(prefix) and port.endswith(fields["request"] + fields["data"])
        )
        record["response_data_bits"] = port_width(response + fields["data"], False)
        record["beat_bytes"] = data_bits // 8 if data_bits and data_bits % 8 == 0 else None
        record["id_bits"] = id_bits
        record["id_space"] = 1 << id_bits if id_bits else None
        if sizes and fields["size_encoding"] == "log2_bytes":
            record["request_sizes_log2"] = sorted(sizes)
            record["max_request_bytes"] = 1 << max(sizes)
            record["max_request_beats"] = (1 << max(sizes)) // record["beat_bytes"] if record["beat_bytes"] else None
            record["size_basis"] = "finite value set of the request size field (upper bound on reachable sizes)"
        else:
            record["max_request_bytes"] = None
            record["size_unknown"] = "the request size field is not a finite constant selection"
        slots = (
            design.index_slots(root, path, response + fields["id"])
            if response + fields["id"] in view.input_value
            else []
        )
        counts = sorted({row["slots"] for row in slots if isinstance(row["slots"], int)})
        if len(counts) == 1:
            limit = counts[0] if record["id_space"] is None else min(counts[0], record["id_space"])
            record["in_flight"] = {
                "status": "derived",
                "value": limit,
                "tracked_slots": counts[0],
                "method": "transaction state slots selected by the response ID (indexed table, one-hot decode or "
                "per-slot comparison), bounded by the request ID space",
                "evidence": slots,
            }
        else:
            record["in_flight"] = _unknown(f"response ID selects differing or no slot counts {counts}", evidence=slots)
        out["clients"][name] = record
    return out


# ---------------------------------------------------------------------- latencies and arrays
def _pins(raw: Any) -> dict[str, dict[str, int]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict) or any(
        not isinstance(ports, dict) or any(not isinstance(v, int) for v in ports.values()) for ports in raw.values()
    ):
        raise ValueError("latency pins must map module -> {port: integer}")
    return raw


def _mode_pins(row: dict[str, Any], dataflow: dict[str, Any]) -> dict[str, dict[str, int]] | str:
    """Explicit pins plus the cell select value of a declared dataflow mode (or why it is unknown)."""
    pins = {module: dict(ports) for module, ports in _pins(row.get("pins")).items()}
    mode = row.get("mode")
    if mode is None:
        return pins
    modes = (dataflow or {}).get("modes") or {}
    values = [int(value) for value, label in modes.items() if label == mode]
    if len(values) != 1 or not dataflow.get("cell_module") or not dataflow.get("select_port"):
        return f"dataflow mode {mode} is not derived"
    pins.setdefault(dataflow["cell_module"], {})[dataflow["select_port"]] = values[0]
    return pins


def fixed_latencies(design: Design, root: str, decl: dict[str, Any], dataflow: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for name, row in sorted(decl.items()):
        module, source, sink = row.get("module"), row.get("from"), row.get("to")
        pins = _mode_pins(row, dataflow)
        if isinstance(pins, str):
            out[name] = _unknown(pins)
            continue
        try:
            view = design.module(module)
        except ValueError:
            out[name] = _unknown(f"module {module} is absent")
            continue
        if source not in view.inputs or sink not in view.outputs:
            out[name] = _unknown(f"{module} has no input {source} and output {sink}")
            continue
        stages = design.pinned(pins).port_stages(module, sink).get(source)
        if stages is None:
            out[name] = _unknown(f"no data path from {source} to {sink}")
            continue
        out[name] = {
            **stages.record(),
            "module": module,
            "from": source,
            "to": sink,
            "occurrences": len(design.occurrences(root, module)),
            "mode": row.get("mode"),
            "pins": pins,
        }
    return out


def array_paths(design: Design, root: str, decl: dict[str, Any], dataflow: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for name, row in sorted(decl.items()):
        module, mode = row.get("module"), row.get("mode")
        pins = _mode_pins(row, dataflow)
        if isinstance(pins, str):
            out[name] = _unknown(pins)
            continue
        view = design.module(module)
        sources = [p for p in view.inputs if p.startswith(row.get("from_prefix", "\0"))]
        sinks = [p for p in view.outputs if p.startswith(row.get("to_prefix", "\0"))]
        if not sources or not sinks:
            out[name] = _unknown("declared port families are empty")
            continue
        pinned = design.pinned(pins)
        matrix, statuses = [], set()
        for sink in sinks:
            stages = pinned.port_stages(module, sink)
            matrix.append(
                [stages[s].minimum if s in stages and stages[s].maximum == stages[s].minimum else None for s in sources]
            )
            statuses |= {stages[s].status for s in sources if s in stages} | (
                {"no_path"} if any(s not in stages for s in sources) else set()
            )
        flat = [v for line in matrix for v in line]
        record = {
            "module": module,
            "mode": mode,
            "pins": pins,
            "sources": len(sources),
            "sinks": len(sinks),
            "statuses": sorted(statuses),
            "min_stages": min((v for v in flat if v is not None), default=None),
            "max_stages": max((v for v in flat if v is not None), default=None),
            "stages_by_sink_then_source": matrix,
            "affine": _affine(matrix),
        }
        record["status"] = "derived" if None not in flat else "partial"
        out[name] = record
    return out


def _affine(matrix: list[list[int | None]]) -> dict[str, int] | None:
    """``stages = source*a + sink*b + c`` over declared port order, when every entry fits."""
    if not matrix or not matrix[0] or len(matrix) < 2 or len(matrix[0]) < 2 or matrix[0][0] is None:
        return None
    c = matrix[0][0]
    a = None if matrix[0][1] is None else matrix[0][1] - c
    b = None if matrix[1][0] is None else matrix[1][0] - c
    if a is None or b is None:
        return None
    ok = all(v == i * a + j * b + c for j, line in enumerate(matrix) for i, v in enumerate(line))
    return {"per_source_index": a, "per_sink_index": b, "offset": c} if ok else None


# ---------------------------------------------------------------------- dataflow
def dataflow_modes(design: Design, root: str, decl: dict[str, Any]) -> dict[str, Any]:
    cell, select = decl.get("cell_module"), decl.get("select_port")
    try:
        view = design.module(cell)
    except ValueError:
        return {"status": "unknown", "reason": f"cell module {cell} is absent"}
    if select not in view.inputs or width(view.input_value[select]) != 1:
        return {"status": "unknown", "reason": f"{cell} has no 1-bit select input {select}"}
    modes, reach = {}, {}
    for value in (0, 1):
        found = design.pinned({cell: {select: value}}).product_reach(cell)
        reach[str(value)] = found
        if found["combinational_outputs"] and not found["registers"]:
            modes[str(value)] = "weight_stationary"
        elif found["registers"] and not found["combinational_outputs"]:
            modes[str(value)] = "output_stationary"
        else:
            modes[str(value)] = "unclassified"
    occurrences = design.occurrences(root, cell)
    influenced = design.root_input_reach(root) if occurrences else set()
    driven = [design.driven_by_root_inputs(root, path, select, influenced) for path in occurrences]
    runtime = bool(occurrences) and all(driven)
    classified = {label for label in modes.values() if label != "unclassified"}
    if runtime and len(classified) == 2:
        support = "both"
    elif not any(driven) and occurrences:
        support = "fixed_select"  # constant-driven: which mode needs the select's constant value
    else:
        support = "unknown"
    return {
        "status": "derived" if support != "unknown" else "unknown",
        "cell_module": cell,
        "select_port": select,
        "modes": modes,
        "support": support,
        "select_runtime_driven": runtime,
        "cell_occurrences": len(occurrences),
        "method": "per select value: a product reaching a cell output with no register is weight-stationary; a "
        "product written only into cell registers is output-stationary; the select is runtime-driven when "
        "every cell occurrence's select is reached from a unit input",
        "evidence": reach,
    }


# ---------------------------------------------------------------------- assembly
def derive(design: Design, *, root: str, decl: dict[str, Any]) -> dict[str, Any]:
    """The complete structure record for ``root`` under the target's declarations."""
    record: dict[str, Any] = {"schema": SCHEMA, "unit_root": root}
    record["buffers"] = buffer_census(design, root)
    record["command_classes"] = (
        command_classes(design, root, decl["dispatch"]) if "dispatch" in decl else {"status": "undeclared"}
    )
    record["memory_clients"] = (
        memory_clients(design, root, decl["memory_clients"], decl.get("memory_protocol") or {})
        if "memory_clients" in decl
        else {"status": "undeclared"}
    )
    record["dataflow"] = (
        dataflow_modes(design, root, decl["dataflow"]) if "dataflow" in decl else {"status": "undeclared"}
    )
    record["fixed_latencies"] = fixed_latencies(design, root, decl.get("latencies") or {}, record["dataflow"])
    record["array_paths"] = array_paths(design, root, decl.get("array_paths") or {}, record["dataflow"])
    record["unknown"] = _unknowns(record)
    record["fit_from_measurement"] = list(FIT_FROM_MEASUREMENT)
    record["status"] = "derived" if not record["unknown"] else "partial"
    return record


def _unknowns(record: dict[str, Any]) -> list[str]:
    found = []

    def walk(node, path):
        if isinstance(node, dict):
            if node.get("status") == "unknown":
                found.append(f"{path}: {node.get('reason', 'unknown')}")
            for key, value in node.items():
                if key != "evidence":
                    walk(value, f"{path}.{key}" if path else key)

    walk({k: v for k, v in record.items() if k not in ("buffers",)}, "")
    return found


def _firtool_version(tool: dict | None) -> str | None:
    import subprocess

    if not tool or not tool.get("path"):
        return None
    try:
        result = subprocess.run([tool["path"], "--version"], capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    for line in result.stdout.splitlines():
        words = line.split()
        if words[:1] == ["CIRCT"] and len(words) >= 2:
            return words[1]
    return None


def add_to_facts(target: str, facts: dict[str, Any]) -> list[str]:
    """Attach ``timing_structure`` for the active source selection; UNKNOWN when it cannot run."""
    from .source_selection import active_selection, digest

    decl = declarations(target)
    selected = active_selection(target)
    if decl is None or selected is None:
        return []
    production = selected.get("production") or {}
    receipt = selected.get("_genericization") or {}
    generic = (receipt.get("output") or {}).get("path")
    root = decl.get("unit_root") or production.get("core_root")
    provenance = {
        "config": selected.get("config"),
        "generator": selected.get("generator"),
        "core_root": production.get("core_root"),
        "firrtl": selected["sources"]["firrtl"],
        "core_hw": selected["sources"]["core_hw"],
        "core_hw_generic": receipt.get("output"),
        "firtool": {**(production.get("tool") or {}), "version": _firtool_version(production.get("tool"))},
        "source_bundle": {"path": selected.get("selection_path"), "sha256": selected.get("selection_sha256")},
        "declarations_sha256": _sha_json(decl),
        "readers": [
            {"path": str(Path(module).resolve()), "sha256": digest(module)}
            for module in (__file__, Path(__file__).with_name("hw_design.py"))
        ],
    }
    if not generic or digest(generic) != receipt["output"]["sha256"]:
        facts["timing_structure"] = {
            "schema": SCHEMA,
            "status": "unknown",
            "reason": "no receipted generic HW serialization of the selected core",
            "provenance": provenance,
        }
        return []
    from .hw_design import load_design

    try:
        record = derive(load_design(Path(generic).read_text()), root=root, decl=decl)
    except (ValueError, KeyError) as exc:  # a declaration the RTL contradicts fails closed, visibly
        record = {"schema": SCHEMA, "status": "unknown", "reason": f"{type(exc).__name__}: {exc}"}
    record["provenance"] = provenance
    facts["timing_structure"] = record
    return [f"timing_structure({record['status']})"]


# ---------------------------------------------------------------------- profile resources
def _resolve(structure: dict[str, Any], ref: Any) -> int | None:
    """One structural integer: a dotted path, or ``{buffer_instance: path}`` (its single location count)."""
    if isinstance(ref, str):
        node: Any = structure
        for key in ref.split("."):
            node = node.get(key) if isinstance(node, dict) else None
        return node if isinstance(node, int) and not isinstance(node, bool) else None
    if isinstance(ref, dict) and set(ref) == {"buffer_instance"}:
        for row in ((structure.get("command_classes") or {}).get("classes") or {}).values():
            hops = [*((row.get("command_queue") or {}).get("chain") or ()), *(row.get("downstream_buffers") or ())]
            for hop in hops:
                if hop.get("instance") == ref["buffer_instance"]:
                    entries = hop.get("entries")
                    entries = entries if isinstance(entries, list) else [entries]
                    return entries[0] if len(entries) == 1 and isinstance(entries[0], int) else None
        return None
    raise ValueError(f"profile resource reference {ref!r} must be a dotted path or {{buffer_instance}}")


def profile_resources(structure: dict[str, Any], mapping: dict[str, Any]) -> dict[str, Any]:
    """Map structure facts onto timing-engine resources under an explicit, target-owned mapping.

    The engine's ``capacity`` is the number of concurrently occupied lanes and ``queue_capacity``
    the number of admitted reservations not yet started. ``mapping`` names, per resource, the
    structural quantity behind each (a reference, or ``{from, minus}`` of two references); ``null``
    leaves the field unmapped. Unresolvable terms stay None with the reason, and the mapping and
    its stated basis are recorded beside every value: a mapping is a reviewed semantic decision,
    never inferred here.
    """

    def term(spec):
        if spec is None:
            return None, "unmapped: no structural quantity is declared for this field"
        if isinstance(spec, dict) and "from" in spec:
            value = _resolve(structure, spec["from"])
            minus = _resolve(structure, spec["minus"]) if spec.get("minus") is not None else 0
            if value is None or minus is None:
                return None, f"{spec} is not derived"
            return (value - minus, None) if value - minus >= 0 else (None, f"{spec} is negative")
        value = _resolve(structure, spec)
        return value, None if value is not None else f"{spec} is not derived"

    out = {}
    for resource, row in sorted(mapping.items()):
        capacity, why_capacity = term(row.get("capacity"))
        queue, why_queue = term(row.get("queue_capacity"))
        out[resource] = {
            "capacity": capacity,
            "queue_capacity": queue,
            "derivation": {"capacity": row.get("capacity"), "queue_capacity": row.get("queue_capacity")},
            "unknown": {key: why for key, why in (("capacity", why_capacity), ("queue_capacity", why_queue)) if why},
            "basis": row.get("basis"),
        }
    return out


def apply_to_profile(profile: dict[str, Any], facts: dict[str, Any], mapping: dict[str, Any]) -> dict[str, Any]:
    """A copy of a timing profile whose resources carry the mapped structural values.

    Fields the mapping leaves unknown stay null, and are listed with every primitive price under
    ``structural_mapping.fit_from_measurement``: those remain to be fitted from measurements.
    """
    import copy

    structure = (facts.get("facts") or {}).get("timing_structure")
    if not isinstance(structure, dict) or structure.get("schema") != SCHEMA:
        raise ValueError("facts carry no timing_structure section of the expected schema")
    mapped = profile_resources(structure, mapping)
    out = copy.deepcopy(profile)
    resources = out.setdefault("resources", {})
    unresolved = []
    for name, row in mapped.items():
        resources.setdefault(name, {})
        resources[name]["capacity"], resources[name]["queue_capacity"] = row["capacity"], row["queue_capacity"]
        unresolved += [f"resources.{name}.{field}" for field in row["unknown"]]
    prices = ((out.get("target") or {}).get("prices")) or {}
    unpriced = [
        f"target.prices.{kind}.{field}"
        for kind, fields in sorted(prices.items())
        for field, value in sorted(fields.items())
        if value is None
    ]
    provenance = structure.get("provenance") or {}
    out["structural_mapping"] = {
        "schema": "merlin.timing_profile_structure.v1",
        "facts_schema": SCHEMA,
        "facts_config": provenance.get("config"),
        "firrtl_sha256": (provenance.get("firrtl") or {}).get("sha256"),
        "declarations_sha256": provenance.get("declarations_sha256"),
        "mapping_sha256": _sha_json(mapping),
        "resources": mapped,
        "fixed_latencies": {
            name: {key: row.get(key) for key in ("status", "min_stages", "max_stages", "module", "from", "to")}
            for name, row in (structure.get("fixed_latencies") or {}).items()
        },
        "fit_from_measurement": sorted(set(unresolved + unpriced + list(FIT_FROM_MEASUREMENT))),
        "qualification": "structural capacities only; no price is derived, and the profile stays unqualified",
    }
    limitations = out.setdefault("limitations", {})
    limitations["physical_queue_capacities"] = (
        "partially derived from RTL structure (structural_mapping.resources); unmapped fields remain unknown"
        if any(row["capacity"] is not None or row["queue_capacity"] is not None for row in mapped.values())
        else "unknown"
    )
    return out


def profile_main(argv: list[str] | None = None) -> int:
    """Write a fresh copy of a timing profile with the target's declared structural resource mapping."""
    import argparse

    parser = argparse.ArgumentParser(description=profile_main.__doc__)
    parser.add_argument("--target", required=True, help="target whose contract declares profile_resources")
    parser.add_argument("--facts", required=True, help="facts artifact carrying timing_structure")
    parser.add_argument("--profile", required=True, help="timing profile JSON to copy")
    parser.add_argument("--out", required=True, help="fresh output path")
    args = parser.parse_args(argv)
    decl = declarations(args.target) or {}
    mapping = decl.get("profile_resources")
    if not isinstance(mapping, dict) or not mapping:
        parser.error(f"{args.target} declares no timing_structure.profile_resources mapping")
    out = Path(args.out)
    if out.exists():
        parser.error(f"{out} exists; outputs are never overwritten")
    profile = json.loads(Path(args.profile).read_text())
    applied = apply_to_profile(profile, json.loads(Path(args.facts).read_text()), mapping)
    out.write_text(json.dumps(applied, indent=2, sort_keys=True) + "\n")
    for name, row in sorted(applied["structural_mapping"]["resources"].items()):
        print(f"{name}: capacity={row['capacity']} queue_capacity={row['queue_capacity']}")
    return 0
