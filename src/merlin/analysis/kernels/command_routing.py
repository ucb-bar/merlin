"""Which dispatch queue each accelerator instruction class enters, derived from the selected RTL.

``command_queue_routing(rtl_facts, structure, hierarchy=None)`` joins three selected records, all
read as data:

1. the operand protocol (``structure["operand_protocol"]``): the register bundles each instruction
   class's fields bind;
2. the operand-protocol layouts (``structure["field_layouts"]``) and field-kind sources
   (``structure["operand_field_kind_sources"]``): the FIRRTL module that consumes each bundle, with
   source lines;
3. the RTL facts' ``timing_structure.command_classes``: per dispatch class, the dispatch-table
   entries and the command-queue chain (instance paths) the issue payload crosses; and the
   elaborated instance hierarchy the facts pin (``source.hierarchy_path``/``hierarchy_sha256``),
   which names the module of every instance on that chain.

A class routes to a dispatch queue when every module consuming its bundles is the module of an
ancestor instance of that queue's command-queue chain, and exactly one queue qualifies. A class
whose fields bind no bundle, whose bundles name no consuming module, or that matches zero or several
queues stays ``unknown`` with the reason. Nothing selects on a class, module or instance spelling.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "merlin.command_queue_routing.v1"
_SOURCE = "timing_structure.command_classes"


def _body(rtl_facts):
    if not isinstance(rtl_facts, dict):
        return {}
    inner = rtl_facts.get("facts")
    return inner if isinstance(inner, dict) else rtl_facts


def _finite(record):
    value = (record or {}).get("value")
    return value if type(value) is int and value > 0 and (record or {}).get("status") == "derived" else None


def queue_depths(rtl_facts) -> dict[str, Any]:
    """Per dispatch class: dispatch-table entries and command-queue entries, each with its source."""
    classes = ((_body(rtl_facts).get("timing_structure") or {}).get("command_classes") or {}).get("classes") or {}
    out = {}
    for name, spec in sorted(classes.items()):
        queue = (spec or {}).get("command_queue") or {}
        total = queue.get("entries_total")
        out[name] = dict(
            dispatch_entries=_finite((spec or {}).get("entries")),
            dispatch_entries_source=f"{_SOURCE}.classes.{name}.entries",
            command_queue_entries=total
            if type(total) is int and total > 0 and queue.get("status") == "derived"
            else None,
            command_queue_source=f"{_SOURCE}.classes.{name}.command_queue.entries_total",
            chain=[
                dict(instance=row.get("instance"), module=row.get("module"))
                for row in queue.get("chain") or ()
                if isinstance(row, dict)
            ],
        )
    return out


def load_hierarchy(rtl_facts):
    """``(tree, reason)``: the instance hierarchy the facts pin, refusing changed bytes."""
    source = _body(rtl_facts).get("source") or {}
    path, expected = source.get("hierarchy_path"), source.get("hierarchy_sha256")
    if not path or not expected:
        return None, "the RTL facts pin no source.hierarchy_path/hierarchy_sha256"
    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        return None, f"pinned instance hierarchy unreadable: {type(error).__name__}"
    if hashlib.sha256(raw).hexdigest() != expected:
        return None, "pinned instance hierarchy differs from source.hierarchy_sha256"
    return json.loads(raw), None


def _unit(tree, root):
    """The first instance node (depth-first) whose module is the facts' unit root."""
    stack = [tree] if isinstance(tree, dict) else []
    while stack:
        node = stack.pop()
        if node.get("module_name") == root:
            return node
        stack.extend(reversed([c for c in node.get("instances") or () if isinstance(c, dict)]))
    return None


def _ancestor_modules(unit, path):
    """Modules of every instance from the unit root down to (and including) ``path``, or ``None``."""
    parts = [p for p in str(path or "").split("/") if p]
    # Facts instance paths start at the unit root, spelled by its module name.
    if not parts or unit is None or parts[0] != unit.get("module_name"):
        return None
    node, out = unit, [(parts[0], unit.get("module_name"))]
    for depth, name in enumerate(parts[1:], start=1):
        node = next((c for c in node.get("instances") or () if c.get("instance_name") == name), None)
        if node is None:
            return None
        out.append(("/".join(parts[: depth + 1]), node.get("module_name")))
    return out


def _consumers(structure):
    """``{bundle: {module: [source citations]}}`` from the protocol layouts and field-kind sources."""
    out: dict[str, dict[str, list[str]]] = {}
    for row in structure.get("field_layouts") or ():
        bundle, module = row.get("bundle"), row.get("module")
        if bundle and module:
            lines = row.get("source_lines") or []
            out.setdefault(bundle, {}).setdefault(module, []).append(
                f"field_layouts#{bundle}.{row.get('name')}" + (f"@firrtl:{lines[0]}" if lines else "")
            )
    for row in structure.get("operand_field_kind_sources") or ():
        bundle, module = row.get("bundle"), row.get("module")
        if bundle and module:
            lines = row.get("firrtl_lines") or []
            out.setdefault(bundle, {}).setdefault(module, []).append(
                f"operand_field_kind_sources#{row.get('label')}" + (f"@firrtl:{lines[0]}" if lines else "")
            )
    return out


def command_queue_routing(rtl_facts, structure, *, hierarchy=None) -> dict[str, Any]:
    """``{status, classes: {class: {status, queue, ...}}, queues, missing}``; see the module docstring."""
    structure = structure or {}
    body = _body(rtl_facts)
    depths = queue_depths(rtl_facts)
    missing = []
    if not depths:
        missing.append(f"{_SOURCE}.classes not derived in the selected RTL facts")
    reason = None
    if hierarchy is None and depths:
        hierarchy, reason = load_hierarchy(rtl_facts)
    root = (body.get("timing_structure") or {}).get("unit_root") or (body.get("census") or {}).get("unit_root")
    unit = _unit(hierarchy, root) if hierarchy is not None and root else None
    if depths and unit is None:
        missing.append(reason or f"instance hierarchy has no node of the unit root module {root!r}")
    queue_modules: dict[str, dict[str, str]] = {}  # queue -> {module: controller instance}
    for name, row in depths.items():
        modules = {}
        for link in row["chain"]:
            for instance, module in (_ancestor_modules(unit, link["instance"]) or [])[:-1]:
                if module and module != root:
                    modules.setdefault(module, instance)
        queue_modules[name] = modules
        row["controller_modules"] = dict(sorted(modules.items()))
    consumers = _consumers(structure)
    classes = {}
    protocol_classes = ((structure.get("operand_protocol") or {}).get("roles") or {}).get("classes") or {}
    for klass, spec in sorted(protocol_classes.items()):
        bundles = sorted({f.get("bundle") for f in (spec or {}).get("fields") or () if f.get("bundle")})
        modules = {m: cites for b in bundles for m, cites in (consumers.get(b) or {}).items()}
        record = dict(
            bundles=bundles,
            consuming_modules=sorted(modules),
            evidence=sorted({c for cites in modules.values() for c in cites})[:8],
        )
        if not bundles:
            classes[klass] = dict(
                record, status="unknown", queue=None, reason="no protocol field of the class binds a register bundle"
            )
            continue
        if not modules:
            classes[klass] = dict(
                record,
                status="unknown",
                queue=None,
                reason="no layout or field-kind source names the module consuming its bundles",
            )
            continue
        hits = [q for q, mods in queue_modules.items() if set(modules) <= set(mods)]
        if len(hits) != 1:
            classes[klass] = dict(
                record,
                status="unknown",
                queue=None,
                reason=(
                    f"consuming modules {sorted(modules)} sit on no dispatch queue's command-queue chain"
                    if not hits
                    else f"consuming modules match several dispatch queues {hits}"
                ),
            )
            continue
        queue = hits[0]
        classes[klass] = dict(
            record,
            status="derived",
            queue=queue,
            controller_instances=sorted({queue_modules[queue][m] for m in modules}),
            queue_source=f"{_SOURCE}.classes.{queue}.command_queue.chain",
        )
    derived = sum(1 for row in classes.values() if row["status"] == "derived")
    return dict(
        schema=SCHEMA,
        status="derived" if derived and not missing else "partial" if derived else "unknown",
        classes=classes,
        queues={name: {k: v for k, v in row.items() if k != "chain"} for name, row in depths.items()},
        missing=missing,
        basis="protocol bundle -> consuming FIRRTL module (layouts / field-kind sources) -> ancestor instance of "
        "a dispatch class's command-queue chain (pinned instance hierarchy)",
        limits=[
            "classes absent from the operand protocol (fences, loop descriptors) are not routed here",
            "routing names the queue a class's operands are consumed behind; it is not a timing model",
        ],
    )


def routed_queue(routing, klass):
    """The derived dispatch queue of ``klass``, or ``None``."""
    row = ((routing or {}).get("classes") or {}).get(klass) or {}
    return row.get("queue") if row.get("status") == "derived" else None
