"""Invocation-owned immutable device layouts for semantic applications."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from merlin.compiler.semantic.codegen import codegen_options, current_codegen_options
from merlin.compiler.planning.cover_selection import application_plan
from merlin.compiler.partition.semantic_regions import outline_semantic_regions
from merlin.compiler.schedule.storage.residency import ResidencyRequest, ReservedInterval, plan_residency

_PUBLICATION = ContextVar("application_residency_publication", default=None)


def publication():
    return _PUBLICATION.get()


def validate_region_bindings(regions, compiled):
    """Require provider packages to consume the complete planned residency ABI."""
    files = publication()
    if files is None:
        return
    record = json.loads(files["residency.json"])
    requested = current_codegen_options().resident_bindings or {}
    if json.dumps(requested, sort_keys=True) != json.dumps(record["bindings"], sort_keys=True):
        raise ValueError("resident bindings differ from the residency table")
    consumed, seen = set(), set()
    for (_, region), package in zip(regions, compiled, strict=True):
        digest = region.digest()
        seen.add(digest)
        expected = requested.get(digest, {})
        reported = package.description.get("resident_bindings", [])
        actual = {}
        if not isinstance(reported, list):
            raise ValueError("region package resident bindings must be a list")
        for item in reported:
            if (not isinstance(item, dict) or not isinstance(item.get("node_id"), str)
                    or item["node_id"] in actual
                    or not {"address", "bytes", "immutable"} <= item.keys()):
                raise ValueError("region package has invalid or duplicate resident bindings")
            actual[item["node_id"]] = {key: item[key] for key in ("address", "bytes", "immutable")}
        # JSON equality also distinguishes Boolean values from integer addresses.
        if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise ValueError("region package did not consume its resident bindings: " + digest)
        consumed.update((node_id, item["address"], item["bytes"]) for node_id, item in actual.items())
    if set(requested) - seen:
        raise ValueError("resident bindings refer to an uncompiled region")
    entries = {row["name"]: row for row in record["entries"]}
    assets = {row["name"] for row in record["entries"] if row["role"] == "immutable_parameter"}
    if assets != {load["name"] for load in record["loads"]}:
        raise ValueError("resident table assets differ from the initial-load inventory")
    for load in record["loads"]:
        row = entries[load["name"]]
        if (load["source_node"], row["address"], row["bytes"]) not in consumed:
            raise ValueError("resident table asset is not bound by any region: " + load["name"])


def compact_runtime_parameters(source, regions, compiled, parameters):
    """Remove host backing only when every regional consumer proves residency.

    Source outputs and unused ports retain host backing. A proof from one device
    consumer does not authorize removing bytes still needed by a host consumer.
    The active allocation publication and complete region reports are rechecked
    here so callers cannot turn an unvalidated provider description into proof.
    """
    if publication() is None or parameters is None:
        return parameters
    regions, compiled = tuple(regions), tuple(compiled)
    validate_region_bindings(regions, compiled)
    original = getattr(parameters, "original", parameters)
    original.check(source)
    consumers = {key: [] for key in original.node_ids}
    for (_, region), package in zip(regions, compiled, strict=True):
        resident = {row["node_id"] for row in package.description.get("resident_bindings", [])}
        for key in set(dict(region.input_storages)) & set(consumers):
            consumers[key].append(key in resident)
    removed = tuple(key for key, uses in consumers.items() if key not in source.outputs and uses and all(uses))
    if not removed:
        return original
    from merlin.compiler.lower.c.compact_parameters import compact_parameters

    return compact_parameters(original, source, resident_only=removed)


@contextmanager
def residency_scope(source, adapter, parameters, state):
    options = current_codegen_options()
    if not options.application_residency:
        token = _PUBLICATION.set(None)
        try:
            yield
        finally:
            _PUBLICATION.reset(token)
        return
    if parameters is None:
        raise ValueError("application residency requires external, authenticated parameters")
    # Mutable state remains an explicit host interface until a device-state ABI
    # can prove successful update publication and recovery ownership.
    if state is not None:
        raise ValueError("persistent device state requires a device-state publication ABI")
    parameters.check(source)
    # Internal fresh verification may receive the compact runtime descriptor.
    # Immutable packing always reads the independently authenticated intake.
    parameters = getattr(parameters, "original", parameters)
    inventory_resident_inputs = getattr(adapter, "inventory_resident_inputs", None)
    pack_resident_input = getattr(adapter, "pack_resident_input", None)
    memory_geometry = getattr(adapter, "resident_memory_geometry", None)
    if not all(callable(hook) for hook in (inventory_resident_inputs, pack_resident_input, memory_geometry)):
        raise ValueError("target does not declare an immutable residency ABI")
    geometry = memory_geometry()
    if not isinstance(geometry, dict) or set(geometry) != {"dma_base", "dma_capacity", "alignment"}:
        raise ValueError("target residency memory geometry fields differ")
    for key, minimum in (("dma_base", 0), ("dma_capacity", 1), ("alignment", 1)):
        if type(geometry[key]) is not int or geometry[key] < minimum:
            raise ValueError("target residency memory geometry has an invalid " + key)
    base, size = options.io_memory_base, options.io_memory_bytes
    if base is None or size is None:
        raise ValueError("application residency requires explicit staging base and size")
    capacity_base, capacity_bytes = options.io_memory_capacity_base, options.io_memory_capacity_bytes
    if base < capacity_base or base + size > capacity_base + capacity_bytes:
        raise ValueError("regional staging lies outside declared residency capacity")
    entries = dict(zip(parameters.node_ids, parameters.entries, strict=True))
    plan = application_plan(source, adapter)
    outlined = outline_semantic_regions(source, entry="residency", plan=plan)
    selected = {row.id: row for row in plan.selected}
    requests, uses, layouts, region_layouts = [], [], {}, {}
    for key, region in outlined.regions:
        if selected[key].placement != "accelerator":
            continue
        inventory = inventory_resident_inputs(region, selected[key])
        parameter_inputs = set(dict(region.input_storages)) & set(entries)
        supported = {item["node_id"] for item in inventory or ()}
        if parameter_inputs - supported:
            raise ValueError("immutable device consumer lacks a resident ABI: "
                             + ", ".join(sorted(parameter_inputs - supported)))
        if inventory is None:
            continue  # The provider explicitly declines this operation's ABI.
        for item in inventory:
            node_id = item["node_id"]
            if node_id not in entries:
                continue
            identity = json.dumps([node_id, item["layout"]], sort_keys=True)
            name = "parameter_" + sha256(identity.encode()).hexdigest()
            region_key = (region.digest(), node_id)
            if region_layouts.setdefault(region_key, name) != name:
                raise ValueError("one regional parameter has incompatible resident layouts")
            if name not in layouts:
                layouts[name] = (region, item, entries[node_id], selected[key])
                requests.append(ResidencyRequest(name, "immutable_parameter", item["bytes"],
                    item["alignment"], "application", "once_at_load"))
            elif layouts[name][1] != item:
                raise ValueError("repeated resident layout has inconsistent inventory")
            uses.append((region.digest(), node_id, name))
    if not requests:
        raise ValueError("application has no supported immutable resident device operands")
    table = plan_residency(requests, base=capacity_base,
        capacity=capacity_bytes, dma_base=geometry["dma_base"], dma_capacity=geometry["dma_capacity"],
        reserved=[ReservedInterval("regional_io", base, size, options.io_memory_alignment or geometry["alignment"],
            "staging", "region invocation", "per_call")])
    allocations = {row.name: row for row in table.entries}
    bindings = {}
    for digest, node_id, name in uses:
        row = allocations[name]
        bindings.setdefault(digest, {})[node_id] = {"address": row.address, "bytes": row.bytes, "immutable": True}
    # Capture an authenticated payload privately, then pack from that snapshot.
    # Reopening the caller's archive after hashing would allow changed bytes to
    # acquire the original payload identity between validation and packing.
    from merlin.compiler.lower.c.weights import checked_payload
    files, loads = {}, []
    with TemporaryDirectory(prefix="merlin-resident-payload-") as temporary:
        payload_path = Path(temporary) / "payload.bin"
        checked = checked_payload(parameters.archive_path, archive_sha256=parameters.archive_sha256,
                                  out_blob=payload_path)
        if (checked["payload_sha256"], checked["payload_bytes"]) != (
            parameters.payload_sha256, parameters.payload_bytes,
        ):
            raise ValueError("resident source payload identity differs")
        with payload_path.open("rb") as payload:
            for name, (region, item, entry, alternative) in sorted(layouts.items()):
                payload.seek(entry.offset)
                dense = payload.read(entry.nbytes)
                if len(dense) != entry.nbytes:
                    raise ValueError("resident source parameter span is truncated")
                packed = pack_resident_input(region, item["node_id"], dense, alternative)
                if len(packed) != allocations[name].bytes:
                    raise ValueError("resident layout byte count differs from allocation")
                path = "residency/" + name + ".bin"
                files[path] = packed
                loads.append({"name": name, "source_name": entry.name, "source_node": item["node_id"],
                    "path": path, "sha256": sha256(packed).hexdigest(), "layout": item["layout"]})
    record = {**table.record(), "plan_sha256": table.digest(), "loads": loads,
        "source_request_digest": source.digest(), "bindings": bindings,
        "host_parameter_payload_sha256": parameters.payload_sha256,
        "host_parameters": parameters.record()["ports"],
        "state": "host interface; device state residency unsupported",
        "images": "provider-owned; existing runtime upload protocol",
        "scratch": "provider-owned; outside selected residency arena"}
    files["residency.json"] = (json.dumps(record, sort_keys=True, indent=2) + "\n").encode()
    token = _PUBLICATION.set(files)
    try:
        with codegen_options(replace(options, resident_bindings=bindings)):
            yield
    finally:
        _PUBLICATION.reset(token)
