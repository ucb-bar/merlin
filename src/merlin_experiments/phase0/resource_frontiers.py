"""Byte-demand frontiers for explicitly declared banks, segments and caches.

Unlike physical resident-store sizing, these declarations do not imply a row
layout. Every selected capacity and reservation is an explicit byte-count fact;
element widths come from the shared format registry. No execution/cache claim
follows from a generated size inequality.
"""

from __future__ import annotations

import copy
import hashlib
import json

from .resource_boundaries import BoundaryUnavailable, _path
from .resource_boundaries import validate as validate_allocations

KIND = "declared_resource_boundary"


def derive(spec, *, owner, axis, target, tile, dtype, fixed, evidence, resolve_extent):
    from merlin.targetgen.capsule_dram import dtype_bits

    allowed = {"derive", "resource", "capacity_fact", "reservation_facts", "quantum", "tail_offsets", "allocations"}
    if not isinstance(spec, dict) or set(spec) - allowed or spec.get("derive") != KIND:
        raise ValueError(f"{owner}: unsupported declared resource boundary")
    resource = spec.get("resource")
    if not isinstance(resource, str) or not resource.strip():
        raise ValueError(f"{owner}: select one explicit resource from the reviewed facts")
    # The allocation language is shared with resident-store boundaries; only the
    # physical row/store proof differs. It validates without reading any facts.
    validate_allocations(
        {key: value for key, value in spec.items() if key != "resource"} | {"store": resource}, owner=owner
    )
    quantum = resolve_extent(spec["quantum"], tile)
    record = {
        "derive": KIND,
        "axis": axis,
        "resource": resource,
        "target": target,
        "declaration": copy.deepcopy(spec),
        "quantum": quantum,
        "missing_scenarios": [],
        "scope": "declared byte demand; resource placement, cache state and execution unverified",
    }
    if evidence is None or evidence.target != target:
        raise BoundaryUnavailable("selected refreshed evidence unavailable", record)
    refreshed = evidence.refreshed_facts
    body = refreshed.get("facts") if isinstance(refreshed, dict) else None
    if not isinstance(body, dict):
        raise BoundaryUnavailable("selected refreshed facts body unavailable", record)
    record["raw_facts_sha256"] = evidence.raw_facts_sha256
    record["facts_sha256"] = hashlib.sha256(
        json.dumps(refreshed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    def fact(path, positive):
        _path(path, owner)
        value = body
        for part in path:
            if isinstance(part, str) and isinstance(value, dict) and part in value:
                value = value[part]
            elif type(part) is int and isinstance(value, list) and part < len(value):
                value = value[part]
            else:
                raise BoundaryUnavailable(f"selected resource byte-count fact missing: {path!r}", record)
        if type(value) is not int or value < int(positive):
            raise BoundaryUnavailable(f"selected resource byte-count fact malformed: {path!r}", record)
        return value

    capacity = fact(spec["capacity_fact"], True)
    reservations = [{"path": path, "bytes": fact(path, False)} for path in spec.get("reservation_facts", [])]
    reserved = sum(row["bytes"] for row in reservations)
    record.update(capacity_bytes=capacity, capacity_fact=spec["capacity_fact"], reservations=reservations)
    if not any(axis in row["shape"] for row in spec["allocations"]):
        raise ValueError(f"{owner}: allocation must depend on the selected frontier axis")

    def dimension(token, extent):
        if token == axis:
            return extent
        if isinstance(token, str) and token in fixed:
            values = set(fixed[token])
            if len(values) != 1 or type(next(iter(values))) is not int:
                raise ValueError(f"{owner}: frontier needs one fixed geometry for {token!r}")
            return next(iter(values))
        return resolve_extent(token, tile)

    def point(extent):
        allocations = []
        for row in spec["allocations"]:
            shape = [dimension(token, extent) for token in row["shape"]]
            allocated = [
                ((value + quantum - 1) // quantum) * quantum if i in row.get("round_up", []) else value
                for i, value in enumerate(shape)
            ]
            selected = dtype if row["dtype"] == "operand" else row["dtype"]
            try:
                bits = dtype_bits(selected)
            except (KeyError, ValueError):
                raise BoundaryUnavailable(
                    "selected resource allocation format has unknown storage width", record
                ) from None
            if bits is None:
                raise BoundaryUnavailable("selected resource allocation format has unknown storage width", record)
            elements = 1
            for value in allocated:
                elements *= value
            per_copy = (elements * bits + 7) // 8
            allocations.append(
                {
                    "name": row["name"],
                    "shape": shape,
                    "allocated_shape": allocated,
                    "dtype": selected,
                    "copies": row.get("copies", 1),
                    "bytes_per_copy": per_copy,
                    "bytes": per_copy * row.get("copies", 1),
                }
            )
        total = reserved + sum(row["bytes"] for row in allocations)
        return {
            "extent": extent,
            "allocations": allocations,
            "reserved_bytes": reserved,
            "total_bytes": total,
            "capacity_bytes": capacity,
            "fits": total <= capacity,
            "inequality": {"lhs": total, "relation": "<=" if total <= capacity else ">", "rhs": capacity},
        }

    if not point(quantum)["fits"]:
        record.update(first_overflow=point(quantum), missing_scenarios=["below", "last_fitting"])
        raise BoundaryUnavailable("no aligned point fits declared resource demand", record)
    low, high = 1, 2
    while point(high * quantum)["fits"]:
        low, high = high, high * 2
        if high > 2**63:
            raise BoundaryUnavailable("resource frontier exceeds integer search bound", record)
    while low + 1 < high:
        middle = (low + high) // 2
        if point(middle * quantum)["fits"]:
            low = middle
        else:
            high = middle
    roles = {}

    def add(extent, role):
        if extent > 0:
            roles.setdefault(extent, []).append(role)
        else:
            record["missing_scenarios"].append(role)

    add((low - 1) * quantum, "below")
    add(low * quantum, "last_fitting")
    add(high * quantum, "first_overflow")
    for anchor, name in ((low, "last_fitting"), (high, "first_overflow")):
        for offset in spec["tail_offsets"]:
            role = f"{name}_tail_{offset}"
            if abs(offset) >= quantum:
                record["missing_scenarios"].append(role)
            else:
                add(anchor * quantum + offset, role)
    record.update(
        status="derived", points=[dict(point(extent), roles=names) for extent, names in sorted(roles.items())]
    )
    return sorted(roles), {extent: "_and_".join(names) for extent, names in roles.items()}, record
