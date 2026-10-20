"""Size obligations around a declared simultaneous footprint in one selected store.

This derives test extents, not a placement plan or a profitable schedule. Facts
come only from the caller's refreshed evidence; footprints use the shared
address-space owner. Allocation concurrency and dimension padding are explicit
external declarations, never inferred from an application or target name.
"""

from __future__ import annotations

import copy
import hashlib
import json
from math import prod

KIND = "resident_allocation_boundary"


class BoundaryUnavailable(ValueError):
    """A well-formed declaration has an unestablished size obligation."""

    def __init__(self, detail: str, record: dict):
        super().__init__(detail)
        self.record = {**record, "status": "unknown", "missing": [detail]}


def _path(value, owner: str) -> None:
    if (
        not isinstance(value, list)
        or not value
        or any(not ((type(part) is int and part >= 0) or (isinstance(part, str) and part)) for part in value)
    ):
        raise ValueError(f"{owner}: fact paths must be nonempty lists of names/nonnegative integer indices")


def validate(spec: dict, *, owner: str) -> None:
    """Check authoring syntax without discovering a target or reading facts."""
    allowed = {
        "derive",
        "store",
        "capacity_fact",
        "reservation_facts",
        "quantum",
        "tail_offsets",
        "allocations",
        "include_exact_capacity",
        "logical_payload_only",
    }
    if set(spec) - allowed:
        raise ValueError(f"{owner}: unknown boundary fields {sorted(set(spec) - allowed)}")
    if "include_exact_capacity" in spec and type(spec["include_exact_capacity"]) is not bool:
        raise ValueError(f"{owner}: include_exact_capacity must be Boolean")
    if "logical_payload_only" in spec and type(spec["logical_payload_only"]) is not bool:
        raise ValueError(f"{owner}: logical_payload_only must be Boolean")
    if not isinstance(spec.get("store"), str) or not spec["store"]:
        raise ValueError(f"{owner}: boundary needs one explicit physical store name")
    _path(spec.get("capacity_fact"), owner)
    reservations = spec.get("reservation_facts", [])
    if not isinstance(reservations, list):
        raise ValueError(f"{owner}: reservation_facts must be a list of byte-count fact paths")
    for path in reservations:
        _path(path, owner)
    if len({tuple(path) for path in reservations}) != len(reservations):
        raise ValueError(f"{owner}: reservation fact paths must be distinct")
    quantum = spec.get("quantum")
    if not ((type(quantum) is int and quantum > 0) or (isinstance(quantum, str) and quantum)):
        raise ValueError(f"{owner}: quantum must use the existing positive extent/tile grammar")
    offsets = spec.get("tail_offsets")
    if not isinstance(offsets, list) or any(type(x) is not int or x == 0 for x in offsets):
        raise ValueError(f"{owner}: tail_offsets must explicitly declare nonzero integer offsets")
    if len(set(offsets)) != len(offsets):
        raise ValueError(f"{owner}: tail offsets must be distinct")
    allocations = spec.get("allocations")
    if not isinstance(allocations, list) or not allocations:
        raise ValueError(f"{owner}: declare every simultaneous allocation in the selected store")
    names = []
    for row in allocations:
        if not isinstance(row, dict) or set(row) - {
            "name",
            "shape",
            "dtype",
            "copies",
            "round_up",
            "logical_element_bits",
        }:
            raise ValueError(f"{owner}: allocation must declare name, shape, dtype and optional copies/round_up")
        if "logical_element_bits" in row and (
            spec.get("logical_payload_only") is not True
            or type(row["logical_element_bits"]) is not int
            or not 1 <= row["logical_element_bits"] <= 64
        ):
            raise ValueError(f"{owner}: logical element bits require an explicit logical-only payload boundary")
        if not isinstance(row.get("name"), str) or not row["name"]:
            raise ValueError(f"{owner}: allocation needs a name")
        names.append(row["name"])
        if not isinstance(row.get("dtype"), str) or not row["dtype"]:
            raise ValueError(f"{owner}: allocation needs a dtype or 'operand'")
        shape = row.get("shape")
        if (
            not isinstance(shape, list)
            or (not shape and spec.get("logical_payload_only") is not True)
            or any(not ((type(x) is int and x > 0) or (isinstance(x, str) and x)) for x in shape)
        ):
            raise ValueError(f"{owner}: allocation shape must have positive extent tokens/axis names")
        if type(row.get("copies", 1)) is not int or row.get("copies", 1) < 1:
            raise ValueError(f"{owner}: copies must be a positive integer")
        rounded = row.get("round_up", [])
        if not isinstance(rounded, list) or any(type(i) is not int or not 0 <= i < len(shape) for i in rounded):
            raise ValueError(f"{owner}: round_up must name dimension indices in the allocation shape")
        if len(set(rounded)) != len(rounded):
            raise ValueError(f"{owner}: rounded dimension indices must be distinct")
    if len(set(names)) != len(names):
        raise ValueError(f"{owner}: allocation names must be distinct")


def derive(spec, *, owner, axis, target, tile, dtype, fixed, evidence, resolve_extent):
    """Return extents, labels and inequalities through the normal sweep resolver."""
    from merlin.targetgen import address_space as AS

    validate(spec, owner=owner)
    quantum = resolve_extent(spec["quantum"], tile)
    if quantum < 1:
        raise ValueError(f"{owner}: quantum must resolve to a positive extent")
    if not any(axis in row["shape"] for row in spec["allocations"]):
        raise ValueError(f"{owner}: at least one resident allocation must depend on axis {axis!r}")
    # Each fixed axis must mean one geometry. A second independent sweep cannot
    # borrow the boundary of the first; author separate declarations instead.
    dimensions = {}
    for row in spec["allocations"]:
        for token in row["shape"]:
            if token == axis:
                continue
            if isinstance(token, str) and token in fixed:
                values = set(fixed[token])
                if len(values) != 1 or next(iter(values)) < 1:
                    raise ValueError(f"{owner}: boundary allocation needs single positive fixed axis {token!r}")
                dimensions[token] = next(iter(values))
            else:
                value = resolve_extent(token, tile)
                if value < 1:
                    raise ValueError(f"{owner}: resident allocation dimensions must be positive")
                dimensions[token] = value
    record = {
        "derive": KIND,
        "axis": axis,
        "target": target,
        "store": spec["store"],
        "quantum": quantum,
        "declaration": copy.deepcopy(spec),
        "missing_scenarios": [],
        "unavailable_tails": [],
        "tail_status": "requested" if spec["tail_offsets"] else "not_declared",
        "scope": "declared simultaneous size in one physical store; no lifetime or schedule proof",
    }
    if evidence is None or getattr(evidence, "target", None) != target:
        raise BoundaryUnavailable("explicit refreshed selected target evidence is unavailable", record)
    facts = getattr(evidence, "refreshed_facts", None)
    body = facts.get("facts") if isinstance(facts, dict) else None
    record["facts_sha256"] = hashlib.sha256(
        json.dumps(facts, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    record["raw_facts_sha256"] = getattr(evidence, "raw_facts_sha256", None)
    if not isinstance(body, dict):
        raise BoundaryUnavailable("selected refreshed facts have no facts mapping", record)

    def fact(path, *, positive):
        value = body
        for part in path:
            if isinstance(part, str) and isinstance(value, dict) and part in value:
                value = value[part]
            elif type(part) is int and isinstance(value, list) and part < len(value):
                value = value[part]
            else:
                raise BoundaryUnavailable(f"missing selected byte-count fact {path!r}", record)
        if type(value) is not int or value < int(positive):
            raise BoundaryUnavailable(
                f"selected byte-count fact {path!r} is not a {'positive' if positive else 'nonnegative'} integer",
                record,
            )
        return value

    capacity_bytes = fact(spec["capacity_fact"], positive=True)
    memories = body.get("memories", [])
    if not isinstance(memories, list):
        raise BoundaryUnavailable("selected facts have no physical memory list", record)
    matching = [i for i, m in enumerate(memories) if isinstance(m, dict) and m.get("name") == spec["store"]]
    if len(matching) != 1:
        raise BoundaryUnavailable("selected physical store is missing or ambiguous", record)
    if spec["capacity_fact"] != ["memories", matching[0], "bytes"]:
        raise ValueError(f"{owner}: capacity_fact must identify the selected store's actual byte capacity")
    space = AS.derive_address_space(target, facts=facts)
    stores = [store for store in space.stores if store.name == spec["store"]]
    if len(stores) != 1 or not stores[0].row_bytes or not stores[0].total_rows:
        record["address_space_unknowns"] = [unknown.to_dict() for unknown in space.unknowns]
        raise BoundaryUnavailable("selected store has no exact physical row width/capacity", record)
    store = stores[0]
    if capacity_bytes != store.nbytes:
        raise BoundaryUnavailable("selected capacity fact disagrees with derived physical store", record)
    reservations = [{"fact": path, "bytes": fact(path, positive=False)} for path in spec.get("reservation_facts", [])]
    if any(row["bytes"] % store.row_bytes for row in reservations):
        raise BoundaryUnavailable("reservation byte count is not an exact physical row multiple", record)
    reserved = sum(row["bytes"] // store.row_bytes for row in reservations)
    record.update(
        capacity_fact={"path": spec["capacity_fact"], "bytes": capacity_bytes},
        reservation_facts=reservations,
        capacity_rows=store.total_rows,
        reserved_rows=reserved,
        physical_store=store.to_dict(),
    )

    def point(extent):
        allocations = []
        for row in spec["allocations"]:
            actual = [extent if token == axis else dimensions[token] for token in row["shape"]]
            shape = [
                ((value + quantum - 1) // quantum) * quantum if i in row.get("round_up", []) else value
                for i, value in enumerate(actual)
            ]
            selected_dtype = dtype if row["dtype"] == "operand" else row["dtype"]
            if spec.get("logical_payload_only") is True:
                bits = row.get("logical_element_bits")
                if type(bits) is not int:
                    raise BoundaryUnavailable("logical payload boundary lacks every source element width", record)
                payload = prod(shape) * ((bits + 7) // 8)
                rows = (payload + store.row_bytes - 1) // store.row_bytes
            else:
                rows = store.working_set_rows(shape, selected_dtype)
            if selected_dtype is None or rows is None:
                raise BoundaryUnavailable(f"allocation {row['name']!r} has an unmeasurable element/row width", record)
            copies = row.get("copies", 1)
            allocations.append(
                {
                    "name": row["name"],
                    "shape": actual,
                    "allocated_shape": shape,
                    "dtype": selected_dtype,
                    "copies": copies,
                    "rows_per_copy": rows,
                    "rows": rows * copies,
                }
            )
        resident = sum(row["rows"] for row in allocations)
        total = reserved + resident
        fits = total <= store.total_rows
        return {
            "extent": extent,
            "allocations": allocations,
            "resident_rows": resident,
            "reserved_rows": reserved,
            "total_rows": total,
            "capacity_rows": store.total_rows,
            "fits": fits,
            "inequality": {"lhs": total, "relation": "<=" if fits else ">", "rhs": store.total_rows},
        }

    if not point(quantum)["fits"]:
        record.update(first_overflow=point(quantum), missing_scenarios=["below", "last_fitting"])
        raise BoundaryUnavailable("no positive aligned extent fits the declared aggregate footprint", record)
    low, high = 1, 2
    while point(high * quantum)["fits"]:
        low, high = high, high * 2
        if high > 2**63:
            raise BoundaryUnavailable("no finite resource boundary found within integer search bound", record)
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
    if spec.get("include_exact_capacity"):
        left, right = low * quantum, high * quantum
        while left < right:
            middle = (left + right) // 2
            if point(middle)["total_rows"] < store.total_rows:
                left = middle + 1
            else:
                right = middle
        exact = point(left)
        record["exact_capacity"] = {
            "status": "reachable" if exact["total_rows"] == store.total_rows else "unreachable",
            "first_at_or_above": exact,
            "previous": point(left - 1) if left > 1 else None,
        }
        if exact["total_rows"] == store.total_rows:
            add(left, "exact_capacity")
    for anchor, name in ((low, "last_fitting"), (high, "first_overflow")):
        for offset in spec["tail_offsets"]:
            direction = "minus" if offset < 0 else "plus"
            role = f"{name}_tail_{direction}_{abs(offset)}"
            if abs(offset) >= quantum:
                record["missing_scenarios"].append(role)
                record["unavailable_tails"].append(
                    {
                        "role": role,
                        "offset": offset,
                        "reason": "offset is not inside the selected quantum",
                        "quantum": quantum,
                    }
                )
            else:
                add(anchor * quantum + offset, role)
    points = [dict(point(extent), roles=names) for extent, names in sorted(roles.items())]
    record.update(status="derived", points=points)
    return list(sorted(roles)), {extent: "_and_".join(names) for extent, names in roles.items()}, record
