"""Load RTL-derived structural timing facts into a device profile, by declared semantic role.

A device profile exported by a simulator repository declares its resources, hardware facts and price
classes, with unknown values left null. Structural facts extracted from the elaborated RTL (queue depths,
reservation entries, in-flight limits, transfer widths, fixed pipeline delays) arrive separately as a
closed ``merlin.timing_capacity_facts.v1`` document, each fact naming its SEMANTIC ROLE. The role, never
the field name, decides where a value may go: a reservation-station entry is a live-command credit held
from allocation to completion, which is not the generic engine's occupancy lane or start queue, so it is
recorded as structure and never copied into a resource capacity.

Every applied value keeps its sources, selector and derivation. A fact can fill only a slot the profile
declares, an already known slot must agree, and nothing here supplies or fits a price.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from . import contracts as C

SCHEMA = "merlin.timing_capacity_facts.v1"
#: role -> (unit, where the value goes). ``resource``: a profile resource field the generic engine
#: models; ``hardware``: an already declared profile hardware fact; ``structure``: recorded for target
#: providers and fitting, with no generic engine field; ``constraint``: a lower bound for fitting a price.
ROLES = {
    "occupancy_lanes": ("commands", ("resource", "capacity")),
    "start_queue_entries": ("commands", ("resource", "queue_capacity")),
    "declared_hardware_fact": (None, ("hardware", None)),
    "live_command_credits": ("commands", ("structure", "live_command_credits")),
    "command_queue_entries": ("commands", ("structure", "command_queue_entries")),
    "outstanding_requests": ("requests", ("structure", "outstanding_requests")),
    "max_transfer_bytes": ("bytes", ("structure", "max_transfer_bytes")),
    "datapath_bytes_per_beat": ("bytes", ("structure", "datapath_bytes_per_beat")),
    "minimum_latency_cycles": ("cycles", ("constraint", "minimum_latency_cycles")),
}
FACT_KEYS = {"id", "role", "applies_to", "value", "unit", "sources", "derivation"}
MAX_FACTS = 512


def _integer(value, where: str, *, positive: bool) -> int:
    if type(value) is not int or value < (1 if positive else 0) or value >= 2**63:
        raise C.ContractError(f"{where} requires a {'positive' if positive else 'nonnegative'} integer")
    return value


def facts_document(document: dict) -> dict:
    """Validate a closed capacity-facts document; return it unchanged."""
    C.closed(document, {"schema", "device_profile_sha256", "elaboration", "facts"}, "capacity facts")
    if document["schema"] != SCHEMA:
        raise C.ContractError("unsupported capacity facts schema")
    C.sha(document["device_profile_sha256"], "device profile identity")
    elaboration = document["elaboration"]
    if not isinstance(elaboration, dict) or not elaboration or len(elaboration) > 16:
        raise C.ContractError("capacity facts must name the elaborated design they were read from")
    for name, digest in elaboration.items():
        C.text(name, "elaboration artifact")
        C.sha(digest, "elaboration artifact " + name)
    facts = document["facts"]
    if not isinstance(facts, list) or not 1 <= len(facts) <= MAX_FACTS:
        raise C.ContractError(f"declare 1..{MAX_FACTS} capacity facts")
    seen, slots = set(), set()
    for fact in facts:
        C.closed(fact, FACT_KEYS, "capacity fact")
        C.text(fact["id"], "fact id")
        if fact["id"] in seen:
            raise C.ContractError("capacity fact ids must be unique")
        seen.add(fact["id"])
        if fact["role"] not in ROLES:
            raise C.ContractError(f"unknown capacity role {fact['role']!r}; roles are {sorted(ROLES)}")
        C.text(fact["applies_to"], "fact target")
        unit, (kind, _) = ROLES[fact["role"]]
        C.text(fact["unit"], "fact unit")
        if unit is not None and fact["unit"] != unit:
            raise C.ContractError(f"{fact['role']} is measured in {unit}, not {fact['unit']}")
        if kind == "hardware":
            C.number(fact["value"], "hardware fact value")
        else:
            _integer(fact["value"], f"{fact['role']} value", positive=fact["role"] != "minimum_latency_cycles")
        if (fact["role"], fact["applies_to"]) in slots:
            raise C.ContractError("one role may be stated once per target")
        slots.add((fact["role"], fact["applies_to"]))
        sources = fact["sources"]
        if not isinstance(sources, list) or not 1 <= len(sources) <= 16:
            raise C.ContractError("every capacity fact cites 1..16 pinned sources")
        for source in sources:
            C.closed(source, {"path", "sha256", "selector"}, "capacity fact source")
            C.text(source["path"], "source path")
            C.sha(source["sha256"], "source digest")
            C.text(source["selector"], "evidence selector")
        C.text(fact["derivation"], "derivation")
    return document


def apply(profile: dict, document: dict, *, profile_sha256: str) -> dict:
    """A copy of ``profile`` with every fact placed by its role, and a receipt of what was placed."""
    facts_document(document)
    if document["device_profile_sha256"] != C.sha(profile_sha256, "profile identity"):
        raise C.ContractError("capacity facts were derived against a different device profile")
    result = copy.deepcopy(profile)
    resources = result.get("resources")
    target = result.get("target")
    if not isinstance(resources, dict) or not isinstance(target, dict) or not isinstance(target.get("hardware"), dict):
        raise C.ContractError("device profile declares no resources or hardware facts")
    prices = target.get("prices") if isinstance(target.get("prices"), dict) else {}
    applied = []
    for fact in document["facts"]:
        _, (kind, field) = ROLES[fact["role"]]
        name, value = fact["applies_to"], fact["value"]
        if kind == "resource":
            slot = resources.get(name)
            if not isinstance(slot, dict) or field not in slot:
                raise C.ContractError(f"profile declares no {field} for resource {name!r}")
            container, key = slot, field
            location = f"resources.{name}.{field}"
        elif kind == "hardware":
            if name not in target["hardware"]:
                raise C.ContractError(f"profile declares no hardware fact {name!r}")
            container, key = target["hardware"], name
            location = f"target.hardware.{name}"
        elif kind == "constraint":
            if name not in prices:
                raise C.ContractError(f"profile declares no price class {name!r} to bound")
            container, key = target.setdefault("constraints", {}).setdefault(name, {}), field
            location = f"target.constraints.{name}.{field}"
        else:
            container, key = target.setdefault("structure", {}).setdefault(field, {}), name
            location = f"target.structure.{field}.{name}"
        if container.get(key) is not None and container[key] != value:
            raise C.ContractError(f"{location} is already {container[key]!r}; the fact says {value!r}")
        container[key] = value
        applied.append({"id": fact["id"], "role": fact["role"], "location": location, "value": value})
    result["capacity_facts"] = {
        "schema": SCHEMA,
        "sha256": C.digest(document),
        "device_profile_sha256": profile_sha256,
        "elaboration": dict(document["elaboration"]),
        "applied": applied,
        "scope": "structural facts placed by declared role; no price is supplied or fitted",
    }
    return result


def apply_files(profile_path: str | Path, facts_path: str | Path) -> dict:
    raw = Path(profile_path).read_bytes()
    profile = json.loads(raw)
    if not isinstance(profile, dict):
        raise C.ContractError("device profile must be a JSON object")
    return apply(profile, C.read(facts_path), profile_sha256=hashlib.sha256(raw).hexdigest())


def model_resources(profile: dict) -> list[dict]:
    """Annotated-model resources for every profile resource whose lanes and start queue are both known.

    A resource with either value unknown stays unbound -- the observer reports it unresolved -- rather
    than being given an invented default."""
    rows = []
    for name, slot in sorted((profile.get("resources") or {}).items()):
        capacity, queue = (slot or {}).get("capacity"), (slot or {}).get("queue_capacity")
        if type(capacity) is int and capacity > 0 and type(queue) is int and queue >= 0:
            rows.append({"name": name, "capacity": capacity, "queue_capacity": queue})
    return rows
