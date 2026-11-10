"""Derive bounded iteration-workload variants from selected hardware facts.

The generated directories are capture inputs, not generated capsules or model
validation. Their copied loader and profile bytes are inventoried by the
ordinary sealed capture selector before execution.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import yaml

SCHEMA = "merlin.iteration_variant_template.v1"
PROFILE_SCHEMA = "merlin.iteration_workload_profile.v1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(document: dict) -> bytes:
    return (json.dumps(document, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _ordinary(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"selected input must be an ordinary file: {path}")
    return path.read_bytes()


def _unique(rows: object, name: str) -> dict:
    if not isinstance(rows, list):
        raise ValueError("selected facts lack a resource roster")
    found = [row for row in rows if isinstance(row, dict) and row.get("name") == name]
    if len(found) != 1:
        raise ValueError(f"selected facts need exactly one resource named {name!r}")
    return found[0]


def _positive(value: object, label: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{label} must be an integer in 1..{maximum}")
    return value


def _simple_name(value: object) -> bool:
    """A portable ASCII identifier for emitted workload paths and profile keys."""
    return (
        isinstance(value, str)
        and bool(value)
        and "a" <= value[0] <= "z"
        and all("a" <= char <= "z" or "0" <= char <= "9" or char == "_" for char in value[1:])
    )


def derive(facts_bytes: bytes, template_bytes: bytes) -> dict:
    """Compute cases on both sides of a selected logical capacity law.

    This is an address-space lower bound, not a model of allocation, fusion,
    usable row capacity, or measured runtime. In the concurrent case, the
    authored live roles still require confirmation in the selected capture.
    """
    facts = json.loads(facts_bytes)
    template = yaml.safe_load(template_bytes)
    if not isinstance(facts, dict) or not isinstance(template, dict):
        raise ValueError("facts and template must be mappings")
    if template.get("schema") != SCHEMA or set(template) != {
        "schema",
        "target",
        "workload_id",
        "profile_constants",
        "boundary",
    }:
        raise ValueError("unsupported iteration-variant template")
    if facts.get("source_consistency", {}).get("status") != "verified":
        raise ValueError("iteration variants require source-consistent selected RTL facts")
    observed = facts.get("facts") or {}
    if observed.get("target") != template["target"]:
        raise ValueError("variant template target differs from selected facts")
    workload_id = template["workload_id"]
    if not _simple_name(workload_id):
        raise ValueError("invalid iteration workload id")
    boundary = template["boundary"]
    if not isinstance(boundary, dict) or boundary.get("kind") not in {
        "square_nchw_single_tensor_capacity",
        "square_nchw_concurrent_tensor_capacity",
    }:
        raise ValueError("unsupported iteration capacity boundary")
    concurrent = boundary["kind"] == "square_nchw_concurrent_tensor_capacity"
    required_keys = {"kind", "memory", "array", "array_axis", "channels_per_array_lane"}
    if concurrent:
        required_keys.add("live_tensor_roles")
    if set(boundary) != required_keys:
        raise ValueError("unsupported iteration capacity boundary")
    roles = boundary.get("live_tensor_roles", ["single_feature"])
    if concurrent and (
        not isinstance(roles, list)
        or not 2 <= len(roles) <= 4
        or any(not _simple_name(role) for role in roles)
        or len(set(roles)) != len(roles)
    ):
        raise ValueError("concurrent capacity boundary needs distinct named live tensors")
    array = _unique(observed.get("arrays"), boundary["array"])
    memory = _unique(observed.get("memories"), boundary["memory"])
    if boundary["array_axis"] not in ("rows", "cols"):
        raise ValueError("array axis must be rows or cols")
    lane_count = _positive(array.get(boundary["array_axis"]), "array lane count", 256)
    channels = lane_count * _positive(boundary["channels_per_array_lane"], "channel multiplier", 16)
    _positive(channels, "channels", 256)
    bits = _positive(memory.get("elem_bits"), "memory element bits", 64)
    if bits % 8:
        raise ValueError("sub-byte element packing has no declared tensor-footprint rule")
    capacity = _positive(memory.get("bytes"), "memory address-space bytes", 1 << 40)
    constants = template["profile_constants"]
    if not isinstance(constants, dict) or "blocks" not in constants or not constants:
        raise ValueError("profile constants must declare model depth")
    for key, value in constants.items():
        if not _simple_name(key):
            raise ValueError("profile constants need simple names")
        if key in {"schema", "workload_id", "channels", "spatial_side"}:
            raise ValueError("profile constant overrides a derived field")
        _positive(value, f"profile constant {key}", 256)
    bytes_per_position = channels * (bits // 8)
    combined_bytes_per_position = len(roles) * bytes_per_position
    below_side = math.isqrt((capacity - 1) // combined_bytes_per_position)
    above_side = math.isqrt(capacity // combined_bytes_per_position) + 1
    if below_side < 1 or above_side > 512 or 3 * above_side * above_side > 4_000_000:
        raise ValueError("resource crossing exceeds the bounded iteration workload")
    cases = {}
    for label, side in (("below", below_side), ("above", above_side)):
        footprint = side * side * bytes_per_position
        combined_footprint = len(roles) * footprint
        profile = {
            "schema": PROFILE_SCHEMA,
            "workload_id": workload_id,
            "channels": channels,
            "spatial_side": side,
            **constants,
        }
        cases[label] = {
            "profile": profile,
            "feature_tensor_bytes": footprint,
            "combined_live_tensor_bytes": combined_footprint,
            "relation": "below" if combined_footprint < capacity else "above",
        }
    if cases["below"]["relation"] != "below" or cases["above"]["relation"] != "above":
        raise ValueError("capacity crossing could not be established")
    if concurrent and any(row["feature_tensor_bytes"] >= capacity for row in cases.values()):
        raise ValueError("concurrent crossing must keep each individual tensor below capacity")
    return {
        "schema": "merlin.iteration_variant_derivation.v1",
        "target": observed["target"],
        "workload_id": workload_id,
        "facts_sha256": _sha(facts_bytes),
        "template_sha256": _sha(template_bytes),
        "boundary": {
            "kind": boundary["kind"],
            "memory": memory["name"],
            "array": array["name"],
            "array_axis": boundary["array_axis"],
            "array_lanes": lane_count,
            "element_bits": bits,
            "address_space_bytes": capacity,
            "bytes_per_spatial_position": bytes_per_position,
            "live_tensor_roles": roles,
            "live_tensor_count": len(roles),
            "combined_bytes_per_spatial_position": combined_bytes_per_position,
            "scope": (
                "authored logical concurrent tensors; selected graph liveness, integer "
                "layout, allocation, placement, transfer and runtime unverified"
                if concurrent
                else "one activation tensor only; no allocation, layout or runtime proof"
            ),
        },
        "cases": cases,
    }


def materialize(*, facts_path: Path, template_path: Path, loader_path: Path, output_root: Path) -> dict:
    """Write fresh, self-contained workload roots suitable for capture selection."""
    facts_bytes = _ordinary(facts_path)
    template_bytes = _ordinary(template_path)
    loader_bytes = _ordinary(loader_path)
    result = derive(facts_bytes, template_bytes)
    if loader_path.name != "loader.py" or loader_path.parent.name != result["workload_id"]:
        raise ValueError("selected independent loader does not match the variant workload id")
    output_root = output_root.absolute()
    if output_root.exists() or not output_root.parent.is_dir():
        raise ValueError("variant output must be a fresh directory with an existing parent")
    if output_root.is_relative_to(loader_path.parent.absolute()) or output_root in {
        facts_path.absolute(),
        template_path.absolute(),
        loader_path.absolute(),
    }:
        raise ValueError("variant output must not overlap selected source inputs")
    result["loader_sha256"] = _sha(loader_bytes)
    result["source_loader"] = str(loader_path.absolute())
    result["source_facts"] = str(facts_path.absolute())
    result["source_template"] = str(template_path.absolute())
    output_root.mkdir(mode=0o700)
    for label, row in result["cases"].items():
        member = output_root / label
        member.mkdir(mode=0o700)
        profile_bytes = _json(row["profile"])
        manifest = {
            "schema": "merlin.iteration_variant_manifest.v1",
            "status": "derived_capture_input_not_performance_evidence",
            "case": label,
            "profile_sha256": _sha(profile_bytes),
            "loader_sha256": result["loader_sha256"],
            "facts_sha256": result["facts_sha256"],
            "template_sha256": result["template_sha256"],
            "boundary": result["boundary"],
            "feature_tensor_bytes": row["feature_tensor_bytes"],
            "combined_live_tensor_bytes": row["combined_live_tensor_bytes"],
            "relation": row["relation"],
            "source_loader": result["source_loader"],
            "source_facts": result["source_facts"],
            "source_template": result["source_template"],
            "capture_input_scope": "independent development model; no headline model source or result",
            "residency_verification": (
                "unverified: requires a selected capture proving named tensor liveness, "
                "dtype/layout, device placement, and measured movement"
            ),
        }
        for name, data in (
            ("loader.py", loader_bytes),
            ("profile.json", profile_bytes),
            ("variant-manifest.json", _json(manifest)),
        ):
            with (member / name).open("xb") as stream:
                stream.write(data)
        row["workload_root"] = str(member)
        row["profile_sha256"] = manifest["profile_sha256"]
    (output_root / "derivation.json").write_bytes(_json(result))
    return result
