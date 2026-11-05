"""Bounded joint-shape planning from explicitly public development metadata.

This creates no source, admission, timing prediction or workload score. Normal
Phase 0 source construction, private exclusion and qualification still apply.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from math import ceil, log2, prod

SUMMARY_SCHEMA = "merlin.phase0.public_shape_summary.v1"
PLAN_SCHEMA = "merlin.phase0.joint_shape_spectrum_plan.v1"


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _identity(row):
    return _digest({name: row[name] for name in ("family", "contract", "shape")})


def _row(row):
    if not isinstance(row, dict) or set(row) != {"family", "contract", "shape", "occurrences"}:
        raise ValueError("shape rows require family, semantic/layout contract, shape and occurrences")
    shape = row["shape"]
    if (
        any(not isinstance(row[key], str) or not row[key] for key in ("family", "contract"))
        or not isinstance(shape, dict)
        or not 1 <= len(shape) <= 8
        or any(
            not isinstance(key, str) or not key or type(value) is not int or not 0 < value <= 2**31
            for key, value in shape.items()
        )
        or type(row["occurrences"]) is not int
        or not 0 < row["occurrences"] <= 2**63
    ):
        raise ValueError("shape summaries require bounded positive integral dimensions and occurrence counts")
    return {**row, "shape": dict(sorted(shape.items()))}


def _morphology(shape, granules):
    if set(shape) != {"M", "K", "N"}:
        return "declared_geometry"
    m, k, n = (shape[axis] for axis in ("M", "K", "N"))
    tags = ["gemv" if m == 1 else "short_m" if m <= granules["M"] else "array_rows"]
    tags.append("tall" if m >= 4 * n else "wide" if n >= 4 * m else "balanced")
    if k >= 4 * max(m, n):
        tags.append("deep_k")
    return "/".join(tags)


def plan(summary, *, axis_granules, resource_sha256, max_members=64, bin_octaves=3, extra_density_points=8):
    """Keep joint tuples, coarse log-size bins, extrema and an upper envelope.

    Dimensions must be independent source parameters, not arbitrary axes of a
    connected graph. ``contract`` partitions layouts/readouts and semantic
    flavors. Occurrences distribute sampling effort; they are never latency.
    """
    if (
        not isinstance(summary, dict)
        or set(summary) != {"schema", "scope", "source_sha256", "members"}
        or summary["schema"] != SUMMARY_SCHEMA
        or summary["scope"] != "public_development"
        or not isinstance(summary["source_sha256"], list)
        or not summary["source_sha256"]
        or any(not _hash(value) for value in summary["source_sha256"])
        or not isinstance(summary["members"], list)
        or not 1 <= len(summary["members"]) <= 4096
        or not _hash(resource_sha256)
        or any(type(value) is not int or value < 1 for value in (max_members, bin_octaves))
        or max_members > 4096
        or bin_octaves > 32
        or type(extra_density_points) is not int
        or not 0 <= extra_density_points <= max_members
    ):
        raise ValueError("select bounded public development metadata and an explicit resource identity")
    if not isinstance(axis_granules, dict) or any(
        not isinstance(key, str) or type(value) is not int or not 0 < value <= 2**31
        for key, value in axis_granules.items()
    ):
        raise ValueError("axis granules must be explicit positive resource inputs")
    merged = {}
    for declaration in summary["members"]:
        row = _row(declaration)
        identity = _identity(row)
        if identity in merged:
            merged[identity]["occurrences"] += row["occurrences"]
        else:
            merged[identity] = row
    rows = sorted(merged.values(), key=lambda row: _digest(row))
    if any(set(row["shape"]) - axis_granules.keys() for row in rows):
        raise ValueError("every independent dimension requires a selected granule")
    partitions, bins = defaultdict(list), defaultdict(list)
    for row in rows:
        key = (row["family"], row["contract"], tuple(row["shape"]))
        partitions[key].append(row)
        volume_bin = (prod(row["shape"].values()).bit_length() - 1) // bin_octaves
        bins[(*key, _morphology(row["shape"], axis_granules), volume_bin)].append(row)
    candidates = {}

    def add(row, purpose, priority):
        key = _identity(row)
        point = candidates.setdefault(
            key,
            {
                "id": key,
                "family": row["family"],
                "contract": row["contract"],
                "shape": row["shape"],
                "purposes": [],
                "priority": priority,
                "occurrence_proxy": 0,
                "admission": "pending_source_semantics_private_exclusion_and_accelerator_qualification",
            },
        )
        if purpose not in point["purposes"]:
            point["purposes"].append(purpose)
        point["priority"] = min(priority, point["priority"])
        point["occurrence_proxy"] = max(point["occurrence_proxy"], row["occurrences"])
        return key

    for key, members in sorted(partitions.items()):
        for axis in key[2]:
            low = min(members, key=lambda row: (row["shape"][axis], _digest(row)))
            high = max(members, key=lambda row: (row["shape"][axis], _digest(row)))
            add(low, "lower_extent:" + axis, 0)
            add(high, "upper_extent:" + axis, 0)
            shape = dict(high["shape"])
            grain = axis_granules[axis]
            # Expand one independent parameter, retaining the other coordinates.
            expanded = ceil(shape[axis] * 5 / 4)
            shape[axis] = expanded if shape[axis] < grain else ceil(expanded / grain) * grain
            if shape[axis] > 2**31:
                raise ValueError("upper envelope exceeds the selected finite shape-planning domain")
            add({**high, "shape": shape}, "above_upper_extent:" + axis, 0)
    cells = []
    for key, members in sorted(bins.items()):
        total = sum(row["occurrences"] for row in members)
        center = {
            axis: sum(log2(row["shape"][axis]) * row["occurrences"] / total for row in members)
            for axis in members[0]["shape"]
        }
        representative = min(
            members,
            key=lambda row: (
                sum(abs(log2(value) - center[axis]) for axis, value in row["shape"].items()),
                _digest(row),
            ),
        )
        identity = add(representative, "occupied_joint_bin", 1)
        cells.append(
            {
                "family": key[0],
                "contract": key[1],
                "axes": list(key[2]),
                "morphology": key[3],
                "volume_bin": key[4],
                "occurrences": total,
                "representative": identity,
                "joint_witnesses": sorted({_identity(row) for row in members}),
            }
        )
    # Optional additional samples favor common bins; never drop the extrema or
    # silently turn an unrepresented bin into coverage when the budget is small.
    density = sorted(rows, key=lambda row: (-row["occurrences"], _digest(row)))
    additions = 0
    for row in density:
        identity = _identity(row)
        if identity not in candidates and additions < extra_density_points:
            add(row, "distribution_density", 2)
            additions += 1
    ordered = sorted(candidates.values(), key=lambda row: (row["priority"], -row["occurrence_proxy"], row["id"]))
    selected, deferred = ordered[:max_members], ordered[max_members:]
    present = {row["id"] for row in selected}
    for cell in cells:
        cell["represented"] = bool(present.intersection(cell["joint_witnesses"]))
    output = {
        "schema": PLAN_SCHEMA,
        "status": "planned_complete" if not deferred else "budget_incomplete",
        "summary_sha256": _digest(summary),
        "resource_sha256": resource_sha256,
        "source_sha256": sorted(summary["source_sha256"]),
        "scope": "public_development_plan_only",
        "policy": {
            "max_members": max_members,
            "bin_octaves": bin_octaves,
            "extra_density_points": extra_density_points,
            "upper_margin": [5, 4],
            "axis_granules": dict(sorted(axis_granules.items())),
        },
        "members": selected,
        "deferred": deferred,
        "occupied_bins": cells,
        "missing_bins": sum(not cell["represented"] for cell in cells),
        "unfulfilled_requirements": [
            "hardware_capacity_knees_and_independent_tails_from_existing_capacity_families",
            "connected_stateful_patterns_and_reduced_depth_public_source_factories",
            "reviewed_software_support_original_numerics_and_private_exclusion",
            "ordinary_capture_materialization_baseline_execution_and_frozen_cohort_admission",
        ],
        "claims": {
            "cohort_admitted": False,
            "measured": False,
            "latency_weighting": False,
            "full_model_coverage": False,
            "validation_execution": False,
        },
    }
    return {**output, "sha256": _digest(output)}
