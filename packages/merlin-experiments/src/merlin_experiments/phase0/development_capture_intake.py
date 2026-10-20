"""Bound capture cost and reopen independent references for unchanged sources.

This is a diagnostic intake, not the sealed capture or publication authority.
Every source remains in the coverage report, including unsupported and costly
members. A passing cost screen is no prediction of compiler size or runtime.
"""

from __future__ import annotations

from math import prod
from pathlib import Path

import numpy as np

from . import development_cohort as F
from . import development_reference_supplement as S


def references(root, *, manifest_sha256, supplements=()):
    """Reopen byte-pinned complete evidence; never call reused outputs fresh."""
    root = Path(root)
    manifest = F.verify(root, manifest_sha256=manifest_sha256)
    catalog = {}
    for member in manifest["members"]:
        if member.get("reference"):
            catalog[member["ordinal"]] = {
                "kind": "original_frozen_npz",
                "path": str(root / member["reference"]),
                "identity": F._identity(root / member["reference"]),
            }
    seen = set()
    for path, digest in supplements:
        path = Path(path)
        if digest in seen:
            raise ValueError("duplicate selected reference supplement")
        seen.add(digest)
        supplement = S.verify(path, index_sha256=digest, sources=root, manifest_sha256=manifest_sha256)
        for row in supplement["members"]:
            if row["status"] != "complete_exact_reference_materialized":
                continue
            if row["ordinal"] in catalog:
                raise ValueError("multiple independent reference products selected for one source")
            catalog[row["ordinal"]] = {
                "kind": "supplement_complete_npy",
                "index": str(path),
                "index_sha256": digest,
                "outputs": [{**item, "path": str(path.parent / item["path"])} for item in row["outputs"]],
            }
    return manifest, catalog


def load_outputs(member, selected):
    """Map every independently frozen output in its original declaration order."""
    abi = member["description"]["outputs"]
    if selected["kind"] == "original_frozen_npz":
        if F._identity(Path(selected["path"])) != selected["identity"]:
            raise ValueError("frozen independent reference bytes changed")
        with np.load(selected["path"], allow_pickle=False) as archive:
            names = {name for name in archive.files if name.startswith("output_")}
            if names != {f"output_{i}" for i in range(len(abi))}:
                raise ValueError("frozen reference output roster changed")
            outputs = [archive[f"output_{i}"] for i in range(len(abi))]
    elif selected["kind"] == "supplement_complete_npy":
        outputs = []
        for item in selected["outputs"]:
            path = Path(item["path"])
            if path.is_symlink() or F._identity(path) != {key: item[key] for key in ("bytes", "sha256")}:
                raise ValueError("supplemental independent reference bytes changed")
            outputs.append(np.load(path, mmap_mode="r", allow_pickle=False))
    else:
        raise ValueError("unrecognized independent reference intake")
    if len(outputs) != len(abi) or any(
        list(value.shape) != row["shape"] or value.dtype != np.dtype(row["dtype"])
        for value, row in zip(outputs, abi, strict=True)
    ):
        raise ValueError("independent reference changed a complete output ABI")
    return outputs


def plan(manifest, catalog, *, max_work, max_elements, max_seed_bytes, max_contractions):
    """Conservative admission of actual dense capture, preserving every member.

    Seed bytes are a lower bound of the unchanged eager Torch arange expression,
    not an upper bound on the process. Contraction count screens graph expansion;
    it supplies no target instruction-count estimate or capacity qualification.
    """
    limits = dict(
        max_work=max_work,
        max_elements=max_elements,
        max_seed_bytes=max_seed_bytes,
        max_contractions=max_contractions,
    )
    if any(type(value) is not int or value < 1 for value in limits.values()):
        raise ValueError("capture intake requires explicit positive finite cost ceilings")
    members = []
    for member in manifest["members"]:
        description = member.get("description")
        row = {
            "ordinal": member["ordinal"],
            "request_sha256": member["request_sha256"],
            "original_status": member["status"],
            "numeric_policy": member.get("numeric_policy"),
            "performance_boundary": member.get("performance_boundary"),
            "status": "capture_cost_refused",
            "reasons": [],
        }
        members.append(row)
        if not description:
            row["reasons"].append(member.get("reason") or "original source unavailable")
            continue
        seed_lower = 16 * max(prod(item["shape"]) for item in description["inputs"])
        costs = dict(
            max_work=description["scalar_products"],
            max_elements=description["logical_elements"],
            max_seed_bytes=seed_lower,
            max_contractions=description["contractions"],
        )
        row.update(costs=costs, compiler_expansion="UNKNOWN until bounded emitted-program checks")
        if description.get("pending_numeric_policy"):
            row["reasons"].append("original selected numerical contract remains pending")
        if member["ordinal"] not in catalog:
            row["reasons"].append("complete independent frozen reference unavailable")
        row["reasons"] += [
            f"{key}: observed {value} exceeds {limits[key]}" for key, value in costs.items() if value > limits[key]
        ]
        if not row["reasons"]:
            row["status"] = "capture_cost_screen_passed_not_admission"
    return {
        "schema": "merlin.phase0.public_development_capture_intake.v1",
        "budgets": limits,
        "phase0_admitted": False,
        "performance_objective_changed": False,
        "members": members,
    }
