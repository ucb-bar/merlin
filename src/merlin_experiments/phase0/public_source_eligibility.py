"""Independent functional sources retain public-only membership and full source checks."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from . import public_boundaries as B
from .operator_exclusion import PublicationRefusal


def require(entries, profile, synth_profile, operator_exclusion):
    """Reopen exact V3 construction; historical boundaries still require exclusion."""
    boundaries = [entry for entry in entries if entry.get("public_boundary")]
    functional = [entry for entry in boundaries if entry["public_boundary"].get("schema") == B.SCHEMA_V3]
    if len(functional) != len(boundaries) and operator_exclusion is None:
        raise PublicationRefusal()
    if not functional:
        return None
    if profile.get("_synth_verification", {}).get("status") != "verified" or synth_profile is None:
        raise PublicationRefusal()
    path = Path(synth_profile)
    declared = (yaml.safe_load(path.read_bytes()) or {}).get("capsules") or []
    names = [entry.get("name") for entry in declared]
    if len(names) != len(set(names)):
        raise ValueError("independent public source names must remain unique")
    selected = {entry["name"]: entry for entry in declared}
    members = []
    for entry in functional:
        original = selected.get(entry["name"])
        if original is None:
            raise PublicationRefusal()
        # This replays original context, cut, parameter roles, selected recipe,
        # exact independent probe geometry, and complete rendered loader bytes.
        loader = B.selected_loader(original, path.parent)
        fields = ("public_boundary", "quant_recipe", "operand_dtype", "kind", "op", "cat", "label")
        if any(entry.get(key) != original.get(key) for key in fields) or Path(entry["loader"]) != loader:
            raise ValueError("independent public source changed after synthesis verification")
        members.append(
            {
                "name": entry["name"],
                "source_basis_sha256": B.digest(original["public_boundary"]),
                "loader_sha256": original["public_boundary"]["loader_sha256"],
            }
        )
    return {
        "schema": "merlin.phase0.independent_public_source_eligibility.v1",
        "basis": "verified_original_public_context_cut_recipe_and_exact_loader",
        "members": copy.deepcopy(members),
        "private_shape_disjointness": "not_claimed",
        "admission": "pending_ordinary_written_source_and_accelerator_checks",
        "execution": "not_established",
    }
