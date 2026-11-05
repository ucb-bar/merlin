"""Explicit release selection of accelerator work, including its required host glue."""

from __future__ import annotations

import copy
import hashlib
import json

from merlin.common.digest import is_sha256

from ..spec import SpecError

SCHEMA = "merlin.corpus.accelerator_scope.v1"


def validate_scope(scope: str, *, generated_only: bool) -> None:
    if not isinstance(scope, str) or scope not in {"all", "accelerator"}:
        raise SpecError("capsule_scope must be all or accelerator")
    if scope == "accelerator" and not generated_only:
        raise SpecError("accelerator capsule selection requires a generated-only release")


def has_accelerator_work(capsule: dict) -> bool:
    """Select from the reviewed program screen, never a name or a previous grade."""
    screen = capsule.get("software_screen") or {}
    placements = screen.get("placements")
    if (
        not isinstance(placements, dict)
        or set(placements) - {"accelerator", "host"}
        or any(type(value) is not int or value < 0 for value in placements.values())
    ):
        raise SpecError("accelerator selection requires an admitted, resolved software screen")
    accelerator = placements.get("accelerator", 0) > 0
    if screen.get("status") == "admitted" and screen.get("constraints_status") == "matched":
        return accelerator
    # A Gemmini refusal test can contain an admitted contraction followed by an
    # unsupported epilogue. Keep the test; selecting it never grants whole-program
    # admission or changes the refusal the ordinary native checker must enforce.
    if accelerator and any(
        row.get("placement") == "accelerator"
        and row.get("classification") == "accelerator_candidate"
        and row.get("status") == "admitted"
        for row in screen.get("decisions") or []
    ):
        return True
    raise SpecError("accelerator selection requires an admitted, resolved software screen")


def select_members(emitted: dict, *, fingerprint) -> tuple[dict, dict]:
    kept, rows = {}, []
    for member, (source, capsule) in sorted(emitted.items()):
        selected = has_accelerator_work(capsule)
        if selected:
            kept[member] = (source, capsule)
        rows.append({"member": member, "sha256": fingerprint(source), "selected": selected})
    if not kept:
        raise SpecError("accelerator capsule selection is empty")
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return kept, {"schema": SCHEMA, "kind": "accelerator", "selection_sha256": digest, "members": rows}


def selected_scope(inputs: dict | None) -> dict | None:
    scope = (inputs or {}).get("capsule_scope")
    if scope is None:
        return None
    if (
        not isinstance(scope, dict)
        or scope.get("schema") != SCHEMA
        or scope.get("kind") != "accelerator"
        or not is_sha256(scope.get("selection_sha256"))
        or not is_sha256(scope.get("source_coverage_inputs_sha256"))
    ):
        raise ValueError("unsupported or unbound capsule scope")
    return scope


def conformance_for_scope(requirement: dict, inputs: dict | None) -> dict:
    """Keep original source obligations; standalone CPU coverage is outside this selection."""
    if selected_scope(inputs) is None:
        return requirement
    result = copy.deepcopy(requirement)
    for axis, key in (("host_lane", "required"), ("host_only", "families")):
        result.setdefault(axis, {})[key] = []
    return result
