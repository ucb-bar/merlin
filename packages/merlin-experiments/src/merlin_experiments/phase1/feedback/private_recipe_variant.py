"""Bind an explicitly reviewed numerical recipe variant to its derived parent.

The operator-private spec carries the review and the complete changes. This does
not expand the hardware's readout capabilities or waive source/numerical checks.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from merlin.targetgen.quant_recipe import digest

SCHEMA = "merlin.phase1.capture_recipe_variant.v1"


def changes(before, after, path=()) -> list[dict]:
    """A deterministic complete diff; explicit presence distinguishes removal from null."""
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        rows = []
        for key in sorted(before.keys() | after.keys()):
            if not path and key == "recipe_sha256":
                continue
            here = [*path, key]
            if key not in before:
                rows.append({"path": here, "before_present": False, "after": after[key]})
            elif key not in after:
                rows.append({"path": here, "before": before[key], "after_present": False})
            else:
                rows.extend(changes(before[key], after[key], here))
        return rows
    same = json.dumps(before, sort_keys=True) == json.dumps(after, sort_keys=True)
    return [] if same else [{"path": list(path), "before": before, "after": after}]


def verify(review, *, parent: bytes, selected: bytes, target: str) -> dict:
    """Require both exact file pins, both self digests and every declared change."""
    if (
        not isinstance(review, Mapping)
        or review.get("schema") != SCHEMA
        or review.get("status") != "reviewed"
        or any(not isinstance(review.get(key), str) or not review[key].strip() for key in ("reviewed_by", "reason"))
    ):
        raise ValueError("capture recipe variant needs an explicit operator review")
    base, variant = json.loads(parent), json.loads(selected)
    for role, raw, body in (("base", parent, base), ("selected", selected, variant)):
        if (
            not isinstance(body, Mapping)
            or body.get("schema") != "quant_recipe_v1"
            or body.get("target") != target
            or body.get("status") != "derived"
            or hashlib.sha256(raw).hexdigest() != review.get(role + "_file_sha256")
            or body.get("recipe_sha256") != review.get(role + "_recipe_sha256")
            or digest(body) != body.get("recipe_sha256")
        ):
            raise ValueError(f"capture recipe variant has a changed {role} recipe")
    delta = changes(base, variant)
    if not delta or json.dumps(review.get("changes"), sort_keys=True) != json.dumps(delta, sort_keys=True):
        raise ValueError("capture recipe variant does not declare its exact changes")
    review_bytes = json.dumps(dict(review), sort_keys=True, separators=(",", ":")).encode()
    return {
        "review_sha256": hashlib.sha256(review_bytes).hexdigest(),
        "base_recipe_sha256": base["recipe_sha256"],
        "base_file_sha256": review["base_file_sha256"],
        "reviewed_by": review["reviewed_by"],
        "scope": "operator-reviewed numerical variant; no additional hardware or numerical admission",
    }
