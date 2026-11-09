"""Explicit application preparation profiles over the existing recipe adapters.

Profiles own calibration, recovery and reviewed model variants. The selected
software specification continues to own mathematical meaning. A profile cannot
quietly replace requested quantization with the target's preferred granularity
or static/dynamic recipe: the caller must select the exact proposed alternative.
The named v1 compatibility policy retains historical output-path derivation.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.compiler.quantization import methodology as M
from merlin.compiler.quantization import static as Q

SCHEMA = "merlin.application_preparation_profile.v1"
PRESERVE_REQUEST = "preserve_request_v1"
LEGACY_OUTPUT_PATH = "legacy_output_path_v1"
_FIELDS = {"schema", "id", "compatibility_policy", "methodology", "gptq_weights", "description"}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def validate_profile(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate policy without adding it to a target's semantic authority."""
    if not isinstance(document, Mapping) or set(document) - _FIELDS or document.get("schema") != SCHEMA:
        raise ValueError("application preparation profile has an unknown schema or field")
    if not isinstance(document.get("id"), str) or not document["id"].strip():
        raise ValueError("application preparation profile requires an id")
    body = copy.deepcopy(dict(document))
    policy = body.setdefault("compatibility_policy", PRESERVE_REQUEST)
    if policy not in (PRESERVE_REQUEST, LEGACY_OUTPUT_PATH):
        raise ValueError("application preparation profile has an unknown compatibility_policy")
    body["methodology"] = M.validate(body.get("methodology"))
    if "gptq_weights" in body:
        from merlin.targetgen import _gptq

        if not isinstance(body["gptq_weights"], Mapping):
            raise ValueError("gptq_weights must state the existing GPTQ parameters")
        body["gptq_weights"] = _gptq.check_parameters(dict(body["gptq_weights"]))
    _canonical(body)
    return body


@dataclass(frozen=True)
class PreparationProfile:
    """Immutable authored bytes and source identity, verified on every use."""

    raw: bytes
    source_path: str | None = None

    def document(self) -> dict[str, Any]:
        if self.source_path is not None and Path(self.source_path).read_bytes() != self.raw:
            raise ValueError("application preparation profile source bytes changed after selection")
        import yaml

        return validate_profile(yaml.safe_load(self.raw.decode("utf-8")))

    def identity(self) -> dict[str, Any]:
        body = self.document()
        return {
            "id": body["id"], "schema": SCHEMA, "source": self.source_path,
            "source_sha256": _sha(self.raw), "profile_sha256": _sha(_canonical(body)),
            "compatibility_policy": body["compatibility_policy"],
        }


def load_profile(path: str | Path) -> PreparationProfile:
    """Select exactly the supplied file; never discover profiles by target/model."""
    source = Path(path).resolve()
    selected = PreparationProfile(source.read_bytes(), str(source))
    selected.document()
    return selected


def profile_from_mapping(document: Mapping[str, Any]) -> PreparationProfile:
    """Select an inline profile with its complete canonical authored identity."""
    validate_profile(document)
    return PreparationProfile(_canonical(document))


class PreparationAlternativeRequired(ValueError):
    """A reviewable proposal, withheld from executable capture candidates."""

    def __init__(self, proposal: Mapping[str, Any]):
        self.proposal = copy.deepcopy(dict(proposal))
        super().__init__(
            "preparation changes requested numerical intent; explicitly select alternative_sha256 "
            + self.proposal["alternative_sha256"]
        )


def _intent(recipe: Mapping[str, Any]) -> dict[str, Any]:
    # Unknown recipe extensions remain visible instead of disappearing from a
    # closed list of numerical algorithms. Only preparation metadata is omitted.
    ignored = {
        "recipe_sha256", "why", "underivable", "status", "methodology", "preparation",
        "recovery", "float_island", "framework_capture_policy", "unquantized",
    }
    body = {key: copy.deepcopy(value) for key, value in recipe.items() if key not in ignored}
    for role in ("weight", "activation"):
        if isinstance(body.get(role), dict):
            body[role].pop("observer", None)
    return body


def _changes(before: Any, after: Any, prefix: str = "") -> list[dict[str, Any]]:
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        result = []
        for key in sorted(set(before) | set(after)):
            name = f"{prefix}.{key}" if prefix else str(key)
            if key not in before or key not in after:
                result.append({"field": name, "requested": before.get(key), "proposed": after.get(key),
                               "requested_present": key in before, "proposed_present": key in after})
            else:
                result.extend(_changes(before[key], after[key], name))
        return result
    return [] if before == after else [{"field": prefix, "requested": before, "proposed": after}]


def select_recipe(
    requested: Mapping[str, Any],
    profile: PreparationProfile,
    output_path: Mapping[str, Any] | None,
    *,
    model: str | None = None,
    alternative_sha256: str | None = None,
) -> dict[str, Any]:
    """Bind an exact application policy and return an admitted preparation recipe.

    This selects existing derivation algorithms, not TorchAO implementations or
    target support. Changed intent raises ``PreparationAlternativeRequired`` with
    requested/proposed recipes and a digest binding both and every selected input.
    Passing that digest explicitly authorizes just that alternative. Independent
    quality gates remain outside the profile and unchanged.
    """
    if not isinstance(profile, PreparationProfile):
        raise TypeError("preparation_profile must be an explicitly selected PreparationProfile")
    if requested.get("recipe_sha256") != Q.digest(requested):
        raise ValueError("requested preparation recipe content identity differs")
    document, identity = profile.document(), profile.identity()
    body = M.derive(requested, document["methodology"], output_path, model=model)
    if body.get("status") == Q.DERIVED and "gptq_weights" in document:
        body = Q.with_gptq_weights(body, **document["gptq_weights"])
    changes = _changes(_intent(requested), _intent(body))
    proposal = {
        "schema": "merlin.preparation_alternative.v1", "profile": identity, "model": model,
        "output_path_sha256": _sha(_canonical(output_path)),
        "requested_recipe_sha256": requested["recipe_sha256"],
        "proposed_recipe_sha256": body["recipe_sha256"],
        "requested_recipe": copy.deepcopy(dict(requested)), "proposed_recipe": copy.deepcopy(body),
        "changes": changes,
    }
    proposal["alternative_sha256"] = _sha(_canonical(proposal))
    if alternative_sha256 is not None and alternative_sha256 != proposal["alternative_sha256"]:
        raise ValueError("selected preparation alternative differs from the exact current proposal")
    if (body.get("status") == Q.DERIVED and changes
            and document["compatibility_policy"] == PRESERVE_REQUEST and alternative_sha256 is None):
        raise PreparationAlternativeRequired(proposal)
    body["preparation"] = {
        "schema": "merlin.preparation_selection.v1", "profile": identity, "model": model,
        "requested_recipe_sha256": requested["recipe_sha256"],
        "derived_recipe_sha256": body["recipe_sha256"], "changes": changes,
        "selected_alternative_sha256": alternative_sha256,
    }
    if profile.identity() != identity:
        raise ValueError("application preparation profile identity changed during selection")
    body["recipe_sha256"] = Q.digest(body)
    return body
