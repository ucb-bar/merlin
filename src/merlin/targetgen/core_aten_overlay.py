"""Additive, evidence-derived target overlays for the portable Core ATen suite.

An overlay is a separate case set whose cases may add to the portable bounded suite but may never
satisfy or replace a portable obligation.  This module owns the target-independent mechanism: build
the candidate pool, run every candidate twice, take an exact minimum cover, and fail closed when a
stored overlay no longer matches its sources, solver proof or eager oracles.

Everything that is a fact about one target -- the pinned configuration sources, the candidate
shapes and the obligation vocabulary -- is supplied by an :class:`OverlayProvider` that lives at the
target's own edge.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from merlin.targetgen.core_aten_cases import case_document_from_arguments
from merlin.targetgen.core_aten_cover import exact_minimum_cover, json_bytes, overload_digest

CORE = "core"
NON_CORE_BRIDGE = "non_core_target_bridge"

#: (case name, overload, argument factory, covered obligations, denominator classification)
CandidateSpec = tuple[str, str, Callable[[], tuple[tuple, dict]], set[str], str]


class OverlayProvider(Protocol):
    """Target-owned inputs of an overlay; implemented beside the target, never in shared code."""

    def profile(self) -> dict[str, Any]:
        """Read and validate the pinned configuration; must carry ``name``, ``target`` and ``facts``."""

    def candidate_specs(self, profile: Mapping[str, Any]) -> Sequence[CandidateSpec]:
        """The finite candidate pool derived from ``profile``."""

    def document_fields(self) -> dict[str, Any]:
        """Descriptive overlay fields: ``scope``, ``claim``, ``non_core_bridge_overloads``, ``deferred``."""

    def summary(self, document: Mapping[str, Any]) -> str:
        """Human-readable Markdown summary of a built overlay."""


def profile_digest(profile: Mapping[str, Any]) -> str:
    return hashlib.sha256(json_bytes(profile)).hexdigest()


def overlay_digest(document: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in document.items() if key != "overlay_sha256"}
    return hashlib.sha256(json_bytes(payload)).hexdigest()


def _case_id(target: str, name: str, case: Mapping[str, Any]) -> str:
    payload = json.dumps(case["arguments"], sort_keys=True, separators=(",", ":"), allow_nan=False)
    suffix = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"{target}::{name}::{suffix}"


def build_overlay(provider: OverlayProvider, *, pytorch_version: str) -> dict[str, Any]:
    """Build, duplicate-run, and exactly minimize the additive candidate pool."""

    profile = provider.profile()
    target = str(profile["target"])
    candidates: dict[str, dict[str, Any]] = {}
    for name, overload, factory, obligations, classification in provider.candidate_specs(profile):
        args, kwargs = factory()
        first = case_document_from_arguments(overload, args, kwargs)
        args, kwargs = factory()
        second = case_document_from_arguments(overload, args, kwargs)
        if json.dumps(first, sort_keys=True, separators=(",", ":"), allow_nan=False) != json.dumps(
            second, sort_keys=True, separators=(",", ":"), allow_nan=False
        ):
            raise RuntimeError(f"{target} overlay candidate is nondeterministic: {name}")
        identifier = _case_id(target, name, first)
        candidates[identifier] = {
            **first,
            "case_id": identifier,
            "overlay_case_name": name,
            "denominator_classification": classification,
            "covered_obligations": sorted(obligations),
        }
    obligations = sorted({obligation for case in candidates.values() for obligation in case["covered_obligations"]})
    cover = exact_minimum_cover(
        obligations,
        {identifier: case["covered_obligations"] for identifier, case in candidates.items()},
    )
    if cover.status != "optimal" or not cover.selected_union_matches_denominator:
        raise RuntimeError(f"{target} overlay exact cover failed: {cover.to_dict()}")
    selected = [candidates[identifier] for identifier in cover.selected_capsules]
    selected.sort(key=lambda case: case["case_id"])
    selected_union = sorted({obligation for case in selected for obligation in case["covered_obligations"]})
    if selected_union != obligations:
        raise RuntimeError(f"{target} overlay selected union does not equal its obligations")
    fields = provider.document_fields()
    document = {
        "schema_version": 1,
        "scope": fields["scope"],
        "claim": fields["claim"],
        "pytorch_version": pytorch_version,
        "profile": profile,
        "profile_sha256": profile_digest(profile),
        "portable_replacement_forbidden": True,
        "core_aten_denominator_additions": 0,
        "non_core_bridge_overloads": list(fields["non_core_bridge_overloads"]),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "obligation_count": len(obligations),
        "obligations": obligations,
        "selected_union_sha256": overload_digest(obligations),
        "cover": cover.to_dict(),
        "candidates": dict(sorted(candidates.items())),
        "selected_cases": selected,
        "deferred_to_existing_target_conformance": list(fields["deferred_to_existing_target_conformance"]),
    }
    document["overlay_sha256"] = overlay_digest(document)
    return document


def validate_overlay(
    provider: OverlayProvider,
    document: Mapping[str, Any],
    *,
    validate_eager_oracles: bool = True,
    validate_solver_certificate: bool = True,
) -> list[dict[str, Any]]:
    """Fail closed unless an overlay still matches its sources, solver proof, and eager oracles."""

    from merlin.targetgen.core_aten_cases import eager_case_observation, resolve_overload

    if document.get("schema_version") != 1 or document.get("portable_replacement_forbidden") is not True:
        raise ValueError("invalid or non-additive overlay")
    current_profile = provider.profile()
    if document.get("profile") != current_profile:
        raise ValueError("overlay profile no longer matches its pinned configuration sources")
    if document.get("profile_sha256") != profile_digest(current_profile):
        raise ValueError("overlay profile digest mismatch")
    if document.get("overlay_sha256") != overlay_digest(document):
        raise ValueError("overlay digest mismatch")
    obligations = list(document.get("obligations") or ())
    candidates = document.get("candidates")
    selected = list(document.get("selected_cases") or ())
    if (
        obligations != sorted(set(str(item) for item in obligations))
        or not isinstance(candidates, Mapping)
        or len(candidates) != int(document.get("candidate_count", -1))
        or len(selected) != int(document.get("selected_count", -1))
    ):
        raise ValueError("overlay is non-canonical or has inconsistent counts")
    selected_ids = [str(case.get("case_id") or "") for case in selected]
    if selected_ids != sorted(set(selected_ids)) or any(
        candidates.get(identifier) != case for identifier, case in zip(selected_ids, selected)
    ):
        raise ValueError("overlay selected cases do not match its candidate pool")
    selected_union = sorted(
        {str(obligation) for case in selected for obligation in (case.get("covered_obligations") or ())}
    )
    if selected_union != obligations or document.get("selected_union_sha256") != overload_digest(obligations):
        raise ValueError("overlay selected cases do not exactly cover its obligations")
    if validate_solver_certificate:
        cover = exact_minimum_cover(
            obligations,
            {str(identifier): list(case.get("covered_obligations") or ()) for identifier, case in candidates.items()},
        )
        if cover.status != "optimal" or cover.selected_capsules != selected_ids:
            raise ValueError("overlay failed exact-cover revalidation")
    bridges = {str(item) for item in document.get("non_core_bridge_overloads") or ()}
    for case in selected:
        expected_bytes = (
            json.dumps(case.get("expected"), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()
        if case.get("expected_sha256") != hashlib.sha256(expected_bytes).hexdigest():
            raise ValueError(f"overlay oracle digest mismatch: {case.get('case_id')}")
        if case.get("schema") != str(resolve_overload(str(case.get("overload")))._schema):
            raise ValueError(f"overlay schema mismatch: {case.get('case_id')}")
        classification = case.get("denominator_classification")
        if classification not in {CORE, NON_CORE_BRIDGE}:
            raise ValueError(f"overlay case has invalid denominator classification: {case.get('case_id')}")
        if case.get("overload") in bridges and classification != NON_CORE_BRIDGE:
            raise ValueError(f"non-Core bridge {case.get('overload')} was misclassified as Core ATen")
        if validate_eager_oracles:
            observation = eager_case_observation(case)
            if observation["output"] != case["expected"]:
                raise ValueError(f"overlay eager oracle changed: {case.get('case_id')}")
    return selected
