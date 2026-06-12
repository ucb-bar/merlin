"""Emit an abstraction_candidate dict (conforming to ``abstraction_candidate.schema.yaml``)."""
from __future__ import annotations

from typing import Iterable

from merlin.common import schemas


def emit_abstraction_candidate(
    name: str,
    kind: str,
    motivation: str,
    evidence: Iterable[str],
    interface_features: Iterable[str],
    extra: dict | None = None,
    validate: bool = True,
) -> dict:
    """Build a schema-shaped abstraction candidate.

    ``evidence`` is the list of real kernel evidence-ids (e.g. ``xnnpack_rvv_gemm``) that
    surfaced the candidate; ``interface_features`` are the proposed interface ops/types.
    """
    candidate = {
        "name": name,
        "kind": kind,
        "motivation": motivation,
        "evidence": sorted(dict.fromkeys(evidence)),
        "interface_features": list(interface_features),
    }
    if extra:
        candidate.update(extra)
    if validate:
        schemas.validate_or_raise(candidate, "abstraction_candidate")
    return candidate
