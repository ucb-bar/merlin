"""Emit a runtime_candidate dict (conforming to ``runtime_candidate.schema.yaml``)."""
from __future__ import annotations

from typing import Iterable

from merlin.common import schemas


def emit_runtime_candidate(
    name: str,
    evidence: Iterable[str],
    compiler_action: Iterable[str],
    runtime_requirement: Iterable[str],
    observed: dict | None = None,
    extra: dict | None = None,
    validate: bool = True,
) -> dict:
    """Build a schema-shaped runtime candidate (L7)."""
    cand = {
        "name": name,
        "evidence": sorted(dict.fromkeys(evidence)),
        "compiler_action": list(compiler_action),
        "runtime_requirement": list(runtime_requirement),
    }
    if observed:
        cand["observed"] = observed
    if extra:
        cand.update(extra)
    if validate:
        schemas.validate_or_raise(cand, "runtime_candidate")
    return cand
