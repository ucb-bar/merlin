"""Host-only golden provenance queries, independent of numerical evaluation.

Only the declared source and its grading regime leave this module. Golden values,
operands and complete YAML documents are never returned or cached here. These
queries remain withheld host metadata, not an exemption for candidate imports.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

__all__ = [
    "golden_source",
    "is_independent_float_golden",
    "selected_golden_source",
    "selected_independent_float_golden",
]


def _declared_source(capsule_dir: str | Path | None) -> Any:
    """Project one provenance field; preserve the legacy parser and failure behavior."""
    if not capsule_dir:
        return None
    import yaml

    path = Path(capsule_dir) / "golden.yaml"
    if not path.is_file():
        return None
    # Parsing a string (not an open stream) preserves PyYAML's legacy diagnostic
    # source labels. Do not suppress malformed YAML or normalize its shape here.
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("golden_source")


def golden_source(
    capsule: dict,
    capsule_dir: str | Path | None = None,
    *,
    metadata_reader: Callable[[str | Path | None], Any] | None = None,
) -> str:
    """The declared independent source, or the existing integer-recompute default.

    The optional reader lets an evaluator preserve its own loader overrides.
    Its document is immediately projected; no full-answer loader is exported.
    Truthy source values are deliberately not cast or newly schema-validated.
    """
    if capsule_dir is None:
        capsule_dir = capsule.get("__dir__")
    source = (
        _declared_source(capsule_dir)
        if metadata_reader is None
        else (metadata_reader(capsule_dir) or {}).get("golden_source")
    )
    return source if (source and source != "merlin_tensor_int") else "merlin_tensor_int"


def is_independent_float_golden(
    capsule: dict,
    capsule_dir: str | Path | None = None,
    *,
    source_reader: Callable[[dict, str | Path | None], Any] | None = None,
) -> bool:
    """Use the same per-capsule regime decision as evaluation, without evaluating.

    Integer comparisons short-circuit without consulting the source or filesystem.
    The injected lookup preserves evaluator-local source overrides.
    """
    compare = (capsule.get("numeric_policy") or {}).get("compare", "exact_int")
    float_policy = compare not in ("exact_int", "exact")
    lookup = golden_source if source_reader is None else source_reader
    return float_policy and lookup(capsule, capsule_dir) != "merlin_tensor_int"


def selected_golden_source(capsule: dict, capsule_dir: str | Path | None = None) -> str:
    """Honor an already-loaded evaluator's overrides without importing that evaluator."""
    evaluator = sys.modules.get("merlin.targetgen.capsule_golden")
    lookup = getattr(evaluator, "golden_source", golden_source)
    return lookup(capsule, capsule_dir)


def selected_independent_float_golden(capsule: dict, capsule_dir: str | Path | None = None) -> bool:
    """Select the active classifier, including legacy source/loader overrides.

    Evaluator adapters delegate to the pure queries above, never these selectors,
    so resolving an active callback cannot recurse through the adapter itself.
    """
    evaluator = sys.modules.get("merlin.targetgen.capsule_golden")
    lookup = getattr(evaluator, "is_independent_float_golden", is_independent_float_golden)
    return lookup(capsule, capsule_dir)
