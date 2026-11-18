"""Shared workload identity and capture discovery; no analysis dependencies."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files

# ---- models ------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def workload_names() -> tuple[str, ...]:
    """Source-owned workload identities, independent of captures and DSE metadata."""
    names = json.loads(files("merlin.capture").joinpath("workload_roster.json").read_text(encoding="utf-8"))
    if not isinstance(names, list) or not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("workload roster must be a nonempty list of names")
    if len(names) != len(set(names)):
        raise ValueError("workload roster contains duplicate names")
    return tuple(names)


def _base_model(dirname: str) -> str | None:
    """Map an output capture dirname to a roster name (longest match)."""
    stem = dirname
    for suffix in (
        "_fp32_consistent",
        "_int8_consistent",
        "_fp8_consistent",
        "_consistent",
        "_fp32_biasfix",
        "_int8_biasfix",
        "_int8_recap",
        "_lower",
        "_phase2",
        "_rvv",
        "_host",
        "_spike",
        "_fixed",
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    # Longest first: 'small_llama' must beat 'small'.
    for key in sorted(workload_names(), key=len, reverse=True):
        if stem == key or stem.startswith(key + "_"):
            return key
    return None


def discover_model_captures() -> dict[str, list[str]]:
    """Map base model -> list of capture dirs (absolute) that have a model.mlir."""
    from merlin.common.artifacts import recaptures_dir

    out_root = recaptures_dir()  # artifacts/recaptures/ (symlinked to legacy output/ in transition)
    found: dict[str, list[str]] = {}
    if not out_root.is_dir():
        return found
    for d in sorted(out_root.iterdir()):
        if not d.is_dir() or not (d / "model.mlir").is_file():
            continue
        base = _base_model(d.name)
        if base is None:
            continue
        found.setdefault(base, []).append(str(d))
    return found
