"""Explicit historical source exclusions; absence preserves legacy input selection."""

from pathlib import Path


def declared_held_out_inputs(document: dict) -> tuple[str, ...]:
    """Validate repo-relative private inputs without probing or granting their paths."""
    values = (document.get("answer_surfaces") or {}).get("held_out_inputs", [])
    if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() or "\x00" in v for v in values):
        raise ValueError("answer_surfaces.held_out_inputs must list repo-relative paths")
    paths = tuple(Path(value) for value in values)
    if any(p.is_absolute() or not p.parts or ".." in p.parts for p in paths):
        raise ValueError("answer_surfaces.held_out_inputs must not escape the source root")
    if len(set(paths)) != len(paths):
        raise ValueError("answer_surfaces.held_out_inputs must be unique")
    return tuple(str(path) for path in paths)
