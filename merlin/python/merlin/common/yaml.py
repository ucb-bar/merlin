"""Deterministic YAML load/dump wrappers.

TargetGen emits YAML artifacts that must be byte-stable across runs (so diffs are
meaningful and tests are reproducible). ``dump_yaml`` therefore sorts keys and disables
PyYAML's line-wrapping and aliasing. Stdlib + PyYAML only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> Any:
    """Parse a YAML file."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def dump_yaml(obj: Any) -> str:
    """Serialize ``obj`` to a deterministic YAML string.

    Keys are sorted, flow style is block, lines are not wrapped, and no anchors/aliases are
    emitted. The same object always yields the same bytes.
    """
    return yaml.safe_dump(
        obj,
        sort_keys=True,
        default_flow_style=False,
        width=10**9,
        allow_unicode=True,
    )


def write_yaml(path: str | Path, obj: Any, header: str | None = None) -> Path:
    """Write ``obj`` as deterministic YAML to ``path`` (creating parents).

    An optional ``header`` comment line is prepended (without the leading ``# ``; it is
    added here). Returns the written path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = dump_yaml(obj)
    if header:
        text = f"# {header}\n{text}"
    p.write_text(text, encoding="utf-8")
    return p
