"""Per-framework contract descriptors — the caller-side assumptions (prepack/transpose/layout/
accumulator/dtype) that are NOT in a kernel's body or assembly, so they can't be mined from code
alone. Hand-authored once per framework (~the XNNPACK-transpose knowledge), agent-refined, and
loaded by the dossier so the agent reads the contract alongside the code facts.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from ...common.yaml import load_yaml

_DIR = Path(__file__).resolve().parent

# kernel.source string -> contract file stem
_SOURCE_TO_FRAMEWORK = {
    "xnnpack": "xnnpack",
    "openblas": "openblas",
    "saturn": "saturn",
    "saturn_vectors": "saturn",
}


@lru_cache(maxsize=None)
def load_contract(framework: str) -> dict[str, Any]:
    """Load a framework contract by source/framework name. Returns {} if none exists (the kernel
    simply has no recorded caller contract — e.g. an unmapped source)."""
    stem = _SOURCE_TO_FRAMEWORK.get((framework or "").lower(), (framework or "").lower())
    path = _DIR / f"{stem}.yaml"
    if not path.is_file():
        return {}
    return load_contract_file(path)


def load_contract_file(path: Path) -> dict[str, Any]:
    return load_yaml(path) or {}


def available_frameworks() -> list[str]:
    return sorted(p.stem for p in _DIR.glob("*.yaml"))
