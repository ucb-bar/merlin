"""The single source of target identity: name -> (paths, backend, kind).

Target identity used to be smeared across three hardcoded maps (`pipeline.DEFAULT_BACKEND`,
`target_lowering._specs()`/`LOWERING_TABLES`, `synthesize.dialect_plan.CURATED_TARGETS`) plus ~6
ad-hoc `parents[N]/"merlin/targets/..."` path readers. This module resolves everything a target needs
from one place, reusing the path resolvers in `merlin.targetgen.rtl.facts`.

Two kinds of target:
- ``reference`` — a curated definition under ``merlin/targets/<name>/`` (toy_npu, saturn, gemmini).
- ``generated`` — an isolated package under ``artifacts/targets/<name>/<run_id>/``, loaded by
  :func:`merlin.targetgen.registry.load_target`. This module resolves the reference kind and the
  base paths; the parametric dialect (from the plan) is built by
  ``merlin.xdsl_dialects.targets.factory``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from merlin.common.paths import targets_dir
from .rtl.facts import dialect_plan_path, rtl_facts_path, target_base, target_contract_path

# Default runtime backend per target (was pipeline.DEFAULT_BACKEND). Unknown -> "simulator".
_DEFAULT_BACKEND = {
    "toy_npu": "simulator",
    "saturn": "baremetal",
    "gemmini": "baremetal",
    "muon": "simulator",
}


@dataclass(frozen=True)
class TargetInfo:
    """Resolved identity + locations for one target."""

    name: str
    kind: str                 # "reference" | "generated"
    base: Path
    contract_path: Path
    dialect_plan_path: Path
    facts_path: Path          # rtl facts pin (may not exist for non-RTL targets)
    backend: str

    def load_contract(self) -> dict[str, Any]:
        return yaml.safe_load(self.contract_path.read_text(encoding="utf-8"))

    def load_dialect_plan(self) -> dict[str, Any]:
        return yaml.safe_load(self.dialect_plan_path.read_text(encoding="utf-8"))


def backend_for(name: str) -> str:
    """Default runtime backend for a target."""
    return _DEFAULT_BACKEND.get(name, "simulator")


def resolve(name: str) -> TargetInfo:
    """Resolve a target's identity + paths. ``kind`` is 'reference' if it has a curated
    ``merlin/targets/<name>/`` definition, else 'generated' (paths point under artifacts/targets)."""
    base = target_base(name)
    kind = "reference" if (targets_dir() / name).is_dir() else "generated"
    return TargetInfo(
        name=name, kind=kind, base=base,
        contract_path=target_contract_path(name),
        dialect_plan_path=dialect_plan_path(name),
        facts_path=rtl_facts_path(name),
        backend=backend_for(name))


def list_targets() -> list[str]:
    """Curated reference targets (dirs under merlin/targets/ with a target_contract.yaml)."""
    root = targets_dir()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir()
                  if (p / "contracts" / "target_contract.yaml").is_file())


def load_contract(name: str) -> dict[str, Any]:
    return resolve(name).load_contract()


def load_dialect_plan(name: str) -> dict[str, Any]:
    return resolve(name).load_dialect_plan()
