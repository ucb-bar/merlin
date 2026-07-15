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

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from merlin.common.paths import targets_dir
from .rtl.facts import dialect_plan_path, rtl_facts_path, target_base, target_contract_path

# Colon/os.pathsep-separated list of out-of-tree target package roots (or dirs containing them). An
# OOT target repo (e.g. a generated radiance-mlir under out/artifacts/targets, later externalized)
# ships its own contracts/target_contract.yaml (+ compute_units) + dialect + lowering entry-point;
# Merlin discovers and plugs it in without any target-specific code committed here.
_ENV_TARGET_PATH = "MERLIN_TARGET_PATH"

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
    kind: str                 # "reference" | "generated" | "external"
    base: Path
    contract_path: Path
    dialect_plan_path: Path
    facts_path: Path          # rtl facts pin (may not exist for non-RTL targets)
    backend: str
    external_root: Path | None = None   # OOT package root, when kind == "external"

    def load_contract(self) -> dict[str, Any]:
        return yaml.safe_load(self.contract_path.read_text(encoding="utf-8"))

    def load_dialect_plan(self) -> dict[str, Any]:
        return yaml.safe_load(self.dialect_plan_path.read_text(encoding="utf-8"))

    def plugin(self) -> dict[str, Any]:
        """The out-of-tree ``plugin`` block from the contract (dialect + lowering entry-points).

        Merlin reads (never executes) these references; importing the dialect / calling the lowering
        is the caller's job, guarded — so nothing target-specific runs at resolution time. The OOT
        package root is injected as ``path`` so a caller can put it on ``sys.path``.
        """
        block = dict(self.load_contract().get("plugin", {}))
        if self.external_root is not None:
            block.setdefault("path", str(self.external_root))
        return block


def backend_for(name: str) -> str:
    """Default runtime backend for a target."""
    return _DEFAULT_BACKEND.get(name, "simulator")


def _is_target_root(p: Path) -> bool:
    return (p / "contracts" / "target_contract.yaml").is_file()


def _target_name(root: Path) -> str:
    """The target's declared name (contract ``name``), falling back to the directory name."""
    try:
        doc = yaml.safe_load((root / "contracts" / "target_contract.yaml").read_text(encoding="utf-8"))
        if isinstance(doc, dict) and doc.get("name"):
            return str(doc["name"])
    except (OSError, yaml.YAMLError):
        pass
    return root.name


def external_targets() -> dict[str, Path]:
    """Discover out-of-tree targets from ``MERLIN_TARGET_PATH`` -> ``{name: package_root}``.

    Each path entry is either a target package root (contains ``contracts/target_contract.yaml``) or a
    directory whose immediate children are such roots. Empty/unset env -> no external targets.
    """
    found: dict[str, Path] = {}
    raw = os.environ.get(_ENV_TARGET_PATH, "")
    for entry in raw.split(os.pathsep):
        if not entry:
            continue
        p = Path(entry)
        roots = [p] if _is_target_root(p) else ([c for c in sorted(p.iterdir()) if _is_target_root(c)]
                                                if p.is_dir() else [])
        for root in roots:
            found[_target_name(root)] = root
    return found


def _resolve_external(name: str, root: Path) -> TargetInfo:
    contracts = root / "contracts"
    return TargetInfo(
        name=name, kind="external", base=root,
        contract_path=contracts / "target_contract.yaml",
        dialect_plan_path=contracts / "dialect_plan.yaml",
        facts_path=contracts / "rtl_facts" / "facts.json",
        backend=str((yaml.safe_load((contracts / "target_contract.yaml").read_text(encoding="utf-8"))
                     or {}).get("runtime", {}).get("default_backend") or backend_for(name)),
        external_root=root)


def resolve(name: str) -> TargetInfo:
    """Resolve a target's identity + paths.

    ``kind`` is 'external' if ``name`` is discovered under ``MERLIN_TARGET_PATH`` (an out-of-tree
    package), else 'reference' if it has a curated ``merlin/targets/<name>/`` definition, else
    'generated' (paths point under artifacts/targets). External discovery is checked first so a target
    repo can be plugged in without a curated in-tree copy."""
    external = external_targets()
    if name in external:
        return _resolve_external(name, external[name])
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


def all_targets() -> list[str]:
    """Curated reference targets plus any discovered out-of-tree (MERLIN_TARGET_PATH) targets."""
    return sorted(set(list_targets()) | set(external_targets()))


def load_contract(name: str) -> dict[str, Any]:
    return resolve(name).load_contract()


def load_dialect_plan(name: str) -> dict[str, Any]:
    return resolve(name).load_dialect_plan()
