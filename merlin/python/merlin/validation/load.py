"""Load the five plan artifacts from a generated target repo's ``contracts/`` dir.

The five plans are the synchronized outputs of TargetGen synthesis:

  target_contract.yaml       what the hardware/runtime exposes
  dialect_plan.yaml          which dialect ops/types/lowerings to scaffold
  runtime_adapter_plan.yaml  how the target implements the Merlin runtime ABI
  zephyr_plan.yaml           the Zephyr backend scaffold to generate
  llvm_extension_plan.yaml   whether/how LLVM changes are needed

Each maps to a schema of the same stem under ``merlin/schemas/``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..common.yaml import load_yaml

# plan stem -> (filename, schema name)
PLAN_FILES: dict[str, tuple[str, str]] = {
    "target_contract": ("target_contract.yaml", "target_contract"),
    "dialect_plan": ("dialect_plan.yaml", "dialect_plan"),
    "runtime_adapter_plan": ("runtime_adapter_plan.yaml", "runtime_adapter_plan"),
    "zephyr_plan": ("zephyr_plan.yaml", "zephyr_plan"),
    "llvm_extension_plan": ("llvm_extension_plan.yaml", "llvm_extension_plan"),
}

# Plans a target is NOT required to ship. The target DIALECT is a design decision the agent owns (it
# writes its own out-of-tree MLIR backend, verified against the CIRCT oracle), so a dialect_plan.yaml is
# optional: validated when present, but its absence is not a structural error.
OPTIONAL_PLANS: frozenset[str] = frozenset({"dialect_plan"})

# Plans every generated target repo must still ship.
REQUIRED_PLANS: tuple[str, ...] = tuple(p for p in PLAN_FILES if p not in OPTIONAL_PLANS)


def contracts_dir(target_repo: str | Path) -> Path:
    """Return the ``contracts/`` directory inside a generated target repo."""
    return Path(target_repo) / "contracts"


def load_plan(target_repo: str | Path, plan: str) -> Any:
    """Load a single plan (by stem, e.g. ``"target_contract"``) from a target repo.

    Raises ``KeyError`` for an unknown plan name and ``FileNotFoundError`` if the file is
    absent.
    """
    if plan not in PLAN_FILES:
        raise KeyError(f"unknown plan '{plan}'; known: {sorted(PLAN_FILES)}")
    filename, _ = PLAN_FILES[plan]
    path = contracts_dir(target_repo) / filename
    if not path.is_file():
        raise FileNotFoundError(f"plan not found: {path}")
    return load_yaml(path)


def load_all_plans(target_repo: str | Path) -> dict[str, Any]:
    """Load every plan present in a target repo's ``contracts/`` dir.

    Missing files are skipped (the caller can detect them by absence). Returns a mapping of
    plan stem -> parsed YAML.
    """
    out: dict[str, Any] = {}
    for plan in PLAN_FILES:
        try:
            out[plan] = load_plan(target_repo, plan)
        except FileNotFoundError:
            continue
    return out
