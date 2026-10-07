"""A target's whole-model driver modules, loaded from the EXPLICITLY selected support provider.

The driver (program generator, kernel binder, open-model dispatch) is target-owned and ships in the
OOT support package selected on ``MERLIN_TARGET_PATH``; no in-tree copy remains. A test of it skips
when no provider for the target is selected -- absence is not a passing qualification.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def require_support(target: str) -> Path:
    """The root of ``target``'s explicitly selected support provider, or skip when none is selected.

    Only ABSENCE skips: a provider that IS selected and then fails to load still fails the test that
    reaches it, so a backend broken by a refactor never reads as a green skip."""
    from merlin.targetgen import target_registry

    selected = target_registry.explicit_targets().get(target)
    if selected is None:
        pytest.skip(f"requires explicit {target} support on MERLIN_TARGET_PATH", allow_module_level=True)
    return Path(selected)


def missing_support(*targets: str) -> list[str]:
    """The targets among ``targets`` with no support provider selected on ``MERLIN_TARGET_PATH``."""
    from merlin.targetgen import target_registry

    selected = target_registry.explicit_targets()
    return [target for target in targets if target not in selected]


def requires_support(*targets: str):
    """A ``skipif`` marker for a test or module whose subject is the named targets' selected support.

    Evaluated when the test module is collected. Only ABSENCE skips, as in :func:`require_support`."""
    absent = missing_support(*targets)
    return pytest.mark.skipif(
        bool(absent), reason=f"requires explicit {', '.join(absent)} support on MERLIN_TARGET_PATH"
    )


def driver_file(target: str, name: str) -> Path:
    """``<selected support>/whole_model/<name>`` for ``target``, or skip when none is selected."""
    from merlin.targetgen import target_registry

    require_support(target)
    path = Path(target_registry.resolve(target).base) / "whole_model" / name
    if not path.is_file():
        pytest.skip(f"the selected {target} support ships no whole_model/{name}", allow_module_level=True)
    return path


def load(target: str, name: str, *, module_name: str | None = None):
    """The driver module ``name`` of the selected provider, imported once under ``module_name``."""
    path = driver_file(target, name)
    key = module_name or f"selected_{target}_{path.stem}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def software_spec_provider(target: str) -> str | None:
    """The support provider ``target``'s phase-0 recipe takes its software spec FROM, or ``None``.

    A recipe may name its software spec as a resource of a selected support provider rather than a
    path in this repo; loading such a target's experiment then needs that provider selected. Read off
    the recipe's own declaration, so no target is named here.
    """
    import yaml
    from merlin_experiments.phase0.declarations import for_target

    try:
        recipe = for_target(target).recipe
    except Exception:  # noqa: BLE001 - a target with no phase-0 declaration has no recipe to read
        return None
    if recipe is None or not Path(recipe).is_file():
        return None
    spec = (yaml.safe_load(Path(recipe).read_text(encoding="utf-8")) or {}).get("software_spec")
    return str(spec["provider"]) if isinstance(spec, dict) and spec.get("provider") else None


def requires_recipe_support(target: str):
    """``requires_support`` for the provider ``target``'s recipe draws its software spec from, if any."""
    provider = software_spec_provider(target)
    return requires_support(provider) if provider else pytest.mark.skipif(False, reason="")
