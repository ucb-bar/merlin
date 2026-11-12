"""Lowering components and named features implemented by the selected vector support provider.

Register-block tables, panel-packed weight layouts, fused requantization epilogues, register-group
widths and the accumulator micro-kernel runner are owned by a support provider, not by Merlin.
Merlin reaches each component through the selected provider's ``<component>_api()`` accessor and
resolves the features they implement from the provider's ``feature_definitions()``; it imports no
provider module by name. Selection is the rule :func:`merlin.mining.from_strategy._codegen` applies.

With no provider selected these components and features are unavailable. A caller that needs one
gets :class:`ProviderUnavailable` naming what it asked for; a caller that only needs to know whether
a provider feature could be present uses the ``optional_*`` forms, which return ``None`` only when
nothing is selected and still raise when a selected provider cannot be loaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from merlin.compiler.selection.features import ImprFeature


class ProviderUnavailable(LookupError):
    """A provider-implemented component or feature was requested and no vector provider is selected."""


def _selected() -> bool:
    from merlin.mining.from_strategy import _codegen_selected

    return _codegen_selected()


def provider(*, purpose: str) -> Any:
    """The selected vector support provider; ``purpose`` names the request in the refusal."""
    if not _selected():
        raise ProviderUnavailable(
            f"{purpose} is implemented by a vector support provider and none is selected: name one on "
            "MERLIN_TARGET_PATH (a provider declaring plugin.rvv_codegen) or install its backend package"
        )
    from merlin.mining.from_strategy import _codegen

    return _codegen()


def optional_provider() -> Any | None:
    """The selected provider, or ``None`` exactly when no provider is selected."""
    if not _selected():
        return None
    from merlin.mining.from_strategy import _codegen

    return _codegen()


def _component_of(selected: Any, name: str) -> Any:
    accessor = getattr(selected, f"{name}_api", None)
    if not callable(accessor):
        raise ProviderUnavailable(f"the selected vector support provider implements no {name!r} component")
    return accessor()


def component(name: str) -> Any:
    """The selected provider's implementation module for ``name`` (its ``<name>_api()``)."""
    return _component_of(provider(purpose=f"the {name!r} component"), name)


def optional_component(name: str) -> Any | None:
    """:func:`component`, or ``None`` exactly when no provider is selected."""
    selected = optional_provider()
    return None if selected is None else _component_of(selected, name)


def abi_bundle(model_dir: str | Path, work: str | Path, prepared: str | Path) -> tuple[Path, dict | None]:
    """The bundle an ABI table must be built from once ``work`` has been prepared.

    The selected provider's ``weight_panel`` component may have repacked weights during preparation;
    its ``abi_bundle`` returns the matching bundle and a report. With no provider selected nothing
    was packed, so the bundle is ``model_dir`` and there is no report.
    """
    panel = optional_component("weight_panel")
    return (Path(model_dir), None) if panel is None else panel.abi_bundle(model_dir, work, prepared)


def guard_abi_bundle(model_dir: str | Path, out_dir: str | Path) -> None:
    """Refuse a stock bundle for a build whose provider packed its weights (see :func:`abi_bundle`)."""
    panel = optional_component("weight_panel")
    if panel is not None:
        panel.guard_planned_pack(model_dir, out_dir)


_DEFINITIONS: dict[int, tuple[Any, dict[str, ImprFeature]]] = {}


def _definitions(selected: Any) -> dict[str, ImprFeature]:
    cached = _DEFINITIONS.get(id(selected))
    if cached is not None and cached[0] is selected:
        return cached[1]
    declare = getattr(selected, "feature_definitions", None)
    features = tuple(declare()) if callable(declare) else ()
    by_name: dict[str, ImprFeature] = {}
    for feature in features:
        if not isinstance(feature, ImprFeature):
            raise TypeError(f"provider feature definition must be ImprFeature, got {type(feature).__name__}")
        if feature.name in by_name:
            raise ValueError(f"provider declares feature {feature.name!r} twice")
        by_name[feature.name] = feature
    _DEFINITIONS[id(selected)] = (selected, by_name)
    return by_name


def feature_names() -> frozenset[str]:
    """Names the selected provider defines; empty when no provider is selected."""
    selected = optional_provider()
    return frozenset() if selected is None else frozenset(_definitions(selected))


def feature_definition(name: str) -> ImprFeature:
    """The selected provider's declaration of ``name``.

    Raises :class:`KeyError` when the selected provider defines no such feature, and
    :class:`ProviderUnavailable` (also a ``LookupError``) when no provider is selected to ask.
    """
    definitions = _definitions(provider(purpose=f"feature {name!r}"))
    if name not in definitions:
        raise KeyError(name)
    return definitions[name]
