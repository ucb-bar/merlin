"""Versioned executable codegen boundary for layered support providers.

The scalar compiler does not resolve a vector provider. A selected vector target loads its
codegen from the provider that declared it, even when a chip package extends that provider.
"""

from __future__ import annotations

from typing import Any

API_VERSION = 1


class BackendAPIError(ValueError):
    """A selected provider does not implement the executable backend interface."""


def _checked(target: str, entries: tuple[tuple[str, Any], ...]) -> Any:
    if len(entries) != 1:
        owners = [owner for owner, _ in entries]
        raise BackendAPIError(f"{target}: expected one vector codegen provider, found {owners}")
    owner, backend = entries[0]
    if getattr(backend, "API_VERSION", None) != API_VERSION:
        raise BackendAPIError(
            f"{target}: {owner} codegen API version {getattr(backend, 'API_VERSION', None)!r}; "
            f"Merlin requires {API_VERSION}"
        )
    required = (
        "build_pipeline",
        "baseline_schedule_text",
        "parallel_transform_schedule",
        "vec_pre_schedule",
        "normalize_features",
        "apply_schedule",
        "apply_pipeline",
        "apply_cflags",
        "vector_cflags",
        "schedule_path",
        "prepare_build_bundle",
        "spike_cflags",
        "spike_harness_cflags",
        "accum_microkernel_marker",
        "accum_microkernel_runner_source",
        "accum_microkernel_rewrite_source",
        "vec_noncontraction_shape",
        "im2col_pack_api",
        "weight_panel_api",
    )
    missing = [name for name in required if not callable(getattr(backend, name, None))]
    if missing:
        raise BackendAPIError(f"{target}: {owner} codegen lacks {missing}")
    return backend


def load_vector_codegen(target: str) -> Any:
    """Return the one version-compatible vector backend in ``target``'s provider chain."""
    from merlin.targetgen.plugins import load_declared_chain

    from .feature_registry import ImprFeature

    owner_backend = load_declared_chain(target, "rvv_codegen")
    backend = _checked(target, owner_backend)
    definitions: list[ImprFeature] = []
    for owner, extension in load_declared_chain(target, "extension"):
        version = getattr(extension, "API_VERSION", None)
        if version != API_VERSION:
            raise BackendAPIError(f"{target}: {owner} extension API version {version!r}; need {API_VERSION}")
        declare = getattr(extension, "feature_definitions", None)
        if declare is None:
            continue
        if not callable(declare):
            raise BackendAPIError(f"{target}: {owner} extension feature_definitions is not callable")
        selected = declare()
        if not isinstance(selected, tuple) or not all(isinstance(feature, ImprFeature) for feature in selected):
            raise BackendAPIError(f"{target}: {owner} extension feature_definitions must return ImprFeature tuple")
        definitions.extend(selected)
    bind = getattr(backend, "bind", None)
    if not callable(bind):
        raise BackendAPIError(f"{target}: vector provider lacks bind(extension_features)")
    bound = bind(tuple(definitions))
    return _checked(target, ((owner_backend[0][0], bound),))
