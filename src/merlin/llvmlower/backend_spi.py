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
    )
    missing = [name for name in required if not callable(getattr(backend, name, None))]
    if missing:
        raise BackendAPIError(f"{target}: {owner} codegen lacks {missing}")
    return backend


def load_vector_codegen(target: str) -> Any:
    """Return the one version-compatible vector backend in ``target``'s provider chain."""
    from merlin.targetgen.plugins import load_declared_chain

    return _checked(target, load_declared_chain(target, "rvv_codegen"))
