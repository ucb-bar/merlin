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
    if not callable(getattr(backend, "build_pipeline", None)):
        raise BackendAPIError(f"{target}: {owner} codegen lacks build_pipeline")
    return backend


def load_vector_codegen(target: str) -> Any:
    """Return the one version-compatible vector backend in ``target``'s provider chain."""
    from merlin.targetgen.plugins import load_declared_chain

    return _checked(target, load_declared_chain(target, "rvv_codegen"))


def optional_vector_codegen(target: str) -> Any | None:
    """Migration adapter while the pinned base still has a contract but no codegen.

    A missing declaration alone retains the old in-tree RVV path. An invalid declaration,
    wrong API version, or multiple providers always fails closed. Remove this adapter when
    Merlin pins the executable RVV provider.
    """
    from merlin.targetgen.plugins import load_declared_chain

    entries = load_declared_chain(target, "rvv_codegen")
    if not entries:
        return None
    return _checked(target, entries)
