"""Compatibility resolver for unit-owned code during package extraction.

The target package owns the implementation. This module keeps legacy import paths
working while callers migrate to the provider-chain extension API.
"""

from __future__ import annotations


def load_unit_module(target: str, filename: str):
    from .plugins import load_module, resolve_support

    selected = resolve_support(target)
    return load_module(selected.base, filename, package_name=target)
