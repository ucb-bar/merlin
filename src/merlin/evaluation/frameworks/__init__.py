"""Independent external-framework builds, execution and comparison records.

Shared capture, toolchain and instruction-audit helpers retain their own owners.
Framework packages and external tools are loaded only by selected runners.
"""

from __future__ import annotations

__all__ = [
    "FRAMEWORKS",
    "REGIONS",
    "BaselineResult",
    "RegionProfile",
    "ScalarFallback",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    from importlib import import_module

    contract = import_module("merlin.evaluation.frameworks.contract")
    return getattr(contract, name)
