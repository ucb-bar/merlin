"""Deprecated import path for package-owned opu_isa.py.

The selected provider owns the implementation; new callers load it directly.
"""

from __future__ import annotations

from merlin.targetgen.unit_module import load_unit_module

_TARGET = "kodiak"  # target-ok: historical import path for this unit is fixed to its owning provider
_FILENAME = "opu_isa.py"


def __getattr__(name: str):
    return getattr(load_unit_module(_TARGET, _FILENAME), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(load_unit_module(_TARGET, _FILENAME))))
