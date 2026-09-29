"""Explicitly selected, installed target binding for native compilation.

The binding owns target facts and physical emission. Merlin owns selection,
extraction, allocation, and checking; an entry point is never a second engine.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path
from typing import Protocol

from .model import KernelRequest
from .search import SearchLimits
from .snapshot import NativeSnapshot, NativeTargetProfile


class NativeTargetBinding(Protocol):
    @staticmethod
    def profile() -> NativeTargetProfile: ...

    @staticmethod
    def compile(
        snapshot: NativeSnapshot,
        request: KernelRequest,
        *,
        fixed_inputs: dict[str, int],
        fixed_outputs: tuple[int, ...],
        target_source: Path,
        destination: Path,
        limits: SearchLimits,
    ) -> dict[str, object]: ...


def load_native_target_binding(name: str) -> NativeTargetBinding:
    """Load one named installed provider; never infer a target from a request."""
    if not name or not name.isidentifier():
        raise ValueError("native target support needs one explicit entry point name")
    matches = [point for point in entry_points(group="merlin.native_target_compilers") if point.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected one installed native target support named {name!r}, found {len(matches)}")
    binding = matches[0].load()
    if not callable(getattr(binding, "profile", None)) or not callable(getattr(binding, "compile", None)):
        raise TypeError("native target support lacks profile or compile interface")
    return binding
