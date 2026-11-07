"""Compatibility entrypoint for Merlin's pinned ``plugin.rvv_codegen`` declaration."""

from .rvv_codegen import backend as _implementation

API_VERSION = _implementation.API_VERSION


def __getattr__(name: str):
    return getattr(_implementation, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_implementation)))
