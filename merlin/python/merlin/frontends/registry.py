"""Frontend-adapter registry — ingest any framework's export into a capture bundle.

A model enters Merlin as a :class:`~merlin.baselines.bundle.CaptureBundle` (the framework-neutral
input every backend consumes). *How* those bytes are produced differs per source: a torch / torchAO
model captured by model2MLIR, a ``.gguf`` checkpoint lifted into the ``quant_ext`` dialect, and so
on. Each such source is a **frontend adapter** conforming to one small protocol; this registry lets a
new framework plug in as one entry + one module — the same instance→registry shape the runtime
backends use (:mod:`merlin.runtime.backends.base`).

The adapter surface is deliberately generic (it says nothing about *which* quantization format or
*which* target): an adapter turns a source into a bundle, and the format each weight carries is
described by the target-agnostic :mod:`merlin.common.quant_formats` registry. That keeps the
ingestion tooling format-agnostic — a new quantization format needs no new adapter, only a registry
entry (and, for a genuinely new container, a new adapter).
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from merlin.baselines.bundle import CaptureBundle


@dataclass(frozen=True)
class AdapterInfo:
    name: str
    module: str                 # dotted import path, loaded lazily via get_adapter()
    source_kinds: tuple[str, ...]  # human-readable tags: what this adapter ingests
    summary: str


# One line per adapter; a new frontend framework is one entry here + its module.
_REGISTRY: dict[str, AdapterInfo] = {
    "m2m": AdapterInfo(
        "m2m",
        "merlin.frontends.adapters.m2m",
        ("torch_model", "hf_model_id"),
        "PyTorch / torchAO models captured by model2MLIR (also the HuggingFace path via its loaders).",
    ),
    "gguf": AdapterInfo(
        "gguf",
        "merlin.frontends.adapters.gguf",
        ("gguf_file",),
        "GGUF checkpoints lifted into the quant_ext dialect (Q8_0/Q6_K/Q4_K/MXFP4/NVFP4).",
    ),
}


@runtime_checkable
class FrontendAdapter(Protocol):
    """The shape every frontend-adapter module exposes (module-level functions/attrs)."""

    NAME: str

    def can_handle(self, source: Any) -> bool: ...
    def ingest(self, source: Any, *, model: str, variant: str, **kw: Any) -> CaptureBundle: ...


def list_adapters() -> list[str]:
    return sorted(_REGISTRY)


def info(name: str) -> AdapterInfo:
    return _REGISTRY[name]


def get_adapter(name: str):
    """Lazily import + return the adapter module for ``name`` (raises KeyError if unregistered)."""
    return importlib.import_module(_REGISTRY[name].module)


def for_source(source: Any):
    """Return the first registered adapter module whose ``can_handle(source)`` is true.

    Adapters are consulted most-specific first (``gguf`` before the catch-all ``m2m``). Raises
    ``LookupError`` if none match.
    """
    for name in sorted(_REGISTRY, key=lambda n: 0 if n != "m2m" else 1):
        adapter = get_adapter(name)
        if adapter.can_handle(source):
            return adapter
    raise LookupError(f"no frontend adapter handles source {source!r}")
