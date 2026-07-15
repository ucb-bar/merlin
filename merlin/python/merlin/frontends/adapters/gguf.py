"""GGUF frontend adapter — lift a ``.gguf`` checkpoint into the quant_ext dialect.

GGUF is the source of the INT8 (Q8_0), FP6-analogous (Q6_K), and FP4-analogous (Q4_K) weights in the
model download matrix, plus true MXFP4/NVFP4. This adapter reads a GGUF with the vendored gguf-py
``GGUFReader`` and emits the same linalg-on-tensors + ``quant_ext``-typed bundle the torch path
produces, so GGUF-quantized models flow through the identical Merlin pipeline as torchAO ones. Each
GGML quantization type maps onto a canonical :mod:`merlin.common.quant_formats` entry
(via ``QuantFormat.ggml_type``).

The reader/writer implementation lands with the P1 vertical slice; the adapter is registered now so
the frontend registry is complete and its contract is exercised.
"""
from __future__ import annotations

from typing import Any

from merlin.baselines.bundle import CaptureBundle

NAME = "gguf"


def can_handle(source: Any) -> bool:
    """True for a GGUF checkpoint path."""
    return str(source).endswith(".gguf")


def ingest(source: Any, *, model: str, variant: str, **_kw: Any) -> CaptureBundle:
    raise NotImplementedError(
        "GGUF -> quant_ext ingestion lands in the P1 vertical slice (merlin.frontends.adapters.gguf). "
        f"Source {source!r} recognised but the reader is not wired yet."
    )
