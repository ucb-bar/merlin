"""model2MLIR frontend adapter — the reference adapter.

Ingests a PyTorch / torchAO model (by model name, resolved through model2MLIR's per-model
``workloads/<model>/loader.py``, including HuggingFace checkpoints) into a
:class:`~merlin.baselines.bundle.CaptureBundle`. The heavy capture (torch export → linalg-on-tensors
MLIR + weights + golden) runs in the model2MLIR venv; this adapter locates an already-captured bundle
and, when asked, drives that capture. It is the catch-all adapter — anything that is not a ``.gguf``
file is handled here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from merlin.baselines import bundle as _bundle

NAME = "m2m"


def can_handle(source: Any) -> bool:
    """True for a torch/HF model source: a model name/id, not a ``.gguf`` file path."""
    return not str(source).endswith(".gguf")


def resolve(model: str, variant: str = "fp32") -> _bundle.CaptureBundle:
    """Locate the capture bundle for ``(model, variant)`` (does not require it to exist)."""
    return _bundle.resolve(model, variant)


def ingest(source: Any, *, model: str, variant: str = "fp32", require: bool = True, **_kw: Any) -> _bundle.CaptureBundle:
    """Return the capture bundle for ``model``/``variant``.

    ``source`` is the model name/id (kept for a uniform adapter signature). With ``require`` the
    bundle's essential inputs (mlir + golden) must already be present — driving a fresh capture (the
    model2MLIR venv subprocess + downloads) is the job of the capture driver used by the P1 slice, so
    this fails closed with a clear message rather than silently returning an empty bundle.
    """
    b = resolve(model, variant)
    if require:
        b.require()
    return b


def torch_loader(model: str) -> Path:
    """Path to the model's PyTorch loader in the external model2MLIR checkout."""
    return _bundle.model2mlir_root() / "workloads" / model / "loader.py"
