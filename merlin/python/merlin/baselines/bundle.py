"""Resolve a ``(model, variant)`` to its capture bundle — the shared input every baseline ingests.

Each baseline framework starts from the SAME captured bundle under
``artifacts/recaptures/<model>_<variant>_consistent/`` so the comparison is apples-to-apples:

  * ``model.mlir``            — linalg-on-tensors export (Buddy/TVM-via-relax can ingest directly)
  * ``weights.safetensors``   (+ ``.manifest.json``)  — HF weights + arg-index map
  * ``inputs.npz`` / ``input_order.json`` — the seeded inputs
  * ``golden.npy``            — the torch reference output for correctness gating
  * ``extra.npz``             — registered buffers + lifted constants

The original PyTorch loader (for frameworks that ingest torch, e.g. ExecuTorch export / TVM
from_pytorch) lives OUTSIDE this repo at
``$MERLIN_MODEL2MLIR/workloads/<model>/loader.py`` (default ``/scratch/agustin/projects/model2MLIR``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from merlin.common.artifacts import recaptures_dir
from merlin.dse_guidance import models as _models

# Per-model correctness tolerances, mirroring merlin/tests/rvv/test_vla_models_rvv.py so a baseline
# is gated exactly as our own runtime is. (min_cos, max_rel). Unlisted models use _DEFAULT_TOL.
_DEFAULT_TOL: tuple[float, float] = (0.9999, 1e-3)
TOLERANCES: dict[str, tuple[float, float]] = {
    "rdt2": (0.9999, 1e-3),
    "rdt": (0.9999, 1e-3),
    "groot_n1d7": (0.9999, 1e-2),
    "molmoact": (0.9999, 1e-3),
    "openvla": (0.9999, 1e-3),
    "openvla_oft": (0.9999, 1e-3),
    "bitvla": (0.999, 5e-3),        # BitNet ternary dequant round-off
    "xr0": (0.999, 5e-2),           # sensitive diffusion head
    "smolvla": (0.978, 5e-2),       # bf16 precision amplification in the denoise head
    "pi05": (0.9999, 1e-2),
    "small_llama": (0.9999, 1e-3),
    "tiny_llama": (0.9999, 1e-3),
}

_VARIANTS = ("fp32", "int8", "fp8")


def tolerance(model: str) -> tuple[float, float]:
    """(min_cos, max_rel) gate for a model — same regime as our own RVV tests."""
    return TOLERANCES.get(model, _DEFAULT_TOL)


def model2mlir_root() -> Path:
    """Root of the external model2MLIR repo (PyTorch loaders live here)."""
    return Path(os.environ.get("MERLIN_MODEL2MLIR", "/scratch/agustin/projects/model2MLIR"))


@dataclass
class CaptureBundle:
    model: str
    variant: str
    root: Path

    @property
    def mlir(self) -> Path:
        return self.root / "model.mlir"

    @property
    def weights(self) -> Path:
        return self.root / "weights.safetensors"

    @property
    def weights_manifest(self) -> Path:
        return self.root / "weights.safetensors.manifest.json"

    @property
    def inputs(self) -> Path:
        return self.root / "inputs.npz"

    @property
    def input_order(self) -> Path:
        return self.root / "input_order.json"

    @property
    def golden(self) -> Path:
        return self.root / "golden.npy"

    @property
    def extra(self) -> Path:
        return self.root / "extra.npz"

    @property
    def torch_loader(self) -> Path:
        """PyTorch loader in the external model2MLIR repo (may not exist for every model)."""
        return model2mlir_root() / "workloads" / self.model / "loader.py"

    @property
    def tolerance(self) -> tuple[float, float]:
        return tolerance(self.model)

    def require(self) -> "CaptureBundle":
        """Raise if the essential inputs (mlir + golden) are missing — fail-closed."""
        missing = [p.name for p in (self.mlir, self.golden) if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"capture bundle {self.model}/{self.variant} at {self.root} missing {missing}")
        return self


def _dirname(model: str, variant: str) -> str:
    return f"{model}_{variant}_consistent"


def resolve(model: str, variant: str = "fp32") -> CaptureBundle:
    """Locate the capture bundle for a model+variant under artifacts/recaptures/.

    Does not require the bundle to exist yet (call ``.require()`` to gate); this keeps discovery
    and validation separate so a runner can report a clean ``gap_reason`` if a capture is absent.
    """
    if variant not in _VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {_VARIANTS}")
    root = recaptures_dir() / _dirname(model, variant)
    return CaptureBundle(model=model, variant=variant, root=root)


def available_models() -> list[str]:
    """Base models that actually have a captured bundle on disk (any variant)."""
    return sorted(_models.discover_model_captures().keys())


def known_models() -> list[str]:
    """All base models in the registry (whether or not captured yet)."""
    return sorted(_models.MODEL_ARCH.keys())
