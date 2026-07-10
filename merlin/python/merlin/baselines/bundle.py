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
``$MERLIN_MODEL2MLIR/workloads/<model>/loader.py`` (default ``/path/to/model2MLIR``).
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
    return Path(os.environ.get("MERLIN_MODEL2MLIR", "/path/to/model2MLIR"))


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


# K1-runnable set: models whose int8 footprint fits the 3.8 GB board (~3.4 GB usable). The three
# 7B-class VLAs (openvla, molmoact, pi05) are RAM-infeasible for whole-model on-board runs even at int8
# (fp32 embeddings dominate) — attempt+RAM-gap, never a false fit. From the full-fidelity recapture.
K1_RUNNABLE: frozenset[str] = frozenset(
    {"tiny_llama", "smolvla", "bitvla", "groot_n1d7", "rdt", "rdt2", "xr0", "small_llama"})
K1_RAM_INFEASIBLE: frozenset[str] = frozenset({"openvla", "molmoact", "pi05"})

# Full-fidelity capture env (the exact loader settings the recapture used to build the REAL/native
# architecture, dropping the truncation defaults). A Phase-2 arm that live-loads the torch model via
# the m2m loader MUST set these first so it ingests the IDENTICAL model the golden was computed on.
FULL_FIDELITY_ENV: dict[str, dict[str, str]] = {
    "tiny_llama": {},                                   # real TinyLlama-1.1B, full 22 layers (no truncation)
    "smolvla": {},                                      # real smolvla_base bf16, native 16 layers
    "bitvla": {"BITVLA_LLM_LAYERS": "30"},              # real BitNet depth (hidden loader-fixed at 256)
    "groot_n1d7": {},                                   # native 16 DiT layers (no M2M_GROOT_LAYERS truncation)
    "rdt": {"M2M_RDT_DEPTH": "28"},                     # real RDT-1B depth
    "rdt2": {"M2M_RDT2_DEPTH": "14"},                   # native depth
    "xr0": {"XR0_DIT_LAYERS": "16"},                    # real DiT-head depth
    "small_llama": {},                                  # synthetic toy reference (no real counterpart)
    "openvla": {},                                      # real Llama-2-7B+ViT config (RAM-infeasible on K1)
    "molmoact": {"M2M_MOLMOACT_LAYERS": "48", "M2M_MOLMOACT_VOCAB": "152064"},  # real 7B (RAM-infeasible)
    "pi05": {},                                         # full PaliGemma+expert 3.6B (RAM-infeasible)
}


def full_env(model: str) -> dict[str, str]:
    """Loader env that reproduces the full-fidelity model the golden was captured on (may be empty)."""
    return dict(FULL_FIDELITY_ENV.get(model, {}))


# Models whose torch loader instantiates RANDOM weights (no pretrained checkpoint) AND whose capture
# bundle ships no ``weights.safetensors``. Their captured golden came from a *different* seeded
# instantiation, so a re-exported fp32 run can never reproduce it — even when the lowering is exact.
#
# MEASURED evidence (do not add a model here without it — an unjustified entry silently WEAKENS that
# model's correctness gate):
#   bitvla/fp32  cos=0.013323 vs captured golden, but 0.9999999999987 (rel 1.6e-06) vs compute_golden
#   openvla/fp32 cos=0.009043 vs captured golden; TVM's host-VM lowering cos=1.0 on the same model
#
# Deliberately EXCLUDED despite being random-init-ish: ``rdt2`` — its recomputed golden reproduces the
# capture exactly (norm 11.4439 both), so its captured golden IS reachable and must stay the gate.
# ``rdt`` is excluded too (recompute 53.24 vs capture 52.88 — close but unresolved; measure first).
# The llama family (tiny_llama/small_llama) loads real checkpoints and passes against captured goldens.
#
# For these models the correctness gate becomes LOWERING-EXACTNESS — "the framework reproduces eager
# torch for THIS instantiation" — NOT a semantic match against a trained model. Every such cell must
# say so; see ``lowering_exactness_note``. This mirrors the TVM arm (openvla cos=1.0) and the int8
# path, which already recomputes its reference for the same reason.
RANDOM_INIT_GOLDEN_UNREPRODUCIBLE: frozenset[str] = frozenset({"bitvla", "openvla"})


def golden_unreproducible(model: str) -> bool:
    """True if ``model``'s captured golden cannot be reproduced by re-instantiating its loader."""
    return model in RANDOM_INIT_GOLDEN_UNREPRODUCIBLE


def lowering_exactness_note(model: str) -> str:
    """The honesty label a cell MUST carry when it is gated on lowering-exactness, not semantics."""
    return (f" random-init({model}): gate=LOWERING-EXACTNESS (framework vs eager-torch on THIS seeded "
            "instantiation), NOT a semantic match — the captured golden's weights are unrecoverable "
            "(bundle ships no weights.safetensors), so cos measures lowering fidelity only.")


def resolve(model: str, variant: str = "fp32") -> CaptureBundle:
    """Locate the capture bundle for a model+variant under artifacts/recaptures/.

    Prefers the full-fidelity recapture (``<model>_<variant>_full``, real/native architecture) over the
    older truncated ``<model>_<variant>_consistent`` bundle when present. Does not require the bundle to
    exist yet (call ``.require()`` to gate) so a runner can report a clean ``gap_reason`` if it's absent.
    """
    if variant not in _VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {_VARIANTS}")
    rd = recaptures_dir()
    full = rd / f"{model}_{variant}_full"
    root = full if full.is_dir() else rd / _dirname(model, variant)
    return CaptureBundle(model=model, variant=variant, root=root)


def available_models() -> list[str]:
    """Base models that actually have a captured bundle on disk (any variant)."""
    return sorted(_models.discover_model_captures().keys())


def known_models() -> list[str]:
    """All base models in the registry (whether or not captured yet)."""
    return sorted(_models.MODEL_ARCH.keys())
