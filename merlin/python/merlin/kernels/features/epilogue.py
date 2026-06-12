"""Epilogue-fusion features.

Decision recorded: *is a bias/requant/activation epilogue fused before the result is
committed to memory*, and a coarse list of which epilogue kinds appear. This seeds the
``accumulator_commit`` (commit-after-epilogue) abstraction.
"""
from __future__ import annotations

from merlin.kernels.types import NormalizedKernel

# Coarse epilogue-kind detection (presence only, order-independent).
_KIND_MARKERS = {
    "requant": ("vfcvt", "vfncvt", "vncvt", "requant", "rescale", "acc_scale", "config_ld"),
    "clamp": ("vfmax", "vfmin", "clamp", "_mm", "vmaxq", "vminq"),
    "bias": ("bias",),
    "relu": ("relu", "RELU", "max(0", "NO_ACTIVATION"),
}


def extract_epilogue(nk: NormalizedKernel, fired: dict[str, list[str]]) -> dict:
    fused = "epilogue_before_commit" in fired
    text = nk.raw_text
    kinds: list[str] = []
    for kind, needles in _KIND_MARKERS.items():
        if kind == "relu" and "NO_ACTIVATION" in text:
            continue  # explicit no-activation -> not a relu epilogue
        if any(n in text for n in needles if n != "NO_ACTIVATION"):
            kinds.append(kind)
    return {"epilogue_fusion": bool(fused), "epilogue_kind": kinds if fused else []}
