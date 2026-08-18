"""SpectFormer-Ti (vanilla architecture) image classifier -> MLIR.

    python workloads/capture.py spectformer --formats fp32,int8
    <venv>/bin/python workloads/capture_consistent.py spectformer int8 <bundle_dir>

Capture unit: ONE forward of the whole classifier on a 224x224 image. The interesting part is
the spectral half: with ``alpha=4``, blocks 0-3 are SpectralGatingNetwork blocks
(``rfft2`` -> learned complex gate -> ``irfft2`` on the 14x14 token grid) and blocks 4-11 are
ordinary multi-head attention. Nothing about the model is rewritten for capture -- the FFT is
lowered by ``m2m.ir.decompositions`` into real DFT contractions.

Env:
    SPECTFORMER_CKPT      path to a trained checkpoint (default: the SpectFormer-Ti checkpoint
                          below if it is readable; otherwise random init, reported on stderr)
    M2M_SPECTFORMER_DEPTH truncate to N blocks (fast smoke; default: 12, the real depth)
    M2M_SPECTFORMER_DIM   embedding dim (default: 256, SpectFormer-Ti)

Upstream: https://github.com/badripatro/SpectFormers  (vanilla_architecture/spectformer.py)
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import torch
from torch import nn

#: SpectFormer-Ti as published: 224/16 -> a 14x14 token grid, embed 256, depth 12.
_IMG, _PATCH, _DIM, _DEPTH, _HEADS = 224, 16, 256, 12, 4

_UPSTREAM = Path(
    os.environ.get("SPECTFORMER_DIR", "<path>")
) / "vanilla_architecture"

#: A trained SpectFormer-Ti (300 epochs, 73.14% top-1 / 91.56% top-5 ImageNet, 9,158,376
#: params). Only used when readable -- a missing checkpoint degrades to random init, which
#: still exercises the full op set but makes the golden a lowering reference, not an accuracy
#: one. Never silently: the fallback prints.
_DEFAULT_CKPT = ("<path>"
                 "logs/spectformer-tiny/checkpoint_best.pth")


def _import_upstream():
    """Import the upstream module, stubbing one dead import it opens with.

    ``spectformer.py`` starts with ``from numpy.lib.arraypad import pad`` -- a path numpy
    removed, and a name the file never uses. Stub it rather than editing the clone, so the
    checkout stays a faithful copy of what upstream publishes.
    """
    if "numpy.lib.arraypad" not in sys.modules:
        import numpy as _np

        shim = types.ModuleType("numpy.lib.arraypad")
        shim.pad = _np.pad
        sys.modules["numpy.lib.arraypad"] = shim
    if str(_UPSTREAM) not in sys.path:
        sys.path.insert(0, str(_UPSTREAM))
    import spectformer  # type: ignore

    return spectformer


def _set_heads(net: nn.Module, dim: int, heads: int) -> None:
    """Retarget the attention head count.

    Upstream ``Block_attention.__init__`` hardcodes ``num_heads = 6`` (its comment says "4 for
    tiny, 6 for small and 12 for base") and ``SpectFormer.__init__`` takes no ``num_heads``, so
    the Ti config does not run as published: 256 is not divisible by 6. ``num_heads`` changes no
    parameter shape -- only the qkv reshape and the qk scale -- so setting both here is exact
    and keeps a trained Ti checkpoint loadable.
    """
    for blk in net.blocks:
        attn = getattr(blk, "attn", None)
        if attn is not None:
            attn.num_heads = heads
            attn.scale = (dim // heads) ** -0.5


def get_model_and_inputs() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    sf = _import_upstream()
    dim = int(os.environ.get("M2M_SPECTFORMER_DIM", _DIM))
    depth = int(os.environ.get("M2M_SPECTFORMER_DEPTH", _DEPTH))
    net = sf.SpectFormer(img_size=_IMG, patch_size=_PATCH, embed_dim=dim, depth=depth,
                         num_classes=1000)
    _set_heads(net, dim, _HEADS)

    ckpt = os.environ.get("SPECTFORMER_CKPT", _DEFAULT_CKPT)
    if ckpt and Path(ckpt).is_file() and dim == _DIM and depth == _DEPTH:
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = blob.get("model", blob) if isinstance(blob, dict) else blob
        missing, unexpected = net.load_state_dict(state, strict=False)
        print(f"[spectformer] loaded {ckpt} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})", file=sys.stderr)
        if missing or unexpected:
            print(f"[spectformer] WARNING: state_dict mismatch — missing={missing[:4]} "
                  f"unexpected={unexpected[:4]}", file=sys.stderr)
    else:
        print(f"[spectformer] no trained checkpoint at {ckpt!r} (or a reduced config was "
              f"requested) — RANDOM INIT; the golden checks lowering, not accuracy",
              file=sys.stderr)

    net = net.eval()
    return net, (torch.randn(1, 3, _IMG, _IMG),)
