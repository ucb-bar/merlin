"""LSTMNetVIT (vitfly) vision-based quadrotor obstacle avoidance -> MLIR.

    python workloads/capture.py lstmnetvit --formats fp32,int8
    <venv>/bin/python workloads/capture_consistent.py lstmnetvit int8 <bundle_dir>

Capture unit: ONE control step. The network is a two-stage SegFormer-style Mix-Transformer
encoder over a 60x90 depth image (overlapping patch-embed convs, efficient self-attention with
spatial reduction, a depthwise-conv MixFFN), a PixelShuffle + bilinear-upsample fusion of the
two feature scales, and a 3-layer LSTM head that emits a 3-DoF command.

The wrapper returns ONLY the command. The real controller also threads the LSTM hidden state
from step to step, but the capture contract is a single output tensor, so this captures one step
from a ZERO initial state -- it exercises the recurrence arithmetic (torch.export unrolls the
LSTM into per-timestep gate matmuls) without proving multi-step recurrent behaviour.

Weights are random init: vitfly publishes its trained models as a password-protected
``pretrained_models.tar`` on Box rather than in the repo, so the golden here checks lowering
exactness, not flight accuracy. Set VITFLY_CKPT once that tarball is unpacked.

Env:
    VITFLY_DIR   upstream checkout (default: /scratch/agustin/projects/vitfly)
    VITFLY_CKPT  optional trained LSTMNetVIT state_dict

Upstream: https://github.com/anish-bhattacharya/vitfly  (models/model.py)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch import nn

_UPSTREAM = Path(os.environ.get("VITFLY_DIR", "/scratch/agustin/projects/vitfly")) / "models"


def _fold_spectral_norm(net: nn.Module) -> int:
    """Fold every spectral-norm reparametrization back into a plain ``weight``.

    ``LSTMNetVIT`` wraps two Linear layers in ``spectral_norm``, which makes ``weight`` a value
    recomputed from ``weight_orig``/``weight_u`` rather than a Parameter. In eval mode that
    value is fixed, so folding it is exact -- and it is required, because torchAO's ``quantize_``
    swaps ``weight`` in place and raises ("cannot assign 'torch.FloatTensor' as parameter
    'weight'") when the attribute is owned by a reparametrization. Returns how many it folded.
    """
    import torch.nn.utils.parametrize as P

    folded = 0
    for mod in net.modules():
        if P.is_parametrized(mod) and "weight" in getattr(mod, "parametrizations", {}):
            P.remove_parametrizations(mod, "weight", leave_parametrized=True)
            folded += 1
            continue
        if any(getattr(h, "__class__", type(None)).__name__ == "SpectralNorm"
               for h in getattr(mod, "_forward_pre_hooks", {}).values()):
            nn.utils.remove_spectral_norm(mod)
            folded += 1
    if folded:
        print(f"[lstmnetvit] folded {folded} spectral-norm reparametrization(s) into weight "
              f"(exact in eval mode)", file=sys.stderr)
    return folded


class _CommandOnly(nn.Module):
    """(depth, desired_velocity, quaternion) -> command, dropping the LSTM state output."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, depth: torch.Tensor, desvel: torch.Tensor, quat: torch.Tensor):
        out, _hidden = self.net([depth, desvel, quat])
        return out


def get_model_and_inputs() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    # models/model.py does `from ViTsubmodules import *`, so its own directory must be on the
    # path (not the repo root).
    if str(_UPSTREAM) not in sys.path:
        sys.path.insert(0, str(_UPSTREAM))
    import model as vitfly_models  # type: ignore

    net = vitfly_models.LSTMNetVIT()
    ckpt = os.environ.get("VITFLY_CKPT")
    if ckpt and Path(ckpt).is_file():
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
        missing, unexpected = net.load_state_dict(state, strict=False)
        print(f"[lstmnetvit] loaded {ckpt} (missing={len(missing)}, "
              f"unexpected={len(unexpected)})", file=sys.stderr)
    else:
        print("[lstmnetvit] no VITFLY_CKPT — RANDOM INIT; the golden checks lowering, not "
              "control accuracy (upstream ships weights in a password-gated Box tarball)",
              file=sys.stderr)
    net = net.eval()
    _fold_spectral_norm(net)

    # The published forward takes a LIST: [depth (N,1,60,90), desired velocity (N,1),
    # quaternion (N,4)]. The LSTM's input_size of 517 = 512 (decoder) + 1 (desvel) + 4 (quat)
    # is what pins desvel to ONE element -- passing a 5-vector makes the concat 521 wide and
    # the model raises.
    depth = torch.randn(1, 1, 60, 90)
    desvel = torch.rand(1, 1) + 1.0
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    return _CommandOnly(net).eval(), (depth, desvel, quat)
