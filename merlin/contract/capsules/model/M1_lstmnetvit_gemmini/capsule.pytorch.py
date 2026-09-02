"""LSTMNetVIT (vitfly) vision-based quadrotor obstacle avoidance -> MLIR.

    python workloads/capture.py lstmnetvit --formats fp32,int8
    <venv>/bin/python workloads/capture_consistent.py lstmnetvit int8 <bundle_dir>

Capture unit: ONE stateful control step. The network is a two-stage SegFormer-style Mix-Transformer
encoder over a 60x90 depth image (overlapping patch-embed convs, efficient self-attention with
spatial reduction, a depthwise-conv MixFFN), a PixelShuffle + bilinear-upsample fusion of the
two feature scales, and a 3-layer LSTM head that emits a 3-DoF command.

The wrapper takes the LSTM hidden/cell tensors explicitly and returns the command plus their updates.
The consistent-bundle capture writes a deterministic sequence of distinct frames and eager trajectory
goldens, so the runtime measures the real recurrent state transition instead of repeating one zero-state
forward.

Synthetic trajectories and random initialization remain available only for compiler smoke tests. A
paper-ready capture requires the complete published state dict loaded strictly, plus an attributed external
trajectory with exactly the requested number of frames, desired velocities, and unit quaternions.

Env:
    VITFLY_DIR   upstream checkout (default: <path>)
    VITFLY_CKPT           complete trained LSTMNetVIT state_dict (required for paper capture)
    VITFLY_SESSION_NPZ    arrays: frames, desired_velocity, quaternions
    VITFLY_SESSION_SOURCE stable provenance label for the trajectory
    VITFLY_PAPER_READY    set to 1 to fail closed unless all paper inputs are valid

Upstream: https://github.com/anish-bhattacharya/vitfly  (models/model.py)
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

_UPSTREAM = Path(os.environ.get("VITFLY_DIR", "<path>")) / "models"
_PAPER_CHECKPOINT = "vitfly/LSTMNetVIT/published_pretrained"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _session_streams(steps: int, example: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                     ) -> tuple[dict[str, torch.Tensor], dict, bool]:
    path_value = os.environ.get("VITFLY_SESSION_NPZ", "")
    source_label = os.environ.get("VITFLY_SESSION_SOURCE", "")
    requested_ready = os.environ.get("VITFLY_PAPER_READY", "0") == "1"
    depth, desvel, quat = example
    if path_value:
        path = Path(path_value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"VITFLY_SESSION_NPZ is absent: {path}")
        with np.load(path) as data:
            required = {"frames", "desired_velocity", "quaternions"}
            if not required <= set(data.files):
                raise ValueError(f"VITFLY_SESSION_NPZ omits {sorted(required - set(data.files))}")
            arrays = {key: np.ascontiguousarray(data[key], dtype=np.float32) for key in required}
        expected = {
            "frames": (steps, *depth.shape),
            "desired_velocity": (steps, *desvel.shape),
            "quaternions": (steps, *quat.shape),
        }
        for key, shape in expected.items():
            if arrays[key].shape != shape:
                raise ValueError(f"{key} has shape {arrays[key].shape}, expected {shape}")
        norms = np.linalg.norm(arrays["quaternions"], axis=-1)
        if not np.all(np.isfinite(arrays["frames"])) or not np.all(np.isfinite(norms)):
            raise ValueError("LSTMNetViT session contains non-finite values")
        if not np.allclose(norms, 1.0, atol=1.0e-4, rtol=1.0e-4):
            raise ValueError("session quaternions must already be unit-normalized")
        streams = {key: torch.from_numpy(value) for key, value in arrays.items()}
        provenance = {"session_source": source_label or "unattributed_external_npz",
                      "session_path": str(path), "session_sha256": _sha256(path),
                      "synthetic_session": False}
        ready = bool(source_label)
    else:
        generator = torch.Generator(device="cpu").manual_seed(20260830)
        frames = torch.randn((steps, *depth.shape), generator=generator, dtype=depth.dtype)
        desired = torch.rand((steps, *desvel.shape), generator=generator,
                             dtype=desvel.dtype) + 1.0
        quaternions = torch.randn((steps, *quat.shape), generator=generator, dtype=quat.dtype)
        quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        streams = {"frames": frames, "desired_velocity": desired,
                   "quaternions": quaternions}
        raw = b"".join(value.detach().cpu().numpy().tobytes() for value in streams.values())
        provenance = {"session_source": "synthetic_seed_20260830",
                      "session_sha256": hashlib.sha256(raw).hexdigest(),
                      "synthetic_session": True}
        ready = False
    if requested_ready and not ready:
        raise ValueError(
            "VITFLY_PAPER_READY=1 requires VITFLY_SESSION_NPZ and an attributed "
            "VITFLY_SESSION_SOURCE")
    return streams, provenance, ready


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


class _StatefulCommand(nn.Module):
    """Flatten the upstream list/tuple ABI into five tensor inputs and three tensor outputs."""

    def __init__(self, net: nn.Module, *, paper_ready: bool, provenance: dict,
                 streams: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.net = net
        self.paper_ready = paper_ready
        self.session_provenance = provenance
        self.session_streams = streams

    def forward(self, depth: torch.Tensor, desvel: torch.Tensor, quat: torch.Tensor,
                hidden: torch.Tensor, cell: torch.Tensor):
        out, updated = self.net([depth, desvel, quat, (hidden, cell)])
        return out, updated[0], updated[1]


def get_model_and_inputs() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    # models/model.py does `from ViTsubmodules import *`, so its own directory must be on the
    # path (not the repo root).
    if str(_UPSTREAM) not in sys.path:
        sys.path.insert(0, str(_UPSTREAM))
    import model as vitfly_models  # type: ignore

    net = vitfly_models.LSTMNetVIT()
    ckpt = os.environ.get("VITFLY_CKPT")
    requested_ready = os.environ.get("VITFLY_PAPER_READY", "0") == "1"
    checkpoint_ready = bool(ckpt and Path(ckpt).is_file())
    checkpoint_provenance = {"checkpoint": "random_init", "full_checkpoint": False}
    if checkpoint_ready:
        checkpoint_path = Path(ckpt).expanduser().resolve()
        blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
        net.load_state_dict(state, strict=True)
        checkpoint_provenance = {
            "checkpoint": _PAPER_CHECKPOINT,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "full_checkpoint": True,
        }
        print(f"[lstmnetvit] strictly loaded {checkpoint_path}", file=sys.stderr)
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
    # Upstream feeds an unbatched 2-D tensor to nn.LSTM, so h/c use the corresponding
    # [num_layers, hidden_size] ABI rather than the batched [layers, batch, hidden] form.
    hidden = torch.zeros(net.lstm.num_layers, net.lstm.hidden_size)
    cell = torch.zeros_like(hidden)
    steps = int(os.environ.get("M2M_SESSION_STEPS", "256"))
    if steps < 1:
        raise RuntimeError("M2M_SESSION_STEPS must be positive")
    streams, session_provenance, session_ready = _session_streams(steps, (depth, desvel, quat))
    paper_ready = bool(requested_ready and checkpoint_ready and session_ready)
    if requested_ready and not checkpoint_ready:
        raise ValueError("VITFLY_PAPER_READY=1 requires a complete VITFLY_CKPT")
    provenance = {**checkpoint_provenance, **session_provenance}
    wrapper = _StatefulCommand(net, paper_ready=paper_ready, provenance=provenance,
                               streams=streams).eval()
    return wrapper, (streams["frames"][0], streams["desired_velocity"][0],
                     streams["quaternions"][0], hidden, cell)


def get_session_spec(model: nn.Module, inputs: tuple[torch.Tensor, ...]) -> dict:
    """The real recurrent controller session, expressed in loader-input/output indices."""
    steps = int(os.environ.get("M2M_SESSION_STEPS", "256"))
    if steps < 1:
        raise RuntimeError("M2M_SESSION_STEPS must be positive")
    streams = getattr(model, "session_streams", None)
    if not isinstance(streams, dict):
        raise RuntimeError("LSTMNetViT session streams were not attached by the loader")
    if int(streams["frames"].shape[0]) != steps:
        raise RuntimeError("attached LSTMNetViT trajectory length changed after model creation")
    return {
        "kind": "recurrent_frames",
        "paper_ready": bool(getattr(model, "paper_ready", False)),
        "stages": ["visual_encode", "recurrent_step", "predict"],
        "stage_schedule": [
            {"name": "visual_encode", "steps": steps,
             "execution": "compiled_recurrent", "timed": True},
            {"name": "recurrent_step", "steps": steps,
             "execution": "compiled_recurrent", "timed": True},
            {"name": "predict", "steps": steps,
             "execution": "compiled_recurrent", "timed": True},
        ],
        "parameters": {"batch": 1, "sequence_length": steps},
        "states": [
            {"name": "hidden_state", "input_index": 3, "output_index": 1},
            {"name": "cell_state", "input_index": 4, "output_index": 2},
        ],
        "streams": [
            {"name": "frame", "input_index": 0, "key": "frames", "values": streams["frames"]},
            {"name": "desired_velocity", "input_index": 1, "key": "desired_velocity",
             "values": streams["desired_velocity"]},
            {"name": "quaternion", "input_index": 2, "key": "quaternions",
             "values": streams["quaternions"]},
        ],
        "quality": {"key": "output0", "output_index": 0},
        "provenance": {**getattr(model, "session_provenance", {}),
                       "paper_ready_requires_strict_checkpoint_and_attributed_session": True},
    }
