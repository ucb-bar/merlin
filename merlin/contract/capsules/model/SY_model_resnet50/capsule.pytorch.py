"""Torchvision ResNet-50 v1.5 capture with a semantic stream of distinct images.

Paper-ready capture requires pretrained IMAGENET1K_V2 weights and a preprocessed image stream supplied
through ``M2M_RESNET_INPUT_NPZ``.  ``M2M_RESNET_RANDOM=1`` is an explicit smoke-only escape hatch; its
session contract is marked not paper-ready.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn


class _Classifier(nn.Module):
    def __init__(self, model: nn.Module, *, paper_ready: bool, provenance: dict) -> None:
        super().__init__()
        self.model = model
        self.paper_ready = paper_ready
        self.session_provenance = provenance

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_dict_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(array.shape)).encode("ascii") + b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def _images() -> tuple[torch.Tensor, dict, bool]:
    source = os.environ.get("M2M_RESNET_INPUT_NPZ", "")
    source_label = os.environ.get("M2M_RESNET_INPUT_SOURCE", "")
    preprocessing = os.environ.get("M2M_RESNET_PREPROCESSING", "")
    requested_ready = os.environ.get("M2M_RESNET_PAPER_READY", "0") == "1"
    steps = int(os.environ.get("M2M_SESSION_STEPS", "256"))
    if steps < 1:
        raise ValueError("M2M_SESSION_STEPS must be positive")
    if source and Path(source).is_file():
        source_path = Path(source).expanduser().resolve()
        with np.load(source_path) as data:
            key = "images" if "images" in data.files else data.files[0]
            images = np.ascontiguousarray(data[key], dtype=np.float32)
        if images.ndim == 4:
            images = images[:, None, :, :, :]
        if images.ndim != 5 or list(images.shape[1:]) != [1, 3, 224, 224]:
            raise RuntimeError(
                "M2M_RESNET_INPUT_NPZ must contain [steps,1,3,224,224] or [steps,3,224,224] images")
        if images.shape[0] != steps:
            raise RuntimeError(f"ResNet image stream has {images.shape[0]} steps, expected {steps}")
        if not np.all(np.isfinite(images)):
            raise RuntimeError("ResNet image stream contains non-finite values")
        provenance = {"input_source": source_label or "unattributed_external_npz",
                      "input_path": str(source_path), "input_sha256": _sha256(source_path),
                      "preprocessing": preprocessing or "unspecified",
                      "synthetic_inputs": False}
        ready = bool(source_label and preprocessing == "IMAGENET1K_V2")
        if requested_ready and not ready:
            raise RuntimeError(
                "M2M_RESNET_PAPER_READY=1 requires M2M_RESNET_INPUT_SOURCE and "
                "M2M_RESNET_PREPROCESSING=IMAGENET1K_V2")
        return torch.from_numpy(images), provenance, ready
    if source:
        raise FileNotFoundError(f"M2M_RESNET_INPUT_NPZ is absent: {Path(source).expanduser()}")
    if os.environ.get("M2M_RESNET_RANDOM") != "1":
        raise RuntimeError(
            "paper capture requires M2M_RESNET_INPUT_NPZ with preprocessed ImageNet images; "
            "set M2M_RESNET_RANDOM=1 only for compiler smoke tests")
    generator = torch.Generator(device="cpu").manual_seed(20260830)
    images = torch.randn((steps, 1, 3, 224, 224), generator=generator)
    provenance = {"input_source": "synthetic_seed_20260830",
                  "input_sha256": hashlib.sha256(images.numpy().tobytes()).hexdigest(),
                  "preprocessing": "synthetic", "synthetic_inputs": True}
    if requested_ready:
        raise RuntimeError("M2M_RESNET_PAPER_READY=1 forbids synthetic images")
    return images, provenance, False


def get_model_and_inputs() -> tuple[nn.Module, tuple[torch.Tensor, ...]]:
    from torchvision.models import ResNet50_Weights, resnet50

    images, input_provenance, inputs_ready = _images()
    requested_ready = os.environ.get("M2M_RESNET_PAPER_READY", "0") == "1"
    random_model = os.environ.get("M2M_RESNET_RANDOM") == "1" and os.environ.get(
        "M2M_RESNET_PRETRAINED", "1") == "0"
    weights = None if random_model else ResNet50_Weights.IMAGENET1K_V2
    if requested_ready and weights is None:
        raise RuntimeError("M2M_RESNET_PAPER_READY=1 requires IMAGENET1K_V2 pretrained weights")
    model = resnet50(weights=weights).eval()
    checkpoint_provenance = {
        "checkpoint": ("torchvision/resnet50/IMAGENET1K_V2" if weights is not None
                       else "random_init"),
        "checkpoint_sha256": _state_dict_sha256(model),
        "full_checkpoint": weights is not None,
    }
    wrapper = _Classifier(
        model, paper_ready=bool(requested_ready and weights is not None and inputs_ready),
        provenance={**checkpoint_provenance, **input_provenance}).eval()
    wrapper.session_images = images
    return wrapper, (images[0],)


def get_session_spec(model: nn.Module, _inputs: tuple[torch.Tensor, ...]) -> dict:
    images = getattr(model, "session_images", None)
    if images is None:
        raise RuntimeError("ResNet session image stream was not attached by the loader")
    return {
        "kind": "image_stream",
        "paper_ready": bool(getattr(model, "paper_ready", False)),
        "stages": ["classify"],
        "stage_schedule": [{"name": "classify", "steps": int(images.shape[0]),
                            "execution": "compiled", "timed": True}],
        "parameters": {"batch": 1, "height": 224, "width": 224, "channels": 3},
        "states": [],
        "streams": [{"name": "image", "input_index": 0, "key": "images", "values": images}],
        "quality": {"key": "output0", "output_index": 0},
        "provenance": {**getattr(model, "session_provenance", {}),
                       "paper_ready_requires_pretrained_and_attributed_real_inputs": True},
    }
