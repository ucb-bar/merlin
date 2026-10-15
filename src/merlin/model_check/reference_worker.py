"""The FP reference of each model, run INSIDE the model's own pinned environment (``python reference_worker.py``).

The float model is built exactly as its capture loader builds it -- torchvision ``resnet50`` with
``IMAGENET1K_V2``, the TinyLlama checkpoint in float32, SmolVLA's ``VLAFlowMatching`` driven through the
loader's own stage modules -- and evaluated on each input's model-ready arrays. It imports only the
standard library, numpy and the framework (plus, for SmolVLA, the capture loader file named in the
request), never Merlin. Results are written with the input worker's byte-stable ``.npz`` writer.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _save(request: dict, path: str, **arrays) -> None:
    _module(request["input_worker"], "_mc_input_worker").save_npz(path, **arrays)


def fp_resnet(request: dict) -> dict:
    """Logits [1, 1000] of torchvision ResNet-50 (IMAGENET1K_V2) per input."""
    import torch
    from torchvision.models import ResNet50_Weights, resnet50

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    out = {}
    with torch.no_grad():
        for item in request["items"]:
            with np.load(item["inputs"]) as data:
                image = torch.from_numpy(np.asarray(data["in0"], np.float32))
            logits = model(image).numpy().astype(np.float32)
            _save(request, item["out"], logits=logits)
            out[item["id"]] = {"top1": int(logits.argmax())}
    return {"items": out}


def fp_tiny_llama(request: dict) -> dict:
    """Next-token logits at positions [prefill-1, prefill+decode) of one causal pass over the segment.

    The chained program (prefill, then teacher-forced decode steps) computes exactly these distributions:
    the decode steps' inputs are the segment's own next tokens.
    """
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        request["checkpoint"], dtype=torch.float32, use_cache=False, tie_word_embeddings=False, local_files_only=True
    ).eval()
    prefill, decode = int(request["prefill_tokens"]), int(request["decode_steps"])
    out = {}
    with torch.no_grad():
        for item in request["items"]:
            with np.load(item["inputs"]) as data:
                tokens = np.asarray(data["tokens"], np.int64)
            ids = torch.from_numpy(tokens[: prefill + decode][None])
            logits = model(input_ids=ids).logits[0, prefill - 1 : prefill + decode].numpy().astype(np.float32)
            _save(request, item["out"], logits=logits)
            out[item["id"]] = {"distributions": int(logits.shape[0])}
    return {"items": out}


def fp_smolvla(request: dict) -> dict:
    """Actions [1, 50, 32] after prefix -> num_steps flow steps -> action_decode, from each input's noise."""
    import torch

    loader = _module(request["loader"], "_mc_smolvla_loader")
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    cfg = loader._load_pretrained_config()  # noqa: SLF001 -- the loader's own pinned config path
    cfg.device = "cpu"
    policy = SmolVLAPolicy.from_pretrained(
        loader._MODEL_ID,
        config=cfg,
        revision=loader._MODEL_REVISION,
        local_files_only=True,
        strict=True,  # noqa: SLF001
    )
    model = policy.model.cpu().eval()
    prefix = loader.SmolVLAPrefixStage(model).eval()
    flow = loader.SmolVLAFlowSession(model).eval()
    steps = int(model.config.num_steps)
    horizon, action_dim = int(model.config.chunk_size), int(model.config.max_action_dim)
    out = {}
    with torch.no_grad():
        for item in request["items"]:
            with np.load(item["inputs"]) as data:
                arrays = {k: torch.from_numpy(np.asarray(data[k])) for k in data.files}
            masks, cache = prefix(
                arrays["image"],
                arrays["image_mask"],
                arrays["language_tokens"],
                arrays["language_mask"],
                arrays["state"],
            )
            state, timestep = arrays["noise"], torch.ones((1,), dtype=torch.float32)
            for _ in range(steps):
                state, cache, timestep = flow(masks, cache, state, timestep)
            actions = state[:, :horizon, :action_dim].numpy().astype(np.float32)
            _save(request, item["out"], actions=actions)
            out[item["id"]] = {"steps": steps}
    return {"items": out}


TASKS = {"fp_resnet": fp_resnet, "fp_tiny_llama": fp_tiny_llama, "fp_smolvla": fp_smolvla}


def main(argv: list[str]) -> int:
    request = json.loads(Path(argv[1]).read_text())
    result = TASKS[request["task"]](request)
    Path(request["result"]).write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
