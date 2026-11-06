"""FP32 Multi-Task DiT diagnostic, random policy weights, no accuracy claim.

Pretrained CLIP encoders are loaded from HF_HOME in offline mode. The default
architecture is retained; DDIM/10 follows the documented inference suggestion.
Robot dimensions and camera count are workload choices, not policy defaults.
"""

import dataclasses
import os
import hashlib

# Set offline policy before importing either Hugging Face client.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from pathlib import Path

import torch

# The offline torchvision 0.26 wheel contains CUDA extensions for torch 2.11.
# This model uses only its Python tensor transforms. Opt in to the missing NMS
# schema for import-time fake registration, never an NMS implementation.
_TV_LIBRARY = None
if os.environ.get("MTDIT_TORCHVISION_SCHEMA_COMPAT") == "1":
    try:
        torch._C._dispatch_find_schema_or_throw("torchvision::nms", "")
    except RuntimeError:
        _TV_LIBRARY = torch.library.Library("torchvision", "FRAGMENT")
        _TV_LIBRARY.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
from huggingface_hub.constants import HF_HUB_CACHE
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig
from lerobot.policies.multi_task_dit.modeling_multi_task_dit import MultiTaskDiTPolicy
from lerobot.utils.constants import (
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)
from transformers import AutoTokenizer

SEED = 0
CLIP_REVISION = "57c216476eefef5ab752ec549e440a49ae4ae5f3"
INSTRUCTION = "Pick up the red block and place it in the bowl."
WEIGHT_SCOPE = "Random DiT and text-projection weights; pretrained CLIP encoders; no accuracy claim."


class Inference(torch.nn.Module):
    """One action chunk with explicit initial noise and static DDIM timesteps.

    Uses the policy's observation encoder, noise predictor, scheduler.step, and
    output slice. Queue management and dataset normalization are outside this ABI.
    Passing noise makes the stochastic boundary explicit without changing math.
    """

    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        scheduler = policy.objective.noise_scheduler
        scheduler.set_timesteps(policy.objective.num_inference_steps)
        self.timesteps = tuple(int(t) for t in scheduler.timesteps)

    def forward(self, images, state, tokens, attention_mask, noise):
        batch = {OBS_IMAGES: images, OBS_STATE: state, OBS_LANGUAGE_TOKENS: tokens,
                 OBS_LANGUAGE_ATTENTION_MASK: attention_mask}
        conditioning = self.policy.observation_encoder.encode(batch)
        sample = noise
        for timestep in self.timesteps:
            prediction = self.policy.noise_predictor(
                sample, torch.full(sample.shape[:1], timestep, dtype=torch.long, device=sample.device),
                conditioning_vec=conditioning,
            )
            sample = self.policy.objective.noise_scheduler.step(prediction, timestep, sample).prev_sample
        start = self.policy.config.n_obs_steps - 1
        return sample[:, start:start + self.policy.config.n_action_steps]


def clip_snapshot():
    return Path(HF_HUB_CACHE) / "models--openai--clip-vit-base-patch16" / "snapshots" / CLIP_REVISION


def checkpoint_metadata():
    files = {}
    for name in ("config.json", "pytorch_model.bin", "merges.txt", "vocab.json",
                 "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "preprocessor_config.json"):
        path = clip_snapshot() / name
        with path.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        files[name] = {"bytes": path.stat().st_size, "sha256": sha}
    return {"revision": CLIP_REVISION, "snapshot": str(clip_snapshot()), "files": files}


def get_model_and_inputs():
    """Return (model, positional inputs), random policy weights, no accuracy claim."""
    torch.manual_seed(SEED)
    config = MultiTaskDiTConfig(
        device="cpu", vision_encoder_name=str(clip_snapshot()),
        text_encoder_name=str(clip_snapshot()), noise_scheduler_type="DDIM", num_inference_steps=10,
        input_features={
            "observation.images.front": PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
            "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
            "observation.state": PolicyFeature(FeatureType.STATE, (14,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (14,))},
    )
    policy = MultiTaskDiTPolicy(config).eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(config.text_encoder_name, revision=CLIP_REVISION, local_files_only=True)
    text = tokenizer(INSTRUCTION, padding="max_length", max_length=config.tokenizer_max_length,
                     truncation=True, return_tensors="pt")
    generator = torch.Generator().manual_seed(SEED)
    # Synthetic, already normalized model-entry observations, not a dataset sample.
    images = torch.rand((1, config.n_obs_steps, 2, 3, 224, 224), generator=generator)
    state = torch.zeros((1, config.n_obs_steps, 14))
    noise = torch.randn((1, config.horizon, 14), generator=generator)
    return Inference(policy).eval(), (images, state, text["input_ids"], text["attention_mask"], noise)


def describe(model, inputs):
    """Serializable construction scope for capture and coverage receipts."""
    return {
        "weight_scope": WEIGHT_SCOPE, "seed": SEED, "dtype": "float32", "quantization": "none",
        "clip_revision": CLIP_REVISION,
        "instruction": INSTRUCTION, "config": dataclasses.asdict(model.policy.config),
        "timesteps": model.timesteps, "output_shape": [1, 24, 14],
        "input_names": ["images", "state", "tokens", "attention_mask", "noise"],
        "inputs": [{"shape": list(x.shape), "dtype": str(x.dtype)} for x in inputs],
        "scope": "One action chunk; two synthetic cameras and 14 state/action coordinates; dataset normalization and queues excluded.",
        "defaults_overridden": {"noise_scheduler_type": ["DDPM", "DDIM"], "num_inference_steps": [None, 10]},
    }
