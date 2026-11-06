"""FP32 Multi-Task DiT diagnostic, random policy weights, no accuracy claim.

Pretrained CLIP encoders are loaded from HF_HOME in offline mode. The default
architecture is retained; DDIM/10 follows the documented inference suggestion.
Robot dimensions and camera count are workload choices, not policy defaults.
"""

import dataclasses
import os
import hashlib
import copy
import json

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

    def __init__(self, policy, *, static_scheduler=False):
        super().__init__()
        self.policy = policy
        scheduler = policy.objective.noise_scheduler
        scheduler.set_timesteps(policy.objective.num_inference_steps)
        self.timesteps = tuple(int(t) for t in scheduler.timesteps)
        self.scheduler = scheduler
        if static_scheduler:
            # Freeze only lookup addressing at construction. Retain the original
            # scheduler.step arithmetic and exact FP32 scalar tensor values.
            self.scheduler = copy.copy(scheduler)
            self.scheduler.alphas_cumprod = {
                i: value.clone() for i, value in enumerate(scheduler.alphas_cumprod)
            }

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
            sample = self.scheduler.step(prediction, timestep, sample).prev_sample
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
    checkpoint = os.environ.get("MTDIT_CHECKPOINT")
    if checkpoint:
        return get_checkpoint_model_and_inputs(Path(checkpoint))
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
    return Inference(policy, static_scheduler=os.environ.get("MTDIT_STATIC_SCHEDULER", "1") == "1").eval(), (images, state, text["input_ids"], text["attention_mask"], noise)


def describe(model, inputs):
    """Serializable construction scope for capture and coverage receipts."""
    if hasattr(model, "checkpoint_scope"):
        return model.checkpoint_scope | {
            "config": dataclasses.asdict(model.policy.config),
            "inputs": [{"shape": list(x.shape), "dtype": str(x.dtype)} for x in inputs],
        }
    return {
        "static_scheduler_lookup": os.environ.get("MTDIT_STATIC_SCHEDULER", "1") == "1",
        "weight_scope": WEIGHT_SCOPE, "seed": SEED, "dtype": "float32", "quantization": "none",
        "clip_revision": CLIP_REVISION,
        "instruction": INSTRUCTION, "config": dataclasses.asdict(model.policy.config),
        "timesteps": model.timesteps, "output_shape": [1, 24, 14],
        "input_names": ["images", "state", "tokens", "attention_mask", "noise"],
        "inputs": [{"shape": list(x.shape), "dtype": str(x.dtype)} for x in inputs],
        "scope": "One action chunk; two synthetic cameras and 14 state/action coordinates; dataset normalization and queues excluded.",
        "defaults_overridden": {"noise_scheduler_type": ["DDPM", "DDIM"], "num_inference_steps": [None, 10]},
    }


class CheckpointInference(torch.nn.Module):
    """Saved flow-matching policy plus saved tensor pre/post transforms.

    Tokens are the explicit boundary for the saved tokenizer. Queue management
    stays outside the single-action-chunk graph. Time-grid .item() calls occur
    once at construction, exactly as the policy's Euler solver obtains them.
    """

    def __init__(self, policy, normalizer, unnormalizer):
        super().__init__()
        self.policy = policy
        self.normalizer = normalizer
        self.unnormalizer = unnormalizer
        self.camera_keys = tuple(policy.config.image_features)
        config = policy.config
        if config.objective != "flow_matching" or config.integration_method != "euler":
            raise ValueError("checkpoint capture currently requires its saved flow-matching Euler objective")
        grid = torch.linspace(0, 1, config.num_integration_steps + 1)
        self.integration_grid = tuple((grid[i].item(), (grid[i + 1] - grid[i]).item())
                                      for i in range(config.num_integration_steps))

    def forward(self, images, state, tokens, attention_mask, noise):
        observations = {key: images[:, :, i:i + 1].squeeze(2) for i, key in enumerate(self.camera_keys)}
        observations[OBS_STATE] = state
        observations = self.normalizer._normalize_observation(observations, inverse=False)
        batch = {OBS_IMAGES: torch.stack([observations[key] for key in self.camera_keys], dim=-4),
                 OBS_STATE: observations[OBS_STATE], OBS_LANGUAGE_TOKENS: tokens,
                 OBS_LANGUAGE_ATTENTION_MASK: attention_mask}
        conditioning = self.policy.observation_encoder.encode(batch)
        sample = noise
        for timestep, dt in self.integration_grid:
            t_batch = torch.full((sample.shape[0],), timestep, dtype=sample.dtype, device=sample.device)
            velocity = self.policy.noise_predictor(sample, t_batch, conditioning_vec=conditioning)
            sample = sample + dt * velocity
        start = self.policy.config.n_obs_steps - 1
        action = sample[:, start:start + self.policy.config.n_action_steps]
        return self.unnormalizer._normalize_action(action, inverse=True)


    def reference_forward(self, images, state, tokens, attention_mask, noise):
        """Unmodified objective solver with the same explicit initial noise."""
        from unittest.mock import patch
        observations = {key: images[:, :, i:i + 1].squeeze(2) for i, key in enumerate(self.camera_keys)}
        observations[OBS_STATE] = state
        observations = self.normalizer._normalize_observation(observations, inverse=False)
        batch = {OBS_IMAGES: torch.stack([observations[key] for key in self.camera_keys], dim=-4),
                 OBS_STATE: observations[OBS_STATE], OBS_LANGUAGE_TOKENS: tokens,
                 OBS_LANGUAGE_ATTENTION_MASK: attention_mask}
        conditioning = self.policy.observation_encoder.encode(batch)
        with patch("torch.randn", return_value=noise):
            sample = self.policy.objective.conditional_sample(
                self.policy.noise_predictor, noise.shape[0], conditioning)
        start = self.policy.config.n_obs_steps - 1
        return self.unnormalizer._normalize_action(
            sample[:, start:start + self.policy.config.n_action_steps], inverse=True)


def validate_checkpoint(checkpoint):
    """Validate all local checkpoint bytes before constructing the policy."""
    if any(token in part.lower() for part in checkpoint.parts for token in ("hammer", "vlsi")):
        raise ValueError("forbidden checkpoint path")
    if checkpoint.is_symlink():
        raise ValueError("checkpoint root must be a local directory")
    files = {}
    def safe_path(name):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid checkpoint manifest path")
        if any(token in part.lower() for part in relative.parts for token in ("hammer", "vlsi")):
            raise ValueError("checkpoint manifest contains a forbidden path")
        path = checkpoint / relative
        current = checkpoint
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("checkpoint manifests must reference local regular files")
        return path
    manifest = safe_path("SHA256SUMS")
    entries = []
    for line in manifest.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        entries.append((name.lstrip("* "), expected))
    entries.extend((name, None) for name in ("SHA256SUMS", "REVISION"))
    for name, expected in entries:
        path = safe_path(name)
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if expected is not None and actual != expected:
            raise ValueError(f"checkpoint SHA256SUMS verification failed for {name}")
        files[name] = {"bytes": path.stat().st_size, "sha256": actual,
                       "manifest_verified": expected is not None}
    return files


def get_checkpoint_model_and_inputs(checkpoint):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
    from safetensors.torch import load_file

    files = validate_checkpoint(checkpoint)
    torch.manual_seed(SEED)
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    # Execution placement and cached pretrained encoder paths are the only overrides.
    config.device = "cpu"
    config.vision_encoder_name = str(clip_snapshot())
    config.text_encoder_name = str(clip_snapshot())
    policy = MultiTaskDiTPolicy.from_pretrained(checkpoint, config=config, strict=True,
                                               local_files_only=True).eval().requires_grad_(False)
    processors = []
    preprocessor = json.loads((checkpoint / "policy_preprocessor.json").read_text())
    tokenizer_step = next(x["config"] for x in preprocessor["steps"]
                          if x["registry_name"] == "tokenizer_processor")
    expected_tokenizer = {"max_length": config.tokenizer_max_length, "padding": config.tokenizer_padding,
                          "padding_side": config.tokenizer_padding_side,
                          "truncation": config.tokenizer_truncation}
    for key, value in expected_tokenizer.items():
        if tokenizer_step[key] != value:
            raise ValueError(f"saved tokenizer processor disagrees with policy config for {key}")
    for name, registry, cls in (("policy_preprocessor", "normalizer_processor", NormalizerProcessorStep),
                                ("policy_postprocessor", "unnormalizer_processor", UnnormalizerProcessorStep)):
        document = json.loads((checkpoint / f"{name}.json").read_text())
        step = next(x for x in document["steps"] if x["registry_name"] == registry)
        processor = cls(**step["config"], device="cpu")
        processor.load_state_dict(load_file(str(checkpoint / step["state_file"])))
        processors.append(processor)
    model = CheckpointInference(policy, *processors).eval()
    tokenizer = AutoTokenizer.from_pretrained(config.text_encoder_name, revision=CLIP_REVISION, local_files_only=True)
    tokenizer.padding_side = tokenizer_step["padding_side"]
    text = tokenizer(INSTRUCTION, padding=tokenizer_step["padding"], max_length=tokenizer_step["max_length"],
                     truncation=tokenizer_step["truncation"], return_tensors="pt")
    generator = torch.Generator().manual_seed(SEED)
    camera_shape = tuple(next(iter(config.image_features.values())).shape)
    action_dim = config.output_features["action"].shape[0]
    inputs = (torch.rand((1, config.n_obs_steps, len(config.image_features), *camera_shape), generator=generator),
              torch.zeros((1, config.n_obs_steps, config.input_features[OBS_STATE].shape[0])),
              text["input_ids"], text["attention_mask"],
              torch.randn((1, config.horizon, action_dim), generator=generator))
    model.checkpoint_scope = {
        "weight_scope": "Local community checkpoint, strictly loaded; synthetic observations; no accuracy claim.",
        "checkpoint": str(checkpoint.resolve()), "checkpoint_files": files,
        "seed": SEED, "dtype": "float32", "quantization": "none",
        "instruction": INSTRUCTION, "tokenizer_processor": tokenizer_step, "camera_keys": model.camera_keys,
        "integration_grid": model.integration_grid,
        "input_names": ["images", "state", "tokens", "attention_mask", "noise"],
        "output_shape": [1, config.n_action_steps, action_dim],
        "scope": "Saved observation normalization, center crop in eval mode, encoders, all Euler steps, action slice, and saved action unnormalization. Tokenizer and queues outside tensor ABI.",
        "config_overrides": {"device": ["cuda", "cpu"], "vision_encoder_name": str(clip_snapshot()),
                             "text_encoder_name": str(clip_snapshot())},
    }
    return model, inputs
