"""SmolVLA (LeRobot) capture loader for m2m.

Builds smolVLA's VLAFlowMatching (SmolVLM2-500M backbone + action expert) and exposes
one flow-matching denoise step (prefix embed + one expert pass) as the capture unit,
on already-preprocessed tensors (host-side preprocessing stays out of the graph).

``M2M_SMOLVLA_SESSION=denoise`` exports the diagnostic recurrent flow step.
``M2M_SMOLVLA_SESSION=e2e`` emits compiled prefix, recurrent flow, and action-decode programs for
the primary continuous session. Paper readiness additionally requires ``M2M_SMOLVLA_PRETRAINED=1``,
an attributed ``M2M_SMOLVLA_INPUT_NPZ``, and explicit ``M2M_SMOLVLA_PAPER_READY=1`` opt-in.
"""

from __future__ import annotations

import os
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching, make_att_2d_masks

_MODEL_ID = "lerobot/smolvla_base"
_MODEL_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"


def _load_pretrained_config() -> SmolVLAConfig:
    """Load the pinned config through LeRobot's registered base-class dispatcher.

    LeRobot serializes a ``type: smolvla`` choice discriminator.  Calling the concrete
    ``SmolVLAConfig.from_pretrained`` asks draccus to decode that discriminator as an ordinary
    dataclass field and fails.  The generic ``PreTrainedConfig`` dispatcher consumes it and returns
    the exact registered SmolVLA subclass, which is also the path used by
    ``PreTrainedPolicy.from_pretrained`` when no explicit config is supplied.
    """
    cfg = PreTrainedConfig.from_pretrained(
        _MODEL_ID, revision=_MODEL_REVISION, local_files_only=True)
    if not isinstance(cfg, SmolVLAConfig):
        raise TypeError(
            f"{_MODEL_ID}@{_MODEL_REVISION} resolved to {type(cfg).__name__}, expected SmolVLAConfig")
    return cfg


class SmolVLADenoiseStep(nn.Module):
    def __init__(self, model: VLAFlowMatching) -> None:
        super().__init__()
        self.model = model

    def forward(self, img, img_mask, lang_tokens, lang_masks, state, noise):
        import os
        m = self.model
        prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(
            [img], [img_mask], lang_tokens, lang_masks, state=state
        )
        if os.environ.get("M2M_SMOLVLA_TAP") == "prefix":   # bisection: embed/conv/dequant only
            return prefix_embs
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1
        vlm_out, past_key_values = m.vlm_with_expert.forward(
            attention_mask=prefix_att_2d,
            position_ids=prefix_pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=m.config.use_cache,
            fill_kv_cache=True,
        )
        if os.environ.get("M2M_SMOLVLA_TAP") == "lm":   # bisection: LM output (RoPE/attention)
            o = vlm_out[0] if isinstance(vlm_out, (list, tuple)) else vlm_out
            return o.float()
        timestep = torch.ones(state.shape[0], dtype=torch.float32)
        return m.denoise_step(prefix_pad_masks, past_key_values, noise, timestep)


class SmolVLAFlowSession(nn.Module):
    """One Euler flow step over explicit prefix-cache, flow-state, and timestep tensors."""

    def __init__(self, model: VLAFlowMatching) -> None:
        super().__init__()
        self.model = model.eval().requires_grad_(False)
        self.dt = -1.0 / int(model.config.num_steps)

    def forward(self, prefix_pad_masks, prefix_kv_cache, flow_state, timestep):
        cache = {
            layer: {"key_states": prefix_kv_cache[0, layer],
                    "value_states": prefix_kv_cache[1, layer]}
            for layer in range(int(self.model.config.num_vlm_layers))
        }
        velocity = self.model.denoise_step(
            prefix_pad_masks, cache, flow_state, timestep)
        next_flow = flow_state + self.dt * velocity
        next_timestep = timestep + self.dt
        return next_flow, prefix_kv_cache, next_timestep


class SmolVLAPrefixStage(nn.Module):
    """Compile image/language/state prefix encoding into explicit mask and cache tensors."""

    def __init__(self, model: VLAFlowMatching) -> None:
        super().__init__()
        self.model = model.eval().requires_grad_(False)

    def forward(self, image, image_mask, lang_tokens, lang_masks, state):
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            [image], [image_mask], lang_tokens, lang_masks, state=state)
        attention = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        positions = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, cache = self.model.vlm_with_expert.forward(
            attention_mask=attention, position_ids=positions, past_key_values=None,
            inputs_embeds=[prefix_embs, None], use_cache=True, fill_kv_cache=True)
        keys = torch.stack([cache[layer]["key_states"]
                            for layer in range(int(self.model.config.num_vlm_layers))])
        values = torch.stack([cache[layer]["value_states"]
                              for layer in range(int(self.model.config.num_vlm_layers))])
        return prefix_pad_masks, torch.stack((keys, values))


class SmolVLAActionDecodeStage(nn.Module):
    """Compiled neural-output boundary: select the configured action chunk and feature extent."""

    def __init__(self, horizon: int, action_dim: int) -> None:
        super().__init__()
        self.horizon, self.action_dim = int(horizon), int(action_dim)

    def forward(self, flow_state):
        return flow_state[:, :self.horizon, :self.action_dim]


class SmolVLAMultiProgramCapture:
    """Deferred capture that quantizes shared SmolVLA weights once before all three programs."""

    def __init__(self, model, inputs, *, paper_ready: bool, full_checkpoint: bool,
                 provenance: dict) -> None:
        self.model, self.inputs = model, tuple(inputs)
        self.paper_ready, self.full_checkpoint = bool(paper_ready), bool(full_checkpoint)
        self.provenance = dict(provenance)

    def _external_runtime_session(self, *, quality_reference: np.ndarray | None = None):
        """Build the exact prefix/flow/action programs and functional tensor routing contract."""
        from m2m.capture.external_runtime import (
            ExternalRuntimeProgram, make_external_runtime_session,
        )

        prefix_inputs = self.inputs[:5]
        steps = int(self.model.config.num_steps)
        prefix = SmolVLAPrefixStage(self.model).eval()
        with torch.no_grad():
            prefix_masks, prefix_cache = prefix(*prefix_inputs)
        flow = SmolVLAFlowSession(self.model).eval()
        timestep = torch.ones(self.inputs[4].shape[0], dtype=torch.float32)
        flow_session = {
            "kind": "action_chunk", "paper_ready": self.paper_ready,
            "stages": ["flow_denoise"], "steps": steps,
            "stage_schedule": [{"name": "flow_denoise", "steps": steps,
                                "execution": "compiled_recurrent", "timed": True}],
            "parameters": {"denoise_steps": steps, "batch": 1},
            "states": [
                {"name": "prefix_kv_cache", "input_index": 1, "output_index": 1},
                {"name": "flow_state", "input_index": 2, "output_index": 0},
                {"name": "timestep", "input_index": 3, "output_index": 2},
            ],
            "streams": [],
            "quality": {"key": "actions", "output_index": 0, "reference": "eager_fp32"},
            "provenance": {"checkpoint": _MODEL_ID, "checkpoint_revision": _MODEL_REVISION,
                           "full_checkpoint": self.full_checkpoint,
                           **self.provenance},
        }
        if quality_reference is not None:
            flow_session["quality"]["reference_values"] = quality_reference
        horizon, action_dim = int(self.model.config.chunk_size), int(self.model.config.max_action_dim)
        action = SmolVLAActionDecodeStage(horizon, action_dim).eval()
        root = {
            "kind": "action_chunk", "paper_ready": self.paper_ready,
            "stages": ["prefix_encode", "flow_denoise", "action_decode"],
            "stage_schedule": [
                {"name": "prefix_encode", "steps": 1,
                 "execution": "compiled", "timed": True},
                {"name": "flow_denoise", "steps": steps,
                 "execution": "compiled_recurrent", "timed": True},
                {"name": "action_decode", "steps": 1,
                 "execution": "compiled", "timed": True},
            ],
            "parameters": {
                "denoise_steps": steps, "action_horizon": horizon, "batch": 1,
                "prefix_policy": "compiled_in_continuous_session",
                "timed_stages": ["prefix_encode", "flow_denoise", "action_decode"],
                "diagnostic_timed_stages": ["flow_denoise"],
                "paper_primary_scope": "end_to_end",
            },
            "bindings": [
                {"name": "prefix_pad_masks",
                 "from": {"program": "prefix_encode", "output_index": 0},
                 "to": {"program": "flow_denoise", "input_index": 0}},
                {"name": "prefix_kv_cache",
                 "from": {"program": "prefix_encode", "output_index": 1},
                 "to": {"program": "flow_denoise", "input_index": 1}},
                {"name": "decoded_flow_state",
                 "from": {"program": "flow_denoise", "output_index": 0},
                 "to": {"program": "action_decode", "input_index": 0}},
            ],
            "states": ["prefix_kv_cache", "flow_state", "timestep"],
            "quality_program": "flow_denoise",
            "provenance": {"checkpoint": _MODEL_ID, "checkpoint_revision": _MODEL_REVISION,
                           "full_checkpoint": self.full_checkpoint,
                           **self.provenance},
        }
        return make_external_runtime_session(
            version=2,
            programs=(
                ExternalRuntimeProgram("prefix_encode", prefix, prefix_inputs, 1),
                ExternalRuntimeProgram(
                    "flow_denoise", flow,
                    (prefix_masks, prefix_cache, self.inputs[5], timestep), steps, flow_session),
                ExternalRuntimeProgram(
                    "action_decode", action, (self.inputs[5],), 1),
            ),
            metadata=root,
        )

    def external_runtime_session(self):
        """Public model-agnostic capability consumed by external runtime exporters."""
        return self._external_runtime_session()

    def write_bundle(self, out: str | Path, *, quant=None) -> dict:
        # Capture FP32 quality before optional in-place torchAO conversion.
        reference = self.external_runtime_session()
        flow_program = reference.programs[1]
        from m2m.capture.bundle import capture_session_trajectory
        quality_reference = capture_session_trajectory(
            flow_program.module, flow_program.inputs, dict(flow_program.session or {}))
        del reference, flow_program

        if quant is not None:
            from m2m.capture.torchao_pipeline import apply_quantization
            self.model = apply_quantization(self.model, quant)
        capture = self._external_runtime_session(quality_reference=quality_reference)
        from m2m.capture.bundle import write_multi_program_bundle
        return write_multi_program_bundle(
            capture.bundle_programs(), dict(capture.metadata), out, quant=quant,
            quantization_preapplied=quant is not None)


def _input_tensors(cfg: SmolVLAConfig) -> tuple[tuple[torch.Tensor, ...], dict, bool]:
    source_value = os.environ.get("M2M_SMOLVLA_INPUT_NPZ", "")
    source_name = os.environ.get("M2M_SMOLVLA_INPUT_SOURCE", "")
    requested_ready = os.environ.get("M2M_SMOLVLA_PAPER_READY", "0") == "1"
    if source_value:
        source = Path(source_value).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"SmolVLA input corpus is absent: {source}")
        with np.load(source) as data:
            required = ("image", "image_mask", "language_tokens", "language_mask", "state", "noise")
            missing = [key for key in required if key not in data.files]
            if missing:
                raise ValueError(f"SmolVLA input npz omits {missing}")
            arrays = tuple(np.ascontiguousarray(data[key]) for key in required)
        inputs = (
            torch.as_tensor(arrays[0], dtype=torch.float32),
            torch.as_tensor(arrays[1], dtype=torch.bool),
            torch.as_tensor(arrays[2], dtype=torch.long),
            torch.as_tensor(arrays[3], dtype=torch.bool),
            torch.as_tensor(arrays[4], dtype=torch.float32),
            torch.as_tensor(arrays[5], dtype=torch.float32),
        )
        real_inputs = bool(source_name)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        source_text = source_name or str(source)
    else:
        b = 1
        configured_size = cfg.resize_imgs_with_padding[0] if cfg.resize_imgs_with_padding else 512
        height = width = int(os.environ.get("M2M_SMOLVLA_IMAGE_SIZE", configured_size))
        tok = cfg.tokenizer_max_length
        state_dim, horizon, action_dim = cfg.max_state_dim, cfg.chunk_size, cfg.max_action_dim
        inputs = (
            torch.rand(b, 3, height, width, dtype=torch.float32) * 2 - 1,
            torch.ones(b, dtype=torch.bool),
            torch.randint(0, 256, (b, tok), dtype=torch.long),
            torch.ones(b, tok, dtype=torch.bool),
            torch.randn(b, state_dim, dtype=torch.float32),
            torch.randn(b, horizon, action_dim, dtype=torch.float32),
        )
        real_inputs = False
        digest = hashlib.sha256(b"synthetic_seeded_by_capture_worker").hexdigest()
        source_text = "synthetic_random"
    if requested_ready and not real_inputs:
        raise ValueError(
            "M2M_SMOLVLA_PAPER_READY=1 requires M2M_SMOLVLA_INPUT_NPZ and "
            "M2M_SMOLVLA_INPUT_SOURCE")
    image, image_mask, language, language_mask, state, noise = inputs
    expected_size = int(os.environ.get(
        "M2M_SMOLVLA_IMAGE_SIZE",
        cfg.resize_imgs_with_padding[0] if cfg.resize_imgs_with_padding else 512))
    expected = {
        "image": (1, 3, expected_size, expected_size),
        "image_mask": (1,),
        "language_tokens": (1, int(cfg.tokenizer_max_length)),
        "language_mask": (1, int(cfg.tokenizer_max_length)),
        "state": (1, int(cfg.max_state_dim)),
        "noise": (1, int(cfg.chunk_size), int(cfg.max_action_dim)),
    }
    actual = {
        "image": tuple(image.shape), "image_mask": tuple(image_mask.shape),
        "language_tokens": tuple(language.shape), "language_mask": tuple(language_mask.shape),
        "state": tuple(state.shape), "noise": tuple(noise.shape),
    }
    mismatches = [f"{name}: expected {expected[name]}, got {actual[name]}"
                  for name in expected if expected[name] != actual[name]]
    if mismatches:
        raise ValueError("SmolVLA input ABI mismatch: " + "; ".join(mismatches))
    return inputs, {"input_source": source_text, "input_sha256": digest,
                    "synthetic_inputs": not real_inputs}, requested_ready and real_inputs


def _prefix_state(model: VLAFlowMatching, inputs: tuple[torch.Tensor, ...]) \
        -> tuple[torch.Tensor, torch.Tensor]:
    image, image_mask, lang_tokens, lang_masks, state, _noise = inputs
    with torch.no_grad():
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            [image], [image_mask], lang_tokens, lang_masks, state=state)
        attention = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        positions = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, cache = model.vlm_with_expert.forward(
            attention_mask=attention, position_ids=positions, past_key_values=None,
            inputs_embeds=[prefix_embs, None], use_cache=True, fill_kv_cache=True)
    expected = list(range(int(model.config.num_vlm_layers)))
    if sorted(cache) != expected:
        raise ValueError(f"SmolVLA prefix cache layers differ: got {sorted(cache)}, expected {expected}")
    keys = torch.stack([cache[layer]["key_states"].detach().clone() for layer in expected])
    values = torch.stack([cache[layer]["value_states"].detach().clone() for layer in expected])
    return prefix_pad_masks.detach().clone(), torch.stack((keys, values))


def get_model_and_inputs():
    pretrained = os.environ.get("M2M_SMOLVLA_PRETRAINED", "0") == "1"
    if pretrained:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        cfg = _load_pretrained_config()
        cfg.device = "cpu"
        policy = SmolVLAPolicy.from_pretrained(
            _MODEL_ID, config=cfg, revision=_MODEL_REVISION, local_files_only=True, strict=True)
        model = policy.model.cpu().eval()
    else:
        cfg = SmolVLAConfig()
        cfg.device = "cpu"
        # Smoke path: real structure, random weights, optionally fewer transformer layers.
        cfg.num_vlm_layers = int(os.environ.get("M2M_SMOLVLA_VLM_LAYERS", cfg.num_vlm_layers))
        if os.environ.get("M2M_SMOLVLA_EXPERT_LAYERS"):
            cfg.num_expert_layers = int(os.environ["M2M_SMOLVLA_EXPERT_LAYERS"])
        model = VLAFlowMatching(cfg).eval()

    inputs, input_provenance, inputs_ready = _input_tensors(cfg)
    session_mode = os.environ.get("M2M_SMOLVLA_SESSION")
    if session_mode not in {"denoise", "e2e"}:
        return SmolVLADenoiseStep(model).eval(), inputs

    full_checkpoint = pretrained and not os.environ.get("M2M_SMOLVLA_VLM_LAYERS") \
        and not os.environ.get("M2M_SMOLVLA_EXPERT_LAYERS")
    if inputs_ready and not full_checkpoint:
        raise ValueError(
            "M2M_SMOLVLA_PAPER_READY=1 requires the complete pretrained checkpoint")
    if session_mode == "e2e":
        return SmolVLAMultiProgramCapture(
            model, inputs, paper_ready=bool(full_checkpoint and inputs_ready),
            full_checkpoint=full_checkpoint, provenance=input_provenance), ()
    prefix_masks, prefix_cache = _prefix_state(model, inputs)
    timestep = torch.ones(inputs[4].shape[0], dtype=torch.float32)
    wrapper = SmolVLAFlowSession(model).eval()
    steps = int(model.config.num_steps)
    wrapper.session_spec = {
        "kind": "action_chunk",
        "paper_ready": bool(full_checkpoint and inputs_ready),
        "stages": ["prefix_encode", "flow_denoise", "action_decode"],
        "steps": steps,
        "stage_schedule": [
            {"name": "prefix_encode", "steps": 1,
             "execution": "eager_reference_initial_state", "timed": False},
            {"name": "flow_denoise", "steps": steps,
             "execution": "compiled_recurrent", "timed": True},
            {"name": "action_decode", "steps": 1,
             "execution": "identity_final_flow_state", "timed": False},
        ],
        "parameters": {
            "denoise_steps": steps, "action_horizon": int(model.config.chunk_size), "batch": 1,
            "prefix_policy": "untimed_eager_reference_initial_state",
            "timed_stages": ["flow_denoise"],
        },
        "states": [
            {"name": "prefix_kv_cache", "input_index": 1, "output_index": 1},
            {"name": "flow_state", "input_index": 2, "output_index": 0},
            {"name": "timestep", "input_index": 3, "output_index": 2},
        ],
        "streams": [],
        "quality": {"key": "actions", "output_index": 0},
        "provenance": {"checkpoint": _MODEL_ID, "checkpoint_revision": _MODEL_REVISION,
                       "full_checkpoint": bool(full_checkpoint),
                       **input_provenance},
    }
    return wrapper, (prefix_masks, prefix_cache, inputs[5], timestep)


def get_session_spec(model: nn.Module, _inputs: tuple[torch.Tensor, ...]) -> dict | None:
    return getattr(model, "session_spec", None)
