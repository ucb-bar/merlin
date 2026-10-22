"""Bounded checkpoint slices for compiler development, independent of policy admission.

These are explicitly isolated workloads, not complete-model quality benchmarks.
Only selected safetensors slices are materialized; no model loader is called.
"""

import hashlib
import json
import math
from pathlib import Path
import struct


def tensor_directory(path):
    """Read a bounded safetensors header without touching weight payloads."""
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < length <= 8 << 20:
            raise ValueError("Unexpected safetensors header extent")
        raw = stream.read(length)
        if len(raw) != length:
            raise ValueError("Truncated safetensors header")
    return path, json.loads(raw), hashlib.sha256(raw).hexdigest()


def workload_geometry(kind, *, tokens=4, action_chunk_size=50, action_hidden_size=720,
                      tiny_hidden_size=2048, tiny_intermediate_size=5632):
    """Admit explicit checkpoint prefixes; original geometry remains the default."""
    dimensions = ((tokens, 1, 16), (action_hidden_size, 1, 720),
                  (tiny_hidden_size, 64, 2048), (tiny_intermediate_size, 16, 5632))
    if any(type(value) is not int or not low <= value <= high for value, low, high in dimensions):
        raise ValueError("Reduced dimensions must be positive bounded integer checkpoint prefixes")
    if action_chunk_size not in (1, 50) or (kind != "smolvla-action" and
                                           (action_chunk_size != 50 or action_hidden_size != 720)):
        raise ValueError("Only the isolated action workload admits action geometry overrides")
    if kind != "tinyllama-head-ffn" and (tiny_hidden_size != 2048 or tiny_intermediate_size != 5632):
        raise ValueError("Only TinyLlama admits hidden/FFN prefix overrides")
    return {"tokens": tokens, "action_chunk_size": action_chunk_size,
            "action_hidden_size": action_hidden_size, "tiny_hidden_size": tiny_hidden_size,
            "tiny_intermediate_size": tiny_intermediate_size}


def preflight(kind, path, *, tokens=4, action_chunk_size=50, action_hidden_size=720,
              tiny_hidden_size=2048, tiny_intermediate_size=5632):
    geometry = workload_geometry(kind, tokens=tokens, action_chunk_size=action_chunk_size,
                                 action_hidden_size=action_hidden_size, tiny_hidden_size=tiny_hidden_size,
                                 tiny_intermediate_size=tiny_intermediate_size)
    path, directory, header_sha = tensor_directory(path)
    def require(name, shape):
        entry = directory.get(name, {})
        if entry.get("shape") != shape or entry.get("dtype") not in ("F32", "BF16"):
            raise ValueError("Unsupported tensor geometry or dtype: " + name)

    if kind == "smolvla-action":
        names = ["model.action_out_proj.weight", "model.action_out_proj.bias"]
        require(names[0], [32, 720])
        require(names[1], [32])
        elements = 32 * action_hidden_size + 32
        macs = action_chunk_size * action_hidden_size * 32
    elif kind == "tinyllama-head-ffn":
        names = [name for name in directory if name.startswith("model.layers.0.")]
        if len(names) != 9:
            raise ValueError("Expected the nine original layer-zero tensors")
        prefix = "model.layers.0."
        for name in ("input_layernorm", "post_attention_layernorm"):
            require(prefix + name + ".weight", [2048])
        for name, shape in (("q", [2048, 2048]), ("k", [256, 2048]), ("v", [256, 2048]), ("o", [2048, 2048])):
            require(prefix + "self_attn." + name + "_proj.weight", shape)
        for name, shape in (("gate", [5632, 2048]), ("up", [5632, 2048]), ("down", [2048, 5632])):
            require(prefix + "mlp." + name + "_proj.weight", shape)
        # Requested FFN prefixes, normalization prefixes, head zero and O columns.
        elements = 3 * tiny_hidden_size * tiny_intermediate_size + 4 * tiny_hidden_size * 64 + 2 * tiny_hidden_size
        macs = tokens * (elements - 2 * tiny_hidden_size) + 2 * tokens * tokens * 64
    elif kind in ("smolvla-denoise", "smolvla-block-action"):
        layers = 16 if kind == "smolvla-denoise" else 1
        action = ("model.action_in_proj.", "model.action_time_mlp_in.",
                  "model.action_time_mlp_out.", "model.action_out_proj.")
        selected = tuple("model.vlm_with_expert.lm_expert.layers." + str(index) + "." for index in range(layers))
        names = [name for name in directory if name.startswith(selected) or name == "model.vlm_with_expert.lm_expert.norm.weight" or name.startswith(action)]
        if len(names) != 9 * layers + 9:
            raise ValueError("Unexpected selected expert/action tensor inventory")
        for index in range(layers):
            prefix = "model.vlm_with_expert.lm_expert.layers." + str(index) + "."
            for name in ("input_layernorm", "post_attention_layernorm"):
                require(prefix + name + ".weight", [720])
            for name, shape in (("q", [960, 720]), ("k", [320, 720 if index % 2 == 0 else 320]),
                                ("v", [320, 720 if index % 2 == 0 else 320]), ("o", [720, 960])):
                require(prefix + "self_attn." + name + "_proj.weight", shape)
            for name, shape in (("gate", [2048, 720]), ("up", [2048, 720]), ("down", [720, 2048])):
                require(prefix + "mlp." + name + "_proj.weight", shape)
        require("model.vlm_with_expert.lm_expert.norm.weight", [720])
        for name, shape in (("action_in_proj", [720, 32]), ("action_time_mlp_in", [720, 1440]),
                            ("action_time_mlp_out", [720, 720]), ("action_out_proj", [32, 720])):
            require("model." + name + ".weight", shape)
            require("model." + name + ".bias", [shape[0]])
        elements = sum(math.prod(directory[name]["shape"]) for name in names)
        macs = 50 * sum(math.prod(directory[name]["shape"]) for name in names if len(directory[name]["shape"]) == 2)
    else:
        raise ValueError("Unknown bounded workload")
    selected_fp32 = elements * 4
    estimate = (1920 << 20) + 7 * selected_fp32
    scope = "Planning allowance updated after the first capped capture/export: 1920 MiB imports/compiler plus seven selected FP32 copies; measured cgroup peak remains authoritative"
    if kind in ("smolvla-denoise", "smolvla-block-action"):
        estimate = (1920 << 20) + 2 * selected_fp32 + (32 << 20)
        scope = "Sequential FP32 reference then W8A8 assembly: 1920 MiB base plus retained FP32 allocator allowance, four INT8 copies and 32 MiB scratch; no simultaneous full FP32 model/export copies"
    return {"workload": kind, "checkpoint": str(path), "checkpoint_file_bytes": path.stat().st_size,
            "header_sha256": header_sha, "selected_names": names, "selected_fp32_bytes_upper_bound": selected_fp32,
            "host_peak_estimate_bytes": estimate, "estimate_scope": scope,
            "contraction_macs_per_call": macs, "tokens": tokens, "action_chunk_size": action_chunk_size if kind == "smolvla-action" else None,
            "requested_geometry": geometry,
            "full_checkpoint_load": False, "model_quality_qualification": False}


def make_workload(kind, checkpoint, *, quantized, tokens=4, seed=20261019, action_chunk_size=50,
                  action_hidden_size=720, tiny_hidden_size=2048, tiny_intermediate_size=5632):
    geometry = workload_geometry(kind, tokens=tokens, action_chunk_size=action_chunk_size,
                                 action_hidden_size=action_hidden_size, tiny_hidden_size=tiny_hidden_size,
                                 tiny_intermediate_size=tiny_intermediate_size)
    import numpy as np
    import torch
    from safetensors import safe_open
    from tinyllama_integer import IntegerProjection

    selected = []

    def load(name, selection=None):
        with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
            value = source.get_tensor(name) if selection is None else source.get_slice(name)[selection]
        original_dtype = str(value.dtype)
        value = value.float().contiguous()
        selected.append({"name": name, "slice": str(selection), "shape": list(value.shape),
                         "original_dtype": original_dtype, "execution_dtype": "float32",
                         "slice_fp32_sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest()})
        return value

    class Linear(torch.nn.Module):
        def __init__(self, weight, bias=None):
            super().__init__()
            self.quantized = quantized
            self.register_buffer("bias", bias)
            if quantized:
                self.integer = IntegerProjection(weight, activation_scale="power_of_two")
            else:
                self.register_buffer("weight", weight)

        def forward(self, value):
            if self.quantized:
                codes, scale = self.integer.integer_inputs(value)
                accumulator = torch._int_mm(codes, self.integer.weight_kn)
                result = accumulator.reshape(*value.shape[:-1], self.integer.out_features).float()
                result = result * scale * self.integer.weight_scale
            else:
                result = torch.nn.functional.linear(value, self.weight)
                accumulator = torch.zeros(1, dtype=torch.int32)
            return result if self.bias is None else result + self.bias, accumulator

    generator = torch.Generator().manual_seed(seed)
    if kind == "smolvla-action":
        projection = Linear(load("model.action_out_proj.weight", (slice(None), slice(0, action_hidden_size))),
                            load("model.action_out_proj.bias"))

        class Action(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = projection

            def forward(self, hidden, noisy_actions, dt):
                velocity, integer = self.projection(hidden)
                updated = noisy_actions + dt * velocity
                return velocity, updated, updated[:, :, :6], integer

        module = Action().eval()
        inputs = (torch.randn(1, action_chunk_size, action_hidden_size, generator=generator),
                  torch.randn(1, action_chunk_size, 32, generator=generator), torch.tensor([-0.1]))
        description = {"scope": "Action projection checkpoint-column prefix, one Euler update and six-dimensional action slice; expert hidden state is an external fixture",
                       "complete_denoiser": False, "chunk_size": action_chunk_size, "hidden_size": action_hidden_size,
                       "original_hidden_size": 720, "reduced_dimension_diagnostic": action_hidden_size != 720,
                       "supplementary_one_position_diagnostic": action_chunk_size == 1,
                       "float_output_names": ["velocity", "updated_latent", "action_slice"], "integer_outputs": [3],
                       "expected_gemmini_contractions": 1}
    elif kind in ("smolvla-denoise", "smolvla-block-action"):
        from smolvla_denoise_block import make_denoiser
        module, inputs, description = make_denoiser(torch, Linear, load, generator,
                                                  expert_layers=16 if kind == "smolvla-denoise" else 1)
    elif kind == "tinyllama-head-ffn":
        prefix = "model.layers.0."
        projections = {}
        for name in ("q", "k", "v"):
            projections[name] = Linear(load(prefix + "self_attn." + name + "_proj.weight", (slice(0, 64), slice(0, tiny_hidden_size))))
        projections["o"] = Linear(load(prefix + "self_attn.o_proj.weight", (slice(0, tiny_hidden_size), slice(0, 64))))
        for name in ("gate", "up"):
            projections[name] = Linear(load(prefix + "mlp." + name + "_proj.weight",
                                            (slice(0, tiny_intermediate_size), slice(0, tiny_hidden_size))))
        projections["down"] = Linear(load(prefix + "mlp.down_proj.weight",
                                         (slice(0, tiny_hidden_size), slice(0, tiny_intermediate_size))))
        before = load(prefix + "input_layernorm.weight", slice(0, tiny_hidden_size))
        after = load(prefix + "post_attention_layernorm.weight", slice(0, tiny_hidden_size))

        class HeadFFN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projections = torch.nn.ModuleDict(projections)
                self.register_buffer("before", before)
                self.register_buffer("after", after)
                angles = torch.arange(tokens).float()[:, None] * (10000.0 ** (-torch.arange(0, 64, 2).float() / 64))[None, :]
                angles = torch.cat((angles, angles), dim=-1)
                self.register_buffer("cosine", angles.cos())
                self.register_buffer("sine", angles.sin())
                mask = torch.triu(torch.full((tokens, tokens), -1e9), diagonal=1)
                self.register_buffer("mask", mask)

            @staticmethod
            def normalize(value, weight):
                return value * torch.rsqrt((value * value).mean(dim=-1, keepdim=True) + 1e-5) * weight

            def rotate(self, value):
                rotated = torch.cat((-value[..., 32:], value[..., :32]), dim=-1)
                return value * self.cosine + rotated * self.sine

            def forward(self, hidden):
                normalized = self.normalize(hidden, self.before)
                q, qi = self.projections["q"](normalized)
                k, ki = self.projections["k"](normalized)
                v, vi = self.projections["v"](normalized)
                scores = self.rotate(q) @ self.rotate(k).transpose(-2, -1) * 0.125 + self.mask
                attention = torch.softmax(scores, dim=-1) @ v
                projected, oi = self.projections["o"](attention)
                residual = hidden + projected
                ffn_input = self.normalize(residual, self.after)
                gate, gi = self.projections["gate"](ffn_input)
                up, ui = self.projections["up"](ffn_input)
                ffn, di = self.projections["down"](torch.nn.functional.silu(gate) * up)
                return residual + ffn, attention, ffn, qi, ki, vi, oi, gi, ui, di

        module = HeadFFN().eval()
        inputs = (torch.randn(1, tokens, tiny_hidden_size, generator=generator),)
        description = {"scope": "Layer-zero query head zero with its shared KV head and O contribution, followed by all three FFN projections at explicitly selected checkpoint-prefix widths",
                       "complete_transformer_layer": False, "other_attention_heads_included": False,
                       "tokens": tokens, "hidden_size": tiny_hidden_size, "head_dim": 64, "intermediate_size": tiny_intermediate_size,
                       "original_hidden_size": 2048, "original_intermediate_size": 5632,
                       "reduced_dimension_diagnostic": (tiny_hidden_size, tiny_intermediate_size) != (2048, 5632),
                       "float_output_names": ["hidden", "attention_head", "ffn"], "integer_outputs": list(range(3, 10)),
                       "expected_gemmini_contractions": 7}
    else:
        raise ValueError("Unknown bounded workload")
    description.update(selected_tensor_slices=selected, quantized=quantized, fixture_seed=seed, requested_geometry=geometry,
                       fixture_kind="synthetic_inputs_pretrained_weight_slices", model_quality_qualified=False,
                       arithmetic="W8A8 learned projections, INT32 accumulation, FP32 dequantization and surrounding math" if quantized else "FP32 reference",
                       numpy_version=np.__version__)
    return module, inputs, description
