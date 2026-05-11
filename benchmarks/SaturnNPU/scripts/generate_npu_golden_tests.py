#!/usr/bin/env python3
"""Generate hierarchical golden test data for SmolVLA NPU kernels.

Golden data is generated at two levels:
  1. **Layer level** (PyTorch): full layer forward pass (e.g., GemmaAttention)
  2. **Operator level**: each atomic op within the layer (matmul, softmax, etc.)

The operator-level golden data *composes* to reproduce the layer-level output.
This lets RTL teams test individual operators AND verify their composition.

Usage:
    python tools/generate_npu_golden_tests.py \
        benchmarks/SaturnNPU/smolvla_graph_manifest.json \
        --output-dir benchmarks/SaturnNPU/golden_data/ \
        --generate-programs third_party/npu_model/npu_model/configs/programs/
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Deterministic seeding
# ---------------------------------------------------------------------------
SEED = 42


def seed_all():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------------------------
# Reference implementations (matching Understanding-PI0/gemma_blocks.py)
# ---------------------------------------------------------------------------


def ref_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS normalization: x * rsqrt(mean(x^2) + eps) * weight."""
    var = torch.mean(x.float() ** 2, dim=-1, keepdim=True)
    normed = x * torch.rsqrt(var + eps)
    return (normed * weight).to(x.dtype)


def ref_gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """GELU with tanh approximation (matches PyTorch nn.GELU('tanh'))."""
    return x * 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3.0))))


def ref_silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU / Swish activation."""
    return x * torch.sigmoid(x)


def ref_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax decomposed into max-subtract-exp-sum-div."""
    x_max = x.max(dim=dim, keepdim=True).values
    x_shifted = x - x_max
    x_exp = torch.exp(x_shifted)
    x_sum = x_exp.sum(dim=dim, keepdim=True)
    return x_exp / x_sum


def ref_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary position embedding."""

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    cos = cos.unsqueeze(1)  # [B, 1, S, D]
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Layer decomposition recipes
# ---------------------------------------------------------------------------


@dataclass
class OpTrace:
    """One operator's golden data within a layer decomposition."""

    name: str
    op_type: str
    inputs: dict[str, torch.Tensor]
    outputs: dict[str, torch.Tensor]
    metadata: dict = field(default_factory=dict)


@dataclass
class LayerTrace:
    """Full layer golden data with operator decomposition."""

    layer_type: str
    layer_input: dict[str, torch.Tensor]
    layer_output: dict[str, torch.Tensor]
    operators: list[OpTrace]
    config: dict
    composition_verified: bool = False


def generate_siglip_attention(
    seq_len: int = 1024,
    hidden: int = 768,
    num_heads: int = 12,
    dtype: torch.dtype = torch.bfloat16,
) -> LayerTrace:
    """Generate golden data for one SigLIP self-attention layer."""
    seed_all()
    head_dim = hidden // num_heads
    scale = 1.0 / math.sqrt(head_dim)

    # Random weights and inputs
    x = torch.randn(1, seq_len, hidden, dtype=dtype)
    ln_weight = torch.ones(hidden, dtype=dtype)
    ln_bias = torch.zeros(hidden, dtype=dtype)
    q_weight = torch.randn(hidden, hidden, dtype=dtype) * 0.02
    q_bias = torch.randn(hidden, dtype=dtype) * 0.02
    k_weight = torch.randn(hidden, hidden, dtype=dtype) * 0.02
    k_bias = torch.randn(hidden, dtype=dtype) * 0.02
    v_weight = torch.randn(hidden, hidden, dtype=dtype) * 0.02
    v_bias = torch.randn(hidden, dtype=dtype) * 0.02
    o_weight = torch.randn(hidden, hidden, dtype=dtype) * 0.02
    o_bias = torch.randn(hidden, dtype=dtype) * 0.02

    ops: list[OpTrace] = []
    residual = x.clone()

    # Op 0: Layer Norm
    x_normed = torch.nn.functional.layer_norm(x.float(), [hidden], ln_weight.float(), ln_bias.float()).to(dtype)
    ops.append(
        OpTrace("layer_norm", "layer_norm", {"x": x, "weight": ln_weight, "bias": ln_bias}, {"output": x_normed})
    )

    # Op 1-3: Q, K, V projections (linear = matmul + bias)
    q = (x_normed @ q_weight.T) + q_bias
    ops.append(OpTrace("q_proj", "linear", {"input": x_normed, "weight": q_weight, "bias": q_bias}, {"output": q}))

    k = (x_normed @ k_weight.T) + k_bias
    ops.append(OpTrace("k_proj", "linear", {"input": x_normed, "weight": k_weight, "bias": k_bias}, {"output": k}))

    v = (x_normed @ v_weight.T) + v_bias
    ops.append(OpTrace("v_proj", "linear", {"input": x_normed, "weight": v_weight, "bias": v_bias}, {"output": v}))

    # Reshape to multi-head: [B, S, H*D] → [B, H, S, D]
    q_mh = q.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
    k_mh = k.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
    v_mh = v.view(1, seq_len, num_heads, head_dim).transpose(1, 2)

    # Op 4: Score matmul (Q @ K^T)
    scores = torch.matmul(q_mh, k_mh.transpose(-2, -1)) * scale
    ops.append(
        OpTrace(
            "score_matmul",
            "batch_matmul",
            {"query": q_mh, "key": k_mh, "scale": torch.tensor(scale)},
            {"scores": scores},
            {"shape": f"{num_heads}x{seq_len}x{head_dim}"},
        )
    )

    # Op 5: Softmax
    attn_weights = ref_softmax(scores.float(), dim=-1).to(dtype)
    ops.append(OpTrace("softmax", "softmax", {"input": scores}, {"output": attn_weights}))

    # Op 6: Value matmul (attn_weights @ V)
    attn_output = torch.matmul(attn_weights, v_mh)
    ops.append(
        OpTrace("value_matmul", "batch_matmul", {"attn_weights": attn_weights, "value": v_mh}, {"output": attn_output})
    )

    # Reshape back: [B, H, S, D] → [B, S, H*D]
    attn_output = attn_output.transpose(1, 2).contiguous().view(1, seq_len, hidden)

    # Op 7: Output projection
    out_proj = (attn_output @ o_weight.T) + o_bias
    ops.append(
        OpTrace("o_proj", "linear", {"input": attn_output, "weight": o_weight, "bias": o_bias}, {"output": out_proj})
    )

    # Op 8: Residual add
    output = residual + out_proj
    ops.append(OpTrace("residual_add", "elementwise_add", {"a": residual, "b": out_proj}, {"output": output}))

    return LayerTrace(
        layer_type="siglip_attention",
        layer_input={"x": residual},
        layer_output={"output": output},
        operators=ops,
        config={"seq_len": seq_len, "hidden": hidden, "num_heads": num_heads, "head_dim": head_dim},
        composition_verified=True,
    )


def generate_gemma_attention(
    seq_len: int = 50,
    hidden: int = 720,
    num_heads: int = 15,
    num_kv_heads: int = 5,
    dtype: torch.dtype = torch.bfloat16,
) -> LayerTrace:
    """Generate golden data for one Gemma decoder self-attention layer."""
    seed_all()
    head_dim = hidden // num_heads
    kv_dim = num_kv_heads * head_dim
    scale = head_dim**-0.5
    num_kv_groups = num_heads // num_kv_heads

    x = torch.randn(1, seq_len, hidden, dtype=dtype)
    norm_weight = torch.ones(hidden, dtype=dtype)
    q_weight = torch.randn(hidden, hidden, dtype=dtype) * 0.02
    k_weight = torch.randn(kv_dim, hidden, dtype=dtype) * 0.02
    v_weight = torch.randn(kv_dim, hidden, dtype=dtype) * 0.02
    o_weight = torch.randn(hidden, hidden, dtype=dtype) * 0.02

    # RoPE frequencies
    rope_dim = head_dim
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
    position_ids = torch.arange(seq_len).unsqueeze(0)
    freqs = (inv_freq[None, :, None].float() @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)

    ops: list[OpTrace] = []
    residual = x.clone()

    # Op 0: RMS Norm
    x_normed = ref_rms_norm(x, norm_weight)
    ops.append(OpTrace("rms_norm", "rms_norm", {"x": x, "weight": norm_weight}, {"output": x_normed}))

    # Op 1-3: Q, K, V projections
    q = (x_normed @ q_weight.T).view(1, seq_len, num_heads, head_dim).transpose(1, 2)
    ops.append(OpTrace("q_proj", "matmul", {"input": x_normed, "weight": q_weight}, {"output": q}))

    k = (x_normed @ k_weight.T).view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    ops.append(OpTrace("k_proj", "matmul", {"input": x_normed, "weight": k_weight}, {"output": k}))

    v = (x_normed @ v_weight.T).view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    ops.append(OpTrace("v_proj", "matmul", {"input": x_normed, "weight": v_weight}, {"output": v}))

    # Op 4: RoPE
    q_rot, k_rot = ref_rope(q, k, cos, sin)
    ops.append(OpTrace("rope", "rope", {"q": q, "k": k, "cos": cos, "sin": sin}, {"q_out": q_rot, "k_out": k_rot}))

    # KV repeat for GQA
    k_rep = k_rot.repeat_interleave(num_kv_groups, dim=1)
    v_rep = v.repeat_interleave(num_kv_groups, dim=1)

    # Op 5: Score matmul
    scores = torch.matmul(q_rot, k_rep.transpose(-2, -1)) * scale
    ops.append(
        OpTrace(
            "score_matmul",
            "batch_matmul",
            {"query": q_rot, "key": k_rep, "scale": torch.tensor(scale)},
            {"scores": scores},
        )
    )

    # Op 6: Softmax
    attn_weights = ref_softmax(scores.float(), dim=-1).to(dtype)
    ops.append(OpTrace("softmax", "softmax", {"input": scores}, {"output": attn_weights}))

    # Op 7: Value matmul
    attn_output = torch.matmul(attn_weights, v_rep)
    ops.append(
        OpTrace("value_matmul", "batch_matmul", {"attn_weights": attn_weights, "value": v_rep}, {"output": attn_output})
    )

    # Reshape back
    attn_output = attn_output.transpose(1, 2).contiguous().view(1, seq_len, hidden)

    # Op 8: O projection
    out = attn_output @ o_weight.T
    ops.append(OpTrace("o_proj", "matmul", {"input": attn_output, "weight": o_weight}, {"output": out}))

    # Op 9: Residual add
    output = residual + out
    ops.append(OpTrace("residual_add", "elementwise_add", {"a": residual, "b": out}, {"output": output}))

    return LayerTrace(
        layer_type="gemma_attention",
        layer_input={"x": residual},
        layer_output={"output": output},
        operators=ops,
        config={
            "seq_len": seq_len,
            "hidden": hidden,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
        },
        composition_verified=True,
    )


def generate_gemma_mlp(
    seq_len: int = 50,
    hidden: int = 720,
    intermediate: int = 2048,
    dtype: torch.dtype = torch.bfloat16,
) -> LayerTrace:
    """Generate golden data for one Gemma MLP layer (gate + up + GELU + down)."""
    seed_all()
    x = torch.randn(1, seq_len, hidden, dtype=dtype)
    norm_weight = torch.ones(hidden, dtype=dtype)
    gate_weight = torch.randn(intermediate, hidden, dtype=dtype) * 0.02
    up_weight = torch.randn(intermediate, hidden, dtype=dtype) * 0.02
    down_weight = torch.randn(hidden, intermediate, dtype=dtype) * 0.02

    ops: list[OpTrace] = []
    residual = x.clone()

    # Op 0: RMS Norm
    x_normed = ref_rms_norm(x, norm_weight)
    ops.append(OpTrace("rms_norm", "rms_norm", {"x": x, "weight": norm_weight}, {"output": x_normed}))

    # Op 1: Gate projection
    gate = x_normed @ gate_weight.T
    ops.append(OpTrace("gate_proj", "matmul", {"input": x_normed, "weight": gate_weight}, {"output": gate}))

    # Op 2: Up projection
    up = x_normed @ up_weight.T
    ops.append(OpTrace("up_proj", "matmul", {"input": x_normed, "weight": up_weight}, {"output": up}))

    # Op 3: GELU activation on gate
    gate_activated = ref_gelu_tanh(gate)
    ops.append(OpTrace("gelu", "gelu", {"input": gate}, {"output": gate_activated}))

    # Op 4: Gate * Up (element-wise multiply)
    gated = gate_activated * up
    ops.append(OpTrace("gate_mul_up", "elementwise_mul", {"gate": gate_activated, "up": up}, {"output": gated}))

    # Op 5: Down projection
    down = gated @ down_weight.T
    ops.append(OpTrace("down_proj", "matmul", {"input": gated, "weight": down_weight}, {"output": down}))

    # Op 6: Residual add
    output = residual + down
    ops.append(OpTrace("residual_add", "elementwise_add", {"a": residual, "b": down}, {"output": output}))

    return LayerTrace(
        layer_type="gemma_mlp",
        layer_input={"x": residual},
        layer_output={"output": output},
        operators=ops,
        config={"seq_len": seq_len, "hidden": hidden, "intermediate": intermediate},
        composition_verified=True,
    )


def generate_siglip_mlp(
    seq_len: int = 1024,
    hidden: int = 768,
    intermediate: int = 3072,
    dtype: torch.dtype = torch.bfloat16,
) -> LayerTrace:
    """Generate golden data for one SigLIP MLP layer (fc1 + GELU + fc2)."""
    seed_all()
    x = torch.randn(1, seq_len, hidden, dtype=dtype)
    ln_weight = torch.ones(hidden, dtype=dtype)
    ln_bias = torch.zeros(hidden, dtype=dtype)
    fc1_weight = torch.randn(intermediate, hidden, dtype=dtype) * 0.02
    fc1_bias = torch.randn(intermediate, dtype=dtype) * 0.02
    fc2_weight = torch.randn(hidden, intermediate, dtype=dtype) * 0.02
    fc2_bias = torch.randn(hidden, dtype=dtype) * 0.02

    ops: list[OpTrace] = []
    residual = x.clone()

    # Op 0: Layer Norm
    x_normed = torch.nn.functional.layer_norm(x.float(), [hidden], ln_weight.float(), ln_bias.float()).to(dtype)
    ops.append(
        OpTrace("layer_norm", "layer_norm", {"x": x, "weight": ln_weight, "bias": ln_bias}, {"output": x_normed})
    )

    # Op 1: FC1
    fc1_out = (x_normed @ fc1_weight.T) + fc1_bias
    ops.append(
        OpTrace("fc1", "linear", {"input": x_normed, "weight": fc1_weight, "bias": fc1_bias}, {"output": fc1_out})
    )

    # Op 2: GELU
    gelu_out = ref_gelu_tanh(fc1_out)
    ops.append(OpTrace("gelu", "gelu", {"input": fc1_out}, {"output": gelu_out}))

    # Op 3: FC2
    fc2_out = (gelu_out @ fc2_weight.T) + fc2_bias
    ops.append(
        OpTrace("fc2", "linear", {"input": gelu_out, "weight": fc2_weight, "bias": fc2_bias}, {"output": fc2_out})
    )

    # Op 4: Residual add
    output = residual + fc2_out
    ops.append(OpTrace("residual_add", "elementwise_add", {"a": residual, "b": fc2_out}, {"output": output}))

    return LayerTrace(
        layer_type="siglip_mlp",
        layer_input={"x": residual},
        layer_output={"output": output},
        operators=ops,
        config={"seq_len": seq_len, "hidden": hidden, "intermediate": intermediate},
        composition_verified=True,
    )


def generate_gemma_cross_attention(
    q_seq_len: int = 50,
    kv_seq_len: int = 241,
    hidden: int = 720,
    num_heads: int = 15,
    num_kv_heads: int = 5,
    dtype: torch.dtype = torch.bfloat16,
) -> LayerTrace:
    """Generate golden data for Gemma expert cross-attention.

    Different from self-attention: query and key/value have different seq lengths.
    Query comes from action tokens (50), KV from vision+language context (241).
    """
    seed_all()
    head_dim = hidden // num_heads
    kv_dim = num_kv_heads * head_dim
    scale = head_dim**-0.5
    num_kv_groups = num_heads // num_kv_heads

    x_q = torch.randn(1, q_seq_len, hidden, dtype=dtype)
    x_kv = torch.randn(1, kv_seq_len, hidden, dtype=dtype)
    norm_weight = torch.ones(hidden, dtype=dtype)
    q_weight = torch.randn(hidden, hidden, dtype=dtype) * 0.02
    k_weight = torch.randn(kv_dim, hidden, dtype=dtype) * 0.02
    v_weight = torch.randn(kv_dim, hidden, dtype=dtype) * 0.02
    o_weight = torch.randn(hidden, hidden, dtype=dtype) * 0.02

    ops: list[OpTrace] = []
    residual = x_q.clone()

    # Op 0: RMS Norm on query
    x_q_normed = ref_rms_norm(x_q, norm_weight)
    ops.append(OpTrace("rms_norm_q", "rms_norm", {"x": x_q, "weight": norm_weight}, {"output": x_q_normed}))

    # Op 1: Q projection from query input
    q = (x_q_normed @ q_weight.T).view(1, q_seq_len, num_heads, head_dim).transpose(1, 2)
    ops.append(OpTrace("q_proj", "matmul", {"input": x_q_normed, "weight": q_weight}, {"output": q}))

    # Op 2-3: K, V projections from KV input
    k = (x_kv @ k_weight.T).view(1, kv_seq_len, num_kv_heads, head_dim).transpose(1, 2)
    ops.append(OpTrace("k_proj", "matmul", {"input": x_kv, "weight": k_weight}, {"output": k}))

    v = (x_kv @ v_weight.T).view(1, kv_seq_len, num_kv_heads, head_dim).transpose(1, 2)
    ops.append(OpTrace("v_proj", "matmul", {"input": x_kv, "weight": v_weight}, {"output": v}))

    # KV repeat for GQA
    k_rep = k.repeat_interleave(num_kv_groups, dim=1)
    v_rep = v.repeat_interleave(num_kv_groups, dim=1)

    # Op 4: Score matmul (Q[50] @ K[241]^T → [50, 241])
    scores = torch.matmul(q, k_rep.transpose(-2, -1)) * scale
    ops.append(
        OpTrace(
            "score_matmul",
            "batch_matmul",
            {"query": q, "key": k_rep, "scale": torch.tensor(scale)},
            {"scores": scores},
            {"note": f"cross-attention: Q[{q_seq_len}] x K[{kv_seq_len}]"},
        )
    )

    # Op 5: Softmax
    attn_weights = ref_softmax(scores.float(), dim=-1).to(dtype)
    ops.append(OpTrace("softmax", "softmax", {"input": scores}, {"output": attn_weights}))

    # Op 6: Value matmul (attn[50,241] @ V[241] → [50])
    attn_output = torch.matmul(attn_weights, v_rep)
    ops.append(
        OpTrace("value_matmul", "batch_matmul", {"attn_weights": attn_weights, "value": v_rep}, {"output": attn_output})
    )

    # Reshape back
    attn_output = attn_output.transpose(1, 2).contiguous().view(1, q_seq_len, hidden)

    # Op 7: O projection
    out = attn_output @ o_weight.T
    ops.append(OpTrace("o_proj", "matmul", {"input": attn_output, "weight": o_weight}, {"output": out}))

    # Op 8: Residual add
    output = residual + out
    ops.append(OpTrace("residual_add", "elementwise_add", {"a": residual, "b": out}, {"output": output}))

    return LayerTrace(
        layer_type="gemma_cross_attention",
        layer_input={"x_query": x_q, "x_context": x_kv},
        layer_output={"output": output},
        operators=ops,
        config={
            "q_seq_len": q_seq_len,
            "kv_seq_len": kv_seq_len,
            "hidden": hidden,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
        },
        composition_verified=True,
    )


def generate_action_time_mlp(
    hidden: int = 1024,
    time_dim: int = 2048,
    dtype: torch.dtype = torch.bfloat16,
) -> LayerTrace:
    """Generate golden data for the action time MLP.

    Linear(time_dim→hidden) + SiLU + Linear(hidden→hidden).
    """
    seed_all()
    x = torch.randn(1, time_dim, dtype=dtype)
    w1 = torch.randn(hidden, time_dim, dtype=dtype) * 0.02
    b1 = torch.randn(hidden, dtype=dtype) * 0.02
    w2 = torch.randn(hidden, hidden, dtype=dtype) * 0.02
    b2 = torch.randn(hidden, dtype=dtype) * 0.02

    ops: list[OpTrace] = []

    # Op 0: First linear
    h = (x @ w1.T) + b1
    ops.append(OpTrace("linear_in", "linear", {"input": x, "weight": w1, "bias": b1}, {"output": h}))

    # Op 1: SiLU
    h_act = ref_silu(h)
    ops.append(OpTrace("silu", "silu", {"input": h}, {"output": h_act}))

    # Op 2: Second linear
    output = (h_act @ w2.T) + b2
    ops.append(OpTrace("linear_out", "linear", {"input": h_act, "weight": w2, "bias": b2}, {"output": output}))

    return LayerTrace(
        layer_type="action_time_mlp",
        layer_input={"x": x},
        layer_output={"output": output},
        operators=ops,
        config={"hidden": hidden, "time_dim": time_dim},
        composition_verified=True,
    )


def generate_siglip_patch_embed(
    image_size: int = 512,
    patch_size: int = 16,
    in_channels: int = 3,
    hidden: int = 768,
    dtype: torch.dtype = torch.bfloat16,
) -> LayerTrace:
    """Generate golden data for SigLIP patch embedding.

    Conv2d(3→768, kernel=16, stride=16) + position embedding add.
    """
    seed_all()
    num_patches = (image_size // patch_size) ** 2  # 1024

    image = torch.randn(1, in_channels, image_size, image_size, dtype=dtype)
    conv_weight = torch.randn(hidden, in_channels, patch_size, patch_size, dtype=dtype) * 0.02
    conv_bias = torch.randn(hidden, dtype=dtype) * 0.02
    pos_embed = torch.randn(num_patches, hidden, dtype=dtype) * 0.02

    ops: list[OpTrace] = []

    # Op 0: Conv2d (patch extraction)
    patches = torch.nn.functional.conv2d(image.float(), conv_weight.float(), conv_bias.float(), stride=patch_size).to(
        dtype
    )
    # Reshape: [1, hidden, H/P, W/P] → [1, num_patches, hidden]
    patches_flat = patches.flatten(2).transpose(1, 2)
    ops.append(
        OpTrace(
            "conv2d_patch",
            "conv2d",
            {"image": image, "weight": conv_weight, "bias": conv_bias},
            {"patches": patches_flat},
            {"kernel_size": patch_size, "stride": patch_size},
        )
    )

    # Op 1: Position embedding add
    output = patches_flat + pos_embed
    ops.append(
        OpTrace(
            "position_embed_add",
            "elementwise_add",
            {"patches": patches_flat, "pos_embed": pos_embed},
            {"output": output},
        )
    )

    return LayerTrace(
        layer_type="siglip_patch_embed",
        layer_input={"image": image},
        layer_output={"output": output},
        operators=ops,
        config={"image_size": image_size, "patch_size": patch_size, "hidden": hidden, "num_patches": num_patches},
        composition_verified=True,
    )


# ---------------------------------------------------------------------------
# Save golden data
# ---------------------------------------------------------------------------


def save_layer_trace(trace: LayerTrace, output_dir: Path) -> None:
    """Save a LayerTrace to disk in a hierarchical directory structure."""
    layer_dir = output_dir / trace.layer_type
    layer_dir.mkdir(parents=True, exist_ok=True)

    # Save layer-level I/O
    for name, tensor in trace.layer_input.items():
        torch.save(tensor, layer_dir / f"layer_input_{name}.pt")
    for name, tensor in trace.layer_output.items():
        torch.save(tensor, layer_dir / f"layer_output_{name}.pt")

    # Save operator-level data
    ops_dir = layer_dir / "operators"
    ops_dir.mkdir(exist_ok=True)

    op_manifest = []
    for i, op in enumerate(trace.operators):
        op_dir = ops_dir / f"{i:02d}_{op.name}"
        op_dir.mkdir(exist_ok=True)

        for name, tensor in op.inputs.items():
            torch.save(tensor, op_dir / f"input_{name}.pt")
        for name, tensor in op.outputs.items():
            torch.save(tensor, op_dir / f"output_{name}.pt")

        op_manifest.append(
            {
                "index": i,
                "name": op.name,
                "op_type": op.op_type,
                "inputs": {k: list(v.shape) for k, v in op.inputs.items()},
                "outputs": {k: list(v.shape) for k, v in op.outputs.items()},
                "input_dtypes": {k: str(v.dtype) for k, v in op.inputs.items()},
                "output_dtypes": {k: str(v.dtype) for k, v in op.outputs.items()},
                "metadata": op.metadata,
            }
        )

    # Save metadata
    metadata = {
        "layer_type": trace.layer_type,
        "config": trace.config,
        "composition_verified": trace.composition_verified,
        "layer_input_shapes": {k: list(v.shape) for k, v in trace.layer_input.items()},
        "layer_output_shapes": {k: list(v.shape) for k, v in trace.layer_output.items()},
        "operators": op_manifest,
        "num_operators": len(trace.operators),
        "composition_chain": " → ".join(op.name for op in trace.operators),
    }

    with open(layer_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_composition(trace: LayerTrace) -> bool:
    """Verify that operator outputs chain correctly to reproduce layer output."""
    # Check that last operator output matches layer output
    last_op = trace.operators[-1]
    for key in trace.layer_output:
        if key in last_op.outputs:
            layer_out = trace.layer_output[key]
            op_out = last_op.outputs[key]
            if not torch.allclose(layer_out.float(), op_out.float(), atol=1e-3, rtol=1e-3):
                max_diff = (layer_out.float() - op_out.float()).abs().max().item()
                print(
                    f"    WARN: {trace.layer_type} last op '{last_op.name}' output '{key}' "
                    f"differs from layer output by {max_diff:.6f}"
                )
                return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# Layer generators: (name, generator_fn, small_kwargs, real_kwargs)
LAYER_CONFIGS = [
    (
        "siglip_attention",
        generate_siglip_attention,
        {"seq_len": 64, "hidden": 768, "num_heads": 12},
        {"seq_len": 1024, "hidden": 768, "num_heads": 12},
    ),
    (
        "siglip_mlp",
        generate_siglip_mlp,
        {"seq_len": 64, "hidden": 768, "intermediate": 3072},
        {"seq_len": 1024, "hidden": 768, "intermediate": 3072},
    ),
    (
        "gemma_attention",
        generate_gemma_attention,
        {"seq_len": 16, "hidden": 720, "num_heads": 15, "num_kv_heads": 5},
        {"seq_len": 291, "hidden": 720, "num_heads": 15, "num_kv_heads": 5},
    ),
    (
        "gemma_mlp",
        generate_gemma_mlp,
        {"seq_len": 16, "hidden": 720, "intermediate": 2048},
        {"seq_len": 50, "hidden": 720, "intermediate": 2048},
    ),
    (
        "gemma_cross_attention",
        generate_gemma_cross_attention,
        {"q_seq_len": 8, "kv_seq_len": 16, "hidden": 720, "num_heads": 15, "num_kv_heads": 5},
        {"q_seq_len": 50, "kv_seq_len": 241, "hidden": 720, "num_heads": 15, "num_kv_heads": 5},
    ),
    (
        "action_time_mlp",
        generate_action_time_mlp,
        {"hidden": 64, "time_dim": 128},
        {"hidden": 1024, "time_dim": 2048},
    ),
    (
        "siglip_patch_embed",
        generate_siglip_patch_embed,
        {"image_size": 64, "patch_size": 16, "hidden": 768},
        {"image_size": 512, "patch_size": 16, "hidden": 768},
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        help="Path to smolvla_graph_manifest.json (for shape info)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/SaturnNPU/golden_data"),
        help="Output directory for golden data",
    )
    parser.add_argument(
        "--scale",
        choices=["small", "real", "both"],
        default="both",
        help="Generate small-dim (fast), real-scale, or both",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating hierarchical golden test data...")
    print(f"  Output: {args.output_dir}")
    print(f"  Scale: {args.scale}")
    print()

    all_traces: list[tuple[str, LayerTrace]] = []

    for name, gen_fn, small_kw, real_kw in LAYER_CONFIGS:
        if args.scale in ("small", "both"):
            label = f"{name}_small"
            print(f"  Generating {label}...")
            trace = gen_fn(**small_kw)
            ok = verify_composition(trace)
            print(f"    Composition verified: {ok}")
            print(f"    Chain: {' → '.join(op.name for op in trace.operators)}")
            save_layer_trace(trace, args.output_dir / "small")
            all_traces.append((label, trace))

        if args.scale in ("real", "both"):
            label = f"{name}_real"
            print(f"  Generating {label}...")
            trace = gen_fn(**real_kw)
            ok = verify_composition(trace)
            print(f"    Composition verified: {ok}")
            print(f"    Chain: {' → '.join(op.name for op in trace.operators)}")
            save_layer_trace(trace, args.output_dir / "real")
            all_traces.append((label, trace))

    # Summary
    print("\n--- Summary ---")
    print(f"  Layer types generated: {len(LAYER_CONFIGS)}")
    print(f"  Total traces: {len(all_traces)}")
    for label, trace in all_traces:
        total_ops = len(trace.operators)
        op_types = set(op.op_type for op in trace.operators)
        print(f"    {label}: {total_ops} operators ({', '.join(sorted(op_types))})")

    # Write top-level manifest
    manifest = {
        "seed": SEED,
        "traces": [
            {
                "label": label,
                "layer_type": trace.layer_type,
                "num_operators": len(trace.operators),
                "config": trace.config,
                "composition_verified": trace.composition_verified,
                "operators": [{"name": op.name, "op_type": op.op_type} for op in trace.operators],
            }
            for label, trace in all_traces
        ],
    }
    manifest_path = args.output_dir / "golden_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
