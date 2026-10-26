"""Frozen static-W8A8 diagnostic source; not the selected EL4 recipe or ABI.

This defines numerical computation and references, never schedules or tuning
decisions. Learned weights are grouped by 16 output channels. Both attention
operands use frozen per-tensor activation scales; cache bytes are signed INT8.
See EL4_INPUT_AUDIT.md before using these diagnostics in a handoff.
"""

import numpy as np
import torch
from torch import nn


def rne_codes(value, scale, low=-128, high=127):
    """Independent FP32 division and floor/parity ties-to-even encoding."""
    values = np.asarray(value, dtype=np.float32) / np.asarray(scale, dtype=np.float32)
    floor = np.floor(values)
    fraction = values - floor
    rounded = floor + ((fraction > 0.5) | ((fraction == 0.5) & (np.remainder(floor, 2) != 0)))
    return np.clip(rounded, low, high).astype(np.int8)


def tensor_scale(value):
    maximum = float(np.max(np.abs(np.asarray(value, dtype=np.float32))))
    return float(np.float32(maximum / 127)) if maximum else 1.0


def grouped_weight(value, group=16):
    weight = np.asarray(value, dtype=np.float32)
    if weight.ndim != 2 or not np.isfinite(weight).all():
        raise ValueError("Require a finite [output,input] weight matrix")
    scale = np.empty(weight.shape[0], np.float32)
    for start in range(0, len(scale), group):
        scale[start : start + group] = tensor_scale(weight[start : start + group])
    return rne_codes(weight, scale[:, None], -127, 127), scale


def encode(x, scale, oracle=False):
    if oracle:
        return torch.from_numpy(rne_codes(x.detach().numpy(), scale.detach().numpy()))
    return torch.clamp(torch.round(x / scale), -128, 127).to(torch.int8)


def integer_mm(left, right, oracle=False):
    if oracle:
        result = left.numpy().astype(np.int64) @ right.numpy().astype(np.int64)
        if result.min() < -(2**31) or result.max() >= 2**31:
            raise ValueError("Independent contraction exceeds INT32")
        return torch.from_numpy(result.astype(np.int32))
    return torch.ops.aten._int_mm.default(
        left.clone(memory_format=torch.contiguous_format), right.clone(memory_format=torch.contiguous_format)
    )


def rms(x, weight):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-5) * weight


def rope(x, cosine, sine):
    half = x.shape[-1] // 2
    return x * cosine + torch.cat((-x[..., half:], x[..., :half]), dim=-1) * sine


class StaticLinear(nn.Module):
    def __init__(self, weight, activation_scale):
        super().__init__()
        codes, scales = grouped_weight(weight.detach().float().numpy())
        self.register_buffer("weight", torch.from_numpy(codes.T.copy()))
        self.register_buffer("weight_scale", torch.from_numpy(scales))
        self.register_buffer("activation_scale", torch.tensor(activation_scale, dtype=torch.float32))

    def run(self, x, oracle=False):
        shape = x.shape
        codes = encode(x.reshape(-1, shape[-1]), self.activation_scale, oracle)
        raw = integer_mm(codes, self.weight, oracle)
        result = (raw.float() * self.activation_scale) * self.weight_scale
        return result.reshape(*shape[:-1], self.weight.shape[1])

    def forward(self, x):
        return self.run(x)


class Layer(nn.Module):
    def __init__(self, archive, index, scales, config):
        super().__init__()
        prefix = f"model.layers.{index}."
        self.heads = config["num_attention_heads"]
        self.kv_heads = config["num_key_value_heads"]
        self.width = config["hidden_size"] // self.heads
        self.capacity = 64
        for name, key in (("norm", "input_layernorm.weight"), ("post_norm", "post_attention_layernorm.weight")):
            self.register_buffer(name, archive.get_tensor(prefix + key).float())
        for name, group, scale_key in (
            ("q", "self_attn", "qkv"),
            ("k", "self_attn", "qkv"),
            ("v", "self_attn", "qkv"),
            ("o", "self_attn", "o"),
            ("gate", "mlp", "gate_up"),
            ("up", "mlp", "gate_up"),
            ("down", "mlp", "down"),
        ):
            setattr(
                self, name, StaticLinear(archive.get_tensor(prefix + f"{group}.{name}_proj.weight"), scales[scale_key])
            )
        for name in ("query", "key", "value", "probability"):
            self.register_buffer(name + "_scale", torch.tensor(scales[name], dtype=torch.float32))
        positions = torch.arange(64, dtype=torch.float32)[:, None]
        inverse = 1.0 / (config["rope_theta"] ** (torch.arange(0, self.width, 2).float() / self.width))
        angles = positions * inverse
        angles = torch.cat((angles, angles), -1)
        # This is frozen numerical source data, not a compiler/table optimization.
        self.register_buffer("cosine", torch.cos(angles))
        self.register_buffer("sine", torch.sin(angles))

    def qkv(self, hidden, positions, oracle=False):
        normalized = rms(hidden, self.norm)
        count = hidden.shape[0]
        q = self.q.run(normalized, oracle).reshape(count, self.heads, self.width).transpose(0, 1)
        k = self.k.run(normalized, oracle).reshape(count, self.kv_heads, self.width).transpose(0, 1)
        v = self.v.run(normalized, oracle).reshape(count, self.kv_heads, self.width).transpose(0, 1)
        cosine = torch.nn.functional.embedding(positions, self.cosine)
        sine = torch.nn.functional.embedding(positions, self.sine)
        return rope(q, cosine, sine), rope(k, cosine, sine), v

    def attention(self, hidden, cache, position, prefill, oracle=False, trace=None):
        count = hidden.shape[0]
        positions = torch.arange(0, count) if prefill else position
        q, k, v = self.qkv(hidden, positions, oracle)
        kc, vc = encode(k, self.key_scale, oracle), encode(v, self.value_scale, oracle)
        if prefill:
            pad = torch.full((self.kv_heads, self.capacity - count, self.width), 0, dtype=torch.int8)
            keys, values = torch.cat((kc, pad), 1), torch.cat((vc, pad), 1)
        else:
            selected = (torch.arange(0, self.capacity) == position.reshape(1, 1)).reshape(1, self.capacity, 1)
            keys, values = torch.where(selected, kc, cache[0]), torch.where(selected, vc, cache[1])
        query = encode(q, self.query_scale, oracle)
        scores = []
        repeat = self.heads // self.kv_heads
        for head in range(self.heads):
            raw = integer_mm(query[head], keys[head // repeat].transpose(0, 1), oracle)
            scores.append((raw.float() * self.query_scale) * self.key_scale)
        scores = torch.stack(scores) * (self.width**-0.5)
        causal = torch.arange(0, self.capacity)[None, :] <= positions[:, None]
        probabilities = torch.softmax(torch.where(causal, scores, torch.full_like(scores, -float("inf"))), -1)
        pc = encode(probabilities, self.probability_scale, oracle)
        heads = []
        for head in range(self.heads):
            raw = integer_mm(pc[head], values[head // repeat], oracle)
            heads.append((raw.float() * self.probability_scale) * self.value_scale)
        attended = torch.stack(heads).transpose(0, 1).reshape(count, -1)
        if trace is not None:
            trace.update(
                hidden=hidden.clone(),
                positions=positions.clone(),
                query=q.clone(),
                keys=keys.clone(),
                values=values.clone(),
                probabilities=probabilities.clone(),
                attended=attended.clone(),
                cache=cache.clone(),
                position=position.clone(),
            )
        result = hidden + self.o.run(attended, oracle)
        return result, torch.stack((keys, values))

    def mlp(self, hidden, oracle=False):
        normalized = rms(hidden, self.post_norm)
        gated = torch.nn.functional.silu(self.gate.run(normalized, oracle)) * self.up.run(normalized, oracle)
        return hidden + self.down.run(gated, oracle)


class SourceModel(nn.Module):
    def __init__(self, archive, calibration, config, prefill):
        super().__init__()
        self.prefill = prefill
        self.register_buffer("initial_position", torch.tensor([0], dtype=torch.int64))
        self.register_buffer("embedding", archive.get_tensor("model.embed_tokens.weight").float())
        self.register_buffer("norm", archive.get_tensor("model.norm.weight").float())
        self.layers = nn.ModuleList(Layer(archive, i, row, config) for i, row in enumerate(calibration["layers"]))
        self.head = StaticLinear(archive.get_tensor("lm_head.weight"), calibration["head"])

    def run(self, tokens, cache=None, position=None, traces=None):
        hidden = torch.nn.functional.embedding(tokens, self.embedding).reshape(tokens.shape[-1], -1)
        next_cache = []
        if self.prefill:
            position = self.initial_position
        for index, layer in enumerate(self.layers):
            incoming = (
                torch.full((2, layer.kv_heads, 64, layer.width), 0, dtype=torch.int8) if self.prefill else cache[index]
            )
            trace = {} if traces is not None and index == 0 else None
            hidden, state = layer.attention(hidden, incoming, position, self.prefill, trace=trace)
            if trace is not None:
                trace["mlp_hidden"] = hidden.clone()
                traces.update(trace)
            hidden = layer.mlp(hidden)
            next_cache.append(state)
        logits = self.head(rms(hidden[-1:], self.norm)).reshape(1, 1, -1)
        return logits, torch.stack(next_cache), position + tokens.shape[-1]


class Prefill(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens):
        return self.model.run(tokens)


class Decode(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, cache, position):
        return self.model.run(tokens, cache, position)


class Chunk(nn.Module):
    def __init__(self, layer, kind, prefill):
        super().__init__()
        self.layer, self.kind, self.prefill = layer, kind, prefill

    def run(self, args, oracle=False):
        layer = self.layer
        if self.kind == "qkv":
            return layer.qkv(args[0], args[1], oracle)
        if self.kind == "o":
            return layer.o.run(args[0], oracle)
        if self.kind == "mlp":
            return layer.mlp(args[0], oracle)
        if self.kind == "down":
            return layer.down.run(args[0], oracle)
        if self.kind in ("qk", "pv"):
            left_scale = layer.query_scale if self.kind == "qk" else layer.probability_scale
            right_scale = layer.key_scale if self.kind == "qk" else layer.value_scale
            codes = encode(args[0], left_scale, oracle)
            right = args[1].transpose(0, 1) if self.kind == "qk" else args[1]
            return (integer_mm(codes, right, oracle).float() * left_scale) * right_scale
        if self.kind == "attention":
            return layer.attention(*args, self.prefill, oracle)
        raise ValueError("Unknown explicit source chunk kind")

    def forward(self, *args):
        return self.run(args)


def calibrate(archive, config, tokens):
    """FP32 source calibration on an explicitly separate synthetic token roster."""
    hidden = archive.get_tensor("model.embed_tokens.weight")[tokens].float()
    heads, kv = config["num_attention_heads"], config["num_key_value_heads"]
    width = config["hidden_size"] // heads
    count = tokens.shape[-1]
    angles = torch.arange(count).float()[:, None] / (
        config["rope_theta"] ** (torch.arange(0, width, 2).float() / width)
    )
    angles = torch.cat((angles, angles), -1)
    causal = torch.arange(count)[None, :] <= torch.arange(count)[:, None]
    records = []
    for index in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{index}."

        def weight(name):
            return archive.get_tensor(prefix + name).float()

        normalized = rms(hidden, weight("input_layernorm.weight"))
        q = (normalized @ weight("self_attn.q_proj.weight").T).reshape(-1, count, heads, width).transpose(1, 2)
        k = (normalized @ weight("self_attn.k_proj.weight").T).reshape(-1, count, kv, width).transpose(1, 2)
        v = (normalized @ weight("self_attn.v_proj.weight").T).reshape(-1, count, kv, width).transpose(1, 2)
        q, k = rope(q, angles.cos(), angles.sin()), rope(k, angles.cos(), angles.sin())
        scores = (q @ k.repeat_interleave(heads // kv, 1).transpose(-1, -2)) * width**-0.5
        probabilities = torch.softmax(torch.where(causal, scores, -float("inf")), -1)
        attended = (probabilities @ v.repeat_interleave(heads // kv, 1)).transpose(1, 2).reshape_as(hidden)
        after = hidden + attended @ weight("self_attn.o_proj.weight").T
        post = rms(after, weight("post_attention_layernorm.weight"))
        gated = torch.nn.functional.silu(post @ weight("mlp.gate_proj.weight").T) * (
            post @ weight("mlp.up_proj.weight").T
        )
        hidden = after + gated @ weight("mlp.down_proj.weight").T
        row = {
            name: tensor_scale(value.numpy())
            for name, value in (
                ("qkv", normalized),
                ("o", attended),
                ("gate_up", post),
                ("down", gated),
                ("query", q),
                ("key", k),
                ("value", v),
            )
        }
        row["probability"] = float(np.float32(1 / 127))
        records.append(row)
        print(f"calibrated layer {index + 1}/{config['num_hidden_layers']}", flush=True)
    return dict(layers=records, head=tensor_scale(rms(hidden, archive.get_tensor("model.norm.weight").float()).numpy()))
