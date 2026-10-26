"""Prepare the prescribed TinyLlama range transforms and canonical cache/session ABI.

This is numerical input preparation, not a TVM schedule or hardware implementation.
Checkpoint weights, public training roster and every derived scale/width are recorded.
"""

import ast
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from prepare_selected_inputs import digest
from torch import nn


class ScaledLinear(nn.Module):
    """Explicit SmoothQuant input migration for a projection with no preceding affine norm."""

    def __init__(self, projection, scale):
        super().__init__()
        self.projection = projection
        self.register_buffer("input_scale", scale)
        with torch.no_grad():
            self.projection.weight.mul_(scale[None])

    def forward(self, value):
        return self.projection(value / self.input_scale)


def migration_scale(absmax, projections, alpha=0.5):
    weight_max = torch.stack([p.weight.detach().abs().amax(0) for p in projections]).amax(0)
    return (absmax.clamp_min(1e-5).pow(alpha) / weight_max.clamp_min(1e-5).pow(1 - alpha)).clamp_min(1e-5)


def fold_migration(norm, projections, scale):
    with torch.no_grad():
        norm.weight.div_(scale)
        for projection in projections:
            projection.weight.mul_(scale[None])


def split_mlp(mlp, absmax, ratio=8, alignment=16):
    """Split high-range gated channels algebraically, then zero-pad to the selected buckets."""
    positive = absmax[absmax > 0]
    threshold = float(positive.median()) * ratio if len(positive) else 0.0
    repeats = (
        torch.ceil(absmax / threshold).clamp_min(1).to(torch.int64) if threshold else torch.ones_like(absmax).long()
    )
    indices = torch.repeat_interleave(torch.arange(len(repeats)), repeats)
    width = len(indices)
    aligned = math.ceil(width / alignment) * alignment
    buckets = [n for n in (5728, 6096) if n >= aligned]
    padded = min(buckets) if buckets else aligned
    gate, up, down = mlp.gate_proj, mlp.up_proj, mlp.down_proj
    if any(p.bias is not None for p in (gate, up, down)):
        raise ValueError("Selected TinyLlama checkpoint has no projection biases")
    with torch.no_grad():
        gate_weight = torch.zeros((padded, gate.in_features), dtype=gate.weight.dtype)
        up_weight = torch.zeros_like(gate_weight)
        down_weight = torch.zeros((down.out_features, padded), dtype=down.weight.dtype)
        gate_weight[:width] = gate.weight[indices]
        up_weight[:width] = up.weight[indices] / repeats[indices, None]
        down_weight[:, :width] = down.weight[:, indices]
        for projection, value in ((gate, gate_weight), (up, up_weight), (down, down_weight)):
            projection.weight = nn.Parameter(value, requires_grad=False)
            projection.out_features, projection.in_features = value.shape
    return dict(original_width=len(repeats), expanded_width=width, padded_width=padded, repeats=repeats.tolist())


def range_statistics(model, segments, path):
    records, handles = {}, []
    for index, layer in enumerate(model.model.layers):
        for role, projection in (
            ("qkv", layer.self_attn.q_proj),
            ("gate_up", layer.mlp.gate_proj),
            ("o", layer.self_attn.o_proj),
            ("down", layer.mlp.down_proj),
        ):
            key = f"layer_{index}_{role}"

            def observe(_module, args, key=key):
                value = args[0].detach().abs().reshape(-1, args[0].shape[-1]).amax(0)
                records[key] = value if key not in records else torch.maximum(records[key], value)

            handles.append(projection.register_forward_pre_hook(observe))
    try:
        with torch.no_grad():
            for index, segment in enumerate(segments[:32]):
                model(input_ids=segment[None], use_cache=False)
                print(f"range statistics {index + 1}/32", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    np.savez_compressed(path, **{k: v.numpy() for k, v in records.items()})
    return records


def prepare_ranges(model, calibration, directory):
    path = directory / "tiny-range-statistics.npz"
    inputs_path = Path(os.environ["SELECTED_INPUTS"]) / "tiny-calibration.npz"
    identity = dict(
        checkpoint="6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933",
        calibration=digest(inputs_path),
        statistics_segments=32,
        alpha=0.5,
        split_ratio=8,
        alignment=16,
        splitting_threshold="8 times the median positive observed gated-channel absmax",
        source=digest(__file__),
    )
    identity_path = directory / "tiny-range-identity-v2.json"
    if path.is_file():
        if identity_path.is_file():
            previous = json.loads(identity_path.read_text())
            if any(previous.get(k) != v for k, v in identity.items()) or previous["statistics_sha256"] != digest(path):
                raise ValueError("Range statistics provenance differs; use a new release directory")
        else:
            # Reuse the already observed arrays only after checking the exact
            # original producer and unchanged statistics algorithm. Retain its
            # original identity and failed transform realization untouched.
            original = json.loads((directory / "tiny-range-identity.json").read_text())
            producer = directory / "source/tiny-range-prototype-loader.py"
            if (
                original["source"] != digest(producer)
                or original["calibration"] != identity["calibration"]
                or original["checkpoint"] != identity["checkpoint"]
                or original["statistics_sha256"] != digest(path)
            ):
                raise ValueError("Original statistics producer/input identity differs")

            def algorithm(path):
                tree = ast.parse(path.read_text())
                return ast.dump(
                    next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "range_statistics")
                )

            if algorithm(producer) != algorithm(Path(__file__)):
                raise ValueError("Statistics algorithm changed; recalculate in a fresh release")
            identity_path.write_text(
                json.dumps(
                    {**identity, "statistics_sha256": digest(path), "statistics_producer_sha256": digest(producer)},
                    indent=2,
                )
                + "\n"
            )
        with np.load(path, allow_pickle=False) as arrays:
            stats = {k: torch.from_numpy(arrays[k].copy()) for k in arrays}
    else:
        stats = range_statistics(model, calibration, path)
        identity_path.write_text(json.dumps({**identity, "statistics_sha256": digest(path)}, indent=2) + "\n")
    rows = []
    for index, layer in enumerate(model.model.layers):
        qkv = [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]
        gate_up = [layer.mlp.gate_proj, layer.mlp.up_proj]
        qkv_scale = migration_scale(stats[f"layer_{index}_qkv"], qkv)
        gate_scale = migration_scale(stats[f"layer_{index}_gate_up"], gate_up)
        out_scale = migration_scale(stats[f"layer_{index}_o"], [layer.self_attn.o_proj])
        fold_migration(layer.input_layernorm, qkv, qkv_scale)
        fold_migration(layer.post_attention_layernorm, gate_up, gate_scale)
        layer.self_attn.o_proj = ScaledLinear(layer.self_attn.o_proj, out_scale)
        split = split_mlp(layer.mlp, stats[f"layer_{index}_down"])
        rows.append(dict(layer=index, **split))
    (directory / "tiny-range-realization-v2.json").write_text(json.dumps(dict(layers=rows), indent=2) + "\n")


def get_model_and_inputs():
    from m2m.capture.causal_session import FixedCacheDecodeStep, FixedCachePrefillStage, make_causal_program_capture
    from transformers import AutoModelForCausalLM

    root = Path(os.environ["SELECTED_RELEASE"])
    checkpoint = root / "source/tiny/model"
    if digest(checkpoint / "model.safetensors") != "6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933":
        raise ValueError("TinyLlama checkpoint differs")
    model, info = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.float32, attn_implementation="eager", output_loading_info=True
    )
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("Full checkpoint did not load exactly")
    model.eval().requires_grad_(False)
    with np.load(Path(os.environ["SELECTED_INPUTS"]) / "tiny-calibration.npz", allow_pickle=False) as data:
        calibration = torch.from_numpy(data["tokens"].copy())
    prepare_ranges(model, calibration, root)
    stage = os.environ.get("SELECTED_TINY_STAGE", "")
    if stage:
        tokens = torch.randint(0, 32000, (1, 64), generator=torch.Generator().manual_seed(0))
        prefill = FixedCachePrefillStage(model, 64).eval()
        if stage == "prefill":
            return prefill, (tokens[:, :32],)
        if stage != "decode":
            raise ValueError("Unknown explicit TinyLlama stage")
        with torch.no_grad():
            _, cache, positions = prefill(tokens[:, :32])
        return FixedCacheDecodeStep(model, 64).eval(), (tokens[:, 32:33], cache, positions)
    return make_causal_program_capture(
        model,
        env_prefix="SELECTED_TINY_CAPTURE",
        checkpoint="TinyLlama/TinyLlama-1.1B-Chat-v1.0@fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        full_checkpoint=True,
        prefill_tokens=32,
        decode_tokens=32,
    )


def get_calibration_inputs(module, inputs):
    from m2m.capture.causal_session import FixedCacheDecodeStep, FixedCachePrefillStage

    path = Path(os.environ["SELECTED_INPUTS"]) / "tiny-calibration.npz"
    with np.load(path, allow_pickle=False) as data:
        segments = torch.from_numpy(data["tokens"].copy())
    if len(inputs) == 1:
        for tokens in segments:
            yield (tokens[None, :32].clone(),)
        return
    if len(inputs) != 3 or not isinstance(module, FixedCacheDecodeStep):
        raise ValueError("Unknown calibration stage ABI")
    prefill = FixedCachePrefillStage(module.lm, 64).eval()
    directory = Path(os.environ["SELECTED_RELEASE"]) / "tiny-decode-calibration"
    directory.mkdir(exist_ok=True)
    manifest = directory / "manifest.json"
    calibration_identity = dict(
        calibration_sha256=digest(path),
        range_identity_sha256=digest(Path(os.environ["SELECTED_RELEASE"]) / "tiny-range-identity-v2.json"),
        loader_source_sha256=digest(__file__),
    )
    if manifest.is_file():
        saved = json.loads(manifest.read_text())
        if saved["identity"] != calibration_identity:
            raise ValueError("Cached calibration inputs or preparation identity changed")
        records = saved["records"]
        if len(records) != 1024:
            raise ValueError("Cached calibration roster is incomplete")
        for record in records:
            cache_path = directory / record["cache"]
            if digest(cache_path) != record["sha256"]:
                raise ValueError("Frozen calibration cache changed")
            yield (
                torch.tensor(record["token"], dtype=torch.int64),
                torch.from_numpy(np.load(cache_path, mmap_mode="c", allow_pickle=False)),
                torch.tensor(record["positions"], dtype=torch.int64),
            )
        return
    if any(directory.iterdir()):
        raise ValueError("Incomplete calibration attempt retained; use a fresh release directory")
    records = []
    with torch.no_grad():
        for index, tokens in enumerate(segments):
            _, cache, positions = prefill(tokens[None, :32])
            for step in range(8):
                arguments = (tokens[None, 32 + step : 33 + step].clone(), cache.clone(), positions.clone())
                _, cache, positions = module(*arguments)
                path = directory / f"cache-{len(records):04d}.npy"
                np.save(path, arguments[1].numpy(), allow_pickle=False)
                records.append(
                    dict(
                        cache=path.name,
                        sha256=digest(path),
                        token=arguments[0].tolist(),
                        positions=arguments[2].tolist(),
                    )
                )
                # Clean file-backed tensors can be evicted under memory pressure.
                # Values/dtype and functional state remain exactly the real FP32 chain's.
                yield (arguments[0], torch.from_numpy(np.load(path, mmap_mode="c", allow_pickle=False)), arguments[2])
            if index % 8 == 0:
                print(f"decode calibration contexts {(index + 1) * 8}/1024", flush=True)
    manifest.write_text(json.dumps(dict(identity=calibration_identity, records=records), indent=2) + "\n")
