"""Capture all Tiny layers with target arithmetic and genuine greedy feedback.

This emits arithmetic references and actual predecessor operands. It does not
execute Atlas instructions, qualify model quality, or measure target timing.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import tiny_workload_native as native
from capture_tinyllama_native import array_identity, identity, safe_path
from compare_tinyllama_attention import raw_array
from tiny_fp8_pack import encode
from tiny_workload_full import load_arithmetic
from tiny_workload_inventory import column_shards

SITES = tuple(f"model.layers.{layer}.{site}" for layer in range(22) for site in native.PROJECTIONS) + ("lm_head",)


def greedy_token(logits):
    values = np.asarray(logits)
    if values.dtype != np.float32 or values.shape != (32000,) or not np.isfinite(values).all():
        raise ValueError("require 32000 finite FP32 logits")
    # NumPy, Torch and the target C helper select the first maximum on ties.
    return int(values.argmax())


def padded_words(words, rows):
    if words.dtype != np.uint16 or words.ndim != 2 or words.shape[0] != rows or rows not in (1, 128):
        raise ValueError("require logical M1/M128 BF16 words")
    if rows == 128:
        return np.ascontiguousarray(words)
    return np.concatenate((words, np.zeros((31, words.shape[1]), dtype=np.uint16)))


def capture(args, projection_factory=None):
    import torch

    layer_count = getattr(args, "layer_count", 22)
    block_only = getattr(args, "block_only", False)
    attention_python = getattr(args, "attention_worker_python", None)
    if type(args.decode_count) is not int or not 0 <= args.decode_count <= 32:
        raise ValueError("decode count must be 0 through 32")
    if type(layer_count) is not int or not 1 <= layer_count <= 22:
        raise ValueError("layer count must be 1 through 22")
    if block_only and args.decode_count:
        raise ValueError("block-only capture requires prefill without decode")
    if layer_count != 22 and not block_only:
        raise ValueError("partial layers require an explicit block-only capture")
    sites = tuple(f"model.layers.{layer}.{site}" for layer in range(layer_count) for site in native.PROJECTIONS)
    if not block_only:
        sites += ("lm_head",)
    out = safe_path(args.out)
    if out.exists():
        raise ValueError("fresh capture output required")
    records = []
    oracle = load_arithmetic(args.atlas_arithmetic, args.atlas_arithmetic_sha256, records)
    from source_relocation import SourceRelocation

    relocation = SourceRelocation(getattr(args, "relocate", ()))
    model, tokens, _, torch, cache_utils, modeling_llama, _ = native.prepare(
        args.historical, records, relocation=relocation
    )
    records.extend(
        identity(Path(__file__).with_name(name))
        for name in (
            "capture_tinyllama_full.py",
            "tiny_workload_native.py",
            "tiny_workload_full.py",
            "tiny_workload_inventory.py",
            "tiny_fp8_pack.py",
            "capture_tinyllama_native.py",
            "compare_tinyllama_attention.py",
            "source_relocation.py",
        )
    )
    initial = native.state(model, torch)
    rng, nrng, prng = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
    out.mkdir(parents=True)
    (out / "parameters").mkdir()
    (out / "calls").mkdir()
    report = {
        "status": "failed",
        "atlas_instruction_execution": False,
        "timing_qualified": False,
        "quality_qualified": False,
        "decode_count": args.decode_count,
        "source_records": records,
        "layer_count": layer_count,
        "block_only": block_only,
        "include_lm_head": not block_only,
        "scope": "layer0-prefill" if block_only and layer_count == 1 else "full-model",
        "source_relocation": relocation.report(),
    }
    started = time.monotonic()
    handles, calls, parameters, saved, weights = [], [], {}, {}, {}
    mode, counter = "capture", 0
    attention, projection = None, None
    original_attention = modeling_llama.eager_attention_forward

    def save(relative, words):
        data = np.ascontiguousarray(words)
        path = out / relative
        path.write_bytes(data.tobytes())
        return {"file": relative, **array_identity(data), "bytes": data.nbytes}

    def hook(site):
        def linear(module, inputs, output):
            nonlocal counter
            step, index = divmod(counter, len(sites))
            rows = 128 if step == 0 else 1
            if (
                step > args.decode_count
                or sites[index] != site
                or len(inputs) != 1
                or module.bias is not None
                or inputs[0].dtype != torch.bfloat16
                or output.dtype != torch.bfloat16
                or module.weight.dtype != torch.bfloat16
                or tuple(inputs[0].shape) != (1, rows, module.in_features)
                or tuple(output.shape) != (1, rows, module.out_features)
            ):
                raise ValueError("full linear order or BF16 ABI changed")
            a = raw_array(inputs[0][0], torch)
            a_id = array_identity(a)
            key = (step, site)
            counter += 1
            if projection is not None:
                return projection.linear(site, module, a, step, index, mode)
            if mode == "replay":
                expected_a, words = saved[key]
                if a_id != expected_a:
                    raise ValueError("replay actual predecessor changed")
                return torch.from_numpy(words.copy()).view(torch.bfloat16).unsqueeze(0)
            if site not in weights:
                raw_b = raw_array(module.weight.T, torch)
                parameters[site] = {"B_bf16": save(f"parameters/{index:03d}-B.bin", raw_b)}
                pieces = []
                for shard in column_shards(module.in_features, module.out_features):
                    lo, hi = shard["columns"]
                    encoded, scale = encode(np.ascontiguousarray(raw_b[:, lo:hi]))
                    pieces.append((lo, hi, encoded, scale))
                weights[site] = (module.weight._version, module.weight.data_ptr(), pieces)
            version, address, pieces = weights[site]
            if (version, address) != (module.weight._version, module.weight.data_ptr()):
                raise ValueError("cached parameter changed")
            encoded_a, a_scale = encode(a)
            print(json.dumps(dict(milestone="projection_start", step=step, site=site)), flush=True)
            panels, result = [], []
            for panel, (lo, hi, b, b_scale) in enumerate(pieces):
                words = oracle.rescale_bf16_array(oracle.ordered_matmul(encoded_a, b), a_scale + b_scale)
                result.append(words)
                stem = f"calls/{step:02d}-{index:03d}-{panel:02d}"
                panels.append(
                    {
                        "columns": [lo, hi],
                        "logical_mkn": [rows, module.in_features, hi - lo],
                        "physical_mkn": [128 if rows == 128 else 32, module.in_features, hi - lo],
                        "A_scale": a_scale,
                        "B_scale": b_scale,
                        "rescale": a_scale + b_scale,
                        "expected_bf16": save(stem + "-expected.bin", padded_words(words, rows)),
                    }
                )
                print(
                    json.dumps(dict(milestone="projection_panel", step=step, site=site, panel=panel, columns=[lo, hi])),
                    flush=True,
                )
            words = np.concatenate(result, axis=1)
            saved[key] = (a_id, words)
            calls.append(
                {
                    "step": step,
                    "site": site,
                    "site_index": index,
                    "A_bf16": save(f"calls/{step:02d}-{index:03d}-A.bin", a),
                    "panels": panels,
                }
            )
            return torch.from_numpy(words.copy()).view(torch.bfloat16).unsqueeze(0)

        return linear

    class StopBlock(Exception):
        pass

    def session(label):
        if projection is not None:
            projection.begin_session(label)
        cache = cache_utils.DynamicCache(config=model.config)
        next_token, trace = None, []
        for step in range(args.decode_count + 1):
            ids = tokens[:, :128] if step == 0 else torch.tensor([[next_token]], dtype=tokens.dtype)
            previous = (
                [(raw_array(layer.keys, torch), raw_array(layer.values, torch)) for layer in cache.layers[:layer_count]]
                if step
                else []
            )
            boundary = {}

            def stop(_module, _inputs, output):
                value = output[0] if isinstance(output, tuple) else output
                boundary["hidden"] = raw_array(value[0], torch)
                raise StopBlock()

            stop_handle = model.model.layers[layer_count - 1].register_forward_hook(stop) if block_only else None
            try:
                with torch.inference_mode():
                    result = model(input_ids=ids, past_key_values=cache, use_cache=True)
            except StopBlock:
                if not block_only or "hidden" not in boundary:
                    raise
            finally:
                if stop_handle is not None:
                    stop_handle.remove()
            row = {"step": step, "input_tokens": ids.tolist(), "cache": []}
            if block_only:
                row["hidden"] = array_identity(boundary["hidden"])
                if label == "target":
                    row["hidden_payload"] = save(f"calls/hidden-{step:02d}.bin", boundary["hidden"])
            else:
                if result.past_key_values is not cache or len(cache.layers) != 22:
                    raise ValueError("full cache identity or layer count changed")
                logits = result.logits[0, -1].float().numpy().copy()
                next_token = greedy_token(logits)
                row.update(next_token=next_token, logits=array_identity(logits))
            for layer, item in enumerate(cache.layers[:layer_count]):
                pair = (raw_array(item.keys, torch), raw_array(item.values, torch))
                if any(v.shape != (1, 4, 128 + step, 64) for v in pair):
                    raise ValueError("cache length or geometry changed")
                for field, value, old in zip(("keys", "values"), pair, previous[layer] if step else (None, None)):
                    if old is not None and not np.array_equal(value[:, :, :-1], old):
                        raise ValueError("KV prefix changed during greedy decode")
                    descriptor = array_identity(value)
                    if label == "target":
                        descriptor = save(f"calls/cache-{step:02d}-{layer:02d}-{field}.bin", value)
                    row["cache"].append({"layer": layer, "field": field, **descriptor})
            if label == "target" and not block_only:
                row["logits_payload"] = save(f"calls/logits-{step:02d}.bin", logits)
            trace.append(row)
            print(json.dumps({"milestone": label, "step": step, "next_token": next_token}), flush=True)
        return trace

    try:
        linear_weights = {site + ".weight" for site in SITES}
        host_parameters = {
            name: save(f"parameters/host-{index:03d}.bin", raw_array(value, torch))
            for index, (name, value) in enumerate(model.named_parameters())
            if name not in linear_weights
        }
        positions = torch.arange(160, dtype=torch.int64).unsqueeze(0)
        with torch.inference_mode():
            cosine, sine = model.model.rotary_emb(torch.zeros((1, 160, 2048), dtype=torch.bfloat16), positions)
        host_parameters["rotary_cosine"] = save("parameters/rotary-cosine.bin", raw_array(cosine[0], torch))
        host_parameters["rotary_sine"] = save("parameters/rotary-sine.bin", raw_array(sine[0], torch))
        host_parameters["prefill_tokens"] = save("parameters/prefill-tokens.bin", tokens[0, :128].numpy())
        native_trace = session("native")
        modules = {name: item for name, item in model.named_modules() if type(item) is torch.nn.Linear}
        if not set(sites).issubset(modules) or (not block_only and set(modules) != set(sites)):
            raise ValueError("selected model linear inventory changed")
        if projection_factory is not None:
            projection = projection_factory(args, out, save, torch, model, sites, records, calls, parameters)
            projection.install()
        if attention_python is not None:
            from capture_atlas_attention import AttentionCapture

            records.append(identity(Path(__file__).with_name("capture_atlas_attention.py")))
            attention = AttentionCapture(
                out,
                attention_python,
                args.attention_worker_script,
                save,
                torch,
                records,
                model_root=getattr(args, "attention_model_root", None),
            )
            if projection is not None and hasattr(projection, "attach_attention"):
                projection.attach_attention(attention)

            def atlas_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
                step = (counter - 1) // len(sites)
                return attention.apply(step, module, query, key, value, attention_mask, scaling)

            modeling_llama.eager_attention_forward = atlas_attention
        for site in sites:
            handles.append(modules[site].register_forward_hook(hook(site)))
        target_trace = session("target")
        if counter != len(sites) * (args.decode_count + 1):
            raise ValueError("incomplete target invocation ledger")
        mode, counter = "replay", 0
        if attention is not None:
            attention.mode = "replay"
        replay_trace = session("replay")
        # File descriptors occur only in the target trace; compare raw identities.
        normalized = json.loads(json.dumps(target_trace))
        for row in normalized:
            row.pop("logits_payload", None)
            row.pop("hidden_payload", None)
            for entry in row["cache"]:
                entry.pop("file", None)
                entry.pop("bytes", None)
        if replay_trace != normalized or counter != len(sites) * (args.decode_count + 1):
            raise ValueError("complete greedy target replay changed")
        while handles:
            handles.pop().remove()
        if projection is not None:
            projection.restore()
        modeling_llama.eager_attention_forward = original_attention
        if session("restored") != native_trace or native.state(model, torch) != initial:
            raise ValueError("native restoration or parameters changed")
        if not torch.equal(rng, torch.get_rng_state()):
            raise ValueError("caller Torch RNG changed")
        for record in records:
            identity(record["path"], record["sha256"])
        report.update(
            status="PASS_TINY_ATTENTION_BLOCK_CAPTURE_REPLAY"
            if block_only
            else "PASS_FULL_TINY_ARITHMETIC_CAPTURE_REPLAY",
            parameters=parameters,
            host_parameters=host_parameters,
            calls=calls,
            native_trace=native_trace,
            target_trace=target_trace,
            greedy_feedback_actual=not block_only,
            all22_cache_prefixes_exact=layer_count == 22,
            cache_prefixes_exact=True,
            projection_calls=len(calls),
            attention_calls=[] if attention is None else attention.calls,
            atlas_instruction_execution=attention is not None,
            source_bookends_verified=True,
            target_replay_exact=True,
            native_restored_exact=True,
        )
        if projection is not None:
            report.update(projection.summary())
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        modeling_llama.eager_attention_forward = original_attention
        while handles:
            handles.pop().remove()
        if projection is not None:
            projection.restore()
        try:
            if attention is not None:
                attention.close()
        finally:
            try:
                if projection is not None:
                    projection.close()
            finally:
                torch.set_rng_state(rng)
                np.random.set_state(nrng)
                random.setstate(prng)
                report["wall_seconds"] = time.monotonic() - started
                (out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("historical", "atlas-arithmetic", "out"):
        parser.add_argument("--" + name, type=safe_path, required=True)
    parser.add_argument("--atlas-arithmetic-sha256", required=True)
    parser.add_argument("--decode-count", type=int, default=2)
    parser.add_argument("--layer-count", type=int, default=22)
    parser.add_argument("--block-only", action="store_true")
    parser.add_argument("--relocate", action="append", default=[])
    parser.add_argument("--attention-worker-python", type=Path)
    parser.add_argument("--attention-worker-script", type=safe_path)
    parser.add_argument(
        "--attention-model-root",
        type=safe_path,
        help="explicit pinned npu_model checkout passed to the delivered attention worker",
    )
    args = parser.parse_args()
    if bool(args.attention_worker_python) != bool(args.attention_worker_script):
        parser.error("both attention worker arguments are required together")
    if args.attention_worker_script and not args.attention_model_root:
        parser.error("--attention-model-root is required for instruction capture")
    if args.attention_model_root and not args.attention_worker_script:
        parser.error("--attention-model-root requires an attention worker")
    capture(args)


if __name__ == "__main__":
    main()
