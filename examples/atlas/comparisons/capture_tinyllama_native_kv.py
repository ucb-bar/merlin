"""Observe full native K/V projections in the unchanged attention producer.

Process-local observation hooks return None and capture the original Linear
inputs/outputs. The complete native attention/cache descriptor must still match
the authenticated baseline. This performs no target or language-quality run.
"""

from merlin.common.jsonio import write_pretty_json
import argparse
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import zipfile

import numpy as np

import compare_tinyllama_attention as attention
from capture_tinyllama_native import (array_identity, check_memory, identity,
                                     safe_path, verify_declared_sources)


def projection_dimensions(config, projection):
    hidden = config.hidden_size
    head_dim = getattr(config, "head_dim", None) or hidden // config.num_attention_heads
    width = config.num_key_value_heads * head_dim
    if (hidden <= 0 or head_dim <= 0 or width <= 0 or projection.in_features != hidden or
            projection.out_features != width or tuple(projection.weight.shape) != (width, hidden)):
        raise ValueError("Native projection dimensions disagree with original config/weight")
    return hidden, width, head_dim


def validate_projection_arrays(arrays, rows, hidden, width, expected=None):
    shapes = {"A": (rows, hidden), "B": (hidden, width), "native_C": (rows, width)}
    if set(arrays) != {name + suffix for name in shapes for suffix in ("", "_bf16")}:
        raise ValueError("Unexpected projection bundle fields")
    for name, shape in shapes.items():
        value, words = arrays[name], arrays[name + "_bf16"]
        if (value.shape != shape or value.dtype != np.float32 or words.shape != shape or
                words.dtype != np.uint16 or not np.isfinite(value).all() or
                not np.array_equal(value.view(np.uint32), words.astype(np.uint32) << 16)):
            raise ValueError("Projection shapes/finite views/raw words disagree")
        if expected is not None and not np.array_equal(words, expected[name]):
            raise ValueError("Projection mapping differs from observed native source")


def source_controls(arrays, rows, hidden, width, expected):
    results = {}
    for kind in ("raw_word", "shape", "nonfinite", "K_order", "output_order", "position_order"):
        changed = {key: value.copy() for key, value in arrays.items()}
        if kind == "raw_word":
            changed["A_bf16"][0, 0] ^= np.uint16(1)
        elif kind == "shape":
            changed["B"] = changed["B"][:, :-1].copy()
        elif kind == "nonfinite":
            changed["A"][0, 0] = np.nan
        else:
            name, axis = {"K_order": ("B", 0), "output_order": ("native_C", 1),
                          "position_order": ("A", 0)}[kind]
            changed[name] = np.roll(changed[name], 1, axis=axis)
            changed[name + "_bf16"] = np.roll(changed[name + "_bf16"], 1, axis=axis)
        try:
            validate_projection_arrays(changed, rows, hidden, width, expected)
        except ValueError:
            results[kind + "_refused"] = True
        else:
            raise AssertionError("Source negative control admitted: " + kind)
    return results


def load_query(bundle, manifest):
    path = bundle / "operands.npz"
    identity(path, manifest["npz_sha256"])
    required = {name + suffix + ".npy" for name in ("A", "B", "native_C") for suffix in ("", "_bf16")}
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if (len(members) != len(required) or {member.filename for member in members} != required or
                sum(member.file_size for member in members) > 32 * 1024**2):
            raise ValueError("Unexpected source query archive members")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    validate_projection_arrays(arrays, 128, 2048, 2048)
    return arrays


def capture(args, out, report):
    import torch
    from transformers.models.llama import modeling_llama

    started = time.monotonic()
    query_id = identity(args.query_bundle / "manifest.json", args.query_manifest_sha256)
    query_manifest = json.loads(Path(query_id["path"]).read_text())
    query = load_query(args.query_bundle, query_manifest)
    baseline_id = identity(args.attention_bundle / "receipt.json", args.attention_receipt_sha256)
    baseline = json.loads(Path(baseline_id["path"]).read_text())
    if (baseline["status"] != "passed_native_attention_baseline" or baseline["source_manifest"] != query_id or
            baseline["native_repeat_raw_exact"] is not True or baseline["source_bookends_verified"] is not True):
        raise ValueError("Require authenticated original native attention baseline")
    baseline_npz = identity(args.attention_bundle / "native-attention.npz", baseline["native_attention_npz"]["sha256"])
    records = [query_id, baseline_id, baseline_npz, identity(__file__), identity(attention.__file__),
               identity(inspect.getfile(torch.nn.modules.module.register_module_forward_pre_hook))]
    observed, holders, handles = {"k_proj": [], "v_proj": []}, [], []

    def observe_model(module, _inputs):
        if type(module) is not modeling_llama.LlamaModel:
            return
        if holders:
            if holders[0] is not module:
                raise AssertionError("More than one native model was observed")
            return
        holders.append(module)
        for name in observed:
            projection = getattr(module.layers[0].self_attn, name)
            projection_dimensions(module.config, projection)

            def observe(_projection, inputs, output, name=name):
                observed[name].append((inputs[0].detach().clone(), output.detach().clone()))

            handles.append(projection.register_forward_hook(observe))

    global_hook = torch.nn.modules.module.register_module_forward_pre_hook(observe_model)
    native_out = out / "native-attention"
    native_out.mkdir()
    native_report = {"status": "failed", "observational_projection_hooks": list(observed),
                     "observer_source": identity(__file__)}
    try:
        attention.evaluate(SimpleNamespace(native_bundle=args.query_bundle,
                           native_manifest_sha256=args.query_manifest_sha256, provider_receipt=None),
                           native_out, native_report)
    except Exception as exc:
        native_report.update(status="failed", error=repr(exc))
        raise
    finally:
        global_hook.remove()
        for handle in handles:
            handle.remove()
        write_pretty_json(native_out / "receipt.json", native_report, allow_nan=False, sort_keys=False)
    if native_report["native_arrays"] != baseline["native_arrays"]:
        raise AssertionError("Observation changed native attention/cache/positions/masks")
    if len(holders) != 1 or any(len(values) != 2 for values in observed.values()):
        raise AssertionError("Wrong native observation invocation counts")
    model = holders[0]
    outputs, manifests = {}, {}
    for name, values in observed.items():
        projection = getattr(model.layers[0].self_attn, name)
        hidden, width, head_dim = projection_dimensions(model.config, projection)
        first, repeated = values
        if any(attention.tensor_identity(a, torch) != attention.tensor_identity(b, torch) for a, b in zip(first, repeated)):
            raise AssertionError("Native K/V projection repeat changed raw bytes")
        activation, current = first
        if activation.shape != (1, 128, hidden) or current.shape != (1, 128, width):
            raise ValueError("Native projection invocation dimensions differ")
        if not np.array_equal(attention.raw_array(activation[0], torch), query["A_bf16"]):
            raise ValueError("K/V RMSNorm input differs from original full query input")
        if name == "v_proj":
            cache_view = current.view(1, 128, model.config.num_key_value_heads, head_dim).transpose(1, 2)
            if attention.tensor_identity(cache_view, torch) != baseline["native_arrays"]["values"]:
                raise ValueError("Original V projection does not match native cache layout raw")
        parameter = attention.tensor_identity(projection.weight, torch)
        if parameter != native_report["materialized_parameters"]["model.layers.0.self_attn." + name + ".weight"]:
            raise ValueError("Observed K/V parameter differs from original checkpoint")
        arrays = {}
        for key, tensor in (("A", activation[0]), ("B", projection.weight.T), ("native_C", current[0])):
            arrays[key] = tensor.detach().contiguous().float().numpy().copy()
            arrays[key + "_bf16"] = attention.raw_array(tensor, torch)
        expected = {key: arrays[key + "_bf16"].copy() for key in ("A", "B", "native_C")}
        validate_projection_arrays(arrays, 128, hidden, width, expected)
        controls = source_controls(arrays, 128, hidden, width, expected)
        destination = out / name
        destination.mkdir()
        np.savez(destination / "operands.npz", **arrays)
        source = dict(query_manifest["source"])
        source.update(original_query_capture_script=source["capture_script"], capture_script=identity(__file__),
                      original_query_manifest=query_id, original_attention_baseline=baseline_id,
                      original_attention_npz=baseline_npz, native_attention_observer=identity(attention.__file__))
        manifest = {"schema_version": 1, "status": "native_selective_capture_passed", "source_dtype": "bfloat16",
                    "npz_sha256": identity(destination / "operands.npz")["sha256"], "source": source,
                    "site": {"module": "model.layers.0.self_attn." + name, "batch": 0,
                             "positions": [0, 128], "output_channels": [0, width], "k_range": [0, hidden],
                             "native_execution_shape": [1, 128, width], "full_k_reduction": True,
                             "native_head_dim": head_dim, "native_kv_heads": model.config.num_key_value_heads},
                    "arrays": {key: array_identity(value) for key, value in arrays.items()},
                    "negative_controls": controls,
            "source_binding_control_domain": "Pure source-array admission, not target program mutations",
                    "capture": {"native_repeats_raw_exact": True, "entire_native_attention_and_cache_match_baseline_raw": True,
                                "native_attention_receipt": identity(native_out / "receipt.json"),
                                "projection_boundary": "Original native Linear output before native RoPE/cache",
                                "native_value_cache_raw_layout_checked": name == "v_proj",
                                "dimension_basis": "Original config and actual native Linear weight/out_features",
                                "observer_returns_none": True, "full_model_executed": False, "mlp_executed": False},
                    "claim_limits": ["source_native_projection_only", "no_target_execution", "no_language_quality_claim"]}
        manifests[name] = manifest
    verify_declared_sources(query_manifest["source"])
    for record in records + native_report["source_records"]:
        identity(record["path"], record["sha256"])
    peak_rss = check_memory()
    for name, manifest in manifests.items():
        destination = out / name
        identity(destination / "operands.npz", manifest["npz_sha256"])
        manifest["capture"]["source_bookends_verified"] = True
        write_pretty_json(destination / "manifest.json", manifest, allow_nan=False, sort_keys=False)
        outputs[name] = {"manifest": identity(destination / "manifest.json"), "npz_sha256": manifest["npz_sha256"],
                         "shape_M_K_N": [128, manifest["site"]["k_range"][1], manifest["site"]["output_channels"][1]],
                         "controls": manifest["negative_controls"]}
    report.update(status="passed_native_kv_capture", outputs=outputs, source_records=records,
                  source_bookends_verified=True, existing_native_attention_cache_raw_exact=True,
                  full_model_executed=False, mlp_executed=False, target_execution=False,
                  wall_seconds=time.monotonic() - started, peak_rss_bytes=peak_rss, cpu_threads=1)


def main():
    from workload_cli import run_report
    from merlin.common.paths import artifacts_dir

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-bundle", type=safe_path, required=True)
    parser.add_argument("--query-manifest-sha256", required=True)
    parser.add_argument("--attention-bundle", type=safe_path, required=True)
    parser.add_argument("--attention-receipt-sha256", required=True)
    parser.add_argument("--out", type=safe_path, required=True)
    args = parser.parse_args()
    if not args.out.is_relative_to(safe_path(artifacts_dir())):
        parser.error("--out must remain inside this Merlin worktree's artifacts_dir")
    report = {"status": "failed", "argv": sys.argv, "python": sys.version}
    run_report(args, lambda args, report: capture(args, args.out, report),
        filename='receipt.json', output=args.out, report=report,
        summary=("status", "outputs", "wall_seconds", "peak_rss_bytes"), reraise=True, error_repr=True,
        memory_unit=None)


if __name__ == "__main__":
    main()
