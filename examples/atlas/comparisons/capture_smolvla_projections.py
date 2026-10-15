"""Capture original layer-zero K/V or conditional actual-Q-context O in SmolVLA.

The selected pinned native Linear is replayed on authenticated retained operands.
K is captured before native RoPE; K/RoPE and V close the original cache. O uses
the accepted actual-Q/native-KV attention context, before residual addition.
Only exported matrices gain fifteen zero rows; native positions remain177.
"""

import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np

import capture_smolvla as original
import capture_smolvla_full_query as full_query
import capture_tinyllama_native as common
from merlin.common.paths import artifacts_dir


HELPER_HASHES = {
    "capture_smolvla": "b1c32b63288eb99881af8f0812cdf5898a8e586618503392118773bab3b269c1",
    "capture_smolvla_full_query": "ae524c00aa29f68eb65cb1d48013e303f9346622e35880d9a235d6ec3f7e0b2b",
    "capture_tinyllama_native": "2070992ba201017de9ccbf92f653c61d1b66f2d7cda7ec46dabda63d555a4ff6",
}


def same(actual, expected, message):
    if common.array_identity(actual) != common.array_identity(expected):
        raise ValueError(message)


def parameter_identity(parameter, word_dtype):
    return {"dtype": str(parameter.dtype), "shape": list(parameter.shape),
            "words": common.array_identity(parameter.detach().contiguous().view(word_dtype).numpy())}


def require_active_parameters(attention, expected, word_dtype):
    actual = {part: parameter_identity(getattr(attention, part + "_proj").weight, word_dtype) for part in expected}
    if actual != expected:
        raise ValueError("Active native projection parameter dtype/shape/bytes differ from checkpoint")
    return actual


def validate_export(arrays, native_a, native_b, native_c):
    width = native_c.shape[1]
    if native_a.shape != (177, 960) or native_c.shape not in ((177, 320), (177, 960)):
        raise ValueError("Wrong native projection extent")
    if set(arrays) != {name + suffix for name in ("A", "B", "native_C") for suffix in ("", "_bf16")}:
        raise ValueError("Unexpected exported projection fields")
    for name, shape in (("A", (192, 960)), ("B", (960, width)), ("native_C", (192, width))):
        floats, words = arrays[name], arrays[name + "_bf16"]
        if floats.shape != shape or words.shape != shape or floats.dtype != np.float32 or words.dtype != np.uint16:
            raise ValueError("Projection export shape/dtype differs: " + name)
        if not np.isfinite(floats).all():
            raise ValueError("Nonfinite projection operand")
        same(floats.view(np.uint32), common.bf16_words_to_float(words).view(np.uint32), "Projection BF16 view differs")
    same(arrays["A_bf16"][:177], native_a, "Native projection input changed")
    same(arrays["B_bf16"], native_b, "Projection checkpoint transpose/layout changed")
    same(arrays["native_C_bf16"][:177], native_c, "Native projection output changed")
    if np.any(arrays["A_bf16"][177:]) or np.any(arrays["native_C_bf16"][177:]):
        raise ValueError("Synthetic matrix rows must be positive zero")


def load_conditional_context(args, manifest, records):
    record = common.identity(args.attention_receipt, args.attention_receipt_sha256)
    receipt = json.loads(Path(record["path"]).read_text())
    expected_checks = ("native_Q_A_B_C_raw_exact", "actual_Q_adaptation_raw_exact", "native_attention_source_used",
                       "original_masks_positions_remain177", "unhooked_instrumented_restored_context_raw_exact",
                       "candidate_repeats_raw_exact", "source_input_parameter_cache_bookends")
    if (receipt["status"] != "completed_hybrid_diagnostic" or receipt["source"] != manifest["source"]
            or not all(receipt["checks"][key] is True for key in expected_checks)
            or receipt["checks"]["original_K_RoPE_cache_words_exact"] != 56640
            or receipt["checks"]["original_V_cache_words_exact"] != 56640
            or receipt["source_script"]["sha256"] != "ef956ec105421b58be4ebb439d60a179a21f323e06c6c9d30bf29537fdb6df8d"):
        raise ValueError("Require the accepted actual-Q/native-KV attention boundary")
    command = receipt["command"]
    if command[command.index("--native-manifest-sha256") + 1] != args.native_manifest_sha256:
        raise ValueError("Conditional attention belongs to another native query bundle")
    bindings = receipt["source_bookends"] + receipt["provider_bookends"]
    if not receipt["source_bookends"] or not receipt["provider_bookends"]:
        raise ValueError("Conditional attention lacks source or physical-output bookends")
    for binding in bindings:
        common.identity(binding["path"], binding["sha256"])
    outputs_id = common.identity(receipt["outputs"]["path"], receipt["outputs"]["sha256"])
    with np.load(outputs_id["path"], allow_pickle=False) as archive:
        contexts = archive["candidate_attention_bf16"].copy()
        original_context = archive["original_attention_bf16"].copy()
        actual_q = archive["actual_query_bf16"].copy()
        masks = {key: archive[key].copy() for key in ("original_pad_mask", "original_attention_pattern", "position_ids")}
    if (contexts.shape != (2, 1, 177, 960) or contexts.dtype != np.uint16
            or original_context.shape != (1, 177, 960) or original_context.dtype != np.uint16
            or actual_q.shape != (177, 960) or actual_q.dtype != np.uint16):
        raise ValueError("Conditional context shape/dtype changed")
    same(contexts[0], contexts[1], "Conditional attention repeats differ")
    if common.array_identity(actual_q) != receipt["actual_query"]:
        raise ValueError("Conditional attention Q payload differs from admitted physical Q")
    observations = [item for item in receipt["observations"] if item["candidate"]]
    if len(observations) != 2 or any(item["context"] != common.array_identity(contexts[0]) for item in observations):
        raise ValueError("Conditional context payload differs from observed native attention")
    originals = [item for item in receipt["observations"] if not item["candidate"]]
    if len(originals) != 3 or any(item["context"] != common.array_identity(original_context) for item in originals):
        raise ValueError("Original context payload differs from observed native attention")
    records.extend([record, outputs_id, *bindings])
    return contexts[0, 0].copy(), original_context[0].copy(), masks, record


def capture(args, report):
    records = [common.identity(__file__)]
    for module in (original, full_query, common):
        records.append(common.identity(module.__file__, HELPER_HASHES[module.__name__]))
    manifest_id = common.identity(args.native_bundle / "manifest.json", args.native_manifest_sha256)
    manifest = json.loads(Path(manifest_id["path"]).read_text())
    site, source = manifest["site"], copy.deepcopy(manifest["source"])
    if (manifest["status"] != "native_selective_capture_passed" or manifest["source_dtype"] != "bfloat16"
            or source["revision"] != original.CHECKPOINT_REVISION
            or not site["module"].endswith(".layers.0.self_attn.q_proj")
            or site["matrix_shape"] != [192, 960, 960] or site["native_rows"] != 177
            or site["attention_valid_positions"] != 141 or site["original_masked_positions"] != 36
            or site["padding"]["native_masks_or_positions_extended"] is not False):
        raise ValueError("Require the original complete native prefix query capture")
    records += [manifest_id, common.identity(args.native_bundle / "operands.npz", manifest["npz_sha256"]),
                common.identity(manifest["native_query"]["path"], manifest["native_query"]["sha256"])]
    with np.load(records[-2]["path"], allow_pickle=False) as archive:
        query_arrays = {key: archive[key].copy() for key in archive.files}
    with np.load(records[-1]["path"], allow_pickle=False) as archive:
        native_q = archive["Q_bf16"].copy()
        native_masks = {key: archive[key].copy() for key in ("prefix_pad_mask", "prefix_attention_mask", "position_ids")}
    full_query.validate_padding(query_arrays, native_q, query_arrays["A_bf16"][:177])
    native_a = query_arrays["A_bf16"][:177].copy()
    conditional = None
    if args.projection == "o":
        native_a, original_context, context_masks, conditional = load_conditional_context(args, manifest, records)
        for left, right in (("original_pad_mask", "prefix_pad_mask"), ("original_attention_pattern", "prefix_attention_mask"),
                            ("position_ids", "position_ids")):
            same(context_masks[left], native_masks[right], "Conditional context changed original masks/positions")
    common.verify_declared_sources(source)
    common.identity(source["weights"]["path"], original.WEIGHTS_SHA)
    archive_id = common.identity(source["historical_original_archive"]["path"], original.ARCHIVE_SHA)
    records.append(archive_id)
    with np.load(archive_id["path"], allow_pickle=False) as archive:
        expected_k = archive["prefix_key_cache"][0].copy()
        expected_v = archive["prefix_value_cache"][0].copy()
        same(archive["prefix_pad_mask"], native_masks["prefix_pad_mask"], "Original mask binding changed")
        same(archive["prefix_attention_mask"], native_masks["prefix_attention_mask"], "Original attention pattern changed")
    if expected_k.shape != (1, 177, 5, 64) or expected_v.shape != expected_k.shape:
        raise ValueError("Original layer-zero cache geometry changed")
    import torch
    from safetensors import safe_open
    from transformers.models.llama import modeling_llama
    from lerobot.policies.smolvla import smolvlm_with_expert
    versions = {key: importlib.metadata.version(key) for key in ("torch", "transformers", "lerobot", "numpy")}
    if versions != source["versions"] or sys.version != source["python"]:
        raise ValueError("Native environment differs from the original query capture")
    records += [common.identity(modeling_llama.__file__, original.LLAMA_SHA),
                common.identity(smolvlm_with_expert.__file__, original.EXPERT_SOURCE_SHA),
                common.identity(Path(torch.__file__).parent / "lib/libtorch_cpu.so", source["torch_cpu_library"]["sha256"])]
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = modeling_llama.LlamaConfig(**json.loads(Path(source["backbone_config"]["path"]).read_text())["text_config"])
    if (config.hidden_size, config.num_attention_heads, config.num_key_value_heads, config.head_dim) != (960, 15, 5, 64):
        raise ValueError("Original Smol projection topology changed")
    prefix = "model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn."
    parts = ("k", "v", "o") if args.projection == "o" else ("k", "v")
    names = {part: prefix + part + "_proj.weight" for part in parts}
    with safe_open(source["weights"]["path"], framework="pt", device="cpu") as handle:
        weights = {part: handle.get_tensor(name) for part, name in names.items()}
    rng = torch.random.get_rng_state().clone()
    try:
        with torch.device("meta"):
            layer = modeling_llama.LlamaDecoderLayer(config, 0)
        for part, weight in weights.items():
            if weight.dtype != torch.bfloat16 or tuple(weight.shape) != ((960, 960) if part == "o" else (320, 960)):
                raise ValueError("Original projection checkpoint geometry changed")
            if names[part] in source["parameters"] and common.array_identity(weight.view(torch.uint16).numpy()) != source["parameters"][names[part]]:
                raise ValueError("Original projection checkpoint bytes changed")
            getattr(layer.self_attn, part + "_proj").weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.eval()
        module = getattr(layer.self_attn, args.projection + "_proj")
        params = {part: common.array_identity(weight.view(torch.uint16).numpy()) for part, weight in weights.items()}
        active_before = {part: parameter_identity(weight, torch.uint16) for part, weight in weights.items()}
        require_active_parameters(layer.self_attn, active_before, torch.uint16)
        expected_b = weights[args.projection].t().contiguous().view(torch.uint16).numpy().copy()
        inputs = torch.from_numpy(native_a.copy()).view(torch.bfloat16).reshape(1, 177, 960)
        positions = torch.from_numpy(native_masks["position_ids"].copy())
        same(positions.numpy(), np.cumsum(native_masks["prefix_pad_mask"], axis=1) - 1, "Original positions changed")
        observations = []

        def observe(_, arguments, output):
            if len(arguments) != 1 or arguments[0].dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
                raise ValueError("Original projection invocation/dtype changed")
            same(arguments[0].view(torch.uint16).numpy()[0], native_a, "Projection received another context/input")
            if output.shape != (1, 177, module.out_features) or not torch.isfinite(output).all():
                raise ValueError("Projection native output geometry/finite check failed")
            observations.append(common.array_identity(output.view(torch.uint16).numpy()))

        with torch.inference_mode():
            baseline = module(inputs).clone()
            hook = module.register_forward_hook(observe)
            try:
                repeated = [module(inputs).clone() for _ in range(2)]
            finally:
                hook.remove()
            restored = module(inputs).clone()
            for value in (*repeated, restored):
                same(value.view(torch.uint16).numpy(), baseline.view(torch.uint16).numpy(), "Native projection repeat/hook removal changed words")
            if len(observations) != 2:
                raise ValueError("Projection observation callback count changed")
            original_inputs = torch.from_numpy(query_arrays["A_bf16"][:177].copy()).view(torch.bfloat16).reshape(1, 177, 960)
            key = layer.self_attn.k_proj(original_inputs).view(1, 177, 5, 64)
            value = layer.self_attn.v_proj(original_inputs).view(1, 177, 5, 64)
            rotated_key = smolvlm_with_expert.apply_rope(key, positions)
            same(rotated_key.view(torch.uint16).numpy(), expected_k, "Native K/RoPE differs from archived cache")
            same(value.view(torch.uint16).numpy(), expected_v, "Native V differs from archived cache")
            original_output = module(torch.from_numpy(original_context.copy()).view(torch.bfloat16).reshape(1, 177, 960)).clone() if conditional else None
        require_active_parameters(layer.self_attn, active_before, torch.uint16)
        same(inputs.view(torch.uint16).numpy()[0], native_a, "Projection mutated caller inputs")
        if not torch.equal(rng, torch.random.get_rng_state()):
            raise ValueError("Native projection consumed RNG state")
        native_c = baseline[0].contiguous().view(torch.uint16).numpy().copy()
        b_words = module.weight.t().contiguous().view(torch.uint16).numpy().copy()
        same(b_words, expected_b, "Exported B differs from selected authenticated active checkpoint transpose")
        if args.projection in ("k", "v"):
            selected = key if args.projection == "k" else value
            same(native_c, selected.reshape(177, 320).view(torch.uint16).numpy(), "Selected native K/V differs from cache-bound projection")
        closure = {"K_after_native_RoPE": common.array_identity(rotated_key.view(torch.uint16).numpy()),
                   "V": common.array_identity(value.view(torch.uint16).numpy()), "archived_words_each": 56640}
        controls = []
        original_parameter = module.weight
        changed_parameter = original_parameter.detach().clone()
        changed_parameter.view(torch.int16).flatten()[0] ^= 1
        module.weight = torch.nn.Parameter(changed_parameter, requires_grad=False)
        try:
            try:
                require_active_parameters(layer.self_attn, active_before, torch.uint16)
            except ValueError:
                controls.append({"control": "rebound_active_projection_weight", "refused": True})
            else:
                raise AssertionError("Rebound active native parameter was admitted")
        finally:
            module.weight = original_parameter
        require_active_parameters(layer.self_attn, active_before, torch.uint16)
        wrong = inputs.clone(); wrong.view(torch.int16).flatten()[0] ^= 1
        try:
            observe(module, (wrong,), baseline)
        except ValueError:
            controls.append({"control": "changed_native_or_conditional_input", "refused": True})
        else:
            raise AssertionError("Changed projection input was admitted")
        changed_cache = expected_v.copy(); changed_cache.flat[0] ^= np.uint16(1)
        try:
            same(value.view(torch.uint16).numpy(), changed_cache, "Changed V cache")
        except ValueError:
            controls.append({"control": "changed_original_V_cache", "refused": True})
        else:
            raise AssertionError("Changed V cache was admitted")
        with torch.inference_mode():
            wrong_key = smolvlm_with_expert.apply_rope(key, positions + 1)
        try:
            same(wrong_key.view(torch.uint16).numpy(), expected_k, "Wrong K RoPE positions")
        except ValueError:
            controls.append({"control": "changed_native_K_positions", "refused": True})
        else:
            raise AssertionError("Wrong K positions were admitted")
        if conditional:
            try:
                same(original_context, native_a, "Original context is not actual-Q conditional context")
            except ValueError:
                controls.append({"control": "original_context_instead_of_actual_Q_context", "refused": True})
            else:
                raise AssertionError("Conditional-context negative lacks an observed difference")
    finally:
        torch.random.set_rng_state(rng)
    padded_a, padded_c = np.zeros((192, 960), np.uint16), np.zeros((192, native_c.shape[1]), np.uint16)
    padded_a[:177], padded_c[:177] = native_a, native_c
    arrays = {"A_bf16": padded_a, "B_bf16": b_words, "native_C_bf16": padded_c}
    arrays.update({name: common.bf16_words_to_float(arrays[name + "_bf16"]) for name in ("A", "B", "native_C")})
    validate_export(arrays, native_a, b_words, native_c)
    wrong_layout = {key: value.copy() for key, value in arrays.items()}
    wrong_layout["B_bf16"] = wrong_layout["B_bf16"].reshape(-1)[::-1].reshape(b_words.shape).copy()
    wrong_layout["B"] = common.bf16_words_to_float(wrong_layout["B_bf16"])
    try:
        validate_export(wrong_layout, native_a, b_words, native_c)
    except ValueError:
        controls.append({"control": "changed_checkpoint_B_layout", "refused": True})
    else:
        raise AssertionError("Changed checkpoint layout was admitted")
    for name in ("A", "native_C"):
        changed = {key: value.copy() for key, value in arrays.items()}
        changed[name + "_bf16"][177, 0] = 1
        changed[name] = common.bf16_words_to_float(changed[name + "_bf16"])
        try:
            validate_export(changed, native_a, b_words, native_c)
        except ValueError:
            controls.append({"control": "nonzero_export_padding_" + name, "refused": True})
        else:
            raise AssertionError("Nonzero synthetic rows were admitted")
    common.verify_declared_sources(source)
    for record in records:
        common.identity(record["path"], record["sha256"])
    source["projection_capture"] = {"script": records[0], "helpers": records[1:4], "native_query_manifest": manifest_id,
                                     "native_query_operands": records[5], "conditional_attention": conditional,
                                     "parameters": {names[part]: identity for part, identity in params.items()},
                                     "active_parameter_bookends": active_before}
    np.savez(args.output / "operands.npz", **arrays)
    np.savez(args.output / "native-projection.npz", native_A_bf16=native_a, native_C_bf16=native_c,
             original_O_bf16=original_output.view(torch.uint16).numpy()[0] if conditional else np.empty((0, 0), np.uint16),
             **native_masks)
    report.update(status="native_selective_capture_passed", source_dtype="bfloat16", source=source,
                  npz_sha256=common.digest(args.output / "operands.npz"), arrays={key: common.array_identity(value) for key, value in arrays.items()},
                  native_projection=common.identity(args.output / "native-projection.npz"),
                  site={"module": prefix + args.projection + "_proj", "positions": [0, 177], "k_range": [0, 960],
                        "output_channels": [0, native_c.shape[1]], "matrix_shape": [192, 960, native_c.shape[1]],
                        "native_rows": 177, "valid_rows": 177, "attention_valid_positions": 141, "original_masked_positions": 36,
                        "full_k_reduction": True, "before_RoPE": args.projection == "k", "before_residual": args.projection == "o",
                        "conditional_context": "actual_Q_original_native_KV" if conditional else "original_native_RMSNorm_cast_input",
                        "padding": {"kind": "exported_matrix_zero_rows_only", "rows": [177, 192], "native_masks_or_positions_extended": False,
                                    "downstream_strip": "target_C[:177,:] before native RoPE/cache/residual use"}},
                  capture={"route": "pinned_native_Linear_replay_on_authenticated_retained_boundary", "full_policy_executed": False,
                           "native_reference_replaced": False, "native_repeats_hook_removal_raw_exact": True,
                           "inputs_parameters_preserved": True, "RNG_preserved": True, "cache_closure": closure},
                  negative_controls=controls, source_bookends=records,
                  claim_limits=manifest["claim_limits"] + ["named_projection_replay_not_fresh_policy_or_vision",
                      "conditional_O_requires_recapture_if_actual_K_or_V_changes_attention_context", "matrix_padding_is_not_native_policy_positions"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection", choices=("k", "v", "o"), required=True)
    parser.add_argument("--native-bundle", type=common.safe_path, required=True)
    parser.add_argument("--native-manifest-sha256", required=True)
    parser.add_argument("--attention-receipt", type=common.safe_path)
    parser.add_argument("--attention-receipt-sha256")
    parser.add_argument("--output-dir", type=common.safe_path, required=True)
    args = parser.parse_args()
    if args.projection == "o" and (args.attention_receipt is None or args.attention_receipt_sha256 is None):
        parser.error("Conditional O requires both the accepted actual-Q attention receipt and its hash")
    if args.projection != "o" and (args.attention_receipt is not None or args.attention_receipt_sha256 is not None):
        parser.error("Original K/V capture does not accept a conditional attention source")
    args.output = args.output_dir
    if args.output == common.safe_path(artifacts_dir()) or not args.output.is_relative_to(common.safe_path(artifacts_dir())):
        parser.error("Output must be a fresh directory beneath the configured artifacts root")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"schema_version": 1, "status": "failed", "command": sys.argv, "source_script": common.identity(__file__),
              "full_policy_executed": False, "vision_prefix_recomputed": False, "task_quality_qualified": False}
    code = 1
    try:
        capture(args, report)
        code = 0
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
    report.update(wall_seconds=time.monotonic() - started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "manifest": str(args.output / "manifest.json"), "error": report.get("error")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
