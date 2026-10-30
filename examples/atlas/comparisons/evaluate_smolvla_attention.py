"""Propagate actual full-Q model output through retained native SmolVLA attention.

Only layer-zero prefix attention before o_proj is evaluated. Original vision,
policy execution and task quality are not established by this semantic hybrid.
"""

import argparse
import importlib.metadata
from pathlib import Path
import sys
from types import SimpleNamespace, MethodType

import numpy as np

import source_relocation
from source_relocation import argument_relocator, read_metadata

from merlin.common.paths import artifacts_dir
from capture_smolvla import (CHECKPOINT_REVISION, WEIGHTS_SHA, RECEIPT_SHA,
                            ARCHIVE_SHA, OBSERVATION_SHA, EXPERT_SOURCE_SHA, LLAMA_SHA)
from capture_smolvla_full_query import validate_padding
from capture_tinyllama_native import (array_identity, bf16_words_to_float, identity,
                                     safe_path, verify_declared_sources, load_provider_validator)

MASK_SOURCE_SHA = "be4463140f1abc5bce61f8a8c849ba6ba23f37777d19ad9159c2e1ffff60c1c5"


def require_same(actual, expected, message):
    if array_identity(actual) != array_identity(expected):
        raise ValueError(message)


def evaluate(args, report):
    producer = safe_path(args.producer_root)
    validator, validator_id = load_provider_validator(
        producer, args.validator_sha256, args.provider_receipt, args.provider_receipt_sha256)
    manifest, arrays, target, records, admission = validator.admit_contraction(
        receipt_path=safe_path(args.provider_receipt), receipt_sha256=args.provider_receipt_sha256,
        native_bundle=safe_path(args.native_bundle), native_manifest_sha256=args.native_manifest_sha256,
        producer_root=producer, model_root=safe_path(args.model_root),
        metadata_path_mapper=argument_relocator(args).path)
    site, source = manifest["site"], manifest["source"]
    if (manifest["status"] != "native_selective_capture_passed" or manifest["source_dtype"] != "bfloat16"
            or source["revision"] != CHECKPOINT_REVISION or not site["module"].endswith(".layers.0.self_attn.q_proj")
            or site["positions"] != [0, 177] or site["output_channels"] != [0, 960]
            or site["k_range"] != [0, 960] or site["matrix_shape"] != [192, 960, 960]
            or site["native_rows"] != 177 or site["valid_rows"] != 177 or site["attention_valid_positions"] != 141
            or site["padding"]["native_masks_or_positions_extended"] is not False):
        raise ValueError("Require the unchanged complete native prefix query with export-only padding")
    native_query_id = identity(manifest["native_query"]["path"], manifest["native_query"]["sha256"])
    with np.load(native_query_id["path"], allow_pickle=False) as archive:
        native_q = archive["Q_bf16"].copy()
        saved_masks = archive["prefix_pad_mask"].copy()
        saved_attention = archive["prefix_attention_mask"].copy()
        saved_positions = archive["position_ids"].copy()
    validate_padding(arrays, native_q, arrays["A_bf16"][:177])
    if target.shape != (192, 960) or np.any(target[177:].view(np.uint32)):
        raise ValueError("Physical export padding must be positive zero")
    actual_q = target[:177].copy()
    actual_words = (actual_q.view(np.uint32) >> 16).astype(np.uint16)
    require_same(actual_q.view(np.uint32), bf16_words_to_float(actual_words).view(np.uint32),
        "Physical Q is not an exact BF16 view")
    import torch
    from safetensors import safe_open
    from transformers.models.llama import modeling_llama
    from lerobot.policies.smolvla import smolvlm_with_expert, modeling_smolvla
    versions = {key: importlib.metadata.version(key) for key in ("torch", "transformers", "lerobot", "numpy")}
    if versions != source["versions"] or sys.version != source["python"]:
        raise ValueError("Native environment differs from the original query capture")
    bindings = [identity(__file__), identity(source_relocation.__file__), validator_id, native_query_id,
                identity(Path(__file__).with_name("capture_smolvla_full_query.py")),
                identity(Path(__file__).with_name("capture_smolvla.py")),
                identity(Path(__file__).with_name("capture_tinyllama_native.py")),
                identity(smolvlm_with_expert.__file__, EXPERT_SOURCE_SHA), identity(modeling_llama.__file__, LLAMA_SHA),
                identity(modeling_smolvla.__file__, MASK_SOURCE_SHA),
                identity(Path(torch.__file__).parent / "lib/libtorch_cpu.so", source["torch_cpu_library"]["sha256"]),
                identity(source["weights"]["path"], WEIGHTS_SHA),
                identity(source["historical_source_receipt"]["path"], RECEIPT_SHA),
                identity(source["historical_original_archive"]["path"], ARCHIVE_SHA),
                identity(source["observation_receipt"]["path"], OBSERVATION_SHA)]
    verify_declared_sources(source)
    historical = read_metadata(args, report, source["historical_source_receipt"]["path"], RECEIPT_SHA)
    if not historical["source_reference_unchanged"] or historical["source_run"]["output_file"]["sha256"] != ARCHIVE_SHA:
        raise ValueError("Require the original policy archive, not the integer candidate")
    with np.load(source["historical_original_archive"]["path"], allow_pickle=False) as archive:
        embeddings_np = archive["prefix_embeddings"].copy()
        masks_np = archive["prefix_pad_mask"].copy()
        attention_np = archive["prefix_attention_mask"].copy()
        expected_k = archive["prefix_key_cache"][0].copy()
        expected_v = archive["prefix_value_cache"][0].copy()
    for name, value in (("prefix_embeddings", embeddings_np), ("prefix_pad_mask", masks_np), ("prefix_attention_mask",
        attention_np)):
        if array_identity(value)["sha256"] != historical["source_run"]["captures"][name]["sha256"]:
            raise ValueError("Original retained prefix binding changed: " + name)
    if (embeddings_np.shape != (1, 177, 960) or embeddings_np.dtype != np.float32
            or masks_np.shape != (1, 177) or masks_np.dtype != np.bool_ or int(masks_np.sum()) != 141
            or attention_np.shape != (1, 177) or expected_k.shape != (1, 177, 5, 64)
            or expected_v.shape != expected_k.shape or expected_k.dtype != np.uint16 or expected_v.dtype != np.uint16):
        raise ValueError("Original prefix/cache geometry changed")
    require_same(masks_np, saved_masks, "Captured masks differ from original prefix")
    require_same(attention_np, saved_attention, "Captured attention pattern differs from original prefix")
    config = modeling_llama.LlamaConfig(**read_metadata(args, report, source["backbone_config"]["path"], source[
        "backbone_config"]["sha256"])["text_config"])
    if (config.hidden_size != 960 or config.num_attention_heads != 15 or config.num_key_value_heads != 5
            or config.head_dim != 64 or config.rms_norm_eps != 1e-5):
        raise ValueError("Native attention geometry changed")
    config._attn_implementation = "eager"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with torch.device("meta"):
        layer = modeling_llama.LlamaDecoderLayer(config, 0)
    prefix = "model.vlm_with_expert.vlm.model.text_model.layers.0."
    names = [prefix + "input_layernorm.weight"] + [prefix + "self_attn." + part + "_proj.weight" for part in ("q", "k", "v")]
    with safe_open(source["weights"]["path"], framework="pt", device="cpu") as handle:
        weights = {name: handle.get_tensor(name) for name in names}
    for name, shape in zip(names, ((960,), (960, 960), (320, 960), (320, 960))):
        if weights[name].dtype != torch.bfloat16 or tuple(weights[name].shape) != shape:
            raise ValueError("Native BF16 projection geometry changed")
        if name in source["parameters"] and array_identity(weights[name].view(torch.uint16).numpy()) != source["parameters"][
            name]:
            raise ValueError("Native checkpoint parameter changed")
    layer.input_layernorm.weight = torch.nn.Parameter(weights[names[0]], requires_grad=False)
    for part, name in zip(("q", "k", "v"), names[1:]):
        getattr(layer.self_attn, part + "_proj").weight = torch.nn.Parameter(weights[name], requires_grad=False)
    layer.eval()

    def parameter_state():
        values = [layer.input_layernorm.weight] + [getattr(layer.self_attn, part + "_proj").weight for part in ("q", "k", "v")]
        return {name: array_identity(value.detach().view(torch.uint16).numpy()) for name, value in zip(names, values)}

    params_before = parameter_state()
    require_same(weights[names[1]].t().contiguous().view(torch.uint16).numpy(), arrays["B_bf16"], "Captured Q weights changed")
    embeddings = torch.from_numpy(embeddings_np.copy())
    masks, attention = torch.from_numpy(masks_np.copy()), torch.from_numpy(attention_np.copy())
    positions = torch.cumsum(masks, dim=1) - 1
    require_same(positions.numpy(), saved_positions, "Original positions changed")
    mask2d = modeling_smolvla.make_att_2d_masks(masks, attention)
    if mask2d.shape != (1, 177, 177) or mask2d.dtype != torch.bool:
        raise ValueError("Native Boolean attention mask changed")
    original_mask2d = mask2d.numpy().copy()
    replacement = torch.from_numpy(actual_words.copy()).view(torch.bfloat16).reshape(1, 177, 960)
    require_same(replacement.view(torch.uint16).numpy()[0], actual_words, "Actual physical Q adaptation changed words")
    cls = smolvlm_with_expert.SmolVLMWithExpertModel
    facade = SimpleNamespace(num_attention_heads=15, num_key_value_heads=5)
    facade.get_attention_interface = MethodType(cls.get_attention_interface, facade)
    observations, current = [], {}

    def attend(mask, batch, dimension, query, key, value):
        require_same(mask.numpy(), original_mask2d, "Native attention mask changed at the actual interface")
        if batch != 1 or dimension != 64 or current["attention_calls"] != 0:
            raise ValueError("Native attention invoked with wrong geometry or count")
        expected_query = smolvlm_with_expert.apply_rope(current["q"].reshape(1, 177, 15, 64), positions)
        require_same(query.view(torch.uint16).numpy(), expected_query.view(torch.uint16).numpy(),
            "Attention did not receive the selected actual/native Q through native RoPE")
        require_same(key.view(torch.uint16).numpy(), expected_k, "Native K/RoPE no longer matches archived words")
        require_same(value.view(torch.uint16).numpy(), expected_v, "Native V no longer matches archived words")
        current["attention_calls"] += 1
        return cls.eager_attention_forward(facade, mask, batch, dimension, query, key, value)

    facade.eager_attention_forward = attend

    def query_hook(_, inputs, output):
        if current["q_calls"] != 0 or inputs[0].dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
            raise ValueError("Query callback invocation/dtype changed")
        require_same(inputs[0].view(torch.uint16).numpy()[0], arrays["A_bf16"][:177], "Native normalized query input changed")
        require_same(output.view(torch.uint16).numpy()[0], native_q, "Original native query output changed")
        current["q_calls"] += 1
        return replacement.clone() if current["candidate"] else None

    def run(candidate, hooked):
        current.clear()
        current.update(candidate=candidate, q_calls=0, attention_calls=0,
                       q=replacement if candidate else torch.from_numpy(native_q.copy()).view(torch.bfloat16).reshape(1, 177,
            960))
        with torch.inference_mode():
            outputs, cache = cls.forward_attn_layer(facade, [[layer], [None]], [embeddings, None], 0,
                                                   positions, mask2d, 1, 64, use_cache=True, fill_kv_cache=True)
        if current["attention_calls"] != 1 or current["q_calls"] != int(hooked) or len(outputs) != 1:
            raise ValueError("Native pipeline callbacks/cache boundary changed")
        output = outputs[0].detach().clone()
        if output.shape != (1, 177, 960) or output.dtype != torch.bfloat16 or not torch.isfinite(output).all():
            raise ValueError("Native attention context geometry/dtype/finite check failed")
        require_same(cache[0]["key_states"].view(torch.uint16).numpy(), expected_k, "Returned native K cache changed")
        require_same(cache[0]["value_states"].view(torch.uint16).numpy(), expected_v, "Returned native V cache changed")
        observations.append({"candidate": candidate, "hooked": hooked, "query_callbacks": current["q_calls"],
                             "attention_callbacks": current["attention_calls"], "context": array_identity(output.view(
            torch.uint16).numpy())})
        return output

    original = run(False, False)
    hook = layer.self_attn.q_proj.register_forward_hook(query_hook)
    try:
        instrumented = run(False, True)
        candidates = [run(True, True) for _ in range(2)]
    finally:
        hook.remove()
    restored = run(False, False)
    for left, right, message in ((original, instrumented, "Native observation hook changed context"),
                                 (original, restored, "Hook removal did not restore original context"),
                                 (candidates[0], candidates[1], "Actual-Q candidate context did not repeat exactly")):
        require_same(left.view(torch.uint16).numpy(), right.view(torch.uint16).numpy(), message)
    # Exercise the same real attention-interface guard with deliberately wrong inputs.
    current.update(q=replacement, attention_calls=0)
    rope_q = smolvlm_with_expert.apply_rope(replacement.reshape(1, 177, 15, 64), positions)
    key = torch.from_numpy(expected_k.copy()).view(torch.bfloat16)
    value = torch.from_numpy(expected_v.copy()).view(torch.bfloat16)
    changed_q = rope_q.clone()
    changed_q.view(torch.int16).flatten()[0] ^= 1
    wrong_mask = mask2d.clone()
    wrong_mask.flatten()[0] = ~wrong_mask.flatten()[0]
    controls = []
    for name, values in (("changed_actual_query_words_at_attention", (mask2d, 1, 64, changed_q, key, value)),
                         ("changed_original_mask_at_attention", (wrong_mask, 1, 64, rope_q, key, value)),
                         ("synthetic_positions_in_native_attention", (torch.ones((1, 192, 192), dtype=torch.bool), 1, 64, rope_q,
        key, value))):
        try:
            attend(*values)
        except ValueError:
            controls.append({"control": name, "refused": True})
        else:
            raise AssertionError("Attention callback negative admitted: " + name)
    for value, original_value, message in ((embeddings.numpy(), embeddings_np, "Caller prefix changed"),
                                           (masks.numpy(), masks_np, "Caller mask changed"),
                                           (attention.numpy(), attention_np, "Caller attention pattern changed"),
                                           (positions.numpy(), saved_positions, "Caller positions changed"),
                                           (mask2d.numpy(), original_mask2d, "Caller native Boolean mask changed")):
        require_same(value, original_value, message)
    if params_before != parameter_state():
        raise ValueError("Native parameters changed")
    verify_declared_sources(source)
    for binding in bindings:
        identity(binding["path"], binding["sha256"])
    validator.recheck(records)
    native = original.float().numpy()
    actual = candidates[0].float().numpy()
    delta = actual.astype(np.float64) - native.astype(np.float64)
    close = np.isclose(actual, native, rtol=1e-4, atol=1e-4)
    norm = float(np.linalg.norm(native.astype(np.float64)))
    np.savez(args.output / "outputs.npz", original_attention_bf16=original.view(torch.uint16).numpy(),
             candidate_attention_bf16=np.stack([value.view(torch.uint16).numpy() for value in candidates]),
             actual_query_bf16=actual_words, native_query_bf16=native_q,
             original_pad_mask=masks_np, original_attention_pattern=attention_np, original_mask_2d=mask2d.numpy(),
        position_ids=saved_positions)
    report.update(status="completed_hybrid_diagnostic", source=source, native_environment=versions, physical_admission=admission,
                  actual_query=array_identity(actual_words), native_query=array_identity(native_q),
        original_mask_2d=array_identity(mask2d.numpy()),
                  parameters=params_before, observations=observations, negative_controls=controls,
                  checks={"native_Q_A_B_C_raw_exact": True, "actual_Q_adaptation_raw_exact": True,
                          "original_K_RoPE_cache_words_exact": expected_k.size, "original_V_cache_words_exact": expected_v.size,
                          "native_attention_source_used": True, "original_masks_positions_remain177": True,
                          "unhooked_instrumented_restored_context_raw_exact": True, "candidate_repeats_raw_exact": True,
                          "source_input_parameter_cache_bookends": True},
                  source_fidelity={"rtol": 1e-4, "atol": 1e-4, "passed": bool(close.all()), "failed_values": int((~close).sum()),
                                   "max_abs": float(np.abs(delta).max()), "relative_l2": float(np.linalg.norm(
        delta) / norm) if norm else None},
                  source_bookends=bindings, provider_bookends=records, outputs=identity(args.output / "outputs.npz"),
                  claim_limits=manifest["claim_limits"] + ["new_selective_native_attention_reference_not_historical_terminal",
        "stops_before_native_output_projection"])


def main():
    from workload_cli import run_report
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native-bundle", "provider-receipt", "producer-root", "model-root", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("native-manifest-sha256", "provider-receipt-sha256", "validator-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--relocate", action="append", default=[], metavar="OLD=NEW")
    args = parser.parse_args()
    args.output = safe_path(args.output_dir)
    if args.output == safe_path(artifacts_dir()) or not args.output.is_relative_to(safe_path(artifacts_dir())):
        parser.error("Output must be fresh and beneath the configured artifact root")
    report = {"status": "failed", "claim": __doc__, "command": sys.argv, "cwd": str(Path.cwd()),
              "source_script": identity(__file__), "python_executable": sys.executable,
              "complete_atlas_execution": False, "timing_qualified": False, "application_quality_qualified": False,
              "vision_prefix_recomputed": False, "full_policy_executed": False}
    return run_report(args, evaluate,
        filename='receipt.json', output=args.output, report=report, atomic=True)


if __name__ == "__main__":
    raise SystemExit(main())
