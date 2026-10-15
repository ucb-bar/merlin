"""Recover a native first-layer SmolVLA contraction from retained source captures.

The original-policy prefix archive supplies real TRAIN inputs. Only RMSNorm and
Q/K weights are loaded; a native LeRobot forward hook captures Q, and the archived
native layer-zero K cache authenticates the reconstructed normalization route.
"""

import argparse
import importlib.metadata
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

from capture_tinyllama_native import (
    array_identity, bf16_words_to_float, digest, identity, round_bf16_words, safe_path,
    verify_declared_sources, artifact_output,
)


CHECKPOINT_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
WEIGHTS_SHA = "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb"
RECEIPT_SHA = "c68fdc9bc2002cf9bb889777c04e405d0ef8af59fc2582840f667114fbe3ee67"
ARCHIVE_SHA = "d41c48264756e01d897be778f96e6428f26f23a4cd054a8ef7f372bf7257271a"
OBSERVATION_SHA = "434b0d3789100fd7f53fb9837471031f3e4d7a2fc7b1cf1a99166c29777bcbb4"
EXPERT_SOURCE_SHA = "b70356145870c7da1e92a2195ee4626f7e9f9387576c6bed5ba2bfefae2a38d9"
LLAMA_SHA = "31bf660a663259134324bc65da4e155951dc89c5ca46471d2325a9938e859e26"


def capture(args):
    import torch
    from safetensors import safe_open
    from transformers.models.llama import modeling_llama
    from lerobot.policies.smolvla import smolvlm_with_expert
    from merlin.common.paths import artifacts_dir

    started = time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if importlib.metadata.version("lerobot") != "0.5.1":
        raise ValueError("Retained LeRobot 0.5.1 implementation is required")
    source = {
        "checkpoint": "lerobot/smolvla_base", "revision": CHECKPOINT_REVISION,
        "weights": identity(args.checkpoint / "model.safetensors", WEIGHTS_SHA),
        "historical_source_receipt": identity(args.receipt, RECEIPT_SHA),
        "historical_original_archive": identity(args.archive, ARCHIVE_SHA),
        "observation_receipt": identity(args.observation_manifest, OBSERVATION_SHA),
        "smolvlm_with_expert": identity(smolvlm_with_expert.__file__, EXPERT_SOURCE_SHA),
        "modeling_llama": identity(modeling_llama.__file__, LLAMA_SHA),
        "backbone_config": identity(args.backbone / "config.json"),
        "capture_script": identity(__file__),
        "capture_helpers": identity(Path(__file__).with_name("capture_tinyllama_native.py")),
        "torch_cpu_library": identity(Path(torch.__file__).parent / "lib/libtorch_cpu.so"),
        "python": sys.version,
        "versions": {key: importlib.metadata.version(key) for key in ("torch", "transformers", "lerobot", "numpy")},
        "source_policies": {"precision": "original_mixed_BF16_FP32_CPU",
                            "projection_quantization": False, "original_checkpoint_parameters": True},
    }
    receipt = json.loads(args.receipt.read_text())
    observation = json.loads(args.observation_manifest.read_text())
    if (receipt["fixture_kind"] != "dataset" or not receipt["source_reference_unchanged"]
            or receipt["source_run"]["output_file"]["sha256"] != ARCHIVE_SHA):
        raise ValueError("Archive is not the retained original-policy source")
    if (observation["checkpoint_revision"] != CHECKPOINT_REVISION
            or observation["dataset_revision"] != "3d6d687a25cdf1565cdf24550814f72d999a861d"
            or observation["split"] != "train"
            or (observation["episode_index"], observation["frame_index"]) != (0, 0)):
        raise ValueError("Expected retained TRAIN episode-zero frame-zero provenance")
    with np.load(args.archive, allow_pickle=False) as archive:
        embeddings_np = archive["prefix_embeddings"].copy()
        masks_np = archive["prefix_pad_mask"].copy()
        attention_np = archive["prefix_attention_mask"].copy()
        expected_k = archive["prefix_key_cache"][0].copy()
    for name, value in (("prefix_embeddings", embeddings_np), ("prefix_pad_mask", masks_np),
                        ("prefix_attention_mask", attention_np)):
        if array_identity(value)["sha256"] != receipt["source_run"]["captures"][name]["sha256"]:
            raise ValueError(f"Original source tensor identity differs: {name}")
    if embeddings_np.shape != (1, 177, 960) or embeddings_np.dtype != np.float32:
        raise ValueError("Unexpected source prefix layout")
    if expected_k.shape != (1, 177, 5, 64) or expected_k.dtype != np.uint16:
        raise ValueError("Expected original BF16 key-cache words")
    config = modeling_llama.LlamaConfig(**json.loads((args.backbone / "config.json").read_text())["text_config"])
    if config.hidden_size != 960 or config.rms_norm_eps != 1e-5 or config.num_key_value_heads != 5:
        raise ValueError("Unexpected native text-backbone geometry")
    config._attn_implementation = "eager"
    with torch.device("meta"):
        layer = modeling_llama.LlamaDecoderLayer(config, 0)
    prefix = "model.vlm_with_expert.vlm.model.text_model.layers.0."
    names = (prefix + "input_layernorm.weight", prefix + "self_attn.q_proj.weight",
             prefix + "self_attn.k_proj.weight")
    with safe_open(str(args.checkpoint / "model.safetensors"), framework="pt", device="cpu") as handle:
        weights = {name: handle.get_tensor(name) for name in names}
    if any(value.dtype != torch.bfloat16 for value in weights.values()):
        raise ValueError("Native checkpoint parameters must remain BF16")
    layer.input_layernorm.weight = torch.nn.Parameter(weights[names[0]], requires_grad=False)
    layer.self_attn.q_proj.weight = torch.nn.Parameter(weights[names[1]], requires_grad=False)
    layer.self_attn.k_proj.weight = torch.nn.Parameter(weights[names[2]], requires_grad=False)
    layer.eval()
    params_before = {key: array_identity(value.view(torch.uint16).numpy()) for key, value in weights.items()}
    embeddings = torch.from_numpy(embeddings_np)
    before = embeddings.clone()
    positions = torch.cumsum(torch.from_numpy(masks_np), dim=1) - 1
    observed = {}

    class ProjectionCaptured(Exception):
        pass

    def stop_query(_module, inputs, output):
        observed["input"] = inputs[0].detach().clone()
        observed["output"] = output.detach().clone()
        raise ProjectionCaptured()

    hook = layer.self_attn.q_proj.register_forward_hook(stop_query)
    try:
        with torch.inference_mode():
            for repeat in range(2):
                try:
                    smolvlm_with_expert.SmolVLMWithExpertModel.forward_attn_layer(
                        None, [[layer], [None]], [embeddings, None], 0,
                        positions, torch.from_numpy(attention_np), 1, 64)
                except ProjectionCaptured:
                    pass
                else:
                    raise AssertionError("Native LeRobot query projection was not observed")
                current = {key: array_identity(value.view(torch.uint16).numpy()) for key, value in observed.items()}
                if repeat == 0:
                    first = current
                elif current != first:
                    raise AssertionError("Native repeated query capture differs")
            native_k = layer.self_attn.k_proj(observed["input"]).view(1, 177, 5, 64)
            native_k = smolvlm_with_expert.apply_rope(native_k, positions)
    finally:
        hook.remove()
    actual_k = native_k.view(torch.uint16).numpy()
    if not np.array_equal(actual_k, expected_k):
        raise AssertionError(f"Reconstructed source K-cache differs on {np.count_nonzero(actual_k != expected_k)} words")
    if not torch.equal(embeddings, before):
        raise AssertionError("Capture modified archived source inputs")
    if params_before != {key: array_identity(value.view(torch.uint16).numpy()) for key, value in weights.items()}:
        raise AssertionError("Capture modified checkpoint parameters")
    a, b = observed["input"][0, :32].contiguous(), weights[names[1]][:32].T.contiguous()
    c = observed["output"][0, :32, :32].contiguous()
    arrays = {}
    for key, value in (("A", a), ("B", b), ("native_C", c)):
        arrays[key] = value.float().numpy().copy()
        arrays[key + "_bf16"] = value.view(torch.uint16).numpy().copy()
        if not np.array_equal(arrays[key].view(np.uint32), bf16_words_to_float(arrays[key + "_bf16"]).view(np.uint32)):
            raise AssertionError("Raw BF16 word authentication failed")
        if not np.isfinite(arrays[key]).all():
            raise AssertionError("Nonfinite native source contraction")
    ideal = arrays["A"].astype(np.float64) @ arrays["B"].astype(np.float64)
    wrong_b = arrays["B"].copy()
    wrong_b[[0, 1]] = wrong_b[[1, 0]]
    controls = {
        "wrong_k_order_detected": bool(np.any(round_bf16_words(arrays["A"].astype(np.float64) @ wrong_b.astype(np.float64)) != arrays["native_C_bf16"])),
        "wrong_output_channels_detected": not torch.equal(c, observed["output"][0, :32, 32:64]),
        "wrong_cached_key_word_detected": not np.array_equal(actual_k ^ np.uint16(1), expected_k),
    }
    if not all(controls.values()):
        raise AssertionError(f"Negative control failed: {controls}")
    verified_sources = verify_declared_sources(source)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if peak > 2 * 1024**3:
        raise MemoryError("Selective capture exceeded 2 GiB RSS lease")
    out = artifact_output(args.out if args.out else artifacts_dir() / "compare" / "atlas-sota" / "smolvla-native-s1")
    out.mkdir(parents=True, exist_ok=False)
    np.savez(out / "operands.npz", **arrays)
    source["observation"] = {key: observation[key] for key in (
        "dataset_id", "dataset_revision", "episode_index", "frame_index", "split", "task", "noise_seed",
        "camera_mapping", "training_camera_mapping_verified", "video_upstream_hashes_verified",
        "source_closure_verified", "backbone_revision_verified")}
    source["parameters"] = params_before
    report = {
        "schema_version": 1, "status": "native_selective_capture_passed",
        "npz_sha256": digest(out / "operands.npz"), "source_dtype": "bfloat16", "source": source,
        "site": {"module": prefix.rstrip(".") + ".self_attn.q_proj", "batch": 0,
                 "positions": [0, 32], "output_channels": [0, 32], "k_range": [0, 960],
                 "position_meaning": "first_camera_visual_tokens", "padding": False,
                 "native_execution_shape": [1, 177, 960], "full_k_reduction": True},
        "capture": {"route": "retained_original_prefix_then_native_LeRobot_forward_attn_layer_to_query_hook",
                    "native_repeats_raw_exact": True, "caller_inputs_preserved": True,
                    "parameters_preserved": True, "source_artifacts_preserved": True,
                    "declared_source_bookends_verified": verified_sources,
                    "layer_zero_key_cache_raw_exact": True,
                    "layer_zero_key_cache_words": expected_k.size,
                    "full_policy_executed": False, "vision_prefix_recomputed": False},
        "arrays": {key: array_identity(value) for key, value in arrays.items()},
        "independent_dot": {"kind": "numpy_FP64_from_original_BF16_operands_diagnostic",
                            "rounded_bf16_mismatches": int(np.count_nonzero(round_bf16_words(ideal) != arrays["native_C_bf16"])),
                            "max_abs_native_vs_ideal": float(np.abs(ideal - arrays["native_C"]).max()),
                            "native_reference_replaced": False},
        "negative_controls": controls, "wall_seconds": time.monotonic() - started,
        "peak_rss_bytes": peak, "cpu_threads": 1,
        "claim_limits": ["source_archive_reuse_not_fresh_vision_run", "no_policy_quality_claim",
                         "no_target_execution", "original_native_policy_not_W8A8_candidate"],
        "invocation": {"argv": sys.argv, "cwd": str(Path.cwd())},
    }
    (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "out": str(out), "npz_sha256": report["npz_sha256"],
                      "independent_dot": report["independent_dot"], "negative_controls": controls,
                      "key_cache_raw_exact_words": expected_k.size, "wall_seconds": report["wall_seconds"],
                      "peak_rss_bytes": peak}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=safe_path, required=True)
    parser.add_argument("--backbone", type=safe_path, required=True)
    parser.add_argument("--receipt", type=safe_path, required=True)
    parser.add_argument("--archive", type=safe_path, required=True)
    parser.add_argument("--observation-manifest", type=safe_path, required=True)
    parser.add_argument("--out", type=artifact_output)
    capture(parser.parse_args())


if __name__ == "__main__":
    main()
