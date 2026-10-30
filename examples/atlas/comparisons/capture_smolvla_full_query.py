"""Capture original SmolVLA Q 177x960; pad exported matrix rows to 192 only.

Native prefix/masks remain length 177 (141 attention-valid positions). This
reuses retained TRAIN prefix provenance; it does not recompute vision or policy.
"""

import argparse
import copy
import importlib.metadata
import os
from pathlib import Path
import sys

import numpy as np

import source_relocation
from source_relocation import argument_relocator, read_metadata

from capture_smolvla import (CHECKPOINT_REVISION, WEIGHTS_SHA, RECEIPT_SHA,
                            ARCHIVE_SHA, OBSERVATION_SHA, EXPERT_SOURCE_SHA, LLAMA_SHA)
from capture_tinyllama_native import (array_identity, bf16_words_to_float, digest,
                                     identity, round_bf16_words, safe_path, verify_declared_sources)


def validate_padding(arrays, native_q, native_a):
    required = {"A", "B", "native_C", "A_bf16", "B_bf16", "native_C_bf16"}
    if set(arrays) != required:
        raise ValueError("Unexpected full-query operand fields")
    for name, shape in (("A", (192, 960)), ("B", (960, 960)), ("native_C", (192, 960))):
        value, words = arrays[name], arrays[name + "_bf16"]
        if value.shape != shape or words.shape != shape or value.dtype != np.float32 or words.dtype != np.uint16:
            raise ValueError("Wrong padded full-query shape/dtype: " + name)
        if not np.isfinite(value).all() or not np.array_equal(value.view(np.uint32), bf16_words_to_float(words).view(np.uint32)):
            raise ValueError("BF16 words do not authenticate exact finite operands: " + name)
    if (not np.array_equal(arrays["A_bf16"][:177], native_a)
            or not np.array_equal(arrays["native_C_bf16"][:177], native_q)
            or np.any(arrays["A_bf16"][177:]) or np.any(arrays["native_C_bf16"][177:])):
        raise ValueError("Native rows changed or synthetic padding rows are not positive zero")


def capture(args, report):
    import torch
    from safetensors import safe_open
    from transformers.models.llama import modeling_llama
    from lerobot.policies.smolvla import smolvlm_with_expert
    tile_id = identity(args.native_tile / "manifest.json", args.native_tile_manifest_sha256)
    tile = read_metadata(args, report, tile_id["path"], tile_id["sha256"])
    if (tile["status"] != "native_selective_capture_passed" or tile["source_dtype"] != "bfloat16"
            or tile["source"]["revision"] != CHECKPOINT_REVISION):
        raise ValueError("Require the retained original native S1 source")
    tile_npz_id = identity(args.native_tile / "operands.npz", tile["npz_sha256"])
    source = copy.deepcopy(tile["source"])
    source["historical_native_tile"] = tile_id
    source["historical_native_tile_operands"] = tile_npz_id
    source["capture_script"] = identity(__file__)
    source["relocation_helper"] = identity(source_relocation.__file__)
    source["capture_helpers"] = identity(Path(__file__).with_name("capture_tinyllama_native.py"))
    source["original_capture_edge"] = identity(Path(__file__).with_name("capture_smolvla.py"))
    versions = {key: importlib.metadata.version(key) for key in ("torch", "transformers", "lerobot", "numpy")}
    if versions != tile["source"]["versions"] or sys.version != tile["source"]["python"]:
        raise ValueError("Imported native environment differs from the retained S1 source")
    identity(smolvlm_with_expert.__file__, EXPERT_SOURCE_SHA)
    identity(modeling_llama.__file__, LLAMA_SHA)
    identity(Path(torch.__file__).parent / "lib/libtorch_cpu.so", source["torch_cpu_library"]["sha256"])
    identity(source["weights"]["path"], WEIGHTS_SHA)
    identity(source["historical_source_receipt"]["path"], RECEIPT_SHA)
    identity(source["historical_original_archive"]["path"], ARCHIVE_SHA)
    identity(source["observation_receipt"]["path"], OBSERVATION_SHA)
    verify_declared_sources(source)
    receipt = read_metadata(args, report, source["historical_source_receipt"]["path"], RECEIPT_SHA)
    if (receipt["fixture_kind"] != "dataset" or not receipt["source_reference_unchanged"]
            or receipt["source_run"]["output_file"]["sha256"] != ARCHIVE_SHA):
        raise ValueError("Archive must be the original policy source, not W8A8 candidate output")
    observation = source["observation"]
    if (observation["split"] != "train" or observation["episode_index"] != 0 or observation["frame_index"] != 0
            or observation["noise_seed"] != 0 or observation["dataset_revision"] != "3d6d687a25cdf1565cdf24550814f72d999a861d"):
        raise ValueError("Expected retained TRAIN episode 0/frame 0/noise 0")
    with np.load(source["historical_original_archive"]["path"], allow_pickle=False) as archive:
        embeddings_np = archive["prefix_embeddings"].copy()
        masks_np = archive["prefix_pad_mask"].copy()
        attention_np = archive["prefix_attention_mask"].copy()
        expected_k = archive["prefix_key_cache"][0].copy()
    for name, value in (("prefix_embeddings", embeddings_np), ("prefix_pad_mask", masks_np), ("prefix_attention_mask",
        attention_np)):
        if array_identity(value)["sha256"] != receipt["source_run"]["captures"][name]["sha256"]:
            raise ValueError("Original prefix identity differs: " + name)
    if (embeddings_np.shape != (1, 177, 960) or embeddings_np.dtype != np.float32
            or masks_np.shape != (1, 177) or masks_np.dtype != np.bool_ or int(masks_np.sum()) != 141
            or attention_np.shape != (1, 177) or expected_k.shape != (1, 177, 5, 64) or expected_k.dtype != np.uint16):
        raise ValueError("Original native prefix/cache geometry differs")
    config_data = read_metadata(args, report, source["backbone_config"]["path"], source["backbone_config"]["sha256"])[
        "text_config"]
    config = modeling_llama.LlamaConfig(**config_data)
    if (config.hidden_size != 960 or config.num_attention_heads != 15 or config.head_dim != 64
            or config.num_key_value_heads != 5 or config.rms_norm_eps != 1e-5):
        raise ValueError("Unexpected native query width or backbone geometry")
    config._attn_implementation = "eager"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with torch.device("meta"):
        layer = modeling_llama.LlamaDecoderLayer(config, 0)
    prefix = "model.vlm_with_expert.vlm.model.text_model.layers.0."
    names = (prefix + "input_layernorm.weight", prefix + "self_attn.q_proj.weight", prefix + "self_attn.k_proj.weight")
    with safe_open(source["weights"]["path"], framework="pt", device="cpu") as handle:
        weights = {name: handle.get_tensor(name) for name in names}
    for name, shape in zip(names, ((960,), (960, 960), (320, 960))):
        if weights[name].dtype != torch.bfloat16 or tuple(weights[name].shape) != shape:
            raise ValueError("Original BF16 parameter geometry changed")
        if array_identity(weights[name].view(torch.uint16).numpy()) != source["parameters"][name]:
            raise ValueError("Native parameter bytes differ from retained S1")
    layer.input_layernorm.weight = torch.nn.Parameter(weights[names[0]], requires_grad=False)
    layer.self_attn.q_proj.weight = torch.nn.Parameter(weights[names[1]], requires_grad=False)
    layer.self_attn.k_proj.weight = torch.nn.Parameter(weights[names[2]], requires_grad=False)
    layer.eval()
    params_before = {key: array_identity(value.view(torch.uint16).numpy()) for key, value in weights.items()}
    embeddings = torch.from_numpy(embeddings_np.copy())
    masks = torch.from_numpy(masks_np.copy())
    attention = torch.from_numpy(attention_np.copy())
    input_before = {"embeddings": array_identity(embeddings_np), "mask": array_identity(masks_np), "attention": array_identity(
        attention_np)}
    positions = torch.cumsum(masks, dim=1) - 1
    captured = []

    class QueryCaptured(Exception):
        pass

    def stop_query(_, inputs, output):
        if (tuple(inputs[0].shape) != (1, 177, 960) or tuple(output.shape) != (1, 177, 960)
                or inputs[0].dtype != torch.bfloat16 or output.dtype != torch.bfloat16):
            raise ValueError("Native query boundary changed or included synthetic rows")
        captured.append((inputs[0].detach().clone(), output.detach().clone()))
        raise QueryCaptured()

    hook = layer.self_attn.q_proj.register_forward_hook(stop_query)
    try:
        with torch.inference_mode():
            for repeat in range(2):
                count = len(captured)
                try:
                    smolvlm_with_expert.SmolVLMWithExpertModel.forward_attn_layer(
                        None, [[layer], [None]], [embeddings, None], 0, positions, attention, 1, 64)
                except QueryCaptured:
                    pass
                else:
                    raise AssertionError("Native query hook did not stop the original pipeline")
                if len(captured) != count + 1:
                    raise AssertionError("Query must be captured exactly once per native call")
            for left, right in zip(captured[0], captured[1]):
                if not torch.equal(left.view(torch.uint16), right.view(torch.uint16)):
                    raise AssertionError("Repeated native query capture changed words")
            native_k = layer.self_attn.k_proj(captured[0][0]).view(1, 177, 5, 64)
            native_k = smolvlm_with_expert.apply_rope(native_k, positions)
    finally:
        hook.remove()
    with torch.inference_mode():
        replay = layer.self_attn.q_proj(captured[0][0])
    if not torch.equal(replay.view(torch.uint16), captured[0][1].view(torch.uint16)):
        raise AssertionError("Native query replay after hook removal changed output words")
    actual_k = native_k.view(torch.uint16).numpy()
    if not np.array_equal(actual_k, expected_k):
        raise AssertionError("Native K/RoPE no longer closes the archived 56640 words")
    a_words = captured[0][0][0].contiguous().view(torch.uint16).numpy().copy()
    q_words = captured[0][1][0].contiguous().view(torch.uint16).numpy().copy()
    b_words = weights[names[1]].t().contiguous().view(torch.uint16).numpy().copy()
    padded_a = np.zeros((192, 960), np.uint16)
    padded_c = np.zeros((192, 960), np.uint16)
    padded_a[:177], padded_c[:177] = a_words, q_words
    arrays = {"A_bf16": padded_a, "B_bf16": b_words, "native_C_bf16": padded_c}
    arrays.update({name: bf16_words_to_float(arrays[name + "_bf16"]) for name in ("A", "B", "native_C")})
    validate_padding(arrays, q_words, a_words)
    with np.load(args.native_tile / "operands.npz", allow_pickle=False) as archive:
        for name, current in (("A_bf16", a_words[:32]), ("B_bf16", b_words[:, :32]), ("native_C_bf16", q_words[:32, :32])):
            if not np.array_equal(current, archive[name]):
                raise AssertionError("Full query does not reproduce retained native S1 tile: " + name)
    controls = {}
    for name, key, row in (("nonzero_synthetic_input", "A_bf16", 177), ("nonzero_synthetic_output", "native_C_bf16", 177),
                           ("changed_native_query_row", "native_C_bf16", 176)):
        changed = {key: value.copy() for key, value in arrays.items()}
        changed[key][row, 0] ^= np.uint16(1)
        changed[key.removesuffix("_bf16")] = bf16_words_to_float(changed[key])
        try:
            validate_padding(changed, q_words, a_words)
        except ValueError:
            controls[name] = True
        else:
            raise AssertionError("Full-query padding control admitted: " + name)
    if (input_before != {"embeddings": array_identity(embeddings.numpy()), "mask": array_identity(masks.numpy()),
        "attention": array_identity(attention.numpy())}
            or params_before != {key: array_identity(value.view(torch.uint16).numpy()) for key, value in weights.items()}):
        raise AssertionError("Native query capture changed caller input or parameters")
    verified = verify_declared_sources(source)
    np.savez(args.out / "operands.npz", **arrays)
    np.savez(args.out / "native-query.npz", Q_bf16=q_words, Q_float32=bf16_words_to_float(q_words),
             prefix_pad_mask=masks_np, prefix_attention_mask=attention_np, position_ids=positions.numpy())
    ideal = bf16_words_to_float(a_words).astype(np.float64) @ bf16_words_to_float(b_words).astype(np.float64)
    report.update(status="native_selective_capture_passed", source_dtype="bfloat16", npz_sha256=digest(args.out / "operands.npz"),
        source=source,
                  site={"module": prefix + "self_attn.q_proj", "batch": 0, "positions": [0, 177], "output_channels": [0, 960],
        "k_range": [0, 960],
                        "native_execution_shape": [1, 177, 960], "full_k_reduction": True, "full_query_projection": True,
                        "matrix_shape": [192, 960, 960], "native_rows": 177, "valid_rows": 177, "attention_valid_positions": 141,
                        "original_masked_positions": 36, "padding": {"kind": "exported_matrix_zero_rows_only", "rows": [177, 192],
                        "native_masks_or_positions_extended": False,
        "downstream_strip": "target_C[:177,:] before any native RoPE/attention/policy use"}},
                  capture={"route": "retained_original_prefix_then_native_LeRobot_RMSNorm_cast_query_hook_before_RoPE",
                           "native_repeats_raw_exact": True, "native_query_replay_after_hook_removal_raw_exact": True,
                           "retained_native_tile_raw_exact": True, "caller_inputs_preserved": True, "parameters_preserved": True,
                           "declared_source_bookends_verified": verified, "layer_zero_key_cache_raw_exact": True,
                           "layer_zero_key_cache_words": int(expected_k.size), "full_policy_executed": False,
        "vision_prefix_recomputed": False},
                  arrays={key: array_identity(value) for key, value in arrays.items()}, native_query=identity(
        args.out / "native-query.npz"),
                  native_query_words=array_identity(q_words), native_input_words=array_identity(a_words),
        negative_controls=controls,
                  independent_dot={"kind": "FP64_from_original_BF16_operands_diagnostic", "native_reference_replaced": False,
                                   "rounded_bf16_mismatches": int(np.count_nonzero(round_bf16_words(ideal) != q_words)),
                                   "max_abs_native_vs_ideal": float(np.max(np.abs(ideal - bf16_words_to_float(q_words))))},
                  claim_limits=tile["claim_limits"] + ["matrix_padding_is_not_native_policy_positions",
        "no_full_source_closure_claim"])


def main():
    from workload_cli import run_report
    from merlin.common.paths import artifacts_dir
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-tile", type=safe_path, required=True)
    parser.add_argument("--native-tile-manifest-sha256", required=True)
    parser.add_argument("--out", type=safe_path, required=True)
    parser.add_argument("--relocate", action="append", default=[], metavar="OLD=NEW")
    args = parser.parse_args()
    if not args.out.is_relative_to(safe_path(artifacts_dir())) or args.out == safe_path(artifacts_dir()):
        parser.error("Output must be a fresh directory beneath the configured artifacts root")
    report = {"schema_version": 1, "status": "failed", "scope": __doc__, "command": sys.argv, "cwd": str(Path.cwd()),
              "python_executable": sys.executable, "source_script": identity(__file__), "cpu_threads": 1,
              "environment": {key: os.environ.get(key) for key in ("PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS")}}
    return run_report(args, capture,
        filename='manifest.json', output=args.out, report=report,
        summary=lambda report: {"status": report["status"], "out": str(args.out), "error": report.get("error")},
        memory_unit='bytes')


if __name__ == "__main__":
    raise SystemExit(main())
