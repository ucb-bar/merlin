import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]
REPO = ROOT.parents[4]
spec = importlib.util.spec_from_file_location(
    "analyze_narrow_epilogues", ROOT / "analyze_narrow_epilogues.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

proof_spec = importlib.util.spec_from_file_location(
    "prove_conv1_native_requant", ROOT / "prove_conv1_native_requant.py")
proof = importlib.util.module_from_spec(proof_spec)
assert proof_spec.loader is not None
proof_spec.loader.exec_module(proof)


def test_exact_resnet_currently_needs_channel_partitioned_narrow_store() -> None:
    bundle = REPO / "out/artifacts/perf-bench/gemmini/resnet50_qdq_v2_quantfix_exact_smallread_firesim_bundle_20260908"
    result = module.audit(
        bundle / "provenance/model2mlir_direct.mlir",
        bundle / "provenance/weights.safetensors.manifest.json",
        REPO / "out/artifacts/recaptures/resnet50_pt2e_w8a8_universal_dog_direct_20260906/weights.safetensors")
    summary = result["summary"]
    assert summary["prepared_integer_convolutions"] == 53
    assert summary["convolutions_audited"] == 53
    assert summary["native_exact_admitted"] == 0
    assert summary["native_exact_refused"] == 53
    assert summary["all_channels_nonzero_bias"] == 53
    assert summary["all_bias_channels_exact_integer_accumulator_offset"] == 0
    assert summary["per_channel_scale"] == 53
    assert summary["output_channels_total"] == 26560
    assert summary["minimum_channel_partitioned_loop_conv_launches"] == 26560
    assert summary["residual_add_paths"] == 20
    assert summary["maxpool_paths"] == 1
    assert summary["global_tail_paths"] == 1
    assert all("nonzero_bias_requires_live_d_preload" not in item["refusal_reasons"]
               for item in result["convolutions"])
    assert all(item["bias"]["native_transport"] == "loop_conv_load3_to_accumulator"
               for item in result["convolutions"])
    assert all("float_bias_not_exact_integer_accumulator_offset"
               in item["refusal_reasons"] for item in result["convolutions"])
    assert all("per_channel_scale_requires_channel_partitioned_store_configuration"
               in item["refusal_reasons"] for item in result["convolutions"])


def test_conv1_proves_current_and_per_tensor_pt2e_are_not_native_exact() -> None:
    bundle = REPO / "out/artifacts/perf-bench/gemmini/resnet50_qdq_v2_quantfix_exact_smallread_firesim_bundle_20260908"
    result = proof.compare(
        bundle / "payload/input_quantized_i8.bin",
        REPO / "out/artifacts/recaptures/resnet50_pt2e_w8a8_universal_dog_direct_20260906/weights.safetensors")
    current = result["captured_per_channel"]
    assert current["pre_pool"] == {
        "elements": 802816, "mismatches": 143, "channels_with_mismatch": 41,
        "max_abs_difference": 1}
    assert current["post_pool"] == {
        "elements": 200704, "mismatches": 33, "channels_with_mismatch": 17,
        "max_abs_difference": 1}
    assert current["nearby_integer_bias_and_scale_search"]["exact_channels_count"] == 25
    assert current["nearby_integer_bias_and_scale_search"]["best_total_mismatches"] == 114
    assert current["pre_pool_narrow_only_repair_ambiguity"]["channels_count"] == 41
    assert current["post_pool_narrow_only_repair_ambiguity"]["channels_count"] == 17
    per_tensor = result["per_tensor_weight_ablation"]
    assert per_tensor["exact_integer_bias_channels"] == 0
    assert per_tensor["pre_pool"]["mismatches"] == 364
    assert per_tensor["post_pool"]["mismatches"] == 110
