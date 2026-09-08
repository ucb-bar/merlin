import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]
REPO = ROOT.parents[4]
spec = importlib.util.spec_from_file_location(
    "analyze_narrow_epilogues", ROOT / "analyze_narrow_epilogues.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


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
    assert summary["per_channel_scale"] == 53
    assert summary["residual_add_paths"] == 20
    assert summary["maxpool_paths"] == 1
    assert summary["global_tail_paths"] == 1
    assert all("nonzero_bias_requires_live_d_preload" not in item["refusal_reasons"]
               for item in result["convolutions"])
    assert all(item["bias"]["native_transport"] == "loop_conv_load3_to_accumulator"
               for item in result["convolutions"])
    assert all("per_channel_scale_requires_channel_partitioned_store_configuration"
               in item["refusal_reasons"] for item in result["convolutions"])
