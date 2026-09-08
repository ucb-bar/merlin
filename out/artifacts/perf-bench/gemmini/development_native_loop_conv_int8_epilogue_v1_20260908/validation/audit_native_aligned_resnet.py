#!/usr/bin/env python3
"""Structural 53-convolution census for the declared native-aligned INT8 contract.

This is deliberately not a model correctness run.  It asks the copied compiler whether each
prepared ResNet geometry is representable once its boundary is NHWC/HWIO i8, its bias is an i32
accumulator vector, and requantization is one positive scalar CONFIG_ST scale.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[6]
ARTIFACT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ARTIFACT / "compiler"), str(ROOT / "merlin" / "python")]

from mlir_oot.gemmini_opt import Pipeline


SOURCE = (ROOT / "out/artifacts/perf-bench/gemmini/"
          "universal_narrow_epilogue_recovery_v1_20260908/"
          "resnet50_narrow_epilogue_inventory.json")


def spatial_parameters(g: dict) -> tuple[int, int]:
    matches = []
    for stride in (1, 2, 3, 4):
        for padding in range(g["kh"]):
            ho = (g["hi"] + 2 * padding - g["kh"]) // stride + 1
            wo = (g["wi"] + 2 * padding - g["kw"]) // stride + 1
            if (ho, wo) == (g["ho"], g["wo"]):
                matches.append((stride, padding))
    if not matches:
        raise ValueError(f"cannot derive stride/padding for {g}")
    # Standard ResNet uses the smallest matching stride and symmetric padding.
    return min(matches)


def interface(g: dict, index: int) -> str:
    stride, padding = spatial_parameters(g)
    k = g["kh"] * g["kw"] * g["ci"]
    out_rows = g["n"] * g["ho"] * g["wo"]
    return f'''module attributes {{merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"}} {{
  %IFM = merlin_iface.tensor {{name = "IFM", role = "input"}} : tensor<{g['n']}x{g['hi']}x{g['wi']}x{g['ci']}xi8>
  %W = merlin_iface.tensor {{name = "W", role = "weight"}} : tensor<{k}x{g['co']}xi8>
  %B = merlin_iface.tensor {{name = "B", role = "bias"}} : tensor<{g['co']}xi32>
  %R = merlin_iface.resident_pack %W {{layout = "packed_conv_rhs"}} : (tensor<{k}x{g['co']}xi8>) -> !merlin_iface.resident
  %Y = merlin_iface.conv2d %IFM, %R {{kernel = [{g['kh']}, {g['kw']}, {g['ci']}, {g['co']}], stride = [{stride}, {stride}], padding = [{padding}, {padding}, {padding}, {padding}], dilation = [1, 1], name = "Y", epilogue = ["bias", "relu", "acc_scale"], bias = "B", acc_scale = 0.125 : f32, output_dtype = "i8", layout = "nhwc"}} : (tensor<{g['n']}x{g['hi']}x{g['wi']}x{g['ci']}xi8>, !merlin_iface.resident) -> tensor<{out_rows}x{g['co']}xi8>
  merlin_iface.evict %R : (!merlin_iface.resident) -> ()
}}
'''


def main() -> int:
    source = json.loads(SOURCE.read_text())
    rows = []
    for conv in source["convolutions"]:
        stride, padding = spatial_parameters(conv["geometry"])
        pipe = Pipeline(interface(conv["geometry"], conv["index"])).run()
        selection = pipe.plan.command_buffer["params"]["convolution_lowering"]["selections"][0]
        rows.append({
            "index": conv["index"], "fqn": conv["fqn"], "geometry": conv["geometry"],
            "stride": stride, "padding": padding,
            "selected": selection["selected"], "reason": selection["reason"],
            "loop_conv_descriptors": sum(i.kind == "loop_conv_ws" for i in pipe.instrs),
            "load3_bias_descriptors": sum(
                i.kind == "loop_conv_ws" and not i.attrs["no_bias"] for i in pipe.instrs),
        })
    admitted = sum(row["selected"] == "gemmini_loop_conv_ws" for row in rows)
    receipt = {
        "schema": "native_aligned_resnet50_conv_structural_census_v1",
        "status": "passed" if admitted == len(rows) == 53 else "failed",
        "contract": {
            "activation": "per_tensor_symmetric_i8_nhwc",
            "weight": "per_tensor_symmetric_i8_hwio",
            "bias": "per_output_channel_i32_accumulator_units_via_load3",
            "requant": "one_positive_f32_config_st_scale_per_layer_rne_saturate",
        },
        "admitted_convolutions": admitted,
        "total_convolutions": len(rows),
        "total_loop_conv_descriptors": sum(r["loop_conv_descriptors"] for r in rows),
        "total_load3_bias_descriptors": sum(r["load3_bias_descriptors"] for r in rows),
        "scope": "isolated_structural_admission_not_a_full_graph_compile_or_correctness_run",
        "full_model_blockers": [
            "produce_native_aligned_quantized_golden_and parameters",
            "global_NHWC_activation_layout_and_HWIO_weight_conversion",
            "lower_residual_add_pool_and_dense_graph_operations",
        ],
        "hardware_qualification": "not_run",
        "layers": rows,
    }
    target = ARTIFACT / "validation/native_aligned_resnet50_conv_census.json"
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: receipt[k] for k in (
        "status", "admitted_convolutions", "total_convolutions",
        "total_loop_conv_descriptors", "total_load3_bias_descriptors", "scope")},
        sort_keys=True))
    return 0 if receipt["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
