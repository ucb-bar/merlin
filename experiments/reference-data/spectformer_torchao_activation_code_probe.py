"""Inspect exact TorchAO activation codes at SpectFormer block-0 fc1."""

import argparse
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--trained-tap", required=True, type=Path)
parser.add_argument("--holdout-tap", required=True, type=Path)
parser.add_argument("--model2mlir", required=True, type=Path)
parser.add_argument("--spectformer-dir", required=True, type=Path)
parser.add_argument("--checkpoint", required=True, type=Path)
args = parser.parse_args()
sys.path[:0] = [str(args.model2mlir), str(args.model2mlir / "workloads/spectformer")]
os.environ["SPECTFORMER_DIR"] = str(args.spectformer_dir)
os.environ["SPECTFORMER_CKPT"] = str(args.checkpoint)
get_model_and_inputs = importlib.import_module("loader").get_model_and_inputs
torchao_pipeline = importlib.import_module("m2m.capture.torchao_pipeline")
QuantizationConfig = torchao_pipeline.QuantizationConfig
apply_quantization = torchao_pipeline.apply_quantization

torch.set_num_threads(1)
torch.manual_seed(0)
np.random.seed(0)
model, inputs = get_model_and_inputs()
model.eval()
model = apply_quantization(model, QuantizationConfig(scheme="int8_dyn_act_int8_weight"), example_inputs=tuple(inputs))
block = model.blocks[0]
quant_module = importlib.import_module("torchao.quantization.quantize_.workflows.int8.int8_tensor")
original_quant = quant_module._choose_quant_func_and_quantize_tensor
current_quant = []


def capture_quant(*args, **kwargs):
    result = original_quant(*args, **kwargs)
    current_quant.append((result.qdata.detach().cpu().numpy().copy(), result.scale.detach().cpu().numpy().copy()))
    return result


quant_module._choose_quant_func_and_quantize_tensor = capture_quant


class IntMM(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.calls = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func == torch.ops.aten._int_mm.default:
            self.calls.append((args[0].detach().cpu().numpy().copy(), result.detach().cpu().numpy().copy()))
        return result


def sha(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def run_fc1(tensor):
    current_quant.clear()
    observe = IntMM()
    with torch.no_grad(), observe:
        output = block.mlp.fc1(torch.from_numpy(tensor)).detach().cpu().numpy().copy()
    if len(current_quant) != 1 or len(observe.calls) != 1:
        raise AssertionError((len(current_quant), len(observe.calls)))
    codes, scale = current_quant[0]
    lhs, intmm = observe.calls[0]
    if not np.array_equal(codes.reshape(lhs.shape), lhs):
        raise AssertionError("quantized activation code differs from _int_mm LHS")
    return output, codes.reshape(lhs.shape), scale.reshape(-1, 1), intmm


records = []
trained = np.load(args.trained_tap)
holdout = np.load(args.holdout_tap)
for name, saved in [("trained", trained), ("holdout", holdout)]:
    host_filter = saved["view_2"]
    host_norm2 = saved["layer_norm_1"]
    host_fc1 = saved["add_2"]
    host_codes = saved["dtype_cast_1"].reshape(196, 256)
    host_invscale = saved["mul_0"].reshape(196, 1)
    with torch.no_grad():
        torch_norm2 = block.norm2(torch.from_numpy(host_filter)).detach().cpu().numpy().copy()
    same_out, same_codes, same_scale, same_intmm = run_fc1(host_norm2)
    drift_out, drift_codes, drift_scale, drift_intmm = run_fc1(torch_norm2)
    if not np.array_equal(same_out, host_fc1):
        raise AssertionError(name + " same-input fc1 is not bitwise")
    if not np.array_equal(same_codes, host_codes):
        raise AssertionError(name + " host code differs from Torch same-input code")
    if not np.array_equal(same_intmm, saved["matmul_0"]):
        raise AssertionError(name + " host intmm differs from Torch same-input intmm")
    torch_invscale = np.reciprocal(same_scale.astype(np.float32))
    codes_delta = drift_codes.astype(np.int16) - same_codes.astype(np.int16)
    int_delta = drift_intmm.astype(np.int64) - same_intmm.astype(np.int64)
    w_scale = block.mlp.fc1.weight.scale.detach().cpu().numpy().reshape(1, -1)
    code_effect = int_delta.astype(np.float64) * same_scale.astype(np.float64) * w_scale.astype(np.float64)
    scale_effect = (
        same_intmm.astype(np.float64)
        * (drift_scale.astype(np.float64) - same_scale.astype(np.float64))
        * w_scale.astype(np.float64)
    )
    actual_delta = drift_out.astype(np.float64) - same_out.astype(np.float64)
    cross_effect = (
        int_delta.astype(np.float64)
        * (drift_scale.astype(np.float64) - same_scale.astype(np.float64))
        * w_scale.astype(np.float64)
    )
    changed_locations = np.argwhere(codes_delta != 0)
    code_transitions = {}
    for i, j in changed_locations:
        key = f"{int(same_codes[i, j])}->{int(drift_codes[i, j])}"
        code_transitions[key] = code_transitions.get(key, 0) + 1
    flat_host = host_norm2.reshape(196, 256)
    flat_torch = torch_norm2.reshape(196, 256)
    host_maxabs = np.max(np.abs(flat_host), axis=1)
    torch_maxabs = np.max(np.abs(flat_torch), axis=1)
    at_host_extreme = sum(bool(abs(flat_host[i, j]) == host_maxabs[i]) for i, j in changed_locations)
    at_torch_extreme = sum(bool(abs(flat_torch[i, j]) == torch_maxabs[i]) for i, j in changed_locations)
    host_scaled = [float(flat_host[i, j]) / float(same_scale[i, 0]) for i, j in changed_locations]
    torch_scaled = [float(flat_torch[i, j]) / float(drift_scale[i, 0]) for i, j in changed_locations]
    examples = [
        {
            "token": int(i),
            "channel": int(j),
            "host_code": int(same_codes[i, j]),
            "torch_code": int(drift_codes[i, j]),
            "host_float": float(host_norm2.reshape(196, 256)[i, j]),
            "torch_float": float(torch_norm2.reshape(196, 256)[i, j]),
            "host_torch_scale": float(same_scale[i, 0]),
            "drift_torch_scale": float(drift_scale[i, 0]),
        }
        for i, j in changed_locations[:5]
    ]
    record = {
        "image": name,
        "host_norm2_sha256": sha(host_norm2),
        "torch_norm2_sha256": sha(torch_norm2),
        "host_codes_sha256": sha(host_codes),
        "same_codes_sha256": sha(same_codes),
        "drift_codes_sha256": sha(drift_codes),
        "host_inverse_scale_sha256": sha(host_invscale),
        "same_torch_scale_sha256": sha(same_scale),
        "drift_torch_scale_sha256": sha(drift_scale),
        "same_host_code_bitwise": True,
        "same_host_inverse_scale_bitwise": bool(np.array_equal(host_invscale, torch_invscale)),
        "max_abs_same_invscale_delta": float(np.max(np.abs(host_invscale - torch_invscale))),
        "same_host_intmm_bitwise": True,
        "same_host_fc1_bitwise": True,
        "changed_activation_codes": int(np.count_nonzero(codes_delta)),
        "total_activation_codes": int(codes_delta.size),
        "changed_token_rows": int(np.count_nonzero(np.any(codes_delta != 0, axis=1))),
        "max_abs_code_delta": int(np.max(np.abs(codes_delta))),
        "changed_int32_outputs": int(np.count_nonzero(int_delta)),
        "total_int32_outputs": int(int_delta.size),
        "max_abs_int32_delta": int(np.max(np.abs(int_delta))),
        "changed_scale_rows": int(np.count_nonzero(same_scale.view(np.uint32) != drift_scale.view(np.uint32))),
        "max_abs_scale_delta": float(np.max(np.abs(drift_scale - same_scale))),
        "max_rel_scale_delta": float(
            np.max(
                np.abs((drift_scale.astype(np.float64) - same_scale.astype(np.float64)) / same_scale.astype(np.float64))
            )
        ),
        "fc1_rel_l2": float(np.linalg.norm(actual_delta) / np.linalg.norm(same_out.astype(np.float64))),
        "fc1_max_abs": float(np.max(np.abs(actual_delta))),
        "code_effect_l2": float(np.linalg.norm(code_effect)),
        "scale_effect_l2": float(np.linalg.norm(scale_effect)),
        "actual_delta_l2": float(np.linalg.norm(actual_delta)),
        "code_effect_fraction_l2": float(np.linalg.norm(code_effect) / np.linalg.norm(actual_delta)),
        "scale_effect_fraction_l2": float(np.linalg.norm(scale_effect) / np.linalg.norm(actual_delta)),
        "residual_after_code_only_fraction_l2": float(
            np.linalg.norm(actual_delta - code_effect) / np.linalg.norm(actual_delta)
        ),
        "residual_after_code_scale_cross_fraction_l2": float(
            np.linalg.norm(actual_delta - code_effect - scale_effect - cross_effect) / np.linalg.norm(actual_delta)
        ),
        "changed_code_examples": examples,
        "code_transitions": code_transitions,
        "changed_at_host_row_extreme": int(at_host_extreme),
        "changed_at_torch_row_extreme": int(at_torch_extreme),
        "max_abs_host_distance_from_negative_127_5": max((abs(value + 127.5) for value in host_scaled), default=0.0),
        "max_abs_torch_distance_from_negative_127_5": max((abs(value + 127.5) for value in torch_scaled), default=0.0),
    }
    records.append(record)
print(
    json.dumps(
        {"schema": "spectformer.activation_codes.v1", "torch": torch.__version__, "rows": records},
        indent=2,
        sort_keys=True,
    ),
    flush=True,
)
