"""Independent normalized-attention expectation from graph buffers and model primitives.

This adapter consumes no generated program, compiler evaluator, kernel library,
framework softmax, or emitted result. Ordinary source-node device equivalence
remains pending: the explicit selected arithmetic below is its comparison scope.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

MODEL_PIN = "47b8599d8ffbe5ddd96059ffe2b5be2427a410f5"
POLICY = "bf16-masked-normalized-softmax-tree32-fp32-rne-bf16-rne-tile-ltr-chop-rcplut-v1"


def _raw_bf16(value):
    value = np.asarray(value)
    if value.dtype != np.uint16:
        raise ValueError("attention boundary requires unchanged raw BF16 uint16 buffers")
    return value


def _torch_bf16(bits):
    import torch

    return torch.from_numpy(np.ascontiguousarray(_raw_bf16(bits)).copy()).view(torch.bfloat16)


def _bits(value):
    import torch

    return value.contiguous().view(torch.uint16).numpy().copy()


def authenticate_model(model_root):
    """Authenticate the exact arithmetic implementation and both exhaustive LUTs."""
    root = Path(model_root).resolve()
    names = ("npu_model/configs/numerics.py", "npu_model/configs/rtl_math.py",
             "npu_model/configs/data/exp.bin.gz", "npu_model/configs/data/rcp.bin.gz")
    hashes = {}
    for name in names:
        live = (root / name).read_bytes()
        pinned = subprocess.check_output(["git", "-C", str(root), "cat-file", "blob", f"{MODEL_PIN}:{name}"])
        if live != pinned:
            raise ValueError(f"instruction-model arithmetic source differs from pin: {name}")
        hashes[name] = sha256(live).hexdigest()
    return hashes


def normalized_softmax(scores_bf16, keep):
    """Use model primitives with adjacent 32-lane trees and one global divisor.

    Every masked or padded score uses -32 before maxima and subtraction, then
    its exponential is multiplied by +0. Denominators combine BF16 tile sums
    left to right with FP32 RNE followed by BF16 chop. No exponential packing or
    packed-sum correction replaces normalized probabilities.
    """
    from npu_model.configs.numerics import RtlNumerics as rtl

    scores = _raw_bf16(scores_bf16)
    keep = np.asarray(keep)
    if keep.dtype != np.bool_ or keep.shape != scores.shape or scores.ndim < 2 or not 0 < scores.shape[-1] <= 65536:
        raise ValueError("softmax requires a same-shape Boolean keep mask and 1..65536 keys")
    if not np.all(np.any(keep, axis=-1)):
        raise ValueError("selected causal policy requires every row to retain at least one key")
    magnitude = scores & 0x7fff
    exponent = (scores >> 7) & 255
    if np.any(keep & ((magnitude > 0x4200) | ((magnitude != 0) & ((exponent < 8) | (exponent > 132))))):
        raise ValueError("kept score leaves the selected normal/signed-zero [-32,32] domain")
    tiles, masks, maxima = [], [], []
    for start in range(0, scores.shape[-1], 32):
        count = min(32, scores.shape[-1] - start)
        tile = np.full((*scores.shape[:-1], 32), 0xc200, dtype=np.uint16)
        valid = np.zeros(tile.shape, dtype=np.bool_)
        tile[..., :count] = np.where(keep[..., start:start + count], scores[..., start:start + count], 0xc200)
        valid[..., :count] = keep[..., start:start + count]
        tensor = _torch_bf16(tile)
        tiles.append(tensor)
        masks.append(_torch_bf16(np.where(valid, 0x3f80, 0).astype(np.uint16)))
        maxima.append(rtl.amax(tensor, -1))
    maximum = maxima[0]
    for partial in maxima[1:]:
        maximum = rtl.maximum(maximum, partial)
    shifted, exponentials, sums = [], [], []
    for tile, mask in zip(tiles, masks, strict=True):
        offset = rtl.sub(tile, maximum)
        exponential = rtl.mul(rtl.unary("exp", offset), mask)
        shifted.append(offset)
        exponentials.append(exponential)
        sums.append(rtl.row_sum(exponential))
    denominator = sums[0]
    for partial in sums[1:]:
        denominator = rtl.add(denominator, partial)
    reciprocal = rtl.unary("rcp", denominator)
    probabilities = [rtl.mul(exponential, reciprocal) for exponential in exponentials]
    trim = lambda values: np.concatenate([_bits(value) for value in values], axis=-1)[..., :scores.shape[-1]]
    return {
        "scores_bf16": trim(tiles), "keep": keep.copy(), "row_max_bf16": _bits(maximum),
        "shifted_bf16": trim(shifted), "exp_bf16": trim(exponentials),
        "tile_sums_bf16": np.concatenate([_bits(value) for value in sums], axis=-1),
        "denominator_bf16": _bits(denominator), "reciprocal_bf16": _bits(reciprocal),
        "probabilities_bf16": trim(probabilities),
    }


def reference_sources():
    import atlas_ref.activation as activation
    import atlas_ref.arithmetic as arithmetic
    import atlas_ref.contractions as contractions

    return {Path(module.__file__).name: sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in (activation, arithmetic, contractions)}


def attention(q_bf16, k_bf16, mask_bf16, v_bf16):
    """Compose graph per-head quantizers, MXU1 QK, normalized P, and MXU1 PV.

    This source template explicitly uses score scaling by 0.25, BF16 causal
    mask {+0,0xff7f}, static probability exponent -8, and unfolded PV. The source
    boundary and policy identities must be bound by the caller before claiming
    agreement for an exported graph.
    """
    from atlas_ref.activation import quantize_act
    from atlas_ref.arithmetic import bf16_bits, bf16_values
    from atlas_ref.contractions import contraction_nt

    q, k, mask, v = map(_raw_bf16, (q_bf16, k_bf16, mask_bf16, v_bf16))
    if q.ndim < 2 or k.ndim < 2 or v.ndim < 2 or q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]:
        raise ValueError("attention Q/K/V contraction dimensions differ")
    if np.any((mask != 0) & (mask != 0xff7f)):
        raise ValueError("causal mask must preserve the source +0 or BF16 minfinite sentinel")
    q_codes, q_exp = quantize_act(q, "dynamic", "head", "ref", None)
    k_codes, k_exp = quantize_act(k, "dynamic", "head", "ref", None)
    v_codes, v_exp = quantize_act(v, "dynamic", "head", "ref", None)
    qk = contraction_nt(q_codes, q_exp, k_codes, k_exp, unit="mxu1")
    scaled = np.multiply(bf16_values(qk.output_bf16), np.float32(0.25), dtype=np.float32)
    # Scaling by 2^-2 is exact in this admitted normal domain. Keep the f32
    # boundary bits and the explicit BF16 policy narrowing separately.
    if np.any(~np.isfinite(scaled)) or np.any(np.abs(scaled) > 32):
        raise ValueError("scaled source scores leave the bounded mask-sentinel mapping domain")
    source_scores = np.add(scaled, bf16_values(mask), dtype=np.float32)
    scores = bf16_bits(source_scores)
    keep = np.broadcast_to(mask == 0, scores.shape)
    softmax = normalized_softmax(scores, keep)
    p_codes, p_exp = quantize_act(softmax["probabilities_bf16"], "static", "tensor", "ref", -8)
    vt_codes = np.swapaxes(v_codes, -1, -2).copy()
    pv = contraction_nt(p_codes, p_exp, vt_codes, v_exp, unit="mxu1")
    return {
        "q_codes": q_codes, "q_exp": q_exp, "k_codes": k_codes, "k_exp": k_exp,
        "v_codes": v_codes, "v_exp": v_exp, "vt_codes": vt_codes,
        "qk_accumulator_bf16": qk.accumulator_bf16, "qk_readout_bf16": qk.readout_bf16,
        "qk_output_bf16": qk.output_bf16, "scaled_scores_f32_bits": scaled.view(np.uint32).copy(),
        "source_scores_f32_bits": source_scores.view(np.uint32).copy(), "source_scores_bf16": scores,
        "source_mask_bf16": mask.copy(), "keep_bf16": np.where(keep, 0x3f80, 0).astype(np.uint16),
        **softmax, "p_codes": p_codes, "p_exp": p_exp,
        "pv_accumulator_bf16": pv.accumulator_bf16, "pv_readout_bf16": pv.readout_bf16,
        "output_bf16": pv.output_bf16,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True, help="NPZ raw BF16 input_0=q,input_1=k,input_2=mask,input_3=v")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validation", type=Path, help="passing separate handmade instruction validation receipt")
    args = parser.parse_args()
    before = authenticate_model(args.model_root)
    sys.path.insert(0, str(args.model_root.resolve()))
    sys.path.insert(0, str(args.reference_root.resolve()))
    import torch
    import npu_model.configs.numerics as numerics
    if Path(numerics.__file__).resolve() != args.model_root.resolve() / "npu_model/configs/numerics.py":
        raise ValueError("loaded instruction-model package differs from selected source")
    torch.set_num_threads(1)
    validation = None
    if args.validation:
        validation = json.loads(args.validation.read_text())
        if (validation.get("status") != "PASS" or validation.get("policy") != POLICY
                or validation.get("adapter_sha256") != sha256(Path(__file__).read_bytes()).hexdigest()
                or validation.get("model_arithmetic_sources") != before
                or validation.get("reference_sources") != reference_sources()
                or validation.get("mismatches") != 0):
            raise ValueError("handmade validation differs from this adapter, reference, policy, or pinned model")
    with np.load(args.inputs, allow_pickle=False) as archive:
        data = {f"input_{index}": archive[f"input_{index}"] for index in range(4)}
    result = attention(*(data[f"input_{index}"] for index in range(4)))
    args.out.mkdir(parents=True, exist_ok=False)
    np.savez(args.out / "expected.npz", **result)
    if authenticate_model(args.model_root) != before:
        raise ValueError("instruction-model arithmetic changed while computing expectations")
    record = {
        "schema": "atlas.independent_normalized_attention.v1", "status": "EXPECTED",
        "policy": POLICY, "model_pin": MODEL_PIN, "model_arithmetic_sources": before,
        "reference_sources": reference_sources(),
        "adapter_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_archive_sha256": sha256(args.inputs.read_bytes()).hexdigest(),
        "inputs": {name: {"dtype": value.dtype.str, "shape": list(value.shape),
                          "sha256": sha256(value.tobytes()).hexdigest()} for name, value in data.items()},
        "boundaries": {name: {"dtype": value.dtype.str, "shape": list(value.shape),
                              "sha256": sha256(value.tobytes()).hexdigest()} for name, value in result.items()},
        "compiler_or_emitted_program_inputs": False, "normalized_probability_then_static_pack": True,
        "source_score_scale": "0.25", "source_probability_exponent": -8,
        "mask": "preserved BF16 +0 or minfinite sentinel; explicit keep multiplication",
        "whole_layer_device_exactness": "PENDING ordinary source policy qualification",
        "validation": ({"status": "PASS", "sha256": sha256(args.validation.read_bytes()).hexdigest(),
                        "cases": len(validation["cases"]), "instructions": validation["instructions"]}
                       if validation else "PENDING separate handmade instruction tests"),
    }
    (args.out / "reference.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": "EXPECTED", "output_elements": result["output_bf16"].size,
                      "output_sha256": record["boundaries"]["output_bf16"]["sha256"]}))


if __name__ == "__main__":
    main()
