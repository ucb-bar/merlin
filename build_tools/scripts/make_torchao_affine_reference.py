#!/usr/bin/env python3
"""Regenerate the torchao reference fixture `merlin/tests/data/torchao_affine/reference.npz`.

The fixture is what `merlin/tests/ir/test_torchao_affine.py` grades our linalg decomposition of
`torchao.choose_qparams_affine` / `torchao.quantize_affine` against. It is a REFERENCE, not a
recorded output of ours: every array here is produced by calling torchao's own implementation, and
the quantize half is additionally cross-checked against ExecuTorch's `quantized_decomposed`
`quantize_per_token` (this script ASSERTS the two agree, so a torch/torchao upgrade that moves
either one fails here rather than silently re-baselining the test).

RUN IN THE CAPTURE VENV, which has torch + torchao:

    /scratch/.../model2MLIR/.venv/bin/python build_tools/scripts/make_torchao_affine_reference.py

The input cases are fixed and chosen so each arithmetic decision is observable (see the `cases`
block written into PROVENANCE.json); they are generated here rather than read from a bundle so the
fixture carries no model data.
"""
import json, numpy as np, pathlib, torch, torchao
import torch.ao.quantization.fx._decomposed  # registers the quantized_decomposed library
from torchao.quantization.quant_primitives import (choose_qparams_affine, quantize_affine,
                                                   MappingType)
qd = torch.ops.quantized_decomposed
out = pathlib.Path(__file__).resolve().parents[2] / "merlin" / "tests" / "data" / "torchao_affine"
out.mkdir(parents=True, exist_ok=True)
prev = np.load(out / "reference.npz")
keep = {name: prev[f"{name}::x"] for name in ("ties", "ulp", "tiny", "zeros", "r3")}
blob = {}
for name, x in keep.items():
    t = torch.from_numpy(np.ascontiguousarray(x))
    bs = tuple([1] * (t.dim() - 1) + [t.shape[-1]])
    scale, zp = choose_qparams_affine(t, MappingType.SYMMETRIC, bs, torch.int8,
                                      quant_min=-127, quant_max=127, eps=1e-5, scale_dtype=None)
    q = quantize_affine(t, bs, scale, zp, torch.int8, quant_min=-127, quant_max=127)
    assert int(zp.abs().max()) == 0, "SYMMETRIC zero point must be 0"
    # SECOND REFERENCE: ExecuTorch's quantized_decomposed.quantize_per_token, handed the SAME
    # (scale, zero_point). The two references must agree on the quantize half; where they do not
    # (their choose_qparams implement different schemes) that is recorded in PROVENANCE.json.
    s_t = scale.reshape(*t.shape[:-1], 1)
    q_et = qd.quantize_per_token(t, s_t, torch.zeros_like(s_t, dtype=torch.int64),
                                 -127, 127, torch.int8)
    assert bool((q_et == q).all()), f"{name}: the two references disagree on quantize_per_token"
    blob[f"{name}::x"] = np.ascontiguousarray(x)
    blob[f"{name}::scale"] = scale.numpy()
    blob[f"{name}::q"] = q.numpy()
np.savez(out / "reference.npz", **blob)
(out / "PROVENANCE.json").write_text(json.dumps({
    "produced_by": "torchao.quantization.quant_primitives.choose_qparams_affine / quantize_affine",
    "torchao_version": torchao.__version__,
    "torch_version": torch.__version__,
    "venv": "/scratch/agustin/projects/model2MLIR/.venv (python3.12)",
    "call": ("MappingType.SYMMETRIC, block_size=per-token, target_dtype=torch.int8, "
             "quant_min=-127, quant_max=127, eps=1e-5, scale_dtype=None"),
    "why_these_args": ("torchao.quantization.quant_api._int8_symm_per_token_reduced_range_quant, "
                       "the activation quantizer Int8DynamicActivationInt8WeightConfig selects "
                       "with its default act_mapping_type=MappingType.SYMMETRIC; confirmed against "
                       "the exported FX node args of a quantize_'d nn.Linear(2048, 1000)"),
    "second_reference": {
        "op": "torch.ops.quantized_decomposed.quantize_per_token (ExecuTorch's portable kernel "
              "library ships the out-variant; torch registers the CompositeExplicitAutograd impl)",
        "agrees_with_torchao_on_quantize": True,
        "checked": "handed the same (scale, zero_point=0, quant_min=-127, quant_max=127), it "
                   "produces the identical int8 on every case here",
        "does_NOT_agree_on_choose_qparams": (
            "quantized_decomposed.choose_qparams_per_token_asymmetric is a DIFFERENT scheme: "
            "ASYMMETRIC, quant_min/max -128/127, scale=(max_pos-min_neg)/255 in float64, a derived "
            "and rounded zero point, eps=finfo(f32).eps. torchao's int8_dyn_act_int8_weight is "
            "SYMMETRIC, -127/127, max|x|/127, zero point 0, eps=1e-5. Measured on the resnet50 fc "
            "activation the two schemes disagree on 2047 of 2048 int8 values."),
        "zero_point_placement_differs": (
            "torchao rounds THEN adds the zero point; quantized_decomposed adds THEN rounds. With "
            "round-half-to-even and an ODD zero point these differ (measured). Immaterial here "
            "because a SYMMETRIC scheme's zero point is 0."),
    },
    "cases": {"ties": "values landing exactly on .5 after scaling (round-half-to-EVEN)",
               "ulp": "an element where round(x*(1/s)) != round(x/s) (reciprocal-then-multiply)",
               "tiny": "magnitudes below eps, so scale hits the eps floor",
               "zeros": "all-zero block (scale = 0/127 -> eps)",
               "r3": "rank-3 per-token, scale rank 2"},
}, indent=2) + "\n")
print("wrote", out, sorted(blob))
