#!/usr/bin/env python3
"""Generate the ATLAS capsule corpus — fp8-e4m3 operands, bf16 accumulate/output, FLOAT goldens.

Atlas is a float MXU (operand fp8_e4m3, accumulate bf16, E8M0 block scale, 32x32 mesh). It physically
CANNOT reproduce the shared gemmini INTEGER corpus (i8xi8->i32, exact_int). This script emits an
atlas-CORRECT corpus: every capsule declares fp8-e4m3 inputs, a bf16 output, and a ``tolerance_float``
numeric_policy — and its golden is produced by an INDEPENDENT oracle (the spec-definition project
``specir``'s reference model), NOT by the atlas RTL (which would be circular).

The independent oracle
-----------------------
We reuse ``specir.oracle.dtypes`` (the fp8_e4m3 / bf16 IEEE-style codecs + correctly-rounded
``round_to_format``) and ``specir.oracle.refmodel.fp_reduce`` (an order/cadence-specified fp reduction).
These are the SAME primitives ``specc testbench --gen atlas-npu --op op.matmul_mxu0`` uses to emit the
E4M3FMA cell golden. We compose that cell reference into a TILE matmul, faithful to the atlas mxu0
datapath declared in ``targets/atlas-npu/atlas.t1.mlir``:

    acc[i,j] <- round_bf16( acc + round_bf16( a[i,k] * w[k,j] ) )   # per-step round, k index-sequential

with fp8 subnormal operands flushed to zero (FTZ) — here a no-op, since the input palette is all exact
fp8 normals. mxu0's own declared precision bound is ``relative tol = 2e-2`` (per-step bf16 lossiness);
that is the rtol we grade with.

Determinism
-----------
Leaves are materialized deterministically from the tensor NAME (no RNG) into a small palette of
exactly-fp8-representable values, so the corpus is reproducible and the golden is a genuine
non-degenerate fp8->bf16 matmul (unlike reusing merlin's int 0..3 fill, whose bytes-as-fp8 collapse to
subnormal/zero). The exact fp8 encodings used are recorded in each ``golden.yaml`` provenance block.

Run:  PYTHONPATH=/scratch2/agustin/mvp-lhwir/spec .venv/bin/python \
          merlin/contract/capsules/atlas/generate_atlas_corpus.py
"""
from __future__ import annotations

import math
import os
import sys
from fractions import Fraction
from pathlib import Path

import yaml

# --- reach the independent oracle (specir) ------------------------------------------------------
_SPECIR = os.environ.get("SPECIR_ROOT", "/scratch2/agustin/mvp-lhwir/spec")
if _SPECIR not in sys.path:
    sys.path.insert(0, _SPECIR)
from specir.oracle import dtypes as D                     # noqa: E402
from specir.oracle.refmodel import fp_reduce              # noqa: E402

FP8 = D.FP8_E4M3
BF16 = D.BF16
HERE = Path(__file__).resolve().parent

# mxu0 declared precision (atlas.t1.mlir: effect.precision.tol, relative) — the grade-time rtol.
RTOL = 2.0e-2
# absolute floor: bf16 ULP is 2^(e-7); for the O(10..10^2) tile magnitudes here ULP ~ 0.06..0.5.
# atol covers near-zero outputs where rtol underflows; 0.25 == one bf16 ULP at magnitude ~32.
ATOL = 0.25

# exactly-fp8_e4m3-representable palette (no subnormals -> FTZ is a no-op, non-degenerate products).
PALETTE = [0.5, 1.0, 1.5, 2.0, -1.0, 2.5, -0.5, 3.0]


# --- deterministic fp8 materialization (name -> fp8 raws) ---------------------------------------
def _prod(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def det_fp8(name: str, shape) -> tuple[list[int], list[float]]:
    """Deterministic fp8_e4m3 fill from ``name`` (stable across runs). Returns (raws, decoded_floats)."""
    n = _prod(shape)
    seed = sum((i + 1) * ord(c) for i, c in enumerate(name)) or 1
    vals = [PALETTE[(seed * (k + 1) + k * k) % len(PALETTE)] for k in range(n)]
    raws = [D.encode_float(v, FP8) for v in vals]
    return raws, vals


# --- the mxu0 datapath, cell-composed into a tile -----------------------------------------------
def _bf16_round(x: Fraction) -> int:
    return D.round_to_format(x, BF16, "rne")


def _fp8_val(raw: int) -> Fraction:
    return D.decode_float_exact(raw, FP8)


def mxu0_matmul(a_raw: list[int], a_shape, w_raw: list[int], w_shape) -> list[list[int]]:
    """(m,k) fp8 x (k,n) fp8 -> (m,n) bf16 raws, per the mxu0 per-step / index-sequential contract."""
    m, k = a_shape
    k2, n = w_shape
    assert k == k2, f"matmul shape mismatch {a_shape} x {w_shape}"
    out = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            prods = [_bf16_round(_fp8_val(a_raw[i * k + p]) * _fp8_val(w_raw[p * n + j]))
                     for p in range(k)]
            out[i][j] = fp_reduce(prods, BF16, order="index_sequential",
                                  cadence="per_step", rm="rne")
    return out


def apply_scale(y_raw: list[list[int]], scale: float) -> list[list[int]]:
    s = Fraction(scale).limit_denominator(1 << 20)
    return [[_bf16_round(D.decode_float_exact(v, BF16) * s) for v in row] for row in y_raw]


def apply_relu(y_raw: list[list[int]]) -> list[list[int]]:
    out = []
    for row in y_raw:
        r = []
        for v in row:
            fv = D.decode_float(v, BF16)
            r.append(v if fv > 0 else 0)   # +0.0 for non-positive (bf16 0x0000)
        out.append(r)
    return out


def bf16_floats(y_raw: list[list[int]]) -> list[list[float]]:
    return [[D.decode_float(v, BF16) for v in row] for row in y_raw]


def transpose(raw: list[int], shape) -> tuple[list[int], tuple[int, int]]:
    r, c = shape
    out = [0] * (r * c)
    for i in range(r):
        for j in range(c):
            out[j * r + i] = raw[i * c + j]
    return out, (c, r)


# --- capsule specs (fp8 in / bf16 out) ----------------------------------------------------------
# DIM = 32 is the atlas MXU mesh; tiles are 32-multiples. Movement/plumbing capsules use a small tile.
DIM = 32

SPECS = [
    # ---- ISA primitives ----
    dict(cat="isa", name="AT0_config_smoke", kind="isa", label="public",
         source_role="handauthored_compiler_test",
         source_reference="MXU config plumbing via a single 32x32 fp8 matmul",
         op="matmul", A=("A0", (DIM, DIM)), W=("W", (DIM, DIM)), out="Y0",
         instr=["CONFIG_EX", "GMEM_LD", "FMA", "GMEM_ST"], modes={}),
    dict(cat="isa", name="AT1_mvin_mvout", kind="isa", label="public",
         source_role="handauthored_compiler_test",
         source_reference="load fp8 tile -> store as bf16 (dequant movement)",
         op="movement", A=("X", (16, 16)), out="Y0",
         instr=["GMEM_LD", "GMEM_ST"], modes={"movement": True},
         forbidden=["FMA"]),
    dict(cat="isa", name="AT2_single_tile_matmul", kind="isa", label="public",
         source_role="handauthored_compiler_test",
         source_reference="single 32x32 fp8 MXU tile -> bf16",
         op="matmul", A=("A0", (DIM, DIM)), W=("W", (DIM, DIM)), out="Y0",
         instr=["CONFIG_EX", "GMEM_LD", "FMA", "GMEM_ST"], modes={}),
    dict(cat="isa", name="AT3_k_accumulation", kind="isa", label="public",
         source_role="handauthored_compiler_test",
         source_reference="K=64 (multi-tile) bf16 accumulate",
         op="matmul", A=("A0", (DIM, 64)), W=("W", (64, DIM)), out="Y0",
         instr=["CONFIG_EX", "GMEM_LD", "FMA", "GMEM_ST"], modes={"k_accumulate": True}),
    dict(cat="isa", name="AT4_bf16_scale", kind="isa", label="public",
         source_role="handauthored_compiler_test",
         source_reference="matmul + bf16 output scale (VPU requant precursor)",
         op="matmul", A=("A0", (DIM, DIM)), W=("W", (DIM, DIM)), out="Y0",
         epilogue=["acc_scale"], acc_scale=0.5,
         instr=["CONFIG_EX", "GMEM_LD", "FMA", "GMEM_ST"], modes={"acc_scale": True}),
    dict(cat="isa", name="AT5_relu_epilogue", kind="isa", label="public",
         source_role="handauthored_compiler_test",
         source_reference="matmul + relu epilogue (bf16)",
         op="matmul", A=("A0", (DIM, DIM)), W=("W", (DIM, DIM)), out="Y0",
         epilogue=["relu"], instr=["CONFIG_EX", "GMEM_LD", "FMA", "GMEM_ST"],
         modes={"relu": True}),
    dict(cat="isa", name="AT6_resident_reuse", kind="isa", label="public",
         source_role="handauthored_compiler_test",
         source_reference="one resident weight, two activation tiles",
         op="resident_reuse", W=("W", (DIM, DIM)),
         matmuls=[("A0", (DIM, DIM), "Y0"), ("A1", (DIM, DIM), "Y1")],
         instr=["CONFIG_EX", "GMEM_LD", "FMA", "GMEM_ST"], modes={"resident_reuse": True}),

    # ---- layers ----
    dict(cat="layers", name="BT0_quantized_linear", kind="layer", label="public",
         source_role="pytorch_model_slice",
         source_reference="nn.Linear fp8 x fp8 -> bf16",
         op="linear", A=("X", (DIM, DIM)), W=("W", (DIM, DIM)), out="Y0",
         instr=["CONFIG_EX", "GMEM_LD", "FMA", "GMEM_ST"], modes={"k_accumulate": True}),
    dict(cat="layers", name="BT1_linear_relu", kind="layer", label="public",
         source_role="pytorch_model_slice",
         source_reference="nn.Linear + ReLU, fp8 -> bf16",
         op="linear", A=("X", (DIM, DIM)), W=("W", (DIM, DIM)), out="Y0",
         epilogue=["relu"], instr=["CONFIG_EX", "GMEM_LD", "FMA", "GMEM_ST"],
         modes={"relu": True, "k_accumulate": True}),

    # ---- model slices ----
    dict(cat="model_slices", name="CT0_mlp_linear1", kind="model_slice", label="public",
         source_role="pytorch_model_slice",
         source_reference="MLP first projection (64-wide hidden)",
         op="linear", A=("X", (DIM, DIM)), W=("W", (DIM, 64)), out="Y0",
         instr=["CONFIG_EX", "GMEM_LD", "FMA", "GMEM_ST"], modes={"k_accumulate": True}),
    dict(cat="model_slices", name="CT1_attention_qk", kind="model_slice", label="public",
         source_role="pytorch_model_slice",
         source_reference="attention Q @ K^T scores, fp8 -> bf16",
         op="attention_qk", Q=("Q", (DIM, DIM)), K=("K", (DIM, DIM)), out="Y0",
         instr=["CONFIG_EX", "GMEM_LD", "FMA", "GMEM_ST"], modes={}),

    # ---- hidden ----
    dict(cat="hidden", name="HT0_matmul_hidden", kind="isa", label="hidden",
         source_role="handauthored_compiler_test", source_reference="hidden single-tile matmul",
         op="matmul", A=("A0", (DIM, DIM)), W=("W", (DIM, DIM)), out="Y0",
         instr=["CONFIG_EX", "GMEM_LD", "FMA", "GMEM_ST"], modes={}),
    dict(cat="hidden", name="HT1_scale_hidden", kind="isa", label="hidden",
         source_role="handauthored_compiler_test", source_reference="hidden matmul + bf16 scale",
         op="matmul", A=("A0", (DIM, DIM)), W=("W", (DIM, DIM)), out="Y0",
         epilogue=["acc_scale"], acc_scale=0.25,
         instr=["CONFIG_EX", "GMEM_LD", "FMA", "GMEM_ST"], modes={"acc_scale": True}),
    dict(cat="hidden", name="HT2_k_accum_hidden", kind="isa", label="hidden",
         source_role="handauthored_compiler_test", source_reference="hidden K=64 accumulate",
         op="matmul", A=("A0", (DIM, 64)), W=("W", (64, DIM)), out="Y0",
         instr=["CONFIG_EX", "GMEM_LD", "FMA", "GMEM_ST"], modes={"k_accumulate": True}),
]


# --- emit one capsule dir -----------------------------------------------------------------------
FP8_MLIR = "f8E4M3FN"   # MLIR spelling of fp8_e4m3


def _tensor_line(var, name, role, shape, dt):
    dims = "x".join(str(s) for s in shape)
    return f'  %{var} = merlin_iface.tensor {{name = "{name}", role = "{role}"}} : tensor<{dims}x{dt}>'


def emit_interface_mlir(spec) -> str:
    op = spec["op"]
    L = ['module attributes {merlin_iface.version = "0.1", merlin_iface.target = "atlas", '
         'merlin_iface.abi_version = "0.1"} {']
    if op in ("matmul", "linear"):
        an, ashape = spec["A"]; wn, wshape = spec["W"]
        epi = spec.get("epilogue", [])
        L.append(_tensor_line("W", wn, "weight", wshape, FP8_MLIR))
        L.append(_tensor_line("A0", an, "input", ashape, FP8_MLIR))
        L.append(f'  %W_res = merlin_iface.resident_pack %W {{layout = "packed_rhs"}} : '
                 f'(tensor<{"x".join(map(str, wshape))}x{FP8_MLIR}>) -> !merlin_iface.resident')
        L.append(f'  %acc0 = merlin_iface.matmul %A0, %W_res : '
                 f'(tensor<{"x".join(map(str, ashape))}x{FP8_MLIR}>, !merlin_iface.resident) '
                 f'-> !merlin_iface.acc<bf16>')
        m, n = ashape[0], wshape[1]
        L.append(f'  %{spec["out"]} = merlin_iface.commit %acc0 {{name = "{spec["out"]}", '
                 f'epilogue = {epi!r}, output_dtype = "bf16"}} : '
                 f'(!merlin_iface.acc<bf16>) -> tensor<{m}x{n}xbf16>')
        L.append('  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()')
    elif op == "movement":
        xn, xshape = spec["A"]
        L.append(_tensor_line("X", xn, "input", xshape, FP8_MLIR))
        m, n = xshape
        L.append(f'  %{spec["out"]} = merlin_iface.movement %X {{name = "{spec["out"]}", '
                 f'semantic = "mvin_mvout", output_dtype = "bf16"}} : '
                 f'(tensor<{m}x{n}x{FP8_MLIR}>) -> tensor<{m}x{n}xbf16>')
    elif op == "resident_reuse":
        wn, wshape = spec["W"]
        L.append(_tensor_line("W", wn, "weight", wshape, FP8_MLIR))
        L.append(f'  %W_res = merlin_iface.resident_pack %W {{layout = "packed_rhs"}} : '
                 f'(tensor<{"x".join(map(str, wshape))}x{FP8_MLIR}>) -> !merlin_iface.resident')
        for idx, (an, ashape, oname) in enumerate(spec["matmuls"]):
            L.append(_tensor_line(an, an, "input", ashape, FP8_MLIR))
            L.append(f'  %acc{idx} = merlin_iface.matmul %{an}, %W_res : '
                     f'(tensor<{"x".join(map(str, ashape))}x{FP8_MLIR}>, !merlin_iface.resident) '
                     f'-> !merlin_iface.acc<bf16>')
            m, n = ashape[0], wshape[1]
            L.append(f'  %{oname} = merlin_iface.commit %acc{idx} {{name = "{oname}", '
                     f'epilogue = [], output_dtype = "bf16"}} : '
                     f'(!merlin_iface.acc<bf16>) -> tensor<{m}x{n}xbf16>')
        L.append('  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()')
    elif op == "attention_qk":
        qn, qshape = spec["Q"]; kn, kshape = spec["K"]
        L.append(_tensor_line("Q", qn, "input", qshape, FP8_MLIR))
        L.append(_tensor_line("K", kn, "input", kshape, FP8_MLIR))
        m = qshape[0]; n = kshape[0]
        L.append(f'  %{spec["out"]} = merlin_iface.attention_qk %Q, %K {{name = "{spec["out"]}", '
                 f'output_dtype = "bf16"}} : '
                 f'(tensor<{qshape[0]}x{qshape[1]}x{FP8_MLIR}>, '
                 f'tensor<{kshape[0]}x{kshape[1]}x{FP8_MLIR}>) -> tensor<{m}x{n}xbf16>')
    L.append("}")
    return "\n".join(L) + "\n"


def build_capsule(spec) -> tuple[dict, dict, dict, dict]:
    """Return (capsule.yaml, golden.yaml, expected_instruction_coverage.yaml, provenance-in-golden)."""
    op = spec["op"]
    inputs = []
    prov_inputs = {}
    attrs = {"out": spec.get("out", "Y0")}
    outputs = {}

    def _reg(name, role, shape):
        raw, vals = det_fp8(name, shape)
        inputs.append({"name": name, "role": role, "shape": list(shape), "dtype": "fp8_e4m3"})
        prov_inputs[name] = {"shape": list(shape), "fp8_raw_hex": [f"0x{r:02x}" for r in raw],
                             "decoded": vals}
        return raw

    if op in ("matmul", "linear"):
        an, ashape = spec["A"]; wn, wshape = spec["W"]
        a_raw = _reg(an, "input", ashape)
        w_raw = _reg(wn, "weight", wshape)
        y = mxu0_matmul(a_raw, ashape, w_raw, wshape)
        epi = spec.get("epilogue", [])
        if "acc_scale" in epi:
            y = apply_scale(y, spec["acc_scale"])
        if "relu" in epi:
            y = apply_relu(y)
        attrs.update({"lhs": an, "weight": wn, "epilogue": epi, "output_dtype": "bf16"})
        if "acc_scale" in epi:
            attrs["acc_scale"] = spec["acc_scale"]
        outputs[spec["out"]] = bf16_floats(y)

    elif op == "movement":
        xn, xshape = spec["A"]
        x_raw = _reg(xn, "input", xshape)
        m, n = xshape
        y = [[_bf16_round(_fp8_val(x_raw[i * n + j])) for j in range(n)] for i in range(m)]
        attrs.update({"src": xn, "semantic": "mvin_mvout", "output_dtype": "bf16"})
        outputs[spec["out"]] = bf16_floats(y)

    elif op == "resident_reuse":
        wn, wshape = spec["W"]
        w_raw = _reg(wn, "weight", wshape)
        mm = []
        for an, ashape, oname in spec["matmuls"]:
            a_raw = _reg(an, "input", ashape)
            y = mxu0_matmul(a_raw, ashape, w_raw, wshape)
            outputs[oname] = bf16_floats(y)
            mm.append({"lhs": an, "out": oname, "epilogue": [], "output_dtype": "bf16"})
        attrs = {"weight": wn, "matmuls": mm, "output_dtype": "bf16"}

    elif op == "attention_qk":
        qn, qshape = spec["Q"]; kn, kshape = spec["K"]
        q_raw = _reg(qn, "input", qshape)
        k_raw = _reg(kn, "input", kshape)
        kt_raw, kt_shape = transpose(k_raw, kshape)     # Q @ K^T
        y = mxu0_matmul(q_raw, qshape, kt_raw, kt_shape)
        attrs = {"q": qn, "k": kn, "out": spec["out"], "epilogue": [], "output_dtype": "bf16"}
        outputs[spec["out"]] = bf16_floats(y)

    else:
        raise ValueError(f"unhandled op {op}")

    numeric_policy = {"compare": "tolerance_float", "dtype": "bf16",
                      "atol": ATOL, "rtol": RTOL}
    if spec.get("epilogue") and "acc_scale" in spec["epilogue"]:
        numeric_policy["acc_scale"] = spec["acc_scale"]

    expected = {"instruction_classes": spec["instr"]}
    if spec.get("forbidden"):
        expected["forbidden_classes"] = spec["forbidden"]
    if spec.get("modes"):
        expected["modes"] = spec["modes"]

    capsule = {
        "name": spec["name"], "kind": spec["kind"],
        "source_role": spec["source_role"], "source_reference": spec["source_reference"],
        "label": spec["label"], "interface_mlir": "capsule.interface.mlir",
        "inputs": inputs,
        "operation": {"op": op, "attributes": attrs},
        "numeric_policy": numeric_policy,
        "expected": expected,
        "required_oracle_tiers": ["L0", "L1", "L3"],
        "vcs": "optional", "firesim": "optional",
    }
    golden = {
        "golden_source": "specir_refmodel_fp8_bf16",
        "oracle_provenance": {
            "engine": "specir.oracle.dtypes + specir.oracle.refmodel.fp_reduce",
            "spec": "targets/atlas-npu/atlas.t1.mlir :: op.matmul_mxu0 (E4M3FMA cell)",
            "datapath": "acc <- round_bf16(acc + round_bf16(a*w)); k index_sequential; per_step; rne; fp8 FTZ",
            "operand_dtype": "fp8_e4m3", "accum_dtype": "bf16", "output_dtype": "bf16",
            "note": "INDEPENDENT of the atlas RTL (not self-oracle); torch cross-corroboration "
                    "unavailable in this env (no torch) — specir refmodel is the reference.",
            "grade_policy": {"compare": "tolerance_float", "atol": ATOL, "rtol": RTOL,
                             "rtol_rationale": "mxu0 effect.precision.tol (relative 2e-2)"},
            "inputs": prov_inputs,
        },
        "outputs": outputs,
    }
    return capsule, golden, dict(expected), {}


def main() -> int:
    n = 0
    for spec in SPECS:
        d = HERE / spec["cat"] / spec["name"]
        d.mkdir(parents=True, exist_ok=True)
        capsule, golden, coverage, _ = build_capsule(spec)
        (d / "capsule.yaml").write_text(yaml.safe_dump(capsule, sort_keys=False), encoding="utf-8")
        (d / "golden.yaml").write_text(yaml.safe_dump(golden, sort_keys=False), encoding="utf-8")
        (d / "expected_instruction_coverage.yaml").write_text(
            yaml.safe_dump(coverage, sort_keys=False), encoding="utf-8")
        (d / "capsule.interface.mlir").write_text(emit_interface_mlir(spec), encoding="utf-8")
        n += 1
        print(f"  wrote {spec['cat']}/{spec['name']}")
    print(f"\n{n} atlas capsules written under {HERE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
