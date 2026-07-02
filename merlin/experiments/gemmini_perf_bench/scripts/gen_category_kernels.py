#!/usr/bin/env python3
"""Extend the perf corpus beyond matmul to ALL the multi-way-runnable op categories the bareMetalC
golden suite + the model2MLIR models exercise on Gemmini:

  conv2d    — golden conv.c family + (models are transformer-heavy, so conv is golden-driven)
  movement  — golden mvin_mvout family (data movement, no compute)
  attention — QK^T (Q@Kt) + PV (P@V); capsule_bench models these as matmul w/ semantic attn_qk/attn_pv
              (the Gemmini-relevant attention op harvested from the transformer models)

Each is authored in the exact capsule_bench format (capsule.yaml + capsule.interface.mlir + golden.yaml
via capsule_golden, which supports matmul/conv2d/movement/attention_qk/attention_pv). The runner
(capsule_runner) already lowers + grades these for baseline/merlin/native; golden(a) uses conv.c /
mvin_mvout.c. Residual/pooling/transpose/fusions are golden-reference-only (separate, op unsupported by
the MLIR arms) and handled elsewhere.

Usage: gen_category_kernels.py [--dry-run]
"""
from __future__ import annotations

import argparse

import yaml

import _pbcommon as PB
from merlin.targetgen import capsule_golden as CG  # noqa: E402
from merlin.targetgen import model_slice_export as MSE  # noqa: E402

_HDR = ('module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", '
        'merlin_iface.abi_version = "0.1"} {')
ACC = 0.0625

# --- conv2d kernels (NHWC, weight packed [kh*kw*ci, co]) -----------------------------------------
# (id, N,H,W,Ci, Kh,Kw,Co, stride, note). Golden = conv.c family.
CONV = [
    ("K_conv_std_3x3_8x8x4_8",   1, 8, 8, 4,   3, 3, 8,  1, "standard 3x3 conv (capsule B3 shape)"),
    ("K_conv_3x3_16x16x16_16",   1, 16, 16, 16, 3, 3, 16, 1, "16ch 3x3 conv (one-tile weight)"),
    ("K_conv_1x1_16x16x32_32",   1, 16, 16, 32, 1, 1, 32, 1, "1x1 conv (pointwise = matmul-like)"),
    ("K_conv_3x3_stride2_16x16x16_32", 1, 16, 16, 16, 3, 3, 32, 2, "strided 3x3 (downsample)"),
]
# --- movement kernels (mvin->mvout, identity) ----------------------------------------------------
MOVE = [("K_move_16x16", 16, 16, "single DIM tile movement (capsule A1 shape)"),
        ("K_move_64x64", 64, 64, "multi-tile movement"),
        ("K_move_16x128", 16, 128, "wide movement")]
# --- attention kernels (matmul: QK^T = Q[S,d]@Kt[d,S]; PV = P[S,S]@V[S,d]) ------------------------
# harvested-representative seq/head dims from the transformer models (flattened to 2D matmul).
ATTN = [("K_attn_qk_64x64x64",  "attn_qk", 64, 64, 64,  "Q", "Kt", "QK^T (seq64, head64)"),
        ("K_attn_pv_64x64x64",  "attn_pv", 64, 64, 64,  "P", "V",  "PV (seq64, head64)"),
        ("K_attn_qk_128x64x128", "attn_qk", 128, 64, 128, "Q", "Kt", "QK^T (seq128, head64)")]

_CLASSES = ["FLUSH", "CONFIG_EX", "CONFIG_LD", "MVIN", "CONFIG_ST", "PRELOAD",
            "COMPUTE_PRELOADED", "MVOUT"]


def _conv_out(h, w, kh, kw, s):
    return (h - kh) // s + 1, (w - kw) // s + 1


def conv_capsule(kid, N, H, W, Ci, Kh, Kw, Co, s, note):
    Ho, Wo = _conv_out(H, W, Kh, Kw, s)
    Krow = Kh * Kw * Ci
    Orow = N * Ho * Wo
    cap = {
        "name": kid, "kind": "layer", "source_role": "uplifted_from_bareMetalC", "source_reference": "bareMetalC/conv.c", "label": "dev",
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [{"name": "W", "role": "weight", "shape": [Krow, Co], "dtype": "i8"},
                   {"name": "IFM", "role": "input", "shape": [N, H, W, Ci], "dtype": "i8"}],
        "operation": {"op": "conv2d", "attributes": {
            "ifm": "IFM", "weight": "W", "out": "Y0", "ci": Ci, "kh": Kh, "kw": Kw,
            "stride": [s, s], "padding": [0, 0, 0, 0], "dilation": [1, 1], "layout": "nhwc",
            "epilogue": [], "output_dtype": "i32", "semantic": "conv2d_im2col"}},
        "numeric_policy": {"compare": "exact_int", "dtype": "i32"},
        "expected": {"instruction_classes": _CLASSES,
                     "modes": {"conv2d": True, "i8": False, "relu": False, "acc_scale": False,
                               "k_accumulate": Krow > PB.DIM}},
        "required_oracle_tiers": ["L0", "L1", "L2", "L3"], "vcs": "optional", "firesim": "optional",
    }
    macs = Orow * Co * Krow
    mlir = (f"// {kid} ({note})\n{_HDR}\n"
            f'  %IFM = merlin_iface.tensor {{name = "IFM", role = "input"}} : tensor<{N}x{H}x{W}x{Ci}xi8>\n'
            f'  %W = merlin_iface.tensor {{name = "W", role = "weight"}} : tensor<{Krow}x{Co}xi8>\n'
            f'  %W_res = merlin_iface.resident_pack %W {{layout = "packed_conv_rhs"}} : '
            f'(tensor<{Krow}x{Co}xi8>) -> !merlin_iface.resident\n'
            f'  %Y0 = merlin_iface.conv2d %IFM, %W_res {{kernel = [{Kh}, {Kw}, {Ci}, {Co}], '
            f'stride = [{s}, {s}], padding = [0, 0, 0, 0], dilation = [1, 1], name = "Y0", '
            f'epilogue = [], output_dtype = "i32", layout = "nhwc"}} : '
            f'(tensor<{N}x{H}x{W}x{Ci}xi8>, !merlin_iface.resident) -> tensor<{Orow}x{Co}xi32>\n'
            f'  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()\n}}\n')
    entry = {"id": kid, "op": "conv2d", "dtype": "i8", "shape": f"{N}x{H}x{W}x{Ci}_k{Kh}x{Kw}x{Co}s{s}",
             "macs": macs, "sim_hint": "L2+L3" if macs <= 2_000_000 else "L2_only",
             "source": "baremetalc:conv.c", "note": note}
    return cap, mlir, entry


def move_capsule(kid, M, N, note):
    cap = {
        "name": kid, "kind": "isa", "source_role": "uplifted_from_bareMetalC", "source_reference": "bareMetalC/mvin_mvout.c", "label": "dev",
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [{"name": "X", "role": "input", "shape": [M, N], "dtype": "i8"}],
        "operation": {"op": "movement", "attributes": {"src": "X", "out": "Y0",
                                                       "semantic": "mvin_mvout"}},
        "numeric_policy": {"compare": "exact_int", "dtype": "i8"},
        "expected": {"instruction_classes": ["FLUSH", "CONFIG_LD", "MVIN", "CONFIG_ST", "MVOUT"],
                     "modes": {"movement": True, "i8": True}},
        "required_oracle_tiers": ["L0", "L1", "L2", "L3"], "vcs": "optional", "firesim": "optional",
    }
    mlir = (f"// {kid} ({note})\n{_HDR}\n"
            f'  %X = merlin_iface.tensor {{name = "X", role = "input"}} : tensor<{M}x{N}xi8>\n'
            f'  %Y0 = merlin_iface.movement %X {{name = "Y0"}} : '
            f'(tensor<{M}x{N}xi8>) -> tensor<{M}x{N}xi8>\n}}\n')
    entry = {"id": kid, "op": "movement", "dtype": "i8", "shape": f"{M}x{N}", "macs": 0,
             "sim_hint": "L2+L3", "source": "baremetalc:mvin_mvout.c", "note": note}
    return cap, mlir, entry


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    PB.KERNELS.mkdir(parents=True, exist_ok=True)
    conv_e, move_e, attn_e = [], [], []

    for spec in CONV:
        cap, mlir, entry = conv_capsule(*spec)
        conv_e.append(entry)
        print(f"  conv {entry['id']:34s} macs={entry['macs']:,} sim={entry['sim_hint']}")
        if not a.dry_run:
            _write(cap, mlir)
    for spec in MOVE:
        cap, mlir, entry = move_capsule(*spec)
        move_e.append(entry)
        print(f"  move {entry['id']:34s} {entry['shape']}")
        if not a.dry_run:
            _write(cap, mlir)
    for kid, sem, M, K, N, lhs, w, note in ATTN:
        cap = MSE.make_matmul_capsule(name=kid, semantic=sem, M=M, K=K, N=N, lhs=lhs, weight=w,
                                      out="Y0", epilogue=[], output_dtype="i32", label="dev",
                                      source_reference=f"model_attention:{sem}")
        macs = PB.matmul_macs(M, K, N)
        attn_e.append({"id": kid, "op": "matmul", "dtype": "i8", "shape": f"{M}x{K}x{N}",
                       "macs": macs, "sim_hint": "L2+L3" if macs <= 2_000_000 else "L2_only",
                       "source": f"model_attention:{sem}", "note": note})
        print(f"  attn {kid:34s} {M}x{K}x{N} ({sem})")
        if not a.dry_run:
            MSE.export_capsule_dir(PB.KERNELS, cap, comment=f"{kid} ({note})")

    if a.dry_run:
        print("\n[dry-run] nothing written")
        return 0
    cp = PB.KERNELS / "kernel_corpus.yaml"
    doc = yaml.safe_load(cp.read_text()) if cp.exists() else {}
    doc = doc or {}
    doc["conv_kernels"] = conv_e
    doc["movement_kernels"] = move_e
    doc["attention_kernels"] = attn_e
    cp.write_text(yaml.safe_dump(doc, sort_keys=False))
    print(f"\nemitted {len(conv_e)} conv + {len(move_e)} movement + {len(attn_e)} attention kernels")
    return 0


def _write(cap, mlir):
    """Author capsule.yaml + interface.mlir + golden.yaml (+ coverage) for a conv2d/movement capsule."""
    d = PB.KERNELS / cap["name"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "capsule.yaml").write_text(yaml.safe_dump(cap, sort_keys=False))
    (d / "capsule.interface.mlir").write_text(mlir)
    gold = CG.golden(cap)
    (d / "golden.yaml").write_text(yaml.safe_dump(
        {"golden_source": "merlin_tensor_int", "outputs": gold}, sort_keys=False))
    (d / "expected_instruction_coverage.yaml").write_text(yaml.safe_dump(cap["expected"], sort_keys=False))


if __name__ == "__main__":
    raise SystemExit(main())
