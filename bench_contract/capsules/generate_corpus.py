#!/usr/bin/env python3
"""Generate the capsule_bench_v0 corpus (ISA + layer + model-slice + hidden) deterministically.

Every capsule is a compiler INPUT (interface MLIR + capsule.yaml + golden + expected coverage +
README); nothing here is a kernel. Re-runnable: same output every time. Model slices come from
``model_slice_export``; the matmul-family ISA/layer capsules are emitted here; movement (A1) and
conv (B3/B4) capsules are emitted by their own generators once the v1 backend supports them.

Usage:  .venv/bin/python bench_contract/capsules/generate_corpus.py
"""
from __future__ import annotations

import pathlib
import sys

import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "merlin" / "python"))

from merlin.targetgen import capsule_golden as CG          # noqa: E402
from merlin.targetgen import model_slice_export as MSE      # noqa: E402
from merlin.targetgen.contract import schemas as S          # noqa: E402

CAP_ROOT = REPO / "bench_contract" / "capsules"
COMMON = ["FLUSH", "CONFIG_EX", "CONFIG_LD", "MVIN", "CONFIG_ST",
          "PRELOAD", "COMPUTE_PRELOADED", "MVOUT"]


def _matmul_capsule(*, name, kind, label, source_role, source_reference, M, K, N,
                    lhs, weight, epilogue, output_dtype, acc_scale=None,
                    modes=None, required=("L0", "L1", "L2", "L3"),
                    forbidden=None, semantic=None):
    modes = modes if modes is not None else {
        "i8": output_dtype == "i8", "relu": "relu" in epilogue, "acc_scale": "acc_scale" in epilogue}
    attrs = {"lhs": lhs, "weight": weight, "out": "Y0", "epilogue": epilogue,
             "output_dtype": output_dtype}
    if acc_scale is not None:
        attrs["acc_scale"] = acc_scale
    if semantic:
        attrs["semantic"] = semantic
    exp = {"instruction_classes": list(COMMON), "modes": modes}
    if forbidden:
        exp["forbidden_classes"] = forbidden
    return {
        "name": name, "kind": kind, "source_role": source_role,
        "source_reference": source_reference, "label": label,
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [{"name": weight, "role": "weight", "shape": [K, N], "dtype": "i8"},
                   {"name": lhs, "role": "input", "shape": [M, K], "dtype": "i8"}],
        "operation": {"op": "matmul", "attributes": attrs},
        "numeric_policy": {"compare": "exact_int", "dtype": output_dtype,
                           **({"acc_scale": acc_scale} if acc_scale is not None else {})},
        "expected": exp, "required_oracle_tiers": list(required),
        "vcs": "optional", "firesim": "optional",
    }


def _write_single_matmul(cap, comment):
    a = cap["operation"]["attributes"]
    M, K = cap["inputs"][1]["shape"]
    N = cap["inputs"][0]["shape"][1]
    text = MSE.emit_interface_mlir(lhs=a["lhs"], weight=a["weight"], out="Y0", M=M, K=K, N=N,
                                   epilogue=a["epilogue"], output_dtype=a["output_dtype"],
                                   acc_scale=a.get("acc_scale"), comment=comment)
    return _write_capsule_dir(cap, text, comment)


def _write_capsule_dir(cap, interface_text, comment):
    S.validate(cap, "capsule")
    cap_for_golden = {**cap, "__dir__": ""}
    gold = CG.golden(cap_for_golden)
    sub = CAP_ROOT / ({"isa": "isa", "layer": "layers",
                       "model_slice": "model_slices"}.get(cap["kind"], cap["kind"]))
    d = sub / cap["name"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "capsule.yaml").write_text(yaml.safe_dump(cap, sort_keys=False))
    (d / "capsule.interface.mlir").write_text(interface_text)
    (d / "golden.yaml").write_text(yaml.safe_dump(
        {"golden_source": "merlin_tensor_int", "outputs": gold}, sort_keys=False))
    (d / "expected_instruction_coverage.yaml").write_text(yaml.safe_dump(cap["expected"], sort_keys=False))
    (d / "README.md").write_text(f"# {cap['name']}\n\n{comment}\n\n"
                                 f"kind={cap['kind']} label={cap['label']} "
                                 f"op={cap['operation']['op']} "
                                 f"modes={cap['expected'].get('modes')}\n")
    return d


# --- A6 resident-reuse (two matmuls, one resident weight) -------------------------------------
def _resident_reuse_capsule(name, label="public"):
    K = N = 16
    matmuls = [{"lhs": "A0", "out": "Y0", "epilogue": [], "output_dtype": "i32"},
               {"lhs": "A1", "out": "Y1", "epilogue": ["relu"], "output_dtype": "i32"}]
    cap = {
        "name": name, "kind": "isa", "source_role": "handauthored_compiler_test",
        "source_reference": "resident weight reuse across two matmuls", "label": label,
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [{"name": "W", "role": "weight", "shape": [K, N], "dtype": "i8"},
                   {"name": "A0", "role": "input", "shape": [16, K], "dtype": "i8"},
                   {"name": "A1", "role": "input", "shape": [16, K], "dtype": "i8"}],
        "operation": {"op": "resident_reuse",
                      "attributes": {"weight": "W", "matmuls": matmuls, "semantic": "resident_reuse"}},
        "numeric_policy": {"compare": "exact_int", "dtype": "i32"},
        "expected": {"instruction_classes": list(COMMON),
                     "modes": {"resident_reuse": True}},
        "required_oracle_tiers": ["L0", "L1", "L2", "L3"], "vcs": "optional", "firesim": "optional",
    }
    text = (
        '// A6 resident reuse: one resident weight reused across two matmuls (16x16 tiles).\n'
        'module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", '
        'merlin_iface.abi_version = "0.1"} {\n'
        '  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<16x16xi8>\n'
        '  %A0 = merlin_iface.tensor {name = "A0", role = "input"} : tensor<16x16xi8>\n'
        '  %A1 = merlin_iface.tensor {name = "A1", role = "input"} : tensor<16x16xi8>\n'
        '  %W_res = merlin_iface.resident_pack %W {layout = "packed_rhs"} '
        ': (tensor<16x16xi8>) -> !merlin_iface.resident\n'
        '  %acc0 = merlin_iface.matmul %A0, %W_res '
        ': (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>\n'
        '  %Y0 = merlin_iface.commit %acc0 {name = "Y0", epilogue = [], output_dtype = "i32"} '
        ': (!merlin_iface.acc<i32>) -> tensor<16x16xi32>\n'
        '  %acc1 = merlin_iface.matmul %A1, %W_res '
        ': (tensor<16x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>\n'
        '  %Y1 = merlin_iface.commit %acc1 {name = "Y1", epilogue = ["relu"], output_dtype = "i32"} '
        ': (!merlin_iface.acc<i32>) -> tensor<16x16xi32>\n'
        '  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()\n'
        '}\n')
    return cap, text


def main() -> int:
    written = []

    # --- ISA capsules (matmul family) ---
    written.append(_write_single_matmul(_matmul_capsule(
        name="A0_config_smoke", kind="isa", label="public",
        source_role="handauthored_compiler_test",
        source_reference="CONFIG_EX/LD/ST smoke via a single 16x16 matmul",
        M=16, K=16, N=16, lhs="A0", weight="W", epilogue=[], output_dtype="i32"),
        "A0: CONFIG_EX/CONFIG_LD/CONFIG_ST smoke (trace must contain all three config classes)."))

    written.append(_write_single_matmul(_matmul_capsule(
        name="A2_single_tile_matmul", kind="isa", label="public",
        source_role="uplifted_from_bareMetalC", source_reference="bareMetalC/matmul_ws.c",
        M=16, K=16, N=16, lhs="A0", weight="W", epilogue=[], output_dtype="i32"),
        "A2: single 16x16 i8 matmul -> i32 (CONFIG, MVIN, PRELOAD, COMPUTE_PRELOADED, MVOUT)."))

    written.append(_write_single_matmul(_matmul_capsule(
        name="A3_k_accumulation", kind="isa", label="public",
        source_role="uplifted_from_bareMetalC", source_reference="bareMetalC/tiled_matmul_ws.c",
        M=16, K=32, N=16, lhs="A0", weight="W", epilogue=[], output_dtype="i32",
        modes={"k_accumulate": True}),
        "A3: K=32 (> tile dim) forces K-accumulation (accumulate-onto PRELOAD across K tiles)."))

    written.append(_write_single_matmul(_matmul_capsule(
        name="A4_acc_scale_i8", kind="isa", label="public",
        source_role="uplifted_from_bareMetalC", source_reference="bareMetalC/transpose_scale.c",
        M=16, K=16, N=16, lhs="A0", weight="W", epilogue=["acc_scale"], output_dtype="i8",
        acc_scale=0.0625, modes={"i8": True, "acc_scale": True}),
        "A4: int32 accumulator -> f32 acc_scale (0.0625) -> saturating i8 readout."))

    written.append(_write_single_matmul(_matmul_capsule(
        name="A5_relu_epilogue", kind="isa", label="public",
        source_role="handauthored_compiler_test", source_reference="relu activation",
        M=16, K=16, N=16, lhs="A0", weight="W", epilogue=["relu"], output_dtype="i32",
        modes={"relu": True}),
        "A5: relu activation (activation bits set in CONFIG_ST; exact relu numerics)."))

    cap, text = _resident_reuse_capsule("A6_resident_reuse")
    written.append(_write_capsule_dir(cap, text,
        "A6: one resident (packed/stationary) weight reused across two matmuls without reload."))

    written.append(_write_single_matmul(_matmul_capsule(
        name="A7_edge_padding", kind="isa", label="public",
        source_role="handauthored_compiler_test", source_reference="bareMetalC/padded.c",
        M=20, K=24, N=12, lhs="A0", weight="W", epilogue=[], output_dtype="i32",
        modes={"padded_edge": True}),
        "A7: non-16-multiple dims (20x24x12); backend zero-pads to tiles, valid window is exact."))

    # --- layer capsules ---
    written.append(_write_single_matmul(_matmul_capsule(
        name="B0_quantized_linear_i8", kind="layer", label="public",
        source_role="pytorch_model_slice", source_reference="nn.Linear + per-tensor requant",
        M=16, K=32, N=16, lhs="X", weight="W", epilogue=["acc_scale"], output_dtype="i8",
        acc_scale=0.0625, modes={"i8": True, "acc_scale": True, "k_accumulate": True},
        semantic="quantized_linear"),  # K=32 => 2 K-tiles: trace must show accumulate (cf. H2_k_accum)
        "B0: quantized linear (i8 x i8 -> i32 -> acc_scale -> i8)."))

    written.append(_write_single_matmul(_matmul_capsule(
        name="B1_linear_relu_i8", kind="layer", label="public",
        source_role="pytorch_model_slice", source_reference="nn.Linear + ReLU",
        M=16, K=32, N=16, lhs="X", weight="W", epilogue=["relu"], output_dtype="i32",
        modes={"relu": True}, semantic="linear_relu"),
        "B1: linear + relu (i32 readout)."))

    written.append(_write_single_matmul(_matmul_capsule(
        name="B2_linear_acc_scale_relu_i8", kind="layer", label="public",
        source_role="pytorch_model_slice", source_reference="nn.Linear + requant + ReLU",
        M=16, K=32, N=16, lhs="X", weight="W", epilogue=["acc_scale", "relu"], output_dtype="i8",
        acc_scale=0.0625, modes={"i8": True, "acc_scale": True, "relu": True},
        semantic="linear_acc_scale_relu"),
        "B2: linear + acc_scale + relu -> i8."))

    # --- model slices C0..C6 ---
    for cap in MSE.standard_model_slices():
        d = MSE.export_capsule_dir(CAP_ROOT / "model_slices", cap)
        written.append(d)

    print(f"wrote {len(written)} capsules:")
    for d in written:
        print(f"  {d.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
