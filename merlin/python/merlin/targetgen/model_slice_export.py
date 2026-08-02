"""Deterministic model-slice capsule exporter (MLP + attention Gemmini-relevant matmuls).

Each model slice reduces to a single weight-stationary matmul on the *existing* certified backend
path: MLP linears are ``A @ W``; attention Q/K/V projections are ``X @ Wproj``; QK^T is ``Q @ Kt``
(K provided pre-transposed as the resident weight leaf, so the device does a plain matmul and never
needs a transpose); PV is ``P @ V``. No softmax. All dims are multiples of 16 so traces stay
canonical.

Leaves are materialized via :meth:`Tensor.deterministic` — the SAME function the harness and the
golden use — so the L0 golden cannot diverge from L2/L3 on leaf data. torch is not installed here;
if it were, ``_torch_crosscheck`` would assert the torch CPU result equals the Tensor-engine golden.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from merlin.targetgen import capsule_golden as CG

def _header(target: str) -> str:
    return ('module attributes {merlin_iface.version = "0.1", '
            f'merlin_iface.target = "{target}", merlin_iface.abi_version = "0.1"}} {{')


def _dt(odt: str) -> str:
    return odt


def emit_interface_mlir(*, lhs: str, weight: str, out: str, M: int, K: int, N: int,
                        epilogue: list[str], output_dtype: str,
                        acc_scale: float | None = None, comment: str = "",
                        target: str = "gemmini", operand_dtype: str = "i8",
                        acc_dtype: str = "i32") -> str:
    """Emit a single-matmul merlin_iface module (weight-stationary). ``target``/``operand_dtype``/
    ``acc_dtype`` default to the gemmini integer path (so existing callers are byte-identical); pass the
    target's derived MLIR dtype spellings (e.g. ``f8E4M3FN``/``bf16`` for a float MXU) to emit its ISA."""
    epi = ", ".join(f'"{e}"' for e in epilogue)
    commit_attrs = f'name = "{out}", epilogue = [{epi}], output_dtype = "{output_dtype}"'
    if acc_scale is not None:
        commit_attrs += f", acc_scale = {acc_scale} : f32"
    lines = []
    if comment:
        lines.append(f"// {comment}")
    lines += [
        _header(target),
        f'  %{weight} = merlin_iface.tensor {{name = "{weight}", role = "weight"}} : '
        f'tensor<{K}x{N}x{operand_dtype}>',
        f'  %{lhs} = merlin_iface.tensor {{name = "{lhs}", role = "input"}} : '
        f'tensor<{M}x{K}x{operand_dtype}>',
        f'  %{weight}_res = merlin_iface.resident_pack %{weight} {{layout = "packed_rhs"}} '
        f': (tensor<{K}x{N}x{operand_dtype}>) -> !merlin_iface.resident',
        f'  %acc0 = merlin_iface.matmul %{lhs}, %{weight}_res '
        f': (tensor<{M}x{K}x{operand_dtype}>, !merlin_iface.resident) -> !merlin_iface.acc<{acc_dtype}>',
        f'  %{out} = merlin_iface.commit %acc0 {{{commit_attrs}}} '
        f': (!merlin_iface.acc<{acc_dtype}>) -> tensor<{M}x{N}x{_dt(output_dtype)}>',
        f'  merlin_iface.evict %{weight}_res : (!merlin_iface.resident) -> ()',
        "}",
    ]
    return "\n".join(lines) + "\n"


def make_matmul_capsule(*, name: str, semantic: str, M: int, K: int, N: int,
                        lhs: str, weight: str, out: str = "Y0",
                        epilogue: list[str] | None = None, output_dtype: str = "i32",
                        acc_scale: float | None = None, label: str = "public",
                        source_reference: str = "") -> dict:
    epilogue = epilogue or []
    modes = {"i8": output_dtype == "i8", "relu": "relu" in epilogue,
             "acc_scale": "acc_scale" in epilogue}
    attrs: dict[str, Any] = {"lhs": lhs, "weight": weight, "out": out,
                             "epilogue": epilogue, "output_dtype": output_dtype,
                             "semantic": semantic}
    if acc_scale is not None:
        attrs["acc_scale"] = acc_scale
    classes = ["FLUSH", "CONFIG_EX", "CONFIG_LD", "MVIN", "CONFIG_ST",
               "PRELOAD", "COMPUTE_PRELOADED", "MVOUT"]
    return {
        "name": name, "kind": "model_slice", "source_role": "pytorch_model_slice",
        "source_reference": source_reference or semantic, "label": label,
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [
            {"name": weight, "role": "weight", "shape": [K, N], "dtype": "i8"},
            {"name": lhs, "role": "input", "shape": [M, K], "dtype": "i8"},
        ],
        "operation": {"op": "matmul", "attributes": attrs},
        "numeric_policy": {"compare": "exact_int", "dtype": output_dtype,
                           **({"acc_scale": acc_scale} if acc_scale is not None else {})},
        "expected": {"instruction_classes": classes, "modes": modes},
        "required_oracle_tiers": ["L0", "L1", "L2", "L3"],
        "vcs": "optional", "firesim": "optional",
    }


def _torch_crosscheck(capsule: dict, tensor_golden: dict) -> str:
    """If torch is available, assert torch CPU result == Tensor-engine golden. Returns a note."""
    try:
        import torch  # noqa: F401
    except Exception:
        return "torch_unavailable (golden source = merlin_tensor_int)"
    import numpy as np
    import torch
    env = CG.materialize_capsule_leaves(capsule)
    a = capsule["operation"]["attributes"]
    lhs = np.array(env[a["lhs"]].to_list(), dtype=np.int64)
    w = np.array(env[a["weight"]].to_list(), dtype=np.int64)
    res = torch.from_numpy(lhs) @ torch.from_numpy(w)
    res = res.numpy()
    if "acc_scale" in a.get("epilogue", []):
        return "torch_crosscheck_skipped (acc_scale rounding compared via Tensor engine)"
    if "relu" in a.get("epilogue", []):
        res = np.maximum(res, 0)
    exp = np.array(tensor_golden[a["out"]], dtype=np.int64)
    return "torch_crosscheck_pass" if np.array_equal(res, exp) else "torch_crosscheck_FAIL"


def export_capsule_dir(root: str | Path, capsule: dict, *, comment: str = "",
                       readme: str = "") -> Path:
    """Write a full capsule directory (capsule.yaml, interface.mlir, golden.yaml,
    expected_instruction_coverage.yaml, README.md). Returns the directory path."""
    a = capsule["operation"]["attributes"]
    M = capsule["inputs"][1]["shape"][0]
    K = capsule["inputs"][1]["shape"][1]
    N = capsule["inputs"][0]["shape"][1]
    text = emit_interface_mlir(lhs=a["lhs"], weight=a["weight"], out=a.get("out", "Y0"),
                               M=M, K=K, N=N, epilogue=a.get("epilogue", []),
                               output_dtype=a.get("output_dtype", "i32"),
                               acc_scale=a.get("acc_scale"), comment=comment or capsule["name"])
    gold = CG.golden(capsule)
    note = _torch_crosscheck(capsule, gold)

    d = Path(root) / capsule["name"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "capsule.yaml").write_text(yaml.safe_dump(capsule, sort_keys=False), encoding="utf-8")
    (d / "capsule.interface.mlir").write_text(text, encoding="utf-8")
    (d / "golden.yaml").write_text(
        yaml.safe_dump({"golden_source": "merlin_tensor_int", "crosscheck": note, "outputs": gold},
                       sort_keys=False), encoding="utf-8")
    (d / "expected_instruction_coverage.yaml").write_text(
        yaml.safe_dump(capsule["expected"], sort_keys=False), encoding="utf-8")
    (d / "README.md").write_text(
        readme or f"# {capsule['name']}\n\nModel-slice capsule "
        f"({a.get('semantic')}). Single weight-stationary matmul "
        f"[{M}x{K}] x [{K}x{N}] -> output_dtype={a.get('output_dtype')}. "
        f"Golden: {note}.\n", encoding="utf-8")
    return d


# -- convenience builders for the C0-C6 model slices ------------------------------------------
def standard_model_slices(label: str = "public") -> list[dict]:
    """The seven required model-slice capsules (C0-C6), all multiples of 16."""
    SEQ, DM, DH = 16, 64, 16
    return [
        make_matmul_capsule(name="C0_mlp_linear1", semantic="mlp_linear1",
                            M=SEQ, K=DM, N=DM, lhs="X", weight="W1", label=label,
                            source_reference="mlp.fc1"),
        make_matmul_capsule(name="C1_mlp_activation_linear2", semantic="mlp_relu_linear2",
                            M=SEQ, K=DM, N=DM, lhs="H", weight="W2", epilogue=["relu"],
                            label=label, source_reference="mlp.relu+fc2"),
        make_matmul_capsule(name="C2_attention_q_projection", semantic="attn_q_proj",
                            M=SEQ, K=DM, N=DH, lhs="X", weight="Wq", label=label,
                            source_reference="attn.q_proj"),
        make_matmul_capsule(name="C3_attention_k_projection", semantic="attn_k_proj",
                            M=SEQ, K=DM, N=DH, lhs="X", weight="Wk", label=label,
                            source_reference="attn.k_proj"),
        make_matmul_capsule(name="C4_attention_v_projection", semantic="attn_v_proj",
                            M=SEQ, K=DM, N=DH, lhs="X", weight="Wv", label=label,
                            source_reference="attn.v_proj"),
        make_matmul_capsule(name="C5_attention_qk_matmul", semantic="attn_qk",
                            M=SEQ, K=DH, N=SEQ, lhs="Q", weight="Kt", label=label,
                            source_reference="attn.q@k^T (Kt = K transposed leaf)"),
        make_matmul_capsule(name="C6_attention_pv_matmul", semantic="attn_pv",
                            M=SEQ, K=SEQ, N=DH, lhs="P", weight="V", label=label,
                            source_reference="attn.p@v"),
    ]
