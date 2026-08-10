"""Section compile: one REAL int8 linear from a model2MLIR whole model, run end-to-end.

Increment 4 of the whole-model compiler. Two layers of proof:

* The engine's int8 weight-only dequant-pack (RES_PACK carrying a per-channel scale) computes
  ``A @ (W_i8 * scale)`` and the residency-bypassing reference agrees — a pure command-buffer test,
  no xDSL or artifact needed, so it runs everywhere.
* A real ``linalg.matmul`` section is carved out of tiny_llama_int8 (matmul + its
  ``dequantize_per_channel`` weight prep), lowered through ``lower_module``, and executed — the
  int8 dequant idiom handled by the compiler, numerically exact vs numpy. Gated on the m2m checkout.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from merlin.xdsl_dialects import _common


def _tiny_llama_int8():
    for var in ("MERLIN_M2M_DIR", "MERLIN_MODEL2MLIR"):
        base = os.environ.get(var)
        if base:
            p = Path(base) / "workloads" / "tiny_llama" / "tiny_llama_int8.mlir"
            if p.is_file():
                return p
    return None


def test_tensor_dequant_per_channel():
    from merlin.runtime.tensor import Tensor

    w = Tensor((2, 3), [1, 2, 3, 4, 5, 6], "i8")
    deq = w.dequant_per_channel(Tensor((3,), [0.5, 2.0, 10.0], "f32"), axis=1)
    assert deq.dtype == "f32"
    assert deq.to_list() == [[0.5, 4.0, 30.0], [2.0, 10.0, 60.0]]


def test_engine_int8_dequant_pack_end_to_end():
    """A RES_PACK carrying a per-channel scale dequantizes the i8 weight; simulator and the
    independent reference agree and both match ``A @ (W * scale)``."""
    from merlin.runtime.reference import reference_outputs
    from merlin.runtime.simulator import simulate

    M, K, N = 4, 5, 3
    cb = {
        "abi_version": "0.1", "target": "toy_npu", "backend": "simulator",
        "tensors": {
            "W": {"shape": [K, N], "dtype": "i8", "role": "weight"},
            "S": {"shape": [N], "dtype": "f32", "role": "input"},
            "A": {"shape": [M, K], "dtype": "f32", "role": "input"},
            "Wr": {"shape": [K, N], "dtype": "f32", "role": "input"},
            "acc": {"shape": [M, N], "dtype": "f32", "role": "input"},
            "Y": {"shape": [M, N], "dtype": "f32", "role": "output"},
        },
        "outputs": ["Y"],
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "Wr", "scale": "S"},
             "attributes": {"layout": "packed_rhs", "dequant_axis": 1}},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A", "rhs": "Wr", "dst": "acc"}},
            {"opcode": "COMMIT", "operands": {"src": "acc", "dst": "Y"},
             "attributes": {"epilogue": [], "output_dtype": "f32"}},
            {"opcode": "EVICT", "operands": {"handle": "Wr"}},
        ],
    }
    rng = np.random.default_rng(0)
    W = rng.integers(-8, 8, size=(K, N)).astype(np.int8)
    S = (rng.random(N).astype(np.float32) * 0.1 + 0.01)
    A = rng.standard_normal((M, K)).astype(np.float32)
    inj = {"W": W.tolist(), "S": S.tolist(), "A": A.tolist()}

    sim = simulate(cb, inj)
    ref = reference_outputs(cb, inj)
    got = np.array(sim["outputs"]["Y"], dtype=np.float32)
    exp = A @ (W.astype(np.float32) * S[None, :])
    assert np.allclose(got, exp, rtol=1e-4, atol=1e-3)
    assert np.allclose(np.array(ref["Y"], dtype=np.float32), exp, rtol=1e-4, atol=1e-3)


@pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")
@pytest.mark.skipif(_tiny_llama_int8() is None,
                    reason="model2MLIR checkout not resolvable (set MERLIN_M2M_DIR)")
def test_real_tiny_llama_int8_section_compiles_and_runs():
    """Carve matmul_1 out of the real int8 tiny_llama, lower it, and run it: the compiler handles
    the int8 weight-only dequant idiom (folded into RES_PACK) and the result matches numpy."""
    from merlin.frontends import linalg_mlir as fl
    from merlin.xdsl_dialects.lowering import execute, lower_module
    from merlin.xdsl_dialects.lowering.section_extract import section_from_matmul

    mod = fl.parse_mlir_file(_tiny_llama_int8())
    sec, boundary, _outs = section_from_matmul(mod, "matmul_1")
    res = lower_module(sec)
    for m in res.modules():
        m.verify()

    cb = res.command_buffer
    # The int8 idiom lowered to a dequant-pack: RES_PACK carries the scale + axis.
    pack = next(c for c in cb["commands"] if c["opcode"] == "RES_PACK")
    assert "scale" in pack["operands"]
    assert pack["attributes"]["dequant_axis"] == 1

    # Boundary shapes tell us the real layer dims (M x K, K x N weight, N scale).
    shapes = {tuple(b.type.get_shape()): str(b.type.element_type) for b in boundary}
    (K, N), = [s for s in shapes if len(s) == 2 and shapes[s] == "i8"]
    (M, K2), = [s for s in shapes if len(s) == 2 and shapes[s] != "i8"]
    assert K2 == K

    rng = np.random.default_rng(0)
    W = rng.integers(-8, 8, size=(K, N)).astype(np.int8)
    S = (rng.random(N).astype(np.float32) * 0.02 + 0.001)
    A = rng.standard_normal((M, K)).astype(np.float32)
    inj = {}
    for name, spec in cb["tensors"].items():
        sh = tuple(spec["shape"])
        if sh == (K, N):
            inj[name] = W.tolist()
        elif sh == (N,):
            inj[name] = S.tolist()
        elif sh == (M, K):
            inj[name] = A.tolist()

    run = execute(res, inj)
    got = np.array(next(iter(run["outputs"].values())), dtype=np.float32)
    exp = A @ (W.astype(np.float32) * S[None, :])
    assert run["correct"] is True
    assert np.allclose(got, exp, rtol=1e-4, atol=1e-2)
