"""Per-dispatch outliner (``merlin-outline-dispatches``).

Structural invariants run everywhere xDSL is present; the value-preserving check lowers
both the monolithic and the outlined module through the real host toolchain and asserts
bit-identical outputs (auto-skips without the m2m venv / clang-23).
"""
from __future__ import annotations

import ctypes
import numpy as np
import pytest

from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")

CHAIN = """
builtin.module {
  func.func @forward(%w: tensor<8x6xf32>, %x: tensor<4x8xf32>, %z: tensor<6x5xf32>)
      -> tensor<4x5xf32> {
    %e0 = tensor.empty() : tensor<4x6xf32>
    %c0 = arith.constant 0.0 : f32
    %f0 = linalg.fill ins(%c0 : f32) outs(%e0 : tensor<4x6xf32>) -> tensor<4x6xf32>
    %y0 = linalg.matmul ins(%x, %w : tensor<4x8xf32>, tensor<8x6xf32>)
          outs(%f0 : tensor<4x6xf32>) -> tensor<4x6xf32>
    %e1 = tensor.empty() : tensor<4x5xf32>
    %c1 = arith.constant 0.0 : f32
    %f1 = linalg.fill ins(%c1 : f32) outs(%e1 : tensor<4x5xf32>) -> tensor<4x5xf32>
    %y1 = linalg.matmul ins(%y0, %z : tensor<4x6xf32>, tensor<6x5xf32>)
          outs(%f1 : tensor<4x5xf32>) -> tensor<4x5xf32>
    func.return %y1 : tensor<4x5xf32>
  }
}
"""

QUANT = """
builtin.module {
  func.func @forward(%w: tensor<8x6xi8>, %s: tensor<6xf32>, %zp: tensor<6xi32>,
                     %x: tensor<4x8xf32>) -> tensor<4x6xf32> {
    %wf = "quant_ext.dequantize_per_channel"(%w, %s, %zp)
        <{axis = 1 : i64, input_dtype = "i8"}>
        : (tensor<8x6xi8>, tensor<6xf32>, tensor<6xi32>) -> tensor<8x6xf32>
    %e = tensor.empty() : tensor<4x6xf32>
    %c0 = arith.constant 0.0 : f32
    %f = linalg.fill ins(%c0 : f32) outs(%e : tensor<4x6xf32>) -> tensor<4x6xf32>
    %y = linalg.matmul ins(%x, %wf : tensor<4x8xf32>, tensor<8x6xf32>)
         outs(%f : tensor<4x6xf32>) -> tensor<4x6xf32>
    func.return %y : tensor<4x6xf32>
  }
}
"""


REPO = __import__("pathlib").Path(__file__).resolve().parents[3]


def _count(module, name):
    return sum(1 for op in module.walk() if op.name == name)


def test_outline_splits_each_compute_op_into_a_kernel():
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects.lowering.outline import outline_dispatches

    res = outline_dispatches(parse_mlir_text(CHAIN))
    assert res.n_kernels == 2
    assert [d.root_op for d in res.dispatches] == ["linalg.matmul", "linalg.matmul"]
    # Each matmul takes its two activations/weights as operands (fill/empty/const cloned in).
    assert [d.n_operands for d in res.dispatches] == [2, 2]
    # The module verifies as valid SSA / typed IR.
    res.module.verify()


def test_outline_kernels_are_self_contained():
    """Each kernel owns its accumulator init: fill/empty/constant cloned inside."""
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects.lowering.outline import outline_dispatches

    res = outline_dispatches(parse_mlir_text(CHAIN))
    kernels = [op for op in res.module.walk()
               if op.name == "func.func" and "$kernel_" in op.sym_name.data]
    assert len(kernels) == 2
    for k in kernels:
        body = list(k.body.blocks[0].ops)
        names = [o.name for o in body]
        assert "linalg.fill" in names          # accumulator zero-init lives in the kernel
        assert "tensor.empty" in names
        assert "arith.constant" in names
        assert names.count("linalg.matmul") == 1


def test_outline_conserves_every_compute_op():
    """No root compute op is lost or duplicated across the kernels."""
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects.lowering.outline import outline_dispatches

    src = parse_mlir_text(CHAIN)
    n_matmul = _count(src, "linalg.matmul")
    res = outline_dispatches(src)
    kernel_funcs = [op for op in res.module.walk()
                    if op.name == "func.func" and "$kernel_" in op.sym_name.data]
    in_kernels = sum(
        sum(1 for o in k.body.blocks[0].ops if o.name == "linalg.matmul")
        for k in kernel_funcs)
    assert in_kernels == n_matmul == res.n_kernels
    # The driver body holds calls, not the compute ops.
    driver = next(op for op in res.module.walk()
                  if op.name == "func.func" and "$kernel_" not in op.sym_name.data)
    driver_names = [o.name for o in driver.body.blocks[0].ops]
    assert driver_names.count("func.call") == res.n_kernels
    assert "linalg.matmul" not in driver_names


def test_outline_separates_dequant_from_matmul():
    """The dequant generic and the matmul land in distinct kernels (fusion is future)."""
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.passes_xdsl import lower_quant_ext
    from merlin.xdsl_dialects.lowering.outline import outline_dispatches

    m = parse_mlir_text(QUANT)
    assert lower_quant_ext(m) == 1
    res = outline_dispatches(m)
    assert res.n_kernels == 2
    assert res.dispatches[0].root_op == "linalg.generic"   # dequant
    assert res.dispatches[1].root_op == "linalg.matmul"


def test_missing_forward_raises():
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects.lowering.outline import OutlineError, outline_dispatches

    with pytest.raises(OutlineError):
        outline_dispatches(parse_mlir_text(CHAIN), forward="nope")


# --- scales to a real whole model ------------------------------------------------------

@pytest.mark.skipif(not (REPO / "output/small_consistent/model.mlir").is_file(),
                    reason="small_llama capture not present")
def test_outline_scales_to_small_llama():
    """The outliner forms a verified dispatch table from the real small LLaMA model."""
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.xdsl_dialects.lowering.outline import outline_dispatches

    m = parse_mlir_file(REPO / "output/small_consistent/model.mlir")
    res = outline_dispatches(m)            # verifies internally (IsolatedFromAbove etc.)
    roots = [d.root_op for d in res.dispatches]
    assert res.n_kernels == len(roots) > 100
    assert roots.count("linalg.matmul") == 15      # one kernel per dense layer
    # Every kernel is a real private func with a body terminated by a return.
    kfuncs = [op for op in res.module.walk()
              if op.name == "func.func" and "$kernel_" in op.sym_name.data]
    assert len(kfuncs) == res.n_kernels
    assert all(list(op.body.blocks[0].ops)[-1].name == "func.return" for op in kfuncs)


@pytest.mark.skipif(not (REPO / "output/tiny_consistent/model.mlir").is_file(),
                    reason="tiny_llama capture not present")
def test_outline_scales_to_tiny_llama():
    """Real TinyLlama-1.1B: 155 matmul dispatches, all region-captures parameterized."""
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.xdsl_dialects.lowering.outline import outline_dispatches

    m = parse_mlir_file(REPO / "output/tiny_consistent/model.mlir")
    res = outline_dispatches(m)
    roots = [d.root_op for d in res.dispatches]
    assert roots.count("linalg.matmul") == 155
    assert res.n_kernels > 1000


# --- value-preserving end-to-end (real toolchain) --------------------------------------

def _toolchain():
    from merlin.llvmlower import toolchain

    return toolchain.available()


def _run_host(text, tag, tmp_path, args):
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.lower import lower_model

    res = lower_model(text, tmp_path / tag, targets=("host",))
    model = HostModel.load(str(res.host_so))
    model(args)


# Note: each toolchain test loads exactly one model .so. Loading several MLIR model
# libraries in one process clashes on their shared RTLD_GLOBAL ciface symbols
# (``_mlir_ciface_forward``/``memrefCopy``), so we gate the outlined module against a
# numpy ground-truth reference rather than a second in-process monolithic load.


@pytest.mark.skipif(not _toolchain(), reason="m2m venv / clang-23 missing")
def test_outline_runs_correctly_on_host(tmp_path):
    """The outlined module computes the correct result end to end through the toolchain."""
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects._common import text as to_text
    from merlin.xdsl_dialects.lowering.outline import outline_dispatches

    rng = np.random.default_rng(0)
    W = rng.standard_normal((8, 6)).astype(np.float32)
    X = rng.standard_normal((4, 8)).astype(np.float32)
    Z = rng.standard_normal((6, 5)).astype(np.float32)
    ref = (X @ W) @ Z

    outlined = to_text(outline_dispatches(parse_mlir_text(CHAIN)).module)
    y = np.zeros((4, 5), np.float32)
    _run_host(outlined, "outlined", tmp_path,
              [(W.ctypes.data, (8, 6)), (X.ctypes.data, (4, 8)),
               (Z.ctypes.data, (6, 5)), (y.ctypes.data, (4, 5))])
    assert np.allclose(y, ref, rtol=1e-4, atol=1e-4), np.abs(y - ref).max()


@pytest.mark.skipif(not _toolchain(), reason="m2m venv / clang-23 missing")
def test_outline_quantized_runs_correctly_on_host(tmp_path):
    """A dequant kernel feeding a matmul kernel across the call boundary is correct."""
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.passes_xdsl import lower_quant_ext
    from merlin.xdsl_dialects._common import text as to_text
    from merlin.xdsl_dialects.lowering.outline import outline_dispatches

    w = np.array([(i % 7) - 3 for i in range(48)], np.int8).reshape(8, 6)
    s = np.array([0.5, 1.0, 0.25, 2.0, 1.5, 0.125], np.float32)
    zp = np.array([1, 0, -2, 3, 0, 1], np.int32)
    x = np.array([(i % 5) - 2 for i in range(32)], np.float32).reshape(4, 8)
    ref = x @ ((w.astype(np.float32) - zp.astype(np.float32)) * s)

    m = parse_mlir_text(QUANT)
    lower_quant_ext(m)
    outlined = to_text(outline_dispatches(m).module)
    y = np.zeros((4, 6), np.float32)
    _run_host(outlined, "outlined_q", tmp_path,
              [(w.ctypes.data, (8, 6)), (s.ctypes.data, (6,)), (zp.ctypes.data, (6,)),
               (x.ctypes.data, (4, 8)), (y.ctypes.data, (4, 6))])
    assert np.allclose(y, ref, rtol=1e-4, atol=1e-4), np.abs(y - ref).max()
