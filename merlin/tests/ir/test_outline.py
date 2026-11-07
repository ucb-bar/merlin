"""Per-dispatch outliner (``merlin-outline-dispatches``).

Structural invariants run everywhere xDSL is present; the value-preserving check lowers
both the monolithic and the outlined module through the real host toolchain and asserts
bit-identical outputs (auto-skips without the m2m venv / clang-23).
"""

from __future__ import annotations

import numpy as np
import pytest

from merlin.common.paths import repo_root
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


REPO = repo_root()


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


#: QUANT with the scale's reshape printed AFTER the dequantize that reads it, the shape a model2MLIR
#: revision's Q/DQ fusion emitted for every per-channel int8 Linear. xDSL parses and verifies it.
USE_BEFORE_DEF = """
builtin.module {
  func.func @forward(%w: tensor<8x6xi8>, %s2: tensor<6x1xf32>, %zp: tensor<6xi32>,
                     %x: tensor<4x8xf32>) -> tensor<4x6xf32> {
    %wf = "quant_ext.dequantize_per_channel"(%w, %s, %zp)
        <{axis = 1 : i64, input_dtype = "i8"}>
        : (tensor<8x6xi8>, tensor<6xf32>, tensor<6xi32>) -> tensor<8x6xf32>
    %e = tensor.empty() : tensor<4x6xf32>
    %c0 = arith.constant 0.0 : f32
    %f = linalg.fill ins(%c0 : f32) outs(%e : tensor<4x6xf32>) -> tensor<4x6xf32>
    %y = linalg.matmul ins(%x, %wf : tensor<4x8xf32>, tensor<8x6xf32>)
         outs(%f : tensor<4x6xf32>) -> tensor<4x6xf32>
    %s = tensor.collapse_shape %s2 [[0, 1]] : tensor<6x1xf32> into tensor<6xf32>
    func.return %y : tensor<4x6xf32>
  }
}
"""


def test_a_module_that_uses_a_value_before_defining_it_is_refused_by_name():
    """It used to die inside the outliner on a bare KeyError naming an xDSL object."""
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects.lowering.outline import OutlineError, outline_dispatches

    with pytest.raises(OutlineError, match=r"op #0 \(quant_ext.dequantize_per_channel\) uses a result of "
                                           r"tensor.collapse_shape before that op defines it"):  # fmt: skip
        outline_dispatches(parse_mlir_text(USE_BEFORE_DEF))
    outline_dispatches(parse_mlir_text(QUANT))  # the same program in order outlines


def test_runtime_inventory_matches_exact_outlined_symbols():
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.runtime.dispatch_runtime import outlined_dispatch_inventory
    from merlin.xdsl_dialects.lowering.outline import outline_dispatches

    result = outline_dispatches(parse_mlir_text(CHAIN))
    rows = outlined_dispatch_inventory(result)
    assert [row["symbol"] for row in rows] == [dispatch.symbol for dispatch in result.dispatches]
    assert [row["root_op"] for row in rows] == ["linalg.matmul", "linalg.matmul"]
    assert all(row["region_id"] is None and row["prov_op"] is None for row in rows)


def test_runtime_inventory_preserves_split_rewrite_roles():
    from merlin.runtime.dispatch_runtime import outlined_dispatch_inventory
    from merlin.xdsl_dialects.lowering.outline import DispatchInfo, OutlineResult

    dispatch = DispatchInfo(
        index=0,
        symbol="forward$kernel_0__rmatmul_0",
        root_op="linalg.generic",
        n_operands=3,
        result_types=["tensor<2x2xi32>"],
        prov={
            "prov.region_id": "matmul_0",
            "prov.op": "batch_matmul",
            "prov.role": "contraction",
            "prov.rewrite": "w8a8_integer_compute",
        },
    )
    row = outlined_dispatch_inventory(OutlineResult(module=None, dispatches=[dispatch]))[0]
    assert row["prov_role"] == "contraction"
    assert row["prov_rewrite"] == "w8a8_integer_compute"


def test_runtime_transform_audit_retains_exact_replayable_stages(tmp_path):
    import hashlib
    import json

    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.runtime.dispatch_runtime import qualify_model_transform_audit, record_model_transform_audit
    from merlin.xdsl_dialects.lowering.outline import outline_dispatches

    source = tmp_path / "model.mlir"
    source.write_text(CHAIN, encoding="utf-8")
    module = parse_mlir_text(CHAIN)
    outlined = outline_dispatches(module)
    assert record_model_transform_audit(source, tmp_path, module, outlined, enabled=False) is None
    assert not list(tmp_path.glob("ir-audit-*"))
    index_path = record_model_transform_audit(source, tmp_path, module, outlined, enabled=True)
    assert index_path is not None
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["outcome"] == "completed"
    assert [stage["name"] for stage in index["stages"]] == ["captured-model", "normalized-model", "outlined-model"]
    assert index["stages"][0]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert all(
        hashlib.sha256((index_path.parent / stage["file"]).read_bytes()).hexdigest() == stage["sha256"]
        for stage in index["stages"]
    )
    qualified = qualify_model_transform_audit(index_path)
    assert qualified["status"] == "structural_replay_matched"
    assert qualified["semantic_equivalence"] == "not_proven"
    archived = index_path.parent / index["stages"][2]["file"]
    archived.write_text(archived.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from its index"):
        qualify_model_transform_audit(index_path)


def test_model_transform_audit_replays_normalization_from_captured_ir(tmp_path):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.runtime.dispatch_runtime import qualify_model_transform_audit, record_model_transform_audit
    from merlin.xdsl_dialects.lowering.outline import outline_dispatches

    source = tmp_path / "model.mlir"
    source.write_text(CHAIN, encoding="utf-8")
    recipe = {
        "int8_compute": False,
        "quant_passes": None,
        "prequant_gather": False,
        "selection_policy": "all",
    }
    normalized = parse_mlir_text(CHAIN)
    index_path = record_model_transform_audit(
        source,
        tmp_path,
        normalized,
        outline_dispatches(normalized),
        enabled=True,
        normalization_recipe=recipe,
    )
    assert qualify_model_transform_audit(index_path)["normalization_replay"] == "matched"

    # A self-consistent archive of a *different* valid transform is not a valid
    # replay of the declared pass sequence, even when all recorded hashes match.
    changed = parse_mlir_text(CHAIN.replace("arith.constant 0.0", "arith.constant 1.0", 1))
    bad_index = record_model_transform_audit(
        source,
        tmp_path,
        changed,
        outline_dispatches(changed),
        enabled=True,
        normalization_recipe=recipe,
    )
    with pytest.raises(ValueError, match="normalization differs from replay"):
        qualify_model_transform_audit(bad_index)


def test_model_transform_audit_replays_the_int8_rewrite_and_abstains_on_custom_selection(tmp_path):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.runtime.dispatch_runtime import (
        _normalize_model_module,
        qualify_model_transform_audit,
        record_model_transform_audit,
    )
    from merlin.xdsl_dialects.lowering.outline import outline_dispatches

    source = tmp_path / "model.mlir"
    source.write_text(QUANT, encoding="utf-8")
    normalized = parse_mlir_text(QUANT)
    _normalize_model_module(normalized, int8_compute=True, quant_passes=None, prequant_gather=False)
    recipe = {
        "int8_compute": True,
        "quant_passes": None,
        "prequant_gather": False,
        "selection_policy": "all",
    }
    index_path = record_model_transform_audit(
        source,
        tmp_path,
        normalized,
        outline_dispatches(normalized),
        enabled=True,
        normalization_recipe=recipe,
    )
    assert qualify_model_transform_audit(index_path)["normalization_replay"] == "matched"
    custom_index = record_model_transform_audit(
        source,
        tmp_path,
        normalized,
        outline_dispatches(normalized),
        enabled=True,
        normalization_recipe={**recipe, "selection_policy": "custom_unreplayable"},
    )
    assert qualify_model_transform_audit(custom_index)["normalization_replay"] == "unsupported_custom_selection"


def test_model_transform_audit_replays_captured_custom_tensor_slice(tmp_path):
    """The audit must replay the same custom-form capture that production can parse."""
    import subprocess

    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower.toolchain import compiler_python
    from merlin.runtime.dispatch_runtime import (
        _normalize_model_module,
        qualify_model_transform_audit,
        record_model_transform_audit,
    )
    from merlin.xdsl_dialects.lowering.outline import outline_dispatches

    python = compiler_python()
    if not python.is_file() or subprocess.run([str(python), "-c", "import torch_mlir"], capture_output=True).returncode:
        pytest.skip("MLIR-capable compiler Python is unavailable")
    source = tmp_path / "custom_slice.mlir"
    source.write_text(
        """builtin.module {
  func.func @forward(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = tensor.insert_slice %arg0 into %arg1[0] [4] [1] : tensor<4xf32> into tensor<4xf32>
    func.return %0 : tensor<4xf32>
  }
}
"""
    )
    normalized = parse_mlir_file(source)
    _normalize_model_module(
        normalized,
        int8_compute=False,
        quant_passes=None,
        prequant_gather=False,
        torchao_cpu_avx2_mean=False,
        torchao_cpu_avx2_layer_norm=False,
    )
    receipt = record_model_transform_audit(
        source,
        tmp_path / "audit",
        normalized,
        outline_dispatches(normalized),
        enabled=True,
        normalization_recipe={
            "int8_compute": False,
            "quant_passes": None,
            "prequant_gather": False,
            "selection_policy": "all",
        },
    )
    archived = {path.name for path in receipt.parent.iterdir()}
    assert qualify_model_transform_audit(receipt)["normalization_replay"] == "matched"
    assert {path.name for path in receipt.parent.iterdir()} == archived


def test_run_model_default_audit_option_reaches_argument_binding(tmp_path, monkeypatch):
    from merlin.runtime import dispatch_runtime as runtime

    model_dir = tmp_path / "capture"
    model_dir.mkdir()
    (model_dir / "model.mlir").write_text(CHAIN, encoding="utf-8")

    class ReachedArgumentBinding(Exception):
        pass

    def stop_at_argument_binding(_model_dir):
        raise ReachedArgumentBinding

    monkeypatch.setattr(runtime, "resolve_forward_args", stop_at_argument_binding)
    with pytest.raises(ReachedArgumentBinding):
        runtime.run_model(model_dir, tmp_path / "work")


def test_outline_kernels_are_self_contained():
    """Each kernel owns its accumulator init: fill/empty/constant cloned inside."""
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects.lowering.outline import outline_dispatches

    res = outline_dispatches(parse_mlir_text(CHAIN))
    kernels = [op for op in res.module.walk() if op.name == "func.func" and "$kernel_" in op.sym_name.data]
    assert len(kernels) == 2
    for k in kernels:
        body = list(k.body.blocks[0].ops)
        names = [o.name for o in body]
        assert "linalg.fill" in names  # accumulator zero-init lives in the kernel
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
    kernel_funcs = [op for op in res.module.walk() if op.name == "func.func" and "$kernel_" in op.sym_name.data]
    in_kernels = sum(sum(1 for o in k.body.blocks[0].ops if o.name == "linalg.matmul") for k in kernel_funcs)
    assert in_kernels == n_matmul == res.n_kernels
    # The driver body holds calls, not the compute ops.
    driver = next(op for op in res.module.walk() if op.name == "func.func" and "$kernel_" not in op.sym_name.data)
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
    assert res.dispatches[0].root_op == "linalg.generic"  # dequant
    assert res.dispatches[1].root_op == "linalg.matmul"


EXTERNAL_CALL = """
builtin.module {
  func.func private @quantize_activations(tensor<4x8xf32>) -> tensor<4x8xf32>
  func.func @forward(%w: tensor<8x6xf32>, %x: tensor<4x8xf32>) -> tensor<4x6xf32> {
    %q = func.call @quantize_activations(%x) : (tensor<4x8xf32>) -> tensor<4x8xf32>
    %e0 = tensor.empty() : tensor<4x6xf32>
    %c0 = arith.constant 0.0 : f32
    %f0 = linalg.fill ins(%c0 : f32) outs(%e0 : tensor<4x6xf32>) -> tensor<4x6xf32>
    %y0 = linalg.matmul ins(%q, %w : tensor<4x8xf32>, tensor<8x6xf32>)
          outs(%f0 : tensor<4x6xf32>) -> tensor<4x6xf32>
    func.return %y0 : tensor<4x6xf32>
  }
}
"""


def test_an_external_declaration_is_not_mistaken_for_the_driver():
    """A body-less ``func.func private`` printed first must not be picked as the function to outline.

    A capture that leaves an operation to an outside implementation prints its declaration ahead of
    ``@forward``; selecting it used to die on ``fn.body.blocks[0]`` with a bare ``IndexError`` that
    named neither the module nor the symbol. The undefined symbol is what should be reported, and
    it should be reported BY NAME.
    """
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects.lowering.outline import OutlineError, outline_dispatches

    with pytest.raises(OutlineError) as excinfo:
        outline_dispatches(parse_mlir_text(EXTERNAL_CALL))
    message = str(excinfo.value)
    assert "@quantize_activations" in message
    assert "forward" in message


def test_missing_forward_raises():
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects.lowering.outline import OutlineError, outline_dispatches

    with pytest.raises(OutlineError):
        outline_dispatches(parse_mlir_text(CHAIN), forward="nope")


# --- scales to a real whole model ------------------------------------------------------


@pytest.mark.skipif(
    not (REPO / "out/artifacts/recaptures/small_consistent/model.mlir").is_file(),
    reason="small_llama capture not present",
)
def test_outline_scales_to_small_llama():
    """The outliner forms a verified dispatch table from the real small LLaMA model."""
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.xdsl_dialects.lowering.outline import outline_dispatches

    m = parse_mlir_file(REPO / "out/artifacts/recaptures/small_consistent/model.mlir")
    res = outline_dispatches(m)  # verifies internally (IsolatedFromAbove etc.)
    roots = [d.root_op for d in res.dispatches]
    assert res.n_kernels == len(roots) > 100
    assert roots.count("linalg.matmul") == 15  # one kernel per dense layer
    # Every kernel is a real private func with a body terminated by a return.
    kfuncs = [op for op in res.module.walk() if op.name == "func.func" and "$kernel_" in op.sym_name.data]
    assert len(kfuncs) == res.n_kernels
    assert all(list(op.body.blocks[0].ops)[-1].name == "func.return" for op in kfuncs)


@pytest.mark.skipif(
    not (REPO / "out/artifacts/recaptures/tiny_consistent/model.mlir").is_file(),
    reason="tiny_llama capture not present",
)
def test_outline_scales_to_tiny_llama():
    """Real TinyLlama-1.1B: 155 matmul dispatches, all region-captures parameterized."""
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.xdsl_dialects.lowering.outline import outline_dispatches

    m = parse_mlir_file(REPO / "out/artifacts/recaptures/tiny_consistent/model.mlir")
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
    _run_host(
        outlined,
        "outlined",
        tmp_path,
        [(W.ctypes.data, (8, 6)), (X.ctypes.data, (4, 8)), (Z.ctypes.data, (6, 5)), (y.ctypes.data, (4, 5))],
    )
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
    _run_host(
        outlined,
        "outlined_q",
        tmp_path,
        [
            (w.ctypes.data, (8, 6)),
            (s.ctypes.data, (6,)),
            (zp.ctypes.data, (6,)),
            (x.ctypes.data, (4, 8)),
            (y.ctypes.data, (4, 6)),
        ],
    )
    assert np.allclose(y, ref, rtol=1e-4, atol=1e-4), np.abs(y - ref).max()
