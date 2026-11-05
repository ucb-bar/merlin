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

SPLAT_MATMUL = """module {
  func.func @forward(%a: tensor<2x3xf32>, %b: tensor<3x4xf32>) -> tensor<2x4xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %init = tensor.splat %zero : tensor<2x4xf32>
    %result = linalg.matmul {prov.region_id = "matmul_0"}
      ins(%a, %b : tensor<2x3xf32>, tensor<3x4xf32>)
      outs(%init : tensor<2x4xf32>) -> tensor<2x4xf32>
    func.return %result : tensor<2x4xf32>
  }
}"""

CAPTURED_GENERIC = """module {
  func.func @forward(%x: tensor<2xf32>, %bias: f32, %extra: f32) -> tensor<2xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %init = tensor.splat %zero : tensor<2xf32>
    %result = linalg.generic {
      indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>],
      iterator_types = ["parallel"]
    } ins(%x : tensor<2xf32>) outs(%init : tensor<2xf32>) {
    ^bb0(%value: f32, %acc: f32):
      %sum = arith.addf %value, %bias : f32
      %both = arith.addf %sum, %extra : f32
      linalg.yield %both : f32
    } -> tensor<2xf32>
    func.return %result : tensor<2xf32>
  }
}"""


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


def test_positive_zero_splat_stays_inside_outlined_matmul():
    from io import StringIO

    from xdsl.printer import Printer

    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.targetgen.contract.model_host_object import _check_entry_dataflow
    from merlin.xdsl_dialects.lowering.outline import externalize_dispatches, outline_dispatches

    outlined = outline_dispatches(parse_mlir_text(SPLAT_MATMUL), forward="forward")
    assert len(outlined.dispatches) == 1
    dispatch = outlined.dispatches[0]
    assert dispatch.n_operands == 2
    kernel = next(op for op in outlined.module.body.block.ops
                  if op.name == "func.func" and op.sym_name.data == dispatch.symbol)
    assert [op.name for op in kernel.body.block.ops] == [
        "arith.constant", "tensor.splat", "linalg.matmul", "func.return",
    ]

    def printed(module):
        stream = StringIO()
        Printer(stream=stream, print_generic_format=True).print_op(module)
        return stream.getvalue() + "\n"

    retained = _check_entry_dataflow(SPLAT_MATMUL, printed(outlined.module), "forward", [])
    assert retained["host_initializer_check_count"] == 1
    external = externalize_dispatches(outlined, [dispatch.symbol])
    selected = _check_entry_dataflow(
        SPLAT_MATMUL, printed(external.module), "forward", [dispatch.symbol],
    )
    assert selected["native_external_call_count"] == 1
    assert selected["native_zero_initialization_obligation_count"] == 1

    negative_zero = SPLAT_MATMUL.replace("0.000000e+00", "-0.000000e+00")
    negative = outline_dispatches(parse_mlir_text(negative_zero), forward="forward")
    external_negative = externalize_dispatches(negative, [negative.dispatches[0].symbol])
    with pytest.raises(ValueError, match="nonzero or unknown initializer"):
        _check_entry_dataflow(
            negative_zero, printed(external_negative.module),
            "forward", [negative.dispatches[0].symbol],
        )


def test_retained_generic_keeps_splat_and_captured_scalar_wiring():
    from io import StringIO

    from xdsl.printer import Printer

    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.targetgen.contract.model_host_object import _check_entry_dataflow
    from merlin.xdsl_dialects.lowering.outline import outline_dispatches

    def printed(module):
        stream = StringIO()
        Printer(stream=stream, print_generic_format=True).print_op(module)
        return stream.getvalue() + "\n"

    outlined = outline_dispatches(parse_mlir_text(CAPTURED_GENERIC), forward="forward")
    assert [dispatch.n_operands for dispatch in outlined.dispatches] == [3]
    checked = _check_entry_dataflow(CAPTURED_GENERIC, printed(outlined.module), "forward", [])
    assert checked["host_defined_call_count"] == 1
    assert checked["host_initializer_check_count"] == 1

    kernel = next(op for op in outlined.module.body.block.ops
                  if op.name == "func.func" and "$kernel_" in op.sym_name.data)
    generic = next(op for op in kernel.body.block.ops if op.name == "linalg.generic")
    additions = [op for op in generic.regions[0].block.ops if op.name == "arith.addf"]
    additions[1].operands[1] = kernel.body.block.args[1]
    with pytest.raises(ValueError, match="changed its Linalg computation"):
        _check_entry_dataflow(CAPTURED_GENERIC, printed(outlined.module), "forward", [])


def test_externalize_exact_dispatch_keeps_driver_and_other_kernel():
    from xdsl.dialects import func

    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects.lowering.outline import externalize_dispatches, outline_dispatches

    outlined = outline_dispatches(parse_mlir_text(CHAIN))
    first, second = (dispatch.symbol for dispatch in outlined.dispatches)
    selected = externalize_dispatches(outlined, [first])

    functions = {
        op.sym_name.data: op for op in selected.module.body.block.ops if isinstance(op, func.FuncOp)
    }
    assert selected.symbols == (first,)
    assert not functions[first].body.blocks
    assert functions[second].body.blocks
    assert functions["forward"].body.blocks
    assert functions[first].function_type == next(
        op.function_type
        for op in outlined.module.body.block.ops
        if isinstance(op, func.FuncOp) and op.sym_name.data == first
    )
    assert all(op.body.blocks for op in outlined.module.body.block.ops if isinstance(op, func.FuncOp))
    selected.module.verify()


def test_externalize_dispatch_rejects_unknown_duplicate_and_bad_symbol():
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects.lowering.outline import OutlineError, externalize_dispatches, outline_dispatches

    outlined = outline_dispatches(parse_mlir_text(CHAIN))
    symbol = outlined.dispatches[0].symbol
    with pytest.raises(OutlineError, match="not an outlined dispatch"):
        externalize_dispatches(outlined, ["forward"])
    with pytest.raises(OutlineError, match="unique"):
        externalize_dispatches(outlined, [symbol, symbol])
    with pytest.raises(OutlineError, match="nonempty strings"):
        externalize_dispatches(outlined, [""])


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
