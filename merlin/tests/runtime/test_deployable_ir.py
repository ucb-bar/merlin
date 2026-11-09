"""Deployable keys and compiler inputs retain the complete operation form."""

import copy
import subprocess

import numpy as np
import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import DenseArrayBase, StringAttr, i64
from xdsl.rewriter import Rewriter

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.deployable_ir import checked_generic_text
from merlin.llvmlower.deployable_outline import canonical_kernel, kernel_code_key
from merlin.xdsl_dialects._common import text

SOURCE = """builtin.module attributes {custom.module = "retained"} {
func.func @transpose(%a:tensor<2x2xf32>)->tensor<2x2xf32> {
  %e=tensor.empty():tensor<2x2xf32>
  %t=linalg.transpose ins(%a:tensor<2x2xf32>) outs(%e:tensor<2x2xf32>) permutation=[1,0]
  func.return %t:tensor<2x2xf32>
}
func.func @forward(%a:tensor<2x3xf32>,%b:tensor<3x4xf32>)->tensor<2x4xf32>
  attributes {llvm.emit_c_interface} {
  %e=tensor.empty():tensor<2x4xf32>
  %z=arith.constant 0.0:f32
  %i=linalg.fill ins(%z:f32) outs(%e:tensor<2x4xf32>)->tensor<2x4xf32>
  %m=linalg.matmul ins(%a,%b:tensor<2x3xf32>,tensor<3x4xf32>) outs(%i:tensor<2x4xf32>)->tensor<2x4xf32>
  func.return %m:tensor<2x4xf32>
}}
"""


def fixture(*, nonstandard=False):
    module = parse_mlir_text(SOURCE)
    for index, op in enumerate(module.walk()):
        if op.name in {"tensor.empty", "linalg.transpose", "linalg.matmul", "arith.mulf", "arith.addf"}:
            op.attributes["custom.obligation"] = StringAttr(op.name + str(index))
    if nonstandard:
        product = next(op for op in module.walk() if op.name == "arith.mulf")
        replacement = arith.SubfOp(*product.operands)
        replacement.attributes = copy.copy(product.attributes)
        Rewriter.replace_op(product, replacement)
    return module


def test_full_roundtrip_retains_attributes_properties_regions_and_types():
    module = fixture(nonstandard=True)
    serialized = checked_generic_text(module)
    parsed = parse_mlir_text(serialized)
    assert module.is_structurally_equivalent(parsed)
    before, after = list(module.walk()), list(parsed.walk())
    assert len(before) == len(after)
    for a, b in zip(before, after, strict=True):
        assert a.name == b.name and a.attributes == b.attributes and a.properties == b.properties
        assert [v.type for v in a.results] == [v.type for v in b.results]
        assert len(a.regions) == len(b.regions)
    assert '"arith.subf"' in serialized and '"arith.mulf"' not in serialized
    assert '"tensor.empty"' in serialized and '"linalg.transpose"' in serialized


@pytest.mark.parametrize("operation", ["tensor.empty", "linalg.transpose", "linalg.matmul", "arith.mulf"])
def test_unknown_operation_attributes_change_key(operation):
    module = fixture()
    before = kernel_code_key(module, {"compiler": "same"})
    op = next(op for op in module.walk() if op.name == operation)
    op.attributes["custom.obligation"] = StringAttr("a different contract")
    assert kernel_code_key(module, {"compiler": "same"}) != before


def test_permutation_property_and_nonstandard_named_body_change_key():
    module = fixture()
    before = kernel_code_key(module, {})
    transpose = next(op for op in module.walk() if op.name == "linalg.transpose")
    transpose.properties["permutation"] = DenseArrayBase.from_list(i64, [0, 1])
    assert kernel_code_key(module, {}) != before
    assert kernel_code_key(fixture(nonstandard=True), {}) != before


def test_custom_printer_loss_is_detected_instead_of_admitted(monkeypatch):
    from merlin.xdsl_dialects import _common

    module = fixture(nonstandard=True)
    lossy = text(module)
    assert "custom.obligation" not in text(next(op for op in module.walk() if op.name == "tensor.empty"))
    monkeypatch.setattr(_common, "text", lambda module, **kwargs: lossy)
    with pytest.raises(ValueError, match="changed attributes"):
        checked_generic_text(module)


def test_canonical_key_still_normalizes_only_names():
    a = canonical_kernel(fixture(nonstandard=True), "forward")
    b = a.clone()
    for op in b.walk():
        for value in op.results:
            value.name_hint = "newspelling"
    b = canonical_kernel(b, "merlin_canonical_kernel")
    assert kernel_code_key(a, {}) == kernel_code_key(b, {})


def test_actual_compiler_refuses_nonstandard_body_without_reconstructing_it(tmp_path):
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.compiler_api import PipelineError
    from merlin.llvmlower.kernel_backend import compile_deployable_kernel, extract_kernel

    if not toolchain.available():
        pytest.skip("explicit MLIR compiler environment unavailable")
    kernel = extract_kernel(fixture(nonstandard=True), "forward")
    # xDSL permits this body, but the native named-op verifier requires a MAC.
    # Retaining it must therefore refuse, instead of silently printing a normal
    # matmul and compiling different arithmetic.
    with pytest.raises(PipelineError, match="expected add/mul op in the body"):
        compile_deployable_kernel(
            kernel,
            tmp_path / "kernel",
            entry="forward",
            extra_cflags=("-march=rv64gcv", "-fno-vectorize", "-fno-slp-vectorize"),
        )
    upstream = parse_mlir_text((tmp_path / "kernel/model.upstream.mlir").read_text())
    sub = next(op for op in upstream.walk() if op.name == "arith.subf")
    assert "custom.obligation" in sub.attributes
    assert not (tmp_path / "kernel/kernel.o").exists()


def test_actual_compiler_retains_attributes_and_complete_standard_result(tmp_path):
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
    from merlin.llvmlower.kernel_backend import compile_deployable_kernel, extract_kernel

    if not toolchain.available():
        pytest.skip("explicit MLIR compiler environment unavailable")
    kernel = extract_kernel(fixture(), "forward")
    source = checked_generic_text(kernel)
    built = compile_deployable_kernel(
        kernel,
        tmp_path / "kernel",
        entry="forward",
        extra_cflags=("-march=rv64gcv", "-fno-vectorize", "-fno-slp-vectorize"),
    )
    assert built["source_serialization"] == "checked_generic_operation_form"
    llvm = (tmp_path / "kernel/model.ll").read_text()
    assert "fmul float" in llvm and "fsub float" not in llvm
    upstream = parse_mlir_text((tmp_path / "kernel/model.upstream.mlir").read_text())
    mul = next(op for op in upstream.walk() if op.name == "arith.mulf")
    assert "custom.obligation" in mul.attributes
    assert '"arith.mulf"' in source
    library = tmp_path / "result.so"
    subprocess.run(
        [
            str(toolchain.clang()),
            "--target=x86_64-linux-gnu",
            "-shared",
            "-fPIC",
            "-O0",
            built["llvm"],
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    model = HostModel.load(str(library), image_policy=PrivateHostImagePolicy(tmp_path))
    a = np.array([[1, 3, -2], [7, -4, 5]], dtype=np.float32)
    b = np.arange(12, dtype=np.float32).reshape(3, 4) / 4
    expected = np.zeros((2, 4), dtype=np.float32)
    for k in range(3):
        expected = np.add(expected, np.multiply(a[:, k, None], b[k, None, :], dtype=np.float32), dtype=np.float32)
    original_a, original_b = a.copy(), b.copy()
    output = np.full((2, 4), np.nan, dtype=np.float32)
    model([(v.ctypes.data, v.shape) for v in [a, b, output]])
    assert np.array_equal(output.view(np.uint32), expected.view(np.uint32))
    assert np.array_equal(a, original_a) and np.array_equal(b, original_b)
