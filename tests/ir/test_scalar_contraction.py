from __future__ import annotations

import hashlib
import subprocess

import numpy as np
import pytest
from corefixtures.compiler_programs import contraction_source as source
from corefixtures.compiler_programs import ordered_contraction as ordered

from merlin.compiler.lower.native.pipeline import lower_to_llvm_ir
from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.codegen import mlir_runtime_c
from merlin.llvmlower.scalar_contraction import (
    EIGHT_OUTPUTS_FEATURE,
    FEATURE,
    FOUR_OUTPUTS_FEATURE,
    RECTANGULAR_FEATURE,
    TWO_OUTPUTS_FEATURE,
    apply_for_test,
)
from merlin.llvmlower.toolchain import clang

OUTPUTS = {FEATURE: 1, TWO_OUTPUTS_FEATURE: 2, FOUR_OUTPUTS_FEATURE: 4, EIGHT_OUTPUTS_FEATURE: 8}


def run(tmp_path, src, args, shape, *, features, opt="O3", pipeline=None):
    work = tmp_path / ("selected" if features else "control") / opt
    work.mkdir(parents=True)
    llvm = lower_to_llvm_ir(src, workdir=work, features=features, pipeline=pipeline)
    path = work / "model.ll"
    path.write_text(llvm)
    shared = work / ("model_" + hashlib.sha256(llvm.encode()).hexdigest()[:12] + ".so")
    subprocess.run(
        [
            str(clang()),
            "-" + opt,
            "-fPIC",
            "-shared",
            "-ffp-contract=off",
            str(path),
            str(mlir_runtime_c()),
            "-lm",
            "-o",
            str(shared),
        ],
        check=True,
        capture_output=True,
    )
    output = np.empty(shape, dtype=np.float32)
    HostModel.load(str(shared))([(a.ctypes.data, a.shape) for a in args] + [(output.ctypes.data, output.shape)])
    return (output, llvm)


@pytest.mark.usefixtures("upstream_host_tools")
@pytest.mark.parametrize("feature,batch,transpose,reversed_add", [(FEATURE, True, True, True)])
def test_actual_native_cancellation_nonzero_init_and_transpose(tmp_path, batch, transpose, reversed_add, feature):
    rng = np.random.default_rng(9)
    prefix = (2,) if batch else ()
    width = max(4, OUTPUTS[feature])
    a = rng.normal(size=prefix + (3, 5)).astype(np.float32)
    b = rng.normal(size=prefix + (5, width)).astype(np.float32)
    c = rng.normal(size=prefix + (3, width)).astype(np.float32)
    a[..., 0, :] = np.array([16777216, 1, -16777216, 0, 0], dtype=np.float32)
    b[..., :, 0] = 1
    c[..., 0, 0] = 0
    expected = ordered(a, b, c)
    storage = np.ascontiguousarray(b.swapaxes(-1, -2)) if transpose else b
    src = source(a.shape, storage.shape, c.shape, transpose=transpose, reversed_add=reversed_add)
    transformed, count = apply_for_test(src, outputs=OUTPUTS[feature])
    for opt in ("O0", "O3"):
        actual, _ = run(tmp_path, src, [a, storage, c.copy()], c.shape, features={feature}, opt=opt)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.usefixtures("upstream_host_tools")
@pytest.mark.parametrize("feature", [FEATURE])
def test_actual_native_aliasing_input_and_initial_destination(tmp_path, feature):
    width = max(4, OUTPUTS[feature])
    a = np.arange(1, width * width + 1, dtype=np.float32).reshape(width, width)
    b = np.arange(width * width + 1, 2 * width * width + 1, dtype=np.float32).reshape(width, width) / np.float32(
        2 * width * width
    )
    original = a.copy()
    expected = ordered(original, b, original)
    src = source(a.shape, b.shape, a.shape, alias=True)
    actual, _ = run(tmp_path, src, [a, b], a.shape, features={feature})
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(a, original)


@pytest.mark.usefixtures("upstream_host_tools")
@pytest.mark.parametrize("feature", [FEATURE])
def test_separate_product_rounding_survives_actual_native_optimization(tmp_path, feature):
    a = np.array([[1.0000001192092896]], dtype=np.float32)
    width = max(4, OUTPUTS[feature])
    b = np.repeat(a, width, axis=1)
    c = np.full((1, width), -1.000000238418579, dtype=np.float32)
    actual, _ = run(tmp_path, source(a.shape, b.shape, c.shape), [a, b, c], c.shape, features={feature})
    assert np.all(actual.view(np.uint32) == 0), "A fused FMA yields a positive nonzero value here"


@pytest.mark.usefixtures("upstream_host_tools")
@pytest.mark.parametrize("feature", [FEATURE])
def test_zero_reduction_preserves_initial_destination(tmp_path, feature):
    a = np.empty((2, 0), dtype=np.float32)
    width = max(4, OUTPUTS[feature])
    b = np.empty((0, width), dtype=np.float32)
    c = np.array([[1, -0.0, 2, 3], [4, 5, 6, 7]], dtype=np.float32)
    c = np.tile(c, (1, width // 4))
    actual, _ = run(tmp_path, source(a.shape, b.shape, c.shape), [a, b, c], c.shape, features={feature})
    np.testing.assert_array_equal(actual.view(np.uint32), c.view(np.uint32))


@pytest.mark.usefixtures("upstream_host_tools")
def test_rectangular_special_values_match_compiled_source_order(tmp_path):
    a = np.array([[2139095040, 1, 2147483648], [2143363909, 8388608, 4286578688]], dtype="u4").view("f4")
    b = np.array(
        [
            [0, 1065353217, 2147483648, 2139095039, 4286578688, 2139095041, 1, 2147483649],
            [2139095039, 1065353216, 8388608, 0, 3212836864, 8388608, 2143433814, 8388608],
            [1065353216, 0, 3212836864, 8388608, 2139095040, 1, 2147483648, 1056964608],
        ],
        dtype="u4",
    ).view("f4")
    c = np.arange(16, dtype="f4").reshape(2, 8)
    src = source(a.shape, b.shape, c.shape)
    control, _ = run(tmp_path / "base", src, [a, b, c.copy()], c.shape, features={EIGHT_OUTPUTS_FEATURE})
    candidate, _ = run(tmp_path / "candidate", src, [a, b, c.copy()], c.shape, features={RECTANGULAR_FEATURE})
    nan = np.isnan(control)
    np.testing.assert_array_equal(np.isnan(candidate), nan)
    np.testing.assert_array_equal(candidate.view("u4")[~nan], control.view("u4")[~nan])
    assert candidate.view("u4")[1, 5] in (2143289344, 2143363909, 2143289345)
    assert control.view("u4")[1, 5] in (2143289344, 2143363909, 2143289345)
