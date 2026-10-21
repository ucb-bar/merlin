"""Exact source-order scalar contractions, including tensor alias semantics."""

from __future__ import annotations

import hashlib
import subprocess

import numpy as np
import pytest

from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.codegen import mlir_runtime_c
from merlin.llvmlower.pipeline import _upstream_pipeline, lower_to_llvm_ir
from merlin.llvmlower.scalar_contraction import (
    EIGHT_OUTPUTS_FEATURE,
    FEATURE,
    FOUR_OUTPUTS_FEATURE,
    FOUR_REDUCTION_UNROLL_FEATURE,
    MARKER,
    RECTANGULAR_FEATURE,
    RECTANGULAR_MARKER,
    REDUCTION_UNROLL_FEATURE,
    SQUARE_FEATURE,
    SQUARE_MARKER,
    TWO_OUTPUTS_FEATURE,
    WIDE_RECTANGULAR_FEATURE,
    WIDE_RECTANGULAR_MARKER,
    apply_for_test,
)
from merlin.llvmlower.toolchain import clang

OUTPUTS = {FEATURE: 1, TWO_OUTPUTS_FEATURE: 2, FOUR_OUTPUTS_FEATURE: 4, EIGHT_OUTPUTS_FEATURE: 8}


def source(a_shape, b_shape, c_shape, *, transpose=False, alias=False, fastmath=False, reversed_add=False):
    rank = len(c_shape)
    dims = [f"d{i}" for i in range(rank + 1)]
    batch = dims[:-3]
    i, j, k = dims[-3:]
    maps = [batch + [i, k], batch + ([j, k] if transpose else [k, j]), batch + [i, j]]
    amap = ",".join(f"affine_map<({','.join(dims)})->({','.join(m)})>" for m in maps)
    iters = ",".join([*(['"parallel"'] * rank), '"reduction"'])

    def typ(shape):
        return "tensor<" + "x".join(map(str, shape)) + "xf32>"

    ta, tb, tc = map(typ, [a_shape, b_shape, c_shape])
    arguments = f"%a: {ta}, %b: {tb}" + ("" if alias else f", %c: {tc}")
    initial = "%a" if alias else "%c"
    flags = " fastmath<contract>" if fastmath else ""
    add_args = "%z,%p" if reversed_add else "%p,%z"
    return f"""module {{ func.func @forward({arguments}) -> {tc} attributes {{llvm.emit_c_interface}} {{
      %r = linalg.generic {{indexing_maps=[{amap}], iterator_types=[{iters}]}}
      ins(%a,%b:{ta},{tb}) outs({initial}:{tc}) {{
      ^bb0(%x:f32,%y:f32,%z:f32):
        %p = arith.mulf %x,%y{flags}:f32
        %s = arith.addf {add_args}:f32
        linalg.yield %s:f32
      }} -> {tc}
      return %r:{tc}
    }} }}"""


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
    return output, llvm


def ordered(a, b, c):
    result = c.copy()
    for k in range(a.shape[-1]):
        product = a[..., :, k, None] * b[..., k, None, :]
        result = np.float32(product + result)
    return result


@pytest.mark.parametrize("feature", OUTPUTS)
@pytest.mark.parametrize(
    "batch,transpose,reversed_add",
    [(False, False, False), (False, True, True), (True, False, False), (True, True, True)],
)
def test_actual_native_cancellation_nonzero_init_and_transpose(tmp_path, batch, transpose, reversed_add, feature):
    rng = np.random.default_rng(9)
    prefix = (2,) if batch else ()
    width = max(4, OUTPUTS[feature])
    a = rng.normal(size=prefix + (3, 5)).astype(np.float32)
    b = rng.normal(size=prefix + (5, width)).astype(np.float32)
    c = rng.normal(size=prefix + (3, width)).astype(np.float32)
    # A strict cancellation witness differs after reduction reassociation.
    a[..., 0, :] = np.array([16777216, 1, -16777216, 0, 0], dtype=np.float32)
    b[..., :, 0] = 1
    c[..., 0, 0] = 0
    expected = ordered(a, b, c)
    storage = np.ascontiguousarray(b.swapaxes(-1, -2)) if transpose else b
    src = source(a.shape, storage.shape, c.shape, transpose=transpose, reversed_add=reversed_add)
    transformed, count = apply_for_test(src, outputs=OUTPUTS[feature])
    result_types = "-> (" + ", ".join(["f32"] * OUTPUTS[feature]) + ")"
    assert count == 1 and result_types in transformed and "linalg.generic" not in transformed
    for opt in ("O0", "O3"):
        actual, llvm = run(tmp_path, src, [a, storage, c.copy()], c.shape, features={feature}, opt=opt)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        assert "fmul float" in llvm and "fadd float" in llvm
        assert "llvm.fma" not in llvm and "fmul contract" not in llvm


@pytest.mark.parametrize("feature", OUTPUTS)
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


@pytest.mark.parametrize("feature", OUTPUTS)
def test_separate_product_rounding_survives_actual_native_optimization(tmp_path, feature):
    a = np.array([[1.0000001192092896]], dtype=np.float32)
    width = max(4, OUTPUTS[feature])
    b = np.repeat(a, width, axis=1)
    c = np.full((1, width), -1.000000238418579, dtype=np.float32)
    actual, _ = run(tmp_path, source(a.shape, b.shape, c.shape), [a, b, c], c.shape, features={feature})
    assert np.all(actual.view(np.uint32) == 0), "A fused FMA yields a positive nonzero value here"


@pytest.mark.parametrize("feature", OUTPUTS)
def test_zero_reduction_preserves_initial_destination(tmp_path, feature):
    a = np.empty((2, 0), dtype=np.float32)
    width = max(4, OUTPUTS[feature])
    b = np.empty((0, width), dtype=np.float32)
    c = np.array([[1, -0.0, 2, 3], [4, 5, 6, 7]], dtype=np.float32)
    c = np.tile(c, (1, width // 4))
    actual, _ = run(tmp_path, source(a.shape, b.shape, c.shape), [a, b, c], c.shape, features={feature})
    np.testing.assert_array_equal(actual.view(np.uint32), c.view(np.uint32))


def test_fastmath_is_left_unchanged():
    src = source((2, 3), (3, 4), (2, 4), fastmath=True)
    transformed, count = apply_for_test(src)
    assert count == 0 and "linalg.generic" in transformed


@pytest.mark.parametrize("change", ["dynamic", "constant_map", "extra_body_op"])
def test_unsupported_patterns_are_left_unchanged(change):
    src = source((2, 3), (3, 4), (2, 4))
    if change == "dynamic":
        src = src.replace("tensor<2x3xf32>", "tensor<?x3xf32>")
    elif change == "constant_map":
        src = src.replace("tensor<2x3xf32>", "tensor<1x3xf32>").replace(
            "affine_map<(d0,d1,d2)->(d0,d2)>", "affine_map<(d0,d1,d2)->(0,d2)>"
        )
    else:
        src = src.replace("linalg.yield %s:f32", "%v = math.absf %s:f32\n linalg.yield %v:f32")
    transformed, count = apply_for_test(src)
    assert count == 0 and "linalg.generic" in transformed


@pytest.mark.parametrize("feature", OUTPUTS)
def test_permuted_output_map_preserves_initial_destination(tmp_path, feature):
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    width = max(4, OUTPUTS[feature])
    b = np.arange(3 * width, dtype=np.float32).reshape(3, width)
    c = np.arange(2 * width, dtype=np.float32).reshape(width, 2)
    expected = ordered(a, b, c.T.copy()).T.copy()
    src = source(a.shape, b.shape, c.shape).replace(
        "affine_map<(d0,d1,d2)->(d0,d1)>", "affine_map<(d0,d1,d2)->(d1,d0)>"
    )
    actual, _ = run(tmp_path, src, [a, b, c], c.shape, features={feature})
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_marker_is_before_bufferization_and_default_pipeline_is_unchanged():
    baseline = _upstream_pipeline(frozenset())
    selected = _upstream_pipeline(frozenset({FEATURE}))
    assert selected.replace(MARKER + ",", "") == baseline
    assert selected.index(MARKER) < selected.index("one-shot-bufferize")
    assert selected.index("linalg-fuse-elementwise-ops") < selected.index(MARKER)
    assert selected.index("linalg-generalize-named-ops") < selected.index(MARKER)


@pytest.mark.parametrize("feature", [*OUTPUTS, RECTANGULAR_FEATURE])
def test_no_matching_contraction_preserves_prepared_view_lowering(tmp_path, feature):
    # A scalar schedule must not generalize these named materializations before
    # ordinary elementwise fusion. The read-only callbacks may observe their
    # buffers, so a numerical-output check alone misses changed borrow admission.
    src = """module {
      func.func private @remember(memref<3x2xf32> {bufferization.access = "read"})
      func.func private @observe(memref<3x2xf32> {bufferization.access = "read"})
      func.func private @prepare(%x: tensor<3x2xf32> {bufferization.access = "read"}) {
        %b = bufferization.to_buffer %x read_only : tensor<3x2xf32> to memref<3x2xf32>
        func.call @remember(%b) : (memref<3x2xf32>) -> ()
        return
      }
      func.func private @use(%x: tensor<3x2xf32> {bufferization.access = "read"}) {
        %b = bufferization.to_buffer %x read_only : tensor<3x2xf32> to memref<3x2xf32>
        func.call @observe(%b) : (memref<3x2xf32>) -> ()
        return
      }
      func.func @forward(%x: tensor<2x3xf32>) -> tensor<3x2xf32>
          attributes {llvm.emit_c_interface} {
        %e = tensor.empty() : tensor<3x2xf32>
        %t = linalg.transpose ins(%x : tensor<2x3xf32>)
          outs(%e : tensor<3x2xf32>) permutation=[1,0]
        %e2 = tensor.empty() : tensor<3x2xf32>
        %shared = linalg.copy ins(%t : tensor<3x2xf32>)
          outs(%e2 : tensor<3x2xf32>) -> tensor<3x2xf32>
        func.call @prepare(%shared) : (tensor<3x2xf32>) -> ()
        func.call @use(%shared) : (tensor<3x2xf32>) -> ()
        func.call @use(%shared) : (tensor<3x2xf32>) -> ()
        return %shared : tensor<3x2xf32>
      }
    }"""
    control = lower_to_llvm_ir(src, workdir=tmp_path / "control", features=set())
    selected = lower_to_llvm_ir(src, workdir=tmp_path / "selected", features={feature})

    def without_module_id(text):
        return "\n".join(line for line in text.splitlines() if not line.startswith("; ModuleID = "))

    assert without_module_id(control) == without_module_id(selected)


@pytest.mark.parametrize("tiled", [TWO_OUTPUTS_FEATURE, FOUR_OUTPUTS_FEATURE, EIGHT_OUTPUTS_FEATURE])
def test_accumulator_schedules_are_explicit_alternatives(tiled):
    with pytest.raises(ValueError, match="alternatives"):
        _upstream_pipeline(frozenset({FEATURE, tiled}))


@pytest.mark.parametrize("outputs", [2, 4, 8])
def test_tiled_output_schedule_refuses_partial_tiles(outputs):
    src = source((2, 3), (3, 5), (2, 5))
    transformed, count = apply_for_test(src, outputs=outputs)
    assert count == 0 and "linalg.generic" in transformed


@pytest.mark.parametrize("hint", [REDUCTION_UNROLL_FEATURE, FOUR_REDUCTION_UNROLL_FEATURE])
def test_reduction_unroll_requires_a_scalar_accumulator_schedule(hint):
    with pytest.raises(ValueError, match="requires exactly one"):
        _upstream_pipeline(frozenset({hint}))


@pytest.mark.parametrize("feature", OUTPUTS)
@pytest.mark.parametrize("hint", [REDUCTION_UNROLL_FEATURE, FOUR_REDUCTION_UNROLL_FEATURE])
def test_schedule_metadata_describes_valid_scalar_choices(feature, hint):
    from merlin.llvmlower import impr_features as I

    selected = I.normalize({feature, hint})
    assert selected == frozenset({feature, hint})
    assert I.get(hint).requires_exactly_one_of == frozenset(OUTPUTS) | {
        RECTANGULAR_FEATURE,
        SQUARE_FEATURE,
        WIDE_RECTANGULAR_FEATURE,
    }
    assert I.get(feature).alternative_group == I.get(FEATURE).alternative_group
    assert I.get(hint).alternative_group != I.get(feature).alternative_group


@pytest.mark.parametrize("batch,transpose,reversed_add", [(False, False, False), (True, True, True)])
@pytest.mark.parametrize("feature,rows", [(RECTANGULAR_FEATURE, 2), (SQUARE_FEATURE, 4)])
def test_rectangular_actual_native_nonzero_init_cancellation_and_input_liveness(
    tmp_path, batch, transpose, reversed_add, feature, rows
):
    rng = np.random.default_rng(31)
    prefix = (3,) if batch else ()
    a = rng.normal(size=prefix + (4, 7)).astype(np.float32)
    b = rng.normal(size=prefix + (7, 12)).astype(np.float32)
    c = rng.normal(size=prefix + (4, 12)).astype(np.float32)
    a[..., 0, :] = [16777216, 1, -16777216, 0, 0, 0, 0]
    b[..., :, 0] = 1
    c[..., 0, 0] = 0
    storage = np.ascontiguousarray(b.swapaxes(-1, -2)) if transpose else b
    untouched = [x.copy() for x in (a, storage, c)]
    src = source(a.shape, storage.shape, c.shape, transpose=transpose, reversed_add=reversed_add)
    transformed, count = apply_for_test(src, outputs=4, rows=rows, reduction_unroll=2)
    assert count == 1 and f"_{rows}x4_outputs" in transformed
    assert "-> (" + ", ".join(["f32"] * (rows * 4)) + ")" in transformed
    expected = ordered(a, b, c)
    actual, llvm = run(tmp_path, src, [a, storage, c.copy()], c.shape, features={feature, REDUCTION_UNROLL_FEATURE})
    np.testing.assert_array_equal(actual.view("u4"), expected.view("u4"))
    for value, before in zip((a, storage), untouched[:2]):
        np.testing.assert_array_equal(value.view("u4"), before.view("u4"))
    assert "llvm.fma" not in llvm


@pytest.mark.parametrize("feature", [RECTANGULAR_FEATURE, SQUARE_FEATURE])
def test_rectangular_actual_native_source_destination_alias(tmp_path, feature):
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    b = np.arange(16, dtype=np.float32).reshape(4, 4) / np.float32(31)
    before = a.copy()
    actual, _ = run(tmp_path, source(a.shape, b.shape, a.shape, alias=True), [a, b], a.shape, features={feature})
    np.testing.assert_array_equal(actual.view("u4"), ordered(before, b, before).view("u4"))
    np.testing.assert_array_equal(a, before)


@pytest.mark.parametrize("k", [0, 1, 3])
@pytest.mark.parametrize("feature,rows", [(RECTANGULAR_FEATURE, 2), (SQUARE_FEATURE, 4)])
def test_rectangular_actual_native_empty_reduction_and_signed_zero(tmp_path, k, feature, rows):
    a = np.ones((rows, k), dtype=np.float32)
    b = np.ones((k, 4), dtype=np.float32)
    c = np.array([[1, -0.0, 2, 3], [4, 5, 6, 7]], dtype=np.float32)
    c = np.tile(c, (rows // 2, 1))
    expected = ordered(a, b, c)
    actual, _ = run(tmp_path, source(a.shape, b.shape, c.shape), [a, b, c.copy()], c.shape, features={feature})
    np.testing.assert_array_equal(actual.view("u4"), expected.view("u4"))


@pytest.mark.parametrize("change", ["odd_rows", "column_tail", "fastmath", "strictfp", "shared_row_input"])
@pytest.mark.parametrize("rows", [2, 4])
def test_rectangular_refuses_unproved_tiles_and_contracts(change, rows):
    src = source((rows, 3), (3, 4), (rows, 4))
    if change == "odd_rows":
        src = source((3, 3), (3, 4), (3, 4))
    elif change == "column_tail":
        src = source((rows, 3), (3, 5), (rows, 5))
    elif change == "fastmath":
        src = source((rows, 3), (3, 4), (rows, 4), fastmath=True)
    elif change == "strictfp":
        src = src.replace("llvm.emit_c_interface", "llvm.emit_c_interface, strictfp")
    else:
        src = src.replace("tensor<3x4xf32>", f"tensor<{rows}x3xf32>").replace(
            "affine_map<(d0,d1,d2)->(d2,d1)>", "affine_map<(d0,d1,d2)->(d0,d2)>"
        )
    transformed, count = apply_for_test(src, outputs=4, rows=rows)
    assert count == 0 and "linalg.generic" in transformed


@pytest.mark.parametrize("feature,marker", [(RECTANGULAR_FEATURE, RECTANGULAR_MARKER), (SQUARE_FEATURE, SQUARE_MARKER)])
def test_rectangular_default_pipeline_identity_and_alternative_refusal(feature, marker):
    baseline = _upstream_pipeline(frozenset())
    selected = _upstream_pipeline(frozenset({feature}))
    assert selected.replace(marker + ",", "") == baseline
    for other in set(OUTPUTS) | {RECTANGULAR_FEATURE, SQUARE_FEATURE} - {feature}:
        with pytest.raises(ValueError, match="alternatives"):
            _upstream_pipeline(frozenset({feature, other}))


@pytest.mark.parametrize("feature", [RECTANGULAR_FEATURE, SQUARE_FEATURE])
def test_rectangular_permuted_output_coordinates(tmp_path, feature):
    a = np.arange(12, dtype="f4").reshape(4, 3)
    b = np.arange(24, dtype="f4").reshape(3, 8) / np.float32(17)
    c = np.arange(32, dtype="f4").reshape(8, 4)
    src = source(a.shape, b.shape, c.shape).replace(
        "affine_map<(d0,d1,d2)->(d0,d1)>", "affine_map<(d0,d1,d2)->(d1,d0)>"
    )
    expected = ordered(a, b, c.T.copy()).T.copy()
    actual, _ = run(tmp_path, src, [a, b, c.copy()], c.shape, features={feature})
    np.testing.assert_array_equal(actual.view("u4"), expected.view("u4"))


def test_rectangular_live_original_initializer_is_preserved_by_upstream(tmp_path):
    a = np.arange(12, dtype="f4").reshape(4, 3) / np.float32(13)
    b = np.arange(24, dtype="f4").reshape(3, 8) / np.float32(17)
    c = np.arange(32, dtype="f4").reshape(4, 8) / np.float32(7)
    src = source(a.shape, b.shape, c.shape)
    src = src.replace(
        "return %r:tensor<4x8xf32>",
        """
      %empty = tensor.empty():tensor<4x8xf32>
      %observed = linalg.generic {
        indexing_maps=[affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>,
          affine_map<(d0,d1)->(d0,d1)>], iterator_types=["parallel","parallel"]}
      ins(%r,%c:tensor<4x8xf32>,tensor<4x8xf32>) outs(%empty:tensor<4x8xf32>) {
      ^bb0(%x:f32,%y:f32,%unused:f32):
        %v = arith.addf %x,%y:f32
        linalg.yield %v:f32
      } -> tensor<4x8xf32>
      return %observed:tensor<4x8xf32>""",
    )
    expected = np.float32(ordered(a, b, c) + c)
    actual, _ = run(tmp_path, src, [a, b, c.copy()], c.shape, features={RECTANGULAR_FEATURE})
    np.testing.assert_array_equal(actual.view("u4"), expected.view("u4"))


@pytest.mark.parametrize("shape", [(0, 8), (2, 0)])
def test_rectangular_empty_parallel_extent_does_not_evaluate_reduction(tmp_path, shape):
    m, n = shape
    a = np.full((m, 3), np.inf, dtype="f4")
    b = np.zeros((3, n), dtype="f4")
    c = np.empty(shape, dtype="f4")
    actual, _ = run(tmp_path, source(a.shape, b.shape, c.shape), [a, b, c], c.shape, features={RECTANGULAR_FEATURE})
    assert actual.shape == shape and actual.size == 0


def test_rectangular_special_values_match_compiled_source_order(tmp_path):
    a = np.array([[0x7F800000, 0x00000001, 0x80000000], [0x7FC12345, 0x00800000, 0xFF800000]], dtype="u4").view("f4")
    b = np.array(
        [
            [0x00000000, 0x3F800001, 0x80000000, 0x7F7FFFFF, 0xFF800000, 0x7F800001, 0x00000001, 0x80000001],
            [0x7F7FFFFF, 0x3F800000, 0x00800000, 0x00000000, 0xBF800000, 0x00800000, 0x7FC23456, 0x00800000],
            [0x3F800000, 0x00000000, 0xBF800000, 0x00800000, 0x7F800000, 0x00000001, 0x80000000, 0x3F000000],
        ],
        dtype="u4",
    ).view("f4")
    c = np.arange(16, dtype="f4").reshape(2, 8)
    src = source(a.shape, b.shape, c.shape)
    control, _ = run(tmp_path / "base", src, [a, b, c.copy()], c.shape, features={EIGHT_OUTPUTS_FEATURE})
    candidate, _ = run(tmp_path / "candidate", src, [a, b, c.copy()], c.shape, features={RECTANGULAR_FEATURE})
    # Unconstrained arith.mulf/addf preserve the numeric operation order, but do
    # not promise which input NaN payload is propagated. LLVM LangRef permits
    # quiet propagation from either NaN input (and canonical NaNs). A register
    # reuse choice can therefore change a payload without changing source
    # semantics. Non-NaN words remain exact; empty-K bit preservation is checked
    # separately because it performs no arithmetic on the source initializer.
    nan = np.isnan(control)
    np.testing.assert_array_equal(np.isnan(candidate), nan)
    np.testing.assert_array_equal(candidate.view("u4")[~nan], control.view("u4")[~nan])
    assert candidate.view("u4")[1, 5] in (0x7FC00000, 0x7FC12345, 0x7FC00001)
    assert control.view("u4")[1, 5] in (0x7FC00000, 0x7FC12345, 0x7FC00001)


def test_runtime_guards_still_reject_invalid_marker_composition():
    from merlin.llvmlower.scalar_contraction import _edit_pipeline, _edit_reduction_unroll

    with pytest.raises(ValueError, match="alternatives"):
        _edit_pipeline([MARKER], outputs=8)
    with pytest.raises(ValueError, match="requires one scalar accumulator"):
        _edit_reduction_unroll([])


def test_reduction_unroll_counts_are_explicit_alternatives():
    with pytest.raises(ValueError, match="alternatives"):
        _upstream_pipeline(frozenset({FOUR_OUTPUTS_FEATURE, REDUCTION_UNROLL_FEATURE, FOUR_REDUCTION_UNROLL_FEATURE}))


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("hint,count", [(REDUCTION_UNROLL_FEATURE, 2), (FOUR_REDUCTION_UNROLL_FEATURE, 4)])
def test_reduction_unroll_survives_actual_bufferization_and_native(tmp_path, fused, hint, count):
    a = np.array([[1e20, 1, -1e20, 3, -2, 5, -4, 6]], dtype=np.float32)
    b = np.ones((8, 4), dtype=np.float32)
    b[:, 1] = np.float32(1.0000001192092896)
    c = np.array([[7, -1.000000238418579, 9, -0.0]], dtype=np.float32)
    features = {FOUR_OUTPUTS_FEATURE, hint}
    if fused:
        features.add("fuse_activation_polynomial_fma")
    actual, llvm = run(tmp_path, source(a.shape, b.shape, c.shape), [a, b, c], c.shape, features=features)
    np.testing.assert_array_equal(actual.view(np.uint32), ordered(a, b, c).view(np.uint32))
    assert '!{!"llvm.loop.unroll.count", i32 ' + str(count) + "}" in llvm
    assert "fmul float" in llvm and "fadd float" in llvm and "llvm.fma" not in llvm


@pytest.mark.parametrize("feature", OUTPUTS)
def test_runner_composes_with_fused_activation_and_exact_fma(tmp_path, feature):
    a = np.ones((2, 3), dtype=np.float32)
    width = max(4, OUTPUTS[feature])
    b = np.ones((3, width), dtype=np.float32)
    c = np.zeros((2, width), dtype=np.float32)
    actual, _ = run(
        tmp_path,
        source(a.shape, b.shape, c.shape),
        [a, b, c],
        c.shape,
        features={feature, "fuse_activation_polynomial_fma"},
    )
    np.testing.assert_array_equal(actual, np.full(c.shape, 3, dtype=np.float32))


@pytest.mark.parametrize("feature", OUTPUTS)
def test_named_matmul_is_generalized_before_the_rewrite(tmp_path, feature):
    src = """module {
      func.func @forward(%a: tensor<2x3xf32>, %b: tensor<3x4xf32>, %c: tensor<2x4xf32>)
          -> tensor<2x4xf32> attributes {llvm.emit_c_interface} {
        %r = linalg.matmul ins(%a,%b:tensor<2x3xf32>,tensor<3x4xf32>)
          outs(%c:tensor<2x4xf32>) -> tensor<2x4xf32>
        return %r:tensor<2x4xf32>
      }
    }"""
    width = max(4, OUTPUTS[feature])
    src = src.replace("x4xf32>", "x" + str(width) + "xf32>")
    a, b, c = [np.ones(shape, dtype=np.float32) for shape in [(2, 3), (3, width), (2, width)]]
    actual, llvm = run(tmp_path, src, [a, b, c], c.shape, features={feature})
    np.testing.assert_array_equal(actual, np.full(c.shape, 4, dtype=np.float32))
    assert "phi float" in llvm


@pytest.mark.parametrize("feature", OUTPUTS)
def test_scalarize_runner_dispatches_the_marker_in_its_first_stage(tmp_path, feature):
    from merlin.llvmlower.accum_microkernel import SCALARIZE_MARKER

    pipeline = _upstream_pipeline(frozenset({feature})).replace(
        "one-shot-bufferize", SCALARIZE_MARKER + ",one-shot-bufferize", 1
    )
    width = max(4, OUTPUTS[feature])
    a, b, c = [np.ones(shape, dtype=np.float32) for shape in [(2, 3), (3, width), (2, width)]]
    actual, llvm = run(
        tmp_path, source(a.shape, b.shape, c.shape), [a, b, c], c.shape, features={feature}, pipeline=pipeline
    )
    np.testing.assert_array_equal(actual, np.full(c.shape, 4, dtype=np.float32))
    assert "phi float" in llvm


@pytest.mark.parametrize(
    "batch,transpose,reversed_add", [(False, False, False), (True, False, True), (False, True, True)]
)
def test_two_by_eight_preserves_source_order_and_views(tmp_path, batch, transpose, reversed_add):
    batch_shape = (2,) if batch else ()
    a = (
        np.linspace(-0.5, 0.5, 48, dtype=np.float32).reshape(*batch_shape, 4, 6)
        if batch
        else np.linspace(-0.5, 0.5, 24, dtype=np.float32).reshape(4, 6)
    )
    b = np.linspace(-1, 1, (2 if batch else 1) * 6 * 16, dtype=np.float32).reshape(*batch_shape, 6, 16)
    c = np.zeros((*batch_shape, 4, 16), dtype=np.float32)
    physical = np.swapaxes(b, -2, -1).copy() if transpose else b
    src = source(a.shape, physical.shape, c.shape, transpose=transpose, reversed_add=reversed_add)
    actual, _ = run(tmp_path, src, [a, physical, c.copy()], c.shape, features={WIDE_RECTANGULAR_FEATURE})
    expected = ordered(a, b, c)
    np.testing.assert_array_equal(actual.view("u4"), expected.view("u4"))
    transformed, count = apply_for_test(src, outputs=8, rows=2)
    assert count == 1 and "_2x8_outputs" in transformed


def test_two_by_eight_is_explicit_and_keeps_partial_tile_fallback():
    src = source((3, 6), (6, 15), (3, 15))
    transformed, count = apply_for_test(src, outputs=8, rows=2)
    assert count == 0 and "linalg.generic" in transformed
    passes = _upstream_pipeline(frozenset({WIDE_RECTANGULAR_FEATURE}))
    assert WIDE_RECTANGULAR_MARKER in passes
