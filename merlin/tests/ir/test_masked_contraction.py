"""Independent complete-use and source arithmetic tests for mask scheduling."""

from __future__ import annotations

import ctypes
import hashlib
import subprocess

import numpy as np
import pytest

from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.codegen import mlir_runtime_c
from merlin.llvmlower.masked_contraction import MaskEffectContract, apply_for_test
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.toolchain import clang

EFFECTS = MaskEffectContract(nontrapping=True, floating_flags_unobserved=True)


def source(*, batch=2, m=4, n=8, k=5, extra_use=False, invert=True, scale=True, strict=False, unsupported_view=False):
    a, b, c = (f"tensor<{batch}x{m}x{k}xf32>", f"tensor<{batch}x{k}x{n}xf32>", f"tensor<{batch}x{m}x{n}xf32>")
    out = f"tensor<1x{batch}x{m}x{n}xf32>"
    mask = f"tensor<1x1x{m}x{n}xi1>"
    rank4map = "affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>"
    scalar = "affine_map<(d0,d1,d2,d3)->()>"
    attr = "llvm.strictfp," if strict else ""
    viewed = (
        f"%v = tensor.reshape %r(%shape): ({c}, tensor<4xindex>) -> {out}"
        if unsupported_view
        else f"""%flat = tensor.collapse_shape %r [[0,1,2]]:{c} into tensor<{batch * m * n}xf32>
      %v = tensor.expand_shape %flat [[0,1,2,3]] output_shape [1,{batch},{m},{n}]:tensor<{batch * m * n}xf32> into {out}"""
    )
    inverted = (
        f"""%mask0 = tensor.empty():{mask}
      %inv = linalg.generic {{indexing_maps=[{rank4map},{rank4map}],iterator_types=["parallel","parallel","parallel","parallel"]}}
        ins(%mask:{mask}) outs(%mask0:{mask}) {{
        ^bb0(%x:i1,%unused:i1):
          %true = arith.constant true
          %not = arith.xori %x,%true:i1
          linalg.yield %not:i1
      }} -> {mask}"""
        if invert
        else ""
    )
    select_choices = "%fill,%x" if invert else "%x,%fill"
    selected_mask = "%inv" if invert else "%mask"
    return_type = f"({out},{c})" if extra_use else out
    returned = f"return %selected,%r:{out},{c}" if extra_use else f"return %selected:{out}"
    mul = "%p = arith.mulf %x,%s:f32" if scale else "%p = arith.divf %x,%s:f32"
    extra_arg = ",%shape:tensor<4xindex>" if unsupported_view else ""
    return f"""module {{ func.func @forward(%a:{a},%b:{b},%c:{c} {{bufferization.writable = false}},%mask:{mask}{extra_arg}) -> {return_type}
      attributes {{{attr}llvm.emit_c_interface}} {{
      %r = linalg.generic {{indexing_maps=[affine_map<(d0,d1,d2,d3)->(d0,d1,d3)>,affine_map<(d0,d1,d2,d3)->(d0,d3,d2)>,affine_map<(d0,d1,d2,d3)->(d0,d1,d2)>],iterator_types=["parallel","parallel","parallel","reduction"]}}
        ins(%a,%b:{a},{b}) outs(%c:{c}) {{
        ^bb0(%x:f32,%y:f32,%seed:f32):
          %p = arith.mulf %x,%y:f32
          %sum = arith.addf %p,%seed:f32
          linalg.yield %sum:f32
      }} -> {c}
      {viewed}
      %scale = arith.constant 0.125:f32
      %scale_tensor = tensor.splat %scale:tensor<f32>
      %empty = tensor.empty():{out}
      %scaled = linalg.generic {{indexing_maps=[{rank4map},{scalar},{rank4map}],iterator_types=["parallel","parallel","parallel","parallel"]}}
        ins(%v,%scale_tensor:{out},tensor<f32>) outs(%empty:{out}) {{
        ^bb0(%x:f32,%s:f32,%unused:f32):
          {mul}
          linalg.yield %p:f32
      }} -> {out}
      {inverted}
      %inf = arith.constant 0xFF800000:f32
      %fill_tensor = tensor.splat %inf:tensor<f32>
      %out0 = tensor.empty():{out}
      %selected = linalg.generic {{indexing_maps=[affine_map<(d0,d1,d2,d3)->(d0,0,d2,d3)>,{scalar},{rank4map},{rank4map}],iterator_types=["parallel","parallel","parallel","parallel"]}}
        ins({selected_mask},%fill_tensor,%scaled:{mask},tensor<f32>,{out}) outs(%out0:{out}) {{
        ^bb0(%condition:i1,%fill:f32,%x:f32,%unused:f32):
          %selected = arith.select %condition,{select_choices}:f32
          linalg.yield %selected:f32
      }} -> {out}
      {returned}
    }} }}"""


def test_effect_permission_is_explicit():
    for effects in (MaskEffectContract(False, True), MaskEffectContract(True, False)):
        with pytest.raises(ValueError, match="explicit"):
            apply_for_test(source(), effects=effects)


@pytest.mark.parametrize(
    "mutation", [dict(extra_use=True), dict(scale=False), dict(strict=True), dict(unsupported_view=True)]
)
def test_complete_use_or_precision_refusal(mutation):
    result, count = apply_for_test(source(**mutation), effects=EFFECTS, outputs=4, rows=2)
    assert count == 0 and "scf.if" not in result


@pytest.mark.parametrize(
    "kwargs,outputs,rows",
    [(dict(), 4, 2), (dict(batch=3, m=3, n=5), 1, 1), (dict(k=0), 4, 2), (dict(invert=False), 4, 2)],
)
def test_typed_source_observer_rewrite(kwargs, outputs, rows):
    result, count = apply_for_test(source(**kwargs), effects=EFFECTS, outputs=outputs, rows=rows)
    assert count == 1 and "scf.if" in result
    if outputs * rows > 1:
        assert "arith.ori" in result
    assert "arith.mulf" in result and "arith.addf" in result and "math.fma" not in result


def compile_native(path, text):
    path.mkdir()
    llvm = lower_to_llvm_ir(text, workdir=path)
    (path / "model.ll").write_text(llvm)
    shared = path / ("model_" + hashlib.sha256(llvm.encode()).hexdigest()[:16] + ".so")
    subprocess.run(
        [
            str(clang()),
            "-O2",
            "-fPIC",
            "-shared",
            "-ffp-contract=off",
            str(path / "model.ll"),
            str(mlir_runtime_c()),
            "-lm",
            "-o",
            str(shared),
        ],
        check=True,
        capture_output=True,
    )
    return HostModel.load(str(shared))


@pytest.mark.parametrize("m,n,k,outputs,rows", [(4, 8, 5, 4, 2), (3, 5, 7, 1, 1), (4, 8, 0, 4, 2)])
def test_actual_compiled_source_random_masks_modes(tmp_path, m, n, k, outputs, rows):
    original = source(m=m, n=n, k=k)
    selected, count = apply_for_test(original, effects=EFFECTS, outputs=outputs, rows=rows)
    assert count == 1
    control = compile_native(tmp_path / "control", original)
    candidate = compile_native(tmp_path / "selected", selected)
    rng = np.random.default_rng(723)
    arrays = [rng.standard_normal(shape).astype(np.float32) for shape in ((2, m, k), (2, k, n), (2, m, n))]
    arrays[2].reshape(-1)[::3] = np.float32(-0.0)
    env = ctypes.CDLL(None)
    assert env.fegetround() == 0
    try:
        for mode in (0, 0x400, 0x800, 0xC00):
            assert env.fesetround(mode) == 0
            for mask in (
                np.zeros((1, 1, m, n), np.bool_),
                np.ones((1, 1, m, n), np.bool_),
                rng.integers(0, 2, (1, 1, m, n), dtype=np.uint8).astype(np.bool_),
            ):
                args = [*arrays, mask]
                padded = []
                for array in args:
                    storage = np.full(array.size + 128, 73, array.dtype)
                    storage[64:-64] = array.reshape(-1)
                    padded.append((storage, storage[64:-64].reshape(array.shape)))
                args = [array for _, array in padded]
                before = [a.tobytes() for a in args]
                output_storage = [np.full(2 * m * n + 128, -73, np.float32) for _ in range(2)]
                reference, result = [storage[64:-64].reshape((1, 2, m, n)) for storage in output_storage]
                control([(a.ctypes.data, a.shape) for a in [*args, reference]])
                candidate([(a.ctypes.data, a.shape) for a in [*args, result]])
                assert np.array_equal(reference.view(np.uint32), result.view(np.uint32))
                assert before == [a.tobytes() for a in args]
                for storage, _ in padded:
                    assert np.all(storage[:64] == np.array(73, dtype=storage.dtype))
                    assert np.all(storage[-64:] == np.array(73, dtype=storage.dtype))
                for storage in output_storage:
                    assert np.all(storage[:64] == -73) and np.all(storage[-64:] == -73)
    finally:
        assert env.fesetround(0) == 0


def test_output_tile_types_refuse_before_toolchain():
    for outputs, rows in ((True, 1), (4, True), (4.0, 2), (4, 2.0)):
        with pytest.raises(ValueError, match="tile"):
            apply_for_test(source(), effects=EFFECTS, outputs=outputs, rows=rows)
    with pytest.raises(ValueError, match="MaskEffectContract"):
        apply_for_test(source(), effects=None)


def test_normal_pipeline_requires_permission_and_scalar_policy(tmp_path):
    from merlin.llvmlower.masked_contraction import FEATURE
    from merlin.llvmlower.pipeline import PipelineError
    from merlin.llvmlower.scalar_contraction import RECTANGULAR_FEATURE

    with pytest.raises(ValueError, match="requires exactly one"):
        lower_to_llvm_ir(
            source(),
            workdir=tmp_path / "missing_schedule",
            features=frozenset({FEATURE}),
            masked_contraction_effects=EFFECTS,
        )
    with pytest.raises(PipelineError, match="explicit MaskEffectContract"):
        lower_to_llvm_ir(
            source(), workdir=tmp_path / "missing_effects", features=frozenset({FEATURE, RECTANGULAR_FEATURE})
        )
    with pytest.raises(PipelineError, match="without masked"):
        lower_to_llvm_ir(source(), workdir=tmp_path / "unused_effects", masked_contraction_effects=EFFECTS)


@pytest.mark.parametrize("integer_softmax", [False, True])
def test_normal_pipeline_emits_identical_selected_llvm(tmp_path, integer_softmax):
    import json

    from merlin.llvmlower import int_softmax_table
    from merlin.llvmlower.masked_contraction import ARGV_INDEX, FEATURE
    from merlin.llvmlower.scalar_contraction import RECTANGULAR_FEATURE

    original = source()
    rewritten, count = apply_for_test(original, effects=EFFECTS, outputs=4, rows=2)
    assert count == 1
    prepared = lower_to_llvm_ir(rewritten, workdir=tmp_path / "prepared")
    features = {FEATURE, RECTANGULAR_FEATURE}
    if integer_softmax:
        features.add(int_softmax_table.FEATURE)
    normal = lower_to_llvm_ir(
        original,
        workdir=tmp_path / "normal",
        features=frozenset(features),
        masked_contraction_effects=EFFECTS,
    )
    assert normal == prepared
    recipe = json.loads((tmp_path / "normal/lowering_recipe.json").read_text())
    argv = recipe["commands"][0]["argv"][1:]
    assert ARGV_INDEX == int_softmax_table.ARGV_INDEX + 1
    assert len(argv) == ARGV_INDEX + 1
    assert argv[int_softmax_table.ARGV_INDEX] == ("1" if integer_softmax else "0")
    assert argv[ARGV_INDEX] == "1"
    assert (tmp_path / "normal/int_softmax_table_report.json").exists() == integer_softmax


def test_captured_outer_mask_refuses_without_fabricated_dominance():
    text = source()
    prefix, body = text.split("attributes {llvm.emit_c_interface} {", 1)
    body = body.rsplit("} }", 1)[0].replace("return %selected:", "scf.yield %selected:")
    out = "tensor<1x2x4x8xf32>"
    nested = (
        prefix
        + "attributes {llvm.emit_c_interface} {\n"
        + f"%outer = scf.execute_region -> {out} {{\n"
        + body
        + "}\n"
        + f"return %outer:{out}\n"
        + "} }"
    )
    rewritten, count = apply_for_test(nested, effects=EFFECTS, outputs=4, rows=2)
    assert count == 0 and "scf.execute_region" in rewritten and "scf.if" not in rewritten


def test_normal_report_refuses_missing_duplicate_and_negative(tmp_path):
    from merlin.llvmlower.masked_contraction import FEATURE, require_report

    for text in ("OK\n", f"OK {FEATURE} 1\nOK {FEATURE} 1\n", f"OK {FEATURE} -1\n"):
        with pytest.raises(ValueError):
            require_report(text, tmp_path)
    assert require_report(f"OK {FEATURE} 0\n", tmp_path) == 0
    assert '"rewritten_contractions": 0' in (tmp_path / "masked_contraction_report.json").read_text()
