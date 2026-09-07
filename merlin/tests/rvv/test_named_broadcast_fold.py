"""Fold only broadcasts left behind by named ``linalg.add``/``linalg.mul`` consumers."""
from __future__ import annotations

import subprocess

import pytest

from merlin.llvmlower import toolchain
from merlin.llvmlower.impr_features import apply_pipeline, normalize
from merlin.llvmlower.named_broadcast_fold import FEATURE, MARKER, require_report, run_source


_needs_m2m = pytest.mark.skipif(not toolchain.available(), reason="m2m toolchain unavailable")


def _run(tmp_path, text: str) -> tuple[str, str]:
    driver = tmp_path / "driver.py"
    driver.write_text(run_source(), encoding="utf-8")
    src, out = tmp_path / "in.mlir", tmp_path / "out.mlir"
    src.write_text(text, encoding="utf-8")
    proc = subprocess.run([str(toolchain.m2m_python()), str(driver), str(src), str(out)],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout, out.read_text(encoding="utf-8")


_ADD = """
module {
  func.func @forward(%x: tensor<3x4xf32>, %bias: tensor<4xf32>) -> tensor<3x4xf32> {
    %be = tensor.empty() : tensor<3x4xf32>
    %b = linalg.broadcast ins(%bias : tensor<4xf32>) outs(%be : tensor<3x4xf32>) dimensions = [0]
    %oe = tensor.empty() : tensor<3x4xf32>
    %r = linalg.add ins(%x, %b : tensor<3x4xf32>, tensor<3x4xf32>)
         outs(%oe : tensor<3x4xf32>) -> tensor<3x4xf32>
    return %r : tensor<3x4xf32>
  }
}
"""


def test_stage_is_after_existing_fusion_and_before_generalization():
    from merlin.llvmlower.impr_features import FUSE_ELEMENTWISE_NAME
    from merlin.llvmlower.pipeline import _FUSE_ELEMENTWISE, _GENERALIZE_NAMED

    passes = ["transform-interpreter{entry-point=__transform_main}", _GENERALIZE_NAMED,
              "one-shot-bufferize"]
    out = apply_pipeline(passes, normalize({FUSE_ELEMENTWISE_NAME, FEATURE}))
    assert out.index(_FUSE_ELEMENTWISE) < out.index(MARKER) < out.index(_GENERALIZE_NAMED)


def test_every_runner_carries_the_stage_and_argv15_gate():
    from merlin.llvmlower.accum_microkernel import run_source as accum_runner
    from merlin.llvmlower.pipeline import EMIT_TRANSLATE, _RUNNER, _activation_poly_runner

    for name, source in {
        "plain": _RUNNER,
        "activation": _activation_poly_runner(EMIT_TRANSLATE),
        "accumulator": accum_runner().replace("__MERLIN_EMIT__", EMIT_TRANSLATE),
    }.items():
        assert "_PRE_GENERALIZE_STAGES" in source, name
        assert "len(sys.argv) > 15" in source, name
        assert "_targeted_named_broadcast_fold" in source, name


@_needs_m2m
def test_named_add_broadcast_becomes_projected_generic(tmp_path):
    stdout, text = _run(tmp_path, _ADD)
    assert "CENSUS targeted_named_broadcast_fold add 1 mul 0" in stdout
    assert "FOLDED 1" in stdout
    assert "linalg.broadcast" not in text
    assert "linalg.add" not in text
    assert "affine_map<(d0, d1) -> (d1)>" in text


_SHARED = _ADD.replace(
    "return %r : tensor<3x4xf32>",
    """%oe2 = tensor.empty() : tensor<3x4xf32>
    %r2 = linalg.mul ins(%r, %b : tensor<3x4xf32>, tensor<3x4xf32>)
          outs(%oe2 : tensor<3x4xf32>) -> tensor<3x4xf32>
    return %r2 : tensor<3x4xf32>""",
)


@_needs_m2m
def test_shared_broadcast_is_refused(tmp_path):
    stdout, text = _run(tmp_path, _SHARED)
    assert "CENSUS targeted_named_broadcast_fold add 0 mul 0" in stdout
    assert "FOLDED 0" in stdout
    assert "linalg.broadcast" in text


_REDUCTION = """
#lhs = affine_map<(d0, d1, d2) -> (d0, d2)>
#rhs = affine_map<(d0, d1, d2) -> (d2, d1)>
#out = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @forward(%x: tensor<3x4xf32>, %w: tensor<4x2xf32>, %s: tensor<2xf32>)
      -> tensor<3x2xf32> {
    %be = tensor.empty() : tensor<3x2xf32>
    %b = linalg.broadcast ins(%s : tensor<2xf32>) outs(%be : tensor<3x2xf32>) dimensions = [0]
    %oe = tensor.empty() : tensor<3x2xf32>
    %r = linalg.generic {indexing_maps = [#lhs, #rhs, #out, #out],
                         iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%x, %w, %b : tensor<3x4xf32>, tensor<4x2xf32>, tensor<3x2xf32>)
        outs(%oe : tensor<3x2xf32>) {
    ^bb0(%a: f32, %q: f32, %scale: f32, %acc: f32):
      %p = arith.mulf %a, %q : f32
      %v = arith.addf %p, %acc : f32
      linalg.yield %v : f32
    } -> tensor<3x2xf32>
    return %r : tensor<3x2xf32>
  }
}
"""


@_needs_m2m
def test_contraction_reduction_consumer_is_not_rewritten(tmp_path):
    stdout, text = _run(tmp_path, _REDUCTION)
    assert "FOLDED 0" in stdout
    assert "linalg.broadcast" in text
    assert 'iterator_types = ["parallel", "parallel", "reduction"]' in text


def test_receipt_records_exact_real_census(tmp_path):
    stdout = ("CENSUS targeted_named_broadcast_fold add 52 mul 10\n"
              "OK targeted_named_broadcast_fold 62\n")
    report = require_report(stdout, tmp_path)
    assert report == {"add": 52, "mul": 10, "folded": 62}
    assert "add=52" in (tmp_path / "named_broadcast_fold_report.txt").read_text()
    with pytest.raises(ValueError, match="missing or inconsistent"):
        require_report("OK targeted_named_broadcast_fold 0\n", tmp_path)


def test_receipt_parser_rejects_partial_and_duplicate_lines(tmp_path):
    partial = ("prefix CENSUS targeted_named_broadcast_fold add 1 mul 2\n"
               "OK targeted_named_broadcast_fold 3 suffix\n")
    with pytest.raises(ValueError, match="missing or inconsistent"):
        require_report(partial, tmp_path)
    duplicate = ("CENSUS targeted_named_broadcast_fold add 1 mul 2\n"
                 "CENSUS targeted_named_broadcast_fold add 1 mul 2\n"
                 "OK targeted_named_broadcast_fold 3\n")
    with pytest.raises(ValueError, match="missing or inconsistent"):
        require_report(duplicate, tmp_path)
