"""Selective elimination of materialized ``linalg.broadcast`` intermediates.

The whole-model LSTMNetVIT profile attributes 4.863 ms/inference to the 161 broadcasts whose
only reader is an all-parallel ``linalg.generic``.  A broadcast is an indexing map, not arithmetic:
the consumer can read the source directly with the broadcast dimensions projected out.  This test
pins the narrow rewrite; it must not recreate the previously-refuted blanket elementwise fusion
that can absorb dequantization into contractions.  The whole-model experiment later proved this
lever converges to byte-identical uninstrumented code: profile markers had prevented the normal
post-contraction fusion and made the broadcasts look like residual work.
"""
from __future__ import annotations

import subprocess

import pytest

from merlin.llvmlower import toolchain
from merlin.llvmlower.broadcast_fold import FEATURE, require_report, run_source
from merlin.llvmlower.impr_features import get, known, normalize


_needs_m2m = pytest.mark.skipif(not toolchain.available(),
                                reason="m2m venv / clang not configured")


def _fold(tmp_path, mlir_text: str, name: str = "f") -> tuple[str, str]:
    driver = tmp_path / f"{name}_driver.py"
    driver.write_text(run_source(), encoding="utf-8")
    src = tmp_path / f"{name}.mlir"
    src.write_text(mlir_text, encoding="utf-8")
    out = tmp_path / f"{name}.out.mlir"
    proc = subprocess.run([str(toolchain.m2m_python()), str(driver), str(src), str(out)],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout, out.read_text(encoding="utf-8")


_SOLE_USE = """
#id2 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @forward(%scale: tensor<4xf32>, %x: tensor<3x4xf32>) -> tensor<3x4xf32> {
    %be = tensor.empty() : tensor<3x4xf32>
    %b = linalg.broadcast ins(%scale : tensor<4xf32>) outs(%be : tensor<3x4xf32>)
         dimensions = [0]
    %oe = tensor.empty() : tensor<3x4xf32>
    %r = linalg.generic {indexing_maps = [#id2, #id2, #id2],
                         iterator_types = ["parallel", "parallel"]}
        ins(%x, %b : tensor<3x4xf32>, tensor<3x4xf32>)
        outs(%oe : tensor<3x4xf32>) {
    ^bb0(%a: f32, %s: f32, %o: f32):
      %v = arith.mulf %a, %s : f32
      linalg.yield %v : f32
    } -> tensor<3x4xf32>
    return %r : tensor<3x4xf32>
  }
}
"""


def test_feature_is_registered_and_default_off():
    assert FEATURE in known()
    assert get(FEATURE).action_class == "PASS"
    assert get(FEATURE).edit_pipeline is None
    assert normalize(None) == frozenset()
    assert FEATURE in normalize([FEATURE])


@_needs_m2m
def test_projects_broadcast_dimensions_into_the_consumer_map(tmp_path):
    stdout, text = _fold(tmp_path, _SOLE_USE)
    assert "FOLDED 1" in stdout, stdout
    assert "linalg.broadcast" not in text
    assert "tensor.empty() : tensor<3x4xf32>" in text  # consumer output remains
    assert text.count("tensor.empty() : tensor<3x4xf32>") == 1  # dead broadcast init erased
    assert "affine_map<(d0, d1) -> (d1)>" in text
    assert "ins(%arg1, %arg0 : tensor<3x4xf32>, tensor<4xf32>)" in text


@_needs_m2m
def test_folded_module_verifies(tmp_path):
    _stdout, text = _fold(tmp_path, _SOLE_USE, "verify")
    opt = toolchain.mlir_translate().parent / "mlir-opt"
    if not opt.is_file():
        pytest.skip("standalone mlir-opt not present")
    src = tmp_path / "folded.mlir"
    src.write_text(text, encoding="utf-8")
    proc = subprocess.run([str(opt), str(src), "-o", str(tmp_path / "checked.mlir")],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr


_SHARED = _SOLE_USE.replace(
    "return %r : tensor<3x4xf32>",
    """%oe2 = tensor.empty() : tensor<3x4xf32>
    %r2 = linalg.generic {indexing_maps = [#id2, #id2],
                          iterator_types = [\"parallel\", \"parallel\"]}
        ins(%b : tensor<3x4xf32>) outs(%oe2 : tensor<3x4xf32>) {
    ^bb0(%a: f32, %o: f32):
      linalg.yield %a : f32
    } -> tensor<3x4xf32>
    return %r2 : tensor<3x4xf32>""",
)


@_needs_m2m
def test_shared_broadcast_is_refused_without_partial_rewrite(tmp_path):
    stdout, text = _fold(tmp_path, _SHARED, "shared")
    assert "FOLDED 0" in stdout, stdout
    assert "2 readers" in stdout, stdout
    assert "linalg.broadcast" in text


_REDUCTION = _SOLE_USE.replace(
    'iterator_types = ["parallel", "parallel"]',
    'iterator_types = ["parallel", "reduction"]',
)


@_needs_m2m
def test_reduction_consumer_is_refused(tmp_path):
    stdout, text = _fold(tmp_path, _REDUCTION, "reduction")
    assert "FOLDED 0" in stdout, stdout
    assert "non-parallel iterator" in stdout, stdout
    assert "linalg.broadcast" in text


def test_every_runner_variant_calls_the_fold_and_reads_argv14():
    from merlin.llvmlower.accum_microkernel import run_source as scalarize_source
    from merlin.llvmlower.pipeline import EMIT_TRANSLATE, _RUNNER, _activation_poly_runner

    variants = {
        "plain": _RUNNER,
        "act_poly": _activation_poly_runner(EMIT_TRANSLATE),
        "scalarize": scalarize_source().replace("__MERLIN_EMIT__", EMIT_TRANSLATE),
    }
    for name, source in variants.items():
        assert "_fold_broadcasts(" in source, f"{name} runner omits broadcast fold"
        assert "_FOLD_BROADCAST" in source, f"{name} runner omits broadcast gate"
        assert "sys.argv[14]" in source, f"{name} runner reads wrong broadcast gate"


def test_requested_rewrite_requires_one_nonzero_runner_receipt(tmp_path):
    assert require_report("noise\nOK fold_broadcast_into_generic folded 161\n", tmp_path) == 161
    assert (tmp_path / "broadcast_fold_report.txt").read_text() == "folded=161\n"
    with pytest.raises(ValueError, match="did not report exactly once"):
        require_report("noise only", tmp_path)
    with pytest.raises(ValueError, match="folded zero"):
        require_report("OK fold_broadcast_into_generic folded 0\n", tmp_path)


def test_receipt_parser_accepts_only_complete_exact_lines(tmp_path):
    noise = ("prefix OK fold_broadcast_into_generic folded 7\n"
             "OK fold_broadcast_into_generic folded 8 suffix\n")
    with pytest.raises(ValueError, match="did not report exactly once"):
        require_report(noise, tmp_path)
    with pytest.raises(ValueError, match="did not report exactly once"):
        require_report("OK fold_broadcast_into_generic folded 1\n" * 2, tmp_path)
