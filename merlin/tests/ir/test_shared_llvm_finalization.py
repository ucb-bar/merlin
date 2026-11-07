"""Scalar and extension pipelines share one terminal LLVM conversion contract."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys

import pytest

from merlin.compiler.lower.llvm.passes import (
    append_finalization,
    finalization_passes,
    finalize_pipeline,
)


def test_shared_suffix_is_the_scalar_and_vector_llvm_boundary():
    common = finalization_passes()
    vector = finalization_passes(vector=True)
    parallel = finalization_passes(openmp=True)
    assert vector[:2] == ("convert-vector-to-llvm", "convert-ub-to-llvm")
    assert parallel[0] == "convert-openmp-to-llvm"
    assert vector[2:] == parallel[1:] == common
    assert append_finalization(["lower-affine"], vector=True) == ["lower-affine", *vector]
    assert "reconcile-unrealized-casts" in common
    assert common[-1] == "symbol-dce"


def test_scalar_and_openmp_pass_strings_remain_byte_identical(monkeypatch):
    from merlin.llvmlower import pipeline

    monkeypatch.delenv("MERLIN_NO_DEALLOC", raising=False)
    monkeypatch.delenv("MERLIN_SINK_DEALLOCS", raising=False)
    serial = pipeline._upstream_pipeline()
    parallel = pipeline._parallel_pipeline()
    assert (
        hashlib.sha256(serial.encode()).hexdigest()
        == "32b16a8e509702f8df80caf4d7ed16736c3ab2389dc95122de0fbaaa0cc21ac6"
    )
    assert (
        hashlib.sha256(parallel.encode()).hexdigest()
        == "16763baf16d48418ecdc4aaaa7192e2a3da8571ee346fa2c1c010ca194d206c5"
    )
    assert finalize_pipeline(serial) == serial
    assert finalize_pipeline(parallel) == parallel


def test_incomplete_backend_pipeline_is_refused_before_runner():
    with pytest.raises(ValueError, match="lacks ordered LLVM finalization"):
        finalize_pipeline("canonicalize,convert-func-to-llvm,symbol-dce")


def test_shared_llvm_package_imports_without_driver_or_backend(tmp_path):
    from merlin.common.paths import merlin_dir

    source = (
        "import sys\n"
        "from merlin.compiler.lower.llvm import abi, normalize, passes\n"
        "assert passes.finalization_passes()\n"
        "assert normalize._fix_float_literals('float inf') == 'float 0x7FF0000000000000'\n"
        "assert abi.BaseBuffer('x', 16, 64).name == 'x'\n"
        "assert 'merlin.llvmlower.pipeline' not in sys.modules\n"
        "assert 'merlin.llvmlower.backend_spi' not in sys.modules\n"
    )
    env = {**os.environ, "PYTHONPATH": str(merlin_dir() / "python")}
    result = subprocess.run([sys.executable, "-c", source], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
