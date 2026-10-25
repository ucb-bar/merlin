"""Native valid-reference comparisons preserve every NaN payload counterexample."""

import os
from pathlib import Path

import pytest
from lean_overrank_native import CASES, compare


@pytest.mark.parametrize("batch,k,n", CASES)
@pytest.mark.parametrize("kind", ["f32", "f64"])
def test_native_valid_reference(tmp_path, batch, k, n, kind):
    compiler = Path(os.environ.get("MERLIN_M2M_PYTHON", "/home/jeremy/Projects/model2MLIR/.venv/bin/python"))
    assert compiler.is_file()
    report = compare(batch, k, n, kind, compiler, tmp_path)
    assert not report["non_nan_mismatches"]
    assert report["samples"] == 6
