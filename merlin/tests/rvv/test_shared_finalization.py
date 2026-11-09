"""Shared scalar/vector finalization keeps ordering and target-width admission."""

from __future__ import annotations

import subprocess
import sys

import pytest

from merlin.common.paths import repo_root
from merlin.compiler.lower.llvm.passes import (
    append_finalization,
    bind_index_width,
    finalization_passes,
    finalize_pipeline,
)
from merlin.compiler.lower.scalar.pipeline import scalar_passes


@pytest.mark.parametrize("parallel", [False, True])
def test_scalar_allocation_and_parallelism_contract(parallel):
    passes = scalar_passes(parallel=parallel)
    assert passes.index("__DEALLOC__") < passes.index("convert-scf-to-cf")
    if parallel:
        assert "func.func(buffer-hoisting,buffer-loop-hoisting)" not in passes
        assert passes.index("convert-scf-to-openmp") < passes.index("convert-openmp-to-llvm")
        assert "func.func(convert-linalg-to-parallel-loops)" in passes
    else:
        assert passes.index("func.func(buffer-hoisting,buffer-loop-hoisting)") < passes.index("__DEALLOC__")
        assert "func.func(convert-linalg-to-loops)" in passes
        assert "convert-openmp-to-llvm" not in passes
    assert "convert-vector-to-llvm" not in passes


def test_component_composition_preserves_repeated_passes_and_caller_input():
    prefix = ["canonicalize", "canonicalize", "lower-affine"]
    selected = append_finalization(prefix, vector=True, openmp=True)
    assert prefix == ["canonicalize", "canonicalize", "lower-affine"]
    assert selected[:3] == prefix
    assert selected.count("canonicalize") == 3
    assert selected.index("convert-vector-to-llvm") < selected.index("convert-openmp-to-llvm")
    assert selected.index("convert-openmp-to-llvm") < selected.index("convert-func-to-llvm")
    assert finalize_pipeline(",".join(selected), index_bits=32).count("index-bitwidth=32") == 5


def test_incomplete_or_ambiguous_conversion_cannot_bind_target_width():
    selected = ",".join(finalization_passes())
    for invalid in (True, 0, -1, "64"):
        with pytest.raises(ValueError):
            bind_index_width(selected, invalid)
    with pytest.raises(ValueError, match="no unique bare"):
        bind_index_width(selected + ",convert-func-to-llvm", 64)
    with pytest.raises(ValueError, match="no unique bare"):
        bind_index_width(bind_index_width(selected, 32), 64)
    with pytest.raises(ValueError, match="lacks ordered"):
        finalize_pipeline(selected.replace("reconcile-unrealized-casts,", ""))


def test_scalar_and_finalization_import_without_driver_or_framework():
    script = """
import sys
sys.path.insert(0, sys.argv[1])
from merlin.compiler.lower.scalar.pipeline import scalar_passes
from merlin.compiler.lower.llvm.passes import finalize_pipeline
from merlin.compiler.passes.parallel import parallel_transform_schedule
assert scalar_passes()
assert "num_threads [0, 4]" in parallel_transform_schedule(4)
for forbidden in ('merlin.llvmlower.pipeline', 'merlin.compile_cli', 'torch', 'xdsl'):
    assert forbidden not in sys.modules, forbidden
"""
    subprocess.run([sys.executable, "-I", "-c", script, str(repo_root() / "src")], check=True)
