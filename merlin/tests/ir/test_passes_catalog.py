"""The Merlin-authored pass catalog and the whole-model dialect-plane entry point."""
from __future__ import annotations
from merlin.common.paths import repo_root, merlin_dir

import importlib
from pathlib import Path

import pytest

from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")

REPO = repo_root()


def test_every_authored_pass_entry_resolves():
    """Every catalogued entry point must import and be callable -- in ALL THREE catalogs.

    The catalog was one flat list until the obligation gate split it three ways: production passes
    that discharge a target obligation, target-independent numeric/frontend normalizations, and the
    staged research pipeline. This test asserted the flat membership and kept demanding
    `merlin-lower-quant-ext` from `catalog()` after that pass moved to `normalization_catalog()` --
    which is where it belongs, since it normalizes quantization extensions in the target-independent
    plane and discharges no target obligation. Assert each catalog's own membership instead, and
    resolve the entry points of all three: a pass whose entry point stopped importing is the failure
    this test exists for, and checking only the production four would stop looking at 12 of them.
    """
    from merlin.xdsl_dialects.lowering.passes import (catalog, normalization_catalog,
                                                      prototype_catalog)

    production = catalog()
    assert {p.name for p in production} >= {
        "merlin-outline-dispatches", "merlin-emit-dispatch-program",
        "merlin-partition-dispatches", "merlin-add-c-interface"}
    # The normalizations are NOT production obligations, and that separation is the point of the
    # split -- so name one here rather than letting it go unchecked in either place.
    assert "merlin-lower-quant-ext" in {p.name for p in normalization_catalog()}
    assert not ({p.name for p in production} & {p.name for p in normalization_catalog()}), (
        "a pass in both catalogs would let a normalization be credited as discharging a target "
        "obligation, which is exactly what the split prevents")

    for p in (*production, *normalization_catalog(), *prototype_catalog()):
        mod, _, fn = p.entry.rpartition(".")
        obj = getattr(importlib.import_module(mod), fn)
        assert callable(obj), p.entry


@pytest.mark.skipif(not (REPO / "out/artifacts/recaptures/small_consistent/model.mlir").is_file(),
                    reason="small_llama capture not present")
def test_dialect_plane_runs_on_a_real_model():
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.xdsl_dialects.lowering.dispatch_program import verify_program
    from merlin.xdsl_dialects.lowering.passes import run_dialect_plane

    res = run_dialect_plane(parse_mlir_file(REPO / "out/artifacts/recaptures/small_consistent/model.mlir"))
    assert res.stats["kernels"] == 183
    assert res.program.n_dispatches == 183
    assert verify_program(res.program) == []
