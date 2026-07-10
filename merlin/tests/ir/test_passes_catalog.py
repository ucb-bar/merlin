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
    from merlin.xdsl_dialects.lowering.passes import catalog

    cat = catalog()
    assert {p.name for p in cat} >= {
        "merlin-lower-quant-ext", "merlin-outline-dispatches",
        "merlin-emit-dispatch-program", "merlin-add-c-interface"}
    for p in cat:
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
