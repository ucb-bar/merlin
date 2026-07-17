"""C8: assemble a model SECTION into a K1-buildable bundle (rvvgen.section_build).

Gated on a real capture with weights (the committed prov-only corpus has no weights). Asserts the
assembled section is a well-formed, self-contained "model" directory the whole-model K1 build consumes
— exercised without touching the board (the on-board run is validated separately).
"""
from __future__ import annotations

import json

import pytest

from merlin.common.paths import repo_root
from merlin.xdsl_dialects import _common

_CAP = repo_root() / "out/artifacts/recaptures/small_consistent"
pytestmark = pytest.mark.skipif(
    not _common.HAS_XDSL or not (_CAP / "model.mlir").is_file(),
    reason="small_consistent capture (with weights) not present")


def test_section_bundle_is_a_valid_buildable_model_dir(tmp_path):
    import numpy as np

    from merlin.llvmlower.weights_pack import load_safetensors_header
    from merlin.rvvgen.section_build import build_section_bundle

    out = tmp_path / "sec_mm0"
    summary = build_section_bundle(str(_CAP), {"matmul_0"}, out)

    # a self-contained section "model" directory the K1 build (build_k1_binary) consumes unchanged.
    for f in ("model.mlir", "weights.safetensors", "weights.safetensors.manifest.json",
              "inputs.npz", "input_order.json"):
        assert (out / f).is_file(), f

    text = (out / "model.mlir").read_text()
    assert "func.func @forward(" in text and text.count("func.call") == 1   # exactly the one section kernel

    man = json.loads((out / "weights.safetensors.manifest.json").read_text())
    assert len(man) == summary["n_boundary_args"]
    # every boundary arg is classified (weight -> param carried in the section safetensors; else input).
    assert summary["n_weights"] + summary["n_inputs"] == summary["n_boundary_args"]
    load_safetensors_header(out / "weights.safetensors")                    # section safetensors parses

    z = np.load(out / "inputs.npz")
    assert len(z.files) == summary["n_inputs"]
    assert summary["output_shape"]                                          # the section produces an output


def test_bad_region_id_fails_closed(tmp_path):
    from merlin.xdsl_dialects.lowering.outline import OutlineError
    from merlin.rvvgen.section_build import build_section_bundle

    with pytest.raises((OutlineError, ValueError)):
        build_section_bundle(str(_CAP), {"no_such_region"}, tmp_path / "nope")
