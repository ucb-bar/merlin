"""The spatial convolution-backward chain (padded 3x3, stride 1): built, graded exactly, and admitted region by region."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from merlin_experiments.phase0.program_admission import account_interface, summarize
from phase0_fixture_selection import fixture_resolved

from merlin.common.paths import repo_root
from merlin.common.yaml import load_yaml
from merlin.targetgen import capsule_golden
from merlin.targetgen import corpus_form_table as ft
from merlin.targetgen import corpus_spec as SPEC
from merlin.targetgen.capsule_inputs import materialize_capsule_leaves

pytestmark = pytest.mark.target("gemmini")


def _build():
    binding = SPEC.CorpusBinding(
        target="gemmini", tile_dim=16, operand_dtype="int8", accum_dtype="int32", integer=True,
        tiers=["L2", "L3"], compare="exact",
    )  # fmt: skip
    entry = {
        "name": "T_cbs", "kind": "isa", "source_role": "derived_sweep", "source_reference": "test",
        "op": "conv_backward_spatial", "label": "public", **ft.entry_fields("contraction", "convolution_backward_spatial", 16, 16),
    }  # fmt: skip
    return SPEC.build(entry, binding)


def test_golden_matches_an_independent_numpy_convolution_backward():
    capsule, _text = _build()
    g = capsule["operation"]["attributes"]["geometry"]
    leaves = materialize_capsule_leaves(capsule)
    xcol = np.array(leaves["Xcol"].data, dtype=np.int64).reshape(g["L"], g["K"])
    dy = np.array(leaves["dY"].data, dtype=np.int64).reshape(g["L"], g["N"])
    w = np.array(leaves["W"].data, dtype=np.int64).reshape(g["K"], g["N"])
    got = capsule_golden.golden(capsule, None)
    assert np.array_equal(np.array(got["grad_weight"]), xcol.T @ dy)
    assert np.array_equal(np.array(got["grad_bias"]), dy.sum(axis=0, keepdims=True))
    # grad_input: requantize W @ dY^T to int8 (RNE of the power-of-two gain, saturating), then overlap-add by hand.
    scale = capsule["operation"]["attributes"]["grad_input_col_scale"]
    col = np.clip(np.rint((w @ dy.T) * scale), -128, 127).astype(np.int64)
    kh, kw = g["kernel"]
    (ph, pw), c, h, wd = g["padding"], g["C"], g["H"], g["W"]
    image = np.zeros((c, h, wd), dtype=np.int64)
    for ch in range(c):
        for i in range(kh):
            for j in range(kw):
                for block in range(g["L"]):
                    y, x = block // wd - ph + i, block % wd - pw + j
                    if 0 <= y < h and 0 <= x < wd:
                        image[ch, y, x] += col[(ch * kh + i) * kw + j, block]
    assert np.array_equal(np.array(got["grad_input"]).reshape(c, h, wd), image)


def test_every_region_is_placed_on_the_accelerator_and_tagged():
    _capsule, text = _build()
    assert text.count('prov.aten = "aten.convolution_backward.default"') == 4
    base = repo_root() / "examples/gemmini/target"
    host = load_yaml(repo_root() / "examples/gemmini/platform/host-capabilities-core-aten.yaml")
    evidence = SimpleNamespace(
        software_spec=fixture_resolved(base, "gemmini"),
        contract=load_yaml(base / "contracts/target_contract.yaml"),
        host_capabilities={
            "profile": {
                "capability_spec": host, "package_sha256": host["compiler"]["package_sha256"],
                "capability_spec_sha256": "b" * 64, "dtype_strategy": host["compiler"]["dtype_strategy"],
            }
        },
        whole_program_admission=None,
    )  # fmt: skip
    path = Path(tempfile.mkdtemp()) / "capsule.interface.mlir"
    path.write_text(text)
    rows = account_interface(path, target="gemmini", evidence=evidence, numeric_screens=[])
    compute = [r for r in rows if (r.get("accelerator_admission") or {}).get("status") != "not_applicable"]
    assert {r["operation"] for r in compute} >= {"matmul", "movement", "window_sum"}
    assert all(r["accelerator_admission"]["status"] == "admitted" for r in compute), compute
    assert summarize(rows, host_only=False, scope="test")["status"] == "admitted"


def test_form_is_declared_by_the_contraction_row():
    from merlin.targetgen.software_spec import validate_software_spec

    doc = validate_software_spec(load_yaml(repo_root() / "examples/gemmini/target/software-spec.yaml"), "gemmini")
    kept, refused = ft.declared_forms(doc, {})
    assert kept.get("contraction") == ["convolution_backward", "convolution_backward_spatial"], (kept, refused)
