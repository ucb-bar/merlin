"""Movement chains and the added movement forms: built, graded exactly (independent numpy), admitted per command."""

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

EXTENTS = [(16, 16), (15, 16), (2, 4), (1, 4), (3, 5), (7, 3)]
FORMS = ["select", "split_sizes", "select_scatter", "squeeze_dims", "clone", "copy_into", "alias"]


def _build(form, m, n):
    binding = SPEC.CorpusBinding(
        target="gemmini", tile_dim=16, operand_dtype="int8", accum_dtype="int32", integer=True,
        tiers=["L2", "L3"], compare="exact",
    )  # fmt: skip
    row = ft.FORM_TABLE[("movement", form)]
    entry = {
        "name": f"T_{form}", "kind": "isa", "source_role": "derived_sweep", "source_reference": "test",
        "op": row.op, "label": "public", **ft.entry_fields("movement", form, m, n),
    }  # fmt: skip
    return SPEC.build(entry, binding)


def _leaves(capsule):
    out = materialize_capsule_leaves(capsule)
    return {k: np.array(v.data, dtype=np.int64).reshape(v.shape) for k, v in out.items()}


@pytest.mark.parametrize("m,n", EXTENTS)
def test_select_drops_the_dim(m, n):
    cap, text = _build("select", m, n)
    x = _leaves(cap)["X"]
    got = capsule_golden.golden(cap, None)
    assert np.array_equal(np.array(got["Y0"]), x[m // 2])
    assert np.array(got["Y0"]).shape == (n,)
    assert text.count("merlin_iface.movement") == 2 and f"-> tensor<{n}xi8>" in text


@pytest.mark.parametrize("m,n", EXTENTS)
def test_split_is_a_multi_output_partition(m, n):
    if m < 2 and n < 2:
        pytest.skip("no dim to split")
    cap, _text = _build("split_sizes", m, n)
    x = _leaves(cap)["X"]
    a = cap["operation"]["attributes"]
    dim, sizes, outs = a["dim"], a["sizes"], a["outs"]
    got = capsule_golden.golden(cap, None)
    assert len(outs) == len(sizes) >= 2 and sum(sizes) == x.shape[dim]
    if x.shape[dim] > 2:
        assert len(set(sizes)) >= 2
    assert np.array_equal(np.concatenate([np.array(got[o]) for o in outs], axis=dim), x)
    assert [np.array(got[o]).shape[dim] for o in outs] == sizes


@pytest.mark.parametrize("m,n", EXTENTS)
def test_select_scatter_writes_one_index_from_a_rank_reduced_source(m, n):
    cap, _text = _build("select_scatter", m, n)
    env = _leaves(cap)
    d, s = env["D"], env["S"]
    assert s.shape == (n,)
    want = d.copy()
    want[m // 2] = s
    assert np.array_equal(np.array(capsule_golden.golden(cap, None)["Y0"]), want)


@pytest.mark.parametrize("m,n", EXTENTS)
def test_squeeze_dims_removes_two_dims(m, n):
    cap, _text = _build("squeeze_dims", m, n)
    x = _leaves(cap)["X"]
    assert x.shape == (1, m, 1, n)
    assert np.array_equal(np.array(capsule_golden.golden(cap, None)["Y0"]), x.reshape(m, n))


@pytest.mark.parametrize("form", ["clone", "alias"])
@pytest.mark.parametrize("m,n", EXTENTS)
def test_clone_and_alias_preserve_values_and_dtype(form, m, n):
    cap, text = _build(form, m, n)
    x = _leaves(cap)["X"]
    assert np.array_equal(np.array(capsule_golden.golden(cap, None)["Y0"]), x)
    assert "xi32" not in text and f"tensor<{m}x{n}xi8>) -> tensor<{m}x{n}xi8>" in text


@pytest.mark.parametrize("m,n", EXTENTS)
def test_copy_overwrites_the_whole_destination(m, n):
    cap, text = _build("copy_into", m, n)
    env = _leaves(cap)
    assert np.array_equal(np.array(capsule_golden.golden(cap, None)["Y0"]), env["S"])
    assert "merlin_iface.insert_slice" in text


def test_scalar_tensor_is_rank_zero():
    cap = _build("fill", 1, 1)  # sanity: the table row exists
    del cap
    binding = SPEC.CorpusBinding(
        target="gemmini", tile_dim=16, operand_dtype="int8", accum_dtype="int32", integer=True,
        tiers=["L2", "L3"], compare="exact",
    )  # fmt: skip
    row = ft.FORM_TABLE[("movement", "fill_scalar")]
    entry = {
        "name": "T_fs", "kind": "isa", "source_role": "derived_sweep", "source_reference": "test",
        "op": row.op, "label": "public", **ft.entry_fields("movement", "fill_scalar", 16, 16),
    }  # fmt: skip
    capsule, text = SPEC.build(entry, binding)
    assert "-> tensor<i8>" in text
    got = capsule_golden.golden(capsule, None)
    assert np.array(got["Y0"]).shape == () and int(np.array(got["Y0"])) == 5


@pytest.mark.parametrize("form", FORMS + ["fill_scalar"])
def test_every_command_is_admitted_on_the_accelerator(form):
    m, n = (16, 16)
    cap, text = _build(form, m, n) if form != "fill_scalar" else _fs()
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
    assert compute and all(r["accelerator_admission"]["status"] == "admitted" for r in compute), compute
    assert summarize(rows, host_only=False, scope="test")["status"] == "admitted"


def _fs():
    binding = SPEC.CorpusBinding(
        target="gemmini", tile_dim=16, operand_dtype="int8", accum_dtype="int32", integer=True,
        tiers=["L2", "L3"], compare="exact",
    )  # fmt: skip
    row = ft.FORM_TABLE[("movement", "fill_scalar")]
    entry = {
        "name": "T_fs", "kind": "isa", "source_role": "derived_sweep", "source_reference": "test",
        "op": row.op, "label": "public", **ft.entry_fields("movement", "fill_scalar", 16, 16),
    }  # fmt: skip
    return SPEC.build(entry, binding)


def test_forms_are_declared_by_the_reviewed_movement_row():
    from merlin.targetgen.software_spec import validate_software_spec

    doc = validate_software_spec(load_yaml(repo_root() / "examples/gemmini/target/software-spec.yaml"), "gemmini")
    contract = load_yaml(repo_root() / "examples/gemmini/target/contracts/target_contract.yaml")
    kept, refused = ft.declared_forms(doc, contract)
    assert set(FORMS + ["fill_scalar"]) <= set(kept.get("movement", [])), (kept, refused)
    assert "split_chunk" not in kept.get("movement", [])


SECOND_WINDOW_FORMS = [
    "pool1d_k2", "pool2d_k4", "pool3d_overlap", "adaptive1d_even", "adaptive2d_even", "adaptive3d_even",
    "pool2d_backward_overlap", "adaptive2d_backward_even", "mean_dim_h", "mean_all_small",
]


@pytest.mark.parametrize("form", SECOND_WINDOW_FORMS)
def test_second_window_geometries_are_built_graded_and_admitted(form):
    binding = SPEC.CorpusBinding(
        target="gemmini", tile_dim=16, operand_dtype="int8", accum_dtype="int32", integer=True,
        tiers=["L2", "L3"], compare="exact",
    )  # fmt: skip
    row = ft.FORM_TABLE[("reduction", form)]
    assert row.once
    entry = {
        "name": f"T_{form}", "kind": "isa", "source_role": "derived_sweep", "source_reference": "test",
        "op": row.op, "label": "public", **ft.entry_fields("reduction", form, 16, 16),
    }  # fmt: skip
    capsule, text = SPEC.build(entry, binding)
    assert capsule_golden.golden(capsule, None)
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
    assert summarize(rows, host_only=False, scope="test")["status"] == "admitted"
