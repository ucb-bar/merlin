"""Tests for the compute-unit capability model (merlin.targetgen.compute_units) + RTL cross-check."""
from __future__ import annotations

import pytest

from merlin.common.yaml import load_yaml
from merlin.common.paths import targets_dir
from merlin.targetgen import compute_units as cu


def _contract(name: str) -> dict:
    return load_yaml(targets_dir() / name / "contracts" / "target_contract.yaml")


def test_parse_gemmini_and_toy_contracts():
    for name in ("gemmini", "toy_npu"):
        units = cu.compute_units(_contract(name))
        assert units, f"{name} declares compute_units"
        mesh = units[0]
        assert mesh.kind == "systolic"
        assert "int8" in mesh.dtypes
        assert mesh.scaling == "per_channel"
        # requant is carried opaquely, never interpreted.
        assert isinstance(mesh.requant, dict) and "ref" in mesh.requant
    # gemmini no longer re-declares the (in,weight)->acc `accumulate` matrix — it IS the CIRCT
    # datapath facts (input i8 / accumulator i32) — so its unit carries no accumulate rules; toy_npu
    # (a plain reference contract) still declares its i8xi8->i32 rule.
    assert cu.compute_units(_contract("gemmini"))[0].accumulate == ()
    assert cu.compute_units(_contract("toy_npu"))[0].accumulate[0].acc == "i32"


def test_unknown_format_or_kind_rejected():
    with pytest.raises(ValueError):
        cu.compute_units({"compute_units": [{"name": "u", "kind": "systolic", "dtypes": ["nope_fmt"]}]})
    with pytest.raises(ValueError):
        cu.compute_units({"compute_units": [{"name": "u", "kind": "warp_thing", "dtypes": ["int8"]}]})
    with pytest.raises(ValueError):
        cu.compute_units({"compute_units": [{"name": "u", "kind": "vector", "scaling": "bogus"}]})


def test_composition_union():
    contract = {
        "compute_units": [
            {"name": "mx_pe", "kind": "systolic", "dtypes": ["mxfp4", "mxfp6", "mxfp8"],
             "ops": ["matmul"], "accumulate": [{"in": "mxfp8", "weight": "mxfp8", "acc": "f32"}]},
            {"name": "cluster", "kind": "simt", "dtypes": ["fp16", "bf16"], "ops": ["matmul", "elementwise"],
             "contains": ["mx_pe"]},
        ]
    }
    units = cu.compute_units(contract)
    cluster = next(u for u in units if u.name == "cluster")
    eff = cu.effective(cluster, units)
    # gemmini-mx-as-a-sub-unit: the cluster's effective dtypes include the embedded mx formats.
    assert {"fp16", "bf16", "mxfp4", "mxfp6", "mxfp8"} <= set(eff.dtypes)
    assert {"matmul", "elementwise"} <= set(eff.ops)


def test_datatype_tokens_resolve_aliases():
    unit = cu.compute_units(_contract("gemmini"))[0]
    # int8 carries the "i8" alias, so the RTL datapath token i8 is covered.
    assert "i8" in cu.datatype_tokens(unit)


def test_rtl_cross_check_passes_for_gemmini():
    # The RTL-derived gemmini datapaths (input i8, accumulator i32) are covered by the contract's
    # compute_units, so validate_against_contract reports no compute-unit problems.
    from merlin.targetgen.rtl import introspect

    datapaths = {"input": "i8", "accumulator": "i32"}
    assert introspect._check_compute_units(datapaths, _contract("gemmini")) == []
    # A datapath the units do not cover is flagged.
    bad = introspect._check_compute_units({"input": "f16", "accumulator": "i32"}, _contract("gemmini"))
    assert bad and "not covered" in bad[0]
