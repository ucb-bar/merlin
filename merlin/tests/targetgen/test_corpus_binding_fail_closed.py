"""A corpus binding cannot turn absent hardware facts into an int8/16-wide claim."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from merlin.targetgen import corpus_spec as CS


def test_fixed_array_geometry_requires_a_fact(monkeypatch):
    from merlin.targetgen.rtl import facts

    def missing(_target):
        raise FileNotFoundError("no RTL facts")

    monkeypatch.setattr(facts, "load_facts", missing)
    fixed = {"compute_units": [{"name": "array", "kind": "systolic", "dtypes": ["int8"]}]}
    with pytest.raises(ValueError, match="no derived tile geometry"):
        CS._tile_dim("synthetic", fixed, operand="int8")

    monkeypatch.setattr(facts, "load_facts", lambda _target: {"facts": {"arrays": [{"name": "mesh", "rows": 32}]}})
    assert CS._tile_dim("synthetic", fixed, operand="int8") == 32

    spatial = {"compute_units": [{"name": "tile", "kind": "spatial", "dtypes": ["int8"]}]}
    monkeypatch.setattr(
        facts,
        "load_facts",
        lambda _target: {"facts": {"fields": {"tile_dim": {"value": {"rows": 8, "cols": 8}}}}},
    )
    assert CS._tile_dim("synthetic", spatial, operand="int8") == 8

    monkeypatch.setattr(facts, "load_facts", missing)

    software = {"compute_units": [{"name": "lanes", "kind": "simt", "dtypes": ["fp32"]}]}
    assert CS._tile_dim("synthetic", software, operand="f32") == CS._DEFAULT_SW_TILE


def test_binding_refuses_missing_and_unsupported_datapath(monkeypatch):
    from merlin.targetgen import oracle_policy, target_experiment
    from merlin.targetgen.rtl import facts

    contract = {"compute_units": [{"name": "array", "kind": "systolic", "dtypes": []}]}
    monkeypatch.setattr(
        target_experiment, "load_capability_manifest", lambda _target: SimpleNamespace(contract=contract)
    )
    monkeypatch.setattr(oracle_policy, "inferred_oracle_tiers", lambda *_: ["L0"])
    monkeypatch.setattr(facts, "load_facts", lambda _target: {"facts": {"arrays": []}})
    te = SimpleNamespace(target="synthetic", sim_via="sim")

    with pytest.raises(ValueError, match="no compute-unit operand dtypes"):
        CS.derive_binding(te, {})

    contract["compute_units"][0]["dtypes"] = ["fp16"]
    with pytest.raises(ValueError, match="not admitted"):
        CS.derive_binding(te, {"operand_dtype": "int8"})

    monkeypatch.setattr(CS, "_classes_source", lambda *_: lambda **_: [])
    with pytest.raises(ValueError, match="no derived tile geometry"):
        CS.derive_binding(te, {})

    contract["capabilities"] = {"mesh": {"rows": 8}}
    binding = CS.derive_binding(te, {})
    assert binding.operand_dtype == "fp16" and binding.tile_dim == 8
