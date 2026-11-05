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


def test_source_bound_structural_grid_supplies_geometry_only(monkeypatch):
    from copy import deepcopy
    from merlin.targetgen import capability_derive, conformance, target_registry
    from merlin.targetgen.rtl import facts

    sha = "a" * 64
    bundle = {
        "inputs": {"fir_sha256": sha},
        "source_consistency": {"status": "verified", "production": {"firrtl_sha256": sha}},
        "facts": {
            "source": {"fir_sha256": sha},
            "arrays": [],
            "structural_observations": [{"kind": "mesh_tile_grid", "source": "selected_firrtl",
                                         "firrtl_sha256": sha, "rows": 12, "cols": 8, "instances": 96,
                                         "compute_engine_established": False}],
        },
    }
    contract = {"compute_units": [{"name": "array", "kind": "systolic", "dtypes": ["int8"]}]}
    assert CS._tile_dim("synthetic", contract, operand="int8", facts=bundle) == 12
    assert "contraction" not in capability_derive.derive("synthetic", contract, bundle).supported

    monkeypatch.setattr(facts, "load_facts", lambda _target: bundle)
    with target_registry.observed_contract("synthetic", contract):
        b = conformance.boundaries("synthetic")
    assert b.tile_edge == 12 and b.tile_edge_is_hardware_fact
    assert "geometry only" in b.tile_edge_source

    corrupted = deepcopy(bundle)
    corrupted["facts"]["structural_observations"][0]["instances"] = 95
    with pytest.raises(ValueError, match="geometry does not match"):
        CS._tile_dim("synthetic", contract, operand="int8", facts=corrupted)
    corrupted = deepcopy(bundle)
    corrupted["facts"]["structural_observations"][0]["firrtl_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="geometry does not match"):
        CS._tile_dim("synthetic", contract, operand="int8", facts=corrupted)


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

    monkeypatch.setattr(CS, "_classes_source", lambda *_, **__: lambda **_: [])
    with pytest.raises(ValueError, match="no derived tile geometry"):
        CS.derive_binding(te, {})

    contract["capabilities"] = {"mesh": {"rows": 8}}
    binding = CS.derive_binding(te, {})
    assert binding.operand_dtype == "fp16" and binding.tile_dim == 8


def test_diagnostic_binding_defers_but_does_not_admit_missing_instruction_classes():
    contract = {"compute_units": [{"name": "array", "kind": "systolic", "dtypes": ["int8"],
                                  "accumulate": [{"in": "int8", "weight": "int8", "acc": "i32"}]}]}
    facts = {"facts": {"arrays": [{"name": "mesh", "rows": 8, "cols": 8}]}}
    te = SimpleNamespace(target="synthetic", sim_via="sim")
    selected = {"contract": contract, "facts": facts, "taxonomy": {}}
    datapath = {"required_oracle_tiers": ["L0"]}
    with pytest.raises(ValueError, match="corpus_issue_order"):
        CS.derive_binding(te, datapath, **selected)
    binding = CS.derive_binding(te, datapath, defer_classes=True, **selected)
    assert binding.operand_dtype == "int8" and binding.accum_dtype == "i32" and binding.tile_dim == 8
    with pytest.raises(ValueError, match="corpus_issue_order"):
        binding.classes_for(op="matmul")


def test_explicit_effect_classes_accept_equivalent_selectors_without_discovery(monkeypatch):
    from merlin.targetgen import isa_taxonomy

    def forbidden(*args, **kwargs):
        raise AssertionError("explicit effect declarations must not discover an ISA provider")

    monkeypatch.setattr(isa_taxonomy, "derive_isa_taxonomy", forbidden)
    contract = {"encoding": {"semantic_class": {"17": "Execute", "29": "Execute"},
                             "corpus_required_classes": ["Execute"],
                             "class_family": {"Execute": "contraction"}}}
    classes = CS._classes_source(SimpleNamespace(target="synthetic"), contract)
    assert classes(op="matmul") == ["Execute"]
    assert classes(op="conv2d") == ["Execute"]
    with pytest.raises(ValueError, match="no required effect class"):
        classes(op="movement", movement=True)


def test_selected_readout_data_drives_commits_without_backend_discovery(monkeypatch):
    from merlin.targetgen import readout_facet

    def forbidden(*args, **kwargs):
        raise AssertionError("selected readout data must not discover an execution backend")

    monkeypatch.setattr(readout_facet, "_backend_hook", forbidden)
    contract = {
        "compute_units": [{"kind": "systolic", "dtypes": ["int8"],
                           "accumulate": [{"in": "int8", "weight": "int8", "acc": "i32"}]}],
        "encoding": {"semantic_class": {"17": "Execute"}, "corpus_required_classes": ["Execute"],
                     "class_family": {"Execute": "contraction"}},
        "runner": {"backend": "chipyard_rocc", "chipyard_rocc": {"readout": {
            "epilogue_capability": [{"selector": "i8", "applies": ["relu", "acc_scale"]},
                                   {"selector": "i32", "applies": []}],
            "stage_routes": [],
        }}},
    }
    binding = CS.derive_binding(
        SimpleNamespace(target="synthetic", sim_via="sim"), {"required_oracle_tiers": ["L0"]},
        contract=contract, facts={"facts": {"arrays": [{"name": "mesh", "rows": 8, "cols": 8}]}}, taxonomy={},
    )
    assert CS._resolve_output_dtype(binding, ["relu"]) == "i8"
    assert CS._resolve_output_dtype(binding, []) == "i32"
    with pytest.raises(ValueError, match="no readout applies"):
        CS._resolve_output_dtype(binding, ["relu"], {"output_dtype": "i32"})
    with pytest.raises(ValueError, match="no readout or contraction route"):
        CS._resolve_output_dtype(binding, ["bias_add"])
    entry = {"name": "reuse", "kind": "layer", "source_role": "synthesized",
             "source_reference": "independent readout declaration", "K": 8, "N": 8,
             "matmuls": [{"lhs": "A0", "out": "Y0", "M": 8, "epilogue": ["relu"]},
                         {"lhs": "A1", "out": "Y1", "M": 8, "epilogue": []}]}
    cap, mlir = CS.build_resident_reuse(entry, binding)
    assert [row["output_dtype"] for row in cap["operation"]["attributes"]["matmuls"]] == ["i8", "i32"]
    assert 'tensor<8x8xi8>' in mlir and 'tensor<8x8xi32>' in mlir


@pytest.mark.parametrize("change", [
    {"corpus_required_classes": []},
    {"corpus_required_classes": ["Execute", "Execute"]},
    {"class_family": {"Execute": "undeclared"}},
    {"semantic_class": {"17": "Different"}},
])
def test_malformed_effect_declaration_cannot_fall_back_to_issue_order(change):
    encoding = {"semantic_class": {"17": "Execute"}, "corpus_required_classes": ["Execute"],
                "class_family": {"Execute": "contraction"}, "corpus_issue_order": ["Execute"]}
    encoding.update(change)
    with pytest.raises(ValueError):
        CS._classes_source(SimpleNamespace(target="synthetic"), {"encoding": encoding})
