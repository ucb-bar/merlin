"""The ordinary recipe derivation selects the optional source demand after software admission."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase0 import program_admission, requirements

from merlin.targetgen import column_readout_capsules as CR
from merlin.targetgen import quantization_spec


@pytest.mark.parametrize("profile", [None, CR.THREE_BLOCKS_V1, CR.DISTINCT_SCALES_V2, "unrecognized"])
def test_actual_recipe_derivation_forwards_the_explicit_profile(tmp_path, monkeypatch, profile):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump({} if profile is None else {"column_readout_profile": profile}))
    software = tmp_path / "software.yaml"
    software.write_text("{}\n")
    declaration = SimpleNamespace(recipe=recipe, descriptor=tmp_path / "descriptor.yaml")
    selected = SimpleNamespace(
        contract={"name": "synthetic"},
        software_spec={},
        host_capabilities={},
        refreshed_facts={},
        quantization_snapshot={},
        readout_facets=[],
        readout_numerics={},
    )
    full = {"applications": {}}
    digest = hashlib.sha256(json.dumps(full, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    row = {
        "op": "matmul",
        "dtype": "int8",
        "epilogue": ["bias_add", "acc_scale", "relu"],
        "signature": "matmul|independent_column_scale|int8|7|aligned",
        "column_readout": {
            "columns_per_scale": 7,
            "columns": 14,
            "case": "aligned",
            "logical_granularity": "per_channel",
        },
    }
    base = {"groups": {"required": [row]}, "scope": {}, "application_demands": {"full_inventory_sha256": digest}}
    monkeypatch.setattr(requirements, "from_definition", lambda _: declaration)
    monkeypatch.setattr(
        requirements,
        "load_target_experiment",
        lambda _: SimpleNamespace(target="synthetic", workload_spec={"applications": ["iteration"]}),
    )
    monkeypatch.setattr(requirements, "performance_scale_selection", lambda *a: {})
    monkeypatch.setattr(
        requirements,
        "load_spec",
        lambda _: SimpleNamespace(document={"phases": {"0": {"config": {}}}}, prohibited_instruction_roles=[]),
    )
    monkeypatch.setattr(requirements, "selected_software_spec_path", lambda *a: software)
    monkeypatch.setattr(requirements, "select_evidence", lambda *a, **k: selected)
    monkeypatch.setattr(requirements.application_inventory, "application_demand_inventory", lambda *a, **k: full)
    monkeypatch.setattr(requirements.conformance, "derive_spec", lambda *a, **k: copy.deepcopy(base))
    monkeypatch.setattr(requirements, "incomplete_inventory_blocker", lambda *a: None)
    monkeypatch.setattr(requirements, "unresolved_epilogue_blocker", lambda *a: None)
    monkeypatch.setattr(requirements, "typed_required_instances", lambda *a: [])
    monkeypatch.setattr(program_admission, "derive_inventory", lambda *a, **k: {"sha256": "a" * 64})
    monkeypatch.setattr(quantization_spec, "build_quantization_contract", lambda *a, **k: {})
    monkeypatch.setattr(quantization_spec, "capture_recipe_candidates", lambda *a: [])
    monkeypatch.setattr(requirements, "_validate_capture_recipes", lambda *a, **k: {})
    events, observed = [], []
    monkeypatch.setattr(requirements, "intersect_requirement", lambda req, *a: events.append("software") or req)
    original = CR.with_profile

    def select(req, value):
        events.append("profile")
        result = original(req, value)
        observed.append(result)
        return result

    monkeypatch.setattr(CR, "with_profile", select)

    class AfterProfile(Exception):
        pass

    def stop(*a):
        raise AfterProfile

    monkeypatch.setattr(requirements, "derive_performance_scope", stop)
    if profile == "unrecognized":
        with pytest.raises(ValueError, match="unknown column_readout_profile"):
            requirements.derive(
                "unused",
                {"iteration": tmp_path / "capture.mlir"},
                rtl_facts=tmp_path / "facts.json",
                output_root=tmp_path / "out",
            )
        assert not observed
    else:
        with pytest.raises(AfterProfile):
            requirements.derive(
                "unused",
                {"iteration": tmp_path / "capture.mlir"},
                rtl_facts=tmp_path / "facts.json",
                output_root=tmp_path / "out",
            )
        rows = observed[0]["groups"]["required"]
        assert rows[0] == row
        assert len(rows) == (1 if profile is None else 2)
        if profile is not None:
            assert rows[-1]["column_readout"]["columns"] == 21
    assert events == ["software", "profile"]
