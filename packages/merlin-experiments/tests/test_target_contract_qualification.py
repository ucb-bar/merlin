"""Coarse target-contract screening and actual native-cell qualification checks."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase0.cell_probe import characterize, verify_characterization
from merlin_experiments.phase0.software_screen import diagnostic_entry, intersect_requirement, screen_entry

from merlin.common.paths import repo_root
from merlin.targetgen.host_capabilities import validate_host_capabilities
from merlin.targetgen.operation_numerics import integer_partial_sum_bound
from merlin.targetgen.software_spec import (
    admit_operation,
    capability_contract,
    load_software_spec,
    validate_software_spec,
)
from merlin.targetgen.transfer_contracts import screen_transfer_contract, validate_transfer_contracts
from merlin.targetgen.target_experiment import load_target_experiment


def test_pre_emission_epilogue_matches_written_residual_carrier(monkeypatch):
    from merlin_experiments.phase0 import software_screen

    observed = []

    def record(_spec, op, signature, placement):
        observed.append((op, signature, placement))
        return {"status": "admitted", "constraints_status": "matched", "reason": "test"}

    monkeypatch.setattr(software_screen, "admit_operation", record)
    entry = {"op": "residual_add", "operand_dtype": "i8", "accum_dtype": "i32", "epilogue": ["relu"]}
    assert screen_entry({}, entry)["status"] == "admitted"
    stage, signature, placement = observed[-1]
    assert (stage, placement) == ("relu", "fused_accelerator")
    assert signature["operand_dtype"] == "i8"
    assert signature["composed_with"] == ["residual_add"]


def test_named_author_inputs_normalize_without_granting_review_or_mutating_bytes():
    from copy import deepcopy

    root = Path(__file__).resolve().parents[3]
    raw = yaml.safe_load((root / "examples/gemmini/target/software-spec.yaml").read_bytes())
    # The example is reviewed; normalization itself must never grant review to an unmarked spec.
    raw.pop("status", None)
    raw.pop("review", None)
    raw["transfer_contracts"].pop("status", None)
    for name in ("operand_load", "accumulator_readout"):
        raw["transfer_contracts"][name].pop("evidence", None)
    original = deepcopy(raw)
    selected = validate_software_spec(raw)
    assert raw == original and "status" not in raw
    assert selected["status"] == "unreviewed"
    assert selected["operations"][0]["id"] == "contraction"
    assert selected["operations"][0]["families"] == ["contraction", "window_mean"]
    assert selected["transfer_contracts"]["status"] == "unreviewed"
    assert selected["transfer_contracts"]["declarations"][0]["id"] == "operand_load"
    assert selected["transfer_contracts"]["declarations"][0]["evidence"] == {}
    assert validate_software_spec(selected) == selected
    assert (
        selected["operations"][0]["signature"]["operand_dtypes"] == (raw["operations"]["contraction"]["operand_dtypes"])
    )
    assert selected["transfer_contracts"]["declarations"][0]["signature"] == {
        "operand_dtype": "int8",
        "result_dtype": "int8",
        "operand_layout": "row_major_contiguous",
        "result_layout": "row_major_contiguous",
    }
    # Compact author inputs and expanded declarations share one consumer contract.
    legacy = deepcopy(raw)
    legacy["operations"] = deepcopy(selected["operations"])
    legacy["transfer_contracts"] = deepcopy(selected["transfer_contracts"])
    assert validate_software_spec(legacy) == selected
    for conflicting in ({"signature": {"ranks": [3]}}, {"rank": 3}):
        bad = deepcopy(raw)
        bad["operations"]["contraction"].update(conflicting)
        with pytest.raises(ValueError, match="mixes flat|unknown fields"):
            validate_software_spec(bad)
    for conflicting in ({"signature": {"operand_dtype": "f32"}}, {"copy": {"dtype": "int8"}}):
        bad = deepcopy(raw)
        bad["transfer_contracts"]["operand_load"].update(conflicting)
        with pytest.raises(ValueError, match="mix compact|explicit dtype and layout"):
            validate_software_spec(bad)
    with pytest.raises(ValueError, match="explicit declarations"):
        validate_transfer_contracts({"status": "reviewed"})
    bad = deepcopy(raw)
    bad["transfer_contracts"]["accumulator_readout"]["valid_window"] = "ignored"
    with pytest.raises(ValueError, match="valid_window"):
        validate_software_spec(bad)
    custom = deepcopy(raw)
    custom["operations"] = {"custom": {"placement": "accelerator", "signature": {"dtypes": ["int8"]}}}
    # Replacing the declaration roster also removes the target-specific declaration a restriction
    # named; this normalization fixture is about explicit ops/families, not stale restrictions.
    custom["restrictions"] = []
    custom["quantization"]["formats"][0]["eligible_operations"] = ["custom"]
    with pytest.raises(ValueError, match="explicit ops or families"):
        validate_software_spec(custom)
    custom["operations"]["custom"]["ops"] = ["matmul"]
    assert validate_software_spec(custom)["operations"][0]["ops"] == ["matmul"]
    reviewed = deepcopy(raw["transfer_contracts"])
    reviewed["status"] = "reviewed"
    with pytest.raises(ValueError, match="review evidence"):
        validate_transfer_contracts(reviewed)
    reviewed["operand_load"]["evidence"] = {"status": "unqualified"}
    reviewed["accumulator_readout"]["evidence"] = {"status": "unqualified"}
    assert (
        screen_transfer_contract(
            {"transfer_contracts": reviewed},
            source_placement="host",
            destination_placement="accelerator",
            operand_dtype="int8",
            result_dtype="int8",
            operand_layout="row_major_contiguous",
            result_layout="row_major_contiguous",
        )["status"]
        == "unknown"
    )


def test_backend_candidates_and_actual_nested_stages_respect_authored_sw_scope():
    from merlin.targetgen import spec_fact_drift as drift

    root = Path(__file__).resolve().parents[3]
    spec = load_software_spec(root / "examples/gemmini/target/software-spec.yaml")
    contract = yaml.safe_load((root / "examples/gemmini/target/contracts/target_contract.yaml").read_bytes())
    # A hardware: form is authored intent, not a screenable capability. Resolve it
    # against a small, explicit test fact view before testing the screen itself.
    facts = drift.fact_capabilities(
        target="gemmini",
        contract=contract,
        raw_facts={
            "facts": {
                "arrays": [
                    {"name": "mesh", "rows": 4, "cols": 4, "corroborated": True, "mac_idiom": {"muls": 1, "adds": 1}}
                ],
                "datapaths": [{"name": "input", "dtype": "i8"}],
                "interfaces": [{"name": "mesh_dma"}],
                "storage_datapaths": [{"name": "input", "dtype": "i8"}, {"name": "accumulator", "dtype": "i32"}],
            }
        },
        readout_facets=[
            {
                "unit": "mesh",
                "readouts": [
                    {"selector": "i8", "applies": ["acc_scale", "relu", "maxpool"], "evidence": "test fact view"}
                ],
                "operand_sum": {"operands": 2, "operand_dtype": "i8"},
                "unknown": {},
                "scale": {"granularities": ["tensor"], "granularities_complete": True, "carriers": []},
            }
        ],
        quantization_candidates=[
            {
                "status": "derived",
                "unit": "mesh",
                "format": "int8",
                "recipe": {"families": ["contraction", "operand_sum"]},
            }
        ],
        taxonomy={},
        # Fact-derived standalone forms (and the operand sum built on them) need the checked
        # semantic fact; an unchecked contract declaration is not evidence.
        semantic_facts={"facts": {"accumulate_on_load": {"status": "derived"}}},
    )
    spec, resolution = drift.resolve_spec(spec, facts)
    assert resolution["status"] == "resolved"
    defaults = spec["numerical_semantics"]
    signature = {
        "rank": 2,
        "layout": "row_major_contiguous",
        "tails": "zero_pad_valid_window",
        "broadcasting": "none",
        "aliasing": "disjoint_inputs_outputs",
    }
    entry = {
        "name": "probe",
        "op": "matmul",
        "cat": "isa",
        "operand_dtype": "int8",
        "placement": "accelerator",
        "operation_signature": signature,
    }
    assert screen_entry(spec, entry, defaults=defaults)["constraints_status"] == "matched"
    micro = {"name": "micro", "kind": "model", "micro_model": True}
    assert screen_entry(spec, micro, defaults=defaults)["status"] == "unknown"
    compound = {
        "name": "compound",
        "kind": "model_slice",
        "op": "host_island_seam",
        "performance": {"emitter": {"knobs": {"operation": "host_island_seam", "accelerator_regions": 2}}},
    }
    assert screen_entry(spec, compound, defaults=defaults)["status"] == "unknown"
    observed_compound = {
        "kind": "model_slice",
        "operation": {"op": "host_island_seam", "attributes": {"accelerator_contractions": 2}},
    }
    assert screen_entry(spec, compound, defaults=defaults, capsule=observed_compound)["status"] == "unknown"
    assert screen_entry(spec, {"op": "undeclared", "kind": "model_slice"}, defaults=defaults)["status"] == "unsupported"
    unknown_compound = {
        **compound,
        "op": "undeclared",
        "performance": {"emitter": {"knobs": {"operation": "undeclared", "accelerator_regions": 2}}},
    }
    assert screen_entry(spec, unknown_compound, defaults=defaults)["status"] == "unsupported"
    standalone_bias = {**entry, "kind": "model_slice", "op": "bias_add", "epilogue": ["bias_add"]}
    assert screen_entry(spec, standalone_bias, defaults=defaults)["status"] == "unsupported"
    # The fact view admits i8 maxpool after a contraction; an integer requant
    # is not one of its selected readout stages.
    assert screen_entry(spec, {**entry, "epilogue": ["maxpool"]}, defaults=defaults)["status"] == "admitted"
    refused = screen_entry(spec, {**entry, "epilogue": ["requant"]}, defaults=defaults)
    assert refused["status"] == "unsupported" and refused["decisions"][1]["constraints_status"] == "refused"
    assert (
        screen_entry(spec, {**entry, "kind": "model_slice", "epilogue": ["requant"]}, defaults=defaults)["status"]
        == "unsupported"
    )
    assert diagnostic_entry(entry, refused)["cat"] == "_diagnostic"
    actual = {
        "inputs": [{"role": "input", "dtype": "i8"}],
        "operation": {"op": "matmul", "attributes": {"epilogue": ["requant"], "output_dtype": "i8"}},
    }
    assert screen_entry(spec, entry, defaults=defaults, capsule=actual)["status"] == "unsupported"
    actual["operation"]["attributes"]["epilogue"] = ["bias_add"]
    bias = screen_entry(spec, entry, defaults=defaults, capsule=actual)
    assert bias["status"] == "unsupported" and bias["decisions"][1]["constraints_status"] == "refused"
    actual["inputs"][0]["dtype"] = "f32"
    assert screen_entry(spec, entry, defaults=defaults, capsule=actual)["status"] == "unsupported"
    host = {**entry, "operand_dtype": "f32", "generalization": {"must_accelerate": False, "eligible": False}}
    assert screen_entry(spec, host, defaults=defaults)["status"] == "unknown"
    assert screen_entry(spec, host, defaults=defaults)["scope"].startswith("explicit host-only")
    requirement = {
        "cells": [
            {"cell": "contraction/i8/aligned", "family": "contraction", "dtype": "i8"},
            {"cell": "reduction/i8/aligned", "family": "reduction", "dtype": "i8"},
        ],
        "epilogue": {"required": [{"stage": "requant"}, {"stage": "bias_add"}]},
        "application_demands": {"n_operations": 751},
    }
    selected = intersect_requirement(requirement, spec, contract)
    assert [row["family"] for row in selected["cells"]] == ["contraction", "reduction"]
    assert selected["epilogue"]["required"] == []
    assert len(selected["software_intersection"]["rejected_backend_capabilities"]) == 2
    assert selected["application_demands"] == requirement["application_demands"]
    assert len(requirement["cells"]) == 2


def test_example_contracts_are_typed_candidates_not_blanket_admission():
    root = repo_root()
    for target, operand, accumulator in (("gemmini", "int8", "i32"), ("atlas", "fp8_e4m3", "bf16")):
        selected = root / "examples" / target / "target"
        spec = load_software_spec(selected / "software-spec.yaml", target=target)
        assert "capability_contract" not in spec and "source_observations" not in spec
        with pytest.raises(ValueError, match="explicit same-target"):
            capability_contract(spec)
        with pytest.raises(ValueError, match="explicit same-target"):
            capability_contract(spec, base_contract={"name": "different"})
        backend = {"name": target, "opaque_provider_policy": {"protocol": ["selected"]}}
        projected = capability_contract(spec, base_contract=backend)
        projected["opaque_provider_policy"]["protocol"].append("mutated")
        assert backend["opaque_provider_policy"]["protocol"] == ["selected"]
        legacy = {**spec, "capability_contract": {"name": target, "legacy": True}}
        assert capability_contract(legacy, base_contract=backend) == legacy["capability_contract"]
        experiment = load_target_experiment(selected / "descriptor.yaml", source_root=root)
        host_path = root / experiment.host_lane.capability_spec
        host = validate_host_capabilities(yaml.safe_load(host_path.read_bytes()))
        assert host["operations"] and host["status"] in {"reviewed", "unreviewed"}
        # Review status is authored in the public document; session-specific
        # decision metadata is not part of the software capability contract.
        assert host["status"] == "unreviewed" or (host.get("review") or {}).get("decision")
        assert spec["status"] == "unreviewed" or (spec["operations"] and spec["numerical_semantics"])
        signature = {
            "family": "contraction",
            "operand_dtype": operand,
            "accum_dtype": accumulator,
            "rank": 2,
            "layout": "row_major_contiguous",
            "tails": "zero_pad_valid_window",
            "broadcasting": "none",
            "aliasing": "disjoint_inputs_outputs",
        }
        decision = admit_operation(spec, "matmul", signature, "accelerator")
        # Matching constraints admit only under a reviewed spec; otherwise the review stays unresolved.
        assert decision["constraints_status"] == "matched"
        assert decision["status"] == ("admitted" if spec["status"] == "reviewed" else "unknown")
        assert (
            admit_operation(spec, "matmul", {**signature, "operand_dtype": "f32"}, "accelerator")["status"]
            == "unsupported"
        )
        assert admit_operation(spec, "lstm", {**signature, "family": None}, "accelerator")["status"] == "unsupported"
        transfer = screen_transfer_contract(
            spec,
            source_placement="host",
            destination_placement="accelerator",
            operand_dtype=operand,
            result_dtype=operand,
            operand_layout="row_major_contiguous",
            result_layout="row_major_contiguous",
        )
        reviewed = (spec.get("transfer_contracts") or {}).get("status") == "reviewed"
        assert transfer["matching_declarations"] and transfer["status"] == ("admitted" if reviewed else "unknown")
    integer = load_software_spec(root / "examples/gemmini/target/software-spec.yaml")["numerical_semantics"]
    # i32 storage and a small final sum cannot rescue an overflowing i20 prefix.
    assert (
        integer_partial_sum_bound(integer, reduction_extent=31, lhs_values=[-128], rhs_values=[-128])["status"]
        == "proven_safe"
    )
    assert (
        integer_partial_sum_bound(integer, reduction_extent=32, lhs_values=[-128], rhs_values=[-128])["status"]
        == "may_overflow"
    )
    assert (
        integer_partial_sum_bound(integer, reduction_extent=1, lhs_values=[1], rhs_values=[1], initial_values=[524287])[
            "status"
        ]
        == "may_overflow"
    )


def test_native_cell_receipt_checks_differential_execution_and_artifact_integrity(tmp_path):
    circt, verilator = os.environ.get("MERLIN_TEST_CIRCT_OPT"), os.environ.get("MERLIN_TEST_VERILATOR")
    if not circt or not verilator:
        pytest.skip("explicit CIRCT/Verilator tool selection required for native characterization")
    source = tmp_path / "cell.mlir"
    source.write_text(
        "module {\n  hw.module @Cell(in %a: i8, in %b: i8, out d: i8) {\n"
        "    %sum = comb.add %a, %b : i8\n    hw.output %sum : i8\n  }\n}\n"
    )
    cases = [
        {"inputs": {"a": a, "b": b}, "expected": {"d": (a + b) & 255}}
        for a in (0, 127, 128, 255)
        for b in (0, 1, 128, 255)
    ]
    output = tmp_path / "good"
    result = characterize(
        hw_source=source,
        module="Cell",
        inputs={"a": 8, "b": 8},
        outputs={"d": 8},
        cases=cases,
        circt_opt=circt,
        verilator=verilator,
        output=output,
        reference={"engine": "independent Python modular sum"},
        domain={"vectors": "explicit signed-wrap boundaries"},
    )
    assert result["status"] == "passed", json.dumps(result)
    assert verify_characterization(output / "characterization.json")["whole_operation_support"] == "not_qualified"
    (output / "vectors.txt").write_text("0 0 1\n")
    with pytest.raises(ValueError, match="artifact changed"):
        verify_characterization(output / "characterization.json")
    wrong = [{"inputs": {"a": 0, "b": 0}, "expected": {"d": 1}}]
    mismatch = characterize(
        hw_source=source,
        module="Cell",
        inputs={"a": 8, "b": 8},
        outputs={"d": 8},
        cases=wrong,
        circt_opt=circt,
        verilator=verilator,
        output=tmp_path / "wrong",
        reference={"engine": "deliberately wrong counterfactual"},
        domain={"vectors": "one incorrect output"},
    )
    assert mismatch["status"] == "mismatch"
    with pytest.raises(ValueError, match="successful cell execution"):
        verify_characterization(tmp_path / "wrong/characterization.json")
