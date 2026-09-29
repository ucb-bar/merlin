"""Quantization authoring, selected hardware recipes, and observed admission join."""

import json
from copy import deepcopy

import pytest

from merlin.targetgen import quant_recipe, readout_facet, software_spec
from merlin.targetgen.quantization_spec import (
    build_quantization_contract,
    capture_recipe_candidates,
    validate_quantization_declarations,
)


def test_mx_software_spec_keeps_three_formats_separate_and_epilogues_on_host():
    from merlin.common.paths import repo_root

    spec = software_spec.load_software_spec(
        repo_root() / "examples/mx_gemmini/target/software-spec.yaml", target="mx_gemmini"
    )
    assert spec["status"] == "unreviewed"
    declarations = validate_quantization_declarations(spec)
    assert {row["operand_dtype"] for row in declarations} == {"mxfp8", "mxfp6", "mxfp4"}
    expected_site_modes = {
        "linear": {"lhs": "dynamic", "rhs": "static"},
        "functional_matmul": {"lhs": "dynamic", "rhs": "dynamic"},
    }
    assert all(row["site_modes"] == expected_site_modes for row in declarations)
    contract = build_quantization_contract(spec, {"target": "mx_gemmini"})
    assert all(
        row["unselected_parameters"]["site_modes"]["value"] == expected_site_modes
        for row in contract["formats"]
    )
    missing_scale_rule = deepcopy(spec["numerical_semantics"])
    del missing_scale_rule["scale_rule"]
    with pytest.raises(ValueError, match="scale_rule"):
        software_spec.validate_numerical_semantics(missing_scale_rule)
    for dtype, tile in (("mxfp8", 16), ("mxfp6", 32), ("mxfp4", 32)):
        signature = {
            "operand_dtype": dtype,
            "accum_dtype": "bf16",
            "rank": 2,
            "layout": "row_major_contiguous",
            "tails": "none",
            "broadcasting": "none",
            "aliasing": "disjoint_inputs_outputs",
            "dimensions": {"M": tile, "N": tile, "K": 32},
        }
        selected = software_spec.admit_operation(spec, "matmul", signature, "accelerator")
        assert selected["declaration"] == f"contraction_{dtype}"
        assert selected["status"] == "unknown"  # authored, not yet RTL-reviewed
        assert selected["constraints_status"] == "matched"
        bad = software_spec.admit_operation(
            spec, "matmul", {**signature, "dimensions": {"M": tile, "N": tile, "K": 31}}, "accelerator"
        )
        assert bad["status"] == "unsupported"
    relu = software_spec.admit_operation(
        spec, "relu", {"operand_dtype": "bf16"}, "accelerator"
    )
    assert relu["status"] == "unsupported"
    host_relu = software_spec.admit_operation(
        spec, "relu", {"operand_dtype": "f32"}, "host"
    )
    assert host_relu["constraints_status"] == "matched"
    assert host_relu["status"] == "unknown"  # host policy still needs review
    host_norm = software_spec.admit_operation(
        spec, "layernorm", {"operand_dtype": "f32"}, "host"
    )
    assert host_norm["constraints_status"] == "matched"
    assert host_norm["status"] == "unknown"


def test_site_quantization_modes_refuse_ambiguous_global_or_malformed_policy():
    spec = _spec()
    row = spec["quantization"]["formats"][0]
    row["site_modes"] = {"linear": {"lhs": "dynamic", "rhs": "static"}}
    row["weight_mode"] = "static"
    with pytest.raises(ValueError, match="cannot coexist"):
        validate_quantization_declarations(spec)
    row.pop("weight_mode", None)
    row.pop("activation_mode", None)
    row["site_modes"] = {"linear": {"lhs": "dynamic"}}
    with pytest.raises(ValueError, match="declare lhs and rhs"):
        validate_quantization_declarations(spec)
    row["site_modes"] = {"linear": {"lhs": "dynamic", "rhs": "automatic"}}
    with pytest.raises(ValueError, match="static, dynamic, or unknown"):
        validate_quantization_declarations(spec)


def _spec():
    return {
        "schema": software_spec.SCHEMA,
        "target": "test_device",
        "status": "reviewed",
        "capability_contract": {"name": "test_device", "compute_units": []},
        "numerical_semantics": {
            "model": {"engine": "integer_reference"},
            "operand_dtype": "int8",
            "accumulator_dtype": "i32",
            "readout_dtype": "i32",
            "subnormal_operand_flush": False,
            "overflow": "wrap",
            "rounding": "rne",
        },
        "operations": [
            {
                "id": "matrix",
                "ops": ["matmul"],
                "placement": "accelerator",
                "families": ["contraction"],
                "signature": {"operand_dtypes": ["int8", "fp32"], "accumulator_dtype": "i32", "ranks": [2]},
            },
            {"id": "recurrent", "ops": ["lstm"], "placement": "host", "signature": {"ranks": [2]}},
        ],
        "quantization": {
            "formats": [
                {"operand_dtype": "int8", "accumulator_dtype": "i32", "eligible_operations": ["matrix", "recurrent"]}
            ]
        },
        "evidence": {"unknowns": []},
    }


def _hardware(formats=None, *, element="i8", accumulator="i32"):
    formats = formats or ["int8"]
    contract = {
        "name": "test_device",
        "compute_units": [
            {
                "name": "unit",
                "kind": "systolic",
                "dtypes": formats,
                "semantic_capabilities": [{"family": "contraction", "dtypes": formats}],
            }
        ],
    }
    facet = readout_facet.ReadoutFacet(
        target="test_device",
        unit="unit",
        element_dtype=element,
        accumulator_dtype=accumulator,
        scale_granularities=("tensor",),
        scale_dtype="fp32",
        zero_point_carried=False,
        clamp=(-128, 127),
    )
    return {
        "contract": contract,
        "quantization_candidates": [row.to_dict() for row in quant_recipe.derive_candidates(contract, [facet])],
        "readout_facets": [facet.to_dict()],
        "readout_numerics": [facet.numerics_handoff()],
    }


def _accounting():
    return {
        "applications": [
            {
                "id": "observed_model",
                "signatures": [
                    {
                        "operation": "linalg.matmul",
                        "count": 3,
                        "matching_declarations": ["matrix"],
                        "observed_admission_signature": {
                            "family": "contraction",
                            "operand_dtype": "int8",
                            "accum_dtype": "i32",
                            "rank": 2,
                        },
                        "classification": "accelerator_candidate",
                        "hardware_admission": {
                            "status": "admitted",
                            "units": ["unit"],
                            "basis": "selected declared hardware signature",
                        },
                    }
                ],
            }
        ]
    }


def test_selected_recipe_and_observed_admission_do_not_license_host_or_framework():
    report = build_quantization_contract(_spec(), _hardware(), _accounting())
    row = report["formats"][0]
    assert row["id"] == "int8__int32"
    assert row["status"] == "candidate"
    assert [operation["status"] for operation in row["operation_eligibility"]] == ["candidate", "ineligible"]
    assert row["hardware_matches"][0]["parameters"]["weight_granularity"]["value"] == "tensor"
    assert row["framework_route"] == "not_evaluated"
    assert row["operation_eligibility"][0]["layer_eligibility"] == "not_evaluated"
    assert report["torchao_adapter"]["source_modification"] is False
    json.dumps(report, allow_nan=False)
    assert build_quantization_contract(None, _hardware())["status"] == "not_configured"
    assert (
        build_quantization_contract(_spec(), _hardware())["formats"][0]["operation_eligibility"][0]["status"]
        == "unknown"
    )
    accounting = _accounting()
    accounting["applications"][0]["signatures"][0]["hardware_admission"]["units"] = ["different_unit"]
    assert (
        build_quantization_contract(_spec(), _hardware(), accounting)["formats"][0]["operation_eligibility"][0][
            "status"
        ]
        == "unknown"
    )
    accounting = _accounting()
    accounting["applications"][0]["signatures"][0]["classification"] = "component"
    assert (
        build_quantization_contract(_spec(), _hardware(), accounting)["formats"][0]["operation_eligibility"][0][
            "status"
        ]
        == "ineligible"
    )


def test_scale_encoding_alias_matches_selected_carrier_without_accepting_different_format():
    spec = _spec()
    spec["quantization"]["formats"][0]["scale_encoding"] = "fp32"
    hardware = _hardware()
    hardware["readout_facets"][0]["scale"]["dtype"] = "f32"

    match = build_quantization_contract(spec, hardware, _accounting())["formats"][0]["hardware_matches"][0]
    assert match["status"] == "candidate"
    assert match["conflicts"] == []
    assert match["parameters"]["scale_encoding"]["value"] == "fp32"

    spec["quantization"]["formats"][0]["scale_encoding"] = "bf16"
    mismatch = build_quantization_contract(spec, hardware, _accounting())["formats"][0]["hardware_matches"][0]
    assert mismatch["status"] == "incompatible"
    assert "scale_encoding" in mismatch["conflicts"][0]


def test_multi_format_candidates_never_borrow_the_selected_format_readout():
    spec = _spec()
    spec["quantization"]["formats"].append(
        {
            "id": "float_alternative",
            "operand_dtype": "fp32",
            "accumulator_dtype": "i32",
            "eligible_operations": ["matrix"],
        }
    )
    report = build_quantization_contract(spec, _hardware(["int8", "fp32"]), _accounting())
    assert [row["status"] for row in report["formats"]] == ["candidate", "unknown"]
    alternative = report["formats"][1]
    assert alternative["operation_eligibility"][0]["status"] == "unknown"
    assert "hardware.quantization_recipe" in alternative["hardware_matches"][0]["unknowns"]
    assert alternative["hardware_matches"][0]["candidate"]["recipe"] is None
    float_hardware = _hardware(["fp32"], element="f32")
    alternative = build_quantization_contract(spec, float_hardware, _accounting())["formats"][1]
    assert alternative["hardware_matches"][0]["candidate"]["status"] == "derived"
    assert "numerical_semantics.format_specific_selection" in alternative["hardware_matches"][0]["unknowns"]
    assert alternative["status"] == "unknown"


def test_float_semantics_and_unknowns_are_explicit_and_refusals_fail_closed():
    spec = _spec()
    spec["numerical_semantics"] = {
        "model": {"engine": "specir_fp_reduce"},
        "operand_dtype": "fp32",
        "accumulator_dtype": "bf16",
        "readout_dtype": "bf16",
        "subnormal_operand_flush": False,
        "product_rounding": "accumulator_format",
        "rounding": "rne",
        "reduction_order": "tree",
        "reduction_cadence": "per_step",
    }
    spec["quantization"]["formats"] = [
        {"operand_dtype": "fp32", "accumulator_dtype": "bf16", "eligible_operations": ["matrix"]}
    ]
    spec["operations"][0]["signature"]["accumulator_dtype"] = "bf16"
    accounting = _accounting()
    signature = accounting["applications"][0]["signatures"][0]
    signature["observed_admission_signature"].update(operand_dtype="fp32", accum_dtype="bf16")
    hardware = _hardware(["fp32"], element="f32", accumulator="bf16")
    assert build_quantization_contract(spec, hardware, accounting)["formats"][0]["status"] == "candidate"
    spec["quantization"]["formats"][0]["scale_encoding"] = "unknown until reviewed"
    report = build_quantization_contract(spec, hardware, accounting)
    assert report["formats"][0]["status"] == "unknown"
    assert "scale_encoding" in report["formats"][0]["hardware_matches"][0]["unknowns"]
    spec["quantization"]["formats"][0].pop("scale_encoding")
    signature["hardware_admission"]["status"] = "unsupported"
    assert (
        build_quantization_contract(spec, hardware, accounting)["formats"][0]["operation_eligibility"][0]["status"]
        == "ineligible"
    )
    signature["hardware_admission"]["status"] = "admitted"
    spec["status"] = "unreviewed"
    assert (
        build_quantization_contract(spec, hardware, accounting)["formats"][0]["operation_eligibility"][0]["status"]
        == "unknown"
    )


def test_authoring_validation_rejects_bad_identity_references_and_parameters():
    assert validate_quantization_declarations(_spec())[0]["accumulator_dtype"] == "int32"
    spec = _spec()
    bad_rows = [
        {"eligible_operations": ["missing"]},
        {"eligible_operations": ["matrix", "matrix"]},
        {"operand_dtype": "invented_float"},
        {"block_size": 0},
        {"block_size": True},
        {"activation_zero_point": 1.5},
        {"subnormal_operand_flush": "yes"},
        {"activation_mode": "automatic"},
    ]
    for changes in bad_rows:
        bad = deepcopy(spec)
        bad["quantization"]["formats"][0].update(changes)
        with pytest.raises(ValueError):
            software_spec.validate_software_spec(bad)
    duplicate = deepcopy(spec)
    duplicate["quantization"]["formats"].append(
        {"operand_dtype": "i8", "accumulator_dtype": "int32", "eligible_operations": ["matrix"]}
    )
    with pytest.raises(ValueError, match="duplicate quantization format id"):
        validate_quantization_declarations(duplicate)
    spec["quantization"]["formats"][0]["activation_zero_point"] = 9
    match = build_quantization_contract(spec, _hardware(), _accounting())["formats"][0]["hardware_matches"][0]
    assert match["status"] == "incompatible"
    assert "activation_zero_point" in match["conflicts"][0]


def test_capture_recipe_is_scoped_without_manually_authored_framework_bookkeeping():
    spec = _spec()
    recipes = capture_recipe_candidates(spec, build_quantization_contract(spec, _hardware()))
    assert len(recipes) == 1
    recipe = recipes[0]["recipe"]
    assert recipe["families"] == ["contraction"]
    assert [row["id"] for row in recipe["software_admission"]["operations"]] == ["matrix"]
    assert recipe["software_admission"]["operations"][0]["ops"] == ["matmul"]
    assert recipe["activation"]["observer"] == "histogram"
    assert recipe["weight"]["observer"] == "minmax"
    assert recipe["software_numerical_engine"] == "integer_reference"
    assert recipe["framework_capture_policy"]["observer_epsilon"] > 0
    assert recipe["recipe_sha256"] == quant_recipe.digest(recipe)
    spec["quantization"]["formats"][0]["activation_zero_point"] = 1
    assert capture_recipe_candidates(spec, build_quantization_contract(spec, _hardware())) == []
    spec["quantization"]["formats"][0].pop("activation_zero_point")
    spec["quantization"]["formats"][0]["framework"] = {"calibration_dataset": "unbound"}
    with pytest.raises(ValueError, match="cannot enforce framework parameters"):
        capture_recipe_candidates(spec, build_quantization_contract(spec, _hardware()))


def test_fp8_capture_recipe_uses_supported_observer_default():
    spec = _spec()
    spec["numerical_semantics"].update(operand_dtype="fp8_e4m3", accumulator_dtype="bf16", readout_dtype="bf16")
    spec["operations"][0]["signature"].update(operand_dtypes=["fp8_e4m3"], accumulator_dtype="bf16")
    spec["quantization"]["formats"] = [
        {"operand_dtype": "fp8_e4m3", "accumulator_dtype": "bf16", "eligible_operations": ["matrix"]}
    ]
    hardware = _hardware(formats=["fp8_e4m3"], element="fp8_e4m3", accumulator="bf16")
    recipes = capture_recipe_candidates(spec, build_quantization_contract(spec, hardware))
    assert len(recipes) == 1
    assert recipes[0]["recipe"]["activation"]["observer"] == "minmax"
    assert recipes[0]["recipe"]["weight"]["observer"] == "minmax"


def test_unknown_block_size_does_not_block_a_derived_per_tensor_format():
    spec = _spec()
    spec["quantization"]["formats"][0]["block_size"] = "unknown"
    match = build_quantization_contract(spec, _hardware())["formats"][0]["hardware_matches"][0]
    assert match["status"] == "candidate"
    assert match["parameters"]["block_size"]["status"] == "not_applicable"
