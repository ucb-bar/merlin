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
        row["unselected_parameters"]["site_modes"]["value"] == expected_site_modes for row in contract["formats"]
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
    relu = software_spec.admit_operation(spec, "relu", {"operand_dtype": "bf16"}, "accelerator")
    assert relu["status"] == "unsupported"
    host_relu = software_spec.admit_operation(spec, "relu", {"operand_dtype": "f32"}, "host")
    assert host_relu["constraints_status"] == "matched"
    assert host_relu["status"] == "unknown"  # host policy still needs review
    host_norm = software_spec.admit_operation(spec, "layernorm", {"operand_dtype": "f32"}, "host")
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


def test_operation_review_cannot_inherit_a_reviewed_document_status():
    spec = _spec()
    spec["operations"][0]["status"] = "unreviewed"
    decision = software_spec.admit_operation(
        spec,
        "matmul",
        {"family": "contraction", "operand_dtype": "int8", "accum_dtype": "i32", "rank": 2},
        "accelerator",
    )
    assert decision["constraints_status"] == "matched"
    assert decision["status"] == "unknown"
    assert decision["review_status"] == "unreviewed"
    spec["operations"][0]["status"] = "accepted"
    with pytest.raises(ValueError, match="invalid review status"):
        software_spec.validate_software_spec(spec, target="test_device")


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


def test_an_authored_calibration_count_reaches_the_recipe_and_its_digest():
    """The number of calibration samples changes the scales, so it is a recipe field (see
    test_recipe_calibration_count.py); absent, the recipe and its digest are what they were."""
    spec = _spec()
    unchanged = capture_recipe_candidates(spec, build_quantization_contract(spec, _hardware()))[0]["recipe"]
    assert "calibration_count" not in unchanged["framework_capture_policy"]
    spec["quantization"]["formats"][0]["framework"] = {"calibration_count": 260}
    recipe = capture_recipe_candidates(spec, build_quantization_contract(spec, _hardware()))[0]["recipe"]
    assert recipe["framework_capture_policy"]["calibration_count"] == 260
    assert recipe["recipe_sha256"] == quant_recipe.digest(recipe) != unchanged["recipe_sha256"]


@pytest.mark.parametrize("bad", [0, 1.5, "all", True])
def test_an_unusable_calibration_count_is_refused_at_derivation(bad):
    spec = _spec()
    spec["quantization"]["formats"][0]["framework"] = {"calibration_count": bad}
    with pytest.raises(ValueError, match="calibration_count"):
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


def test_a_declared_methodology_derives_each_capture_recipe_over_the_selected_output_path(tmp_path):
    """With ``quantization.methodology`` declared, the Phase 0 capture recipe is the methodology's
    recipe over the selected contract's derived output path; without that path it is underivable
    (never the hardware-only recipe), and a spec without a methodology is unchanged."""
    from merlin.targetgen import quant_methodology as QM

    spec = _spec()
    plain = capture_recipe_candidates(spec, build_quantization_contract(spec, _hardware()))[0]["recipe"]
    assert "methodology" not in build_quantization_contract(spec, _hardware())
    spec["quantization"]["methodology"] = {
        "scale_granularity": "from_output_path",
        "bias": "accumulator_preload_int32",
        "epilogue": "integer_domain",
        "rounding": "from_output_path",
        "clamp": "from_output_path",
        "calibration": {"activation": "mse", "weight": "minmax"},
        "recovery": {"cross_layer_equalization": True, "bias_correction": False},
        "float_island": {"dtype": "fp32", "ops": ["softmax"]},
        "model_variants": [
            {
                "model": "m",
                "name": "longer-calibration",
                "status": "reviewed",
                "reviewed_by": "test",
                "reason": "more samples",
                "measured_accuracy": {"top1_agreement": 0.9},
                "changes": {"calibration_count": 512},
            }
        ],
    }
    path = {
        "schema": "output_path_v1",
        "accumulator": {"dtype": "i32"},
        "accumulator_stages": [{"kind": "preload", "dtype": "i32", "range": [-(2**31), 2**31 - 1]}],
        "readout": [
            {"kind": "convert", "to": "f32", "rounding": "half_even"},
            {
                "kind": "scale",
                "dtype": "f32",
                "rounding": "half_even",
                "granularity": "per_command",
                "columns_per_scale": 8,
            },
            {"kind": "requant", "to": "i8", "rounding": "half_even", "clamp": [-128, 127]},
        ],
        "exits": {"accumulator": False, "requant": True},
    }
    hardware = _hardware()
    hardware["contract"] = {**hardware["contract"], "output_epilogue": {"output_path": path}}
    quantization = build_quantization_contract(spec, hardware)
    assert quantization["methodology"]["status"] == "selected"
    rows = capture_recipe_candidates(spec, quantization)
    recipe = rows[0]["recipe"]
    assert recipe["status"] == quant_recipe.DERIVED and recipe["recipe_sha256"] != plain["recipe_sha256"]
    assert recipe["weight"]["channels_per_scale"] == 8 and recipe["epilogue"]["activation"] is None
    assert recipe["epilogue"]["bias"]["mode"] == "accumulator_preload"
    assert recipe["software_admission"] == plain["software_admission"]
    assert recipe["methodology"]["output_path_sha256"] == QM.digest(path)
    (variant,) = rows[0]["model_variants"]
    assert variant["model"] == "m" and variant["recipe"]["framework_capture_policy"]["calibration_count"] == 512
    unresolved = build_quantization_contract(spec, _hardware())
    assert unresolved["methodology"]["status"] == "unresolved"
    assert capture_recipe_candidates(spec, unresolved)[0]["recipe"]["status"] == quant_recipe.UNDERIVABLE
    from merlin_experiments.phase0.capture_recipe import _selected_capture_recipe, recipe_outputs

    outputs = recipe_outputs(
        rows,
        target=spec["target"],
        software_review=spec["status"],
        encode=lambda value: json.dumps(value, sort_keys=True).encode(),
    )
    for member, raw in outputs.items():
        dest = tmp_path / member
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
    index = json.loads(outputs["software/quantization-recipes.json"])
    assert len(index["recipes"]) == len(index["model_variants"]) == 1
    assert json.loads(outputs[index["model_variants"][0]["path"]]) == variant["recipe"]
    assert (
        _selected_capture_recipe(tmp_path, target=spec["target"], operand_dtype="int8", accumulator_dtype="int32")
        == recipe
    )
    (tmp_path / index["recipes"][0]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="bytes differ"):
        _selected_capture_recipe(tmp_path, target=spec["target"], operand_dtype="int8", accumulator_dtype="int32")


def test_evidence_prose_is_not_an_unknown_numerical_parameter():
    from merlin.targetgen.quantization_spec import _unresolved

    semantics = {
        "domain": {
            "minimum": -1.0,
            "maximum": 1.0,
            "provenance": "authored fixed observer range; pending operator review",
            "evidence": {"review_status": "authored, pending operator review"},
        },
        "rounding": "unknown",
    }
    assert _unresolved(semantics, "numerical_semantics") == ["numerical_semantics.rounding"]


def test_every_real_software_spec_string_is_not_an_implicit_unknown():
    from merlin.common.paths import repo_root
    from merlin.targetgen.quantization_spec import _unknown, _unresolved

    spec = software_spec.load_software_spec(repo_root() / "examples/gemmini/target/software-spec.yaml")

    def strings(value, path=""):
        if isinstance(value, dict):
            for key, member in value.items():
                yield from strings(member, f"{path}.{key}")
        elif isinstance(value, list):
            for index, member in enumerate(value):
                yield from strings(member, f"{path}[{index}]")
        elif isinstance(value, str):
            yield path, value

    assert [(path, value) for path, value in strings(spec) if _unknown(value)] == []
    # This subtree is actually traversed by hardware candidate selection. The
    # calibration review basis must not suppress every static int8 recipe.
    assert _unresolved(spec["numerical_semantics"], "numerical_semantics") == []
    selected = _spec()
    selected["numerical_semantics"] = spec["numerical_semantics"]
    contract = build_quantization_contract(selected, _hardware())
    assert contract["formats"][0]["hardware_matches"][0]["status"] == "candidate"
    assert capture_recipe_candidates(selected, contract)


@pytest.mark.parametrize("text", ["Operator R6; reviewed convention", "(0, 1]", "source\nreviewed", "scale\tpolicy"])
def test_whitespace_is_not_an_unknown_value_marker(text):
    from merlin.targetgen.quantization_spec import _record, _unknown, _unresolved, unresolved_values

    assert not _unknown(text)
    assert _record(text, "authored")["status"] == "known"
    assert _unresolved({"future_field": text}, "parameters") == []
    assert unresolved_values({"future_field": text}) == []


@pytest.mark.parametrize(
    "field",
    [
        "evidence",
        "provenance",
        "policy_reviews",
        "basis",
        "note",
        "notes",
        "rationale",
        "description",
        "reason",
        "justification",
    ],
)
def test_annotation_subtrees_are_not_numeric_parameters_even_when_they_mention_unknown(field):
    from merlin.targetgen.quantization_spec import _unresolved, unresolved_values

    annotations = {field: {"text": ["UNKNOWN source width", "from_facts", None]}}
    value = {**annotations, "rounding": "unknown until reviewed", "nested": [None, "from_output_path"]}
    original = deepcopy(value)
    assert _unresolved(value, "parameters") == ["parameters.rounding", "parameters.nested[0]"]
    assert unresolved_values(value) == ["rounding", "nested[0]", "nested[1]"]
    assert value == original


@pytest.mark.parametrize("value", [None, "unknown", "UNKNOWN", "unknown until reviewed", "UNKNOWN source width"])
def test_explicit_unknown_parameters_still_block_candidates(value):
    spec = _spec()
    spec["quantization"]["formats"][0]["scale_encoding"] = value
    match = build_quantization_contract(spec, _hardware())["formats"][0]["hardware_matches"][0]
    assert match["status"] == "unknown"
    assert "scale_encoding" in match["unknowns"]


@pytest.mark.parametrize("field", ["basis", "notes", "rationale", "description", "policy_reviews"])
def test_format_annotations_are_retained_without_becoming_capture_parameters(field):
    spec = _spec()
    baseline = build_quantization_contract(spec, _hardware())
    text = "unknown historical detail; the numerical parameters are reviewed"
    spec["quantization"]["formats"][0][field] = text
    report = build_quantization_contract(spec, _hardware())
    row = report["formats"][0]
    assert row["declaration"][field] == text
    assert row["hardware_matches"] == baseline["formats"][0]["hardware_matches"]
    assert capture_recipe_candidates(spec, report) == capture_recipe_candidates(_spec(), baseline)


@pytest.mark.parametrize("field", ["block_size", "activation_zero_point", "subnormal_operand_flush", "activation_mode"])
def test_prose_cannot_satisfy_a_typed_quantization_parameter(field):
    spec = _spec()
    spec["quantization"]["formats"][0][field] = "a reviewed numerical parameter"
    with pytest.raises(ValueError, match=field):
        validate_quantization_declarations(spec)


def _schemes_spec():
    spec = _spec()
    spec["numerical_semantics"]["readout"] = {"strategy": "exact_rational_host", "requant_rounding": "half_even"}
    spec["quantization"]["schemes"] = {
        "schema": "merlin.quantization_schemes.v1",
        "status": "authored_pending_operator_review",
        "axes": {
            "scale_value": ["power_of_two", "arbitrary_f32"],
            "weight_scale_granularity": ["tensor", "column_block", "column"],
            "activation_scale_granularity": ["tensor"],
            "readout_exit": ["narrow_saturating", "accumulator_full"],
            "operand_sum_rounding": ["single_rounding", "per_operand"],
            "residual_coefficient": ["none", "integer_constant"],
        },
        "constraints": [
            {
                "when": {"scale_value": "arbitrary_f32"},
                "readout_strategy": ["f32_product_rne"],
                "oracle": {"spike_extension": "rounding_fixed"},
            },
            {"when": {"operand_sum_rounding": "per_operand"}, "operand_multiplier": {"min_exclusive": 0, "max": 1}},
            {"when": {"readout_exit": "accumulator_full"}, "forbids_stages": "from_facts"},
            {
                "when": {"residual_coefficient": "integer_constant"},
                "coefficient_digits": {"dtype": "from_facts", "max_k_extension": "from_facts"},
            },
        ],
        "schemes": [
            {
                "id": "baseline",
                "scale_value": "power_of_two",
                "weight_scale_granularity": "tensor",
                "operand_sum_rounding": "single_rounding",
            },
            *[
                {
                    "id": granularity,
                    "scale_value": "arbitrary_f32",
                    "weight_scale_granularity": granularity,
                    "operand_sum_rounding": "single_rounding",
                    "readout_strategy": "f32_product_rne",
                }
                for granularity in ("tensor", "column_block", "column")
            ],
            {
                "id": "residual",
                "scale_value": "arbitrary_f32",
                "weight_scale_granularity": "column_block",
                "operand_sum_rounding": "per_operand",
                "readout_strategy": "f32_product_rne",
                "weight_layout": {"kind": "packed_panel", "panel_width": "from_facts"},
            },
        ],
    }
    return spec


def _scheme_path(width):
    return {
        "schema": "output_path_v1",
        "accumulator": {"dtype": "i32"},
        "accumulator_stages": [],
        "readout": [
            {"kind": "convert", "to": "f32", "rounding": "half_even"},
            {
                "kind": "scale",
                "dtype": "f32",
                "rounding": "half_even",
                "granularity": "per_command",
                "columns_per_scale": width,
            },
            {"kind": "requant", "to": "i8", "rounding": "half_even", "clamp": [-128, 127]},
        ],
        "exits": {"accumulator": True, "requant": True},
    }


@pytest.mark.parametrize("width", [3, 7, 29])
def test_schemes_resolve_physical_width_and_grading_without_changing_baseline(width):
    from merlin.targetgen.spec_fact_drift import FactField, resolve_spec

    spec = _schemes_spec()
    original = deepcopy(spec)
    resolved, record = resolve_spec(
        spec,
        {
            "composition": {
                "output_path": _scheme_path(width),
                "weight_panel_width": FactField((width + 2,), True, ("provider layout fact",)),
                "dtype": FactField(("i8",), True, ("operand storage fact",)),
                "max_k_extension": FactField((width * 5,), True, ("capacity bound",)),
            }
        },
    )
    assert record["status"] == "resolved"
    assert spec == original
    rows = {row["id"]: row for row in resolved["quantization_scheme_resolution"]["schemes"]}
    baseline = rows["baseline"]
    assert baseline["readout_strategy"] == "exact_rational_host"
    assert baseline["readout_scale_granularity"] == "tensor"
    assert baseline["exactness_gate"]["proof"] == "prove_fp32_products"
    assert rows["column_block"]["columns_per_scale"] == width
    assert rows["column_block"]["graded"] is True
    assert rows["column"]["graded"] is False
    assert rows["column"]["realization"] == "output_scale_partition"
    residual = rows["residual"]
    assert residual["residual_scope"] == "every_residual_add"
    assert residual["weight_layout"]["panel_width"] == width + 2
    assert residual["weight_layout"]["logical_order"] == ["N/panel_width", "K", "panel_width"]
    assert residual["exactness_gate"]["definitional"] == "f32_product_rne"
    assert residual["oracle_requirement"] == {"spike_extension": "rounding_fixed"}


@pytest.mark.parametrize("width", [None, 0, -2, True, "unknown"])
def test_unresolved_scheme_geometry_is_listed_and_never_defaulted(width):
    from merlin.targetgen.spec_fact_drift import resolve_spec

    resolved, record = resolve_spec(_schemes_spec(), {"composition": {"output_path": _scheme_path(width)}})
    row = resolved["quantization_scheme_resolution"]["schemes"][2]
    assert row["columns_per_scale"] == "UNKNOWN"
    assert row["status"] == "unresolved"
    assert "columns_per_scale" in row["unknown"]
    assert record["status"] == "unresolved"
    assert any(
        item.get("scheme") == "column_block" and item["field"] == "columns_per_scale" for item in record["unresolved"]
    )
    assert any("coefficient_digits" in item["field"] for item in record["unresolved"])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda b: b["axes"]["scale_value"].append("model_specific"),
        lambda b: b["schemes"].append(deepcopy(b["schemes"][0])),
        lambda b: b["axes"].__setitem__("weight_scale_granularity", ["tensor"]),
        lambda b: b["schemes"][0].__setitem__("operation", "matmul"),
        lambda b: b["constraints"][0].__setitem__("tolerance", 1),
        lambda b: b["schemes"][-1]["weight_layout"].__setitem__("panel_width", 16),
        lambda b: b["schemes"][-1]["weight_layout"].pop("panel_width"),
        lambda b: b["schemes"][0].pop("operand_sum_rounding"),
        lambda b: b["constraints"][0].__setitem__("when", {}),
    ],
)
def test_scheme_axes_and_points_are_closed_and_fact_shaped(mutation):
    spec = _schemes_spec()
    mutation(spec["quantization"]["schemes"])
    with pytest.raises(ValueError):
        validate_quantization_declarations(spec)


def test_arbitrary_scales_cannot_select_the_rational_readout():
    from merlin.targetgen.spec_fact_drift import resolve_spec

    spec = _schemes_spec()
    spec["quantization"]["schemes"]["schemes"][1]["readout_strategy"] = "exact_rational_host"
    with pytest.raises(ValueError, match="arbitrary_f32 requires"):
        resolve_spec(spec, {})


def test_missing_strategy_and_unproved_panel_layout_remain_unknown():
    from merlin.targetgen.spec_fact_drift import FactField, resolve_spec

    spec = _schemes_spec()
    del spec["numerical_semantics"]["readout"]
    resolved, record = resolve_spec(
        spec,
        {
            "composition": {
                "output_path": _scheme_path(5),
                "weight_panel_width": FactField((19,), False, ("incomplete",)),
            }
        },
    )
    rows = resolved["quantization_scheme_resolution"]["schemes"]
    assert rows[0]["readout_strategy"] == "UNKNOWN"
    assert rows[-1]["weight_layout"]["panel_width"] == "UNKNOWN"
    assert record["status"] == "unresolved"


def test_contract_exposes_scheme_unknowns_and_legacy_contract_has_no_scheme_axis():
    baseline = build_quantization_contract(_spec(), _hardware())
    assert "schemes" not in baseline
    selected = build_quantization_contract(_schemes_spec(), _hardware())
    assert selected["schemes"]["unresolved"]
    assert selected["schemes"]["schemes"][2]["status"] == "unresolved"


def test_example_software_specs_validate_with_or_without_scheme_axes():
    from merlin.common.paths import repo_root

    paths = sorted((repo_root() / "examples").glob("*/target/software-spec.yaml"))
    assert paths
    legacy = software_spec.load_software_spec(
        repo_root() / "examples/gemmini/phase0/history/software-spec-seals1-3.yaml"
    )
    assert "schemes" not in legacy["quantization"] and "compositions" not in legacy
    software_spec.validate_software_spec(legacy)
    for path in paths:
        selected = software_spec.load_software_spec(path)
        if "schemes" in (selected.get("quantization") or {}):
            assert selected["quantization"]["schemes"]["schemes"]
            assert selected["compositions"]["grading"] == "placement_and_instruction_audit"


def test_new_schema_stage_vocabulary_matches_the_shared_output_path():
    from merlin.targetgen import output_path as OP
    from merlin.targetgen.contract.schemas import load_schema

    chain = load_schema("software_compositions")["properties"]["chains"]["items"]["properties"]
    assert set(chain["accumulator_stages"]["properties"]) == set(OP.ACCUMULATOR_KINDS)
    assert chain["readout_stages"]["oneOf"][1]["items"]["enum"] == list(OP.READOUT_KINDS)
    assert chain["exits"]["items"]["enum"] == list(OP.EXITS)


def test_new_axes_do_not_modify_legacy_resolution_or_recipe_candidates():
    from merlin.targetgen.spec_fact_drift import resolve_spec

    spec = _spec()
    resolved, record = resolve_spec(spec, {})
    assert resolved == spec
    assert record["unresolved"] == []
    baseline = capture_recipe_candidates(spec, build_quantization_contract(spec, _hardware()))
    declared = _schemes_spec()
    # An explicit scheme axis is prospective metadata; it does not silently select a new quantizer.
    declared["numerical_semantics"] = deepcopy(spec["numerical_semantics"])
    selected = capture_recipe_candidates(declared, build_quantization_contract(declared, _hardware()))
    assert selected == baseline


def test_missing_arbitrary_readout_strategy_is_listed_as_unknown():
    from merlin.targetgen.spec_fact_drift import resolve_spec

    spec = _schemes_spec()
    del spec["numerical_semantics"]["readout"]
    spec["quantization"]["schemes"]["schemes"][1].pop("readout_strategy")
    resolved, record = resolve_spec(spec, {})
    row = resolved["quantization_scheme_resolution"]["schemes"][1]
    assert row["status"] == "unresolved"
    assert row["readout_strategy"] == "UNKNOWN"
    assert any(item.get("scheme") == "tensor" and item["field"] == "readout_strategy" for item in record["unresolved"])


def test_full_accumulator_exit_constraints_and_digit_bounds_resolve_from_facts():
    from merlin.targetgen.spec_fact_drift import FactField, resolve_spec

    spec = _schemes_spec()
    row = spec["quantization"]["schemes"]["schemes"][0]
    row["readout_exit"] = "accumulator_full"
    row["residual_coefficient"] = "integer_constant"
    resolved, _ = resolve_spec(
        spec,
        {
            "composition": {
                "output_path": _scheme_path(5),
                "dtype": FactField(("i8",), True, ("digit storage",)),
                "max_k_extension": FactField((11,), True, ("mesh bound",)),
            }
        },
    )
    selected = resolved["quantization_scheme_resolution"]["schemes"][0]
    assert selected["forbids_stages"] == ["convert", "scale", "requant"]
    assert selected["coefficient_digits"] == {"dtype": "i8", "max_k_extension": 11}


def test_runtime_scale_is_an_explicit_axis_on_the_existing_tensor_scheme():
    import yaml

    from merlin.common.paths import repo_root
    from merlin.targetgen.quantization_spec import resolve_quantization_schemes, select_scheme_axes

    spec = yaml.safe_load((repo_root() / "examples/gemmini/target/software-spec.yaml").read_text())
    block = spec["quantization"]["schemes"]
    context = dict(
        output_path={"readout": [{"kind": "scale", "granularity": "per_command", "columns_per_scale": 3}]},
        dtype="int8",
        max_k_extension=256,
        weight_panel_width=3,
    )
    resolved, _ = resolve_quantization_schemes(spec, context)
    constant = next(row for row in resolved["schemes"] if row["id"] == "q_f32_tensor")
    runtime = select_scheme_axes(constant, block["axes"], {"scale_source": "runtime_input"})
    assert constant["scale_source"] == "constant"
    assert runtime == {**constant, "scale_source": "runtime_input"}
    assert len(resolved["schemes"]) == len(block["schemes"])
    with pytest.raises(ValueError, match="outside the declared domain"):
        select_scheme_axes(constant, block["axes"], {"scale_source": "unknown"})
    with pytest.raises(ValueError, match="undeclared quantization axis"):
        select_scheme_axes(constant, block["axes"], {"scale_operand": "runtime_input"})
    v1 = next(row for row in resolved["schemes"] if row["id"] == "q_pow2_tensor")
    with pytest.raises(ValueError, match="arbitrary_f32 tensor"):
        select_scheme_axes(v1, block["axes"], {"scale_source": "runtime_input"})
