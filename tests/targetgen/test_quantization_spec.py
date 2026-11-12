from copy import deepcopy

import pytest

from merlin.compiler.quantization import capabilities as quant_recipe
from merlin.compiler.quantization.admission import (
    build_quantization_contract,
    capture_recipe_candidates,
    validate_quantization_declarations,
)
from merlin.targetgen import readout_facet, software_spec


def test_mx_software_spec_keeps_three_formats_separate_and_epilogues_on_host():
    from merlin.common.paths import repo_root

    spec = software_spec.load_software_spec(
        repo_root() / "examples/compiler-generation/mx_gemmini/target/software-spec.yaml", target="mx_gemmini"
    )
    assert spec["status"] == "unreviewed"
    declarations = validate_quantization_declarations(spec)
    assert {row["operand_dtype"] for row in declarations} == {"mxfp8", "mxfp6", "mxfp4"}
    expected_site_modes = {
        "linear": {"lhs": "dynamic", "rhs": "static"},
        "functional_matmul": {"lhs": "dynamic", "rhs": "dynamic"},
    }
    assert all((row["site_modes"] == expected_site_modes for row in declarations))
    contract = build_quantization_contract(spec, {"target": "mx_gemmini"})
    assert all(
        (row["unselected_parameters"]["site_modes"]["value"] == expected_site_modes for row in contract["formats"])
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
        assert selected["status"] == "unknown"
        assert selected["constraints_status"] == "matched"
        bad = software_spec.admit_operation(
            spec, "matmul", {**signature, "dimensions": {"M": tile, "N": tile, "K": 31}}, "accelerator"
        )
        assert bad["status"] == "unsupported"
    relu = software_spec.admit_operation(spec, "relu", {"operand_dtype": "bf16"}, "accelerator")
    assert relu["status"] == "unsupported"
    host_relu = software_spec.admit_operation(spec, "relu", {"operand_dtype": "f32"}, "host")
    assert host_relu["constraints_status"] == "matched"
    assert host_relu["status"] == "unknown"
    host_norm = software_spec.admit_operation(spec, "layernorm", {"operand_dtype": "f32"}, "host")
    assert host_norm["constraints_status"] == "matched"
    assert host_norm["status"] == "unknown"


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
