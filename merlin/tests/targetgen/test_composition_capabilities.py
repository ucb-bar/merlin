"""Phase 0 projects existing readout facts without inventing executable support."""

import pytest

from merlin.targetgen.operation_capabilities import (
    derive_readout_operation_contract,
    merge_operation_observations,
)


def _manifest(configuration="selected-config"):
    return {
        "runner": {"chipyard_rocc": {"config": configuration}},
        "readout_semantics": {
            "epilogue_capability": [
                {"selector": "narrow", "applies": ["acc_scale", "relu"]},
                {"selector": "raw", "applies": []},
            ],
            "scalar_abi": {"accumulator_dtype": "i32", "output_dtype": "i8"},
        },
    }


def test_projection_preserves_numerical_rules_but_leaves_order_and_execution_unknown():
    registry = derive_readout_operation_contract(_manifest(), dialect="unit_ops")
    scaled, raw = registry["operations"]
    assert scaled["domain"] == "composition" and scaled["status"] == "unknown"
    assert scaled["semantics"]["configuration"] == "selected-config"
    assert scaled["semantics"]["numerical_rules"]["accumulator_dtype"] == "i32"
    assert scaled["semantics"]["ordered_stage_templates"] is None
    assert raw["semantics"]["applies"] == []
    assert derive_readout_operation_contract({}, dialect="other")["operations"] == []


@pytest.mark.parametrize("configuration", [None, "different-config"])
def test_positive_observations_cannot_cross_configurations(configuration):
    registry = derive_readout_operation_contract(_manifest(), dialect="unit_ops")
    with pytest.raises(ValueError, match="selected configuration"):
        merge_operation_observations(
            registry,
            [
                {
                    "domain": "composition",
                    "dialect": "unit_ops",
                    "operation": "readout:narrow",
                    "status": "supported",
                    "configuration": configuration,
                }
            ],
        )


def test_matching_observation_refines_only_its_exact_composition():
    registry = derive_readout_operation_contract(_manifest(), dialect="unit_ops")
    refined = merge_operation_observations(
        registry,
        [
            {
                "domain": "composition",
                "dialect": "unit_ops",
                "operation": "readout:narrow",
                "status": "supported",
                "configuration": "selected-config",
                "evidence": {"kind": "probe", "detail": "synthetic execution observation"},
            }
        ],
    )
    assert [row["status"] for row in refined["operations"]] == ["supported", "unknown"]


def test_a_configuration_must_exist_before_a_positive_observation():
    registry = derive_readout_operation_contract(_manifest(None), dialect="unit_ops")
    with pytest.raises(ValueError, match="selected configuration"):
        merge_operation_observations(
            registry,
            [
                {
                    "domain": "composition",
                    "dialect": "unit_ops",
                    "operation": "readout:narrow",
                    "status": "supported",
                    "configuration": None,
                }
            ],
        )


@pytest.mark.parametrize("evidence", [None, {"kind": "declaration", "detail": "RTL register exists"}])
def test_declarations_alone_cannot_qualify_executable_compositions(evidence):
    registry = derive_readout_operation_contract(_manifest(), dialect="unit_ops")
    with pytest.raises(ValueError, match="evidence"):
        merge_operation_observations(
            registry,
            [
                {
                    "domain": "composition",
                    "dialect": "unit_ops",
                    "operation": "readout:narrow",
                    "status": "supported",
                    "configuration": "selected-config",
                    "evidence": evidence,
                }
            ],
        )
