"""Scaled pooling needs both source decomposition and a separate SW declaration."""

from copy import deepcopy

import pytest

from merlin.targetgen.interface_observations import command_rows
from merlin.targetgen.operation_accounting import admit_operation_row


def _program(scale=0.5):
    return {
        "abi_version": "0.1",
        "target": "fixture",
        "tensors": {
            "X": {"role": "input", "shape": [50, 17], "dtype": "i8"},
            "Y": {"role": "output", "shape": [18, 17], "dtype": "i8"},
        },
        "commands": [
            {
                "opcode": "MAXPOOL2D",
                "operands": {"src": "X", "dst": "Y"},
                "attributes": {
                    "epilogue": ["acc_scale", "maxpool"],
                    "output_dtype": "i8",
                    "acc_scale": scale,
                    "pool_in_dims": [5, 5],
                    "pool_size": [3, 3],
                    "pool_stride": [2, 2],
                    "pool_padding": [1, 1, 1, 1],
                    "pool_pad_value": -128,
                },
            }
        ],
    }


def _inputs(*, movement_scale=False):
    spec = {
        "status": "reviewed",
        "operations": [
            {
                "id": "movement",
                "families": ["movement"],
                "placement": "accelerator",
                "signature": {
                    "dtypes": ["int8", "int32"],
                    "layouts": ["row_major_contiguous"],
                    "aliasing": ["disjoint_inputs_outputs"],
                },
            },
            {
                "id": "elementwise_map",
                "families": ["elementwise_map"],
                "placement": "fused_accelerator",
                "signature": {
                    "dtypes": ["int8", "int32"],
                    "epilogues": ["acc_scale", "bias_add", "relu"],
                    "composed_with": ["contraction"],
                    "scale_granularity": ["tensor"],
                },
            },
            {
                "id": "pooling_on_readout",
                "families": ["reduction"],
                "placement": "fused_accelerator",
                "signature": {"dtypes": ["int8", "int32"], "composed_with": ["contraction", "movement"]},
            },
        ],
    }
    if movement_scale:
        spec["operations"].append(
            {
                "id": "scaled_movement_readout",
                "families": ["elementwise_map"],
                "placement": "fused_accelerator",
                "signature": {
                    "dtypes": ["int8", "int32"],
                    "epilogues": ["acc_scale"],
                    "composed_with": ["movement"],
                    "scale_granularity": ["tensor"],
                },
            }
        )
    contract = {
        "name": "fixture",
        "compute_units": [
            {
                "name": "unit",
                "kind": "systolic",
                "dtypes": ["int8"],
                "ops": ["matmul"],
                "semantic_capabilities": [
                    {"family": "movement", "dtypes": ["int8"], "forms": ["copy"]},
                    {"family": "elementwise_map", "dtypes": ["int8"]},
                    {"family": "reduction", "dtypes": ["int8"], "composed_with": ["contraction", "movement"]},
                ],
            }
        ],
    }
    return spec, contract


def _admit(rows, *, movement_scale=False):
    spec, contract = _inputs(movement_scale=movement_scale)
    return [admit_operation_row(row, software_spec=spec, capability_contract=contract) for row in rows]


@pytest.mark.parametrize("scale", [0.5, 0.9900199770927429])
def test_same_public_macro_requires_a_new_narrow_declaration(scale):
    program = _program(scale)
    original = deepcopy(program)
    rows = command_rows(program)
    assert program == original
    assert [r["operation"] for r in rows] == ["movement", "acc_scale", "maxpool"]
    assert [r["component_index"] for r in rows] == [0, 1, 2]
    assert [r["component_input_dtype"] for r in rows] == ["i8", "i32", "i8"]
    assert [r["component_output_dtype"] for r in rows] == ["i32", "i8", "i8"]
    assert all(r["ordinals"] == [0] and r["command_opcode"] == "MAXPOOL2D" for r in rows)
    assert all(r["source_macro"]["attributes"] == program["commands"][0]["attributes"] for r in rows)
    assert all(r["composed_observation"]["composed_with"] == ["movement"] for r in rows[1:])
    old = _admit(rows)
    assert [r["hardware_admission"]["status"] for r in old] == ["admitted"] * 3
    assert [r["accelerator_admission"]["status"] for r in old] == ["admitted", "unsupported", "admitted"]
    new = _admit(rows, movement_scale=True)
    assert all(r["accelerator_admission"]["status"] == "admitted" for r in new)


@pytest.mark.parametrize(
    "change",
    [
        {"epilogue": []},
        {"epilogue": ["maxpool"]},
        {"epilogue": ["maxpool", "acc_scale"]},
        {"epilogue": ["bias_add", "acc_scale", "maxpool"]},
        {"epilogue": ["acc_scale", "relu", "maxpool"]},
        {"bias": "B"},
        {"acc_scale": [0.5]},
        {"acc_scale": float("nan")},
        {"acc_scale": 10**400},
        {"pool_in_dims": [4, 5]},
        {"pool_stride": [0, 2]},
        {"pool_pad_value": None},
        {"pool_pad_value": 128},
        {"output_dtype": "i32"},
    ],
)
def test_other_epilogues_or_unresolved_geometry_do_not_get_the_decomposition(change):
    program = _program()
    program["commands"][0]["attributes"].update(change)
    rows = command_rows(program)
    assert all("source_macro" not in r for r in rows)
    assert not all(r["accelerator_admission"]["status"] == "admitted" for r in _admit(rows, movement_scale=True))


@pytest.mark.parametrize("mutation", ["alias", "wrong_output", "wrong_dtype", "other_opcode"])
def test_only_a_complete_disjoint_signed_pool_macro_is_decomposed(mutation):
    program = _program()
    if mutation == "alias":
        program["commands"][0]["operands"]["dst"] = "X"
    elif mutation == "wrong_output":
        program["tensors"]["Y"]["shape"] = [9, 17]
    elif mutation == "wrong_dtype":
        program["tensors"]["X"]["dtype"] = "i32"
    else:
        program["commands"][0]["opcode"] = "REDUCE_MAX"
    assert all("source_macro" not in row for row in command_rows(program))


def test_new_grant_does_not_admit_bias_relu_or_plain_standalone_reduction():
    rows = command_rows(_program())
    for stage in ("bias_add", "relu"):
        row = deepcopy(rows[1])
        row["operation"] = stage
        row["mlir_operation"] = "merlin_iface." + stage
        row["composed_observation"]["epilogues"] = [stage]
        assert _admit([row], movement_scale=True)[0]["accelerator_admission"]["status"] == "unsupported"
    row = deepcopy(rows[2])
    row["composed_observation"]["composed_with"] = []
    assert _admit([row], movement_scale=True)[0]["accelerator_admission"]["status"] == "unsupported"
    # No direct ops selector eclipses the existing contraction stage grant.
    row = deepcopy(rows[1])
    row["composed_observation"]["composed_with"] = ["contraction"]
    old, new = _admit([row])[0], _admit([row], movement_scale=True)[0]
    assert old["accelerator_admission"]["status"] == new["accelerator_admission"]["status"] == "admitted"
    assert "elementwise_map" in new["matching_declarations"]
