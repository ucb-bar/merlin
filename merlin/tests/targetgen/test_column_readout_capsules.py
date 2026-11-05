"""Capsule builders preserve column gains through typed input and execution."""

import numpy as np
import pytest

from merlin.runtime.reference import reference_outputs
from merlin.runtime.simulator import simulate
from merlin.targetgen import corpus_spec as CS
from merlin.targetgen.contract import interface_emit as IE
from merlin.targetgen.contract.schemas import validate_command_buffer


def _binding():
    return CS.CorpusBinding(
        target="synthetic", tile_dim=4, operand_dtype="int8", accum_dtype="i32", integer=True,
        tiers=["L0", "L1", "L2", "L3"], compare="exact_int", requant_output_dtype="i8",
        classes_for=lambda **_: [],
    )


def _entry(op, columns):
    return {
        "name": "column_readout", "kind": "layer", "source_role": "derived_sweep", "source_reference": "test",
        "op": op, "M": 3, "K": 5, "N": columns, "Himg": 1, "Wimg": 3, "ci": 5, "kh": 1, "kw": 1,
        "epilogue": ["bias_add", "acc_scale", "relu"],
        "acc_scale_by_column": [0.5] * 4 + [0.125] * (columns - 4),
    }


@pytest.mark.parametrize("op", ["matmul", "conv2d"])
@pytest.mark.parametrize("columns", [5, 8, 11])
def test_capsule_builder_readout_survives_parsing_and_matches_original_tensor_arithmetic(op, columns):
    entry = _entry(op, columns)
    capsule, text = getattr(CS, f"build_{op}")(entry, _binding())
    cb = IE.parse_interface_mlir(text)
    validate_command_buffer(cb)
    command = next(c for c in cb["commands"] if c["opcode"] in ("COMMIT", "CONV2D"))
    attrs = command["attributes"]
    assert attrs["acc_scale_by_column"] == entry["acc_scale_by_column"]
    assert "acc_scale" not in attrs
    assert capsule["operation"]["attributes"]["acc_scale_by_column"] == attrs["acc_scale_by_column"]
    lhs = (np.arange(15).reshape(3, 5) % 9 - 4).astype(np.int64)
    weight = (np.arange(5 * columns).reshape(5, columns) % 7 - 3).astype(np.int64)
    bias = (np.arange(columns) * 3 - 9).astype(np.int64)
    inputs = {}
    for tensor in capsule["inputs"]:
        data = {"input": lhs, "weight": weight, "bias": bias}[tensor["role"]]
        inputs[tensor["name"]] = data.reshape(tensor["shape"]).tolist()
    expected = np.clip(np.maximum(np.rint(
        (lhs @ weight + bias).astype(np.float32) * np.asarray(entry["acc_scale_by_column"], dtype=np.float32)
    ), 0), -128, 127).astype(np.int64).tolist()
    output = attrs["out"] if "out" in attrs else command["operands"]["dst"]
    assert reference_outputs(cb, inputs=inputs)[output] == expected
    assert simulate(cb, inputs=inputs)["outputs"][output] == expected


@pytest.mark.parametrize("op", ["matmul", "conv2d"])
@pytest.mark.parametrize("change", [
    {"acc_scale": 1.0}, {"epilogue": []}, {"acc_scale_by_column": [0.5]},
    {"acc_scale_by_column": [float("nan")] * 5},
])
def test_capsule_builders_refuse_lost_ambiguous_and_invalid_column_parameters(op, change):
    with pytest.raises(ValueError, match="acc_scale|readout parameters"):
        getattr(CS, f"build_{op}")({**_entry(op, 5), **change}, _binding())
