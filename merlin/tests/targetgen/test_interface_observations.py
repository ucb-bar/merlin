"""Constraints a written interface program states are observed from it, never assumed."""

from __future__ import annotations

from merlin.targetgen.interface_observations import command_rows


def _cb(commands, tensors):
    return {"abi_version": "0.1", "target": "fixture", "tensors": tensors, "commands": commands}


_MATMUL = _cb(
    [
        {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "W_res"}, "attributes": {"layout": "packed_rhs"}},
        {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A0", "rhs": "W_res", "dst": "acc0"}},
        {
            "opcode": "COMMIT",
            "operands": {"src": "acc0", "dst": "Y0"},
            "attributes": {"epilogue": ["acc_scale", "relu"], "output_dtype": "i8", "acc_scale": 0.5},
        },
        {"opcode": "EVICT", "operands": {"handle": "W_res"}},
    ],
    {
        "W": {"shape": [32, 16], "dtype": "i8", "role": "weight"},
        "A0": {"shape": [16, 32], "dtype": "i8", "role": "input"},
    },
)


def test_a_resident_matmul_and_its_epilogue_are_observed_from_the_program():
    rows = command_rows(_MATMUL)
    matmul = next(row for row in rows if row["operation"] == "matmul")
    assert matmul["semantic_family"] == "contraction"
    assert matmul["contraction_shape"] == {"M": 16, "K": 32, "N": 16}
    assert (matmul["layout"], matmul["tails"], matmul["broadcasting"], matmul["aliasing"]) == (
        "row_major_contiguous",
        "zero_pad_valid_window",
        "none",
        "disjoint_inputs_outputs",
    )
    stages = [row for row in rows if row.get("composed_observation", {}).get("epilogues")]
    assert [row["operation"] for row in stages] == ["acc_scale", "relu"]
    assert all(row["composed_observation"]["composed_with"] == ["contraction"] for row in stages)
    assert all(row["composed_observation"]["scale_granularity"] == "tensor" for row in stages)
    assert all(row["operand_dtypes"] == ["i8"] for row in stages)
    assert {row["disposition"] for row in rows if row["command_opcode"] in {"RES_PACK", "EVICT"}} == {
        "support_required"
    }


def test_batches_and_standalone_maps_are_observed_as_written():
    batched = command_rows(
        _cb(
            [
                {
                    "opcode": "BATCHED_MATMUL",
                    "operands": {"a": "A0", "w": "W", "dst": "Y0"},
                    "attributes": {"batch": 2, "output_dtype": "i32"},
                }
            ],
            {
                "A0": {"shape": [2, 16, 32], "dtype": "i8", "role": "input"},
                "W": {"shape": [2, 32, 16], "dtype": "i8", "role": "weight"},
                "Y0": {"shape": [2, 16, 16], "dtype": "i32", "role": "output"},
            },
        )
    )[0]
    assert batched["semantic_family"] == "contraction" and batched["broadcasting"] == "independent_batches"
    standalone = command_rows(
        _cb(
            [{"opcode": "BIAS_ADD", "operands": {"src": "X", "bias": "B", "dst": "Y0"}, "attributes": {}}],
            {
                "X": {"shape": [16, 16], "dtype": "i32", "role": "input"},
                "B": {"shape": [16], "dtype": "i32", "role": "bias"},
                "Y0": {"shape": [16, 16], "dtype": "i32", "role": "output"},
            },
        )
    )[0]
    # A standalone command consumes declared tensors: composed with nothing, nobody's epilogue.
    assert standalone["composed_observation"] == {"epilogues": [], "composed_with": [], "scale_granularity": None}
    # A standalone sum that scales each operand by a scalar multiplier states a per-tensor scale.
    scaled = command_rows(
        _cb(
            [
                {
                    "opcode": "RESIDUAL_ADD",
                    "operands": {"lhs": "X0", "rhs": "X1", "dst": "Y0"},
                    "attributes": {"lhs_scale": 0.75, "rhs_scale": 0.25, "bound_lsb": 1, "output_dtype": "i8"},
                }
            ],
            {
                "X0": {"shape": [8, 32], "dtype": "i8", "role": "input"},
                "X1": {"shape": [8, 32], "dtype": "i8", "role": "input"},
                "Y0": {"shape": [8, 32], "dtype": "i8", "role": "output"},
            },
        )
    )[0]
    assert scaled["composed_observation"]["scale_granularity"] == "tensor"
    per_channel = command_rows(
        _cb(
            [
                {
                    "opcode": "RESIDUAL_ADD",
                    "operands": {"lhs": "X0", "rhs": "X1", "dst": "Y0"},
                    "attributes": {"lhs_scale": [0.5, 0.25], "rhs_scale": 0.25},
                }
            ],
            {
                "X0": {"shape": [8, 2], "dtype": "i8", "role": "input"},
                "X1": {"shape": [8, 2], "dtype": "i8", "role": "input"},
            },
        )
    )[0]
    assert per_channel["composed_observation"]["scale_granularity"] is None
    # A float contraction is not observed as zero-pad exact.
    floats = _cb(
        _MATMUL["commands"][:2],
        {
            "W": {"shape": [32, 16], "dtype": "f32", "role": "weight"},
            "A0": {"shape": [16, 32], "dtype": "f32", "role": "input"},
        },
    )
    assert next(row for row in command_rows(floats) if row["operation"] == "matmul")["tails"] is None


def test_named_command_epilogue_is_observed_as_a_fused_stage():
    rows = command_rows(
        _cb(
            [
                {
                    "opcode": "RESIDUAL_ADD",
                    "operands": {"lhs": "X0", "rhs": "X1", "dst": "Y0"},
                    "attributes": {
                        "epilogue": ["relu"],
                        "lhs_scale": 1.0625,
                        "rhs_scale": 0.3125,
                        "output_dtype": "i8",
                    },
                }
            ],
            {
                "X0": {"shape": [256, 8], "dtype": "i8", "role": "input"},
                "X1": {"shape": [256, 8], "dtype": "i8", "role": "input"},
                "Y0": {"shape": [256, 8], "dtype": "i8", "role": "output"},
            },
        )
    )
    assert [row["operation"] for row in rows] == ["residual_add", "relu"]
    assert rows[0]["composed_observation"]["epilogues"] == []
    assert rows[1]["composed_observation"] == {
        "epilogues": ["relu"],
        "composed_with": ["residual_add"],
        "scale_granularity": "tensor",
    }
    assert rows[1]["operand_dtypes"] == ["i8"]

    convolution = command_rows(
        _cb(
            [
                {
                    "opcode": "CONV2D",
                    "operands": {"ifm": "Image", "weight": "Weight", "dst": "Output"},
                    "attributes": {"epilogue": ["relu"], "output_dtype": "i8"},
                }
            ],
            {
                "Image": {"shape": [1, 8, 8, 8], "dtype": "i8", "role": "input"},
                "Weight": {"shape": [3, 3, 8, 16], "dtype": "i8", "role": "weight"},
                "Output": {"shape": [1, 8, 8, 16], "dtype": "i8", "role": "output"},
            },
        )
    )
    assert [row["operation"] for row in convolution] == ["conv2d", "relu"]
    assert convolution[1]["composed_observation"]["composed_with"] == ["contraction"]


def test_a_convolutions_geometry_is_not_mistaken_for_a_per_channel_scale():
    """CONV2D (and a pooled commit) carry integer geometry lists; their readout still scales by one value
    for the whole tensor. A list-valued multiplier is what would make it per-channel."""
    conv = {
        "opcode": "CONV2D",
        "operands": {"ifm": "IFM", "weight": "W_res", "dst": "Y0"},
        "attributes": {
            "kernel": [3, 3, 4, 8],
            "stride": [1, 1],
            "padding": [1, 1, 1, 1],
            "dilation": [1, 1],
            "pool_size": [2, 2],
            "pool_stride": [2, 2],
            "epilogue": ["bias_add", "acc_scale", "relu"],
            "output_dtype": "i8",
            "acc_scale": 0.25,
            "bias": "B",
        },
    }
    tensors = {
        "IFM": {"shape": [1, 8, 8, 4], "dtype": "i8", "role": "input"},
        "W": {"shape": [36, 8], "dtype": "i8", "role": "weight"},
        "B": {"shape": [8], "dtype": "i32", "role": "bias"},
    }
    pack = {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "W_res"}, "attributes": {"layout": "packed_conv_rhs"}}
    rows = command_rows(_cb([pack, conv], tensors))
    stages = [row for row in rows if row.get("composed_observation", {}).get("epilogues")]
    assert [row["operation"] for row in stages] == ["bias_add", "acc_scale", "relu"]
    assert all(row["composed_observation"]["scale_granularity"] == "tensor" for row in stages)
    assert all(row["composed_observation"]["composed_with"] == ["contraction"] for row in stages)
    per_channel = dict(conv, attributes={**conv["attributes"], "acc_scale": [0.25, 0.5]})
    rows = command_rows(_cb([pack, per_channel], tensors))
    stages = [row for row in rows if row.get("composed_observation", {}).get("epilogues")]
    assert all(row["composed_observation"]["scale_granularity"] is None for row in stages)
