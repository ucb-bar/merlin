"""COMMIT applies its epilogue in the order listed, so the RTL's ReLU-before-scale readout is expressible.

The device readout of an int8 contraction is ``COMMIT {epilogue: [bias, relu, acc_scale], output_dtype:
i8}``: the int32 bias is in the accumulator, the ReLU acts on the accumulator, then one binary32 scale
rounded half to even and saturated. Both engines must execute exactly that order (and, for a positive
scale, it agrees with the commuted order).
"""

from __future__ import annotations

import numpy as np

from merlin.runtime.reference import reference_outputs
from merlin.runtime.simulator import simulate


def _buffer(epilogue: list[str], scale: float) -> dict:
    return {
        "abi_version": "0.1",
        "target": "t",
        "backend": "simulator",
        "tensors": {
            "A": {"shape": [4, 8], "dtype": "i8", "role": "input"},
            "W": {"shape": [8, 4], "dtype": "i8", "role": "weight"},
            "B": {"shape": [4], "dtype": "i32", "role": "bias"},
            "Y": {"shape": [4, 4], "dtype": "i8", "role": "output"},
        },
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "Wr"}, "attributes": {"layout": "packed_rhs"}},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A", "rhs": "Wr", "dst": "acc"}},
            {
                "opcode": "COMMIT",
                "operands": {"src": "acc", "dst": "Y", "bias": "B"},
                "attributes": {"epilogue": epilogue, "acc_scale": scale, "output_dtype": "i8"},
            },
        ],
        "params": {"acc_scale": scale},
        "resources": {"buffers": [], "handles": ["Wr", "acc"]},
        "metrics_requested": [],
    }


def test_relu_runs_on_the_accumulator_before_the_scale():
    cb = _buffer(["bias", "relu", "acc_scale"], 0.37)
    reference = reference_outputs(cb)
    assert reference == simulate(cb)["outputs"]
    from merlin.runtime.commandbuffer import materialize_inputs

    env = materialize_inputs(cb)
    a = np.asarray(env["A"].data).reshape(4, 8)
    w = np.asarray(env["W"].data).reshape(8, 4)
    b = np.asarray(env["B"].data)
    acc = np.maximum(a @ w + b, 0)
    expected = np.clip(np.rint(acc.astype(np.float32) * np.float32(0.37)), -128, 127)
    assert np.array_equal(np.asarray(reference["Y"]), expected)
    # For a positive scale the commuted order is the same function.
    assert reference_outputs(_buffer(["bias", "acc_scale", "relu"], 0.37)) == reference
