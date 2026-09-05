"""The integer reference engine models ``ATTENTION_QK``, and agrees with the simulator on it.

Regression for a harness defect, not a submission defect: the command-buffer SIMULATOR implemented
``ATTENTION_QK`` while :data:`merlin.runtime.reference.MODELED_OPCODES` did not list it, so
:func:`reference_outputs` raised ``UnmodeledOp`` on a conformant buffer. A capsule declaring
``op: attention_qk`` therefore had NO expressible command buffer at all — half of the L0 comparison
refused the only opcode the interface grammar defines for that op.

Pure Python (no build, no oracle): synthetic buffers plus an independently computed Q @ Kᵀ.
"""
from __future__ import annotations

import pytest

from merlin.runtime.commandbuffer import materialize_inputs
from merlin.runtime.reference import MODELED_OPCODES, reference_outputs, outputs_match
from merlin.runtime.simulator import simulate


def _cb(m=16, d=32, n=16, *, epilogue=None, output_dtype="i32", **attrs):
    a = {"epilogue": list(epilogue or []), "output_dtype": output_dtype}
    a.update(attrs)
    # the buffer schema requires a target; both engines are target-agnostic on this op, so
    # any name serves — nothing here reads a per-target fact.
    return {"abi_version": "0.1", "target": "toy_npu",
            "tensors": {"Q": {"shape": [m, d], "dtype": "i8", "role": "input"},
                        "K": {"shape": [n, d], "dtype": "i8", "role": "input"},
                        "Y0": {"shape": [m, n], "dtype": output_dtype, "role": "output"}},
            "commands": [{"opcode": "ATTENTION_QK",
                          "operands": {"q": "Q", "k": "K", "dst": "Y0"}, "attributes": a}]}


def test_attention_qk_is_modeled():
    assert "ATTENTION_QK" in MODELED_OPCODES
    assert reference_outputs(_cb())["Y0"]          # no UnmodeledOp, and a non-empty output map


def test_reference_computes_q_at_k_transposed():
    """K is stored ROW-per-key, so the contraction is over the trailing head dim of BOTH operands."""
    cb = _cb(m=5, d=4, n=3)
    env = materialize_inputs(cb, None)
    q = [env["Q"].data[i * 4:(i + 1) * 4] for i in range(5)]
    k = [env["K"].data[j * 4:(j + 1) * 4] for j in range(3)]
    expected = [[sum(q[i][t] * k[j][t] for t in range(4)) for j in range(3)] for i in range(5)]
    assert reference_outputs(cb)["Y0"] == expected


@pytest.mark.parametrize("cb", [
    _cb(),                                          # the shipped C7 geometry, empty epilogue
    _cb(epilogue=["relu"]),
    _cb(epilogue=["acc_scale"], acc_scale=0.5),
    _cb(epilogue=["requant"], requant_shift=3),
    _cb(output_dtype="i8"),                         # scaled/clamped readout
    _cb(m=20, d=12, n=7),                           # extents that do not divide any tile edge
])
def test_reference_and_simulator_agree(cb):
    assert outputs_match(reference_outputs(cb), simulate(cb)["outputs"])


def test_unknown_epilogue_stage_fails_closed():
    """A stage the engine cannot perform RAISES. Skipping it on both sides would make the golden and
    the reference agree on a value neither of them computed."""
    with pytest.raises(ValueError, match="does not implement"):
        reference_outputs(_cb(epilogue=["softcap"]))
