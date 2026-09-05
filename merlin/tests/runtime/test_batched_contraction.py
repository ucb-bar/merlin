"""A rank-3 contraction must be materializable, evaluable and countable.

The target contract declared ``ranks: [2, 4]``, so every batched matmul was ruled ineligible before it
reached the device rewrite -- while the rewrite already supported the (B,M,N,K) signature, the device
builder already produced the (M,N,K) kernel, and the generated shim already looped the batch calling
that kernel once per independent slice. The machinery existed and the contract forbade its use.

Fixing the contract line is not enough on its own, because three things downstream still assumed rank
two and each one fails in a way that does not look like "rank three is unsupported":

* the leaf materializer peeled exactly ONE level of nesting, so a rank-3 operand arrived as a list of
  lists and ``Tensor`` rejected it for having the wrong element COUNT -- a malformed-input error;
* the integer reference engine had no definition for the opcode, so it raised ``UnmodeledOp``, which
  reads as "grade this on hardware instead" and silently removes the tier that compares two
  independent evaluations of the same buffer;
* the work counter had no rule for the opcode, which marks the whole program a lower bound and drops
  it from the achievable-ceiling harvest.

Every operand in the perf corpus is rank-2, so none of this was reachable from a shipped capsule and a
revert of the contract line would be invisible. That is what these tests are for.
"""
from __future__ import annotations

import pytest

from merlin.perf import work_volume as WV
from merlin.runtime.commandbuffer import materialize_inputs
from merlin.runtime.reference import MODELED_OPCODES, reference_outputs
from merlin.runtime.simulator import simulate

#: Two independent slices, chosen so the two batches do NOT compute the same thing -- a batched kernel
#: that ignored the batch index and ran slice 0 twice would still match a fixture whose slices agree.
A0 = [[[1, 2, 3], [4, 5, 6]], [[1, 0, 1], [0, 1, 0]]]
W = [[[1, 0], [0, 1], [1, 1]], [[2, 0], [0, 2], [1, 1]]]
EXPECTED = [[[4, 5], [10, 11]], [[3, 1], [0, 2]]]


def _command_buffer():
    return {
        "abi_version": "0.1", "target": "gemmini", "version": "0.1", "params": {},
        "tensors": {
            "A0": {"role": "input", "shape": [2, 2, 3], "dtype": "i8"},
            "W": {"role": "weight", "shape": [2, 3, 2], "dtype": "i8"},
            "Y0": {"role": "output", "shape": [2, 2, 2], "dtype": "i32"},
        },
        "commands": [{"opcode": "BATCHED_MATMUL",
                      "operands": {"a": "A0", "w": "W", "dst": "Y0"},
                      "attributes": {"batch": 2, "output_dtype": "i32"}}],
        "outputs": ["Y0"],
    }


def test_a_rank_three_operand_materializes_with_every_element():
    """The failure this prevents reported a COUNT, not a rank: 'shape (2, 2, 3) needs 12 elements,
    got 4'. Nothing in that message says the rank is the problem."""
    env = materialize_inputs(_command_buffer(), {"A0": A0, "W": W})
    assert env["A0"].shape == (2, 2, 3)
    assert len(env["A0"].data) == 12
    assert env["A0"].data == [1, 2, 3, 4, 5, 6, 1, 0, 1, 0, 1, 0]


@pytest.mark.parametrize("nested,flat", [
    ([1, 2, 3], [1, 2, 3]),
    ([[1, 2], [3, 4]], [1, 2, 3, 4]),
    ([[[1], [2]], [[3], [4]]], [1, 2, 3, 4]),
])
def test_flattening_is_rank_agnostic(nested, flat):
    """Ranks one and two must keep flattening exactly as before; rank three must now work too."""
    from merlin.runtime.commandbuffer import _flatten
    assert _flatten(nested) == flat


def test_the_reference_engine_models_the_batched_opcode():
    assert "BATCHED_MATMUL" in MODELED_OPCODES


def test_reference_and_simulator_independently_agree_with_the_hand_computed_result():
    """Two independent evaluations plus a third computed by hand. Agreement between the reference and
    the simulator alone would not catch a mistake made the same way in both."""
    cb = _command_buffer()
    reference = reference_outputs(cb, {"A0": A0, "W": W})
    simulated = simulate(cb, {"A0": A0, "W": W})
    produced = simulated["outputs"]["Y0"] if "outputs" in simulated else simulated["Y0"]
    assert reference["Y0"] == EXPECTED
    assert produced == EXPECTED


def test_the_batch_index_is_honoured_rather_than_slice_zero_repeated():
    """The two slices differ, so a kernel that ran slice 0 twice would produce [[4,5],[10,11]] twice."""
    out = reference_outputs(_command_buffer(), {"A0": A0, "W": W})["Y0"]
    assert out[0] != out[1]


def test_batched_work_is_counted_and_not_left_a_lower_bound():
    """An opcode with no work-counting rule makes the WHOLE program a lower bound, which drops it from
    the achievable-ceiling harvest -- so the member contributes nothing and says nothing."""
    work = WV.work_from_command_buffer(_command_buffer())
    assert not work.is_lower_bound, work.refusals
    assert work.exact_macs == 2 * (2 * 3 * 2), "batch x (M x K x N)"


def test_operands_over_different_batches_refuse_rather_than_guess():
    cb = _command_buffer()
    cb["tensors"]["W"]["shape"] = [3, 3, 2]          # three batches against the activation's two
    work = WV.work_from_command_buffer(cb)
    assert work.is_lower_bound
    assert any("batched-matmul" in reason for reason in work.refusals), work.refusals


def test_a_rank_two_operand_pair_is_not_accepted_as_batched():
    cb = _command_buffer()
    cb["tensors"]["A0"]["shape"] = [2, 3]
    cb["tensors"]["W"]["shape"] = [3, 2]
    work = WV.work_from_command_buffer(cb)
    assert work.is_lower_bound, "a rank-2 pair under a batched opcode is not a batch of one"
