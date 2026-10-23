"""A command that reads what an earlier command of the same buffer wrote needs a FENCE between them.

WHY THIS EXISTS. The ABI had no FENCE command, and a contract that states ``hazard_resolution:
explicit`` with ``must_fence_dram_read_after_write`` had nothing to hold a buffer to: the reservation
station tracks no DRAM address, so a chained layer's loads can overtake the previous layer's stores, and
no engine in Merlin models that. ``runtime.fences`` names every such pair; capsule lowering refuses a
buffer with one when the target's contract obliges the fence.
"""

from __future__ import annotations

from merlin.runtime import fences
from merlin.runtime.reference import reference_outputs
from merlin.runtime.simulator import simulate


def _buffer(*commands):
    return {
        "abi_version": "0.1",
        "target": "t",
        "backend": "simulator",
        "tensors": {
            "X": {"shape": [2, 2], "dtype": "i8", "role": "input"},
            "Y": {"shape": [2, 2], "dtype": "i8", "role": "output"},
            "Z": {"shape": [2, 2], "dtype": "i8", "role": "output"},
        },
        "commands": list(commands),
        "resources": {"buffers": [], "handles": []},
        "metrics_requested": [],
    }


_COPY_XY = {"opcode": "MOVEMENT", "operands": {"src": "X", "dst": "Y"}, "attributes": {"output_dtype": "i8"}}
_COPY_YZ = {"opcode": "MOVEMENT", "operands": {"src": "Y", "dst": "Z"}, "attributes": {"output_dtype": "i8"}}
_FENCE = {"opcode": "FENCE", "operands": {}}


def test_a_chained_read_without_a_fence_is_named():
    (violation,) = fences.fence_violations(_buffer(_COPY_XY, _COPY_YZ))
    assert violation == {
        "reader": "MOVEMENT",
        "reader_index": 1,
        "writer": "MOVEMENT",
        "writer_index": 0,
        "tensor": "Y",
    }


def test_a_fence_orders_the_pair_and_changes_no_value():
    fenced = _buffer(_COPY_XY, _FENCE, _COPY_YZ)
    assert fences.fence_violations(fenced) == []
    assert reference_outputs(fenced) == reference_outputs(_buffer(_COPY_XY, _COPY_YZ))
    assert simulate(fenced)["outputs"] == reference_outputs(fenced)


def test_leaf_inputs_and_device_handles_need_no_fence():
    reads_inputs = _buffer(_COPY_XY, {**_COPY_XY, "operands": {"src": "X", "dst": "Z"}})
    assert fences.fence_violations(reads_inputs) == []


def test_only_an_explicit_contract_with_the_obligation_requires_fences():
    explicit = {"memory_model": {"hazard_resolution": "explicit"}, "compiler_obligations": [fences.OBLIGATION]}
    assert fences.requires_fences(explicit)
    assert not fences.requires_fences({"memory_model": {"hazard_resolution": "interlocked"}})
    assert not fences.requires_fences({**explicit, "compiler_obligations": []})
    assert not fences.requires_fences(None)
