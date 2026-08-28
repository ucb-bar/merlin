"""A validator the submitter is told to run must fail on what the runner will refuse.

JSON Schema types a command-buffer operand slot as a bare string, so a value that names nothing -- a
shape, a type, a dimension list -- validates. And `tensors` is not required. So a command buffer with
twelve MATMUL commands whose `dst` was twelve spellings of a shape, declaring no tensors at all, was
SCHEMA-VALID.

That is what a live submission actually emitted 32 minutes into a run, having been told by its task
instructions to "confirm the emitted command_buffer.json validates against the schema". It validated. It
was then rejected downstream for a constraint the contract never expressed, so it spent its session
guessing spellings instead of writing a compiler. The workspace held ~25 probe files with names like
`bracket_shape_verdict.json`, `tuple_dst_verdict.json`, `uppercase_dims_verdict.json`.

The rule asserted here is deliberately the weakest one that is certainly true: commands that reference
operands need SOMETHING to reference. Names PRODUCED by earlier commands (an accumulator, a committed
intermediate) are legitimately absent from `tensors`, so per-name resolution is NOT asserted -- that would
fail conformant buffers. Measured over every command buffer on disk from prior runs: 309 unaffected, and
only the probing ones flagged.
"""
from __future__ import annotations

import pytest

from merlin.runtime.commandbuffer import validate_command_buffer as structural
from merlin.targetgen.contract import schemas as S


def _cb(**kw):
    base = {"abi_version": "0.1", "target": "t", "commands": []}
    base.update(kw)
    return base


def _matmul(dst="Y0"):
    return {"opcode": "MATMUL", "operands": {"lhs": "A0", "rhs": "W", "dst": dst}}


def test_operands_with_no_declared_tensors_is_a_problem():
    probs = structural(_cb(commands=[_matmul("16x16")]))
    assert any("declares no 'tensors'" in p for p in probs)
    assert any("16x16" in p for p in probs), "name the offending value"


def test_the_message_says_what_an_operand_slot_actually_holds():
    p = " ".join(structural(_cb(commands=[_matmul("tensor<16x16xf32>")])))
    assert "NAME of a tensor" in p
    assert "not a" in p and "shape" in p


def test_a_conformant_buffer_is_untouched():
    cb = _cb(tensors={"A0": {"shape": [2, 2], "dtype": "f32"}}, commands=[_matmul()])
    assert not [p for p in structural(cb) if "declares no" in p]


def test_a_produced_intermediate_need_not_be_a_declared_tensor():
    """The reason per-name resolution is NOT asserted: an accumulator is produced, not declared."""
    cb = _cb(tensors={"A0": {"shape": [2, 2], "dtype": "f32"}},
             commands=[{"opcode": "MATMUL", "operands": {"lhs": "A0", "dst": "acc0"}},
                       {"opcode": "COMMIT", "operands": {"src": "acc0", "dst": "Y0"}}])
    assert not [p for p in structural(cb) if "declares no" in p]


def test_a_declined_buffer_may_carry_no_tensors():
    """Declining is a legitimate answer and carries no commands to satisfy."""
    cb = _cb(commands=[], declined={"reason": "unsupported shape", "shape": [9], "op": "matmul"})
    assert not [p for p in structural(cb) if "declares no" in p]


def test_the_contract_validator_refuses_what_the_runner_will_refuse():
    """The load-bearing property: schema-valid is no longer enough to be told 'valid'."""
    bad = _cb(commands=[_matmul("[16,16]")])
    with pytest.raises(S.ContractViolation) as e:
        S.validate_command_buffer(bad)
    assert "declares no 'tensors'" in str(e.value)


def test_the_contract_validator_still_accepts_conformant_buffers():
    S.validate_command_buffer(_cb(tensors={"A0": {"shape": [2, 2], "dtype": "f32"}},
                                  commands=[_matmul()]))
    S.validate_command_buffer(_cb(commands=[],
                                  declined={"reason": "r", "shape": [9], "op": "matmul"}))
