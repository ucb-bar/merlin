"""When operands cannot be derived, the error must name the cause the agent can act on.

Measured on a live A/B round: an unaided baseline arm scored 0/35, every capsule failing with

    cyclotron invocation failed: could not derive harness operands: canonical_inputs ARE present, but
    this harness has no operand rule for the command shape ['COMMIT', 'EVICT', 'MATMUL_RESIDENT',
    'RES_PACK'] -- a TOOLING gap, not a defect in the submitted artifact

Every claim in that sentence pointed away from the defect. The command shape is the one thing IDENTICAL
between a submission that grades and one that does not: on the same capsule, the promoted merlin package
emits exactly those four opcodes -- the vocabulary this target's own `interface_grammar.md` documents --
and grades. The difference is that it declares its operands in `tensors` while the baseline leaves the
block empty, and `tensors` is OPTIONAL in command_buffer.schema.json. So a schema-valid, correctly-spelled
command buffer was told its artifact was fine and the harness was broken. The agent's own round notes read
"every semantically reasonable opcode set is rejected before compilation" -- it spent the round permuting
opcodes, which is where the message sent it.
"""
from __future__ import annotations

import pytest

from merlin.runtime.backends import base as _bk

# The SIMT backend is an OUT-OF-TREE package resolved through the registry (it is not importable as
# `merlin.targets.*`), so load it the way capsule_runner does.
try:
    muon = _bk.get_backend("muon")
except Exception as exc:  # noqa: BLE001 - backend absent in this env
    muon = None
    _why = f"SIMT backend not present in this env: {type(exc).__name__}: {exc}"
pytestmark = pytest.mark.skipif(muon is None, reason="SIMT backend not present in this env")


def _cb(*, tensors: dict | None, canonical: bool = True) -> dict:
    """A GEMM command buffer in the documented resident-matmul vocabulary."""
    cb = {
        "abi_version": "1", "target": "radiance",
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "W_res"}},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A0", "rhs": "W_res", "dst": "acc0"}},
            {"opcode": "COMMIT", "operands": {"src": "acc0", "dst": "Y0"}},
            {"opcode": "EVICT", "operands": {"src": "W_res"}},
        ],
    }
    if tensors is not None:
        cb["tensors"] = tensors
    if canonical:
        cb["canonical_inputs"] = {"W": [[1.0]], "A0": [[1.0]]}
    return cb


def _fail(cb) -> str:
    """Drive the operand-derivation failure and return its message."""
    with pytest.raises(Exception) as ei:
        muon.compile_mlir_forkfree("llvm.func @k() {\n  llvm.return\n}\n", cb, "/tmp", target="radiance")
    return str(ei.value)


def test_an_undeclared_tensors_block_is_named_as_the_cause():
    msg = _fail(_cb(tensors={}))
    assert "declares no `tensors`" in msg, msg
    # and it must NOT send the reader back to the opcodes, which are supported and identical to a
    # submission that grades.
    assert "no operand rule for the command shape" not in msg, msg
    assert "not a defect in the submitted artifact" not in msg, msg


def test_the_message_says_what_to_write():
    """A failure an agent cannot act on costs a whole round; name the fields, not just the fault."""
    msg = _fail(_cb(tensors={}))
    for token in ("shape", "dtype", "role"):
        assert token in msg, f"{token!r} missing from: {msg}"


def test_the_operands_that_have_no_shape_are_listed():
    msg = _fail(_cb(tensors={}))
    for name in ("A0", "W", "Y0"):
        assert name in msg, f"{name!r} missing from: {msg}"


def test_a_missing_stimulus_keeps_its_own_distinct_message():
    """The three causes must stay distinguishable -- collapsing them is what produced the wrong blame."""
    msg = _fail(_cb(tensors={}, canonical=False))
    assert "no canonical_inputs" in msg, msg
    assert "declares no `tensors`" not in msg, msg


def test_a_genuine_tooling_gap_still_reads_as_one():
    """With operands declared and an unsupported shape, the harness must still own the gap."""
    cb = _cb(tensors={"X": {"shape": [4, 4], "dtype": "f32", "role": "input"}})
    cb["commands"] = [{"opcode": "SOMETHING_UNMODELLED", "operands": {"src": "X", "dst": "Y"}}]
    msg = _fail(cb)
    assert "not a defect in the submitted artifact" in msg, msg
