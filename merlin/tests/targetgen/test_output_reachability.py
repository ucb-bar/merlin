"""A declared output nothing writes must be caught BEFORE a simulator runs.

Measured motivation: 13 capsules emitted single-command buffers with no store, validated cleanly, ran on
the RTL oracle, and were reported as numeric mismatches whose counts could not respond to the kernel.
"""

from __future__ import annotations

from merlin.targetgen.output_reachability import (declared_outputs, output_reachability_findings,
                                                 unwritten_outputs)


def _cb(commands, tensors=None):
    return {"abi_version": "0.1", "target": "t",
            "tensors": tensors or {"A": {"role": "input"}, "B": {"role": "input"},
                                   "Y0": {"role": "output"}},
            "commands": commands}


def test_the_measured_af6_buffer_is_caught():
    """The exact shape that cost seven rounds: one compute command, no store."""
    cb = _cb([{"opcode": "VECTOR_MAP", "operands": {"lhs": "A", "rhs": "B", "dst": "Y0"},
               "attributes": {"op": "add"}}])
    # AF6 DID name dst=Y0 -- so reachability alone does not flag it; that is honest and important.
    assert output_reachability_findings(cb) == []


def test_a_buffer_that_names_no_destination_is_flagged():
    cb = _cb([{"opcode": "VREDUCE", "operands": {"src": "X"}, "attributes": {"op": "sum"}}],
             tensors={"X": {"role": "input"}, "Y0": {"role": "output"}})
    f = output_reachability_findings(cb)
    assert len(f) == 1 and "Y0" in f[0]
    assert "untouched fill" in f[0]


def test_a_resident_matmul_chain_passes():
    cb = _cb([{"opcode": "RES_PACK", "operands": {"src": "B", "dst": "B_res"}},
              {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A", "rhs": "B_res", "dst": "acc0"}},
              {"opcode": "COMMIT", "operands": {"src": "acc0", "dst": "Y0"}},
              {"opcode": "EVICT", "operands": {"handle": "B_res"}}])
    assert output_reachability_findings(cb) == []


def test_it_never_asks_for_a_COMMIT_opcode():
    """Two capsules with NO COMMIT passed in the measured run. An opcode check would false-positive
    them; a reachability check must not, as long as some command names the output."""
    cb = _cb([{"opcode": "ATTENTION_QK", "operands": {"q": "A", "k": "B", "out": "Y0"}}])
    assert output_reachability_findings(cb) == []
    src = open("merlin/python/merlin/targetgen/output_reachability.py").read()
    assert '"COMMIT"' not in src, "an opcode literal would make this an assumed-ISA check"


def test_unknown_destination_key_is_indeterminate_not_a_failure():
    """Writing through a key this module does not classify must be reported as indeterminate -- fail
    closed on the CHECK, never on the submission."""
    cb = _cb([{"opcode": "ODD", "operands": {"writeback_target": "Y0"}}])
    f = output_reachability_findings(cb)
    assert len(f) == 1
    assert "indeterminate" in f[0] and "not proven missing" in f[0]


def test_no_declared_outputs_is_not_this_checks_business():
    assert output_reachability_findings(_cb([{"opcode": "X", "operands": {}}],
                                            tensors={"A": {"role": "input"}})) == []


def test_helpers():
    cb = _cb([{"opcode": "C", "operands": {"dst": "Y0"}}])
    assert declared_outputs(cb) == ["Y0"] and unwritten_outputs(cb) == []
    cb2 = _cb([{"opcode": "C", "operands": {"src": "A"}}])
    assert unwritten_outputs(cb2) == ["Y0"]


def test_malformed_input_does_not_raise():
    for bad in ({}, {"tensors": None}, {"tensors": {"Y0": {"role": "output"}}, "commands": None},
                {"tensors": {"Y0": {"role": "output"}}, "commands": [None, 3, "x"]}):
        output_reachability_findings(bad)          # must not raise
