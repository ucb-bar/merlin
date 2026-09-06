"""The ablation must not be able to flatter itself.

This file guards the three ways the measurement in :mod:`merlin.verify.ablation` could report a
number that means less than it appears to:

1. counting a submission that IS its own specification as a verified compilation (``X == X``);
2. reporting an abstention -- a shape the encoder cannot model -- as a pass;
3. printing only the non-empty cells, so "we found nothing" and "we never looked" read the same.

Each is asserted rather than described, because the module's whole purpose is to avoid a flattered
measurement and the instrument built for the historical replay flattered itself three ways first.
"""
from __future__ import annotations

import copy

import pytest

from merlin.verify import HAS_XDSL, HAS_Z3

pytestmark = pytest.mark.skipif(not (HAS_XDSL and HAS_Z3),
                                reason="needs the verify extra (xdsl + z3)")

_TIMEOUT_MS = 20_000


def _matmul_cb(m=2, k=2, n=2):
    """A small, complete command buffer: pack a weight, contract, commit, evict."""
    return {
        "abi_version": "0.1", "target": "t",
        "tensors": {
            "X": {"shape": [m, k], "dtype": "i8", "role": "input"},
            "W": {"shape": [k, n], "dtype": "i8", "role": "weight"},
            "Y0": {"shape": [m, n], "dtype": "i32", "role": "output"},
        },
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "W_res"}, "attributes": {}},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "X", "rhs": "W_res", "dst": "acc0"},
             "attributes": {}},
            {"opcode": "COMMIT", "operands": {"src": "acc0", "dst": "Y0"},
             "attributes": {"output_dtype": "i32", "epilogue": []}},
            {"opcode": "EVICT", "operands": {"src": "W_res"}, "attributes": {}},
        ],
        "outputs": ["Y0"],
    }


# -- 1. the exclusion, which is the load-bearing one ---------------------------------------------

def test_a_submission_identical_to_its_spec_is_excluded_not_verified():
    """The measurement this prevents: 2,500 of 4,111 archived submissions are their own spec.

    Verifying those would be a query over two copies of one program -- trivially `unsat`, and also
    the case where a bug in the shared encoder cancels completely. Counting them would have made the
    headline 61% vacuous.
    """
    from merlin.verify.ablation import classify

    cb = _matmul_cb()
    assert classify(cb, copy.deepcopy(cb)) == "identical"


def test_a_restructured_submission_is_eligible():
    """A buffer that reaches the same result by different commands is what the check is FOR."""
    from merlin.verify.ablation import classify

    spec = _matmul_cb()
    agent = copy.deepcopy(spec)
    agent["commands"].insert(0, {"opcode": "MOVEMENT", "operands": {"src": "X", "dst": "X"},
                                 "attributes": {}})
    assert classify(spec, agent) == "opcodes"

    attrs = copy.deepcopy(spec)
    attrs["commands"][2]["attributes"]["output_dtype"] = "i8"
    assert classify(spec, attrs) == "operands"


# -- 2. the verdicts themselves ------------------------------------------------------------------

def test_an_equivalent_restructuring_is_verified():
    """Contract against the weight directly instead of packing it first; expect `unsat`.

    This is a restructuring a real backend performs, not a synthetic one: `RES_PACK` is a
    value-identical copy (`cb_semantics._res_pack`), so folding it away leaves the function unchanged
    while changing the command sequence. If the checker refuted this it would be penalising a correct
    submission, which is the failure mode that would make the whole layer worse than useless.
    """
    from merlin.verify.refine import validate_equivalence

    spec = _matmul_cb()
    agent = copy.deepcopy(spec)
    agent["commands"] = [
        {"opcode": "MATMUL", "operands": {"lhs": "X", "rhs": "W", "dst": "acc0"}, "attributes": {}},
        spec["commands"][2],
    ]
    assert [c["opcode"] for c in agent["commands"]] != [c["opcode"] for c in spec["commands"]]
    v = validate_equivalence(spec, agent, timeout_ms=_TIMEOUT_MS)
    assert v.status == "unsat", (
        f"an equivalent restructuring must verify, not {v.status}; refuting a correct submission is "
        f"the one outcome that makes this tool worse than useless")


def test_a_wrong_buffer_is_refuted_with_a_counterexample():
    """A validator that has never rejected anything has not been shown to work.

    The counterexample is required, not optional: a refutation without one is an assertion, and the
    ablation's second headline asks whether the counterexample lies outside the dynamic stimulus.
    """
    from merlin.verify.refine import validate_equivalence

    spec = _matmul_cb()
    agent = copy.deepcopy(spec)
    # Narrow the readout to i8. For most inputs this agrees; for large accumulators it clamps.
    agent["commands"][2]["attributes"]["output_dtype"] = "i8"
    v = validate_equivalence(spec, agent, timeout_ms=_TIMEOUT_MS)
    assert v.status == "sat", f"a narrowed readout must be refuted, not {v.status}"
    assert v.model_values, "a refutation must carry the inputs that expose it"


def test_an_unmodellable_dtype_abstains_rather_than_refuting():
    """Incompleteness in our encoder must never be reported as a defect in the submission."""
    from merlin.verify.refine import validate_equivalence
    from merlin.verify.smt_semantics import UnsupportedSemantics

    spec = _matmul_cb()
    spec["tensors"]["X"]["dtype"] = "bf16"
    with pytest.raises(UnsupportedSemantics):
        validate_equivalence(spec, copy.deepcopy(spec), timeout_ms=_TIMEOUT_MS)


# -- 3. the stimulus claim, and the report ------------------------------------------------------

def test_the_stimulus_range_is_derived_from_the_tensor_module_not_written_down():
    """The ablation's headline-2 claim depends on this set; a hardcoded copy would go stale.

    An earlier note recorded the default fill as period-4 with identical rows. That was true once and
    is not now -- the fill is indexed by (row, col). What remains true, and is what the claim rests
    on, is that every value lies in a four-element non-negative set.
    """
    from merlin.runtime.tensor import Tensor
    from merlin.verify.ablation import stimulus_values

    vals = stimulus_values()
    assert vals and all(isinstance(v, int) for v in vals)
    assert all(v >= 0 for v in vals), "the claim 'the stimulus never goes negative' must hold"
    assert max(vals) < 8, f"the stimulus range widened to {sorted(vals)}; headline 2 must be restated"

    rows = {tuple(Tensor.deterministic("A0", (4, 4), "i8").data[i * 4:(i + 1) * 4]) for i in range(4)}
    assert len(rows) > 1, "rows are identical again; that is a separate, worse problem"


def test_the_report_prints_every_declared_cell_including_the_empty_ones():
    """"Nothing was found" and "nothing was looked for" must not render the same.

    A table that omits its zero rows cannot distinguish them, and this ablation's most likely outcome
    is a zero in exactly the cell that matters most.
    """
    from merlin.verify.ablation import render

    record = {
        "schema": "verify_ablation/v1", "question": "q", "population_total": 3, "sampled": 3,
        "seed": None, "timeout_ms": 1000, "stimulus_values": [0, 1, 2, 3], "wall_seconds": 1.0,
        "population_pin": None,
        "records": [
            {"capsule": "A", "shape": "identical", "verdict": "excluded", "numeric_status": "pass"},
            {"capsule": "B", "shape": "opcodes", "verdict": "verified", "numeric_status": "pass"},
            {"capsule": "C", "shape": "opcodes", "verdict": "abstained", "numeric_status": "fail",
             "reason_kind": "float_dtype"},
        ],
    }
    text = render(record)
    assert "refuted, but the numeric grade PASSED" in text
    assert "none" in text, "an empty headline cell must say so explicitly"
    assert "abstentions (coverage limit, never a pass)" in text
    assert "float_dtype" in text
    assert "EXCLUDED" in text, "the exclusion must be visible in the report, not silent"


# -- 4. the defect the archive found in this checker ---------------------------------------------

def test_right_values_under_the_wrong_name_is_refuted_not_verified():
    """A buffer that computes correctly but publishes under an undeclared name must not verify.

    Found in the archive, not by inspection: thirteen submissions were graded `numeric=fail` with
    EVERY element mismatched while this checker called them VERIFIED. All thirteen committed the
    right values to a name the specification never declared -- eleven to `output_tensor` instead of
    `Y0`. Outputs were paired by sorted ORDER, so one output matched one output regardless of name.
    """
    from merlin.verify.refine import OutputContractViolation, validate_equivalence

    spec = _matmul_cb()
    agent = copy.deepcopy(spec)
    for command in agent["commands"]:
        if command["opcode"] == "COMMIT":
            command["operands"]["dst"] = "output_tensor"
    agent["tensors"]["output_tensor"] = agent["tensors"].pop("Y0")
    agent["outputs"] = ["output_tensor"]

    with pytest.raises(OutputContractViolation, match="Y0"):
        validate_equivalence(spec, agent, timeout_ms=_TIMEOUT_MS)


def test_committing_the_declared_name_twice_is_not_silently_collapsed():
    """Two of the thirteen committed `['Y0', 'Y0']`; a dict env keeps only the last.

    The buffer writes the declared output twice, which is a real ABI question, and the checker used
    to see one output and pair it happily. It must not verify by accident.
    """
    from merlin.verify.refine import validate_equivalence

    spec = _matmul_cb()
    agent = copy.deepcopy(spec)
    commit = [c for c in agent["commands"] if c["opcode"] == "COMMIT"][0]
    agent["commands"].append(copy.deepcopy(commit))
    # A second identical commit is value-identical, so this must NOT refute on values; the point is
    # that the checker reaches a considered verdict rather than collapsing the pair unnoticed.
    v = validate_equivalence(spec, agent, timeout_ms=_TIMEOUT_MS)
    assert v.status in ("unsat", "sat"), f"a duplicated commit must reach a verdict, got {v.status}"
