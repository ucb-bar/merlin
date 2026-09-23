"""A quantized residual add rounds once, and a capsule declares how far a target may lie from that."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from merlin.runtime import reference, simulator
from merlin.runtime.commandbuffer import SIGNED_STIMULUS_RANGE, STIMULUS_RANGE_KEY, materialize_inputs
from merlin.targetgen import capsule_golden
from merlin.targetgen import corpus_spec as CS
from merlin.targetgen.contract import interface_emit as IE

_LHS, _RHS = 0.731, 1.377


def _binding():
    return CS.CorpusBinding(
        target="t", tile_dim=16, operand_dtype="int8", accum_dtype="int32", integer=True,
        tiers=["L2", "L3"], compare="exact",
    )  # fmt: skip


def _entry(**over):
    entry = {
        "name": "T_residual", "kind": "layer", "source_role": "derived_sweep", "source_reference": "test",
        "op": "residual_add", "M": 16, "N": 32, "lhs_scale": _LHS, "rhs_scale": _RHS, "bound_lsb": 1,
        "epilogue": ["relu"],
    }  # fmt: skip
    entry.update(over)
    return entry


def _built(**over):
    capsule, text = CS.build_residual_add(_entry(**over), _binding())
    capsule[STIMULUS_RANGE_KEY] = [-128, 127]
    cb = IE.parse_interface_mlir(text)
    cb.setdefault("params", {})[STIMULUS_RANGE_KEY] = [-128, 127]
    return capsule, cb


def test_the_three_engines_are_one_definition() -> None:
    capsule, cb = _built()
    golden = capsule_golden.golden(capsule, None)["Y0"]
    assert golden == reference.reference_outputs(cb)["Y0"] == simulator.simulate(cb)["outputs"]["Y0"]
    values = [v for row in golden for v in row]
    assert min(values) == 0 and 127 in values  # the activation and the saturation both did something


def test_the_declared_bound_is_what_admits_a_unit_that_rounds_each_operand() -> None:
    capsule, cb = _built()
    golden = capsule_golden.golden(capsule, None)["Y0"]
    leaves = materialize_inputs(cb)
    a = np.array(leaves["X0"].data, dtype=np.float32)
    b = np.array(leaves["X1"].data, dtype=np.float32)
    # What a unit with a scaled, rounding load computes: round each operand, then add integers.
    unit = np.maximum(np.clip(np.rint(a * np.float32(_LHS)) + np.rint(b * np.float32(_RHS)), -128, 127), 0)
    unit = unit.reshape(16, 32).astype(int).tolist()
    policy = capsule["numeric_policy"]
    assert policy == {"compare": "bounded_int", "dtype": "i8", "atol": 1, "rtol": 0}
    assert capsule_golden.compare({"Y0": golden}, {"Y0": unit}, policy)["status"] == "pass"
    # The mutation: the same result held to the capture's own arithmetic is refused. The two ARE
    # different functions, and a bound of zero is how a capsule says it wants the first.
    exact = capsule_golden.compare({"Y0": golden}, {"Y0": unit}, {**policy, "atol": 0})
    assert exact["status"] == "fail" and exact["mismatch_count"] > 0 and exact["max_abs_error"] == 1
    # And the bound is a bound: two steps away is refused under it.
    off = copy.deepcopy(golden)
    off[0][0] += 2
    assert capsule_golden.compare({"Y0": golden}, {"Y0": off}, policy)["status"] == "fail"


@pytest.mark.parametrize("missing", ["lhs_scale", "rhs_scale", "bound_lsb"])
def test_nothing_about_the_arithmetic_has_a_default(missing) -> None:
    with pytest.raises(ValueError, match=missing):
        CS.build_residual_add(_entry(**{missing: None}), _binding())
    _capsule, cb = _built()
    command = next(c for c in cb["commands"] if c["opcode"] == "RESIDUAL_ADD")
    del command["attributes"][missing]
    with pytest.raises(ValueError, match=missing):
        reference.reference_outputs(cb)
    with pytest.raises(simulator.SimulationError, match=missing):
        simulator.simulate(cb)


def test_a_stage_the_add_cannot_carry_is_refused() -> None:
    with pytest.raises(ValueError, match="carries only"):
        CS.build_residual_add(_entry(epilogue=["acc_scale"]), _binding())


def test_the_op_is_in_the_grammar_and_owes_no_contraction() -> None:
    from merlin.targetgen import semantic_families

    assert "residual_add" in IE.defined_mnemonics()
    assert semantic_families.from_op("residual_add") == "elementwise_map"
    assert "residual_add" in CS.BUILDERS and SIGNED_STIMULUS_RANGE[0] < 0
