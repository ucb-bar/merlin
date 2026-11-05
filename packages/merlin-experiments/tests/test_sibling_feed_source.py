"""A sibling-feed case: two sibling readouts of one input merge and feed a third contraction."""

from __future__ import annotations

from copy import deepcopy

import pytest
from merlin_experiments.phase0.writer import _integer_reference_bound

from merlin.runtime.reference import reference_outputs
from merlin.runtime.simulator import simulate
from merlin.targetgen import capsule_golden as CG
from merlin.targetgen import corpus_spec as CS
from merlin.targetgen.capsule_inputs import materialize_capsule_leaves
from merlin.targetgen.contract.interface_emit import parse_interface_mlir
from merlin.targetgen.contract.schemas import validate_capsule
from merlin.targetgen.interface_observations import command_rows

SEMANTICS = {"internal_arithmetic": {"full_operation_overflow_policy": "bounded_exact_requires_each_partial_sum",
                                     "mac_result_bits": 20, "signed_operand_bits": 8}}


def source(**override):
    entry = dict(name="feed", kind="model_slice", source_role="model_derived", source_reference="public test",
                 label="public", op="sibling_feed", M=5, K=24, H=20, N=12, gate_acc_scale=0.125,
                 up_acc_scale=0.0625, lhs_scale=1.0, rhs_scale=1.0, bound_lsb=0, stimulus_range=[-4, 3])
    entry.update(override)
    binding = CS.CorpusBinding("fixture", 16, "int8", "i32", True, ["L0", "L1", "L2"], "exact_int",
                               readout_inputs={}, requant_output_dtype="i8")
    return entry, binding


def test_golden_is_the_composition_of_the_ordinary_readout_add_and_contraction():
    entry, binding = source()
    cap, text = CS.build(entry, binding)
    validate_capsule(cap)
    env = materialize_capsule_leaves(cap)
    gate = env["X"].matmul(env["WG"]).requant_acc_scale(0.125).to_i8()
    up = env["X"].matmul(env["WU"]).requant_acc_scale(0.0625).to_i8()
    merged = gate.residual_add_scaled(up, 1.0, 1.0).to_i8()
    expected = merged.matmul(env["WD"]).to_list()
    assert CG.golden(cap) == {"Y0": expected}
    cb = parse_interface_mlir(text)
    inputs = {name: env[name].to_list() for name in ("X", "WG", "WU", "WD")}
    assert reference_outputs(cb, inputs)["Y0"] == expected
    assert simulate(cb, inputs)["outputs"]["Y0"] == expected


def test_both_siblings_read_the_same_input_and_the_merge_reads_committed_readouts():
    entry, binding = source()
    _, text = CS.build(entry, binding)
    cb = parse_interface_mlir(text)
    matmuls = [c for c in cb["commands"] if c["opcode"] == "MATMUL_RESIDENT"]
    assert [c["operands"]["lhs"] for c in matmuls] == ["X", "X", "Hm"]
    merge = next(row for row in command_rows(cb) if row["operation"] == "residual_add")
    # Both operands are committed tensors; each is observed with its readout dtype.
    assert merge["operand_dtypes"] == ["i8"] and len(merge["ordered_operand_types"]) == 2


def test_the_merge_must_be_exact_and_the_bound_covers_all_three_contractions():
    entry, binding = source(bound_lsb=1)
    with pytest.raises(ValueError, match="must be exact"):
        CS.build(entry, binding)
    entry, binding = source()
    cap, _ = CS.build(entry, binding)
    bound = _integer_reference_bound({**entry, "numerical_semantics": SEMANTICS}, cap)
    assert bound["status"] == "proven_safe"
    assert [row["region"] for row in bound["members"]] == ["gate", "up", "down"]


def test_readouts_need_their_declared_scales():
    entry, binding = source()
    del entry["up_acc_scale"]
    with pytest.raises(ValueError, match="up_acc_scale"):
        CS.build(deepcopy(entry), binding)
