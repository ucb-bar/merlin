"""End-to-end: linalg -> contract -> schedule -> interface -> toynpu -> runtime ->
command buffer -> real execution, with simulator == independent reference."""
from __future__ import annotations

import pytest

from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")

REQUIRED_METRICS = {"cycles", "bytes_moved", "command_count", "pack_count",
                    "resident_hits", "evictions", "accumulator_commits"}


@pytest.fixture(scope="module")
def lowered():
    from merlin.xdsl_dialects.lowering import lower_repeated_rhs_matmul

    return lower_repeated_rhs_matmul(reuse=4)


def test_every_stage_verifies(lowered):
    for mod in lowered.modules():
        mod.verify()


def test_stage_dialect_descent(lowered):
    def dialects_of(mod):
        return {op.dialect_name() for op in mod.walk()} - {"builtin", "func", "arith",
                                                           "tensor"}
    assert dialects_of(lowered.input_module) == {"linalg"}
    assert dialects_of(lowered.contract_module) == {"linalg", "contract"}
    assert dialects_of(lowered.schedule_module) == {"linalg", "contract", "schedule"}
    assert dialects_of(lowered.interface_module) == {"interface"}
    assert dialects_of(lowered.target_module) == {"toynpu"}
    assert dialects_of(lowered.runtime_module) == {"runtime"}


def test_command_buffer_is_valid_and_executes_correctly(lowered):
    from merlin.runtime import (outputs_match, reference_outputs, simulate,
                                validate_command_buffer)

    cb = lowered.command_buffer
    assert validate_command_buffer(cb) == []
    assert cb["target"] == "toy_npu"
    assert cb["backend"] == "simulator"
    res = simulate(cb)
    ref = reference_outputs(cb)
    assert outputs_match(res["outputs"], ref)
    assert set(res["outputs"]) == {"Y0", "Y1", "Y2", "Y3"}
    m = res["metrics"]
    assert REQUIRED_METRICS <= {k for k, v in m.items()}
    assert m["pack_count"] == 1
    assert m["resident_hits"] == 4
    assert m["accumulator_commits"] == 4
    assert m["evictions"] == 1
    assert m["cycles"] > 0


def test_execute_helper_reports_correct(lowered):
    from merlin.xdsl_dialects.lowering import execute

    run = execute(lowered)
    assert run["correct"] is True
    names = [e["name"] for e in run["trace"]]
    assert names.count("resident_hit") == 4


def test_lowering_is_deterministic():
    from merlin.xdsl_dialects.lowering import lower_repeated_rhs_matmul

    a = lower_repeated_rhs_matmul(reuse=3)
    b = lower_repeated_rhs_matmul(reuse=3)
    assert a.command_buffer == b.command_buffer
    assert _common.text(a.runtime_module) == _common.text(b.runtime_module)


def test_intermediates_roundtrip(lowered):
    from xdsl.dialects.arith import Arith
    from xdsl.dialects.linalg import Linalg
    from xdsl.dialects.tensor import Tensor as TensorDialect

    from merlin.xdsl_dialects import get_all_dialects
    from merlin.xdsl_dialects.targets import toynpu

    extra = get_all_dialects() + [toynpu.get_dialect(), Linalg, TensorDialect, Arith]
    for mod in lowered.modules():
        m2 = _common.roundtrip(mod, *extra)
        m2.verify()
        assert _common.text(mod) == _common.text(m2)


def test_uses_committed_dialect_plan():
    """The committed toy_npu dialect_plan.yaml drives the same lowering."""
    from merlin.xdsl_dialects.lowering import execute, lower_repeated_rhs_matmul
    from merlin.xdsl_dialects.lowering.target_lowering import load_toy_dialect_plan

    plan = load_toy_dialect_plan()
    res = lower_repeated_rhs_matmul(reuse=2, dialect_plan=plan)
    assert execute(res)["correct"] is True


def test_no_reuse_means_no_residency():
    """Negative control: a single matmul must not select residency."""
    from merlin.xdsl_dialects import schedule as s
    from merlin.xdsl_dialects.lowering.contract_facts import lower_to_contract
    from merlin.xdsl_dialects.lowering.input_workload import build_input_module
    from merlin.xdsl_dialects.lowering.schedule_decisions import lower_to_schedule

    mod = lower_to_schedule(lower_to_contract(build_input_module(reuse=1)))
    assert not [op for op in mod.walk() if isinstance(op, s.SelectInterfaceOp)]


def test_capacity_violation_blocks_interface_lowering():
    """A weight bigger than resident storage must fail the discharged-checks gate."""
    from merlin.xdsl_dialects.lowering import LoweringError, lower_repeated_rhs_matmul

    tiny = {"name": "toy_npu",
            "features": ["resident_packed_tensor", "accumulator_commit",
                         "command_buffer", "metrics"],
            "runtime": {"backends": ["simulator"]},
            "capabilities": {"resident_storage_bytes": 16}}
    with pytest.raises(LoweringError, match="capacity_fit"):
        lower_repeated_rhs_matmul(reuse=4, target_contract=tiny)
