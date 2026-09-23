"""``schedule.group_dispatch`` has a consumer that can contradict it.

The op was constructed on the production lowering path and read nowhere, which makes a recorded
decision indistinguishable from one that silently stopped being recorded. Dispatch-region formation
is its natural consumer: the decision names contraction results that belong in ONE region, and the
formation either puts them in one or does not.

The mutation below is the whole point -- the SAME module, the SAME recorded decision, and a
formation that honours it or contradicts it depending on whether dispatches are formed from compute
groups. A read-back that returned ``True`` either way would be the write-only receipt again, wearing
a test.
"""

from __future__ import annotations

from merlin.common import mlir_query as mq
from merlin.xdsl_dialects.lowering import schedule_decisions as SD
from merlin.xdsl_dialects.lowering.contract_facts import lower_to_contract
from merlin.xdsl_dialects.lowering.passes import run_dialect_plane

_TARGET = "toy_npu"  # in-tree, no hardware; the plane takes its target as a parameter

#: Two contractions over one reusable right-hand side: the shape `lower_to_schedule` records a
#: `group_dispatch` for. Nothing about the target is in the module; the facts are inferred from it.
_REUSED_WEIGHT = """builtin.module {
  func.func @forward(%a: tensor<64x128xi8>, %b: tensor<64x128xi8>,
                     %w: tensor<128x64xi8>) -> (tensor<64x64xi32>, tensor<64x64xi32>) {
    %z = arith.constant 0 : i32
    %e0 = tensor.empty() : tensor<64x64xi32>
    %m0 = linalg.quantized_matmul ins(%a, %w, %z, %z : tensor<64x128xi8>, tensor<128x64xi8>, i32, i32)
          outs(%e0 : tensor<64x64xi32>) -> tensor<64x64xi32>
    %e1 = tensor.empty() : tensor<64x64xi32>
    %m1 = linalg.quantized_matmul ins(%b, %w, %z, %z : tensor<64x128xi8>, tensor<128x64xi8>, i32, i32)
          outs(%e1 : tensor<64x64xi32>) -> tensor<64x64xi32>
    func.return %m0, %m1 : tensor<64x64xi32>, tensor<64x64xi32>
  }
}"""

#: No weight is reused, so nothing is proven reusable and no decision is recorded.
_NO_REUSE = """builtin.module {
  func.func @forward(%a: tensor<64x128xi8>, %w: tensor<128x64xi8>) -> tensor<64x64xi32> {
    %z = arith.constant 0 : i32
    %e0 = tensor.empty() : tensor<64x64xi32>
    %m0 = linalg.quantized_matmul ins(%a, %w, %z, %z : tensor<64x128xi8>, tensor<128x64xi8>, i32, i32)
          outs(%e0 : tensor<64x64xi32>) -> tensor<64x64xi32>
    func.return %m0 : tensor<64x64xi32>
  }
}"""


def _grouping(source: str, target: str | None) -> dict:
    from merlin.xdsl_dialects.lowering.pipeline import load_curated_contract

    contract = load_curated_contract(_TARGET)
    return run_dialect_plane(mq.parse(source), target=target, analysis_contract=contract).stats["abstraction_analysis"][
        "dispatch_grouping"
    ]


def test_the_decision_is_read_back_off_the_module_that_recorded_it() -> None:
    from merlin.xdsl_dialects.lowering.pipeline import load_curated_contract

    scheduled = SD.lower_to_schedule(lower_to_contract(mq.parse(_REUSED_WEIGHT), load_curated_contract(_TARGET)))
    recorded = SD.recorded_group_dispatch(scheduled)
    assert len(recorded) == 1
    assert recorded[0].n_values == 2 and recorded[0].granularity == "command_buffer"


def test_forming_dispatches_from_compute_groups_honours_the_decision() -> None:
    grouping = _grouping(_REUSED_WEIGHT, _TARGET)
    assert grouping["recorded"] == 1 and grouping["values_grouped"] == 2
    assert grouping["contraction_dispatches_in_a_region"] == 2
    assert grouping["honoured"] is True


def test_the_single_op_baseline_contradicts_it_and_the_read_back_says_so() -> None:
    """The mutation. One dispatch per operation is the ABSENCE of the grouping, not a grouping."""
    grouping = _grouping(_REUSED_WEIGHT, None)
    assert grouping["recorded"] == 1 and grouping["values_grouped"] == 2
    assert grouping["contraction_dispatches_in_a_region"] == 0
    assert grouping["honoured"] is False


def test_nothing_recorded_is_reported_as_such_and_never_as_honoured() -> None:
    """A decision nothing records and a decision nothing contradicts are different facts."""
    for target in (None, _TARGET):
        grouping = _grouping(_NO_REUSE, target)
        assert grouping["recorded"] == 0 and grouping["honoured"] is None, target


def test_forming_dispatches_from_groups_changes_the_program_that_results() -> None:
    """The `groups=` argument outline_dispatches has always accepted, now passed on this path too."""
    baseline = run_dialect_plane(mq.parse(_REUSED_WEIGHT), target=None)
    grouped = run_dialect_plane(mq.parse(_REUSED_WEIGHT), target=_TARGET)
    assert baseline.stats["kernels"] == grouped.stats["kernels"] == 2
    assert [d.group for d in baseline.dispatches] == [None, None]
    assert [d.group for d in grouped.dispatches] == [0, 1]
    assert all(d.placement for d in grouped.dispatches)
