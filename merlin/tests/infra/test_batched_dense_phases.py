"""Actual counted dense programs, private byte windows and conservative refusals."""
from dataclasses import replace

import pytest
from test_compact_dense import ADAPTER, mutate
from test_dense_retained_reduction import chosen, inputs

from merlin.compile.scheduling.batched_dense import (
    partition_batched_dense_storage,
    partition_dense_phases,
    verify_dense_phase_partition,
)
from merlin.compile.scheduling.block_schedule import Compute, Load, Preload, Store
from merlin.compile.scheduling.compact_dense import DenseLoopStorage, DenseMemoryView
from merlin.compile.scheduling.dense_panel_pipeline import DRAIN
from merlin.sched.ir import Call, evaluate, instances


def classification(call):
    return {Load: "load", Preload: "compute", Compute: "compute", Store: "store"}.get(type(ADAPTER.project(call)))


def options():
    return dict(phases=("load", "compute", "store"), classify=classification, completion_instruction=DRAIN)


def commands(kernel):
    return [(c.instr, tuple((key, evaluate(value, env)) for key, value in c.args)) for c, env in instances(kernel)]


@pytest.fixture
def program():
    p, r, s, b, a = inputs(shape=(8, 14, 16))
    return chosen(p, r, s, b, a).kernel


def test_real_counted_phase_partition_preserves_every_original_operand_and_product(program):
    phases, record = partition_dense_phases(program, **options())
    for phase in phases:
        expected = [(c.instr, tuple((k, evaluate(v, env)) for k, v in c.args))
                    for c, env in instances(program) if c.instr != DRAIN and classification(c) == phase]
        assert commands(phases[phase]) == expected
        assert record["original_dynamic_calls"][phase] == len(expected)
    assert record["original_dynamic_completions"] == sum(c.instr == DRAIN for c, _ in instances(program))
    assert record["execution_authority"] == "not established"
    assert verify_dense_phase_partition(program, phases, record, **options()) == record


def test_phase_revalidation_refuses_changed_source_and_changed_partition(program):
    phases, record = partition_dense_phases(program, **options())
    changed = mutate(program, "compute", "k", 0)
    with pytest.raises(ValueError, match="changed"):
        verify_dense_phase_partition(changed, phases, record, **options())
    altered = dict(phases)
    altered["load"] = replace(phases["load"], body=())
    with pytest.raises(ValueError, match="changed"):
        verify_dense_phase_partition(program, altered, record, **options())


def test_phase_partition_refuses_unclassified_effect_event_and_proof_overflow(program):
    with pytest.raises(ValueError, match="classification"):
        partition_dense_phases(program, **dict(options(), classify=lambda _: None))
    event = replace(program, body=(*program.body, Call(DRAIN, (), produces="observed")))
    with pytest.raises(ValueError, match="payload|event"):
        partition_dense_phases(event, **options())
    with pytest.raises(ValueError, match="budget"):
        partition_dense_phases(program, **dict(options(), max_static_nodes=1))


def storage():
    return DenseLoopStorage(DenseMemoryView("a", "sp", 2, 0, 48),
                            DenseMemoryView("b", "sp", 3, 32, 32),
                            DenseMemoryView("c", "acc", 8, 0, 32))


def test_batch_private_suballocations_preserve_selected_units_and_actual_parent_bounds():
    original = storage()
    groups = partition_batched_dense_storage(original, groups=2, row_counts=(24, 16, 16))
    assert [(s.lhs.base, s.rhs.base, s.accumulator.base) for s in groups] == [(0, 32, 0), (24, 48, 16)]
    for s in groups:
        for parent, child in zip((original.lhs, original.rhs, original.accumulator),
                                 (s.lhs, s.rhs, s.accumulator), strict=True):
            assert parent.base <= child.base < child.base + child.capacity <= parent.base + parent.capacity
            assert (child.backing, child.unit_bytes) == (parent.backing, parent.unit_bytes)


def test_batch_storage_refuses_capacity_and_byte_alias_even_with_different_row_widths():
    original = storage()
    with pytest.raises(ValueError, match="exceed"):
        partition_batched_dense_storage(original, groups=3, row_counts=(24, 16, 16))
    overlapping = replace(original, rhs=replace(original.rhs, base=31))
    with pytest.raises(ValueError, match="overlap"):
        partition_batched_dense_storage(overlapping, groups=2, row_counts=(24, 16, 16))
