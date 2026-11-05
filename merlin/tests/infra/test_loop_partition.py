"""Ordered counted intervals transport actual operands and proof custody."""

from dataclasses import replace

import pytest
from test_compact_dense import ADAPTER, fixture, schedule
from test_loop_schedule import SOURCE, base, trace
from test_scf_commands import record

from merlin.compile.scheduling.loop_schedule import LoopTransform, partition_counted_loop
from merlin.compile.scheduling.scf_commands import materialize_scf_commands
from merlin.compile.scheduling.schedule_ir import (
    dense_schedule_text,
    make_dense_schedule_plan,
    parse_dense_schedule_plan,
)
from merlin.sched.ir import Kernel, Var, call, loop, select
from merlin.sched.primitives import NotApplicable


def test_ragged_regions_keep_exact_iteration_order_pointer_predicates_and_configuration():
    original = base(17)
    result = partition_counted_loop(original, ("i",), cuts=(4, 10))
    assert result.verify_current(original, result.kernel)["exact_instance_order"]
    assert [(r.start, r.stop) for r in result.regions] == [(0, 4), (4, 10), (10, 17)]
    assert trace(result.kernel) == trace(original)
    assert result.kernel.digest() != original.digest()
    assert result.request.names == tuple(r.path[-1] for r in result.regions)
    assert result.request.cuts == (4, 10)
    # Actual original final-iteration predicate is shifted, not replaced by a
    # guessed last-member extent or an unproved branch simplification.
    assert "select(i_part_2 == 6" in result.kernel.text()


def test_actual_partition_scf_emitter_executes_every_original_index_once():
    original = Kernel("regions", (), (loop("i", 11,
        call("consume", offset=Var("i") * 4, last=select("i", 10, 1, 0))),))
    result = partition_counted_loop(original, ("i",), cuts=(2, 7))
    before, after = (record(materialize_scf_commands(k)) for k in (original, result.kernel))
    assert before == after
    assert [args["offset"] for _, args in after] == list(range(0, 44, 4))
    assert sum(args["last"] for _, args in after) == 1


def test_nested_body_and_unrelated_names_keep_exact_original_scope():
    original = Kernel("nested", (), (loop("i", 13, loop("j", 3,
        call("consume", row=Var("i"), col=Var("j")))), loop("i_part_0", 1, call("other"))))
    result = partition_counted_loop(original, ("i",), cuts=(2, 9))
    assert trace(result.kernel) == trace(original)
    assert "i_part_0" not in result.request.names
    result.verify_current(original, result.kernel)


def test_large_domain_is_counted_not_unfolded():
    original = base(10**12)
    result = partition_counted_loop(original, ("i",), cuts=(11, 10**9))
    assert result.kernel.text().count("read(") == 3
    assert len(result.kernel.text()) < 2000
    assert result.regions[-1].stop == 10**12
    result.verify_current(original, result.kernel)


@pytest.mark.parametrize("cuts", [(), (0,), (-1,), (17,), (18,), (5, 5), (10, 4), (True,), [4], tuple(range(1, 17))])
def test_ambiguous_empty_duplicate_outside_or_unbounded_cut_decisions_refuse(cuts):
    with pytest.raises((ValueError, NotApplicable), match="cuts|inside|interior"):
        partition_counted_loop(base(), ("i",), cuts=cuts)


def test_unknown_control_field_names_and_static_bounds_refuse():
    with pytest.raises(ValueError, match="only partition"):
        LoopTransform("reorder", ("i",), cuts=(2,))
    with pytest.raises(ValueError, match="only.*cuts"):
        LoopTransform("partition", ("i",), cuts=(2,), factor=2)
    with pytest.raises(NotApplicable, match="collides"):
        partition_counted_loop(base(), ("i",), cuts=(2,), names=("i", "other"))
    with pytest.raises(NotApplicable, match="budget"):
        partition_counted_loop(base(), ("i",), cuts=(2, 8), max_static_operations=4)


def test_custody_refuses_changed_original_actual_interval_mapping_and_request():
    original = base()
    result = partition_counted_loop(original, ("i",), cuts=(4, 10))
    with pytest.raises(ValueError, match="source identity"):
        result.verify_current(base(18), result.kernel)
    with pytest.raises(ValueError, match="actual ordered"):
        result.verify_current(original, original)
    bad = replace(result, regions=(replace(result.regions[0], stop=5), *result.regions[1:]))
    with pytest.raises(ValueError, match="interval mapping"):
        bad.verify_current(original, result.kernel)
    unnamed = replace(result, request=replace(result.request, names=()), regions=())
    with pytest.raises(ValueError, match="exact recorded"):
        unnamed.verify_current(original, result.kernel)
    bad = replace(result, request=replace(result.request, cuts=(3, 10)))
    with pytest.raises(ValueError, match="actual ordered"):
        bad.verify_current(original, result.kernel)


def test_partition_decision_roundtrips_and_unchanged_dense_verifier_admits_actual_ragged_order():
    p, r, s = fixture(shape=(8, 6, 20))
    r = replace(r, max_m_blocks=1, max_k_blocks=1, max_n_blocks=1)
    def plan(requests):
        return parse_dense_schedule_plan(dense_schedule_text(make_dense_schedule_plan(
            p, r, s, ADAPTER, source_identity=SOURCE, dtypes=("i8", "i8", "i32"), static_budget=80,
            block_tiles=(1, 1, 1), loop_transforms=requests)))
    original = schedule(p, r, s, schedule_plan=plan(()), source_identity=SOURCE)
    result = partition_counted_loop(original.kernel, ("outer_m", "outer_n"), cuts=(2, 3))
    actual = schedule(p, r, s, schedule_plan=plan((result.request,)), source_identity=SOURCE)
    actual.revalidate()
    assert trace(actual.kernel) == trace(original.kernel)
    result.verify_current(original.kernel, actual.kernel)
    assert "partition" in actual.realization["schedule_ir"]["text"]
    assert "cuts = array<i64: 2, 3>" in actual.realization["schedule_ir"]["text"]
    assert record(materialize_scf_commands(actual.kernel)) == record(materialize_scf_commands(original.kernel))
