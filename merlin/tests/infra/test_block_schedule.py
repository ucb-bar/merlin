"""The block-schedule pass: knobs over DERIVED geometry, and a residency check that fails closed.

Every geometry here is synthesized as a facts artifact and derived through
``targetgen.address_space``, at two different array edges and bank depths, so nothing in the pass can
be shaped around one device. What the knobs are worth in cycles is measured elsewhere (the capsule
head-to-head); what is pinned here is that each knob does what it says, that a geometry the facts
cannot answer is refused rather than defaulted, and that a schedule which would read a block some
earlier load overwrote never leaves the pass.
"""
from __future__ import annotations

import pytest

from merlin.compile.scheduling import (BANK_ALIGNED, CONTIGUOUS, Compute, Contraction, Geometry,
                                       Knobs, LHS, Load, NEST, OPPOSITE_END, Preload, ROLE, Store,
                                       WEIGHT, BlockScheduleError, check_residency,
                                       schedule_contraction, schedule_interface_program)
from merlin.targetgen.address_space import derive_address_space


def _facts(edge, operand_bytes, operand_depth, accum_bytes, accum_depth, *,
           arrays=None, memories=None, operand_dtype="i8"):
    """A synthetic facts artifact in the shape ``load_facts`` returns."""
    body = {
        "arrays": [{"name": "mesh", "rows": edge, "cols": edge}] if arrays is None else arrays,
        # Deliberately NOT merlin's role names: the pass must resolve roles by row width, and a fixture
        # spelled "scratchpad"/"accumulator" would pass whether or not a name lookup crept back in.
        "datapaths": [{"name": "input", "dtype": operand_dtype, "evidence": "sram_a smem UInt<8>"},
                      {"name": "sum", "dtype": "i32", "evidence": "sram_b smem SInt<32>"}],
    }
    if memories is None:
        # Declared wide-row store FIRST, so a resolver leaning on declaration order picks wrong.
        memories = [{"name": "sram_b", "bytes": accum_bytes, "depth": accum_depth},
                    {"name": "sram_a", "bytes": operand_bytes, "depth": operand_depth}]
    body["memories"] = memories
    return {"schema_version": "2.0", "inputs": {}, "facts": body}


def _geometry(name, **kwargs):
    return Geometry.from_address_space(derive_address_space(name, facts=_facts(**kwargs)))


#: Two devices, deliberately unlike each other: different block edge, store size and bank depth.
NARROW = dict(edge=16, operand_bytes=262144, operand_depth=4096, accum_bytes=65536, accum_depth=512)
WIDE = dict(edge=32, operand_bytes=131072, operand_depth=1024, accum_bytes=65536, accum_depth=256)
GEOMETRIES = {"narrow": NARROW, "wide": WIDE}


def _loads(schedule, role=None):
    return [op for op in schedule.ops if isinstance(op, Load) and (role is None or op.role == role)]


def test_geometry_is_derived_from_each_targets_own_facts():
    narrow = _geometry("t_narrow", **NARROW)
    wide = _geometry("t_wide", **WIDE)
    assert (narrow.block, narrow.operand_rows, narrow.operand_bank_rows) == (16, 16384, 4096)
    assert (wide.block, wide.operand_rows, wide.operand_bank_rows) == (32, 4096, 1024)
    assert (narrow.accumulator_rows, wide.accumulator_rows) == (1024, 512)
    assert narrow.sources["operand_rows"] and wide.sources["block"], "provenance travels with the facts"


@pytest.mark.parametrize("broken, why", [
    (dict(arrays=[]), "no array geometry"),
    (dict(arrays=[{"name": "mesh", "rows": 16, "cols": 8}]), "not square"),
    (dict(memories=[{"name": "sram_b", "bytes": 65536, "depth": 512}]), "no accumulator store"),
    (dict(memories=[]), "no operand store"),
])
def test_a_geometry_the_facts_cannot_answer_is_refused_not_defaulted(broken, why):
    """The failure this repo keeps re-learning: an unmeasurable quantity reported as a plausible
    number. Every branch that cannot derive the geometry has to refuse."""
    with pytest.raises(BlockScheduleError, match=why):
        _geometry("t_broken", **{**NARROW, **broken})


@pytest.mark.parametrize("name", sorted(GEOMETRIES))
def test_loading_on_index_change_moves_each_streamed_block_once(name):
    geometry = _geometry(f"t_{name}", **GEOMETRIES[name])
    d = geometry.block
    contraction = Contraction(m=2 * d, k=2 * d, n=3 * d)      # 2 x 2 x 3 blocks
    once = schedule_contraction(contraction, geometry, Knobs(load_on_index_change=True))
    per_block = schedule_contraction(contraction, geometry, Knobs(load_on_index_change=False))
    assert len(_loads(once, LHS)) == 2 * 2, "one load per (m, k) block"
    assert len(_loads(per_block, LHS)) == 2 * 2 * 3, "without the knob, once per resident block as well"
    assert len(_loads(once, WEIGHT)) == len(_loads(per_block, WEIGHT)) == 2 * 3
    assert once.count(Compute) == per_block.count(Compute), "the knob moves bytes, not arithmetic"


@pytest.mark.parametrize("name", sorted(GEOMETRIES))
def test_lookahead_decides_how_far_ahead_of_its_compute_a_load_is_issued(name):
    geometry = _geometry(f"t_{name}", **GEOMETRIES[name])
    d = geometry.block
    contraction = Contraction(m=d, k=4 * d, n=d)              # four reduction steps
    first_compute = {}
    for depth in (0, 1, 2, None):
        schedule = schedule_contraction(contraction, geometry, Knobs(lookahead_steps=depth))
        ops = list(schedule.ops)
        first_compute[depth] = next(i for i, op in enumerate(ops) if isinstance(op, Compute))
        loads_before = sum(1 for op in ops[:first_compute[depth]] if isinstance(op, Load))
        # depth d issues steps 0..d before the first compute; None issues every step's loads.
        assert loads_before == (2 * (depth + 1) if depth is not None else 8), (depth, loads_before)
    assert first_compute[0] < first_compute[1] < first_compute[2] < first_compute[None]


@pytest.mark.parametrize("order", [(LHS, WEIGHT), (WEIGHT, LHS)])
def test_operand_order_orders_the_loads_inside_a_group(order):
    geometry = _geometry("t_narrow", **NARROW)
    d = geometry.block
    schedule = schedule_contraction(Contraction(m=d, k=2 * d, n=d), geometry,
                                    Knobs(lookahead_steps=1, load_grouping=ROLE, operand_order=order))
    assert [op.role for op in _loads(schedule)][:2] == list(order)


def test_nest_grouping_keeps_each_streamed_load_beside_its_own_compute():
    """The interleaved shape: with no lookahead and nest grouping, a streamed block is loaded
    immediately before the compute that reads it, not batched with its step."""
    geometry = _geometry("t_narrow", **NARROW)
    d = geometry.block
    schedule = schedule_contraction(Contraction(m=2 * d, k=d, n=d), geometry,
                                    Knobs(lookahead_steps=0, load_grouping=NEST))
    kinds = [type(op).__name__ for op in schedule.ops]
    assert kinds == ["Load", "Load", "Preload", "Compute", "Store",
                     "Load", "Preload", "Compute", "Store"], kinds


@pytest.mark.parametrize("name", sorted(GEOMETRIES))
def test_each_placement_policy_puts_the_resident_region_where_it_says(name):
    geometry = _geometry(f"t_{name}", **GEOMETRIES[name])
    d = geometry.block
    contraction = Contraction(m=d, k=2 * d, n=d)
    streamed_rows = 2 * d
    bases = {policy: schedule_contraction(contraction, geometry,
                                          Knobs(placement=policy)).regions[WEIGHT][0]
             for policy in (OPPOSITE_END, BANK_ALIGNED, CONTIGUOUS)}
    assert bases[OPPOSITE_END] == geometry.operand_rows - streamed_rows
    assert bases[BANK_ALIGNED] == geometry.operand_bank_rows, "the first bank boundary past the inputs"
    assert bases[CONTIGUOUS] == streamed_rows, "directly after the inputs, sharing their bank"


def test_bank_aligned_placement_is_refused_when_the_facts_have_no_bank_depth():
    geometry = _geometry("t_narrow", **NARROW)
    without_banks = Geometry(block=geometry.block, operand_rows=geometry.operand_rows,
                             operand_bank_rows=None, accumulator_rows=geometry.accumulator_rows,
                             separate_accumulator_space=True)
    with pytest.raises(BlockScheduleError, match="per-bank row count"):
        schedule_contraction(Contraction(m=16, k=32, n=16), without_banks, Knobs(placement=BANK_ALIGNED))


@pytest.mark.parametrize("name, k_blocks", [("narrow", 513), ("wide", 65)])
def test_hoisting_every_load_over_overlapping_regions_is_refused(name, k_blocks):
    """The fail-closed case, and the reason the knobs are checked rather than trusted.

    A reduction deep enough that the two operand regions overlap is schedulable ONLY while each
    overwrite lands after the overlapping block's last read. One step of lookahead satisfies that;
    hoisting every load does not, and an in-order executor cannot see the difference -- it just
    computes on whatever is resident.
    """
    geometry = _geometry(f"t_{name}", **GEOMETRIES[name])
    d = geometry.block
    contraction = Contraction(m=d, k=k_blocks * d, n=d)
    overlapped = schedule_contraction(contraction, geometry, Knobs(lookahead_steps=1))
    lhs_rows, (weight_base, _) = overlapped.regions[LHS][1], overlapped.regions[WEIGHT]
    assert weight_base < lhs_rows, "the shape is only interesting while the regions overlap"
    assert any("OVERLAP" in note for note in overlapped.notes)
    for depth in (None, 0):
        knobs = Knobs(lookahead_steps=depth, load_grouping=ROLE if depth is None else NEST)
        if depth is None:
            with pytest.raises(BlockScheduleError, match="still live"):
                schedule_contraction(contraction, geometry, knobs)
        else:
            schedule_contraction(contraction, geometry, knobs)   # in nest position it is safe


def test_the_residency_check_catches_a_dropped_and_a_late_load():
    """Mutation controls: a check that cannot fail proves nothing."""
    geometry = _geometry("t_narrow", **NARROW)
    schedule = schedule_contraction(Contraction(m=16, k=64, n=16), geometry)
    check_residency(schedule)                                    # the unmutated schedule passes

    first_load = next(i for i, op in enumerate(schedule.ops) if isinstance(op, Load))
    dropped = type(schedule)(tuple(op for i, op in enumerate(schedule.ops) if i != first_load),
                             schedule.contraction, schedule.geometry, schedule.knobs, schedule.regions)
    with pytest.raises(BlockScheduleError, match="nothing loaded"):
        check_residency(dropped)

    ops = list(schedule.ops)
    last_compute = max(i for i, op in enumerate(ops) if isinstance(op, Compute))
    moved = ops[:first_load] + ops[first_load + 1:last_compute] + [ops[first_load]] + ops[last_compute:]
    late = type(schedule)(tuple(moved), schedule.contraction, schedule.geometry, schedule.knobs,
                          schedule.regions)
    with pytest.raises(BlockScheduleError):
        check_residency(late)


def test_an_out_of_range_row_is_refused():
    geometry = _geometry("t_narrow", **NARROW)
    schedule = schedule_contraction(Contraction(m=16, k=32, n=16), geometry)
    ops = list(schedule.ops)
    index = next(i for i, op in enumerate(ops) if isinstance(op, Load))
    ops[index] = Load(ops[index].role, ops[index].block, ops[index].dram_row, ops[index].dram_col,
                      ops[index].rows, ops[index].cols, geometry.operand_rows - 1, ops[index].step)
    with pytest.raises(BlockScheduleError, match="row store"):
        check_residency(type(schedule)(tuple(ops), schedule.contraction, geometry, schedule.knobs,
                                       schedule.regions))


def test_a_contraction_too_large_for_the_store_is_refused_rather_than_wrapped():
    geometry = _geometry("t_wide", **WIDE)
    with pytest.raises(BlockScheduleError, match="has to be tiled"):
        schedule_contraction(Contraction(m=32, k=32 * 5000, n=32), geometry)


def test_the_interface_adapter_schedules_commits_and_refuses_everything_else():
    geometry = _geometry("t_narrow", **NARROW)
    tensors = {"A0": {"shape": [16, 32], "dtype": "i8"}, "W": {"shape": [32, 16], "dtype": "i8"},
               "Y0": {"shape": [16, 16], "dtype": "i32"}}
    commands = [{"opcode": "RES_PACK", "operands": {"src": "W", "dst": "res"}},
                {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A0", "rhs": "res", "dst": "acc"}},
                {"opcode": "COMMIT", "operands": {"src": "acc", "dst": "Y0"},
                 "attributes": {"epilogue": ["relu"]}},
                {"opcode": "EVICT", "operands": {"handle": "res"}}]
    schedules = schedule_interface_program(tensors, commands, geometry)
    assert len(schedules) == 1 and schedules[0].count(Store) == 1
    assert schedules[0].contraction.lhs == "A0" and schedules[0].contraction.out == "Y0"

    with pytest.raises(BlockScheduleError, match="not a resident matmul"):
        schedule_interface_program(tensors, [{"opcode": "CONV2D", "operands": {}}], geometry)
    pooled = commands[:2] + [{"opcode": "COMMIT", "operands": {"src": "acc", "dst": "Y0"},
                              "attributes": {"epilogue": ["maxpool"]}}]
    with pytest.raises(BlockScheduleError, match="epilogue"):
        schedule_interface_program(tensors, pooled, geometry)


def test_the_defaults_are_the_documented_ones():
    """A silent retune of the defaults is a behaviour change for every caller; pin them."""
    knobs = Knobs()
    assert (knobs.load_on_index_change, knobs.lookahead_steps, tuple(knobs.operand_order),
            knobs.load_grouping, knobs.placement) == (True, 1, (LHS, WEIGHT), ROLE, OPPOSITE_END)


@pytest.mark.parametrize("bad", [Knobs(operand_order=(LHS, LHS)), Knobs(load_grouping="sideways"),
                                 Knobs(placement="wherever"), Knobs(lookahead_steps=-1)])
def test_an_unknown_knob_value_is_refused(bad):
    with pytest.raises(BlockScheduleError):
        schedule_contraction(Contraction(m=16, k=16, n=16), _geometry("t_narrow", **NARROW), bad)
