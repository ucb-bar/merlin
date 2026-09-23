"""The block-schedule pass: knobs over DERIVED geometry, and a residency check that fails closed.

Every geometry here is synthesized as a facts artifact and derived through
``targetgen.address_space``, at two different array edges and bank depths, so nothing in the pass can
be shaped around one device. What the knobs are worth in cycles is measured elsewhere (the capsule
head-to-head); what is pinned here is that each knob does what it says, that a geometry the facts
cannot answer is refused rather than defaulted, and that a schedule which would read a block some
earlier load overwrote never leaves the pass.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from merlin.compile.scheduling import (
    AXES,
    BANK_ALIGNED,
    CONTIGUOUS,
    LHS,
    NEST,
    OPPOSITE_END,
    ROLE,
    WEIGHT,
    BlockScheduleError,
    Compute,
    Contraction,
    ConvContraction,
    Geometry,
    K,
    Knobs,
    Load,
    M,
    N,
    Preload,
    Store,
    check_residency,
    execute,
    geometry_from_address_space,
    schedule_contraction,
    schedule_convolution,
    schedule_interface_program,
)
from merlin.targetgen.address_space import derive_address_space


def _facts(
    edge, operand_bytes, operand_depth, accum_bytes, accum_depth, *, arrays=None, memories=None, operand_dtype="i8"
):
    """A synthetic facts artifact in the shape ``load_facts`` returns."""
    body = {
        "arrays": [{"name": "mesh", "rows": edge, "cols": edge}] if arrays is None else arrays,
        # STORE names are deliberately not merlin's role names: the pass must resolve stores by row width,
        # and a fixture spelled "scratchpad"/"accumulator" would pass whether or not a name lookup crept
        # back in. The accumulate DATAPATH does carry the facts schema's role -- that declaration is the
        # evidence that results accumulate in the store it fills.
        "datapaths": [
            {"name": "input", "dtype": operand_dtype, "evidence": "sram_a smem UInt<8>"},
            {"name": "accumulator", "dtype": "i32", "evidence": "sram_b smem SInt<32>"},
        ],
    }
    if memories is None:
        # Declared wide-row store FIRST, so a resolver leaning on declaration order picks wrong.
        memories = [
            {"name": "sram_b", "bytes": accum_bytes, "depth": accum_depth},
            {"name": "sram_a", "bytes": operand_bytes, "depth": operand_depth},
        ]
    body["memories"] = memories
    return {"schema_version": "2.0", "inputs": {}, "facts": body}


def _geometry(name, **kwargs):
    return geometry_from_address_space(derive_address_space(name, facts=_facts(**kwargs)))


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


@pytest.mark.parametrize(
    "broken, why",
    [
        (dict(arrays=[]), "no array geometry"),
        (dict(arrays=[{"name": "mesh", "rows": 16, "cols": 8}]), "not square"),
        (dict(memories=[{"name": "sram_b", "bytes": 65536, "depth": 512}]), "accumulator is unknown"),
        (dict(memories=[]), "no operand store"),
    ],
)
def test_a_geometry_the_facts_cannot_answer_is_refused_not_defaulted(broken, why):
    """The failure this repo keeps re-learning: an unmeasurable quantity reported as a plausible
    number. Every branch that cannot derive the geometry has to refuse."""
    with pytest.raises(BlockScheduleError, match=why):
        _geometry("t_broken", **{**NARROW, **broken})


@pytest.mark.parametrize("name", sorted(GEOMETRIES))
def test_loading_on_index_change_moves_each_streamed_block_once(name):
    geometry = _geometry(f"t_{name}", **GEOMETRIES[name])
    d = geometry.block
    contraction = Contraction(m=2 * d, k=2 * d, n=3 * d)  # 2 x 2 x 3 blocks
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
    contraction = Contraction(m=d, k=4 * d, n=d)  # four reduction steps
    first_compute = {}
    for depth in (0, 1, 2, None):
        schedule = schedule_contraction(contraction, geometry, Knobs(lookahead_steps=depth))
        ops = list(schedule.ops)
        first_compute[depth] = next(i for i, op in enumerate(ops) if isinstance(op, Compute))
        loads_before = sum(1 for op in ops[: first_compute[depth]] if isinstance(op, Load))
        # depth d issues steps 0..d before the first compute; None issues every step's loads.
        assert loads_before == (2 * (depth + 1) if depth is not None else 8), (depth, loads_before)
    assert first_compute[0] < first_compute[1] < first_compute[2] < first_compute[None]


@pytest.mark.parametrize("order", [(LHS, WEIGHT), (WEIGHT, LHS)])
def test_operand_order_orders_the_loads_inside_a_group(order):
    geometry = _geometry("t_narrow", **NARROW)
    d = geometry.block
    schedule = schedule_contraction(
        Contraction(m=d, k=2 * d, n=d), geometry, Knobs(lookahead_steps=1, load_grouping=ROLE, operand_order=order)
    )
    assert [op.role for op in _loads(schedule)][:2] == list(order)


def test_nest_grouping_keeps_each_streamed_load_beside_its_own_compute():
    """The interleaved shape: with no lookahead and nest grouping, a streamed block is loaded
    immediately before the compute that reads it, not batched with its step."""
    geometry = _geometry("t_narrow", **NARROW)
    d = geometry.block
    schedule = schedule_contraction(
        Contraction(m=2 * d, k=d, n=d), geometry, Knobs(lookahead_steps=0, load_grouping=NEST)
    )
    kinds = [type(op).__name__ for op in schedule.ops]
    assert kinds == ["Load", "Load", "Preload", "Compute", "Store", "Load", "Preload", "Compute", "Store"], kinds


@pytest.mark.parametrize("name", sorted(GEOMETRIES))
def test_each_placement_policy_puts_the_resident_region_where_it_says(name):
    geometry = _geometry(f"t_{name}", **GEOMETRIES[name])
    d = geometry.block
    contraction = Contraction(m=d, k=2 * d, n=d)
    streamed_rows = 2 * d
    bases = {
        policy: schedule_contraction(contraction, geometry, Knobs(placement=policy)).regions[WEIGHT][0]
        for policy in (OPPOSITE_END, BANK_ALIGNED, CONTIGUOUS)
    }
    assert bases[OPPOSITE_END] == geometry.operand_rows - streamed_rows
    assert bases[BANK_ALIGNED] == geometry.operand_bank_rows, "the first bank boundary past the inputs"
    assert bases[CONTIGUOUS] == streamed_rows, "directly after the inputs, sharing their bank"


def test_bank_aligned_placement_is_refused_when_the_facts_have_no_bank_depth():
    geometry = _geometry("t_narrow", **NARROW)
    without_banks = Geometry(
        block=geometry.block,
        operand_rows=geometry.operand_rows,
        operand_bank_rows=None,
        accumulator_rows=geometry.accumulator_rows,
        separate_accumulator_space=True,
    )
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
            schedule_contraction(contraction, geometry, knobs)  # in nest position it is safe


def test_the_residency_check_catches_a_dropped_and_a_late_load():
    """Mutation controls: a check that cannot fail proves nothing."""
    geometry = _geometry("t_narrow", **NARROW)
    schedule = schedule_contraction(Contraction(m=16, k=64, n=16), geometry)
    check_residency(schedule)  # the unmutated schedule passes

    first_load = next(i for i, op in enumerate(schedule.ops) if isinstance(op, Load))
    dropped = type(schedule)(
        tuple(op for i, op in enumerate(schedule.ops) if i != first_load),
        schedule.contraction,
        schedule.geometry,
        schedule.knobs,
        schedule.regions,
    )
    with pytest.raises(BlockScheduleError, match="nothing loaded"):
        check_residency(dropped)

    ops = list(schedule.ops)
    last_compute = max(i for i, op in enumerate(ops) if isinstance(op, Compute))
    moved = ops[:first_load] + ops[first_load + 1 : last_compute] + [ops[first_load]] + ops[last_compute:]
    late = type(schedule)(tuple(moved), schedule.contraction, schedule.geometry, schedule.knobs, schedule.regions)
    with pytest.raises(BlockScheduleError):
        check_residency(late)


def test_an_out_of_range_row_is_refused():
    geometry = _geometry("t_narrow", **NARROW)
    schedule = schedule_contraction(Contraction(m=16, k=32, n=16), geometry)
    ops = list(schedule.ops)
    index = next(i for i, op in enumerate(ops) if isinstance(op, Load))
    ops[index] = Load(
        ops[index].role,
        ops[index].block,
        ops[index].dram_row,
        ops[index].dram_col,
        ops[index].rows,
        ops[index].cols,
        geometry.operand_rows - 1,
        ops[index].step,
    )
    with pytest.raises(BlockScheduleError, match="row store"):
        check_residency(type(schedule)(tuple(ops), schedule.contraction, geometry, schedule.knobs, schedule.regions))


def test_a_contraction_too_large_for_the_store_is_refused_rather_than_wrapped():
    geometry = _geometry("t_wide", **WIDE)
    with pytest.raises(BlockScheduleError, match="has to be tiled"):
        schedule_contraction(Contraction(m=32, k=32 * 5000, n=32), geometry)


def test_the_interface_adapter_schedules_commits_and_refuses_everything_else():
    geometry = _geometry("t_narrow", **NARROW)
    tensors = {
        "A0": {"shape": [16, 32], "dtype": "i8"},
        "W": {"shape": [32, 16], "dtype": "i8"},
        "Y0": {"shape": [16, 16], "dtype": "i32"},
    }
    commands = [
        {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "res"}},
        {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A0", "rhs": "res", "dst": "acc"}},
        {"opcode": "COMMIT", "operands": {"src": "acc", "dst": "Y0"}, "attributes": {"epilogue": ["relu"]}},
        {"opcode": "EVICT", "operands": {"handle": "res"}},
    ]
    schedules = schedule_interface_program(tensors, commands, geometry)
    assert len(schedules) == 1 and schedules[0].count(Store) == 1
    assert schedules[0].contraction.lhs == "A0" and schedules[0].contraction.out == "Y0"

    with pytest.raises(BlockScheduleError, match="neither a resident matmul nor a convolution"):
        schedule_interface_program(tensors, [{"opcode": "MOVEMENT", "operands": {}}], geometry)
    with pytest.raises(BlockScheduleError, match="missing"):
        schedule_interface_program(tensors, [{"opcode": "CONV2D", "operands": {}}], geometry)
    pooled = commands[:2] + [
        {"opcode": "COMMIT", "operands": {"src": "acc", "dst": "Y0"}, "attributes": {"epilogue": ["maxpool"]}}
    ]
    with pytest.raises(BlockScheduleError, match="epilogue"):
        schedule_interface_program(tensors, pooled, geometry)


def test_the_defaults_are_the_documented_ones():
    """A silent retune of the defaults is a behaviour change for every caller; pin them."""
    knobs = Knobs()
    assert (
        knobs.load_on_index_change,
        knobs.lookahead_steps,
        tuple(knobs.operand_order),
        knobs.load_grouping,
        knobs.placement,
    ) == (True, 1, (LHS, WEIGHT), ROLE, OPPOSITE_END)


@pytest.mark.parametrize(
    "bad",
    [
        Knobs(operand_order=(LHS, LHS)),
        Knobs(load_grouping="sideways"),
        Knobs(placement="wherever"),
        Knobs(lookahead_steps=-1),
    ],
)
def test_an_unknown_knob_value_is_refused(bad):
    with pytest.raises(BlockScheduleError):
        schedule_contraction(Contraction(m=16, k=16, n=16), _geometry("t_narrow", **NARROW), bad)


# --------------------------------------------------------------------------- loop order + execution


ORDERS = list(itertools.permutations(AXES))


def _operands(contraction, seed=0):
    rng = np.random.default_rng(seed)
    return (
        rng.integers(-128, 128, size=(contraction.m, contraction.k)),
        rng.integers(-128, 128, size=(contraction.k, contraction.n)),
    )


@pytest.mark.parametrize("name", sorted(GEOMETRIES))
@pytest.mark.parametrize("order", ORDERS, ids=lambda o: "".join(o))
@pytest.mark.parametrize(
    "knobs",
    [
        Knobs(load_on_index_change=False, lookahead_steps=0, load_grouping=NEST),
        Knobs(load_on_index_change=True, lookahead_steps=0, load_grouping=NEST),
        Knobs(lookahead_steps=1, load_grouping=ROLE),
        Knobs(lookahead_steps=None, load_grouping=NEST),
    ],
    ids=["v0", "l1", "la1", "hoist"],
)
def test_every_loop_order_computes_the_contraction_exactly(name, order, knobs):
    """Executed at the addresses the schedule names, every nest gives ``lhs @ weight`` -- on ragged edges
    in all three axes, so partial blocks are exercised too."""
    geometry = _geometry(f"t_{name}", **GEOMETRIES[name])
    d = geometry.block
    contraction = Contraction(m=2 * d + 3, k=3 * d + 1, n=2 * d + 5)
    lhs, weight = _operands(contraction)
    schedule = schedule_contraction(contraction, geometry, Knobs(**{**knobs.__dict__, "loop_order": order}))
    assert np.array_equal(execute(schedule, lhs, weight), lhs @ weight)


@pytest.mark.parametrize("order", ORDERS, ids=lambda o: "".join(o))
def test_load_on_index_change_loads_each_block_once_in_any_order(order):
    geometry = _geometry("t_narrow", **NARROW)
    d = geometry.block
    contraction = Contraction(m=2 * d, k=2 * d, n=3 * d)  # 2 x 2 x 3 blocks
    once = schedule_contraction(contraction, geometry, Knobs(loop_order=order))
    assert len(_loads(once, LHS)) == 2 * 2 and len(_loads(once, WEIGHT)) == 2 * 3
    assert once.count(Compute) == 2 * 2 * 3


def test_the_default_order_is_reduction_major():
    assert Knobs().loop_order == (K, N, M)


def test_a_loop_order_that_is_not_a_permutation_is_refused():
    with pytest.raises(BlockScheduleError, match="permutation"):
        schedule_contraction(Contraction(16, 16, 16), _geometry("t_narrow", **NARROW), Knobs(loop_order=(K, K, M)))


def test_execution_catches_a_schedule_that_drains_the_wrong_accumulator_block():
    """The executor is an oracle, not a restatement: corrupt one drain and the product is wrong; drain a
    block nothing computed and it refuses."""
    geometry = _geometry("t_narrow", **NARROW)
    d = geometry.block
    contraction = Contraction(m=2 * d, k=d, n=d)
    schedule = schedule_contraction(contraction, geometry, Knobs())
    lhs, weight = _operands(contraction)
    stores = [i for i, op in enumerate(schedule.ops) if isinstance(op, Store)]
    first, second = schedule.ops[stores[0]], schedule.ops[stores[1]]

    def _with(index, op):
        ops = list(schedule.ops)
        ops[index] = op
        return type(schedule)(
            tuple(ops), schedule.contraction, schedule.geometry, schedule.knobs, schedule.regions, schedule.notes
        )

    stale = _with(stores[1], Store(second.dram_row, second.dram_col, second.rows, second.cols, first.accumulator_row))
    assert not np.array_equal(execute(stale, lhs, weight), lhs @ weight)
    early = _with(stores[0], Store(first.dram_row, first.dram_col, first.rows, first.cols, second.accumulator_row))
    with pytest.raises(BlockScheduleError, match="no compute ever wrote"):
        execute(early, lhs, weight)


# ------------------------------------------------------------------------------------ convolution


def _conv_reference(ifm, weight, conv):
    """Direct NHWC convolution into the ``[pixels, co]`` layout, weights ``[kh*kw*ci, co]``."""
    x = ifm.reshape(conv.batch, conv.in_h, conv.in_w, conv.ci)
    w = weight.reshape(conv.kh, conv.kw, conv.ci, conv.co)
    pt, pl, _, _ = conv.padding
    out = np.zeros((conv.batch, conv.out_h, conv.out_w, conv.co), dtype=np.int64)
    for b in range(conv.batch):
        for oh in range(conv.out_h):
            for ow in range(conv.out_w):
                for kr in range(conv.kh):
                    for kc in range(conv.kw):
                        ih = oh * conv.stride[0] - pt + kr * conv.dilation[0]
                        iw = ow * conv.stride[1] - pl + kc * conv.dilation[1]
                        if 0 <= ih < conv.in_h and 0 <= iw < conv.in_w:
                            out[b, oh, ow] += x[b, ih, iw].astype(np.int64) @ w[kr, kc]
    return out.reshape(conv.pixels, conv.co)


CONVS = {
    "k3_pad0": ConvContraction(1, 8, 8, 4, 3, 3, 8),
    "k3_pad1": ConvContraction(1, 8, 8, 4, 3, 3, 8, padding=(1, 1, 1, 1)),
    "k3_stride2": ConvContraction(1, 8, 8, 4, 3, 3, 8, stride=(2, 2)),
    "ragged_channels_batch2": ConvContraction(2, 7, 9, 20, 3, 3, 40, padding=(1, 1, 1, 1)),
    "dilated": ConvContraction(1, 9, 9, 5, 3, 3, 17, padding=(2, 2, 2, 2), dilation=(2, 2)),
    "one_by_one": ConvContraction(1, 6, 6, 33, 1, 1, 20),
}


@pytest.mark.parametrize("name", sorted(GEOMETRIES))
@pytest.mark.parametrize("conv", sorted(CONVS))
@pytest.mark.parametrize(
    "knobs",
    [
        Knobs(load_on_index_change=False, lookahead_steps=0, load_grouping=NEST),
        Knobs(load_on_index_change=True, lookahead_steps=0, load_grouping=NEST),
        Knobs(lookahead_steps=1, load_grouping=ROLE),
        Knobs(lookahead_steps=1, load_grouping=ROLE, loop_order=(M, K, N)),
    ],
    ids=["v0", "l1", "la1", "la1_mkn"],
)
def test_a_scheduled_convolution_computes_the_convolution_exactly(name, conv, knobs):
    """Padding taps read the zero path, strides and dilations pick the gathered pixels, and ragged
    channel slices and output-channel blocks are partial blocks -- all checked against a direct conv."""
    geometry = _geometry(f"t_{name}", **GEOMETRIES[name])
    spec = CONVS[conv]
    rng = np.random.default_rng(1)
    ifm = rng.integers(-128, 128, size=(spec.batch * spec.in_h * spec.in_w, spec.ci))
    weight = rng.integers(-128, 128, size=(spec.kh * spec.kw * spec.ci, spec.co))
    schedule = schedule_convolution(spec, geometry, knobs)
    assert np.array_equal(execute(schedule, ifm, weight), _conv_reference(ifm, weight, spec))


def test_load_on_index_change_gathers_each_tap_once_per_pixel_block_not_per_output_channel_block():
    geometry = _geometry("t_narrow", **NARROW)
    spec = CONVS["ragged_channels_batch2"]  # 18 tap slices, 8 pixel blocks, 3 output-channel blocks
    every = schedule_convolution(
        spec, geometry, Knobs(load_on_index_change=False, lookahead_steps=0, load_grouping=NEST)
    )
    once = schedule_convolution(spec, geometry, Knobs(load_on_index_change=True, lookahead_steps=0, load_grouping=NEST))
    assert len(_loads(every, LHS)) == 18 * 8 * 3 and len(_loads(once, LHS)) == 18 * 8
    assert every.count(Compute) == once.count(Compute) == 18 * 8 * 3


def test_a_load_carrying_a_different_gather_into_a_live_block_is_refused():
    """The gather is part of a block's identity: same block position, different pixels, different bytes."""
    geometry = _geometry("t_narrow", **NARROW)
    spec = CONVS["k3_pad1"]
    schedule = schedule_convolution(spec, geometry, Knobs(lookahead_steps=0, load_grouping=NEST))
    loads = [i for i, op in enumerate(schedule.ops) if isinstance(op, Load) and op.role == LHS]
    first, other = schedule.ops[loads[0]], schedule.ops[loads[1]]
    assert first.gather != other.gather
    ops = list(schedule.ops)
    ops[loads[0]] = Load(
        first.role,
        first.block,
        first.dram_row,
        first.dram_col,
        first.rows,
        first.cols,
        first.row,
        first.step,
        other.gather,
    )
    corrupted = type(schedule)(
        tuple(ops), schedule.contraction, schedule.geometry, schedule.knobs, schedule.regions, schedule.notes
    )
    with pytest.raises(BlockScheduleError, match="DIFFERENT gather"):
        check_residency(corrupted)


def test_the_adapter_schedules_a_conv_command_and_refuses_pooling():
    geometry = _geometry("t_narrow", **NARROW)
    tensors = {"IFM": {"shape": [1, 8, 8, 4]}, "W": {"shape": [36, 8]}, "Y0": {"shape": [64, 8]}}
    conv = {
        "opcode": "CONV2D",
        "operands": {"ifm": "IFM", "weight": "W_res", "dst": "Y0"},
        "attributes": {
            "kernel": [3, 3, 4, 8],
            "stride": [1, 1],
            "padding": [1, 1, 1, 1],
            "dilation": [1, 1],
            "layout": "nhwc",
            "epilogue": ["relu"],
        },
    }
    pack = {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "W_res"}}
    [schedule] = schedule_interface_program(tensors, [pack, conv], geometry, Knobs())
    direct = schedule_convolution(CONVS["k3_pad1"], geometry, Knobs())
    assert schedule.ops == direct.ops
    pooled = {**conv, "attributes": {**conv["attributes"], "epilogue": ["maxpool"]}}
    with pytest.raises(BlockScheduleError, match="maxpool"):
        schedule_interface_program(tensors, [pack, pooled], geometry, Knobs())


def test_the_pass_module_imports_nothing_from_merlin_so_a_backend_can_vendor_it():
    """Generated backends may not depend on merlin at run time and copy this module verbatim instead; the
    package integrity scan rejects any merlin import, so a stray one here would fail every such package."""
    from merlin.compile.scheduling import block_schedule
    from merlin.targetgen.oot_runner import _py_imports_merlin

    with open(block_schedule.__file__, encoding="utf-8") as handle:
        assert _py_imports_merlin(handle.read()) is None
