"""The input LEFT IN the scratchpad across a convolution's output-channel tiles.

The input is not an axis of the output-channel tile: ``conv_tile_traffic`` prices its movement as
``... * len(o_t) * kch_ld``, so a layer walking 16 output-channel tiles moves the same window 16 times.
Measured over this model's 20 convolution groups, that repetition is 819,654 of 964,808 input row
transfers, and NOT ONE of the 20 has a single output-channel tile -- so there is no shape here for which
the question answers itself.

The device removes it with two facts, both read off ``LoopConv.scala`` at the pinned revision:
``a_ex_spad_id`` replaces the loader's ``addr_start`` AND the execute stage's ``a_addr_start`` with
``(id-1)*(max_addr/concurrent_loops)`` -- the same expression on both sides, so id ``h`` names half
``h-1`` for the write and for the read; and ``LoopConvLdInput`` holds its entire command queue off on
``req.dram_addr =/= 0.U``, so a NULL input pointer moves nothing at all.

AND IT IS ONLY CORRECT WHERE THE HALF IS STAGED ONCE. A descriptor's input window is a function of the
batch, row, column and reduction tile, so a nest walking several of them reloads the pinned half; the
reservation station tracks that reload as a WAR against the descriptors still reading it and it is
still wrong on the device. That is not inferred, it is measured, three shapes on gsim against the
vendor library over byte-identical operands: one window is bit-exact, six windows differ in 4,427
elements, forty differ in 414. The bound the vendor library gates its own ``a_reuse`` on -- at most
``concurrent_loops`` resident windows, never a reload -- is therefore the device's rule and not that
library being careful.

Two things are tested here and they are different things. The CHECKER must model the mode -- every way
of getting it wrong is refused by a name that says which fact it breaks, and each of those refusals is
mutation-tested, because a check that cannot fail is not a check. The EMITTER must use it exactly where
it holds and keep re-staging where it does not, saying which.
"""

from __future__ import annotations

import importlib

import pytest

from merlin.runtime.backends import base
from merlin.sched.check.static import check_kernel
from merlin.sched.ir import TensorArg
from merlin.sched.isa import IsaError


@pytest.fixture(scope="module")
def iset():
    return base.get_backend("gemmini").sched_instruction_set()


@pytest.fixture(scope="module")
def instr(iset):
    return iset.instr("loop_conv_ws")


@pytest.fixture(scope="module")
def sched():
    return importlib.import_module("merlin._oot_backends.gemmini.gemmini_sched")


def _tile() -> dict:
    """One 8x16 output tile of a 3x3 stride-1 pad-1 56x56 64->64 convolution, reduction whole."""
    return dict(
        batch_size=1,
        in_row_dim=56,
        in_col_dim=56,
        in_channels=64,
        out_channels=64,
        out_row_dim=56,
        out_col_dim=56,
        pool_out_row_dim=56,
        pool_out_col_dim=56,
        stride=1,
        padding=1,
        kernel_dim=3,
        kernel_dilation=1,
        pool_size=0,
        pool_stride=0,
        pool_padding=0,
        batches=1,
        porows=8,
        pocols=16,
        pochs=16,
        krows=3,
        kcols=3,
        kchs=64,
        lpad=0,
        rpad=0,
        upad=0,
        dpad=0,
        plpad=0,
        prpad=0,
        pupad=0,
        pdpad=0,
        orows=8,
        ocols=16,
        weights=1,
        output=2,
        bias=3,
        input=4,
        no_bias=0,
        no_pool=1,
        downsample=0,
        wrot180=0,
        input_dilated=0,
        activation=0,
        trans_output_1203=0,
        trans_weight_1203=0,
        trans_weight_0132=0,
        trans_input_3120=0,
        max_pixels_per_row=1,
        in_stride=64,
        weight_stride=64,
        out_stride=64,
        dw=0,
        a_spad_id=0,
        b_spad_id=0,
    )


def _state() -> dict:
    return {"config_st": {"stride": 64, "acc_act": 0, "acc_scale": 1.0}}


def _stage_then_read(instr, state, *, reader: dict) -> list[str]:
    """Stage the input into half 0, then check ``reader`` against the residency it left."""
    assert instr.check(_tile() | {"a_spad_id": 1}, state) == []
    return instr.check(reader, state)


# --- the checker models the mode ---------------------------------------------------------------------


def test_pinning_the_input_and_then_reading_it_is_legal(instr):
    """The two descriptors the mode IS: one stages half 0, the next names it and passes no pointer."""
    state = _state()
    reader = _tile() | {"a_spad_id": 1, "input": None, "pochs": 32}
    assert _stage_then_read(instr, state, reader=reader) == []


def test_a_null_input_without_a_named_half_is_refused(instr):
    """With ``a_spad_id = 0`` the execute stage reads whichever half its loop SLOT owns, and which slot
    the sequencer hands a descriptor is not a property of the schedule."""
    state = _state()
    errs = _stage_then_read(instr, state, reader=_tile() | {"input": None})
    assert any("a_spad_id names a staged scratchpad half" in e for e in errs), errs
    assert any("whichever scratchpad half its loop SLOT owns" in e for e in errs), errs


def test_a_null_input_against_an_unstaged_half_is_refused(instr):
    """Half 0 holds the input; half 1 holds whatever the last loop to own it left."""
    state = _state()
    errs = _stage_then_read(instr, state, reader=_tile() | {"a_spad_id": 2, "input": None})
    assert any("into which nothing before it staged an input" in e for e in errs), errs


@pytest.mark.parametrize(
    "field,value",
    [
        ("kchs", 32),  # `ichs`: the loader's channel bound, and the A address's block stride
        ("orows", 4),  # `irows` is derived from it, so every row address moves
        ("upad", 1),  # the loader writes at `irow_padded`, the execute stage reads at `irow`
        ("downsample", 1),  # both sides shift their row and column counts by it
        ("in_stride", 128),  # the DRAM row pitch the staged rows were gathered at
        ("max_pixels_per_row", 2),  # how many kernel columns the loader packs into one staged row
        ("porows", 4),  # the output tile moved, so this is not the window that was staged
    ],
)
def test_reading_a_half_staged_for_another_window_is_refused(instr, field, value):
    """Every field the staged rows depend on. A reader that differs in ANY of them is reading bytes
    laid out for a different descriptor, and the answer is silently wrong rather than refused."""
    state = _state()
    reader = _tile() | {"a_spad_id": 1, "input": None, field: value}
    errs = _stage_then_read(instr, state, reader=reader)
    assert any("staged for a different window" in e and field in e for e in errs), (field, errs)


def test_the_output_channel_extent_is_the_one_axis_a_reader_may_move_along(instr):
    """The whole point: nothing in either side's A address depends on ``pochs``."""
    state = _state()
    reader = _tile() | {"a_spad_id": 1, "input": None, "pochs": 48}
    assert _stage_then_read(instr, state, reader=reader) == []


def test_an_unpinned_load_beside_a_pinned_half_is_refused(instr):
    """An unpinned load lands at its loop slot's own ``a_addr_start``, which the RTL sets ONCE at reset
    to ``i*(max_addr/concurrent_loops)`` -- so it may be the very half a later descriptor is relying on."""
    state = _state()
    errs = _stage_then_read(instr, state, reader=_tile())
    assert any("unpinned load lands at the a_addr_start" in e for e in errs), errs


def test_a_half_the_unit_does_not_have_is_refused_by_the_partition_it_names(instr, iset, sched):
    """Derived from the scratchpad/accumulator partition, not written: ``a_spad_id`` may name a half
    only as far as ``conv_loop_slots`` says the sequencer splits the memory."""
    slots = sched.conv_loop_slots(iset.facts)
    errs = instr.check(_tile() | {"a_spad_id": slots + 1}, _state())
    assert any(f"partitions the scratchpad {slots} ways" in e for e in errs), errs
    assert instr.check(_tile() | {"a_spad_id": slots}, _state()) == []


def test_pinning_the_weights_is_still_refused_by_name(instr):
    """``b_spad_id`` moves ``b_addr_end``, which this checker does not model -- so it is refused rather
    than admitted along with the input side."""
    errs = instr.check(_tile() | {"b_spad_id": 1}, _state())
    assert any("b_spad_id pins the WEIGHTS" in e for e in errs), errs


def test_a_null_input_inside_a_split_reduction_is_refused(instr):
    """A chain is verified by how far the input POINTER advanced into the reduction; a NULL pointer
    advances nothing, so the chain would be certified by a rule that never looked at it."""
    state = _state()
    assert instr.check(_tile() | {"a_spad_id": 1}, state) == []
    # withholding its output: this descriptor OPENS a chain
    opener = _tile() | {"a_spad_id": 1, "input": None, "output": None, "kchs": 32}
    assert any("inside a split reduction" in e for e in instr.check(opener, state)), instr.check(opener, state)


def test_re_staging_a_pinned_half_is_refused(instr):
    """THE ONE THE MEASUREMENT PUT HERE. The reservation station records an mvin's destination rows and
    a compute's A rows and makes the reload depend on the read, so a second staging of a half looks like
    it should be ordered behind the descriptors still reading the first. It is not. Pinned across a nest
    that walks 40 input windows, ResNet-50's stem returned 414 of 200,704 elements wrong against the
    vendor library on gsim over byte-identical operands -- while running 1,320,729 cycles against
    1,945,160, which is exactly why a checker that only prices movement would have shipped it."""
    state = _state()
    first = _tile() | {"a_spad_id": 1}
    assert instr.check(first, state) == []
    assert instr.check(_tile() | {"a_spad_id": 1, "input": None, "pochs": 32}, state) == []
    # the next spatial tile of the same layer: a new window, same half
    again = _tile() | {"a_spad_id": 1, "porows": 8, "orows": 8, "upad": 1}
    errs = instr.check(again, state)
    assert any("re-stages scratchpad half 0" in e for e in errs), errs
    assert any("wrong answer rather than a slow one" in e for e in errs), errs


# --- the emitter uses it where it holds, and says where it does not ----------------------------------


def _conv(sched, facts, **kw):
    in_dim, ci, co, k = kw["in_dim"], kw["ci"], kw["co"], kw["kernel"]
    stride, padding = kw["stride"], kw["padding"]
    ps, pst, ppad = kw.get("pool", (0, 0, 0))
    out_dim = (in_dim + 2 * padding - k) // stride + 1
    final = (out_dim + 2 * ppad - ps) // pst + 1 if pst else out_dim
    ops = {
        "input": TensorArg("p_in", (1, in_dim, in_dim, ci), "i8", "read"),
        "weights": TensorArg("p_w", (k, k, ci, co), "i8", "read"),
        "bias": TensorArg("p_bias", (co,), "i32", "read"),
        "output": TensorArg("p_out", (1, final, final, co), "i8", "write"),
    }
    return sched.conv_reference(
        name=kw.get("name", "g"),
        batch=1,
        in_dim=in_dim,
        in_channels=ci,
        out_channels=co,
        kernel=k,
        stride=stride,
        padding=padding,
        operands=ops,
        relu=True,
        scale=0.02,
        facts=facts,
        pool_size=ps,
        pool_stride=pst,
        pool_padding=ppad,
        pin_input=kw.get("pin_input"),
    )


def _descriptors(kernel) -> list:
    """Every DYNAMIC convolution descriptor of a kernel, concrete, in program order."""
    from merlin.sched.ir import concretize, instances

    return [concretize(c, env) for c, env in instances(kernel) if c.instr == "loop_conv_ws"]


#: The three ResNet-50 convolution shapes that stand for the three answers the recipe gives. Each one
#: has been run both ways on gsim against the vendor library over byte-identical operands, and the
#: verdicts are why the rule is what it is:
#:
#:   g38, ONE window, 16 output-channel tiles -- pinned is BIT_EXACT and saves 1,149 of 474,674 cycles;
#:   g1, 40 windows -- pinned DIFFERS in 414 of 200,704 elements (and runs 1,320,729 against 1,945,160,
#:     which is the trap: it looks like a 32% win);
#:   g16, 6 windows -- pinned DIFFERS in 4,427 elements AND runs 590,530 against 557,658.
_BODY = dict(name="g38", in_dim=14, ci=256, co=256, kernel=3, stride=1, padding=1)
_STEM = dict(name="g1", in_dim=224, ci=3, co=64, kernel=7, stride=2, padding=3, pool=(3, 2, 1))
_SPLIT = dict(name="g63", in_dim=7, ci=512, co=512, kernel=3, stride=1, padding=1)


def test_one_window_stages_the_input_once_for_the_whole_kernel(sched, iset):
    """The shape the mode is for: the whole nest stages the window once and 15 descriptors read it.

    The count is the property, not the wording -- the re-staged schedule issues one input load per
    descriptor and the pinned one issues exactly one.
    """
    facts = iset.facts
    kern_off = _conv(sched, facts, **_BODY, pin_input=False)
    kern_on = _conv(sched, facts, **_BODY, pin_input=True)
    off, on = _descriptors(kern_off), _descriptors(kern_on)
    assert len(off) == len(on), "the pin changes which descriptors carry a pointer, not how many there are"
    tiling = dict(t.split("=") for t in dict(kern_on.attrs)["tile"].split(","))
    assert -(-_BODY["co"] // int(tiling["pochs"])) == len(off) > 1, "this shape must walk several output-channel tiles"
    assert sum(1 for d in off if d["input"] is not None) == len(off)
    assert sum(1 for d in on if d["input"] is not None) == 1
    assert all(d["a_spad_id"] != 0 for d in on)
    assert all(d["a_spad_id"] == 0 for d in off)


@pytest.mark.parametrize("shape", [_BODY, _STEM, _SPLIT], ids=lambda s: s["name"])
def test_both_schedules_are_legal_against_the_instruction_set(sched, iset, shape):
    """The checker certifies the recipe's own answer for each shape as a WHOLE program -- residency rule
    included -- and still certifies the re-staged control."""
    for pin in (False, None):
        kern = _conv(sched, iset.facts, **shape, pin_input=pin)
        assert check_kernel(kern, iset) == [], (shape["name"], pin)


def test_the_recipe_pins_only_where_the_kernel_stages_one_window(sched, iset):
    assert dict(_conv(sched, iset.facts, **_BODY).attrs)["input_staging"].startswith("pinned to scratchpad half 0")
    for shape in (_STEM, _SPLIT):
        assert "restaged per output-channel tile" in dict(_conv(sched, iset.facts, **shape).attrs)["input_staging"]


def test_a_nest_that_would_reload_the_pinned_half_is_refused_by_that_name(sched, iset):
    """ResNet-50's stem walks 40 input windows. Pinned, it came back wrong -- so the recipe refuses it
    rather than pricing the movement it would have saved."""
    attrs = dict(_conv(sched, iset.facts, **_STEM).attrs)
    assert "input windows" in attrs["input_staging"]
    with pytest.raises(IsaError, match="re-staged 39 times"):
        _conv(sched, iset.facts, **_STEM, pin_input=True)


def test_a_split_reduction_counts_as_several_windows_and_is_refused(sched, iset):
    """The reduction tile is one of the four axes an input window is a function of, so a split reduction
    is several windows for the same reason a second spatial tile is."""
    attrs = dict(_conv(sched, iset.facts, **_SPLIT).attrs)
    assert "input windows" in attrs["input_staging"]
    with pytest.raises(IsaError, match="reduction tiles"):
        _conv(sched, iset.facts, **_SPLIT, pin_input=True)


def test_one_output_channel_tile_has_nothing_to_pin(sched, iset):
    """A layer the recipe walks with a single output-channel tile re-stages nothing, so the mode buys
    zero and asking for it is a refusal rather than a no-op that reads like a success."""
    small = dict(name="tiny", in_dim=8, ci=16, co=16, kernel=1, stride=1, padding=0)
    attrs = dict(_conv(sched, iset.facts, **small).attrs)
    assert "only one output-channel tile" in attrs["input_staging"]
    with pytest.raises(IsaError, match="only one output-channel tile"):
        _conv(sched, iset.facts, **small, pin_input=True)
