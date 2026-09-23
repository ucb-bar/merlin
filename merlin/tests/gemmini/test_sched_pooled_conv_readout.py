"""A convolution whose READOUT pools, expressed with no second pass over the output.

ResNet-50's stem is a 7x7 stride-2 convolution followed by a 3x3 stride-2 max pool, and it was the one
convolution group our recipes could not express: the recipe stated ``no_pool`` and the checker refused
anything else. The device does not need a second pass for it. ``LOOP_CONV_WS`` carries the window
(``pool_size``, ``pool_stride``, ``pool_padding``, the pooled output extents and the four pool borders),
and ``LoopConvSt`` has a whole second shape for it: a ``CONFIG_STORE`` carrying the window, one mvout
per batch and output-channel block that reduces the accumulated readout window down to the pooled block
at the ``pool_out_col_dim`` row pitch, then a second ``CONFIG_STORE`` that puts the window back.

Three things had to become true together, and each is pinned below.

**The loop nest walks the POOLED axes.** A tile is ``porows`` x ``pocols`` pooled positions; what it
accumulates is the readout window those positions read, ``porows * pool_stride + pool_size - 1`` less
whatever falls outside the convolution's own output. So adjacent tiles' windows OVERLAP by
``pool_size - pool_stride`` and those convolution outputs are computed twice -- which is what the
library's own tiler does, and why the tile search has to be told the window.

**The checker MODELS the window rather than admitting it.** Both shapes of the store stage are stated,
and an operand set that states neither consistently is refused by the property it got wrong.

**The pooling store path is a BUILD gate.** ``pooling_is_enabled`` is a generator parameter, and a unit
built without it answers a pooling descriptor with the UNPOOLED rows rather than refusing -- so the
capability is read from the target's own elaborated RTL and an underived one refuses.
"""

from __future__ import annotations

import importlib

import pytest

from merlin.runtime.backends import base
from merlin.sched.check.static import check_kernel
from merlin.sched.ir import TensorArg, concretize, instances
from merlin.sched.isa import IsaError


@pytest.fixture(scope="module")
def iset():
    return base.get_backend("gemmini").sched_instruction_set()


@pytest.fixture(scope="module")
def instr(iset):
    return iset.instr("loop_conv_ws")


@pytest.fixture(scope="module")
def sched():
    base.get_backend("gemmini")  # registers the out-of-tree package
    return importlib.import_module("merlin._oot_backends.gemmini.gemmini_sched")


def _legal() -> dict:
    """One tile of a 56x56 64->64 3x3 stride-1 convolution whose readout pools 2x2 stride 2.

    28x28 pooled positions; the tile takes 4x8 of them, and the readout window it accumulates is the
    library's ``porows * pool_stride + pool_size - 1`` -- 9x17, one row and one column past what those
    pooled positions strictly read, which is the library's own allocation and what the RTL addresses
    the accumulator with. No pool border: a 2x2 stride-2 window with no pool padding never reaches past
    the layer's own output.
    """
    return dict(
        batch_size=1,
        in_row_dim=56,
        in_col_dim=56,
        in_channels=64,
        out_channels=64,
        out_row_dim=56,
        out_col_dim=56,
        pool_out_row_dim=28,
        pool_out_col_dim=28,
        stride=1,
        padding=1,
        kernel_dim=3,
        kernel_dilation=1,
        pool_size=2,
        pool_stride=2,
        pool_padding=0,
        batches=1,
        porows=4,
        pocols=8,
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
        orows=9,
        ocols=17,
        weights=1,
        output=2,
        bias=3,
        input=4,
        no_bias=0,
        no_pool=0,
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


#: The same tile with the store stage's OTHER shape: no window at all, which is what this recipe
#: emitted before and still emits for every unpooled layer.
_UNPOOLED = {
    "no_pool": 1,
    "pool_size": 0,
    "pool_stride": 0,
    "pool_padding": 0,
    "pool_out_row_dim": 56,
    "pool_out_col_dim": 56,
    "porows": 8,
    "pocols": 16,
    "orows": 8,
    "ocols": 16,
}


def _state() -> dict:
    return {"config_st": {"stride": 64, "acc_act": 0, "acc_scale": 1.0}}


def test_the_target_states_it_built_the_pooling_store_path(iset):
    assert iset.facts["readout_pooling"] is True, "derived from the elaborated RTL, not declared"


def test_a_pooled_readout_is_legal(instr):
    assert instr.check(_legal(), _state()) == []


def test_the_pooled_extent_must_be_the_layers_own(instr):
    """A ``pool_out_*_dim`` that is not the window's own output makes the store walk the wrong pitch,
    which overwrites a neighbouring tile rather than failing."""
    for field in ("pool_out_row_dim", "pool_out_col_dim"):
        errs = instr.check(_legal() | {field: 27}, _state())
        assert any(field in e and "pooled extent" in e for e in errs), (field, errs)


def test_the_readout_window_must_be_the_one_the_tile_needs(instr):
    """``orows`` is what the descriptor ACCUMULATES, and the pooled tile is what it STORES. A window
    smaller than the pooled positions read leaves the last of them reducing over uninitialised rows."""
    for field, other in (("orows", "porows"), ("ocols", "pocols")):
        errs = instr.check(_legal() | {field: _legal()[field] - 1}, _state())
        assert any(field in e and "readout window" in e for e in errs), (field, errs)
        assert any(str(_legal()[other]) in e for e in errs), errs


def test_the_tile_is_bounded_by_the_POOLED_extent(instr):
    """The axis the tile is named in changed, and so did what bounds it. ``porows`` used to be bounded
    by the convolution's output extent; a pooled layer's is a quarter of that, and a tile between the
    two is inside the old bound while walking positions the output buffer does not have."""
    for field, whole in (("porows", "pool_out_row_dim"), ("pocols", "pool_out_col_dim")):
        errs = instr.check(_legal() | {field: 40, "orows": 81, "ocols": 81}, _state())
        assert any(whole in e and "exceeds" in e for e in errs), (field, errs)


def test_the_window_must_be_a_window(instr):
    for mutation, wanted in (
        ({"pool_size": 0}, "positive window"),
        ({"pool_stride": 0}, "positive window"),
        ({"pool_padding": 2}, "not inside"),
        ({"pool_padding": -1}, "not inside"),
    ):
        errs = instr.check(_legal() | mutation, _state())
        assert any(wanted in e for e in errs), (mutation, errs)


def test_a_pool_border_beside_no_pool_is_refused(instr):
    """The two halves of the store stage, stated together: ``no_pool`` takes the shape that has no
    window at all, so a border for a window is a descriptor that means two different things."""
    unpooled = _legal() | _UNPOOLED
    assert instr.check(unpooled, _state()) == [], "the unpooled descriptor this recipe already emitted"
    assert any("no pooling window" in e for e in instr.check(unpooled | {"pupad": 1}, _state()))
    assert any("window stated beside it" in e for e in instr.check(unpooled | {"pool_size": 2}, _state()))
    assert any("readout window" in e for e in instr.check(unpooled | {"orows": 7}, _state()))


def test_a_target_whose_rtl_does_not_state_a_pooling_store_path_refuses(sched, iset):
    """The build gate, not the encoding. ``pooling_is_enabled`` is elaborated away on a unit configured
    without ``has_max_pool``, and such a unit stores the UNPOOLED rows rather than refusing -- so an
    underived fact must refuse here, where the wrong answer is still a refusal.
    """
    for stated in (None, False):
        facts = dict(iset.facts) | {"readout_pooling": stated}
        errs = sched._conv_readout_window(_legal(), facts=facts)
        assert any("pooling store path" in e and repr(stated) in e for e in errs), (stated, errs)
        unpooled = _legal() | _UNPOOLED
        assert sched._conv_readout_window(unpooled, facts=facts) == [], "an unpooled store needs no gate"
    assert sched._conv_readout_window(_legal(), facts=iset.facts) == []


def test_a_reduction_chain_may_not_change_the_readout(instr):
    """The chain's descriptors accumulate into ONE accumulator half and only the last one stores, so a
    chain that changed the window between them would store a block reduced over two different shapes.

    The window therefore has to be part of what names the output tile, beside the geometry that already
    was -- the two halves of the reduction below differ in nothing but the pool stride, and that has to
    be enough to refuse them.
    """
    from merlin.sched.ir import Ptr

    pointers = {"weights": Ptr("w", 0), "input": Ptr("x", 0), "bias": Ptr("d", 0), "output": None}
    first = _legal() | pointers | {"kchs": 32}
    second = _legal() | {
        "kchs": 32,
        "weights": Ptr("w", 32 * 64),
        "input": Ptr("x", 32),
        "bias": None,
        "output": Ptr("y", 0),
    }
    state: dict = _state()
    assert instr.check(first, state) == []
    assert instr.check(second, state) == [], "the chain the recipe emits"
    for mutation in ({"pool_stride": 1, "orows": 5, "ocols": 9}, {"pool_out_row_dim": 27}, {"no_pool": 1}):
        state = _state()
        assert instr.check(first, state) == []
        errs = instr.check(second | mutation, state)
        assert any("another output tile" in e for e in errs), (mutation, errs)


# --- the recipe ----------------------------------------------------------------------------------


#: (in_dim, ci, co, kernel, stride, padding, pool_size, pool_stride, pool_padding).
#: The first is ResNet-50's stem, the one group of the model that kept its vendor call.
_POOLED = [
    (224, 3, 64, 7, 2, 3, 3, 2, 1),
    (56, 64, 64, 3, 1, 1, 2, 2, 0),
    (28, 128, 256, 1, 1, 0, 3, 2, 1),
]


def _conv(sched, facts, in_dim, ci, co, kernel, stride, padding, ps, pst, ppad):
    out_dim = (in_dim + 2 * padding - kernel) // stride + 1
    pooled = (out_dim + 2 * ppad - ps) // pst + 1 if pst else out_dim
    ops = {
        "input": TensorArg("input", (1, in_dim, in_dim, ci), "i8", "read"),
        "weights": TensorArg("weights", (kernel, kernel, ci, co), "i8", "read"),
        "bias": TensorArg("bias", (co,), "i32", "read"),
        "output": TensorArg("output", (1, pooled, pooled, co), "i8", "write"),
    }
    return sched.conv_reference(
        name="c",
        batch=1,
        in_dim=in_dim,
        in_channels=ci,
        out_channels=co,
        kernel=kernel,
        stride=stride,
        padding=padding,
        operands=ops,
        relu=True,
        scale=0.03125,
        facts=facts,
        pool_size=ps,
        pool_stride=pst,
        pool_padding=ppad,
    )


def _descriptors(kernel):
    return [concretize(call, env) for call, env in instances(kernel) if call.instr == "loop_conv_ws"]


def _library_descriptors(in_dim, ci, co, kernel, stride, padding, ps, pst, ppad, tile, *, eb, ab):
    """``tiled_conv``'s own inner loop, transcribed from the curated header.

    Not a restatement of the recipe: these are the header's expressions for one descriptor -- its two
    nested windows, its four pointers and the slice of the reduction it consumes -- driven by the same
    tile the recipe chose, and the recipe reaches its own values through loop-variable expressions it
    builds independently.
    """
    out_dim = (in_dim + 2 * padding - kernel) // stride + 1
    if pst == 0:
        ps, pst, ppad = 1, 1, 0
    pool_out_dim = (out_dim + 2 * ppad - ps) // pst + 1
    porows, pocols, pochs, kchs = tile["porows"], tile["pocols"], tile["pochs"], tile["kchs"]
    got = []
    for porow in range(0, pool_out_dim, porows):
        orow = porow * pst - ppad
        for pocol in range(0, pool_out_dim, pocols):
            ocol = pocol * pst - ppad
            for poch in range(0, co, pochs):
                irow = max(orow, 0) * stride - padding
                icol = max(ocol, 0) * stride - padding
                for kch in range(0, ci, kchs):
                    out_ptr = (porow * pool_out_dim + pocol) * co + poch
                    porows_ = min(porows, pool_out_dim - porow)
                    pocols_ = min(pocols, pool_out_dim - pocol)
                    orows_ = porows_ * pst + ps - 1
                    ocols_ = pocols_ * pst + ps - 1
                    plpad, pupad = max(0, -ocol), max(0, -orow)
                    prpad = max(0, ocol + ocols_ - out_dim)
                    pdpad = max(0, orow + orows_ - out_dim)
                    icols_ = (ocols_ - plpad - prpad) * stride + kernel - 1
                    irows_ = (orows_ - pupad - pdpad) * stride + kernel - 1
                    lpad, upad = max(0, -icol), max(0, -irow)
                    rpad = max(0, icol + icols_ - in_dim)
                    dpad = max(0, irow + irows_ - in_dim)
                    got.append(
                        dict(
                            porows=porows_,
                            pocols=pocols_,
                            pochs=min(pochs, co - poch),
                            kchs=min(kchs, ci - kch),
                            krows=kernel,
                            kcols=kernel,
                            lpad=lpad,
                            rpad=rpad,
                            upad=upad,
                            dpad=dpad,
                            plpad=plpad,
                            prpad=prpad,
                            pupad=pupad,
                            pdpad=pdpad,
                            orows=orows_ - pupad - pdpad,
                            ocols=ocols_ - plpad - prpad,
                            pool_out_row_dim=pool_out_dim,
                            pool_out_col_dim=pool_out_dim,
                            weights=((kch * co) + poch) * eb,
                            output=None if kch + kchs < ci else out_ptr * eb,
                            bias=poch * ab if kch == 0 else None,
                            input=(((irow + upad) * in_dim + (icol + lpad)) * ci + kch) * eb,
                        )
                    )
    return got


@pytest.mark.parametrize("shape", _POOLED)
def test_every_descriptor_is_the_one_the_library_would_have_issued(sched, iset, shape):
    eb, ab = iset.facts["elem_bytes"], iset.facts["acc_bytes"]
    kernel = _conv(sched, iset.facts, *shape)
    tile = {p.split("=")[0]: int(p.split("=")[1]) for p in dict(kernel.attrs)["tile"].split(",")}
    mine = _descriptors(kernel)
    theirs = _library_descriptors(*shape, tile, eb=eb, ab=ab)
    assert len(mine) == len(theirs) > 0
    for index, (got, want) in enumerate(zip(mine, theirs)):
        for field, value in want.items():
            have = got[field]
            if field in ("weights", "output", "bias", "input"):
                have = None if have is None else have.offset
            assert have == value, (shape, index, field, have, value)


@pytest.mark.parametrize("shape", _POOLED)
def test_the_pooled_recipe_passes_the_static_check(sched, iset, shape):
    kernel = _conv(sched, iset.facts, *shape)
    assert check_kernel(kernel, iset) == [], shape
    attrs = dict(kernel.attrs)
    assert attrs["readout"].startswith("max pool "), attrs
    assert [s.instr for s in kernel.body[:2]] == ["config_ex", "config_st"]


@pytest.mark.parametrize("shape", _POOLED)
def test_the_tiles_cover_every_pooled_position_exactly_once(sched, iset, shape):
    """A nest that skips or repeats a POOLED position is a wrong answer no field check would see --
    and the pooled axis is a different axis from the one the tiles used to be named in."""
    in_dim, ci, co, kernel, stride, padding, ps, pst, ppad = shape
    eb = iset.facts["elem_bytes"]
    out_dim = (in_dim + 2 * padding - kernel) // stride + 1
    pooled = (out_dim + 2 * ppad - ps) // pst + 1
    covered: set[tuple[int, int, int]] = set()
    for v in _descriptors(_conv(sched, iset.facts, *shape)):
        if v["output"] is None:
            continue
        start = v["output"].offset // eb
        poch, pocol, porow = start % co, (start // co) % pooled, start // (co * pooled)
        for r in range(v["porows"]):
            for c in range(v["pocols"]):
                for o in range(v["pochs"]):
                    position = (porow + r, pocol + c, poch + o)
                    assert position not in covered, f"{position} written twice"
                    covered.add(position)
    assert len(covered) == pooled * pooled * co, shape


def test_the_readout_windows_of_adjacent_tiles_overlap_by_what_the_window_costs(sched, iset):
    """The property a pooled tiling has and an unpooled one does not: a ``pool_size``-wide window with a
    smaller stride makes neighbouring tiles share convolution outputs, so those are computed twice. It
    is a real cost, and a tiling that did NOT overlap would be storing a differently-shaped pool.
    """
    shape = (56, 64, 64, 3, 1, 1, 3, 2, 1)
    kernel = _conv(sched, iset.facts, *shape)
    rows = [(v["porows"], v["orows"], v["pupad"], v["pdpad"]) for v in _descriptors(kernel)]
    for porows, orows, pupad, pdpad in rows:
        assert orows == porows * 2 + 3 - 1 - pupad - pdpad
    interior = [(p, o) for p, o, up, dp in rows if up == dp == 0]
    assert interior, rows
    porows, orows = interior[0]
    assert orows - porows * 2 == 3 - 1, "the library allocates pool_size - 1 rows past the strided span"
    assert orows > porows, "a pooled tile accumulates more rows than it stores"


def test_the_tile_search_is_told_the_window(sched, iset):
    """The window enlarges what a tile accumulates without enlarging what it stores, so a search that
    ignored it would choose a tile whose working set overruns the accumulator half -- silently, because
    the sequencer generates its own addresses and nothing bounds them."""
    shape = dict(batch=1, out_dim=112, out_channels=64, kernel=7, stride=2, in_channels=3, facts=iset.facts)
    blind = sched.choose_conv_tiles(**shape)
    pooled = sched.choose_conv_tiles(**shape, pool_out_dim=56, pool_size=3, pool_stride=2)
    assert pooled != blind
    rows = sched.conv_working_rows(
        acc=True,
        stride=2,
        batches=1,
        porows=pooled["porows"],
        pocols=pooled["pocols"],
        pochs=pooled["pochs"],
        krows=7,
        kcols=7,
        kchs=pooled["kchs"],
        pool_size=3,
        pool_stride=2,
        dim=iset.facts["dim"],
    )
    assert rows <= iset.facts["acc_rows_per_loop"]


def test_the_tile_padding_reduces_to_the_unpooled_arithmetic(sched):
    """The defaults are the library's own normalisation of "no pooling", so the path that already
    measured must come back through the generalised arithmetic unchanged."""
    for in_dim, out_dim, kernel, stride, padding in ((56, 56, 3, 1, 1), (224, 112, 7, 2, 3), (56, 28, 3, 2, 1)):
        for start, extent in ((0, out_dim), (0, 8), (8, 8)):
            pads = sched.conv_tile_padding(
                in_dim=in_dim,
                out_dim=out_dim,
                kernel=kernel,
                stride=stride,
                padding=padding,
                start=start,
                extent=extent,
            )
            assert pads["readout"] == extent
            assert (pads["plpad"], pads["prpad"], pads["pupad"], pads["pdpad"]) == (0, 0, 0, 0)
            assert pads["origin"] == start * stride - padding
            assert pads["span"] == extent * stride + kernel - 1


def test_the_stems_own_borders(sched):
    """ResNet-50's stem, pinned per tile: a 3x3 stride-2 pad-1 pool over a 112x112 output.

    The first pooled tile's window begins ONE convolution output above the layer (``pupad = 1``), and
    the input origin is the clamped readout origin walked with the layer's stride less its padding --
    ``0 * 2 - 3``, three rows above the image, which is ``upad = 3``.
    """
    first = sched.conv_tile_padding(
        in_dim=224, out_dim=112, kernel=7, stride=2, padding=3, start=0, extent=6,
        pool_size=3, pool_stride=2, pool_padding=1, pool_out_dim=56,
    )  # fmt: skip
    assert (first["pupad"], first["pdpad"]) == (1, 0)
    assert first["readout"] == 6 * 2 + 3 - 1 - 1
    assert (first["origin"], first["upad"]) == (-3, 3)
    last = sched.conv_tile_padding(
        in_dim=224, out_dim=112, kernel=7, stride=2, padding=3, start=54, extent=2,
        pool_size=3, pool_stride=2, pool_padding=1, pool_out_dim=56,
    )  # fmt: skip
    assert last["pupad"] == 0 and last["pdpad"] == 1, "the last window reaches one row past the output"
    assert last["readout"] == 2 * 2 + 3 - 1 - 1


def test_an_empty_pooling_window_is_refused(sched, iset):
    with pytest.raises(IsaError, match="is empty"):
        _conv(sched, iset.facts, 56, 64, 64, 3, 1, 1, 0, 2, 0)
    with pytest.raises(IsaError, match="not inside"):
        _conv(sched, iset.facts, 56, 64, 64, 3, 1, 1, 2, 2, 2)


def test_a_recipe_on_a_target_that_does_not_state_the_gate_refuses(sched, iset):
    facts = dict(iset.facts) | {"readout_pooling": None}
    with pytest.raises(IsaError, match="pooling store path"):
        _conv(sched, facts, 224, 3, 64, 7, 2, 3, 3, 2, 1)
    assert _conv(sched, facts, 224, 3, 64, 7, 2, 3, 0, 0, 0), "the unpooled layer is unaffected"
