"""The device convolution sequencer, as a schedule instruction this target's header defines.

This is the instruction the whole-model emission never issues. Measured on ResNet-50: the program used
8 of the 25 functs the RTL declares, and the 17 it never emitted are this family -- the loop plus its
six config words -- so every convolution's im2col patch generation ran as host scalar code, 37% of all
host dynamic operations.

Two properties are worth pinning. Its operand list is the header's, so a header that changes its macro
changes the instruction rather than silently disagreeing with it; and its legality check REFUSES every
mode it does not model instead of passing it. The loop issues its own mvin/mvout, so nothing downstream
would catch a schedule whose movement the checker never reasoned about.
"""

from __future__ import annotations

import pytest

from merlin.runtime.backends import base


@pytest.fixture(scope="module")
def instr():
    return base.get_backend("gemmini").sched_instruction_set().instr("loop_conv_ws")


def _legal() -> dict:
    """One 8x16 output tile of a 3x3 stride-1 pad-1 56x56 64->64 convolution.

    A tile, not the whole layer: the layer's own output needs 12544 accumulator rows against the 512 one
    loop owns, and :func:`_whole_layer` is that case, kept as the refusal it must be.
    """
    return _whole_layer() | {"porows": 8, "pocols": 16, "orows": 8, "ocols": 16, "pochs": 16}


def _whole_layer() -> dict:
    """The same convolution asked for in ONE descriptor -- the shape that silently overran."""
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
        porows=56,
        pocols=56,
        pochs=64,
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
        orows=56,
        ocols=56,
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


def test_the_operand_list_is_the_headers(instr):
    """54 operands, in the macro's own order -- not a list retyped here that could drift from it."""
    names = [o.name for o in instr.operands]
    assert len(names) == 54
    assert names[0] == "batch_size" and names[-1] == "b_spad_id"
    assert [o.name for o in instr.operands if o.kind == "ptr"] == ["weights", "output", "bias", "input"]
    assert "gemmini_loop_conv_ws" in instr.doc


def test_a_real_convolution_is_legal(instr):
    assert instr.check(_legal(), _state()) == []


def test_the_output_extent_must_be_the_layers_own(instr):
    v = _legal() | {"out_row_dim": 28}
    assert any("output extent" in e for e in instr.check(v, _state()))


def test_the_tile_must_lie_inside_the_problem(instr):
    for field, value, whole in (
        ("kchs", 128, "in_channels"),
        ("pochs", 65, "out_channels"),
        ("krows", 4, "kernel_dim"),
        ("batches", 2, "batch_size"),
    ):
        errs = instr.check(_legal() | {field: value}, _state())
        assert any(whole in e for e in errs), (field, errs)


def test_every_unmodelled_mode_is_refused_rather_than_passed(instr):
    """A checker silent about a mode certifies a schedule it never reasoned about."""
    for field, value in (
        ("dw", 1),
        ("wrot180", 1),
        ("downsample", 1),
        ("input_dilated", 1),
        ("trans_weight_1203", 1),
        ("trans_input_3120", 1),
        ("kernel_dilation", 2),
        # The weight side of the scratchpad pinning is still unmodelled. The INPUT side is modelled now
        # (``test_sched_conv_input_residency``), so ``a_spad_id`` is refused only where it names a half
        # the partition does not have -- not for being set.
        ("b_spad_id", 1),
        ("a_spad_id", 3),
    ):
        assert instr.check(_legal() | {field: value}, _state()), field


def test_a_descriptor_that_states_both_shapes_of_the_store_stage_is_refused(instr):
    """``no_pool`` picks between two shapes of ``LoopConvSt``, and a descriptor that half-states the
    other one means two things at once. The pooled shape itself is modelled; see
    ``test_sched_pooled_conv_readout``."""
    assert any("window stated beside it" in e for e in instr.check(_legal() | {"pool_size": 2}, _state()))
    assert any("no pooling window" in e for e in instr.check(_legal() | {"pupad": 1}, _state()))
    errs = instr.check(_legal() | {"no_pool": 0, "pool_size": 2, "pool_stride": 2}, _state())
    assert any("pooled extent" in e for e in errs), errs


def test_a_descriptor_that_begins_an_output_tile_must_carry_a_bias_pointer(instr):
    """The bias mvin is the ONLY thing that initialises the accumulator half.

    ``LoopConvLdBias`` skips its load outright when the pointer is zero (``skip = dram_addr === 0``),
    and ``LoopConvExecute`` preloads with ``accumulate = true`` unconditionally. So a descriptor that
    starts an output tile with a NULL bias adds its result onto whatever that half last held -- a wrong
    answer with nothing anywhere to say so. ``no_bias`` is not a substitute: it makes the same mvin
    write zeros, which still needs a pointer to be issued at all.
    """
    assert any("accumulator half" in e for e in instr.check(_legal() | {"bias": None}, _state()))
    errs = instr.check(_legal() | {"bias": None, "no_bias": 1}, _state())
    assert any("skip that load entirely" in e for e in errs), errs
    assert instr.check(_legal() | {"no_bias": 1}, _state()) == [], "zeroing the half with a pointer is legal"


def test_max_pixels_per_row_must_fit_one_mesh_row(instr):
    """It packs several kernel columns into one mesh row, so it is bounded by the row, not by taste."""
    assert any("mesh row" in e for e in instr.check(_legal() | {"max_pixels_per_row": 2}, _state()))
    thin = _legal() | {"in_channels": 4, "kchs": 4, "in_stride": 4, "max_pixels_per_row": 3}
    assert instr.check(thin, _state()) == [], "three columns of four channels do fit a 16-wide row"
    assert any("kernel columns" in e for e in instr.check(thin | {"max_pixels_per_row": 4}, _state()))


def test_a_field_that_does_not_fit_its_descriptor_is_caught_from_the_header(instr):
    """The selector spells this as a 16-bit overflow clause. Here it is whatever the header's own
    packing expression admits, so a header that repacks the word moves the limit with it."""
    big = 1 << 20
    v = _legal() | {"in_row_dim": big, "out_row_dim": (big + 2 - 3) + 1}
    errs = instr.check(v, _state())
    assert any("CONFIG" in e and "overflow" in e for e in errs), errs


def test_a_store_configuration_must_precede_the_loop(instr):
    """The loop stores its own output, so the readout it will use is state, not an operand."""
    assert any("config_st" in e for e in instr.check(_legal(), {}))
    mismatched = {"config_st": {"stride": 64, "acc_act": 1, "acc_scale": 1.0}}
    assert any("activation" in e for e in instr.check(_legal(), mismatched))


def test_a_tile_larger_than_the_scratchpad_half_is_refused(instr):
    """The tile must FIT, not merely lie inside the problem.

    A whole 56x56 64-channel layer as one descriptor is what this recipe used to emit, and every field
    of it is inside its problem -- which is why the static check passed it and the ResNet-50 program it
    built came back with all nineteen convolutions corrupt. The sequencer generates its own accumulator
    addresses and nothing bounds them, so the overrun is a wrong answer rather than a refusal.
    """
    errs = instr.check(_whole_layer(), _state())
    assert any("accumulator rows" in e for e in errs), errs
    assert any("12544" in e for e in errs), "the refusal says how many rows the tile actually wants"
    assert instr.check(_legal(), _state()) == [], "the tile that fits is still legal"


def test_a_tile_whose_weights_overrun_the_scratchpad_is_refused(instr):
    """The other half of the same property: the operand side, not the output side."""
    v = _legal() | {
        "in_channels": 512,
        "kchs": 512,
        "out_channels": 512,
        "pochs": 512,
        "porows": 1,
        "pocols": 1,
        "orows": 1,
        "ocols": 1,
    }
    assert any("scratchpad rows" in e for e in instr.check(v, _state()))


# --- the recipe: one layer as a nest of device-convolution descriptors ---------------------------


@pytest.fixture(scope="module")
def sched():
    import importlib

    base.get_backend("gemmini")  # registers the out-of-tree package
    return importlib.import_module("merlin._oot_backends.gemmini.gemmini_sched")


@pytest.mark.parametrize(
    "in_dim,out_dim,kernel,stride,padding,want",
    [
        (56, 56, 3, 1, 1, (1, 1)),  # the 58x58 zero-padded form the external kernels build by hand
        (224, 112, 7, 2, 3, (3, 3)),  # ResNet-50's stem
        (56, 56, 1, 1, 0, (0, 0)),  # a projection: no border at all
        (56, 28, 3, 2, 1, (1, 1)),  # a strided block entry
    ],
)
def test_the_tile_padding_is_the_library_tilers_own(sched, in_dim, out_dim, kernel, stride, padding, want):
    """A padding this gets wrong is a silently wrong output, not a refusal, so it is pinned per shape."""
    pads = sched.conv_tile_padding(in_dim=in_dim, out_dim=out_dim, kernel=kernel, stride=stride, padding=padding)
    assert (pads["lpad"], pads["rpad"]) == want
    assert pads["lpad"] == pads["upad"] and pads["rpad"] == pads["dpad"], "square layer, square border"


#: The convolutions of the ResNet-50 group model, as (in_dim, ci, co, kernel, stride, padding).
_RESNET50_CONVS = [
    (56, 64, 64, 3, 1, 1),
    (56, 128, 128, 3, 2, 1),
    (56, 256, 512, 1, 2, 0),
    (28, 128, 128, 3, 1, 1),
    (28, 256, 256, 3, 2, 1),
    (28, 512, 1024, 1, 2, 0),
    (14, 256, 256, 3, 1, 1),
    (14, 512, 512, 3, 2, 1),
    (14, 1024, 2048, 1, 2, 0),
    (7, 512, 512, 3, 1, 1),
]


def _conv(sched, iset, in_dim, ci, co, kernel, stride, padding, *, relu=True):
    from merlin.sched.ir import TensorArg

    out_dim = (in_dim + 2 * padding - kernel) // stride + 1
    ops = {
        "input": TensorArg("input", (1, in_dim, in_dim, ci), "i8", "read"),
        "weights": TensorArg("weights", (kernel, kernel, ci, co), "i8", "read"),
        "bias": TensorArg("bias", (co,), "i32", "read"),
        "output": TensorArg("output", (1, out_dim, out_dim, co), "i8", "write"),
    }
    return sched.conv_reference(
        name="c1",
        batch=1,
        in_dim=in_dim,
        in_channels=ci,
        out_channels=co,
        kernel=kernel,
        stride=stride,
        padding=padding,
        operands=ops,
        relu=relu,
        scale=0.03125,
        facts=iset.facts,
    )


def _chains(kernel):
    """The descriptors of one kernel grouped into reduction chains, in program order."""
    from merlin.sched.ir import concretize, instances

    chains, current = [], []
    for call, env in instances(kernel):
        if call.instr != "loop_conv_ws":
            continue
        v = concretize(call, env)
        current.append(v)
        if v["output"] is not None:
            chains.append(current)
            current = []
    assert current == [], "a chain was left open"
    return chains


def test_the_recipe_emits_a_tile_nest_that_passes_the_static_check(sched):
    """Every descriptor of every ResNet-50 convolution fits the half the sequencer owns.

    Before this held, the recipe emitted the whole layer as one descriptor: legal by every field check
    and wrong on every output, because the tile's working set is nowhere near the scratchpad and
    accumulator halves one loop is given.
    """
    from merlin.sched.check.static import check_kernel

    iset = base.get_backend("gemmini").sched_instruction_set()
    for shape in _RESNET50_CONVS:
        k = _conv(sched, iset, *shape)
        assert check_kernel(k, iset) == [], shape
        assert [s.instr for s in k.body[:2]] == ["config_ex", "config_st"]
        assert len(_chains(k)) == int(dict(k.attrs)["descriptors"]) // int(dict(k.attrs)["reduction_steps"])


def test_the_strided_layers_row_stride_reaches_the_execute_configuration(sched):
    """``config_ex``'s A_stride is the CONVOLUTION's stride on this path, not the matmul path's 1.

    The library's own convolution configures ``stride >> downsample`` there, and the sequencer walks the
    input rows with it; leaving it 1 reads a stride-2 layer off by a row with nothing to say so.
    """
    iset = base.get_backend("gemmini").sched_instruction_set()
    strided = 0
    for shape in _RESNET50_CONVS:
        k = _conv(sched, iset, *shape)
        # The descriptors' own bit decides it, so this reads the pair off the emitted kernel rather
        # than restating the recipe. With the strided load the loader has already applied the stride
        # and the execute unit must step by one; setting either field alone computes a different
        # convolution and says nothing about it.
        bits = {chain[0]["downsample"] for chain in _chains(k)}
        assert len(bits) == 1, (shape, bits)
        downsample = bits.pop()
        strided += downsample
        assert dict(k.body[0].args)["A_stride"].value == shape[4] >> downsample, shape
        assert dict(k.body[0].args)["sys_act"].value == iset.facts["constants"]["NO_ACTIVATION"], shape
    # And the pair is actually exercised: a model whose every layer loads unstrided would pass the
    # assertions above without ever testing the mode.
    assert strided, "no ResNet-50 convolution took the strided load, so the pairing is untested here"


def test_every_descriptor_is_the_one_the_library_would_have_issued(sched):
    """The nest, instance by instance, against ``tiled_conv``'s own inner-loop arithmetic.

    Not a restatement of the recipe: the expected values below are the header's expressions for one
    descriptor -- its border, its three operand pointers and the slice of the reduction it consumes --
    and the recipe reaches them through loop-variable expressions it builds independently. The tiles
    must also cover every output position exactly once, and every chain must cover every input channel
    exactly once: a nest that skips or repeats one is a wrong answer no field check would see.

    A descriptor reading an input LEFT IN the scratchpad carries no pointer (see
    ``test_sched_conv_input_residency``), and the expected pointer is then checked against the one the
    last staging descriptor actually moved -- which is the whole obligation of that mode: the window a
    reader implies must be the window the stager left.
    """
    iset = base.get_backend("gemmini").sched_instruction_set()
    eb, ab = iset.facts["elem_bytes"], iset.facts["acc_bytes"]
    for in_dim, ci, co, kernel, stride, padding in _RESNET50_CONVS:
        k = _conv(sched, iset, in_dim, ci, co, kernel, stride, padding)
        out_dim = (in_dim + 2 * padding - kernel) // stride + 1
        covered: set[tuple[int, int, int]] = set()
        staged: dict[int, int] = {}
        for chain in _chains(k):
            # Where this chain's output tile starts, read back from the pointer its LAST descriptor
            # carries -- the only one that has it, which is the property being tested.
            start = chain[-1]["output"].offset // eb
            poch = start % co
            pocol = (start // co) % out_dim
            porow = start // (co * out_dim)
            for r in range(chain[-1]["porows"]):
                for c in range(chain[-1]["pocols"]):
                    for o in range(chain[-1]["pochs"]):
                        position = (porow + r, pocol + c, poch + o)
                        assert position not in covered, f"{position} written twice"
                        covered.add(position)
            irow, icol = porow * stride - padding, pocol * stride - padding
            irows = chain[-1]["porows"] * stride + kernel - 1
            icols = chain[-1]["pocols"] * stride + kernel - 1
            kch = 0
            for step, v in enumerate(chain):
                assert (v["upad"], v["lpad"]) == (max(0, -irow), max(0, -icol))
                assert (v["dpad"], v["rpad"]) == (max(0, irow + irows - in_dim), max(0, icol + icols - in_dim))
                assert (v["krows"], v["kcols"]) == (kernel, kernel), "the kernel window is never split"
                # tiled_conv's own slicing: out is NULL until the last reduction step, the bias is NULL
                # after the first, and the two operand pointers step by the channels already consumed.
                assert (v["output"] is None) == (step != len(chain) - 1)
                assert (v["bias"] is None) == (step != 0)
                if step == 0:
                    assert v["bias"].offset == poch * ab
                assert v["weights"].offset == (kch * co + poch) * eb
                want_input = (((irow + v["upad"]) * in_dim + (icol + v["lpad"])) * ci + kch) * eb
                if v["input"] is None:
                    assert v["a_spad_id"], "a descriptor with no input pointer must name the half it reads"
                    assert staged[v["a_spad_id"]] == want_input, "the reader implies a window nobody staged"
                else:
                    assert v["input"].offset == want_input
                    if v["a_spad_id"]:
                        staged[v["a_spad_id"]] = v["input"].offset
                assert (v["orows"], v["ocols"]) == (v["porows"], v["pocols"]), "no pooling: the readout is the tile"
                kch += v["kchs"]
            assert kch == ci, f"the chain reduces {kch} of {ci} input channels"
        assert len(covered) == out_dim * out_dim * co, (in_dim, ci, co, kernel, stride, padding)


#: The layers whose reduction the tile search splits, and what it costs to hold it whole instead.
#: Measured on gsim against the vendor library, the four of them ran 1.7x, 2.0x, 8.4x and 4.9x the
#: library's cycles with the reduction whole -- the whole of a 1.74x convolution gap.
_SPLIT_SHAPES = [(28, 256, 256, 3, 2, 1), (14, 512, 512, 3, 2, 1), (14, 1024, 2048, 1, 2, 0), (7, 512, 512, 3, 1, 1)]

#: The layers it leaves whole, where the movement a split saves does not pay for the loop slot it
#: forfeits. Measured, the recipe is already 0.78-0.92x the library on every one of them.
_WHOLE_SHAPES = [(56, 64, 64, 3, 1, 1), (28, 128, 128, 3, 1, 1), (14, 256, 256, 3, 1, 1), (56, 128, 128, 3, 2, 1)]


def test_the_reduction_is_split_where_holding_it_whole_starves_the_output_tile(sched):
    """The regression this recipe exists to fix.

    A descriptor's weights occupy ``ceil(pochs/DIM) * krows * kcols * kchs`` scratchpad rows, so a deep
    layer that holds its reduction whole pays for it out of the output tile: measured, the 512->512 3x3
    stride-2 layer fell to 16 output channels over 2 output rows -- 128 descriptors, each re-moving the
    whole input window -- and ran 8.4x the vendor library. Splitting the reduction buys the output tile
    back.
    """
    iset = base.get_backend("gemmini").sched_instruction_set()
    for in_dim, ci, co, kernel, stride, padding in _SPLIT_SHAPES:
        out_dim = (in_dim + 2 * padding - kernel) // stride + 1
        tile = sched.choose_conv_tiles(
            batch=1,
            out_dim=out_dim,
            out_channels=co,
            kernel=kernel,
            stride=stride,
            in_channels=ci,
            facts=iset.facts,
        )
        whole = sched._conv_output_tile(
            batch=1,
            out_dim=out_dim,
            out_channels=co,
            kernel=kernel,
            stride=stride,
            depth=ci,
            facts=iset.facts,
        )
        assert tile["kchs"] < ci, (in_dim, ci, co)
        assert tile["pochs"] > whole["pochs"], "the room the split frees goes to the output tile"
        k = _conv(sched, iset, in_dim, ci, co, kernel, stride, padding)
        assert int(dict(k.attrs)["reduction_steps"]) > 1


def test_the_reduction_stays_whole_where_splitting_it_would_only_forfeit_a_loop_slot(sched):
    """The other half of the trade, and the reason the objective is not movement alone.

    The sequencer runs two loops at once and descriptors on distinct output tiles occupy both; the
    descriptors of one reduction chain all accumulate into the same accumulator half and serialise.
    Measured: the library's own search splits the 14x14 256->256 layer into 240 and 16 channels for
    about 40% less movement, and that schedule ran 1.28x the one that keeps the reduction whole. A
    search that priced movement alone would take that trade on every one of these layers.
    """
    iset = base.get_backend("gemmini").sched_instruction_set()
    for in_dim, ci, co, kernel, stride, padding in _WHOLE_SHAPES:
        k = _conv(sched, iset, in_dim, ci, co, kernel, stride, padding)
        assert dict(k.attrs)["reduction_steps"] == "1", (in_dim, ci, co)
        assert all(len(chain) == 1 for chain in _chains(k))


def test_the_chosen_tile_is_the_cheapest_candidate_by_the_stated_model(sched):
    """The search is a model over a candidate family, so the two must not drift apart."""
    iset = base.get_backend("gemmini").sched_instruction_set()
    slots = sched.conv_loop_slots(iset.facts)
    assert slots == iset.facts["spad_rows"] // iset.facts["spad_rows_per_loop"] > 1
    for in_dim, ci, co, kernel, stride, padding in _RESNET50_CONVS:
        out_dim = (in_dim + 2 * padding - kernel) // stride + 1
        shape = dict(batch=1, out_dim=out_dim, out_channels=co, kernel=kernel, stride=stride, in_channels=ci)

        def priced(tile):
            t = sched.conv_tile_traffic(**shape, tile=tile, facts=iset.facts)
            rows = t["input"] + t["weights"] + t["bias"] + t["output"]
            return rows if t["reduction_steps"] > 1 else -(-rows // slots)

        candidates = sched.conv_tile_candidates(**shape, facts=iset.facts)
        chosen = sched.choose_conv_tiles(**shape, facts=iset.facts)
        assert chosen in candidates
        assert priced(chosen) == min(priced(c) for c in candidates), (in_dim, ci, co)


def test_a_layer_whose_kernel_window_alone_cannot_fit_is_refused(sched):
    """Fail closed. The kernel window is never split, so a layer whose window alone overruns is named.

    Splitting the reduction goes as far as one input channel per descriptor; below that the only thing
    left to give up is the kernel window, whose split moves each descriptor's input origin and border
    with the kernel position -- which neither the recipe nor the checker states.
    """
    from merlin.sched.isa import IsaError

    iset = base.get_backend("gemmini").sched_instruction_set()
    with pytest.raises(IsaError, match="kernel window whole"):
        _conv(sched, iset, 128, 64, 64, 128, 1, 0)


def test_a_convolution_with_no_bias_is_refused(sched):
    """The sequencer initialises the accumulator half from the bias mvin and from nothing else."""
    from merlin.sched.ir import TensorArg
    from merlin.sched.isa import IsaError

    iset = base.get_backend("gemmini").sched_instruction_set()
    ops = {
        "input": TensorArg("input", (1, 14, 14, 64), "i8", "read"),
        "weights": TensorArg("weights", (3, 3, 64, 64), "i8", "read"),
        "output": TensorArg("output", (1, 14, 14, 64), "i8", "write"),
    }
    with pytest.raises(IsaError, match="no bias"):
        sched.conv_reference(
            name="c1",
            batch=1,
            in_dim=14,
            in_channels=64,
            out_channels=64,
            kernel=3,
            stride=1,
            padding=1,
            operands=ops,
            relu=False,
            scale=1.0,
            facts=iset.facts,
        )


def test_a_layer_with_no_output_is_refused(sched):
    from merlin.sched.ir import TensorArg
    from merlin.sched.isa import IsaError

    iset = base.get_backend("gemmini").sched_instruction_set()
    ops = {
        "input": TensorArg("input", (1, 2, 2, 64), "i8", "read"),
        "weights": TensorArg("weights", (7, 7, 64, 64), "i8", "read"),
        "bias": TensorArg("bias", (64,), "i32", "read"),
        "output": TensorArg("output", (1, 1, 1, 64), "i8", "write"),
    }
    with pytest.raises(IsaError, match="no output"):
        sched.conv_reference(
            name="c1",
            batch=1,
            in_dim=2,
            in_channels=64,
            out_channels=64,
            kernel=7,
            stride=1,
            padding=0,
            operands=ops,
            relu=False,
            scale=1.0,
            facts=iset.facts,
        )


# --- the reduction chain: what a descriptor that withholds its output owes the next one ------------


def _ptr(tensor: str, offset: int):
    from merlin.sched.ir import ConcretePtr

    return ConcretePtr(tensor, offset)


def _chain_step(kchs: int, kch: int, *, first: bool, last: bool) -> dict:
    """One descriptor of a 64-channel reduction over the legal tile, sliced ``kchs`` at a time."""
    return _legal() | {
        "kchs": kchs,
        "weights": _ptr("w", kch * 64),  # weight_stride = out_channels = 64, one byte an element
        "input": _ptr("in", kch),
        "bias": _ptr("bias", 0) if first else None,
        "output": _ptr("out", 0) if last else None,
    }


def test_a_split_reduction_in_the_shape_the_sequencer_requires_is_legal(instr):
    """Two descriptors, one accumulator half: bias on the first, output on the last, 32 + 32 = 64."""
    state = _state()
    assert instr.check(_chain_step(32, 0, first=True, last=False), state) == []
    assert state["pending_conv_partial"] is not None, "the checker is carrying the partial forward"
    assert instr.check(_chain_step(32, 32, first=False, last=True), state) == []
    assert state["pending_conv_partial"] is None, "and lets go of it once the sum is stored"


def test_a_partial_sum_must_not_reload_the_bias(instr):
    """The bias mvin OVERWRITES the accumulator half; reloading it throws the partial away."""
    state = _state()
    assert instr.check(_chain_step(32, 0, first=True, last=False), state) == []
    errs = instr.check(_chain_step(32, 32, first=True, last=True), state)
    assert any("must not reload the bias" in e for e in errs), errs


def test_a_partial_sum_must_not_be_abandoned_for_another_output_tile(instr):
    state = _state()
    assert instr.check(_chain_step(32, 0, first=True, last=False), state) == []
    other = _chain_step(32, 32, first=False, last=True) | {"porows": 4, "orows": 4}
    errs = instr.check(other, state)
    assert any("abandoned" in e for e in errs), errs


def test_a_split_reduction_must_cover_every_input_channel(instr):
    """The one thing a per-descriptor check cannot see and a wrong answer depends on."""
    state = _state()
    assert instr.check(_chain_step(32, 0, first=True, last=False), state) == []
    errs = instr.check(_chain_step(16, 32, first=False, last=True), state)
    assert any("48 of the layer's 64" in e for e in errs), errs


def test_a_continued_partial_must_start_where_its_predecessor_ended(instr):
    """A mis-stepped slice recomputes or skips channels, which no field check would see."""
    for operand, wrong in (("weights", _ptr("w", 16 * 64)), ("input", _ptr("in", 16))):
        state = _state()
        assert instr.check(_chain_step(32, 0, first=True, last=False), state) == []
        errs = instr.check(_chain_step(32, 32, first=False, last=True) | {operand: wrong}, state)
        assert any(e.startswith(f"{operand} advances") for e in errs), (operand, errs)


def test_a_descriptor_withholding_an_already_complete_reduction_is_refused(instr):
    """Withholding the output of a whole reduction leaves a finished sum nothing will ever store."""
    errs = instr.check(_chain_step(64, 0, first=True, last=False), _state())
    assert any("nothing after it will store" in e for e in errs), errs


def test_a_reduction_split_across_kernel_positions_is_refused(instr):
    """krows/kcols below the kernel extent move the input origin and border with the position."""
    errs = instr.check(_legal() | {"krows": 1}, _state())
    assert any("kernel window" in e for e in errs), errs


def test_the_kernel_may_not_end_holding_a_partial_sum(sched):
    """The whole-kernel rule, on a real schedule with its last store removed."""
    from merlin.runtime.backends import base as _base
    from merlin.sched.check.static import check_kernel
    from merlin.sched.ir import NULL, Call, Kernel

    iset = _base.get_backend("gemmini").sched_instruction_set()
    k = _conv(sched, iset, 14, 512, 512, 3, 2, 1)
    assert check_kernel(k, iset) == []
    body = list(k.body)
    tail = body[-1]
    inner = list(tail.body)
    last = inner[-1]
    inner[-1] = Call(last.instr, tuple((n, NULL if n == "output" else v) for n, v in last.args))
    body[-1] = type(tail)(tail.var, tail.extent, tuple(inner))
    errs = check_kernel(Kernel(k.name, k.args, tuple(body), k.attrs), iset)
    assert any("never stored" in e for e in errs), errs
