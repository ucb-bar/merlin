"""Which convolutions the native emitter can EXPRESS, and which it still refuses by name.

The selector was widened before the emitter was, and the emitter is the half that decides what runs:
it hard-required ``stride == [1, 1]``, ``padding == [0, 0, 0, 0]``, ``dilation == [1, 1]`` and an
epilogue of at most ``["relu"]`` -- between them, a set containing NO convolution ResNet-50 performs.
Every one of its 53 convolutions is strided or padded or both, and its readouts add a bias, scale the
accumulator and rectify. So the device's convolution sequencer was never asked for any of them, and
the im2col patch generation ran as host scalar code.

What is pinned here is the widening AND its limit. The geometry is derived from the pinned header's
own tiler (`tiled_conv` specialized to the single tile that covers the whole image), the padding is
carried in the fields this ELABORATED revision actually holds registers for, and every mode the
emitter does not model is a refusal that names itself -- never a field quietly left at zero.
"""

from __future__ import annotations

import importlib

import pytest

from merlin.runtime.backends import base


@pytest.fixture(scope="module")
def lc():
    base.get_backend("gemmini")
    return importlib.import_module("merlin._oot_backends.gemmini.gemmini_loop_conv")


@pytest.fixture(scope="module")
def contract(lc):
    return lc.derive_native_conv_contract()


def _command(
    *,
    k=3,
    ci=4,
    co=8,
    n=1,
    h=8,
    w=8,
    s=1,
    p=1,
    d=1,
    epilogue=(),
    acc_scale=None,
    bias=False,
    oh=None,
    ow=None,
    out_dtype="i8",
):
    attributes = {
        "kernel": [k, k, ci, co],
        "stride": [s, s],
        "padding": [p, p, p, p],
        "dilation": [d, d],
        "layout": "nhwc",
        "epilogue": list(epilogue),
        "output_dtype": out_dtype,
    }
    if acc_scale is not None:
        attributes["acc_scale"] = acc_scale
    dilated = k + (d - 1) * (k - 1)
    oh = (h + 2 * p - dilated) // s + 1 if oh is None else oh
    ow = (w + 2 * p - dilated) // s + 1 if ow is None else ow
    operands = {"ifm": "X", "weight": "W", "dst": "Y"}
    tensors = {
        "X": {"shape": [n, h, w, ci], "dtype": "i8"},
        "W": {"shape": [k * k * ci, co], "dtype": "i8"},
        "Y": {"shape": [n * oh * ow, co], "dtype": out_dtype},
    }
    pointers = {"ifm": "in_p", "weight": "wt_p", "dst": "out_p"}
    if bias:
        operands["bias"] = "B"
        tensors["B"] = {"shape": [co], "dtype": "i32"}
        pointers["bias"] = "bias_p"
    return (
        {"opcode": "CONV2D", "attributes": attributes, "operands": operands},
        tensors,
        pointers,
        {"ifm": ci, "weight": co, "dst": co},
    )


def _emit(lc, contract, **kwargs):
    command, tensors, pointers, strides = _command(**kwargs)
    return lc.emit_native_conv(command, tensors, contract=contract, pointers=pointers, row_strides=strides)


# --- the geometry, stated without a hardware contract -------------------------------------------


def test_a_padded_window_pads_on_every_side(lc):
    """`tiled_conv` puts the tile origin at (0, 0), so the first input index is -padding and the pad
    is that much on the near edges; the far pads are the window's overhang past the input extent."""
    g = lc.native_conv_single_tile(batch=1, in_rows=8, in_cols=8, kernel=3, stride=1, padding=1, kernel_dilation=1)
    assert (g["out_rows"], g["out_cols"]) == (8, 8)
    assert (g["lpad"], g["rpad"], g["upad"], g["dpad"]) == (1, 1, 1, 1)
    assert g["irows"] == 8 * 1 + 3 - 1 == g["icols"]


def test_the_padded_window_covers_exactly_the_input_rows(lc):
    """The property the four pads exist for: strip them and what is left is the source extent. A
    stride that does not divide the padded image leaves a partial last window, and the overhang is
    carried as MORE far padding rather than as a read past the tensor."""
    for in_rows, stride, padding, kernel in ((8, 2, 1, 3), (9, 2, 1, 3), (14, 2, 1, 3), (224, 2, 3, 7)):
        g = lc.native_conv_single_tile(
            batch=1, in_rows=in_rows, in_cols=in_rows, kernel=kernel, stride=stride, padding=padding, kernel_dilation=1
        )
        assert g["irows"] - g["upad"] - g["dpad"] == in_rows
        assert g["dpad"] >= padding


def test_a_1x1_stride_2_convolution_downsamples(lc):
    """The header's own predicate. It halves the rows staged for ResNet-50's projection shortcuts."""
    assert (
        lc.native_conv_single_tile(batch=1, in_rows=8, in_cols=8, kernel=1, stride=2, padding=0, kernel_dilation=1)[
            "downsample"
        ]
        == 1
    )
    # Any of the predicate's terms failing takes the general path instead of a wrong fast one.
    for kw in ({"kernel": 3}, {"stride": 1}, {"in_rows": 7}, {"padding": 0}):
        base_kw = {"batch": 1, "in_rows": 8, "in_cols": 8, "kernel": 1, "stride": 2, "padding": 0, "kernel_dilation": 1}
        if kw == {"padding": 0}:
            continue
        assert lc.native_conv_single_tile(**{**base_kw, **kw})["downsample"] == 0


def test_a_dilated_kernel_widens_the_window(lc):
    g = lc.native_conv_single_tile(batch=1, in_rows=8, in_cols=8, kernel=3, stride=1, padding=1, kernel_dilation=2)
    assert g["out_rows"] == (8 + 2 - 5) + 1 == 6
    assert g["irows"] == 6 + 5 - 1


def test_padding_at_or_past_the_kernel_is_refused(lc):
    """The pinned header asserts `kernel_dim must be larger than padding`, against the UNdilated
    kernel. Reading it any other way admits a window the sequencer cannot address."""
    with pytest.raises(lc.UnsupportedNativeConv, match="0_le_padding_lt_kernel"):
        lc.native_conv_single_tile(batch=1, in_rows=8, in_cols=8, kernel=3, stride=1, padding=3, kernel_dilation=1)


# --- the descriptor the emitter builds ----------------------------------------------------------


def test_a_strided_convolution_is_expressed_at_all(lc, contract):
    """The regression this file exists for: stride 2 used to be a flat refusal."""
    receipt = _emit(lc, contract, k=3, s=2, p=1, h=8, w=8)
    assert receipt["parameters"]["stride"] == 2
    assert receipt["parameters"]["orows"] == receipt["parameters"]["porows"] == 4
    # The execute unit's A_stride is what turns the sequencer's output walk into a strided input
    # walk. Pinned at 1, the descriptor is well-formed and the arithmetic is not the convolution.
    assert receipt["entry_strides"]["a_stride"] == 2


def test_a_padded_convolution_carries_its_padding_and_its_four_pads(lc, contract):
    """Both, because the device wants both.

    This assertion used to read the other way. `padding` has no `%loops_*` register in the
    elaborated design, and that was taken as proof the device derives everything from lpad/rpad/
    upad/dpad -- so the emitter passed 0 there and the descriptor decode proof, which can only see
    registers, reported success. On hardware the padded convolution then computed the wrong answer:
    282 of 512 outputs differed from the header's own conv_cpu while the header's own tiler, given
    the same inputs, differed in none. Absence of a register is absence of PROOF, not evidence of
    inertness, and the emitter now carries what the reference tiler carries.
    """
    receipt = _emit(lc, contract, k=3, s=1, p=1, h=8, w=8)
    parameters = receipt["parameters"]
    assert (parameters["lpad"], parameters["rpad"], parameters["upad"], parameters["dpad"]) == (1, 1, 1, 1)
    assert parameters["padding"] == 1
    assert "padding" not in contract.descriptor_fields
    assert receipt["descriptor_to_rtl_field_qualification"]["fields_outside_this_proof"]["padding"] == 1


def test_an_absent_bias_is_the_headers_sentinel_and_not_the_null_pointer(lc, contract):
    """`tiled_conv` substitutes its own address for a bias it will not load and sets no_bias. Zero is
    the plausible value and it is not the one the device is given anywhere else."""
    assert contract.absent_bias_address != 0
    receipt = _emit(lc, contract, k=3, s=1, p=1)
    assert receipt["parameters"]["no_bias"] == 1
    assert receipt["parameters"]["bias"] == contract.absent_bias_address


def test_several_pixels_share_a_staged_row_when_the_header_says_they_do(lc, contract):
    """`sp_tiled_conv`'s pixel packing. The sequencer stages AND walks rows by this number, so one
    pixel per row is not a conservative default -- it is a different program from the configured
    one, and it was measured wrong on hardware."""
    dim = contract.dim
    # in_channels well under the row width: several pixels fit, capped by the kernel's width.
    assert _emit(lc, contract, k=3, ci=4, co=8)["parameters"]["max_pixels_per_row"] == min(dim // 4, 3)
    # A full row per pixel leaves room for exactly one.
    assert _emit(lc, contract, k=1, ci=dim, co=dim, p=0)["parameters"]["max_pixels_per_row"] == 1
    # Dilating the kernel makes the packed row's pixels non-adjacent, so the header stops packing.
    assert _emit(lc, contract, k=3, ci=4, co=8, d=2)["parameters"]["max_pixels_per_row"] == 1


def test_a_single_tile_identifies_itself_as_the_first_staged_buffer(lc, contract):
    """The header's reuse predicate: with one tile every operand is staged once, and the descriptor
    names that buffer rather than leaving the field at a zero that means something else."""
    parameters = _emit(lc, contract, k=3, s=1, p=1)["parameters"]
    assert (parameters["a_spad_id"], parameters["b_spad_id"]) == (1, 1)


def test_a_downsampling_convolution_agrees_with_its_execute_stride(lc, contract):
    """downsample and A_stride are one decision: the halved mvin is only the same convolution when
    the execute unit stops striding as well."""
    receipt = _emit(lc, contract, k=1, ci=16, co=16, s=2, p=0, h=8, w=8)
    assert receipt["parameters"]["downsample"] == 1
    assert receipt["entry_strides"]["a_stride"] == 1


def test_a_dilated_convolution_is_expressed(lc, contract):
    receipt = _emit(lc, contract, k=3, s=1, p=1, d=2, h=8, w=8)
    assert receipt["parameters"]["kernel_dilation"] == 2


def test_the_readouts_full_epilogue_is_expressed(lc, contract):
    """bias, then a per-tensor accumulator scale, then rectification -- the readout ResNet-50 asks
    for at every convolution, and the one the emitter used to refuse outright."""
    receipt = _emit(lc, contract, epilogue=("bias_add", "acc_scale", "relu"), acc_scale=0.25, bias=True)
    assert receipt["parameters"]["no_bias"] == 0
    assert receipt["parameters"]["bias"] == "bias_p"
    assert receipt["parameters"]["activation"] == 1
    assert receipt["entry_strides"]["acc_scale"] == 0.25
    assert receipt["epilogue"]["stages"] == ["bias_add", "acc_scale", "relu"]


def test_the_capacity_check_prices_the_padded_window(lc, contract):
    """The staged input is the window the sequencer walks, not the source tensor. Pricing the source
    tensor instead understates a padded convolution and overstates a strided one."""
    receipt = _emit(lc, contract, k=3, s=1, p=1, h=8, w=8, ci=4, co=8)
    # ceil(4/16) * 1 * 10 * 10 for the padded window, against 8*8 for the bare input.
    assert receipt["capacity"]["input_rows"] == 100
    assert receipt["tiles"] == 1, "this one fits whole"


def test_a_convolution_too_large_for_the_buffers_is_cut_into_tiles(lc, contract):
    """ResNet-50's own shape. The header cuts it; so does the emitter, one descriptor per tile."""
    receipt = _emit(lc, contract, k=3, ci=64, co=64, h=56, w=56, s=1, p=1)
    assert receipt["tiles"] > 1
    assert len(receipt["instructions"]) == 7 * receipt["tiles"]
    rows = receipt["capacity"]
    assert rows["input_rows"] + rows["weight_rows"] <= rows["max_spad_rows"]
    assert rows["accumulator_rows"] <= rows["max_acc_rows"]


def test_only_the_last_contributor_to_an_output_stores_it(lc, contract):
    """A convolution cut along its INPUT channels accumulates across tiles, so most of its tiles are
    partial. A tiling that stored on every pass would write partial sums over finished outputs, and
    one that loaded the bias on every pass would add it once per pass. The header passes a null for
    both, and a null is not offset zero.

    ResNet-50's 512-channel 3x3 is such a convolution: 512 input channels do not fit one tile.
    """
    receipt = _emit(lc, contract, k=3, ci=512, co=512, h=7, w=7, s=1, p=1)
    tiles = receipt["tile_parameters"]
    stores = [each for each in tiles if each["output"] != 0]
    biases = [each for each in tiles if each["bias"] != 0]
    assert receipt["tile"]["kchs"] < 512, "this shape is cut along its input channels"
    assert 0 < len(stores) < receipt["tiles"], "a partial tile must not store"
    assert 0 < len(biases) < receipt["tiles"], "the bias enters the accumulator once"
    # One store per output-channel group, and the last tile of the convolution is one of them.
    assert len(stores) == len(biases) == receipt["tiles"] // (512 // receipt["tile"]["kchs"] + 1)
    assert tiles[-1]["output"] != 0


# --- what must still refuse ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        ({"out_dtype": "i32"}, "narrow saturating output"),
        ({"p": 3, "k": 3}, "0_le_padding_lt_kernel"),
        ({"epilogue": ("acc_scale",), "acc_scale": None}, "UNKNOWN acc_scale multiplier"),
        ({"epilogue": ("relu", "acc_scale"), "acc_scale": 0.5}, "not the order the device readout"),
        ({"epilogue": ("maxpool",)}, "unsupported rich epilogue"),
        ({"epilogue": ("requant",)}, "unsupported rich epilogue"),
        ({"epilogue": ("bias_add",)}, "names no bias operand"),
        ({"bias": True}, "no bias stage is declared"),
    ],
)
def test_a_mode_the_emitter_does_not_model_refuses_by_name(lc, contract, kwargs, fragment):
    with pytest.raises(lc.UnsupportedNativeConv, match=fragment):
        _emit(lc, contract, **kwargs)


def test_a_device_whose_buffers_hold_nothing_still_refuses(lc, contract, monkeypatch):
    """A convolution larger than the buffers is now CUT, not refused, so the capacity clause is only
    reached by a device that cannot hold one output pixel of one channel. It is still reached."""
    monkeypatch.setattr(contract, "capacity", {"max_spad_rows": 0, "max_acc_rows": 0})
    with pytest.raises(lc.UnsupportedNativeConv, match="exceeds derived double-buffer capacity"):
        _emit(lc, contract, k=3, s=1, p=1)


def test_a_nonuniform_stride_or_padding_refuses(lc, contract):
    """The descriptor carries ONE stride and one padding; a per-axis one is a different operation,
    not one to take the first component of."""
    command, tensors, pointers, strides = _command(k=3, s=1, p=1)
    command["attributes"]["stride"] = [1, 2]
    with pytest.raises(lc.UnsupportedNativeConv, match="uniform_2d_geometry"):
        lc.emit_native_conv(command, tensors, contract=contract, pointers=pointers, row_strides=strides)
    command, tensors, pointers, strides = _command(k=3, s=1, p=1)
    command["attributes"]["padding"] = [1, 1, 2, 1]
    with pytest.raises(lc.UnsupportedNativeConv, match="uniform_padding"):
        lc.emit_native_conv(command, tensors, contract=contract, pointers=pointers, row_strides=strides)


def test_a_macro_parameter_nothing_assigns_refuses(lc, contract, monkeypatch):
    """The fail-closed property a widening most easily loses. Every parameter of the header's macro
    is named deliberately; one this header grows must stop the emission, not inherit a zero that
    happens to read as 'feature off'."""
    import dataclasses

    header = contract.header
    grown = dataclasses.replace(
        header.macro("gemmini_loop_conv_ws"),
        params=tuple(header.macro("gemmini_loop_conv_ws").params) + ("a_mode_nobody_modelled",),
    )

    class _Grown:
        def macro(self, name):
            return grown if name == "gemmini_loop_conv_ws" else header.macro(name)

    monkeypatch.setattr(contract, "header", _Grown())
    with pytest.raises(lc.UnsupportedNativeConv, match="UNKNOWN native macro parameter"):
        _emit(lc, contract, k=3, s=1, p=1)
