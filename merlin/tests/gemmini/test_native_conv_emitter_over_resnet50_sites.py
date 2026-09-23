"""What the native EMITTER does with ResNet-50's own 53 convolutions, measured.

`test_native_conv_refusals` measures the SELECTOR against the emission as recorded, where all 53
carry an NCHW im2col layout and a full-width readout because the compiler rewrote them that way
upstream. That census cannot see past its own normalization: the three clauses it reports are about
the rewrite, not about the convolutions, so it says nothing about whether the device could run them.

This asks the other question. Take the same 53 geometries, put them in the form the device's own
store path requires -- NHWC, narrow readout -- and ask the emitter to build a descriptor. It refused
all 53 twice over: first on geometry, because it admitted only stride 1, zero padding and no
dilation; then, once that was widened, on capacity, because it issued a single LOOP_CONV_WS covering
the whole image and no ResNet-50 convolution's staged window fits the derived double-buffer halves.

All 53 are now emitted, as the header's tiler emits them: cut into tiles that fit, one descriptor
per tile, accumulating across the tiles that share an output. Pinning the number here is the point --
the next change to this route is measured against 53, not against a remembered "it works now".
"""

from __future__ import annotations

import importlib
import json
from collections import Counter

import pytest

from merlin.common.paths import merlin_dir
from merlin.runtime.backends import base

_SITES = merlin_dir() / "tests/data/recorded_conv_sites_resnet50.json"


@pytest.fixture(scope="module")
def lc():
    base.get_backend("gemmini")
    return importlib.import_module("merlin._oot_backends.gemmini.gemmini_loop_conv")


def _native_form(site):
    """The recorded site restated in the device's own layout and readout width.

    The recorded operand shapes are NCHW because that is what the rewrite produced; the geometry --
    kernel, stride, padding, dilation, channel counts -- is the convolution's own and is carried
    through unchanged. Nothing here weakens a check: it removes the upstream normalization so the
    question asked is about the convolution rather than about the rewrite.
    """
    attributes = site["attributes"]
    kh, kw, ci, co = attributes["kernel"]
    batch, _, rows, cols = site["operands"]["ifm"]["shape"]
    return dict(kh=kh, kw=kw, ci=ci, co=co, batch=batch, rows=rows, cols=cols, attributes=attributes)


def _emit(lc, contract, form):
    kh, kw, ci, co = form["kh"], form["kw"], form["ci"], form["co"]
    n, h, w = form["batch"], form["rows"], form["cols"]
    a = form["attributes"]
    geometry = lc.native_conv_single_tile(
        batch=n,
        in_rows=h,
        in_cols=w,
        kernel=kh,
        stride=a["stride"][0],
        padding=a["padding"][0],
        kernel_dilation=a["dilation"][0],
    )
    oh, ow = geometry["out_rows"], geometry["out_cols"]
    command = {
        "opcode": "CONV2D",
        "attributes": {
            "kernel": [kh, kw, ci, co],
            "stride": a["stride"],
            "padding": a["padding"],
            "dilation": a["dilation"],
            "layout": "nhwc",
            "epilogue": [],
            "output_dtype": "i8",
        },
        "operands": {"ifm": "X", "weight": "W", "dst": "Y"},
    }
    tensors = {
        "X": {"shape": [n, h, w, ci], "dtype": "i8"},
        "W": {"shape": [kh * kw * ci, co], "dtype": "i8"},
        "Y": {"shape": [n * oh * ow, co], "dtype": "i8"},
    }
    return lc.emit_native_conv(
        command,
        tensors,
        contract=contract,
        pointers={"ifm": "in_p", "weight": "wt_p", "dst": "out_p"},
        row_strides={"ifm": ci, "weight": co, "dst": co},
    )


def test_every_recorded_site_is_emitted(lc):
    """The measurement. 53 convolutions, 53 descriptor programs, no refusals."""
    contract = lc.derive_native_conv_contract()
    sites = json.loads(_SITES.read_text())["sites"]
    assert len(sites) == 53
    emitted, reasons, tiles = 0, Counter(), 0
    for site in sites:
        try:
            receipt = _emit(lc, contract, _native_form(site))
        except lc.UnsupportedNativeConv as exc:
            reasons[str(exc)] += 1
        else:
            emitted += 1
            tiles += receipt["tiles"]
            # Every tile the emitter chose fits what the device declares it can hold at once.
            rows = receipt["capacity"]
            assert rows["input_rows"] + rows["weight_rows"] <= rows["max_spad_rows"]
            assert rows["accumulator_rows"] <= rows["max_acc_rows"]
            assert len(receipt["instructions"]) == 7 * receipt["tiles"]
    assert (emitted, dict(reasons)) == (53, {})
    # The model's whole convolution workload as device descriptors. Pinned as an order of magnitude
    # rather than a number to chase: it moves when the tile search or a shape does, and a change of
    # several times over is a different schedule, not a rounding.
    assert 1000 < tiles < 4000


def test_no_site_is_refused_for_its_geometry_any_more(lc):
    """The half that moved. Every one of the 53 is strided or padded (or, once, both plus a 7x7
    kernel), and every one of those used to be a flat refusal before the staged window was priced.
    A geometry refusal reappearing here is a regression, whatever the capacity verdict says.
    """
    sites = json.loads(_SITES.read_text())["sites"]
    strided = padded = 0
    for site in sites:
        form = _native_form(site)
        a = form["attributes"]
        strided += a["stride"][0] != 1
        padded += a["padding"][0] != 0
        # Raises if the geometry itself is inexpressible; the capacity verdict is downstream of it.
        geometry = lc.native_conv_single_tile(
            batch=form["batch"],
            in_rows=form["rows"],
            in_cols=form["cols"],
            kernel=form["kh"],
            stride=a["stride"][0],
            padding=a["padding"][0],
            kernel_dilation=a["dilation"][0],
        )
        # The staged window is the source image plus its padding, exactly -- never a read past it.
        assert geometry["irows"] - geometry["upad"] - geometry["dpad"] == form["rows"]
        assert geometry["icols"] - geometry["lpad"] - geometry["rpad"] == form["cols"]
    assert (strided, padded) == (7, 17), "the recorded model's own mix, pinned so a resample is visible"
