"""The native convolution route is asked for, and its refusal says something true.

Two defects met here. The route ran only when a caller constructed a contract and passed it, and NO
production caller ever did -- so every CONV2D took the host im2col rewrite and the device's own
sequencer was never asked. And the structural check guarding the route compared the resident pack's
layout against ``packed_rhs``, which is the MATMUL spelling; a convolution weight pack is
``packed_conv_rhs``, which is what `corpus_spec` emits. So every convolution capsule in the corpus was
turned away at a check about *packing and ordering* whose packing and ordering were both already
correct, before any property of the convolution was examined.

Neither fix may change what is emitted. Measured when they landed: 0 of 16 conv capsules become
admissible, because the real blockers are the narrow-readout and layout clauses underneath -- so the
value here is a refusal census that names them instead of naming the wrong thing 16 times.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from merlin.common.paths import merlin_dir
from merlin.runtime.backends import base as _bk
from merlin.targetgen.contract.interface_emit import parse_interface_mlir

gm = _bk.get_backend("gemmini").gemmini_codegen_mlir


def _conv_capsules() -> list[str]:
    root = merlin_dir() / "contract" / "capsules" / "layers"
    return sorted(
        p.name for p in root.iterdir() if "conv" in p.name.lower() and (p / "capsule.interface.mlir").is_file()
    )


def _cb(name: str) -> dict:
    path = merlin_dir() / "contract" / "capsules" / "layers" / name / "capsule.interface.mlir"
    return parse_interface_mlir(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def loop_conv():
    import importlib

    _bk.get_backend("gemmini")
    return importlib.import_module("merlin._oot_backends.gemmini.gemmini_loop_conv")


def test_the_contract_derives_from_the_targets_own_sources(loop_conv):
    """No caller should have to build one; every production caller passed None and got im2col."""
    contract = loop_conv.derive_native_conv_contract()
    assert contract.dim > 0
    # The opcode is the target's, read from its facts -- not a constant spelled in the route.
    assert contract.custom_opcode == 0x7B
    assert set(contract.provenance) == {"header_sha256", "params_sha256", "facts_sha256", "core_hw_sha256"}


@pytest.mark.parametrize("name", _conv_capsules())
def test_requesting_the_route_does_not_change_what_is_emitted(name):
    """The safety property that makes requesting-by-default defensible: when the route refuses, the
    fallback is the path that was taken before, byte for byte."""
    suppressed = gm.emit_kernel_mlir(_cb(name), native_conv=False)[0]
    requested = gm.emit_kernel_mlir(_cb(name))[0]
    assert suppressed == requested


@pytest.mark.parametrize("name", _conv_capsules())
def test_every_capsule_records_why_the_route_was_not_taken(name):
    receipt: dict = {}
    gm.emit_kernel_mlir(_cb(name), native_conv_selection_receipt=receipt)
    assert receipt, "a route that is requested must say what happened"
    assert receipt["reason"], receipt
    assert receipt["default_enabled"] is True


def test_no_capsule_is_turned_away_for_packing_or_ordering():
    """The regression this exists for. Before the fix all 16 reported this, and it was never true."""
    for name in _conv_capsules():
        receipt: dict = {}
        gm.emit_kernel_mlir(_cb(name), native_conv_selection_receipt=receipt)
        assert "resident pack" not in receipt["reason"], (name, receipt["reason"])


def test_the_refusals_name_convolution_properties(loop_conv):
    """What a truthful census looks like: the narrow readout and the geometry clauses, which is where
    the work actually is -- not a structural check standing in front of them.

    The geometry half used to arrive as ONE message, `layout/stride/dilation/padding`, for a
    conjunction that refused every stride but 1, every padding but 0 and every dilation but 1 at
    once. That conjunction contained no convolution ResNet-50 performs, and reading its message told
    you nothing about which of its four terms your convolution had tripped. The emitter now expresses
    strided, padded and dilated convolutions, so what remains is per-property and says which.
    """
    reasons = set()
    for name in _conv_capsules():
        receipt: dict = {}
        gm.emit_kernel_mlir(_cb(name), native_conv_selection_receipt=receipt)
        reasons.add(receipt["reason"])
    joined = " | ".join(reasons)
    assert "narrow saturating output" in joined
    assert "layout/stride/dilation/padding" not in joined, "the four-way conjunction is gone"
    assert "loop_conv_requires_uniform_padding" in joined, "a per-edge padding names itself"


def test_suppressing_the_request_records_nothing(loop_conv):
    receipt: dict = {}
    gm.emit_kernel_mlir(_cb(_conv_capsules()[0]), native_conv=False, native_conv_selection_receipt=receipt)
    assert receipt == {}, "a route that was not requested has nothing to report"


def test_a_checkout_that_cannot_derive_the_contract_says_so(monkeypatch, loop_conv):
    """`facts.json` is a REGENERATED, gitignored artifact and is routinely absent in a fresh worktree.
    That must read as 'this checkout cannot derive the contract', never as 'not a native convolution'."""

    def _absent():
        raise loop_conv.UnsupportedNativeConv("UNKNOWN native-conv contract: no rtl facts at /nowhere")

    monkeypatch.setattr(loop_conv, "derive_native_conv_contract", _absent)
    receipt: dict = {}
    name = _conv_capsules()[0]
    text = gm.emit_kernel_mlir(_cb(name), native_conv_selection_receipt=receipt)[0]
    assert receipt["selection"] == "contract_underivable"
    assert "UNKNOWN" in receipt["reason"]
    assert text == gm.emit_kernel_mlir(_cb(name), native_conv=False)[0], "still the unchanged fallback"


def test_a_pack_ordered_after_its_consumer_is_named_as_ordering(loop_conv):
    """The three terms were one conjunction with one message; each must now name itself."""
    import importlib

    nc = importlib.import_module("merlin._oot_backends.gemmini.gemmini_native_conv_codegen")
    contract = loop_conv.derive_native_conv_contract()
    cb = deepcopy(_cb("GC0_conv2d_i8"))
    commands = cb["commands"]
    pack = next(c for c in commands if c["opcode"] == "RES_PACK")
    conv = next(c for c in commands if c["opcode"] == "CONV2D")
    commands.remove(pack)
    commands.insert(commands.index(conv) + 1, pack)
    with pytest.raises(loop_conv.UnsupportedNativeConv, match="ordered after"):
        nc.emit_selected_native_conv(cb, contract=contract)


def test_a_pack_that_does_not_feed_the_convolution_is_named_as_that(loop_conv):
    import importlib

    nc = importlib.import_module("merlin._oot_backends.gemmini.gemmini_native_conv_codegen")
    contract = loop_conv.derive_native_conv_contract()
    cb = deepcopy(_cb("GC0_conv2d_i8"))
    next(c for c in cb["commands"] if c["opcode"] == "RES_PACK")["operands"]["dst"] = "somewhere_else"
    with pytest.raises(loop_conv.UnsupportedNativeConv, match="does not produce the convolution's weight"):
        nc.emit_selected_native_conv(cb, contract=contract)


def test_an_unknown_pack_layout_is_still_refused_and_quotes_itself(loop_conv):
    """Accepting the conv spelling must not have opened the attribute to anything at all."""
    import importlib

    nc = importlib.import_module("merlin._oot_backends.gemmini.gemmini_native_conv_codegen")
    contract = loop_conv.derive_native_conv_contract()
    cb = deepcopy(_cb("GC0_conv2d_i8"))
    next(c for c in cb["commands"] if c["opcode"] == "RES_PACK")["attributes"]["layout"] = "something_else"
    with pytest.raises(loop_conv.UnsupportedNativeConv, match="something_else"):
        nc.emit_selected_native_conv(cb, contract=contract)


def _resnet50_style_conv_buffer(bias: bool) -> dict:
    """One of the recorded model's own convolutions, in the form the device's store path requires."""
    k, ci, co, h, w, p = 3, 64, 64, 56, 56, 1
    oh = ow = (h + 2 * p - k) + 1
    tensors = {
        "X": {"shape": [1, h, w, ci], "dtype": "i8"},
        "W": {"shape": [k * k * ci, co], "dtype": "i8"},
        "Y": {"shape": [oh * ow, co], "dtype": "i8"},
    }
    attributes = {
        "kernel": [k, k, ci, co],
        "stride": [1, 1],
        "padding": [p] * 4,
        "dilation": [1, 1],
        "layout": "nhwc",
        "epilogue": ["bias_add", "acc_scale", "relu"] if bias else [],
        "output_dtype": "i8",
    }
    operands = {"ifm": "X", "weight": "W", "dst": "Y"}
    if bias:
        attributes["acc_scale"] = 0.03125
        operands["bias"] = "B"
        tensors["B"] = {"shape": [co], "dtype": "i32"}
    return {"tensors": tensors, "commands": [{"opcode": "CONV2D", "attributes": attributes, "operands": operands}]}


def test_a_tiled_convolution_lowers_to_one_kernel(loop_conv):
    """The whole convolution, tiles and all, as ONE accelerator kernel: two entry configs and seven
    descriptor instructions per tile. A tiled emission that fell back would lower to nothing here.
    """
    receipt: dict = {}
    text, arguments = gm.emit_kernel_mlir(
        _resnet50_style_conv_buffer(bias=False), native_conv_selection_receipt=receipt
    )
    assert receipt["selection"] == "selected_explicit_opt_in"
    assert receipt["tiles"] > 1
    assert text.count(".insn r ") == 2 + 7 * receipt["tiles"]
    assert arguments == ["W", "X", "Y"]


def test_a_bias_becomes_a_fourth_kernel_pointer(loop_conv):
    """And the convolution WITHOUT one keeps the three-argument order it already had, so adding bias
    support does not renumber anybody else's kernel arguments."""
    receipt: dict = {}
    text, arguments = gm.emit_kernel_mlir(_resnet50_style_conv_buffer(bias=True), native_conv_selection_receipt=receipt)
    assert arguments == ["W", "X", "Y", "B"]
    assert text.count("llvm.ptrtoint") == 4


def test_each_operand_slice_is_materialized_once(loop_conv):
    """A few hundred tiles address a few hundred slices; recomputing the same address per tile would
    inflate the kernel for nothing. One `llvm.add` per distinct slice, and no slice unaccounted for."""
    receipt: dict = {}
    text, _ = gm.emit_kernel_mlir(_resnet50_style_conv_buffer(bias=False), native_conv_selection_receipt=receipt)
    slices = set()
    for parameters in receipt["tile_parameters"]:
        for field in ("weights", "output", "input", "bias"):
            value = parameters[field]
            if isinstance(value, str) and "+" in value:
                slices.add(value)
    assert text.count("llvm.add") == len(slices)
