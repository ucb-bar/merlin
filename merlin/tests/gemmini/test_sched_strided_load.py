"""The strided input load, and the two fields that are only ever correct together.

The sequencer's input loader stages one transfer per input pixel per channel block. A convolution of
stride 2 reads every other pixel in each dimension, so three of every four staged pixels are moved into
the scratchpad and never read. ``LoopConvLdInput``'s ``downsample`` bit removes exactly that waste, and
the recipe pinned it to 0 while the checker refused it outright as "not modelled".

Measured before this landed, on completed GSIM runs with the outputs compared element for element
against the vendor library: 1,126,116 -> 502,703 cycles on the 56x56 256->512 layer, 670,415 -> 417,677
on 28x28 512->1024, 738,313 -> 592,192 on 14x14 1024->2048. One bit, 1,022,272 cycles, bit-exact.

What makes it a checker's business rather than a flag to flip: the execute stage reads the staged window
back with ``irows >> downsample``, so with the bit set the loader has ALREADY applied the stride and
``config_ex``'s ``A_stride`` must come back to one. Set either alone and the kernel computes a different
convolution with nothing to say so. These tests hold the pair together, hold the eligibility predicate
to the library's own, and hold the capacity arithmetic to the window the device actually stages.
"""

from __future__ import annotations

import pytest

from merlin.runtime.backends import base

pytestmark = pytest.mark.target("gemmini")

#: The three ResNet-50 layers entitled to the mode, and one that is not (a 3x3 stride-2).
ENTITLED = ((56, 256, 512), (28, 512, 1024), (14, 1024, 2048))


@pytest.fixture(scope="module")
def sched():
    import importlib

    base.get_backend("gemmini")  # registers the out-of-tree package
    return importlib.import_module("merlin._oot_backends.gemmini.gemmini_sched")


@pytest.fixture(scope="module")
def iset():
    return base.get_backend("gemmini").sched_instruction_set()


def _conv(sched, iset, in_dim, ci, co, kernel, stride, padding, **kw):
    from merlin.sched.ir import TensorArg

    out_dim = (in_dim + 2 * padding - kernel) // stride + 1
    operands = {
        "input": TensorArg("input", (1, in_dim, in_dim, ci), "i8", "read"),
        "weights": TensorArg("weights", (kernel, kernel, ci, co), "i8", "read"),
        "bias": TensorArg("bias", (co,), "i32", "read"),
        "output": TensorArg("output", (1, out_dim, out_dim, co), "i8", "write"),
    }
    return sched.conv_reference(
        name="probe",
        batch=1,
        in_dim=in_dim,
        in_channels=ci,
        out_channels=co,
        kernel=kernel,
        stride=stride,
        padding=padding,
        operands=operands,
        relu=False,
        scale=0.01,
        facts=iset.facts,
        **kw,
    )


def _descriptors(kernel):
    from merlin.sched.ir import concretize, instances

    return [concretize(c, e) for c, e in instances(kernel) if c.instr == "loop_conv_ws"]


@pytest.mark.parametrize("in_dim,ci,co", ENTITLED)
def test_a_layer_the_library_admits_takes_the_strided_load(sched, iset, in_dim, ci, co):
    k = _conv(sched, iset, in_dim, ci, co, 1, 2, 0)
    assert {d["downsample"] for d in _descriptors(k)} == {1}
    assert dict(k.body[0].args)["A_stride"].value == 1, "the loader applied the stride; execute must not"
    from merlin.sched.check.static import check_kernel

    assert check_kernel(k, iset) == []


@pytest.mark.parametrize(
    "kernel,stride,padding,why",
    [
        (3, 2, 1, "a 3x3 window reads more than one input pixel per output pixel"),
        (1, 1, 0, "a unit-stride layer has nothing to skip"),
        (1, 2, 1, "padding shifts which input pixel each output reads"),
    ],
)
def test_a_layer_the_library_does_not_admit_keeps_the_full_load(sched, iset, kernel, stride, padding, why):
    k = _conv(sched, iset, 28, 128, 128, kernel, stride, padding)
    assert {d["downsample"] for d in _descriptors(k)} == {0}, why
    assert dict(k.body[0].args)["A_stride"].value == stride
    from merlin.sched.check.static import check_kernel

    assert check_kernel(k, iset) == []


def test_an_odd_extent_is_not_admitted(sched, iset):
    """Halving a window is the same computation only when the window halves evenly."""
    k = _conv(sched, iset, 7, 128, 128, 1, 2, 0)
    assert {d["downsample"] for d in _descriptors(k)} == {0}


def test_a_pooled_readout_is_not_admitted(sched, iset):
    k = _conv(sched, iset, 56, 64, 64, 1, 2, 0, pool_size=3, pool_stride=2, pool_padding=1)
    assert {d["downsample"] for d in _descriptors(k)} == {0}


# --- the mutations: each must fail a check that did not exist before ------------------------------


@pytest.fixture(scope="module")
def instr(iset):
    return iset.instr("loop_conv_ws")


def _one(sched, iset, in_dim, ci, co, kernel, stride, padding):
    """A real descriptor the recipe emitted, so a mutation perturbs the genuine article."""
    return dict(_descriptors(_conv(sched, iset, in_dim, ci, co, kernel, stride, padding))[0])


def _state(a_stride: int) -> dict:
    return {
        "config_st": {"stride": 64, "acc_act": 0, "acc_scale": 1.0},
        "config_ex": {"A_stride": a_stride, "A_transpose": 0, "B_transpose": 0},
    }


def test_a_descriptor_the_recipe_emitted_passes_its_own_check(sched, iset, instr):
    """The control. Without it every mutation below could be failing for an unrelated reason."""
    assert instr.check(_one(sched, iset, 56, 256, 512, 1, 2, 0), _state(1)) == []


def test_the_bit_without_its_paired_stride_is_refused(sched, iset, instr):
    """The silent-wrongness this check exists for: the loader strides, and the execute unit strides
    again. Both fields are individually plausible; only together are they a convolution."""
    errs = instr.check(_one(sched, iset, 56, 256, 512, 1, 2, 0), _state(2))
    assert any("A_stride must be 1" in e for e in errs), errs


def test_the_bit_on_a_layer_the_predicate_refuses_is_refused(sched, iset, instr):
    v = _one(sched, iset, 28, 128, 128, 3, 2, 1) | {"downsample": 1}
    errs = instr.check(v, _state(1))
    assert any("predicate does not admit" in e for e in errs), errs


def test_a_checkout_that_states_no_header_refuses_the_mode(sched, iset, instr, monkeypatch):
    """`facts` is derived and `facts.json` is a regenerated, gitignored artifact routinely absent in a
    fresh worktree. That must read as 'this checkout cannot state the predicate', never as 'permitted'."""
    # Built while the header is readable, checked after it is gone -- which is the real situation: a
    # descriptor travels, and the checkout that validates it need not be the one that emitted it.
    v = _one(sched, iset, 56, 256, 512, 1, 2, 0)
    assert v["downsample"] == 1, "the control: this layer takes the mode while the header is readable"
    monkeypatch.setitem(iset.facts, "conv_header_text", "")
    errs = instr.check(v, _state(1))
    assert any("states no convolution header" in e for e in errs), errs


def test_the_footprint_prices_the_window_the_device_stages(sched):
    """Capacity is the mirror of the overrun check: pricing the unstrided window would refuse a tile
    the device holds, which is the same mistake pointed the other way."""
    common = dict(
        acc=False,
        stride=2,
        batches=1,
        porows=8,
        pocols=8,
        pochs=16,
        krows=1,
        kcols=1,
        kchs=16,
        pool_size=1,
        pool_stride=1,
        dim=16,
    )
    assert sched.conv_working_rows(**common, downsample=1) < sched.conv_working_rows(**common, downsample=0)


def test_the_traffic_model_sees_the_saving(sched, iset):
    """A search that priced the unstrided window would be blind to the very saving the mode buys."""
    tile = {"batches": 1, "porows": 8, "pocols": 8, "pochs": 64, "kchs": 64}
    shape = dict(
        batch=1, out_dim=28, out_channels=512, kernel=1, stride=2, in_channels=256, tile=tile, facts=iset.facts
    )
    full = sched.conv_tile_traffic(**shape, downsample=0)
    strided = sched.conv_tile_traffic(**shape, downsample=1)
    assert strided["input"] < full["input"]
    for term in ("weights", "bias", "output", "descriptors"):
        assert strided[term] == full[term], f"{term} is not the input and must not move"
