"""The mean-pool ResNet-50 ends on, written as what it already is.

This target's headers offer no average pool: the only pooling ``config_st`` and the convolution
sequencer carry is a MAX pool. So a whole-plane mean is expressed as a contraction against a vector of
ones, with the division folded into ``config_st``'s accumulator scale -- the sum stays exact in the
accumulator's integer domain and the only rounding is the readout's, which is where a requantizing
pool's rounding belongs.

The refusals matter more than the happy path here. A sliding window mean is NOT this contraction -- it
is a depthwise convolution against ones weights, a mode the device expresses (``loop_conv_ws`` carries
``dw``) but the checker does not model. Reducing over the whole plane when a window was asked for
produces a silently wrong number rather than an error, so it is refused.
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
def sched():
    base.get_backend("gemmini")  # registers the out-of-tree package
    return importlib.import_module("merlin._oot_backends.gemmini.gemmini_sched")


def _ops(rows: int, channels: int) -> dict:
    return {
        "a": TensorArg("ones", (1, rows), "i8", "read"),
        "b": TensorArg("x", (rows, channels), "i8", "read"),
        "c": TensorArg("y", (1, channels), "i8", "write"),
    }


def test_the_resnet50_global_mean_pool_passes_the_static_check(sched, iset):
    """7x7 over 2048 channels -- the one mean-pool in the model."""
    k = sched.window_mean_reference(name="gap", rows=49, channels=2048, operands=_ops(49, 2048), facts=iset.facts)
    assert check_kernel(k, iset) == []
    attrs = dict(k.attrs)
    assert attrs["recipe"] == "window_mean_ones_contraction_v1"
    assert attrs["reduction"] == "mean over 49"
    assert attrs["numerics"] == "matches the vendor library on gsim, ResNet-50 group model", (
        "the marker states what was measured"
    )


def test_a_sliding_window_is_refused_rather_than_reduced_over_the_whole_plane(sched, iset):
    """The failure this guard exists for is silent: reducing over 49 when 9 was asked for still
    produces a number, of the right shape, that is simply wrong."""
    with pytest.raises(IsaError, match="depthwise"):
        sched.window_mean_reference(name="g", rows=49, channels=64, operands=_ops(49, 64), facts=iset.facts, window=3)


def test_a_strided_window_is_refused(sched, iset):
    with pytest.raises(IsaError, match="whole-plane"):
        sched.window_mean_reference(name="g", rows=49, channels=64, operands=_ops(49, 64), facts=iset.facts, stride=2)


def test_stating_the_whole_plane_explicitly_is_accepted(sched, iset):
    """A caller that passes the window it means, and means the whole plane, is not fighting the guard."""
    k = sched.window_mean_reference(
        name="g", rows=49, channels=64, operands=_ops(49, 64), facts=iset.facts, window=49, stride=49
    )
    assert check_kernel(k, iset) == []


def test_an_empty_plane_is_refused(sched, iset):
    with pytest.raises(IsaError, match="empty plane"):
        sched.window_mean_reference(name="g", rows=0, channels=64, operands=_ops(1, 64), facts=iset.facts)


def test_the_division_rides_the_accumulator_scale_not_the_host(sched, iset):
    """The whole point: no host arithmetic, and no separate divide instruction -- `1/rows` is the
    readout's scale, so the integer sum reaching it is exact."""
    k = sched.window_mean_reference(name="gap", rows=49, channels=64, operands=_ops(49, 64), facts=iset.facts)
    st = [c for c in k.body if getattr(c, "instr", None) == "config_st"]
    assert len(st) == 1
    assert dict(st[0].args)["acc_scale"] == pytest.approx(1.0 / 49, rel=1e-6)
