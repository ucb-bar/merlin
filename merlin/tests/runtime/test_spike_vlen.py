"""Pinning the simulated AND compiled vector length.

spike takes VLEN from its ISA string, and with no `zvl` extension it uses the V minimum of 128. That is
a silent trap for a wider board: the image gets validated at a vector length the hardware does not have,
and a fixed-width schedule compiled for one VLEN maps to a different LMUL on the other -- the measured
K1 case, where `-march=rv64gcv` assumed 128 against a 256-bit unit and doubled every vector group.

So the build flag and the simulator string have to state the SAME number, which is what these tests pin.
Measured end to end on spectformer int8: VLEN 128 and 256 both build, both pass the gate, and produce
bit-identical output (spike is functional, so cycles are unchanged -- it validates correctness at the
board's vector length, not its cost).
"""
from __future__ import annotations

import pytest

from merlin.runtime.backends import zephyr_model as zm


def test_default_isa_is_unchanged():
    """No vlen argument must leave every existing run byte-identical."""
    assert zm.spike_isa(None) == zm.DEFAULT_SPIKE_ISA
    assert zm.march_with_vlen(["-march=rv64gcv", "-O2"], None) == ["-march=rv64gcv", "-O2"]


@pytest.mark.parametrize("vlen", [128, 256, 512, 1024])
def test_isa_and_march_state_the_same_vlen(vlen):
    """The simulator string and the compile flag must agree, or the test proves nothing."""
    isa = zm.spike_isa(vlen)
    march = zm.march_with_vlen(["-march=rv64gcv"], vlen)[0]
    assert isa.endswith(f"_zvl{vlen}b")
    assert march.endswith(f"_zvl{vlen}b")


def test_an_already_pinned_march_is_not_double_pinned():
    """A package that already states its VLEN must win; appending a second zvl would be invalid."""
    assert zm.march_with_vlen(["-march=rv64gcv_zvl512b"], 256) == ["-march=rv64gcv_zvl512b"]


def test_non_march_flags_are_untouched():
    flags = ["-march=rv64gcv", "-fno-vectorize", "-mabi=lp64d", "-O2"]
    out = zm.march_with_vlen(flags, 256)
    assert out[1:] == flags[1:]


@pytest.mark.parametrize("bad", [96, 100, 0, 64])
def test_a_vlen_that_is_not_a_power_of_two_at_least_128_is_rejected(bad):
    """V's minimum is 128 and groups are powers of two; anything else would silently mis-lower."""
    with pytest.raises(zm.ZephyrModelError):
        zm.spike_isa(bad)


def test_build_and_run_records_the_vlen_it_used():
    """A result that does not say which VLEN it exercised can be misread as another one."""
    import inspect

    src = inspect.getsource(zm.build_and_run)
    assert '"vlen": vlen' in src, "build_and_run must record the vlen in its result"
    assert "vlen=vlen" in src, "build_and_run must pass vlen to BOTH the build and the run"
