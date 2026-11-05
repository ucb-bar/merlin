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


class _Built(Exception):
    """Raised by the stubbed -march composer: the build got as far as choosing its vector length."""


def _stub_until_march(monkeypatch, board_vlen):
    from types import SimpleNamespace

    from merlin.llvmlower import weight_prepack

    seen = {}
    monkeypatch.setattr(
        "merlin.runtime.boards.board",
        lambda name, **kw: SimpleNamespace(
            name=name, vlen=kw.get("vlen", board_vlen), flow="zephyr",
            zephyr_default_ram_bytes=1 << 28, zephyr_link_limit_bytes=1 << 30,
        ),
    )  # fmt: skip
    monkeypatch.setattr(zm, "build_available", lambda: True)
    monkeypatch.setattr(weight_prepack, "prepare_build_bundle", lambda model_dir, work, features: model_dir)

    def march(cflags, vlen):
        seen["vlen"] = vlen
        raise _Built

    monkeypatch.setattr(zm, "march_with_vlen", march)
    return seen


@pytest.mark.parametrize("board_vlen", [256, 512])
def test_an_image_is_compiled_for_the_boards_vlen_when_none_is_given(tmp_path, monkeypatch, board_vlen):
    """The -march of a board image states the board's width, not the V minimum.

    The board default used to be applied AFTER the -march was composed, so `merlin-compile --board X`
    built vector code for 128 while its Kconfig said X's width, and the simulator (given no width) ran
    it at 128 too: a 512-bit board's image was graded at a width the board does not have, and was not
    the image `make_delivery.py` (which passes the width explicitly) ships.
    """
    seen = _stub_until_march(monkeypatch, board_vlen)
    with pytest.raises(_Built):
        zm.build_app(tmp_path, tmp_path / "w", board="wide_board", backend="rvv")
    assert seen["vlen"] == board_vlen


def test_an_explicit_vlen_still_wins_over_the_boards(tmp_path, monkeypatch):
    seen = _stub_until_march(monkeypatch, 512)
    with pytest.raises(_Built):
        zm.build_app(tmp_path, tmp_path / "w", board="wide_board", backend="rvv", vlen=256)
    assert seen["vlen"] == 256


def test_build_and_run_simulates_at_the_width_the_build_reports(tmp_path, monkeypatch):
    """A result that does not say which VLEN it exercised can be misread as another one, and the run must
    use the width the BUILD chose (the board's when the caller gave none), or build and run disagree on
    vlenb -- which corrupts every scalable spill slot rather than merely slowing it down."""
    from types import SimpleNamespace

    seen = {}
    monkeypatch.setattr(
        zm, "build_app", lambda *a, **kw: {"elf": "x.elf", "app_dir": "app", "ram_bytes": 1 << 20, "vlen": 512}
    )
    monkeypatch.setattr(
        "merlin.runtime.boards.board", lambda name, **kw: SimpleNamespace(simulator="spike", dram_base=0x80000000)
    )

    def spike(elf, **kw):
        seen.update(kw)
        return {"prefix": None}

    monkeypatch.setattr(zm, "run_on_spike", spike)
    res = zm.build_and_run(tmp_path, tmp_path / "w", board="wide_board")
    assert seen["vlen"] == 512 and res["vlen"] == 512
