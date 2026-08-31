"""The joint-occupancy bracket in the graded harness is OPT-IN, and off it changes nothing.

This harness is on the L0/L1/L3 grading path. A change that altered every run would make a round's
verdicts incomparable with the rounds before it, so the default must be byte-identical to what it was.
"""
from __future__ import annotations

import os

import pytest

from merlin.targetgen.contract import interface_emit as IE


def _render(capsule="isa/A2_single_tile_matmul"):
    from merlin.common.paths import repo_root
    from merlin.runtime.backends import base as bk

    p = repo_root() / "merlin" / "contract" / "capsules" / capsule / "capsule.interface.mlir"
    if not p.is_file():
        pytest.skip(f"{capsule} is not present in this checkout")
    cb = IE.parse_interface_mlir(p.read_text(encoding="utf-8"))
    return bk.get_backend("gemmini").render_harness(cb, target="gemmini")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("MERLIN_HW_COUNTERS", raising=False)


def test_the_default_harness_carries_no_counter_code(monkeypatch):
    got = _render()
    assert "counter_configure" not in got
    assert "gemmini_counter.h" not in got
    # The window it already measured is untouched.
    assert "METRIC cycles" in got and "read_cycles()" in got


def test_asking_for_counters_adds_the_bracket_around_the_same_window(monkeypatch):
    monkeypatch.setenv("MERLIN_HW_COUNTERS", "1")
    got = _render()
    assert "counter_reset();" in got and "gemmini_counter.h" in got
    # Configured BEFORE the cycle window opens and read AFTER it closes, so realised overlap and the
    # cycle count describe one run rather than two.
    assert got.index("counter_configure") < got.index("read_cycles()")
    assert got.index("counter_read(") > got.rindex("gemmini_fence();")
    assert "METRIC cycles" in got


def test_every_derived_counter_is_configured_and_read_back(monkeypatch):
    monkeypatch.setenv("MERLIN_HW_COUNTERS", "1")
    got = _render()
    assert got.count("counter_configure(") == got.count("counter_read(")
    assert got.count("counter_configure(") >= 3, "a target with fewer than three combinations cannot "\
                                                 "show three-way overlap"


@pytest.mark.parametrize("value,on", [("1", True), ("true", True), ("on", True), ("yes", True),
                                      ("0", False), ("", False), ("no", False)])
def test_the_switch_reads_only_affirmative_values(monkeypatch, value, on):
    monkeypatch.setenv("MERLIN_HW_COUNTERS", value)
    assert ("counter_configure" in _render()) is on


def test_a_counter_failure_never_breaks_the_graded_harness(monkeypatch):
    # The bracket is a diagnostic EXTRA. If deriving it fails, the harness must still render and still
    # grade -- a capsule must not fail on the absence of an optional measurement.
    monkeypatch.setenv("MERLIN_HW_COUNTERS", "1")
    import merlin.perf.hw_counters as hc

    def _boom(*a, **k):
        raise RuntimeError("counter header unavailable")

    monkeypatch.setattr(hc, "counters_for_target", _boom)
    got = _render()
    assert "int main()" in got and "METRIC cycles" in got
    assert "counter_configure" not in got
