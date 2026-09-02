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
    monkeypatch.delenv("MERLIN_HW_COUNTER_UNIT", raising=False)
    monkeypatch.delenv("MERLIN_CACHE_STATE", raising=False)


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
    assert "MERLIN_COUNTER_SCHEMA" in got
    # Configured BEFORE the cycle window opens and read AFTER it closes, so realised overlap and the
    # cycle count describe one run rather than two.
    assert got.index("counter_configure") < got.index("read_cycles()")
    assert got.index("counter_read(") > got.rindex("gemmini_fence();")
    assert "METRIC cycles" in got


def test_every_derived_counter_is_configured_and_read_back(monkeypatch):
    monkeypatch.setenv("MERLIN_HW_COUNTERS", "1")
    got = _render()
    selected = got.count("counter_read(")
    assert selected >= 3, "a target with fewer than three combinations cannot show three-way overlap"
    assert got.count("counter_configure(") >= selected
    assert "padding: disabled event" in got


@pytest.mark.parametrize("value,on", [("1", True), ("true", True), ("on", True), ("yes", True),
                                      ("0", False), ("", False), ("no", False)])
def test_the_switch_reads_only_affirmative_values(monkeypatch, value, on):
    monkeypatch.setenv("MERLIN_HW_COUNTERS", value)
    assert ("counter_configure" in _render()) is on


def test_requested_counter_failure_refuses_the_instrumented_run(monkeypatch):
    # Once explicitly requested, the bracket is campaign evidence. Silently rendering an uninstrumented
    # harness would let the campaign report GO for a measurement it never took.
    monkeypatch.setenv("MERLIN_HW_COUNTERS", "1")
    import merlin.perf.hw_counters as hc

    def _boom(*a, **k):
        raise RuntimeError("counter header unavailable")

    monkeypatch.setattr(hc, "counters_for_target", _boom)
    with pytest.raises(Exception, match="requested counter instrumentation unavailable"):
        _render()


def test_a_counter_unit_is_selected_from_the_shipped_header(monkeypatch):
    monkeypatch.setenv("MERLIN_HW_COUNTERS", "1")
    monkeypatch.setenv("MERLIN_HW_COUNTER_UNIT", "BYTES")
    got = _render()
    configured = [line for line in got.splitlines() if "counter_configure(" in line]
    assert configured
    selected = [line for line in configured if "padding: disabled event" not in line]
    assert all("BYTES" in line for line in selected)
    assert len(selected) == got.count("counter_read(")
    assert len(configured) > len(selected)


def test_warm_condition_executes_one_unmeasured_warmup(monkeypatch):
    monkeypatch.setenv("MERLIN_CACHE_STATE", "warm")
    got = _render()
    assert got.count("gemmini_kernel(") == 3  # declaration + warmup + measured call
    warmup = got.index("warmup completed outside the measured/counter window")
    assert warmup < got.index("uint64_t c0 = read_cycles()")


def test_unknown_cache_condition_is_refused(monkeypatch):
    monkeypatch.setenv("MERLIN_CACHE_STATE", "wishful")
    with pytest.raises(Exception, match="unsupported cache-state"):
        _render()


@pytest.mark.parametrize("kind", ["movement", "whole_op"])
def test_instrumentation_covers_movement_and_whole_op_harnesses(monkeypatch, kind):
    from merlin.runtime.backends import base as bk

    monkeypatch.setenv("MERLIN_HW_COUNTERS", "1")
    monkeypatch.setenv("MERLIN_HW_COUNTER_UNIT", "BYTES")
    tensors = {"X": {"shape": [3, 5], "dtype": "i8", "role": "input"},
               "Y": {"shape": [3, 5], "dtype": "i32", "role": "output"}}
    if kind == "movement":
        commands = [{"opcode": "MOVEMENT", "operands": {"src": "X", "dst": "Y"},
                     "attributes": {"output_dtype": "i32"}}]
    else:
        tensors["K"] = {"shape": [7, 5], "dtype": "i8", "role": "input"}
        tensors["Y"]["shape"] = [3, 7]
        commands = [{"opcode": "ATTENTION_QK",
                     "operands": {"q": "X", "k": "K", "dst": "Y"},
                     "attributes": {"output_dtype": "i32", "epilogue": []}}]
    cb = {"abi_version": "0.1", "target": "gemmini",
          "tensors": tensors, "commands": commands}
    got = bk.get_backend("gemmini").render_harness(cb, target="gemmini")
    assert "counter_configure" in got and "counter_read" in got
    assert got.index("counter_configure") < got.index("uint64_t c0")
    assert got.index("counter_read") > got.index("uint64_t c1")
