"""A fabricated or undeclared engine's accelerator counters must be REFUSED, not annotated.

The defect this guards cost a session hours: an optimization loop read accelerator occupancy
counters off a functional ISS that returns `rand()` for them, and the values looked like small
measurements. Refusal has to be the default for anything not positively declared real.
"""
from __future__ import annotations

import pytest

from merlin.perf.counter_trust import (FABRICATED, REAL, UNKNOWN, declared_engines, is_trusted,
                                       require_trusted, values_or_refusal, verdict_for)

_VALUES = {"MAIN_LD_CYCLES": 3801783, "RDMA_BYTES_REC": 7}


def test_the_functional_iss_is_declared_fabricated_with_evidence():
    v = verdict_for("spike")
    assert v.verdict == FABRICATED
    assert v.evidence and v.evidence_source, "a verdict must cite the engine's own sources"
    assert not v.trusted


def test_the_rtl_engines_are_declared_real():
    for engine in ("firesim", "verilator", "gsim"):
        v = verdict_for(engine)
        assert v.verdict == REAL, engine
        assert v.trusted and v.refusal() is None, engine


def test_an_undeclared_engine_is_unknown_and_refused():
    v = verdict_for("some_new_simulator")
    assert v.verdict == UNKNOWN
    assert not v.trusted
    assert "UNKNOWN" in (v.refusal() or "")


def test_an_empty_or_missing_engine_name_is_refused():
    for engine in (None, "", "   "):
        assert not is_trusted(engine)
        assert verdict_for(engine).refusal()


def test_values_are_withheld_and_a_reason_given_for_a_fabricated_engine():
    values, why = values_or_refusal("spike", _VALUES)
    assert values is None
    assert why and "fabricated" in why


def test_values_pass_through_for_a_real_engine():
    values, why = values_or_refusal("firesim", _VALUES)
    assert why is None
    assert values == _VALUES
    assert values is not _VALUES, "must not alias the caller's mapping"


def test_require_trusted_raises_for_anything_not_real():
    for engine in ("spike", "some_new_simulator", None):
        with pytest.raises(ValueError):
            require_trusted(engine)
    assert require_trusted("gsim").trusted


def test_engine_names_are_matched_case_insensitively_and_trimmed():
    assert is_trusted(" FireSim ")
    assert verdict_for("SPIKE").verdict == FABRICATED


def test_the_table_declares_the_engines_this_tree_actually_runs():
    declared = set(declared_engines())
    assert {"spike", "firesim", "verilator", "gsim"} <= declared


def test_run_on_oracle_refuses_readings_from_a_fabricating_engine():
    """The wiring, not just the policy: an inert module is the trap this repo keeps re-finding.

    `run_on_oracle` used to stamp `status: "measured"` on raw counter readings whatever engine
    produced them. The eta path beside it already refused a non-RTL oracle; this path did not.
    """
    import inspect

    from merlin.targetgen.contract import compile as compile_mod

    body = inspect.getsource(compile_mod.run_on_oracle)
    assert "counter_trust" in body, "run_on_oracle must consult the trust table"
    # the refusal must be recorded, not merely omitted
    assert "_trust.refusal()" in body
    assert '"status": "unknown"' in body
    # and it must be keyed on the engine that ran the program
    assert "verdict_for(simulator)" in body
