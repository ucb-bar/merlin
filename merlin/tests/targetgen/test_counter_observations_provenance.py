"""A counter reading is an occupancy measurement only when an RTL oracle produced it.

The runner brackets a kernel with the target's own combination counters and the readings reach the
console. Attaching them to the tier record is what finally gives a host-dispatched target an activity
source -- but only the tiers entitled to make a hardware claim may carry one. A functional model runs
the program correctly without modelling the engines, so its counter CSRs describe nothing: measured on
one, a 52-cycle window returned per-engine busy totals in the THOUSANDS. Those are not imprecise
numbers, they are numbers about a different machine, and a composition operator derived from them would
be a fabrication carrying a measurement's provenance.

These tests also pin the shape of the guard, because the first version of it was wrong in the way that
is hardest to see: an unimported name inside a broad ``except`` made EVERY call return "no capability",
so the negative cases passed for the wrong reason and the positive case silently never fired.
"""
from __future__ import annotations

from merlin.targetgen.contract import compile as C

_RTL = {"kind": "t-rtl", "derived_from_rtl": True}
_MODEL = {"kind": "t-model", "derived_from_rtl": False}

#: A console recorded verbatim from a real bracketed RTL run (306 cycles).
_CONSOLE = ("MERLIN_HWCOUNTER MAIN_EX_CYCLES 70\n"
            "MERLIN_HWCOUNTER MAIN_LD_EX_CYCLES 28\n"
            "MERLIN_HWCOUNTER MAIN_LD_ST_EX_CYCLES 0\n"
            "MERLIN_HWCOUNTER MAIN_ST_EX_CYCLES 0\n"
            "MERLIN_HWCOUNTER MAIN_LD_CYCLES 39\n"
            "MERLIN_HWCOUNTER MAIN_LD_ST_CYCLES 0\n"
            "MERLIN_HWCOUNTER MAIN_ST_CYCLES 44\n"
            "METRIC cycles 306\nDONE\n")


def _target_with_counters() -> str | None:
    """A target whose shipped header actually derives a counter set, or ``None`` to skip.

    Resolved rather than named: the point of the guard is target-independent, and a test that hardcoded
    a target would pass on a checkout where that target's header is absent.
    """
    from merlin.common.paths import merlin_dir
    from merlin.perf import hw_counters as H
    names = sorted(p.name for p in (merlin_dir() / "targets").iterdir() if p.is_dir())
    for name in names:
        try:
            if H.counters_for_target(name).get("status") == "derived":
                return name
        except Exception:                                      # noqa: BLE001
            continue
    return None


class TestOracleProvenanceGate:
    def test_a_functional_model_carries_no_activity_block(self):
        assert C._counter_observations(_CONSOLE, target="anything", simulator="model",
                                       cycles=52, oracle=_MODEL) == (None, None)

    def test_an_unstated_oracle_fails_closed(self):
        assert C._counter_observations(_CONSOLE, target="anything", simulator="x",
                                       cycles=1, oracle=None) == (None, None)

    def test_an_unbracketed_rtl_run_is_byte_identical_to_before(self):
        assert C._counter_observations("METRIC cycles 5\nDONE\n", target="anything",
                                       simulator="rtl", cycles=5, oracle=_RTL) == (None, None)

    def test_an_rtl_run_with_readings_does_carry_one(self):
        """The positive case. Without it the guard above can pass for the wrong reason."""
        target = _target_with_counters()
        if target is None:
            import pytest
            pytest.skip("no target on this checkout derives a counter set from its shipped header")
        obs, cap = C._counter_observations(_CONSOLE, target=target, simulator="rtl",
                                           cycles=306, oracle=_RTL)
        assert obs, "an RTL oracle with counter readings must produce an activity block"
        quantities = {e["quantity"] for e in obs}
        assert any(q.startswith("busy_cycles.") for q in quantities)
        assert cap is not None and cap["partitioned"] is False, (
            "the block must declare itself non-partitioned or its overlap reading is refused")
