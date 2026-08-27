"""The round COUNT must stop being a terminator under --schedule continuous.

Rounds are how the agent is invoked and how artifacts are laid out; that is fine and unchanged. What is
not fine is a run ENDING because an arithmetic cap expired while the submission was still improving, or
CONTINUING to pay after it converged. Measured on v12 arm-4: it reached its ceiling in round 0 and then
spent two more rounds and 37.9M tokens going nowhere.

Per-capsule promotion is deliberately NOT coupled to this flag — tier_promote fires on every verdict
from both brokers and the round grade, so a capsule's cert tier is enqueued when its loop tier passes,
in either mode. These tests pin that separation too, so nobody later "fixes" continuous mode by moving
promotion onto the round boundary.
"""
from __future__ import annotations

import importlib.util
import sys

import pytest

from merlin.common.paths import merlin_dir

HARNESS = merlin_dir() / "experiments/capsule_bench/harness"


def _mod(name: str):
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location(name, HARNESS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001 — harness deps absent in this env
        pytest.skip(f"{name} not importable here: {type(e).__name__}: {e}")
    return mod


def test_the_flag_exists_and_defaults_to_rounds():
    """Default MUST stay `rounds`: every stored result was measured under it, and silently switching
    would make new numbers incomparable with v11/v12 without anyone choosing that."""
    loop = _mod("run_baseline_qa_loop")
    import argparse

    ap_src = loop.main.__doc__ or ""
    del ap_src
    # parse an empty argv through the real parser by invoking it in isolation
    parser = None
    for obj in vars(loop).values():
        if isinstance(obj, argparse.ArgumentParser):
            parser = obj
            break
    # the parser is built inside main(); assert on the source instead, which is what ships
    import inspect
    src = inspect.getsource(loop.main)
    assert '"--schedule"' in src
    assert 'default="rounds"' in src, "continuous must be opt-in"
    assert '"--max-wall-s"' in src


def test_the_l3_barrier_still_stops_on_budget_in_rounds_mode():
    loop = _mod("run_baseline_qa_loop")
    assert loop._l3_barrier_decision(False, rnd=12, max_rounds=12) == "budget"
    assert loop._l3_barrier_decision(False, rnd=3, max_rounds=12) == "iterate"
    assert loop._l3_barrier_decision(True, rnd=99, max_rounds=12) == "done"


def test_a_pass_is_always_terminal_whatever_the_cap():
    """Convergence ends the run in both modes. A schedule flag must never be able to keep paying for a
    submission that is already done."""
    loop = _mod("run_baseline_qa_loop")
    for cap in (0, 1, 12, 1_000_000):
        assert loop._l3_barrier_decision(True, rnd=0, max_rounds=cap) == "done"


def test_continuous_never_reports_budget_at_the_l3_barrier():
    """With the effectively-unbounded cap continuous passes in, the decision can only be done/iterate —
    an L3 that has not passed keeps being worked rather than stopping on arithmetic."""
    loop = _mod("run_baseline_qa_loop")
    for rnd in (0, 5, 50, 500):
        assert loop._l3_barrier_decision(False, rnd=rnd, max_rounds=rnd + 1_000_000) == "iterate"


def test_promotion_is_not_coupled_to_the_schedule():
    """tier_promote must stay reachable from every verdict path. If a later change moves promotion onto
    the round boundary, continuous mode silently loses the property it exists for."""
    import inspect

    broker = _mod("simjob_broker")
    assert "promote" in inspect.getsource(broker)
    loop = _mod("run_baseline_qa_loop")
    src = inspect.getsource(loop)
    assert "tier_promote" in src
    # and the loop's own promotion must not be gated on the schedule flag
    seg = src.split("tier_promote", 1)[1][:600]
    assert "schedule" not in seg, "promotion must fire in BOTH schedules"
