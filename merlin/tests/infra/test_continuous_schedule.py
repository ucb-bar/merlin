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


def test_formal_completion_requires_numeric_tooling_and_official_hidden_grade():
    loop = _mod("run_baseline_qa_loop")
    assert loop._authoring_completion(True, True) is True
    assert loop._authoring_completion(True, False) is False
    assert loop._formal_completion(True, True, True) is True
    assert loop._formal_completion(True, True, False) is False
    assert loop._formal_completion(True, False, True) is False
    assert loop._formal_completion(False, True, True) is False


def test_continuous_never_reports_budget_at_the_l3_barrier():
    """With the effectively-unbounded cap continuous passes in, the decision can only be done/iterate —
    an L3 that has not passed keeps being worked rather than stopping on arithmetic."""
    loop = _mod("run_baseline_qa_loop")
    for rnd in (0, 5, 50, 500):
        assert loop._l3_barrier_decision(False, rnd=rnd, max_rounds=rnd + 1_000_000) == "iterate"


def test_a_declared_wall_budget_still_ends_the_l3_fix_loop():
    """The one thing that DOES terminate the L3 loop in continuous mode: the budget the operator declared.

    Rounds are not a terminator, but `--max-wall-s` is -- otherwise a "12 h" run would iterate fix rounds
    past its own budget forever. The driver expresses that by collapsing the cap to the current round once
    the budget is spent; the decision function's contract is what makes that stop honest.
    """
    loop = _mod("run_baseline_qa_loop")
    assert loop._l3_barrier_decision(False, rnd=7, max_rounds=7) == "budget"
    src = (HARNESS / "run_baseline_qa_loop.py").read_text(encoding="utf-8")
    assert "_cap, _budget_reason = rnd, \"max_wall_s\"" in src, (
        "the L3 fix loop no longer honours --max-wall-s; a continuous run can outlive its declared budget")


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


# --- the whole-model capstone must be bounded inside a round ------------------------------------
# MEASURED (merlincirct_defcal1, 2026-08-29): the agent round finished in 40 min, the capstone then
# ran 5h30m -- past the round's own 4h --round-timeout -- and the round never graded. --qa-timeout is
# a PER-STEP subprocess cap, so a grade that makes many such calls is not bounded by it.

def test_the_model_budget_defaults_to_the_qa_timeout():
    """Derived, not invented: the capstone may cost at most what the operator already said one grading
    step may. An explicit value wins; 0 means no ceiling (an operator certification run wants that)."""
    loop = _mod("run_baseline_qa_loop")
    import inspect

    src = inspect.getsource(loop.main)
    assert '"--model-budget-s"' in src
    assert "a.qa_timeout if a.model_budget_s is None else a.model_budget_s" in src
    assert 'os.environ["MERLIN_MODEL_BUDGET_S"]' in src
    assert 'os.environ.pop("MERLIN_MODEL_BUDGET_S", None)' in src, "0 must CLEAR the ceiling"


def test_the_batch_launcher_forwards_the_budget():
    """A per-arm terminator the batch does not forward is a batch setting that is a lie."""
    launcher = _mod("launch_ab_batch")
    import inspect

    src = inspect.getsource(launcher._arm_cmd)
    assert '"--model-budget-s"' in src
    assert 'getattr(a, "model_budget_s", None) is not None' in src, "unset must stay byte-identical"
