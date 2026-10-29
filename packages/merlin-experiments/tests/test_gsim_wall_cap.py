"""The operator's gSIM wall cap is policy data; every lane timeout is held to it."""

import pytest

from merlin_experiments.phase1 import gsim_wall_cap as W
from merlin_experiments.phase1 import joint_gsim_measurement as M

FIT = {"setup_s": 50.0, "cycles_per_s": 1430.0, "margin": 2.0}
WINDOW = dict(remaining_drain_s=100_000, remaining_end_s=120_000, final_window_s=1000, long_allowed=True,
              final_slots_free=True)


def test_the_message_names_the_projection_the_cap_and_the_alternative():
    assert W.message(2400, 1800) == (
        "gSIM not run: projected ~40 min > 30 min cap; use the timing model or a smaller representative case"
    )


def test_every_lane_timeout_is_held_to_the_cap():
    held, _ = M.job_plan(FIT, 2_000_000, 2_000_000, max_job_wall_s=1800, **WINDOW)
    assert held["timeout_s"] == 1800
    assert M.job_plan(FIT, 2_000_000, 2_000_000, **WINDOW)[0]["timeout_s"] > 1800  # no cap selected


def test_the_cap_is_policy_data_and_validated():
    assert W.max_job_wall_s("gsim", {"gsim": {"max_job_wall_s": 1800}}) == 1800
    assert W.max_job_wall_s("gsim_correctness", {}) is None
    assert M.settings({"gsim_measurement": {"max_job_wall_s": 1800}})["max_job_wall_s"] == 1800
    assert M.settings({})["max_job_wall_s"] is None  # no default in code: the launch policy selects it
    with pytest.raises(ValueError):
        W.max_job_wall_s("gsim", {"gsim": {"max_job_wall_s": "30 min"}})
    with pytest.raises(ValueError):
        M.validate({"max_job_wall_s": 0})
    M.validate({"max_job_wall_s": 1800})
