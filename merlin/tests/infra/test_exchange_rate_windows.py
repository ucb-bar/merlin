"""The window plan isolates one term per arm, and each way of breaking that is caught.

The point of these tests is not that today's plan is consistent -- it is that a LATER EDIT which
breaks the isolation fails here, rather than quietly measuring two things at once and producing a
slope that looks like a rate.
"""

from __future__ import annotations

import copy

import pytest

from merlin.perf.exchange_rate_windows import (
    WindowPlanError,
    check_plan,
    load_plan,
    window_terms,
)


def _plan() -> dict:
    return copy.deepcopy(load_plan())


def test_the_declared_plan_is_internally_consistent():
    report = check_plan()
    assert report["status"] == "consistent"
    assert report["fit_order"]
    # Every fit arm moves its own term and nothing else.
    for arm in report["arms"]:
        if arm["role"] == "fit":
            assert arm["moved"] == [arm["varies"]], arm


def test_every_arm_holds_useful_macs_constant():
    """The bit-identity precondition. If the useful work moves, the windows are not comparable."""
    for arm in check_plan()["arms"]:
        values = {cols["useful_macs"] for cols in arm["windows"].values()}
        assert len(values) == 1, f"arm {arm['arm']} changes useful_macs: {values}"


def test_a_closure_arm_is_required():
    plan = _plan()
    plan["arms"] = [arm for arm in plan["arms"] if arm.get("role") != "closure"]
    plan["fit_order"] = [arm["id"] for arm in plan["arms"]]
    with pytest.raises(WindowPlanError, match="no closure arm"):
        check_plan(body=plan)


def test_an_order_control_is_required():
    plan = _plan()
    plan["batch"] = {"mechanism": "x"}
    with pytest.raises(WindowPlanError, match="no order control"):
        check_plan(body=plan)


def test_two_points_are_not_a_rate():
    plan = _plan()
    for arm in plan["arms"]:
        if arm["id"] == "D":
            arm["windows"] = arm["windows"][:2]
    with pytest.raises(WindowPlanError, match="slope through two points"):
        check_plan(body=plan)


def test_splitting_k_is_refused():
    """K reassociates the accumulation; the per-group checksums stop being comparable."""
    plan = _plan()
    plan["arms"][0]["windows"][0]["k_chunks"] = 2
    with pytest.raises(WindowPlanError, match="K IS NEVER CUT"):
        check_plan(body=plan)


def test_an_arm_that_moves_a_second_term_is_refused():
    """The failure the prose design actually had: two terms moving at once."""
    plan = _plan()
    for arm in plan["arms"]:
        if arm["id"] == "D":
            # Narrow the last window's N chunk: dispatches still move, but so does the issue count.
            arm["windows"][-1]["n_edges"] = 0.5
    with pytest.raises(WindowPlanError, match="mesh_issue_cycles"):
        check_plan(body=plan)


def test_an_arm_that_secretly_moves_host_config_is_refused():
    plan = _plan()
    for arm in plan["arms"]:
        if arm["id"] == "I":
            arm["windows"][-1]["config_per_tile"] = 1
    with pytest.raises(WindowPlanError, match="host_config_emissions"):
        check_plan(body=plan)


def test_an_arm_whose_term_does_not_actually_move_is_refused():
    """The arm the framing proposed: a reshape the model prices identically."""
    plan = _plan()
    for arm in plan["arms"]:
        if arm["id"] == "I":
            # Every N chunk an exact multiple of the edge: S = N/E, so the issue count is invariant
            # however the chunks are cut. This is exactly why an "aspect" arm measures nothing.
            for window, m_edges in zip(arm["windows"], (4, 8, 16)):
                window["n_edges"] = 1
                window["m_edges"] = m_edges
    with pytest.raises(WindowPlanError, match="CONSTANT"):
        check_plan(body=plan)


def test_a_ragged_chunk_edge_is_refused():
    """A ragged edge adds partial-block waste belonging to no arm."""
    plan = _plan()
    plan["arms"][0]["windows"][0]["m_edges"] = 3  # 16 is not divisible by 3
    with pytest.raises(WindowPlanError, match="does not divide"):
        check_plan(body=plan)


def test_fit_order_must_name_every_arm():
    plan = _plan()
    plan["fit_order"] = plan["fit_order"][:-1]
    with pytest.raises(WindowPlanError, match="does not name exactly"):
        check_plan(body=plan)


def test_a_term_carried_as_previously_fit_must_have_been_fit():
    """A rate claimed from a term no earlier arm fitted is claimed from nothing."""
    plan = _plan()
    for arm in plan["arms"]:
        if arm["id"] == "H":
            arm["previously_fit"] = ["dispatches"]
    with pytest.raises(WindowPlanError, match="previously fit"):
        check_plan(body=plan)


# ---------------------------------------------------------------------------------------------
# The algebra itself, independent of the declared plan.
# ---------------------------------------------------------------------------------------------


def test_m_chunking_never_moves_the_issue_count():
    """Rows are charged linearly, so cutting M cannot change the modelled issue cycles.

    This is the asymmetry the whole design rests on, and it is why a dispatches arm exists at all.
    """
    shape = {"a": 16, "b": 16, "c": 16}
    base = window_terms({"id": "x", "m_edges": 16, "n_edges": 1, "k_chunks": 1}, shape)
    for m_edges in (1, 2, 4, 8, 16):
        terms = window_terms({"id": "x", "m_edges": m_edges, "n_edges": 1, "k_chunks": 1}, shape)
        assert terms["mesh_issue_cycles"] == base["mesh_issue_cycles"]
        assert terms["useful_macs"] == base["useful_macs"]
        assert terms["dispatches"] == 16 // m_edges * 16


def test_a_sub_edge_n_chunk_costs_a_whole_array_width():
    """The partial-block waste `array_issue_time` charges and a division cannot see."""
    shape = {"a": 16, "b": 16, "c": 16}
    full = window_terms({"id": "x", "m_edges": 4, "n_edges": 1, "k_chunks": 1}, shape)
    half = window_terms({"id": "x", "m_edges": 8, "n_edges": 0.5, "k_chunks": 1}, shape)
    quarter = window_terms({"id": "x", "m_edges": 16, "n_edges": 0.25, "k_chunks": 1}, shape)
    assert half["mesh_issue_cycles"] == 2 * full["mesh_issue_cycles"]
    assert quarter["mesh_issue_cycles"] == 4 * full["mesh_issue_cycles"]
    # and the useful work and the dispatch count did not move at all
    for terms in (half, quarter):
        assert terms["useful_macs"] == full["useful_macs"]
        assert terms["dispatches"] == full["dispatches"]


def test_a_sub_edge_m_chunk_is_refused_as_measuring_nothing():
    with pytest.raises(WindowPlanError, match="whole number of array edges"):
        window_terms({"id": "x", "m_edges": 0.5, "n_edges": 1, "k_chunks": 1}, {"a": 16, "b": 16, "c": 16})


def test_isolation_is_independent_of_the_array_edge():
    """Every ratio the plan relies on is E-free, so an isolation proved here holds at any array size.

    The terms are computed in edge multiples, so this is really a statement that no window's terms
    depend on a concrete edge -- which is what keeps the plan target-agnostic.
    """
    for shape in ({"a": 16, "b": 16, "c": 16}, {"a": 32, "b": 32, "c": 32}):
        full = window_terms({"id": "x", "m_edges": 4, "n_edges": 1, "k_chunks": 1}, shape)
        half = window_terms({"id": "x", "m_edges": 8, "n_edges": 0.5, "k_chunks": 1}, shape)
        assert half["mesh_issue_cycles"] / full["mesh_issue_cycles"] == 2
        assert half["dispatches"] == full["dispatches"]
