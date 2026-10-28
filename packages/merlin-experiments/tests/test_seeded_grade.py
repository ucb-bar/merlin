"""A seeded grade checks what changed first, then times by largest gap, and carries proven cases last."""

import json
import os

import pytest

from merlin.targetgen import seeded_grade as S
from merlin.targetgen.capsule_scheduling import admission_relevance, run_pipelined, timing_history


def _carried(run="seedrun"):
    return {"carried": True, "source": f"carried from {run}", "earned_by_run": run}


def _document(name, *, status="pass", carried=False, cycles=100):
    tier = {
        "status": status,
        "cycles": cycles,
        "measured_now": not carried,
        **({"carried": _carried()} if carried else {"timing": {"sim_active_s": 1.0, "build_s": 1.0}}),
    }
    document = {
        "capsule": name,
        "status": status,
        "failure": None if status == "pass" else {"category": "X"},
        "tiers": {"L0": {"status": "pass", "measured_now": True}, "L2": tier},
    }
    label = S.label(document, ("L2",))
    document["tier_reuse"] = {"executed": [], "carried": [], **({"carried_from": label} if label else {})}
    return document


def _write(root, document, mtime=10**18):
    path = root / "runs" / "fixture-capsule-bench" / document["capsule"] / "capsule_result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))
    os.utime(path, ns=(mtime, mtime))


@pytest.fixture()
def seeded(tmp_path, monkeypatch):
    store = tmp_path / "store"
    monkeypatch.setenv("MERLIN_TIER_CERT_CACHE", str(store))
    monkeypatch.setenv("MERLIN_TIER_CERT_PORTABLE", "1")
    monkeypatch.setenv(S.ENV, "seedrun")
    monkeypatch.delenv("MERLIN_CAPSULE_ORDER", raising=False)
    return store


def test_nothing_is_seeded_without_the_plan_setting_or_the_portable_judge(monkeypatch, tmp_path):
    monkeypatch.setenv("MERLIN_TIER_CERT_CACHE", str(tmp_path))
    monkeypatch.delenv(S.ENV, raising=False)
    monkeypatch.setenv("MERLIN_TIER_CERT_PORTABLE", "1")
    assert S.seed_run() is None and not S.screen_carry("case", "L2")
    monkeypatch.setenv(S.ENV, "seedrun")
    assert S.seed_run() == "seedrun"
    monkeypatch.delenv("MERLIN_TIER_CERT_PORTABLE")
    assert S.seed_run() is None, "a carry needs the portable judge identity the seeded plan selects"
    monkeypatch.setenv("MERLIN_TIER_CERT_PORTABLE", "1")
    monkeypatch.setenv("MERLIN_TIER_CERT_CACHE", "0")
    assert S.seed_run() is None


def test_rank_puts_unproven_first_timing_next_and_carried_last():
    assert S.rank(None, proven_by_seed=False) == S.NEEDS_CORRECTNESS
    assert S.rank(None, proven_by_seed=True) == S.PROVEN
    assert S.rank({"passed": False, "carried": False}, proven_by_seed=True) == S.NEEDS_CORRECTNESS
    assert S.rank({"passed": True, "carried": False}, proven_by_seed=True) == S.NEEDS_TIMING
    assert S.rank({"passed": True, "carried": True}, proven_by_seed=False) == S.PROVEN


def test_carried_passes_are_labelled_and_counted():
    carried = _document("a", carried=True)
    executed = _document("b")
    failed = _document("c", status="fail", carried=True)
    assert S.label(carried, ("L2",)) == "carried from seedrun"
    assert S.label(carried) is None, "an in-process check that ran now is a tier that was not carried"
    assert S.label(executed, ("L2",)) is None and S.label(failed, ("L2",)) is None
    assert S.summary([carried, executed, failed], ("L2",)) == {
        "executed": {"L0": 3, "L2": 1},
        "carried": {"L2": 2},
        "fully_carried_cases": 1,
        "cases": 3,
    }


def test_index_round_trip_and_headroom(seeded, tmp_path):
    S.write_index(seeded, "seedrun", {"b": ["L2"], "a": ["L3", "L2", "L2"]})
    assert S.proven() == {"a", "b"}
    assert S.proven("other") == frozenset()
    verdict = {
        "feedback_brief": {
            "performance_priority": {
                "development_headroom": {
                    "largest_headroom": [
                        {"test_case": "x", "headroom_cycles": 50, "cycles": 90},
                        {"test_case": "y", "cycles": 7},
                        "junk",
                    ]
                }
            }
        }
    }
    assert S.headroom(verdict) == {"x": 50.0, "y": 7.0}
    (tmp_path / "ws" / "qa").mkdir(parents=True)
    (tmp_path / "ws" / "qa" / "verdict.json").write_text(json.dumps(verdict))
    assert S.workspace_gap(tmp_path / "ws")({"name": "x"}) == 50.0


def _order(caps):
    order = []
    run_pipelined(caps, lambda c, n: order.append(c["name"]) or {"capsule": c["name"], "status": "pass"}, max_workers=1)
    return order


def _history(tmp_path):
    root = tmp_path / "fruns_1"
    _write(root, _document("failing", status="fail"))
    _write(root, _document("small_gap", cycles=100))
    _write(root, _document("large_gap", cycles=900))
    _write(root, _document("carried_pass", carried=True, cycles=5000))
    return (root,)


NAMES = ("carried_pass", "seed_proven_new", "small_gap", "failing", "large_gap", "unproven_new")


def test_seeded_order_changed_first_then_largest_gap_then_carried(seeded, tmp_path):
    S.write_index(seeded, "seedrun", {"seed_proven_new": ["L2"], "carried_pass": ["L2"]})
    with timing_history(_history(tmp_path)):
        order = _order([{"name": n} for n in NAMES])
    assert order[:2] == ["failing", "unproven_new"] or order[:2] == ["unproven_new", "failing"]
    assert order[2:4] == ["large_gap", "small_gap"], "timing work goes by largest gap first"
    assert set(order[4:]) == {"carried_pass", "seed_proven_new"}, "proven cases go last"


def test_a_supplied_gap_outranks_observed_cycles(seeded, tmp_path):
    gaps = {"small_gap": 10_000.0, "large_gap": 1.0}
    with timing_history(_history(tmp_path)), admission_relevance(gap=lambda cap: gaps.get(cap["name"])):
        order = _order([{"name": n} for n in NAMES])
    assert order.index("small_gap") < order.index("large_gap")


def test_without_a_seed_the_order_is_unchanged(seeded, tmp_path, monkeypatch):
    with timing_history(_history(tmp_path)):
        seeded_order = _order([{"name": n} for n in NAMES])
    monkeypatch.delenv(S.ENV)
    with timing_history(_history(tmp_path)):
        plain = _order([{"name": n} for n in NAMES])
    assert plain == list(NAMES), "no seed, no background owner: the legacy order (input order on ties)"
    assert seeded_order != plain
