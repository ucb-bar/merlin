"""What a target's cost model does when the only timing anybody recorded is WALL CLOCK.

THE DEFECT THIS PINS, measured rather than imagined. One target in this repo has 2,036
``capsule_result.json`` files on disk carrying 1,501 cycle-accurate tier entries, 704 of them
PASSING -- and not one of them records ``timing.sim_active_s``, because the program-driven oracle
adapters emitted no timing block until they were fixed. ``cert_cost`` read only that one field, so
the target reported "no measured certification history", and the absence cascaded: no history means
no cost model, no cost model means ``applications.size_class`` can show no class affordable, and the
whole application axis derived ZERO capsules for a target that had been certified over a thousand
times. The target was never uncertified. It was unmeasured, and from inside the cost model the two
were indistinguishable -- which is the "unknown read as no" failure this file exists to prevent.

So three properties are gated here, and they pull against each other on purpose:

* the projection into the fit must CARRY the wall-clock number (it used to be dropped);
* a wall-clock history with a real size signal must YIELD a fit, and say what it rests on;
* a wall-clock history with NO size signal must be REFUSED, because that is not a cheap capsule --
  it is queue contention, and handing it back is worse than handing back nothing.

The last one has teeth: ``max_elements_within`` reads a non-positive slope as "size did not move
cost" and returns ``elements_max``, i.e. blanket permission to certify a capsule at any size the
history happened to contain. Measured on that target's real history, 105 wall-clock samples fit at
r2 0.0065 with a NEGATIVE per-element slope, and the same capsule appears at 12.106 s and 0.809 s
across two runs because ``adapter_wall_s`` includes ``oracle_wait_s``.
"""
from __future__ import annotations

import json

import pytest

from merlin.targetgen import cert_cost as CC
from merlin.targetgen import program_oracle as PO


def _result(dirpath, capsule, *, sim_active_s=None, adapter_wall_s=None, tier="L3"):
    """One ``capsule_result.json`` shaped exactly as ``capsule_runner`` writes it.

    ``sim_active_s=None`` with a positive ``adapter_wall_s`` is the shape of the whole broken
    history: the runner defaults the field it could not get from the adapter to None and stamps the
    wall it measured itself.
    """
    d = dirpath / f"{capsule}-{tier}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "capsule_result.json").write_text(json.dumps({
        "capsule": capsule,
        "tiers": {tier: {"status": "pass", "cycle_accurate": True, "derived_from_rtl": True,
                         "timing": {"build_s": None, "sim_active_s": sim_active_s,
                                    "oracle_wait_s": adapter_wall_s,
                                    "adapter_wall_s": adapter_wall_s}}},
    }), encoding="utf-8")


def _fit(tmp_path, monkeypatch, sizes, **kw):
    monkeypatch.setattr(CC, "_capsule_sizes", lambda roots: sizes)
    return CC.fit_for("t", corpus_roots=[tmp_path], timing_root=tmp_path, **kw)


# --- the projection ---------------------------------------------------------------------------


def test_the_reshaping_carries_the_wall_clock_number_it_used_to_drop():
    """``_per_tier_from_result`` PROJECTS a subset of keys, and everything outside that subset is
    silently gone. Dropping ``adapter_wall_s`` is how a thousand timed runs read as no runs at all:
    the record had the number and the projection threw it away."""
    doc = {"capsule": "C", "tiers": {"L3": {
        "cycle_accurate": True,
        "timing": {"sim_active_s": None, "adapter_wall_s": 42.5, "oracle_wait_s": 42.5}}}}
    assert CC._per_tier_from_result(doc)["L3"]["adapter_wall_s"] == 42.5


def test_a_wall_clock_sample_is_marked_as_one_in_its_basis():
    """A weaker measurement has to SAY it is weaker at the point of use, not in a docstring. The
    basis is the string every caller already keeps beside the number."""
    precise = CC._cycle_accurate_pick(
        {"by_tier": {"L3": {"cycle_accurate": True,
                            "timing": {}, "sim_active_s": 7.0, "adapter_wall_s": 90.0}}})
    fallback = CC._cycle_accurate_pick(
        {"by_tier": {"L3": {"cycle_accurate": True, "sim_active_s": None, "adapter_wall_s": 90.0}}})
    assert precise[0] == 7.0 and "wall_clock" not in precise[1]
    assert fallback[0] == 90.0 and "wall_clock" in fallback[1]


# --- the fallback -----------------------------------------------------------------------------


def test_a_wall_clock_history_with_a_size_signal_is_a_usable_fit(tmp_path, monkeypatch):
    """A weaker sample that says it is weaker beats no sample that reads as no history."""
    sizes = {}
    for i in range(CC._MIN_SAMPLES + 2):
        name = f"C{i}"
        sizes[name] = 100 * (i + 1)
        _result(tmp_path, name, adapter_wall_s=20.0 + 0.5 * sizes[name])
    fit = _fit(tmp_path, monkeypatch, sizes)
    assert fit is not None, "a target with a thousand timed runs must not read as uncertified"
    assert fit.per_element_s > 0 and fit.r2 > CC._WALL_CLOCK_MIN_R2
    assert fit.rests_on_wall_clock, "the fit must disclose that it leaned on the adapter wall"
    assert all("wall_clock" in s for s in fit.sources)


def test_a_precise_fit_does_not_claim_to_rest_on_wall_clock(tmp_path, monkeypatch):
    """The disclosure has to distinguish, or it is decoration."""
    sizes = {}
    for i in range(CC._MIN_SAMPLES + 2):
        name = f"C{i}"
        sizes[name] = 100 * (i + 1)
        _result(tmp_path, name, sim_active_s=20.0 + 0.5 * sizes[name], adapter_wall_s=999.0)
    fit = _fit(tmp_path, monkeypatch, sizes)
    assert fit is not None and not fit.rests_on_wall_clock


# --- the refusal ------------------------------------------------------------------------------


def test_queue_noise_is_refused_rather_than_returned_as_a_cheap_capsule(tmp_path, monkeypatch):
    """The measured shape of the real history: a NEGATIVE per-element slope, because the wall time
    is dominated by how long the capsule waited for an oracle slot. Returned, it would reach
    ``max_elements_within``, which reads a non-positive slope as "size did not matter" and hands
    back the largest size ever measured -- permission to certify anything."""
    sizes = {}
    for i in range(CC._MIN_SAMPLES + 2):
        name = f"C{i}"
        sizes[name] = 100 * (i + 1)
        _result(tmp_path, name, adapter_wall_s=500.0 - 3.0 * i)   # bigger capsules, LESS wall time
    assert _fit(tmp_path, monkeypatch, sizes) is None


def test_a_wall_clock_fit_that_explains_nothing_is_refused(tmp_path, monkeypatch):
    """A positive slope is not enough. 105 real samples fit at r2 0.0065 -- that is not "size barely
    moves cost over this range", it is contention with no size signal in it at all."""
    sizes = {}
    walls = [400.0, 12.0, 380.0, 9.0, 410.0, 11.0, 395.0]
    for i, w in enumerate(walls):
        name = f"C{i}"
        sizes[name] = 100 * (i + 1)
        _result(tmp_path, name, adapter_wall_s=w)
    fit = _fit(tmp_path, monkeypatch, sizes)
    assert fit is None or fit.r2 >= CC._WALL_CLOCK_MIN_R2


def test_a_weak_but_real_simulator_fit_is_not_held_to_the_wall_clock_bar(tmp_path, monkeypatch):
    """The bar exists because wall clock measures the HOST. Simulator seconds measure the capsule,
    so a weak relationship there is a fact about the target, not a reason to discard the history."""
    sizes = {}
    walls = [400.0, 12.0, 380.0, 9.0, 410.0, 11.0, 395.0]
    for i, w in enumerate(walls):
        name = f"C{i}"
        sizes[name] = 100 * (i + 1)
        _result(tmp_path, name, sim_active_s=w)
    fit = _fit(tmp_path, monkeypatch, sizes)
    assert fit is not None and fit.r2 < CC._WALL_CLOCK_MIN_R2


def test_the_refusal_asks_about_the_samples_under_the_fit_not_the_whole_history(tmp_path,
                                                                                monkeypatch):
    """SCOPING, and it is load-bearing. One simulator-timed record that the fit DISCARDS -- here,
    a capsule whose size cannot be read -- was enough to answer "not wall-clock only" for a line
    drawn entirely through wall-clock points, and the guard then let the queue noise through."""
    sizes = {}
    for i in range(CC._MIN_SAMPLES + 2):
        name = f"C{i}"
        sizes[name] = 100 * (i + 1)
        _result(tmp_path, name, adapter_wall_s=500.0 - 3.0 * i)
    _result(tmp_path, "UNSIZED", sim_active_s=1.0)          # timed, but no size -> not in the fit
    assert "UNSIZED" not in sizes
    assert _fit(tmp_path, monkeypatch, sizes) is None


# --- the repair -------------------------------------------------------------------------------


def test_a_program_oracle_records_the_seconds_the_cost_model_reads():
    """THE ACTUAL REPAIR, pinned across the module boundary it broke on. The program-driven adapters
    now time the simulator call; that is worth nothing unless the block they emit is the block
    ``cert_cost`` reads. Both halves are checked here on purpose -- the field name is the whole
    contract between them, and a rename on either side reintroduces a thousand null records."""
    block = PO._sim_timing(1.45, build_s=0.3)
    assert block["sim_active_s"] == 1.45 and block["build_s"] == 0.3
    # In-process simulator call: there is no queue to attribute, and inventing a wait would be
    # worse than recording none.
    assert block["oracle_wait_s"] == 0.0

    # The tier record capsule_runner assembles from it, read back by the cost model.
    seconds, basis, _engine = CC._cycle_accurate_pick(
        {"by_tier": {"L3": dict(block, cycle_accurate=True)}})
    assert seconds == 1.45, "the cost model must read the adapter's own simulator seconds"
    assert "wall_clock" not in basis, "a timed adapter must not be priced off the fallback"


@pytest.mark.parametrize("missing", ["sim_active_s", "oracle_wait_s"])
def test_the_timing_block_is_complete_rather_than_partially_filled(missing):
    """A partially-filled block is the shape the bug had: the key present, the value absent, and
    every downstream reader treating "no measurement" as "nothing to measure"."""
    assert PO._sim_timing(2.0)[missing] is not None
