"""Sizing a capsule against what its certification actually costs.

A capsule derived at an application's real shape is worthless if nobody can afford to certify it,
and the sweet spot between "too small to generalize" and "too big to simulate" is not something
anyone can pick by eye. These tests pin the two properties that make the cost model usable: it is
fitted from runs already paid for, and it REFUSES rather than guesses everywhere the evidence runs
out -- a target with no history, a budget below the fixed floor, a size past anything measured.

The measured shape of it, on gemmini, is why this module exists at all: a fixed floor of ~115 s that
a capsule pays for existing, and ~0.06 s per operand element on top, so the floor dominates below
~1900 elements while today's capsules are 256-512. The corpus is paying nearly the whole cost of a
certification to exercise a 16x16 tile.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import cert_cost as CC


def _fit_or_skip(target: str = "gemmini"):
    fit = CC.fit_for(target)
    if fit is None:
        pytest.skip(f"{target} has no measured certification history in this checkout")
    return fit


def test_a_target_with_no_certification_history_has_no_cost_model():
    """The refusal that matters most. A default here would be a number nobody measured driving a
    size somebody certifies and then quotes."""
    assert CC.fit_for("definitely_not_a_target") is None


def test_the_fit_rests_on_enough_distinct_sizes_to_be_a_line():
    """Two points define a line through anything, and one x-value defines nothing at all."""
    fit = _fit_or_skip()
    assert fit.n_samples >= 5
    assert fit.elements_max > fit.elements_min
    assert fit.sources, "a fit must name the runs it was built from"


def test_the_fit_predicts_the_runs_it_was_built_from():
    """A fit nobody checked against its own inputs is a straight line with a plausible slope. This
    is deliberately a loose band -- simulator time is noisy -- but it fails on a fit that is simply
    wrong, e.g. one whose intercept and slope have swapped roles."""
    fit = _fit_or_skip()
    mid = (fit.elements_min + fit.elements_max) // 2
    predicted = CC.predict_seconds(fit, mid)
    assert predicted is not None
    assert fit.intercept_s <= predicted <= fit.intercept_s + fit.per_element_s * fit.elements_max * 1.5


def test_a_fixed_floor_is_reported_so_headroom_is_visible():
    """`floor_dominates_below` is the number that says how much larger a capsule can get before it
    is paying for its size rather than for existing -- i.e. where representativeness is nearly free.
    It is the whole reason a derived capsule can be bigger than a tile without costing more."""
    fit = _fit_or_skip()
    assert fit.intercept_s > 0
    if fit.per_element_s > 0:
        assert fit.floor_dominates_below > 0


def test_a_prediction_past_the_measured_range_is_absent_not_large():
    """A fit built on hundreds of elements says nothing about hundreds of thousands. Returning a
    number anyway is how a capsule nobody could afford gets scheduled on the strength of
    arithmetic."""
    fit = _fit_or_skip()
    assert CC.predict_seconds(fit, fit.elements_max * 100) is None
    assert CC.predict_seconds(fit, 0) is None
    assert CC.predict_seconds(None, 1024) is None


def test_a_budget_below_the_fixed_floor_admits_no_capsule_at_all():
    """Not "a very small capsule" -- none. The floor is paid before any work happens, so a budget
    under it is a statement about the budget rather than about the shape."""
    fit = _fit_or_skip()
    assert CC.max_elements_within(fit, fit.intercept_s * 0.5) is None
    assert CC.max_elements_within(None, 600.0) is None


def test_a_generous_budget_is_clamped_to_what_the_evidence_supports():
    """The inverse of the refusal above: a budget large enough to imply a size far past anything
    measured yields the largest size the evidence actually supports, not the arithmetic answer."""
    fit = _fit_or_skip()
    huge = CC.max_elements_within(fit, fit.intercept_s + fit.per_element_s * fit.elements_max * 1000)
    assert huge is not None
    assert huge <= fit.elements_max * 2


def test_a_larger_budget_never_admits_a_smaller_capsule():
    fit = _fit_or_skip()
    small = CC.max_elements_within(fit, fit.intercept_s + 50)
    large = CC.max_elements_within(fit, fit.intercept_s + 500)
    if small is not None and large is not None:
        assert large >= small


def test_the_size_metric_is_the_largest_operand_and_tolerates_a_symbolic_dim():
    """Chosen by measurement, not argument: across the gemmini runs the largest single operand
    predicts cost better than total operand elements, and declared OUTPUT elements is degenerate
    because a capsule records its inputs and not its result shape."""
    assert CC.capsule_elements({"inputs": [
        {"name": "A0", "shape": [16, 32]}, {"name": "W", "shape": [32, 64]}]}) == 2048
    assert CC.capsule_elements({"inputs": []}) == 0
    # A symbolic dim makes THAT operand unmeasurable, not the capsule.
    assert CC.capsule_elements({"inputs": [
        {"name": "A0", "shape": ["?", 32]}, {"name": "W", "shape": [8, 8]}]}) == 64


def test_the_fit_predicts_capsules_it_has_never_seen():
    """The falsification that matters: refit without each capsule, then predict it. A line through
    its own inputs proves nothing — this asks whether the model generalizes to a size it was not
    told about, which is exactly what sizing a NEW capsule requires of it.

    The bound is deliberately loose. Measured on gemmini the median absolute error is 17.5% and the
    worst 51%, so this is a sizing instrument rather than a stopwatch; the assertion exists to catch
    a model that has stopped predicting at all, not to pretend to a precision it does not have."""
    import statistics

    from merlin.common.paths import merlin_dir

    timings = CC._timing_records("gemmini")
    sizes = CC._capsule_sizes([merlin_dir() / "contract" / "capsules"])
    points = [(sizes[n], s) for (n, _eng), (s, _src) in sorted(timings.items()) if sizes.get(n)]
    if len(points) < CC._MIN_SAMPLES + 1:
        pytest.skip("not enough measured capsules to hold one out")

    def _line(pts):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        den = sum((x - mx) ** 2 for x in xs)
        if den == 0:
            return None
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
        return my - slope * mx, slope

    errors = []
    for i in range(len(points)):
        held = _line(points[:i] + points[i + 1:])
        if held is None:
            continue
        intercept, slope = held
        x, y = points[i]
        errors.append(abs((intercept + slope * x) - y) / y)
    assert errors
    assert statistics.median(errors) < 0.35, (
        f"the cost model no longer predicts held-out capsules (median error "
        f"{statistics.median(errors):.0%}); sizing against it would be guessing")
    assert sum(1 for e in errors if e <= 0.5) >= 0.8 * len(errors), (
        "fewer than four in five held-out capsules land within 50% of prediction")


# --- which ENGINE produced the second ---------------------------------------------------------------
#
# Two elaborated-RTL engines answer the same capsule at the same fidelity and are NOT interchangeable as
# cost samples: measured on gemmini against the identical ELF, GSIM answers in 3.31 s where Verilator
# takes 86.83 s (hardware_pins.yaml, `gsim_compiler`). A fit over a mixture prices a capsule at neither
# engine's cost. The per-capsule record carries `engine`; the reshaping into `by_tier` used to drop it,
# so the mixture was not merely unhandled, it was invisible.

def test_the_engine_survives_the_reshaping_into_a_by_tier_block():
    doc = {"tiers": {"L3": {"cycle_accurate": True, "engine": "gsim",
                            "timing": {"sim_active_s": 3.31}}}}
    assert CC._per_tier_from_result(doc)["L3"]["engine"] == "gsim"


def test_the_engine_rides_in_the_basis_so_a_mixed_fit_is_visible():
    """The basis is the string every caller already keeps beside the number, which makes this readable
    off the fit's own sources rather than requiring a new channel."""
    fast = {"by_tier": {"L3": {"cycle_accurate": True, "engine": "gsim",
                               "sim_active_s": 3.31}}}
    slow = {"by_tier": {"L3": {"cycle_accurate": True, "engine": "verilator",
                               "sim_active_s": 86.83}}}
    _s_fast, basis_fast = CC._cycle_accurate_seconds(fast)
    _s_slow, basis_slow = CC._cycle_accurate_seconds(slow)
    assert basis_fast.endswith("@gsim")
    assert basis_slow.endswith("@verilator")
    assert basis_fast != basis_slow


def test_a_sample_with_no_recorded_engine_still_yields_a_basis():
    """Older records predate the field. They must keep contributing rather than start being dropped --
    the point is to make the mixture visible, not to discard history."""
    old = {"by_tier": {"L3": {"cycle_accurate": True, "sim_active_s": 12.0}}}
    seconds, basis = CC._cycle_accurate_seconds(old)
    assert seconds == 12.0
    assert basis and "@" not in basis


# ---------------------------------------------------------------------------------------------
# The engine is part of what identifies a cost sample. Two elaborated-RTL engines answer the same
# capsule at the same FIDELITY and roughly 26x apart in SECONDS (measured on the identical ELF: GSIM
# 3.31 s, Verilator 86.83 s), so a fit that mixes them prices a capsule at neither engine's cost.
# ---------------------------------------------------------------------------------------------

def _result(dirpath, capsule, seconds, engine=None):
    """One capsule_result.json declaring a cycle-accurate L3 tier, optionally naming its engine."""
    import json

    d = dirpath / capsule if engine is None else dirpath / f"{capsule}-{engine}"
    d.mkdir(parents=True)
    tier = {"timing": {"sim_active_s": seconds}, "cycle_accurate": True, "derived_from_rtl": True}
    if engine is not None:
        tier["engine"] = engine
    (d / "capsule_result.json").write_text(
        json.dumps({"capsule": capsule, "tiers": {"L3": tier}}), encoding="utf-8")


def test_one_capsule_on_two_engines_keeps_both_samples(tmp_path):
    """Keyed on the capsule NAME alone, the second engine's run overwrote the first and whichever file
    sorted last silently won -- so the 26x gap could never appear in the evidence at all."""
    _result(tmp_path, "C1", 3.31, engine="gsim")
    _result(tmp_path, "C1", 86.83, engine="verilator")
    recs = CC._timing_records("t", root=tmp_path)
    assert set(recs) == {("C1", "gsim"), ("C1", "verilator")}
    assert recs[("C1", "gsim")][0] == 3.31
    assert recs[("C1", "verilator")][0] == 86.83


def test_an_unrecorded_engine_is_unknown_never_guessed(tmp_path):
    """A record carrying NO discriminator at all names no engine and must not be guessed into a named
    bucket -- a sample of unknown provenance is not evidence about a particular machine."""
    _result(tmp_path, "C2", 10.0)
    assert set(CC._timing_records("t", root=tmp_path)) == {("C2", CC.UNKNOWN_ENGINE)}


def test_an_engine_recorded_only_in_the_EVIDENCE_filename_still_separates_the_buckets(tmp_path):
    """The recoverable history. Most cycle-accurate records on disk predate the ``engine`` field and
    carry the engine only as a console filename -- 772 of one target's 1447, under two different
    spellings of the same engine. Read as unattributed, all of them landed in ONE bucket with the
    genuinely unattributed records; read verbatim, one engine became two.

    The inference is weaker than a statement and the fit says which it rests on
    (:attr:`CycleCostFit.engine_basis`), but separating two engines beats pooling them: where both
    sources coexist on disk today they agree on every one of the 758 records that carry both."""
    import json

    for name, evidence, seconds in (("C3", "verilator_console.log", 90.0),
                                    ("C4", "rtl_verilator_console.log", 92.0),
                                    ("C5", "rtl_gsim_console.log", 4.0),
                                    ("C6", None, 7.0)):
        d = tmp_path / name
        d.mkdir()
        tier = {"timing": {"sim_active_s": seconds}, "cycle_accurate": True,
                "derived_from_rtl": True}
        if evidence:
            tier["evidence"] = evidence
        (d / "capsule_result.json").write_text(
            json.dumps({"capsule": name, "tiers": {"L3": tier}}), encoding="utf-8")

    recs = CC._timing_records("t", root=tmp_path)
    assert set(recs) == {("C3", "verilator"), ("C4", "verilator"), ("C5", "gsim"),
                         ("C6", CC.UNKNOWN_ENGINE)}, (
        "two spellings of one engine must be one bucket, and a record naming none stays unattributed")


def test_a_mixture_recoverable_only_from_the_EVIDENCE_is_still_reported_as_a_mixture(tmp_path,
                                                                                     monkeypatch):
    """THE LIVE PATH. The element fit is the one that sizes capsules today, and its callers pass no
    ``engine=``. It read the ``engine`` field and nothing else, so a history whose engines are only
    recoverable from the console name reported ``mixed_engines is False`` -- a two-engine mixture that
    said it was not one. The fit is unchanged (same samples, same coefficients); what changes is that a
    caller can now SEE it must refit per engine."""
    import json

    sizes = {}
    for i in range(CC._MIN_SAMPLES + 1):
        for tag, evidence, seconds in (("s", "verilator_console.log", 100.0 + 50 * i),
                                       ("f", "rtl_gsim_console.log", 1.0 + i)):
            name = f"{tag}{i}"
            sizes[name] = 100 * (i + 1)
            d = tmp_path / name
            d.mkdir()
            (d / "capsule_result.json").write_text(json.dumps({"capsule": name, "tiers": {"L3": {
                "timing": {"sim_active_s": seconds}, "cycle_accurate": True,
                "derived_from_rtl": True, "evidence": evidence}}}), encoding="utf-8")
    fit = _fit(tmp_path, monkeypatch, sizes)
    assert fit is not None
    assert fit.engines == ("gsim", "verilator")
    assert fit.mixed_engines is True, (
        "a fit averaging two engines 50x apart reported itself as single-engine")
    assert _fit(tmp_path, monkeypatch, sizes, engine="gsim").mixed_engines is False


def _fit(tmp_path, monkeypatch, sizes, **kw):
    monkeypatch.setattr(CC, "_capsule_sizes", lambda roots: sizes)
    return CC.fit_for("t", corpus_roots=[tmp_path], timing_root=tmp_path, **kw)


def test_fit_for_restricts_to_one_engine(tmp_path, monkeypatch):
    sizes = {}
    for i in range(CC._MIN_SAMPLES + 1):
        name = f"C{i}"
        sizes[name] = 100 * (i + 1)
        _result(tmp_path, name, 1.0 + i, engine="gsim")
        _result(tmp_path, name, 100.0 + 50 * i, engine="verilator")

    fast = _fit(tmp_path, monkeypatch, sizes, engine="gsim")
    slow = _fit(tmp_path, monkeypatch, sizes, engine="verilator")
    assert fast is not None and slow is not None
    assert fast.engines == ("gsim",) and slow.engines == ("verilator",)
    assert not fast.mixed_engines and not slow.mixed_engines
    # Each fit describes ITS engine, and the slow one is dramatically more expensive per element.
    assert slow.per_element_s > fast.per_element_s * 5
    assert CC.predict_seconds(slow, 300) > CC.predict_seconds(fast, 300)


def test_an_unfiltered_fit_over_two_engines_says_it_is_a_mixture(tmp_path, monkeypatch):
    """Reported, not refused: a history predating the discriminator is legitimately unattributed and
    must keep working. But a caller sizing a budget for one engine must be able to SEE that this fit
    is not about one engine, instead of reading an average as a measurement."""
    sizes = {}
    for i in range(CC._MIN_SAMPLES + 1):
        name = f"C{i}"
        sizes[name] = 100 * (i + 1)
        _result(tmp_path, name, 1.0 + i, engine="gsim")
        _result(tmp_path, name, 100.0 + 50 * i, engine="verilator")
    mixed = _fit(tmp_path, monkeypatch, sizes)
    assert mixed is not None
    assert mixed.mixed_engines is True
    assert mixed.engines == ("gsim", "verilator")


def test_an_unattributed_history_fits_exactly_as_it_did_before(tmp_path, monkeypatch):
    """The compatibility guarantee: with no engine recorded anywhere, keying on the pair changes
    nothing -- same samples, same coefficients -- and the fit does not claim to be about an engine."""
    sizes = {}
    for i in range(CC._MIN_SAMPLES + 1):
        name = f"C{i}"
        sizes[name] = 100 * (i + 1)
        _result(tmp_path, name, 1.0 + i)
    unfiltered = _fit(tmp_path, monkeypatch, sizes)
    explicit = _fit(tmp_path, monkeypatch, sizes, engine=CC.UNKNOWN_ENGINE)
    assert unfiltered is not None
    assert unfiltered.n_samples == CC._MIN_SAMPLES + 1
    assert unfiltered.engines == (CC.UNKNOWN_ENGINE,)
    assert unfiltered.mixed_engines is False
    assert (explicit.intercept_s, explicit.per_element_s) == (unfiltered.intercept_s,
                                                              unfiltered.per_element_s)
