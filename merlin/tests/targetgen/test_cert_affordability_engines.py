"""Certification cost is per (target, ENGINE), and an unattributed second buys no fit.

The same capsule costs 3.31 s on one elaborated-RTL engine and 86.83 s on another. A cost model fitted
across both prices a capsule at neither engine's cost, so every "can we afford this cohort at L3" number
computed from it is about no machine that exists.

These tests pin the three properties that make the arithmetic mean something, and each is written so it
FAILS against the behaviour it replaces:

  * a fit never crosses engines -- and the undiscriminated fit over the same samples is shown, in the
    same test, to match neither engine;
  * a ``(target, engine)`` with no measured history yields ``None``, never a default, and a cohort
    priced against ``None`` has an unknown total rather than a zero one;
  * a record that does not NAME its engine is counted and reported, never attributed by guessing at the
    console filename -- the runner writes that filename from a static map that a run-time engine
    substitution does not update, so the filename is evidence of what was declared, not of what ran.

Plus a drift guard: over a history that is entirely one engine, this module must reproduce
``cert_cost.fit_for`` exactly, since the two fit the same law over the same metric.
"""
from __future__ import annotations

import json

import pytest

from merlin.targetgen import cert_affordability as CA
from merlin.targetgen import cert_cost as CC

TARGET = "t_under_test"
FAST, SLOW = "engine_fast", "engine_slow"

#: Capsule sizes chosen so a least squares over them is determined and the two engines' slopes differ by
#: the order of magnitude the real engines differ by.
SIZES = {"c1": 256, "c2": 512, "c3": 1024, "c4": 2048, "c5": 4096, "c6": 8192}


def _corpus(tmp_path):
    """A corpus whose capsules declare their operands, so ``cert_cost`` can size them."""
    root = tmp_path / "corpus"
    for name, n in SIZES.items():
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "capsule.yaml").write_text(
            f"name: {name}\nkind: isa\nlabel: public\ninputs:\n  - name: a\n    shape: [{n}]\n",
            encoding="utf-8")
    return root


def _timings(tmp_path, rows, *, name="timings"):
    """``rows`` is ``[(capsule, engine_or_None, seconds), ...]`` written as capsule_result records."""
    root = tmp_path / name
    for i, (capsule, engine, seconds) in enumerate(rows):
        d = root / f"run{i}"
        d.mkdir(parents=True, exist_ok=True)
        tier = {"cycle_accurate": True, "timing": {"sim_active_s": seconds},
                # The only trace an older record carries of its engine, and deliberately a LIE here:
                # a run-time substitution writes the console under the DECLARED engine's name.
                "evidence": f"{SLOW}_console.log"}
        if engine is not None:
            tier["engine"] = engine
        (d / "capsule_result.json").write_text(
            json.dumps({"capsule": capsule, "tiers": {"L3": tier}}), encoding="utf-8")
    return root


def _both_engines(tmp_path):
    """Every capsule certified on BOTH engines: the fast one at 0.01 s/element, the slow one at 0.26."""
    return _timings(tmp_path, [(c, FAST, 5.0 + 0.01 * n) for c, n in SIZES.items()]
                    + [(c, SLOW, 40.0 + 0.26 * n) for c, n in SIZES.items()])


def _fit(tmp_path, engine, timing_root):
    return CA.fit_for(TARGET, engine, corpus_roots=[_corpus(tmp_path)], timing_root=timing_root)


# ---------------------------------------------------------------------------------------------
# 1. a fit never crosses engines
# ---------------------------------------------------------------------------------------------
def test_a_fit_never_crosses_engines(tmp_path):
    """Two engines, one corpus, one target -- and two fits that each recover their own engine's law.

    The undiscriminated fit over the identical samples is computed here too, and asserted to match
    NEITHER: that is not decoration, it is the falsifier. Without it a per-engine split that quietly
    averaged would still satisfy the two assertions above it.
    """
    timing = _both_engines(tmp_path)
    fast, slow = _fit(tmp_path, FAST, timing), _fit(tmp_path, SLOW, timing)

    assert fast is not None and slow is not None
    assert fast.n_samples == slow.n_samples == len(SIZES)
    assert fast.per_element_s == pytest.approx(0.01, rel=1e-6)
    assert slow.per_element_s == pytest.approx(0.26, rel=1e-6)
    assert fast.intercept_s == pytest.approx(5.0, abs=1e-6)
    assert slow.intercept_s == pytest.approx(40.0, abs=1e-6)
    # a sample belongs to exactly one fit
    assert {s.engine for s in CA.samples_for(TARGET, corpus_roots=[_corpus(tmp_path)],
                                             timing_root=timing)["by_engine"][FAST]} == {FAST}

    # THE FALSIFIER: the same seconds fitted with no engine axis describe neither engine.
    mixed = CC.fit_for(TARGET, corpus_roots=[_corpus(tmp_path)], timing_root=timing)
    assert mixed is not None
    assert mixed.per_element_s != pytest.approx(fast.per_element_s, rel=1e-3)
    assert mixed.per_element_s != pytest.approx(slow.per_element_s, rel=1e-3)


def test_the_engines_cohort_totals_differ_by_the_engine_ratio(tmp_path):
    """The number the whole exercise is for: what the cohort costs, per engine, from its own samples."""
    timing = _both_engines(tmp_path)
    fast = CA.cohort_price(_fit(tmp_path, FAST, timing), SIZES)
    slow = CA.cohort_price(_fit(tmp_path, SLOW, timing), SIZES)
    assert fast["priced"] == slow["priced"] == len(SIZES)
    assert slow["total_s"] > 10 * fast["total_s"]
    # and each says what it rests on, with its own sample count
    assert f"{len(SIZES)} measured {FAST} certification(s)" in fast["basis"]


# ---------------------------------------------------------------------------------------------
# 2. no measured history, no number
# ---------------------------------------------------------------------------------------------
def test_a_target_engine_pair_with_no_history_yields_none(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert CA.fit_for(TARGET, FAST, corpus_roots=[_corpus(tmp_path)], timing_root=empty) is None
    got = CA.fits_for(TARGET, corpus_roots=[_corpus(tmp_path)], timing_root=empty)
    assert got["engines"] == {} and got["sample_counts"] == {}


def test_an_engine_with_history_does_not_lend_it_to_another(tmp_path):
    """One engine measured, the other not. The unmeasured one stays ``None`` -- it does not inherit."""
    timing = _timings(tmp_path, [(c, FAST, 5.0 + 0.01 * n) for c, n in SIZES.items()])
    assert _fit(tmp_path, FAST, timing) is not None
    assert _fit(tmp_path, SLOW, timing) is None


def test_a_cohort_priced_against_no_fit_has_an_unknown_total_not_a_zero_one():
    priced = CA.cohort_price(None, SIZES)
    assert priced["total_s"] is None
    assert sorted(priced["unpriceable"]) == sorted(SIZES)
    assert "no measured" in priced["basis"]


def test_a_capsule_past_the_evidence_is_excluded_from_the_total_not_extrapolated_into_it(tmp_path):
    """A cohort total that silently absorbs an unsupported guess is the number nobody should quote."""
    timing = _both_engines(tmp_path)
    fit = _fit(tmp_path, FAST, timing)
    huge = dict(SIZES)
    huge["enormous"] = int(max(SIZES.values()) * CC._EXTRAPOLATION_MARGIN) + 1
    priced = CA.cohort_price(fit, huge)
    assert priced["beyond_evidence"] == ["enormous"]
    assert priced["priced"] == len(SIZES)
    assert priced["total_s"] == pytest.approx(CA.cohort_price(fit, SIZES)["total_s"])


# ---------------------------------------------------------------------------------------------
# 3. an unattributed second is counted, never guessed
# ---------------------------------------------------------------------------------------------
def test_a_record_that_does_not_name_its_engine_is_counted_and_fits_nothing(tmp_path):
    """Every record here carries ``evidence: <SLOW>_console.log`` and NO ``engine`` field.

    Reading the engine off that filename would produce a confident, wrong fit for ``SLOW``. The runner
    writes that name from the contract's static tier map, which a run-time engine substitution does not
    update -- so it says what was declared, not what ran. Fail closed: no fit, and the count surfaced.
    """
    timing = _timings(tmp_path, [(c, None, 40.0 + 0.26 * n) for c, n in SIZES.items()])
    got = CA.fits_for(TARGET, corpus_roots=[_corpus(tmp_path)], timing_root=timing)
    assert got["engines"] == {}
    assert got["unattributed_samples"] == len(SIZES)
    assert CA.fit_for(TARGET, SLOW, corpus_roots=[_corpus(tmp_path)], timing_root=timing) is None


def test_engine_of_reads_the_field_and_only_the_field():
    assert CA.engine_of({"engine": " gsim "}) == "gsim"
    assert CA.engine_of({"evidence": "verilator_console.log"}) is None
    assert CA.engine_of({"engine": ""}) is None
    assert CA.engine_of(None) is None


def test_a_measurement_whose_capsule_is_not_in_the_corpus_is_reported_not_dropped(tmp_path):
    timing = _timings(tmp_path, [(c, FAST, 5.0 + 0.01 * n) for c, n in SIZES.items()]
                      + [("not_in_corpus", FAST, 99.0)])
    got = CA.samples_for(TARGET, corpus_roots=[_corpus(tmp_path)], timing_root=timing)
    assert got["unsized"] == 1
    assert len(got["by_engine"][FAST]) == len(SIZES)


# ---------------------------------------------------------------------------------------------
# 4. drift guard, and the gate that consumes this
# ---------------------------------------------------------------------------------------------
def test_a_single_engine_fit_reproduces_the_undiscriminated_one(tmp_path):
    """Same law, same metric, same samples: when the history IS one engine the two must agree exactly.

    This is what keeps the fit from silently becoming a second, different cost model.
    """
    timing = _timings(tmp_path, [(c, FAST, 5.0 + 0.01 * n) for c, n in SIZES.items()])
    roots = [_corpus(tmp_path)]
    mine = CA.fit_for(TARGET, FAST, corpus_roots=roots, timing_root=timing)
    theirs = CC.fit_for(TARGET, corpus_roots=roots, timing_root=timing)
    assert theirs is not None and mine is not None
    assert mine.n_samples == theirs.n_samples
    assert mine.intercept_s == pytest.approx(theirs.intercept_s)
    assert mine.per_element_s == pytest.approx(theirs.per_element_s)
    assert mine.r2 == pytest.approx(theirs.r2)
    assert mine.metric == theirs.metric


def _gate():
    """The affordability gate, imported by path -- it is a script, not an installed module."""
    import importlib.util

    from merlin.common.paths import repo_root

    path = repo_root() / "build_tools" / "scripts" / "check_cert_affordability.py"
    spec = importlib.util.spec_from_file_location("_afford_gate_engines", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:                       # noqa: BLE001
        pytest.skip(f"gate not importable here: {type(exc).__name__}: {exc}")
    return mod


def test_the_gate_prices_with_a_measured_fit_when_it_has_one():
    """The fitted branch must be REACHABLE.

    It was not: it tested the metric against a literal no fit ever carries, so every capsule in the
    corpus was priced by the single global calibration law while the report read as though a fit had
    been consulted. A dead branch and a missing one are indistinguishable from the output, which is
    exactly why this asserts on the basis string and not just on the number.
    """
    gate = _gate()
    metric = CC.CostFit.__dataclass_fields__["metric"].default
    fit = CC.CostFit(target=TARGET, intercept_s=10.0, per_element_s=0.5, r2=0.9, n_samples=12,
                     elements_min=64, elements_max=1024, metric=metric)
    secs, basis = gate._price(fit, 100)
    assert secs == pytest.approx(60.0)
    assert "fitted" in basis and "12 samples" in basis


def test_the_gate_says_when_it_has_no_measured_basis_rather_than_implying_one():
    gate = _gate()
    secs, basis = gate._price(None, 1024)
    assert secs > 0
    assert "no measured (target, engine) basis" in basis
