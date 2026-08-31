"""The headline experiment must keep its two claims apart, and must not flatter itself.

Two failures this guards against, both of which look like success:

* reporting a prediction result as a recovery result. Off the reference corpus no reference
  implementation exists, so "fraction of reference" has no denominator there. A perfect prediction at a
  shape merlin emitted itself says nothing about whether the emitted code is fast.
* reading a lower bound that EXCEEDS its measurement as a good fit. A structural bound above the thing
  it bounds falsifies an input; quoting it as ~1.0 accuracy would turn a refutation into a headline.
"""
from __future__ import annotations

import importlib.util
import json

import pytest

from merlin.common.paths import repo_root


_MOD = None
_BODY = None
_BODY_SKIP: str | None = None


def _headline():
    global _MOD
    if _MOD is None:
        path = repo_root() / "merlin/experiments/performance_contract/headline.py"
        if not path.is_file():
            pytest.skip("headline driver absent")
        spec = importlib.util.spec_from_file_location("_headline", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MOD = mod
    return _MOD


def _report():
    """``report()`` once for the whole module, shared by the three tests that read it.

    One call is MEASURED at ~180 s: it rebuilds the emitted instruction plan for every generated
    shape up to 704x32x704. Three independent calls put the file at ~540 s and each single call
    within a factor of two of a 300 s per-test ceiling, which is what made these tests fail as
    timeouts rather than on their assertions. The body is identical for all three, so it is computed
    once. Nothing about the assertions changes -- they still read the real report.

    ``SystemExit`` is caught alongside ``Exception`` because it is NOT an ``Exception``:
    ``headline.load_suite()`` raises ``SystemExit`` when ``MERLIN_MLC_DIR`` is unset or the measured
    cycle suite is absent, so the previous ``except Exception`` let it escape and ERROR the test on
    any machine without the model checkout instead of skipping it. ``BaseException`` is deliberately
    NOT caught -- that would swallow a pytest timeout or an interrupt and report it as a skip.

    An EVIDENCE-FREE report is skipped rather than asserted over. The generated-shape measurements
    live in ``out/artifacts/cache/perf_headline/<target>/generated_runs.json``, which the layout
    convention marks PURGEABLE, and ``headline.load_measurements`` fails OPEN on its absence
    (returns ``{"runs": {}}``). The body it then produces still satisfies every assertion below --
    it reads "PREDICTS (7.2), n=0 of 0 ... a median None of measured", carries its ``n=``, carries
    its NOT SUPPORTED lines, and silently DROPS the warning that the program-schedule variant
    exceeds its own measurement. So all three tests would pass over a report with no evidence in it.
    Skipping on ``n_shapes_attempted == 0``, naming the absent cache, is the only reading of that
    state that is neither a false green nor a failure blamed on the machine.
    """
    global _BODY, _BODY_SKIP
    if _BODY is None and _BODY_SKIP is None:
        mod = _headline()
        try:
            body = mod.report("atlas", "vsim")
            if not body["claim_7_2_predicts"]["n_shapes_attempted"]:
                _BODY_SKIP = ("no generated-shape measurements: the purgeable measurement cache "
                              f"{mod.measure_path('atlas')} is absent, so the PREDICTS claim has no "
                              "evidence behind it (re-measure with `headline.py measure`)")
            else:
                _BODY = body
        except (Exception, SystemExit) as exc:  # noqa: BLE001 - absent cache/suite is a skip
            _BODY_SKIP = f"report unavailable here: {type(exc).__name__}: {exc}"
    if _BODY_SKIP is not None:
        pytest.skip(_BODY_SKIP)
    return _BODY


def test_the_driver_exposes_both_claims_separately() -> None:
    """Two named entry points, not one blended score."""
    mod = _headline()
    assert hasattr(mod, "claim_recovers")
    assert hasattr(mod, "claim_predicts")


def test_a_bound_above_its_measurement_is_reported_as_falsification() -> None:
    """The ratio nearest 1.0 must not be presentable as the best result.

    This is the trap: the program-schedule variant lands at ~1.01 of measured, which reads as a near
    perfect model. It is a LOWER bound, so exceeding the measurement means an input is wrong. The
    driver must say so rather than quote the ratio."""
    mod = _headline()
    src = (repo_root() / "merlin/experiments/performance_contract/headline.py").read_text()
    assert "falsifies" in src, (
        "a lower bound exceeding its measurement must be named as falsification, not quoted as accuracy")


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_the_written_result_states_what_it_does_not_support() -> None:
    """An experiment that only reports what it found overstates itself.

    Reported over the cached run, so this needs no oracle."""
    body = _report()
    text = json.dumps(body)
    assert "NOT SUPPORTED" in text, "the result must name what the evidence does not support"
    assert "% of peak" in text or "attainment" in text, (
        "no-attainment must be stated explicitly: speed_of_light is null for this target")


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_every_reported_claim_carries_its_n() -> None:
    """A ratio without its sample size is not a result -- 2 of 21 and 21 of 21 read identically."""
    body = _report()
    written = body["written_result"]
    claims = [line for line in written if line.startswith(("RECOVERS", "PREDICTS"))]
    assert len(claims) == 2, f"expected both claims in the written result, got {len(claims)}"
    for line in claims:
        assert "n=" in line, f"a claim was reported without its sample size: {line[:120]}"


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_the_two_claims_are_kept_apart_in_the_written_result() -> None:
    """The result must say in words that the two denominators are different things.

    The failure this prevents is quoting a prediction accuracy as a recovery fraction -- they read
    identically as a number near 1, and only one of them is about somebody else's kernel."""
    body = _report()
    assert "claim_7_1_recovers" in body and "claim_7_2_predicts" in body
    text = " ".join(body["written_result"])
    assert "NOT the same measurement" in text or "not the same measurement" in text.lower(), (
        "the written result must state that the two claims measure different things")
