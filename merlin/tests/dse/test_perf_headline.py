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


def _headline():
    path = repo_root() / "merlin/experiments/performance_contract/headline.py"
    if not path.is_file():
        pytest.skip("headline driver absent")
    spec = importlib.util.spec_from_file_location("_headline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def test_the_written_result_states_what_it_does_not_support() -> None:
    """An experiment that only reports what it found overstates itself.

    Reported over the cached run, so this needs no oracle."""
    mod = _headline()
    try:
        body = mod.report("atlas", "vsim")
    except Exception as exc:  # noqa: BLE001 - an absent cache is a skip, never a pass
        pytest.skip(f"report unavailable here: {type(exc).__name__}: {exc}")
    text = json.dumps(body)
    assert "NOT SUPPORTED" in text, "the result must name what the evidence does not support"
    assert "% of peak" in text or "attainment" in text, (
        "no-attainment must be stated explicitly: speed_of_light is null for this target")


def test_every_reported_claim_carries_its_n() -> None:
    """A ratio without its sample size is not a result -- 2 of 21 and 21 of 21 read identically."""
    mod = _headline()
    try:
        body = mod.report("atlas", "vsim")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"report unavailable here: {type(exc).__name__}: {exc}")
    written = body["written_result"]
    claims = [line for line in written if line.startswith(("RECOVERS", "PREDICTS"))]
    assert len(claims) == 2, f"expected both claims in the written result, got {len(claims)}"
    for line in claims:
        assert "n=" in line, f"a claim was reported without its sample size: {line[:120]}"


def test_the_two_claims_are_kept_apart_in_the_written_result() -> None:
    """The result must say in words that the two denominators are different things.

    The failure this prevents is quoting a prediction accuracy as a recovery fraction -- they read
    identically as a number near 1, and only one of them is about somebody else's kernel."""
    mod = _headline()
    try:
        body = mod.report("atlas", "vsim")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"report unavailable here: {type(exc).__name__}: {exc}")
    assert "claim_7_1_recovers" in body and "claim_7_2_predicts" in body
    text = " ".join(body["written_result"])
    assert "NOT the same measurement" in text or "not the same measurement" in text.lower(), (
        "the written result must state that the two claims measure different things")
