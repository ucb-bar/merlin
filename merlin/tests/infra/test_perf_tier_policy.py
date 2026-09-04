"""The cert tier is rationed from a measured price, and an unpriced kernel fails CHEAP.

Most performance work must not reach the cycle-accurate tier -- that is the reason the analytical
tooling and the loop tier exist. The tier used to be chosen by a string a generator wrote down
(``sim_hint``, set from ``"L2+L3" if macs <= 2_000_000 else "L2_only"``), and the paired bench
defaulted an unlabelled kernel to ``"L2+L3"``. So the kernel nobody had priced took the MOST
expensive path, which is the one direction that cannot be recovered from: a wrongly-cheap plan
under-certifies and says so, a wrongly-expensive one silently spends the budget.
"""
from __future__ import annotations

import importlib.util
import sys

import pytest

from merlin.common.paths import repo_root


def _bench():
    d = repo_root() / "merlin" / "experiments" / "gemmini_perf_bench" / "scripts"
    if not (d / "run_perf_bench.py").exists():
        pytest.skip("the performance bench is not in this checkout")
    sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location("run_perf_bench", d / "run_perf_bench.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001 - the harness needs its own deps to import
        pytest.skip(f"run_perf_bench is not importable here: {type(exc).__name__}")
    return mod


def test_a_kernel_nobody_can_price_is_held_at_the_loop_tier():
    """The inverted default: unpriced must mean cheap, never the deepest tier available."""
    bench = _bench()
    corpus = [{"id": "UNPRICED", "m": 16, "n": 16}]
    # A target with no certified history prices nothing, which is exactly the unpriced case.
    plan = bench.plan_cert_tier(corpus, target="a-target-nobody-has-certified", budget_s=None)
    admitted, why = plan["UNPRICED"]
    assert admitted is False, f"an unpriced kernel took the cert tier: {why}"
    assert why, "a hold-back with no reason is indistinguishable from an oversight"


def test_a_declared_ceiling_is_never_exceeded():
    """A capsule that caps its own oracle tier must not be promoted past it by a budget."""
    bench = _bench()
    corpus = [{"id": "CAPPED", "m": 16, "n": 16, "max_oracle_tier": "L2"}]
    plan = bench.plan_cert_tier(corpus, target="any-target", budget_s=1e9)
    admitted, why = plan["CAPPED"]
    assert admitted is False and "caps its oracle tier" in why, why


def test_the_budget_admits_cheapest_first_and_says_what_it_dropped():
    """A rationed budget must spend on the most cover it can buy, and name what it could not."""
    bench = _bench()

    class _Fit:  # a stand-in price: seconds == output elements
        pass

    fit = _Fit()
    import merlin.targetgen.cert_cost as cc
    orig_fit_for, orig_predict = cc.fit_for, cc.predict_seconds
    cc.fit_for = lambda target, **kw: fit
    cc.predict_seconds = lambda f, elements: float(elements)
    try:
        corpus = [{"id": "BIG", "m": 100, "n": 100},      # 10000 s
                  {"id": "SMALL", "m": 10, "n": 10},      # 100 s
                  {"id": "MID", "m": 20, "n": 20}]        # 400 s
        plan = bench.plan_cert_tier(corpus, target="t", budget_s=600.0)
    finally:
        cc.fit_for, cc.predict_seconds = orig_fit_for, orig_predict

    assert plan["SMALL"][0] is True, plan["SMALL"]
    assert plan["MID"][0] is True, plan["MID"]      # 100 + 400 <= 600
    assert plan["BIG"][0] is False, plan["BIG"]     # would blow the budget
    assert "exceeds the remaining cert budget" in plan["BIG"][1]


def test_the_paired_bench_no_longer_defaults_to_the_expensive_tier():
    """The literal defect: an absent hint must not select the deepest tier."""
    d = repo_root() / "merlin" / "experiments" / "gemmini_perf_bench" / "scripts"
    src = d / "run_paired_perf_bench.py"
    if not src.exists():
        pytest.skip("the paired bench is not in this checkout")
    text = src.read_text(encoding="utf-8")
    assert 'setdefault("sim_hint", "L2+L3")' not in text, \
        "an unlabelled kernel again defaults to the cycle-accurate tier"
    assert 'setdefault("sim_hint", "L2_only")' in text
