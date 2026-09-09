"""The cert tier's TIME BUDGET, and what an exhausted budget is allowed to mean.

Two defects, one incident. A 97-capsule batch reported 87 L3-clean capsules for every arm that
finished, and the six capsules missing from all three were the six heaviest DRAM movers rather than
six programs with wrong numbers:

1. **The budget came from the wrong engine.** The broker sized its per-capsule wall clock from
   ``.oracle_timing.json::verilator_per_capsule_s`` (a VERILATOR measurement, and the only key that
   file holds) and then did not even apply it: the launch branch read
   ``(vpc * ncaps) if sim == "verilator" else 900``, so every other engine got a flat 900 s.
   ``rtl_engine_policy`` ranks ``gsim`` ABOVE ``verilator`` on cost, so the flat-900 branch was the
   normal path, and ``cert_cost`` fits gsim at a completely different law (a ~41 s floor plus
   ~0.00024 s/cycle, from 135 samples) than verilator (~55 s plus ~0.229 s/cycle, from 34).

2. **An abandoned cert read as a defect.** The tier record became a bare ``L3: fail`` whose reason
   said "timed out after 900 seconds", and ``agg_agentic_results._l3_evidence`` counted it exactly
   like a capsule whose L3 produced wrong numbers. "Not certified because unaffordable" and "not
   certified because wrong" are different findings -- the first is a lowering-cost problem -- and
   collapsing them silently cost an arm six capsules in its headline.

Both fixes are the kind that fail silently when they break: a wrong budget just produces more
timeouts, and a mis-bucketed timeout just produces a lower score. Hence this file.
"""
from __future__ import annotations

import importlib.util
import json
import sys

import pytest

from merlin.common.paths import merlin_dir

HARNESS = merlin_dir() / "experiments/capsule_bench/harness"


def _load(name: str):
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location(name, HARNESS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001 -- harness deps absent in this env
        pytest.skip(f"{name} not importable here: {type(exc).__name__}: {exc}")
    return mod


@pytest.fixture()
def broker():
    return _load("simjob_broker")


@pytest.fixture()
def agg():
    return _load("agg_agentic_results")


# --------------------------------------------------------------------------------------------------
# Defect A: the budget must come from the engine that will actually run.
# --------------------------------------------------------------------------------------------------

def _fits(**laws):
    """``{engine: CycleCostFit}`` with the real dataclass, so the fields the broker reads are real."""
    from merlin.targetgen.cert_cost import CycleCostFit
    return {eng: CycleCostFit(target="t", intercept_s=icept, per_cycle_s=rate, r2=0.5,
                              n_samples=99, cycles_min=1, cycles_max=cmax, engine=eng,
                              engine_basis="engine")
            for eng, (icept, rate, cmax) in laws.items()}


@pytest.fixture()
def stub_cost(monkeypatch):
    """Install a fake cost history whose two engines are far enough apart to tell which one was used."""
    from merlin.targetgen import cert_cost
    laws = _fits(gsim=(41.0, 0.00024, 4_000_000), verilator=(55.0, 0.229, 1_200))
    monkeypatch.setattr(cert_cost, "fits_cycles_for", lambda target, **kw: dict(laws))
    monkeypatch.setattr(cert_cost, "fit_cycles_for",
                        lambda target, *, engine=None, **kw: laws.get(engine))
    monkeypatch.delenv("MERLIN_REQUIRED_RTL_ENGINE", raising=False)
    return laws


def test_budget_is_sized_from_the_policys_preferred_engine_not_verilator(broker, stub_cost):
    """gsim is the engine the policy picks, so gsim's law -- not verilator's -- sets the budget."""
    secs, why = broker._cert_budget_s("t")
    assert secs is not None, why
    assert "gsim" in why, why
    # gsim's own law at 2x its measured range: 41 + 0.00024 * 8e6 = ~1961 s, x1.5 margin.
    assert secs == pytest.approx(int(1.5 * (41.0 + 0.00024 * 8_000_000)), abs=2), why
    # Verilator's law over ITS measured range is a few hundred seconds; sizing from it is the defect.
    veril = int(1.5 * (55.0 + 0.229 * 2_400))
    assert secs > veril, f"budget {secs}s looks like verilator's law ({veril}s), not gsim's: {why}"


def test_a_pinned_engine_decides_the_budget(broker, stub_cost, monkeypatch):
    """When the experiment PINS an engine, that engine's cost law is the one that applies."""
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "verilator")
    secs, why = broker._cert_budget_s("t")
    assert secs is not None and "verilator" in why, why
    assert secs == pytest.approx(int(1.5 * (55.0 + 0.229 * 2_400)), abs=2), why


def test_the_broker_uses_the_derived_budget(broker, stub_cost):
    """End to end through the function main() calls: with a fit on disk, the fit decides."""
    secs, why = broker._per_capsule_timeout(0)
    assert secs == pytest.approx(int(1.5 * (41.0 + 0.00024 * 8_000_000)), abs=2), why
    assert "derived from" in why and "gsim" in why, why


def test_the_budget_never_regresses_below_the_historical_floor(broker, monkeypatch):
    """A cheap fit must not SHORTEN the budget. Derivation may RAISE the wall clock, never lower it:
    the floor is what the old flat default already allowed every capsule."""
    from merlin.targetgen import cert_cost
    laws = _fits(gsim=(1.0, 1e-9, 10))
    monkeypatch.setattr(cert_cost, "fits_cycles_for", lambda target, **kw: dict(laws))
    monkeypatch.delenv("MERLIN_REQUIRED_RTL_ENGINE", raising=False)
    raw, _ = broker._cert_budget_s("t")
    assert raw is not None and raw < broker._CERT_TIMEOUT_FLOOR_S, "fixture must be a cheap fit"
    secs, why = broker._per_capsule_timeout(0)
    assert secs == broker._CERT_TIMEOUT_FLOOR_S, why


def test_an_explicit_request_beats_every_derivation(broker, stub_cost):
    assert broker._per_capsule_timeout(77)[0] == 77


def test_without_a_fit_the_recorded_verilator_measurement_still_applies(broker, monkeypatch,
                                                                       tmp_path):
    """The historical path is KEPT for a target with no cert history, so the new code cannot make an
    unmeasured target worse off than before."""
    from merlin.targetgen import cert_cost
    monkeypatch.setattr(cert_cost, "fits_cycles_for", lambda target, **kw: {})
    monkeypatch.delenv("MERLIN_REQUIRED_RTL_ENGINE", raising=False)
    monkeypatch.setattr(broker, "HERE", tmp_path)
    (tmp_path / ".oracle_timing.json").write_text(json.dumps({"verilator_per_capsule_s": 1500.0}))
    secs, why = broker._per_capsule_timeout(0)
    assert secs == 3000, why
    assert "verilator measurement" in why, why


def test_with_neither_a_fit_nor_a_measurement_the_old_constant_applies(broker, monkeypatch,
                                                                      tmp_path):
    from merlin.targetgen import cert_cost
    monkeypatch.setattr(cert_cost, "fits_cycles_for", lambda target, **kw: {})
    monkeypatch.delenv("MERLIN_REQUIRED_RTL_ENGINE", raising=False)
    monkeypatch.setattr(broker, "HERE", tmp_path)          # no .oracle_timing.json here
    secs, why = broker._per_capsule_timeout(0)
    assert secs == broker._CERT_TIMEOUT_FALLBACK_S, why


def test_no_measured_history_refuses_rather_than_guessing(broker, monkeypatch):
    """``cert_cost`` returning nothing must yield None + a reason, so the caller keeps its default."""
    from merlin.targetgen import cert_cost
    monkeypatch.setattr(cert_cost, "fits_cycles_for", lambda target, **kw: {})
    monkeypatch.delenv("MERLIN_REQUIRED_RTL_ENGINE", raising=False)
    secs, why = broker._cert_budget_s("t")
    assert secs is None and why, "an unmeasured target must refuse, not invent a cost law"


def test_a_broken_cost_reader_can_never_raise_in_the_broker(broker, monkeypatch):
    """This runs inside a broker that HOLDS sim slots. A raise there loses the slots and the round."""
    from merlin.targetgen import cert_cost

    def boom(*a, **kw):
        raise RuntimeError("cost history is a smoking crater")

    monkeypatch.setattr(cert_cost, "fits_cycles_for", boom)
    monkeypatch.setattr(cert_cost, "fit_cycles_for", boom)
    monkeypatch.delenv("MERLIN_REQUIRED_RTL_ENGINE", raising=False)
    secs, why = broker._cert_budget_s("t")
    assert secs is None and "smoking crater" in why


def test_the_budget_derivation_runs_no_availability_probe(broker, monkeypatch, stub_cost):
    """No simulator probe may run here. `rtl_engine_policy.select` needs probes that locate/elaborate
    designs and can raise, and this code path is entered by a broker holding sim slots.

    The call is RECORDED rather than raised out of: every failure inside the derivation is swallowed by
    design, so a test that only raised would pass whether or not the probe ran.
    """
    from merlin.targetgen import rtl_engine_policy
    calls = []

    def record(*a, **kw):
        calls.append(a)
        raise RuntimeError("probe refused")

    monkeypatch.setattr(rtl_engine_policy, "select", record)
    broker._cert_budget_s("t")
    assert calls == [], "the broker probed engine availability while holding sim slots"


@pytest.mark.parametrize("engine", ["vcs", "gsim", "verilator"])
def test_every_elaborated_rtl_engine_gets_the_derived_budget(broker, engine):
    """The launch branch used to name verilator alone, so gsim ran on a flat 900 s."""
    from merlin.targetgen.rtl_engine_policy import ENGINE_PRIORITY
    assert engine in ENGINE_PRIORITY
    assert broker._job_timeout_s(engine, 1, 1367) == 1367, (
        f"{engine} is an elaborated-RTL engine and must get the derived per-capsule budget")
    assert broker._job_timeout_s(engine, 3, 1367) == 3 * 1367


def test_the_functional_screen_keeps_its_own_budget(broker):
    """spike and the contract sentinel are not elaborated RTL; their budget is not the cert budget."""
    assert broker._job_timeout_s("spike", 1, 1367) == broker._SCREEN_TIMEOUT_S
    assert broker._job_timeout_s(broker._NEUTRAL_SIM, 40, 1367) == broker._SCREEN_TIMEOUT_S


def test_engine_list_is_derived_from_the_policy(broker):
    """The engines come from `rtl_engine_policy`, never from a second literal ladder in the broker."""
    from merlin.targetgen.rtl_engine_policy import ENGINE_PRIORITY
    assert broker._rtl_engines() == tuple(ENGINE_PRIORITY)


# --------------------------------------------------------------------------------------------------
# Defect B: an abandoned cert is unaffordable, not wrong.
# --------------------------------------------------------------------------------------------------

_TIMEOUT_REASON = ("elaborated_rtl crash: Command '['/x/emulator', '/y/package_kernel.elf']' "
                   "timed out after 900 seconds")
#: The same reason as it appears in a VERDICT, whose digits are redacted to '#'. Anything anchored on
#: the number would classify every real record wrongly.
_REDACTED_REASON = ("elaborated_rtl invocation failed: Command '['/x/emulator']' timed out after "
                    "# seconds")
_WRONG_NUMBERS = ("declared oracle tier(s) ['L3'] RAN and did not pass (on-mesh execution: # of # "
                  "tile(s) passed, # failed)")


def _verdict(tmp_path, capsules):
    run = tmp_path / "run"
    (run / "qa_history").mkdir(parents=True)
    (run / "qa_history" / "verdict_round_00.json").write_text(json.dumps({
        "n_capsules": len(capsules),
        "n_passed": sum(1 for c in capsules if c.get("status") == "pass"),
        "per_capsule": capsules}))
    return run


def _cap(name, *, status, l3, detail=None):
    return {"capsule": name, "label": "public", "status": status,
            "tiers": {"L0": "skipped", "L1": "skipped", "L2": "pass", "L3": l3},
            "failure_plane": "elaborated_rtl" if l3 == "fail" else None,
            "failure_detail": detail}


def test_an_abandoned_cert_is_unaffordable_and_a_wrong_answer_is_a_failure(agg, tmp_path):
    run = _verdict(tmp_path, [
        _cap("ok", status="pass", l3="pass"),
        _cap("too_slow", status="fail", l3="fail", detail=_TIMEOUT_REASON),
        _cap("wrong", status="fail", l3="fail", detail=_WRONG_NUMBERS),
    ])
    ev = agg._l3_evidence(run)
    assert ev["not_certified_budget"] == 1, ev
    assert ev["not_certified_failed"] == 1, ev
    assert ev["rtl_clean"] == 1, ev


def test_the_graders_structured_abandonment_list_is_authoritative(agg, tmp_path):
    """`qa_check` writes `tiers_abandoned` -- the tiers IT abandoned on budget. Where that field is
    present no text is interpreted at all, which is what keeps this working when the reason text is
    redacted out of the verdict."""
    cap = _cap("too_slow", status="fail", l3="fail", detail=None)
    cap["tiers_abandoned"] = ["L3"]
    ev = agg._l3_evidence(_verdict(tmp_path, [cap]))
    assert (ev["not_certified_budget"], ev["not_certified_failed"]) == (1, 0), ev


def test_an_abandonment_list_naming_another_tier_does_not_excuse_the_cert(agg, tmp_path):
    cap = _cap("wrong", status="fail", l3="fail", detail=_WRONG_NUMBERS)
    cap["tiers_abandoned"] = ["L1"]
    ev = agg._l3_evidence(_verdict(tmp_path, [cap]))
    assert (ev["not_certified_budget"], ev["not_certified_failed"]) == (0, 1), ev


def test_a_redacted_timeout_reason_is_still_recognised(agg, tmp_path):
    """The verdict's own copy of the reason has its digits replaced; it must still classify."""
    run = _verdict(tmp_path, [_cap("too_slow", status="fail", l3="fail", detail=_REDACTED_REASON)])
    ev = agg._l3_evidence(run)
    assert (ev["not_certified_budget"], ev["not_certified_failed"]) == (1, 0), ev


def test_the_brokers_own_abandonment_wording_is_recognised(agg, tmp_path):
    """The broker writes its own sentence when it reaps a job at the wall clock."""
    run = _verdict(tmp_path, [_cap("too_slow", status="fail", l3="fail",
                                   detail="gsim exceeded its time budget")])
    assert agg._l3_evidence(run)["not_certified_budget"] == 1


def test_an_unexplained_cert_failure_is_not_reclassified_as_unaffordable(agg, tmp_path):
    """Silence is not evidence of unaffordability. Moving a real defect out of the failure count is
    the more expensive mistake, so an absent reason stays a failure."""
    run = _verdict(tmp_path, [_cap("mystery", status="fail", l3="fail", detail=None)])
    ev = agg._l3_evidence(run)
    assert (ev["not_certified_budget"], ev["not_certified_failed"]) == (0, 1), ev


def test_a_cert_tier_that_never_ran_is_in_neither_bucket(agg, tmp_path):
    """A skipped cert is an absence. Attributing it to either cause would be an invention."""
    run = _verdict(tmp_path, [_cap("skipped", status="pass", l3="skipped"),
                              _cap("clean", status="pass", l3="pass")])
    ev = agg._l3_evidence(run)
    assert (ev["not_certified_budget"], ev["not_certified_failed"]) == (0, 0), ev


def test_the_grader_tier_record_shape_is_read_too(agg, tmp_path):
    """``capsule_result.json`` carries ``tiers.L3`` as a RECORD with its own ``reason``; the verdict
    carries a status string plus ``failure_detail``. Reading only one shape is how this goes silent."""
    run = _verdict(tmp_path, [{
        "capsule": "too_slow", "label": "public", "status": "fail",
        "tiers": {"L2": {"status": "pass", "reason": None},
                  "L3": {"status": "fail", "mandatory": False, "reason": _TIMEOUT_REASON}}}])
    ev = agg._l3_evidence(run)
    assert (ev["not_certified_budget"], ev["not_certified_failed"]) == (1, 0), ev


def test_existing_keys_keep_their_previous_meaning(agg, tmp_path):
    """A caller printing ``rtl_clean`` must report the same quantity it always did, with the new
    breakdown available BESIDE it rather than replacing it."""
    caps = [_cap("ok", status="pass", l3="pass"),
            _cap("l2_only", status="pass", l3="fail", detail=_TIMEOUT_REASON),
            _cap("wrong", status="fail", l3="fail", detail=_WRONG_NUMBERS)]
    ev = agg._l3_evidence(_verdict(tmp_path, caps))
    # Independently recomputed here exactly as the pre-change function defined them.
    assert ev["rtl_clean"] == len([c for c in caps if c["status"] == "pass"
                                   and c["tiers"]["L3"] == "pass"])
    assert ev["l2_only"] == len([c for c in caps if c["status"] == "pass"
                                 and c["tiers"]["L3"] not in ("pass", None)])
    assert ev["gate_passed"] == 2 and ev["n_capsules"] == 3
    assert ev["l3_source"] == "verdict_round_00.json"


def test_not_measured_still_reads_as_not_measured(agg, tmp_path):
    """No verdict must yield None for EVERY key, new ones included, so absent cannot read as zero."""
    ev = agg._l3_evidence(tmp_path / "no_such_run")
    assert set(ev) >= {"rtl_clean", "l2_only", "gate_passed", "n_capsules", "l3_source",
                       "not_certified_budget", "not_certified_failed"}
    assert all(v is None for v in ev.values()), ev


def test_the_classification_uses_no_regex(agg, broker):
    """This repo forbids regex in this role: a too-narrow pattern silently mis-buckets a
    differently-spelled reason, which is the defect being fixed, not a tool to fix it with."""
    text = (HARNESS / "agg_agentic_results.py").read_text()
    for bad in ("import re", "re.search", "re.match", "re.compile", "re.findall"):
        assert bad not in text, f"agg_agentic_results.py uses {bad}; parse structurally instead"
