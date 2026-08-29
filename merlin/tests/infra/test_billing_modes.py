"""A subscription-seat run must never report money spent, and an unpriced model must never be
priced by accident.

Both rules come from one artifact: a `gpt-5.6-sol` Codex round landed
`estimated_cost_usd: 17.2103` in its ledger. Two separate defects produced that number — the price
table's "conservative" fallback charged an unrecognized model at opus rates, and the ChatGPT
subscription that actually served the round is not billed per token at all. So the ledger now keeps
three distinct things apart: dollars actually owed, dollars the same traffic *would* have cost
metered (notional), and no figure at all with the gap named.
"""
from __future__ import annotations

import json
import sys

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen import experiment_tokens as ET

_HARNESS = merlin_dir() / "experiments/capsule_bench/harness"
_ATLAS = merlin_dir() / "experiments/capsule_bench/targets/atlas/target_experiment.yaml"


def _transcript(tmp_path, model, usage, name="t.jsonl"):
    line = {"type": "assistant",
            "message": {"id": "msg_1", "model": model, "usage": usage,
                        "content": [{"type": "text", "text": "hi"}]}}
    p = tmp_path / name
    p.write_text(json.dumps(line) + "\n")
    return p


_CLAUDE_USAGE = {"input_tokens": 1000, "cache_creation_input_tokens": 0,
                 "cache_read_input_tokens": 0, "output_tokens": 100}


# --------------------------------------------------------------------------- unpriced models
#: A model that is genuinely absent from every price table. These tests used to name gpt-5.6-sol, which
#: WAS unpriced when they were written and has since been given a rate -- so they began asserting that a
#: priced model is unpriced, and failed for a reason that had nothing to do with the behaviour under
#: test. The subject here is "a model we have no rate for", so the fixture has to be a model no table can
#: ever acquire, not whichever real model happened to be unpriced on the day.
_UNPRICED = "vendor.model-that-no-price-table-carries"


def test_an_unknown_model_yields_no_dollar_figure_and_names_the_gap(tmp_path):
    s = ET.parse_transcript(_transcript(tmp_path, _UNPRICED, _CLAUDE_USAGE))
    assert s["available"] is True
    assert s["estimated_cost_usd"] is None
    assert _UNPRICED in s["cost_unavailable_reason"]
    assert "subscription_notional_usd" not in s, "an unpriced model has no notional figure either"


def test_rate_returns_none_rather_than_defaulting_to_the_priciest_model():
    assert ET._rate(_UNPRICED) is None
    assert ET._rate("") is None
    assert ET._rate("claude-opus-4-8") == ET._RATES["opus"]


def test_a_priced_model_does_get_a_rate():
    """The complement of the above: the table is consulted, not ignored."""
    assert ET._rate("gpt-5.6-sol") is not None, "gpt-5.6-sol carries a seat rate; it must be found"


def test_tokens_are_still_accounted_when_the_price_is_unknown(tmp_path):
    """Losing the dollars must not lose the tokens — the token ledger is the primary record."""
    s = ET.parse_transcript(_transcript(tmp_path, _UNPRICED, _CLAUDE_USAGE))
    assert (s["tokens_input"], s["tokens_output"]) == (1000, 100)


# --------------------------------------------------------------------------- billing modes
def test_a_metered_run_reports_dollars_owed(tmp_path):
    s = ET.parse_transcript(_transcript(tmp_path, "claude-opus-4-8", _CLAUDE_USAGE))
    assert s["billing_mode"] == ET.METERED
    assert s["estimated_cost_usd"] == pytest.approx(1000 * 15e-6 + 100 * 75e-6, rel=1e-6)
    assert "subscription_notional_usd" not in s


def test_a_subscription_run_reports_notional_only(tmp_path):
    tp = _transcript(tmp_path, "claude-opus-4-8", _CLAUDE_USAGE)
    metered = ET.parse_transcript(tp)
    seat = ET.parse_transcript(tp, billing_mode=ET.SUBSCRIPTION_NOTIONAL)
    assert seat["estimated_cost_usd"] is None, "a seat is not billed per token"
    assert seat["subscription_notional_usd"] == metered["estimated_cost_usd"]
    assert "not money spent" in seat["cost_unavailable_reason"]


def test_the_billing_mode_never_changes_the_token_counts(tmp_path):
    tp = _transcript(tmp_path, "claude-opus-4-8", _CLAUDE_USAGE)
    a = ET.parse_transcript(tp)
    b = ET.parse_transcript(tp, billing_mode=ET.SUBSCRIPTION_NOTIONAL)
    keys = ("tokens_input", "tokens_cached", "tokens_output", "tokens_total", "tool_calls")
    assert [a[k] for k in keys] == [b[k] for k in keys]


# --------------------------------------------------------------------------- reasoning is a SUBSET
def test_reasoning_tokens_are_recorded_beside_output_never_added(tmp_path):
    usage = dict(_CLAUDE_USAGE, reasoning_output_tokens=40)
    s = ET.parse_transcript(_transcript(tmp_path, "claude-opus-4-8", usage))
    assert s["tokens_reasoning"] == 40
    assert s["tokens_output"] == 100, "reasoning is inside output, not next to it"
    assert s["tokens_total"] == s["tokens_input"] + s["tokens_cached"] + s["tokens_output"]


def test_an_absent_reasoning_field_stays_absent_rather_than_zero(tmp_path):
    s = ET.parse_transcript(_transcript(tmp_path, "claude-opus-4-8", _CLAUDE_USAGE))
    assert "tokens_reasoning" not in s, "unknown is not zero"


# --------------------------------------------------------------------------- the harness asks the driver
@pytest.fixture()
def loop(monkeypatch):
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", str(_ATLAS))
    if str(_HARNESS) not in sys.path:
        sys.path.insert(0, str(_HARNESS))
    import run_baseline_qa_loop as L  # noqa: PLC0415
    return L


def test_the_codex_driver_declares_a_subscription_seat(loop, monkeypatch):
    monkeypatch.setattr(loop, "_DRIVER", "codex")
    assert loop._billing_mode("gpt-5.6-sol") == ET.SUBSCRIPTION_NOTIONAL


def test_the_claude_cli_driver_is_metered(loop, monkeypatch):
    monkeypatch.setattr(loop, "_DRIVER", "claudecode")
    assert loop._billing_mode("claude-opus-4-8") == ET.METERED


def test_a_driver_that_declares_nothing_is_treated_as_metered(loop, monkeypatch):
    """Fail-safe direction: guessing "seat" for a metered driver would hide real spend."""
    monkeypatch.setattr(loop, "_DRIVER", "converse")
    assert loop._billing_mode("zai.glm-5") == ET.METERED


# --------------------------------------------------------------------------- the shared spend store
def test_a_seat_run_logs_zero_cost_and_a_notional_metric(monkeypatch, tmp_path):
    """`aet spend` is a MONEY view. A seat run must add nothing to it, while still keeping its
    projection where an analysis can find it."""
    from merlin.targetgen import aet_bridge as AB

    calls: dict = {"cost": [], "metrics": {}, "params": {}}

    class _Logger:
        @classmethod
        def start(cls, **_kw):
            return cls()

        def log_token_usage(self, **_kw): pass
        def log_model_usage(self, *_a): pass
        def log_agent_turns(self, *_a): pass
        def log_session_id(self, *_a): pass
        def close(self): pass

        def log_cost(self, cost, model=None):
            calls["cost"].append(cost)

        def log_param(self, name, value):
            calls["params"][name] = value

        def log_metric(self, name, value, **_kw):
            calls["metrics"][name] = value

    class _Result:
        cost_usd = 12.5
        model_usage: list = []
        turn_usage: list = []
        total_output_tokens = 10
        total_cache_creation_tokens = 0
        total_cache_read_tokens = 0
        model = "gpt-5.6-sol"
        num_turns = 1
        session_id = None
        tool_call_count = 0

        def estimated_cost_usd(self): return None
        def per_model_usage(self): return {}

    tp = tmp_path / "t.jsonl"
    tp.write_text('{"type":"assistant"}\n')
    import sys as _sys
    import types as _types
    fake = _types.ModuleType("aet.tracking.claude_stream")
    fake.parse_stream = lambda _t: _Result()
    fake_rl = _types.ModuleType("aet.tracking.run_logger")
    fake_rl.EvalRunLogger = _Logger
    for name, mod in (("aet", _types.ModuleType("aet")),
                      ("aet.tracking", _types.ModuleType("aet.tracking")),
                      ("aet.tracking.claude_stream", fake),
                      ("aet.tracking.run_logger", fake_rl)):
        monkeypatch.setitem(_sys.modules, name, mod)

    ok = AB.emit_to_aet(run_dir=tmp_path, run_id="r0", method="arm", model="gpt-5.6-sol",
                        target="t", transcript_paths=[tp], save_trajectory=False,
                        billing_mode="subscription_notional")
    assert ok is True
    assert calls["cost"] == [0.0], "a subscription seat must add zero money to the spend store"
    assert calls["metrics"]["cost.subscription_notional_usd"] == 12.5
    assert calls["params"]["billing_mode"] == "subscription_notional"


# --- repricing: dollars are re-DERIVED from stored tokens, never hand-edited ----------------------

def _cost_rec(**over):
    rec = {"available": True, "billing_mode": ET.SUBSCRIPTION_NOTIONAL, "model": "gpt-5.6-sol",
           "tokens_input": 1_000_000, "tokens_cached": 0, "tokens_cache_write": 0,
           "tokens_output": 100_000, "reasoning_is_subset_of_output": True,
           "estimated_cost_usd": None,
           "cost_unavailable_reason": "no price entry for model(s): gpt-5.6-sol"}
    rec.update(over)
    return rec


def _write_run(tmp_path, rec, manifest_process=None):
    import yaml
    d = tmp_path / "a_run"
    d.mkdir()
    (d / "cost_time_toolcalls.yaml").write_text(yaml.safe_dump(rec, sort_keys=False))
    if manifest_process is not None:
        (d / "run_manifest.yaml").write_text(
            yaml.safe_dump({"run_id": "a_run", "process": manifest_process}, sort_keys=False))
    return d


def test_reprice_fills_a_notional_figure_the_table_can_now_price(tmp_path, monkeypatch):
    monkeypatch.setattr(ET, "_OVERRIDES", {"gpt-5.6-sol": (5e-6, 30e-6)})
    d = _write_run(tmp_path, _cost_rec())
    r = ET.reprice_run(d)
    assert r["changed"]["subscription_notional_usd"][1] == 8.0   # 1M*$5 + 100k*$30 per 1M
    import yaml
    got = yaml.safe_load((d / "cost_time_toolcalls.yaml").read_text())
    assert got["subscription_notional_usd"] == 8.0
    assert got["estimated_cost_usd"] is None, "a seat run never reports a spend"
    assert "no price entry" not in got["cost_unavailable_reason"]


def test_reprice_syncs_the_manifest_copy_so_the_two_files_cannot_disagree(tmp_path, monkeypatch):
    monkeypatch.setattr(ET, "_OVERRIDES", {"gpt-5.6-sol": (5e-6, 30e-6)})
    # The exact drift seen in the wild: the cost file was repriced, its manifest copy was not.
    d = _write_run(tmp_path, _cost_rec(subscription_notional_usd=8.0,
                                       cost_unavailable_reason="subscription_notional: ..."),
                   manifest_process={"subscription_notional_usd": None,
                                     "cost_unavailable_reason": "no price entry for model(s): gpt-5.6-sol"})
    r = ET.reprice_run(d)
    assert r["manifest_changed"]["subscription_notional_usd"] == (None, 8.0)
    import yaml
    man = yaml.safe_load((d / "run_manifest.yaml").read_text())
    assert man["process"]["subscription_notional_usd"] == 8.0


def test_check_mode_reports_drift_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(ET, "_OVERRIDES", {"gpt-5.6-sol": (5e-6, 30e-6)})
    d = _write_run(tmp_path, _cost_rec())
    before = (d / "cost_time_toolcalls.yaml").read_text()
    r = ET.reprice_run(d, write=False)
    assert r["changed"], "drift must be reported"
    assert (d / "cost_time_toolcalls.yaml").read_text() == before, "--check must not write"


def test_reprice_never_overwrites_a_cost_the_provider_cli_reported(tmp_path, monkeypatch):
    # Our list-price table is an ESTIMATE; the CLI priced the call it actually made. A reprice may
    # correct our own arithmetic, never the provider's figure.
    monkeypatch.setattr(ET, "_OVERRIDES", {"opus": (1e-9, 1e-9)})
    rec = _cost_rec(billing_mode=ET.METERED, model="claude-opus-4-8", estimated_cost_usd=168.541)
    rec.pop("cost_unavailable_reason")
    d = _write_run(tmp_path, rec)
    r = ET.reprice_run(d)
    assert r["changed"] == {}
    import yaml
    assert yaml.safe_load((d / "cost_time_toolcalls.yaml").read_text())["estimated_cost_usd"] == 168.541


def test_an_unpriced_model_still_yields_no_dollar_figure(tmp_path, monkeypatch):
    monkeypatch.setattr(ET, "_OVERRIDES", {})
    d = _write_run(tmp_path, _cost_rec(model="some-unknown-model-id"))
    ET.reprice_run(d)
    import yaml
    got = yaml.safe_load((d / "cost_time_toolcalls.yaml").read_text())
    assert "subscription_notional_usd" not in got
    assert "no metered rate" in got["cost_unavailable_reason"]


def test_a_run_without_usage_metadata_is_skipped_not_priced_at_zero(tmp_path):
    d = _write_run(tmp_path, {"available": False, "reason": "no usage metadata in transcript"})
    r = ET.reprice_run(d)
    assert r["skipped"]
    assert r["changed"] == {}
