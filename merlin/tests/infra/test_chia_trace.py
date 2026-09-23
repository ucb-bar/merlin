"""Public trace metadata is diagnostic; AET remains the billing authority."""

import json
from types import SimpleNamespace

import pytest

from merlin.benchharness.chia_trace import record_trace


@pytest.fixture
def handle(tmp_path):
    (tmp_path / "chia").mkdir()
    costs, metrics, events = [], [], []
    logger = SimpleNamespace(
        log_cost=lambda value: costs.append(value),
        log_metric=lambda *args: metrics.append(args),
        log_event=lambda *args: events.append(args),
    )
    return SimpleNamespace(run_dir=tmp_path, run_id="canonical", logger=logger, costs=costs, metrics=metrics)


def event(call_id="one", **extra):
    return {"type": "complete", "call_id": call_id, "extra": extra}


def report(handle):
    return json.loads((handle.run_dir / "chia/accounting.json").read_text())


def test_untyped_cost_missing_usage_and_failed_calls_are_unknown(handle):
    record_trace(
        handle, [event(cost_usd=2), event("empty"), {"type": "dispatch", "call_id": "failed"}], accounting="trace"
    )
    assert report(handle)["unknown_calls"] == 3
    assert report(handle)["metered_usd"] is None
    assert report(handle)["subscription_usd_notional"] is None
    assert handle.costs == []


def test_explicit_metadata_uses_aet_billing_and_deduplicates(handle):
    metered = event(cost_usd=0.25, billing_mode="per_token", cost_source="billed")
    seat = event("seat", cost_usd=1.25, billing_mode="subscription", cost_source="billed")
    record_trace(handle, [metered, metered, seat], accounting="trace")
    assert handle.costs == [0.25]
    assert handle.metrics == [("chia.subscription.cost_equivalent_usd", 1.25)]
    assert len(report(handle)["calls"]) == 2


def test_child_native_ledgers_are_never_charged_again(handle):
    record_trace(
        handle, [event(cost_usd=3, billing_mode="per_token", cost_source="billed")], accounting="child-ledgers"
    )
    assert handle.costs == [] and handle.metrics == []
    assert report(handle)["calls"][0]["status"] == "child_ledger_owned"


def test_conflicting_duplicate_refuses_before_any_cost_write(handle):
    with pytest.raises(ValueError, match="conflicting"):
        record_trace(handle, [event(cost_usd=1), event(cost_usd=2)], accounting="trace")
    assert handle.costs == []
    assert not (handle.run_dir / "chia/accounting.json").exists()


@pytest.mark.parametrize("cost", [-1, float("nan"), float("inf"), True, "0.5"])
def test_malformed_declared_cost_refuses(handle, cost):
    with pytest.raises(ValueError, match="cost"):
        record_trace(handle, [event(cost_usd=cost, billing_mode="per_token", cost_source="billed")], accounting="trace")
    assert handle.costs == []


def test_missing_price_is_not_a_measured_zero(handle):
    record_trace(handle, [event(cost_usd=None, billing_mode="per_token", cost_source="billed")], accounting="trace")
    assert report(handle)["unpriced_calls"] == 1
    assert report(handle)["metered_usd"] is None
    assert handle.costs == []


def test_repeated_failed_and_unknown_events_are_deduplicated(handle):
    pending = {"type": "dispatch", "call_id": "failed"}
    record_trace(handle, [pending, pending, event(), event()], accounting="trace")
    assert report(handle)["unknown_calls"] == 2


def test_invalid_explicit_mode_is_refused_by_aet(handle):
    with pytest.raises(ValueError, match="billing_mode"):
        record_trace(
            handle, [event(cost_usd=None, billing_mode="unrecognized", cost_source="billed")], accounting="trace"
        )
    assert handle.costs == []


def test_same_run_cannot_be_published_twice(handle):
    events = [event(cost_usd=0.25, billing_mode="per_token", cost_source="billed")]
    record_trace(handle, events, accounting="trace")
    with pytest.raises(FileExistsError):
        record_trace(handle, events, accounting="trace")
    assert handle.costs == [0.25]
