"""An unmeasured round is not a free one.

The batch spend ceiling summed `float(cost or 0)`, so a round whose usage never arrived was booked at
$0.00. That is not a rare corner: the driver emits usage only on `turn.completed`, and a round killed by
`--round-timeout` never completes its turn — so the usage of a full four-hour round is simply absent.
Three separate A/B legs (v9 arm-4, v10 baseline, v11 arm-4) each burned exactly that round, and each was
recorded as costing nothing. A metered run could overrun its ceiling by an arbitrary amount while every
gate reported it comfortably under.

The cap cannot be enforced against a number nobody has. What it can do is refuse to pretend the number is
zero: unmeasured rounds are recorded as such, counted separately, and surfaced as a lower bound.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir

HARNESS = merlin_dir() / "experiments/capsule_bench/harness"


def _loop_mod():
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location("run_baseline_qa_loop",
                                                  HARNESS / "run_baseline_qa_loop.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001 — harness deps absent here
        pytest.skip(f"loop module not importable: {type(e).__name__}: {e}")
    return mod


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    f = tmp_path / "spend.jsonl"
    monkeypatch.setenv("MERLIN_SPEND_LEDGER", str(f))
    monkeypatch.setenv("MERLIN_MAX_SPEND_USD", "10")
    return f


def test_a_measured_round_still_counts(ledger):
    M = _loop_mod()
    over, total, cap = M._spend_over_cap(4.0)
    assert (over, total, cap) == (False, 4.0, 10.0)


def test_an_unmeasured_round_is_not_booked_as_zero(ledger):
    """The defect: `float(None or 0)` is 0.0, so four hours of real spend read as free."""
    M = _loop_mod()
    M._spend_over_cap(None)
    rows = [json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
    assert rows[-1]["cost"] is None
    assert rows[-1]["unmeasured"] is True


def test_an_unmeasured_round_does_not_inflate_the_measured_total(ledger):
    """It must not be counted as spend either — inventing a figure is the opposite failure."""
    M = _loop_mod()
    M._spend_over_cap(3.0)
    _, total, _ = M._spend_over_cap(None)
    assert total == 3.0


def test_the_cap_still_trips_on_measured_spend(ledger):
    M = _loop_mod()
    M._spend_over_cap(6.0)
    M._spend_over_cap(None)                       # unknown, neither free nor counted
    over, total, _ = M._spend_over_cap(5.0)
    assert over is True and total == 11.0


def test_zero_is_distinguishable_from_unknown(ledger):
    """A genuinely free round and an unmeasured one must not collapse to the same record — that collapse
    is what made the defect invisible."""
    M = _loop_mod()
    M._spend_over_cap(0.0)
    M._spend_over_cap(None)
    rows = [json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
    assert rows[0]["cost"] == 0.0 and rows[0]["unmeasured"] is False
    assert rows[1]["cost"] is None and rows[1]["unmeasured"] is True


def test_no_ledger_configured_is_still_a_no_op(ledger, monkeypatch):
    monkeypatch.delenv("MERLIN_SPEND_LEDGER")
    M = _loop_mod()
    assert M._spend_over_cap(99.0) == (False, 0.0, 0.0)


def test_a_malformed_line_cannot_defeat_the_cap(ledger):
    ledger.write_text("not json\n" + json.dumps({"cost": 9.0}) + "\n")
    M = _loop_mod()
    over, total, _ = M._spend_over_cap(2.0)
    assert over is True and total == 11.0
