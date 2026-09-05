"""Cost pricing has ONE source of truth: the AET_PRICE_TABLE override file, honored by merlin's
experiment_tokens estimate (so it can't diverge from aet's PriceTable, which reads the same file).

Hermetic: a temp override file + monkeypatched env; no network, no real transcript.
"""
from __future__ import annotations

import importlib

import pytest

from merlin.targetgen import experiment_tokens as ET


@pytest.fixture(autouse=True)
def _reset_price_override_cache():
    yield
    ET._OVERRIDES = None


def test_rate_honors_the_shared_override_file(tmp_path, monkeypatch):
    pf = tmp_path / "prices.yaml"
    pf.write_text("zai.glm-5: [0.6, 2.2, 0.06, 0.75]\nnewmodel-9: {input: 1.5, output: 6.0}\n")
    monkeypatch.setenv("AET_PRICE_TABLE", str(pf))
    importlib.reload(ET)  # drop the module-level override cache so the env is re-read
    # per-Mtok in the file → per-token in the estimate
    assert ET._rate("zai.glm-5") == (0.6e-6, 2.2e-6)
    assert ET._rate("us.newmodel-9-v1") == (1.5e-6, 6.0e-6)   # dict form + substring match
    # a model not in the override still resolves via the built-in table (Anthropic families)
    assert ET._rate("claude-opus-4-8") == ET._RATES["opus"]


def test_four_bucket_override_drives_exact_subscription_notional_cost(tmp_path, monkeypatch):
    pf = tmp_path / "prices.yaml"
    pf.write_text("gpt-5.6-sol: [5, 30, 0.5, 5]\n")
    transcript = tmp_path / "codex.jsonl"
    transcript.write_text(
        '{"type":"turn.started"}\n'
        '{"type":"turn.completed","usage":{"input_tokens":3000000,'
        '"cached_input_tokens":1000000,"cache_write_input_tokens":1000000,'
        '"output_tokens":1000000}}\n')
    monkeypatch.setenv("AET_PRICE_TABLE", str(pf))
    importlib.reload(ET)

    summary = ET.parse_agent_transcript(
        transcript, driver="codex", model="gpt-5.6-sol",
        billing_mode=ET.SUBSCRIPTION_NOTIONAL)

    # 1M fresh*$5 + 1M cache-write*$5 + 1M cache-read*$0.50 + 1M output*$30.
    assert summary["subscription_notional_usd"] == 40.5
    assert summary["estimated_cost_usd"] is None


def test_two_bucket_override_keeps_legacy_cache_multipliers(tmp_path, monkeypatch):
    pf = tmp_path / "prices.yaml"
    pf.write_text("legacy-model: [2, 8]\n")
    monkeypatch.setenv("AET_PRICE_TABLE", str(pf))
    importlib.reload(ET)

    assert ET._bucket_rate("legacy-model-v1") == pytest.approx(
        (2e-6, 8e-6, 0.2e-6, 2.5e-6))


def test_no_override_uses_builtin_table(tmp_path, monkeypatch):
    monkeypatch.delenv("AET_PRICE_TABLE", raising=False)
    monkeypatch.setattr("merlin.common.paths._dotenv", lambda: {}, raising=False)
    importlib.reload(ET)
    assert ET._rate("claude-sonnet-4-6") == ET._RATES["sonnet"]


def test_malformed_override_never_breaks_accounting(tmp_path, monkeypatch):
    pf = tmp_path / "bad.yaml"
    pf.write_text("this: is: not: valid: mapping: [\n")
    monkeypatch.setenv("AET_PRICE_TABLE", str(pf))
    importlib.reload(ET)
    # a broken price file falls back to the built-ins, never raises
    assert ET._rate("claude-opus-4-8") == ET._RATES["opus"]
