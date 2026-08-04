"""Cost pricing has ONE source of truth: the AET_PRICE_TABLE override file, honored by merlin's
experiment_tokens estimate (so it can't diverge from aet's PriceTable, which reads the same file).

Hermetic: a temp override file + monkeypatched env; no network, no real transcript.
"""
from __future__ import annotations

import importlib

from merlin.targetgen import experiment_tokens as ET


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
