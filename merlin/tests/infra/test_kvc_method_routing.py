"""Every agent run in the amortization study routes through the Codex seat or Bedrock.

A standing constraint, pinned here so it cannot drift back. Anthropic and OpenAI models rate-cap
quickly on this account, and a capped run does not fail loudly -- it yields short rounds and a small
constant score, which reads as a weak agent rather than a throttled one. A comparison run that way
measures quota, not capability.

The check has to be driver-aware, since the same id means different things by route: `gpt-5.6-sol` is
the ChatGPT seat's own model on the codex driver and an OpenAI-on-Bedrock profile on converse.
"""
import importlib.util

import pytest

from merlin.common.paths import repo_root

_SCRIPT = (repo_root() / "merlin" / "experiments" / "llm_kernel_vs_compiler_v0"
           / "scripts" / "check_method_models.py")


def _load():
    if not _SCRIPT.is_file():
        pytest.skip(f"routing check not present at {_SCRIPT}")
    spec = importlib.util.spec_from_file_location("kvc_routing", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _v(cfg):
    return _load().violations(cfg, where="t")


CODEX = {"driver": "codex", "model": "gpt-5.6-sol", "billing_mode": "subscription_notional"}
BEDROCK = {"driver": "converse", "model": "qwen.qwen3-coder-480b-a35b-v1:0",
           "billing_mode": "metered"}


def test_the_two_configured_arms_are_allowed():
    assert _v(CODEX) == []
    assert _v(BEDROCK) == []


def test_the_shipped_method_configs_pass():
    """The real configs, not just hand-built dicts -- a config can drift without a test noticing."""
    import yaml
    mod = _load()
    methods = _SCRIPT.parent.parent / "methods"
    found = sorted(methods.glob("*/method.yaml"))
    assert found, f"no method configs under {methods}"
    for p in found:
        cfg = yaml.safe_load(p.read_text()) or {}
        assert mod.violations(cfg, where=p.name) == [], f"{p.name} violates the routing policy"


@pytest.mark.parametrize("driver", ["claudecode", "claude", "anthropic", ""])
def test_a_claude_driver_is_refused(driver):
    out = _v({"driver": driver, "model": "claude-opus-5", "billing_mode": "subscription_notional"})
    assert out and "not an allowed route" in out[0]


@pytest.mark.parametrize("model", [
    "us.anthropic.claude-sonnet-4-6",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    "openai.gpt-5.6-sol",          # the SAME model is fine on codex, refused metered on Bedrock
])
def test_a_rate_capping_vendor_is_refused_on_bedrock(model):
    out = _v({"driver": "converse", "model": model, "billing_mode": "metered"})
    assert any("rate-cap" in o for o in out), out


def test_the_seat_model_stays_allowed_on_its_own_driver():
    """gpt-5.6-sol on the codex seat is the point of that arm; only the Bedrock route is refused."""
    assert _v(CODEX) == []


def test_billing_mode_must_match_the_route():
    """A seat projection and metered spend must never share a total."""
    out = _v({**CODEX, "billing_mode": "metered"})
    assert any("billing_mode" in o for o in out), out
    out = _v({**BEDROCK, "billing_mode": "subscription_notional"})
    assert any("billing_mode" in o for o in out), out


def test_vendor_is_read_as_a_segment_not_a_substring():
    """A model merely NAMED after a vendor must not be rejected, nor a region prefix hide one."""
    mod = _load()
    assert mod._vendor_of("us.anthropic.claude-sonnet-4-6") == "anthropic"
    assert mod._vendor_of("anthropic.claude-x") == "anthropic"
    assert mod._vendor_of("qwen.qwen3-coder-480b-a35b-v1:0") == "qwen"
    # a vendor name inside the MODEL segment is not the vendor
    assert mod._vendor_of("qwen.not-anthropic-at-all") == "qwen"
    assert _v({"driver": "converse", "model": "qwen.not-anthropic-at-all",
               "billing_mode": "metered"}) == []
