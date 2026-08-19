"""Routing rules for the harness x model bridge.

The bridge exists so a model can be driven by a harness that does not natively speak its protocol,
which is what turns "model vs model" into "model x harness". Getting the ROUTING wrong is silent and
expensive in both directions: routing a native pairing through the proxy changes every existing arm of
a campaign without saying so, and failing to route a non-native one runs a different model than the
manifest claims. Both happened while this was being built:

  * a hand-written vendor list parsed ``gpt-5.6-sol`` as vendor "gpt-5" and re-routed the codex arms;
  * ``codex_agent.resolve_model`` mapped ``nemotron`` onto DEFAULT_CODEX_MODEL, so the run would have
    measured OpenAI's default model under nemotron's name.
"""
from __future__ import annotations

import sys

import pytest

from merlin.common.paths import merlin_dir

_H = merlin_dir() / "experiments" / "capsule_bench" / "harness"
if str(_H) not in sys.path:
    sys.path.insert(0, str(_H))


@pytest.fixture()
def BR(monkeypatch):
    monkeypatch.setenv("MERLIN_PROXY_KEY", "test-key")
    monkeypatch.delenv("MERLIN_FORCE_BRIDGE", raising=False)
    import agent_bridge
    return agent_bridge


# ---------------------------------------------------------------- vendor derivation

def test_vendor_is_derived_not_listed(BR):
    """The vendor set comes from the model registry + proxy config, so adding a model needs no edit."""
    vendors = BR._known_vendors()
    assert {"anthropic", "nvidia", "zai"} <= vendors
    assert BR._vendor("nemotron") == "nvidia"
    assert BR._vendor("opus") == "anthropic"


def test_a_dotted_version_is_not_a_vendor(BR):
    """'gpt-5.6-sol' is a model NAME containing dots, not '<vendor>.<model>'.

    Treating its leading token as a vendor is what silently re-routed every native codex arm through
    the proxy -- the exact failure the derived vendor set exists to prevent.
    """
    assert BR._vendor("gpt-5.6-sol") != "gpt-5"
    assert BR._vendor("some-model-9.9-xyz") == "native"


def test_region_prefixed_ids_resolve_to_the_real_vendor(BR):
    assert BR._vendor("us.anthropic.claude-opus-4-6-v1") == "anthropic"


# ---------------------------------------------------------------- routing

def test_opencode_never_needs_the_bridge(BR):
    """opencode is natively multi-provider; routing it through a proxy would only add a confound."""
    for m in ("nemotron", "glm5", "opus"):
        assert BR.bridged_name(m, "opencode") is None


def test_a_non_anthropic_model_is_bridged_for_the_claude_cli(BR):
    assert BR.bridged_name("nemotron", "claude") == "nemotron"
    assert BR.bridged_name("glm5", "claude") == "glm5"


def test_an_anthropic_model_keeps_its_native_claude_path(BR):
    assert BR.bridged_name("opus", "claude") is None


def test_a_non_openai_model_is_bridged_for_codex(BR):
    """codex-cli 0.147 speaks only the Responses API, so anything else must go through the proxy."""
    assert BR.bridged_name("nemotron", "codex") == "nemotron"


def test_codex_keeps_its_own_catalogue_native(BR):
    """The seat model must not be re-routed -- that would change every existing codex arm."""
    assert BR.bridged_name("gpt-5.6-sol", "codex") is None


def test_an_unserved_model_is_never_routed(BR):
    """A model the proxy does not serve has nowhere to go; say so rather than inventing a route."""
    assert BR.bridged_name("kimi", "claude") is None
    assert BR.bridged_name("kimi", "codex") is None


def test_force_bridges_even_a_native_pairing(BR):
    """The proxy-vs-direct CONTROL: same model, same harness, once native and once through the proxy."""
    assert BR.bridged_name("opus", "claude", force=True) == "opus"


def test_force_is_a_parameter_not_only_an_env_var(BR, monkeypatch):
    """The config fragment is computed in the PARENT while the env var goes to the CHILD.

    Reading force only from the environment made the canary's 'bridged' leg emit no provider block and
    run native, so the control compared native to native and passed without touching the proxy.
    """
    # opus is served by the proxy AND native to the claude CLI, so it is the pairing where force is
    # the only thing that can produce a bridged route.
    assert BR.claude_env("opus") == {}, "an Anthropic model takes its native path by default"
    forced = BR.claude_env("opus", force=True)
    assert forced["ANTHROPIC_BASE_URL"] == BR.PROXY_BASE
    assert BR.claude_model_name("opus", force=True) == "opus"


def test_force_cannot_invent_a_route_the_proxy_cannot_serve(BR):
    """force bridges a NATIVE pairing; it does not conjure a backend for a model the proxy lacks.

    gpt-5.6-sol is the live example: it is codex-cli's own model, and ``openai.gpt-5.6-sol`` appears in
    the Bedrock catalogue but is not enabled on this account, so there is no second route to it.
    """
    assert BR._proxy_name("gpt-5.6-sol") is None
    assert BR.codex_config_fragment("gpt-5.6-sol", force=True) == ""
    assert BR.bridged_name("gpt-5.6-sol", "claude", force=True) is None


# ---------------------------------------------------------------- driver wiring

def test_codex_resolves_a_bridged_model_instead_of_a_default(BR):
    """resolve_model used to fall through to DEFAULT_CODEX_MODEL for any non-slug name."""
    import codex_agent
    assert codex_agent.resolve_model("nemotron") == "nemotron"
    assert codex_agent.resolve_model("glm5") == "glm5"
    assert codex_agent.resolve_model("gpt-5.6-sol") == "gpt-5.6-sol"


def test_codex_config_declares_the_measured_window(BR):
    """Without it codex budgets against fallback metadata -- the same class of defect as opencode's."""
    frag = BR.codex_config_fragment("nemotron")
    assert "model_context_window = 131072" in frag
    assert 'wire_api = "responses"' in frag, "codex 0.147 removed the chat wire protocol"


def test_claude_env_turns_bedrock_mode_off(BR):
    """CLAUDE_CODE_USE_BEDROCK speaks the Bedrock runtime directly and would bypass the bridge."""
    env = BR.claude_env("nemotron")
    assert env["CLAUDE_CODE_USE_BEDROCK"] == ""
    assert env["ANTHROPIC_BASE_URL"] == BR.PROXY_BASE


def test_the_record_states_the_bridge_caveats(BR):
    """A bridged number must carry why it is not comparable to a native one on cost or caching."""
    rec = BR.record("nemotron", harness="codex")
    assert rec["bridged"] is True
    assert rec["context_window"] == 131072
    assert any("caching" in c for c in rec["caveats"])
    native = BR.record("opus", harness="claude")
    assert native["bridged"] is False and native["caveats"] == []


def test_served_models_match_the_proxy_config(BR):
    """SERVED advertising a model the proxy cannot serve is how a run dies on round 0."""
    import yaml
    cfg = yaml.safe_load(BR.proxy_config_path().read_text())
    names = {e["model_name"] for e in cfg["model_list"]}
    assert set(BR.SERVED) <= names, f"SERVED has entries the proxy does not define: {set(BR.SERVED)-names}"


# ---------------------------------------------------------------- billing

def test_a_bridged_round_is_metered_whatever_the_driver_declares(BR, monkeypatch):
    """Billing follows the ACCOUNT the traffic was bought on, not the CLI that drove it.

    The codex driver declares subscription_notional because it normally runs on a ChatGPT seat. Bridged,
    it is spending our Bedrock key per token -- and booking that as notional hides real money from
    --max-spend-usd and from the campaign budget. The mirror of the same class of error as trusting a
    CLI's own total_cost_usd for a model it does not bill.
    """
    import run_baseline_qa_loop as R
    from merlin.targetgen import experiment_tokens as ET

    monkeypatch.setattr(R, "_DRIVER", "codex", raising=False)
    assert R._billing_mode("nemotron") == ET.METERED, "bridged codex traffic is real Bedrock spend"
    # the seat model still reports as a seat
    assert R._billing_mode("gpt-5.6-sol") != ET.METERED


# ---------------------------------------------------------------- harness naming

def test_the_driver_name_and_the_wiring_name_route_identically(BR):
    """The harness identifier travels under two spellings and both must reach the same decision.

    ``run_baseline_qa_loop._driver_for`` returns ``"claudecode"``; the runtime wiring in this module
    uses ``"claude"``. While ``bridged_name`` branched only on ``"claude"`` and fell through to
    ``return name`` for anything else, an Opus round under ``--driver claudecode`` was routed through
    the proxy: it ran natively on the subscription but recorded ``bridged: true``, and ``_trust_cli_cost``
    then DISCARDED the claude CLI's own authoritative cost and re-derived it from a rate table. A native
    run mislabelled and mispriced is exactly the defect that would invalidate the Opus control cell.
    """
    for model in ("opus", "nemotron", "glm5", "gpt-5.6-sol"):
        assert BR.bridged_name(model, "claudecode") == BR.bridged_name(model, "claude"), model


def test_native_opus_under_claude_code_is_not_bridged(BR):
    """Opus on the claude CLI is the native pairing — the control run depends on it staying native."""
    assert BR.bridged_name("opus", "claudecode") is None
    assert BR.bridged_name("opus", "claude") is None


def test_a_direct_to_bedrock_harness_never_bridges(BR):
    """opencode and converse both reach Bedrock themselves; putting a proxy in that path would be a
    silent extra hop that the run record would then describe as bridged."""
    for harness in ("opencode", "converse"):
        for model in ("opus", "nemotron", "glm5"):
            assert BR.bridged_name(model, harness) is None, (harness, model)


def test_an_unknown_harness_fails_closed(BR):
    """The old fallthrough guessed, and guessing is what produced the mislabelled Opus round. A harness
    nobody has declared a routing rule for must raise, not silently return a proxy name."""
    import pytest
    with pytest.raises(ValueError, match="unknown harness"):
        BR.bridged_name("opus", "some-new-cli")
