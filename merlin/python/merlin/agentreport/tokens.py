"""Tokens and money for one run, with the three cost states kept apart.

MONEY SPENT, MONEY THE SAME TRAFFIC WOULD HAVE COST, AND NO FIGURE AVAILABLE ARE THREE DIFFERENT
QUANTITIES. Collapsing them is how a notional projection consumes a real budget ceiling and how an
unpriced model masquerades as the priciest one -- both of which have happened here. So a run's cost
is one of:

``metered``
    real per-token billing; the number is money that left an account.
``subscription_notional``
    a seat account, which is not billed per token at all. The number is what the same traffic WOULD
    have cost metered. It may be charted, never summed with metered spend, and never compared to a
    budget.
``unpriced``
    no rate exists for this model. The value is ``None`` -- never ``0.0``, because a free run and an
    unpriceable one look identical once you write a zero.

MODEL NAMES ARE NORMALIZED STRUCTURALLY, NOT BY TABLE. The corpus carries thirteen spellings for
eight models: a deployment path (``amazon-bedrock/``), a region-and-vendor prefix (``us.anthropic.``,
``zai.``, ``nvidia.``), and a version suffix (``-v1``, ``-v1:0``). Stripping those is a parse, so a
new deployment of an existing model needs no edit here. What CANNOT be repaired structurally is a
bare family name -- ``opus`` names no particular Opus -- so those are normalized and then FLAGGED,
because silently folding them into a specific version invents a fact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from merlin.agentreport.availability import Availability, derived, measured, unavailable

METERED = "metered"
NOTIONAL = "subscription_notional"
UNPRICED = "unpriced"

#: Deployment/vendor prefixes a model id may be published under. Stripped left to right; each is a
#: routing fact about WHERE the model was called, never about WHICH model it is.
_PATH_SEPARATORS = ("/",)
_VENDOR_PREFIXES = ("us.", "eu.", "apac.", "anthropic.", "openai.", "zai.", "nvidia.",
                    "amazon.", "moonshotai.", "qwen.", "deepseek.", "meta.", "mistral.")
#: Version/revision suffixes a deployment appends. Stripped only from the END.
_VERSION_SUFFIXES = ("-v1:0", "-v1", ":0")
#: A model id with no version component names a FAMILY, not a model. Listed so the flag can be set;
#: nothing is rewritten on the basis of this.
_BARE_FAMILIES = frozenset({"opus", "sonnet", "haiku", "glm5", "nemotron", "kimi", "qwen-coder"})
#: The harness's own placeholder for an event it could not attribute to a model.
SYNTHETIC = "<synthetic>"


def normalize_model(raw: str) -> tuple[str, bool]:
    """``(normalized_id, names_only_a_family)``.

    Structural: drop everything before the last path separator, then any vendor/region prefix, then a
    trailing version marker. A name that survives all three and still carries no version digit names
    a family rather than a model, and the caller is told so."""
    name = (raw or "").strip()
    if not name or name == SYNTHETIC:
        return name, False
    for sep in _PATH_SEPARATORS:
        if sep in name:
            name = name.rsplit(sep, 1)[-1]
    # Repeatedly, not once: a published id stacks them (``us.`` + ``anthropic.``), and stopping at
    # the first match leaves half a routing prefix welded to the model name.
    while True:
        lowered = name.lower()
        for prefix in _VENDOR_PREFIXES:
            if lowered.startswith(prefix) and len(name) > len(prefix):
                name = name[len(prefix):]
                break
        else:
            break
    lowered = name.lower()
    for suffix in _VERSION_SUFFIXES:
        if lowered.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name, name.lower() in _BARE_FAMILIES


@dataclass
class TokenFacts:
    """What one run consumed, and what that cost -- or why no figure exists."""

    model_raw: str = ""
    model: str = ""
    model_is_family_only: bool = False
    driver_billing_mode: str = ""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    tool_calls: int = 0

    wall_s: float = 0.0
    active_wall_s: float = 0.0
    rate_limit_wait_s: float = 0.0

    cost_kind: str = UNPRICED
    cost_usd: float | None = None        # metered spend ONLY. None unless cost_kind == METERED.
    notional_usd: float | None = None    # what a seat run would have cost. Never summed with cost_usd.
    cost_reason: str = ""

    availability: Availability = field(default_factory=Availability)

    @property
    def cached_share(self) -> float | None:
        billed = self.input_tokens + self.cache_read_tokens + self.cache_creation_tokens
        return self.cache_read_tokens / billed if billed else None


def _int(value) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


def _float(value) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def read_tokens(run_dir: Path) -> TokenFacts:
    """Token buckets and cost for one run, from the harness's own accounting file."""
    facts = TokenFacts()
    path = run_dir / "cost_time_toolcalls.yaml"
    if not path.is_file():
        facts.availability.set("tokens", unavailable(
            f"{run_dir.name} wrote no cost_time_toolcalls.yaml, so this run has no token or cost "
            f"accounting at all"))
        facts.availability.set("cost", unavailable("no accounting file"))
        return facts
    try:
        import yaml
        doc = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore")) or {}
    except Exception:  # noqa: BLE001
        facts.availability.set("tokens", unavailable(
            f"{path.name} could not be parsed as YAML"))
        facts.availability.set("cost", unavailable("accounting file unreadable"))
        return facts

    facts.model_raw = str(doc.get("model") or "")
    facts.model, facts.model_is_family_only = normalize_model(facts.model_raw)
    facts.driver_billing_mode = str(doc.get("billing_mode") or "")
    facts.tool_calls = _int(doc.get("tool_calls"))
    facts.wall_s = _float(doc.get("wall_time_seconds"))
    facts.active_wall_s = _float(doc.get("active_wall_s")) or facts.wall_s
    facts.rate_limit_wait_s = _float(doc.get("rate_limit_wait_s"))

    # The per-model block splits cache READS from cache WRITES, which the flat keys do not: the flat
    # `tokens_cached` is their sum, and they are billed an order of magnitude apart. Prefer the split.
    native = doc.get("tokens_native_by_model")
    if isinstance(native, dict) and native:
        for entry in native.values():
            if not isinstance(entry, dict):
                continue
            facts.input_tokens += _int(entry.get("input"))
            facts.output_tokens += _int(entry.get("output"))
            facts.cache_read_tokens += _int(entry.get("cache_read"))
            facts.cache_creation_tokens += _int(entry.get("cache_create"))
            facts.reasoning_tokens += _int(entry.get("reasoning"))
        facts.availability.set("token_split", measured("tokens_native_by_model"))
    else:
        facts.input_tokens = _int(doc.get("tokens_input"))
        facts.output_tokens = _int(doc.get("tokens_output"))
        facts.reasoning_tokens = _int(doc.get("tokens_reasoning"))
        # `tokens_cached` merges reads and writes. Attributing it to reads would understate cost by
        # the write premium, so it is recorded as the sum it is and the split is declared missing.
        facts.cache_read_tokens = _int(doc.get("tokens_cached"))
        facts.availability.set("token_split", unavailable(
            "this run recorded only the summed `tokens_cached`; cache reads and cache writes are "
            "billed roughly an order of magnitude apart and cannot be separated after the fact",
            source="tokens_cached"))
    facts.total_tokens = _int(doc.get("tokens_total")) or (
        facts.input_tokens + facts.output_tokens + facts.cache_read_tokens + facts.cache_creation_tokens)

    if facts.total_tokens > 0:
        facts.availability.set("tokens", measured("cost_time_toolcalls"))
    else:
        facts.availability.set("tokens", unavailable(
            str(doc.get("reason") or "the accounting file recorded no token usage; a run killed "
                "before its driver reported usage keeps its transcript but not its token counts")))

    metered = doc.get("estimated_cost_usd")
    notional = doc.get("subscription_notional_usd")
    reason = str(doc.get("cost_unavailable_reason") or "")
    if isinstance(metered, (int, float)):
        facts.cost_kind, facts.cost_usd = METERED, float(metered)
        facts.availability.set("cost", measured("estimated_cost_usd"))
    elif isinstance(notional, (int, float)):
        facts.cost_kind, facts.notional_usd = NOTIONAL, float(notional)
        facts.cost_reason = reason or (
            "a subscription seat is not billed per token; this figure is what the same traffic would "
            "have cost metered, not money spent")
        facts.availability.set("cost", derived(facts.cost_reason, source="subscription_notional_usd"))
    else:
        facts.cost_kind = UNPRICED
        facts.cost_reason = reason or (
            f"no rate is available for model {facts.model_raw!r}, so this run has no dollar figure. "
            f"A zero here would be indistinguishable from a free run.")
        facts.availability.set("cost", unavailable(facts.cost_reason))
    return facts
