"""Harness-vs-model bridge: drive ANY registered model through ANY of the three agentic harnesses.

WHY
---
capsule-bench compares models, but every result so far confounded *model* with *harness*: the open models
were only ever driven by opencode, and gpt-5.6-sol was only ever driven by codex-cli. The measured gap
(20/20 in 67 actions vs 0/20 in 599) is therefore a gap between (model, harness) PAIRS, and the campaign
cannot say how much of it is either factor alone.

Each harness is locked to one wire protocol, which is what made the cross-product impossible:

    opencode 1.18.10     multi-provider natively            -> no bridge needed
    codex-cli 0.147.0    OpenAI **Responses** API ONLY      -> needs /v1/responses
                         (``wire_api = "chat"`` was removed in 0.147; the binary contains exactly one
                          endpoint string, ``/v1/responses`` -- even ``--oss`` goes through it)
    claude 2.1.234       Anthropic **Messages** API         -> needs /v1/messages, via ANTHROPIC_BASE_URL

A LiteLLM proxy serves both shapes over the same Bedrock-backed models, so the harness becomes the only
variable. Verified live on 2026-08-19: codex-cli and the claude CLI each completed a full agentic
tool-using turn against ``nvidia.nemotron-super-3-120b`` through this bridge.

THE BRIDGE IS A CONFOUND UNTIL PROVEN OTHERWISE
-----------------------------------------------
Routing through a translation layer can itself change a result. Two measured asymmetries that MUST travel
with any number produced this way:

  * **No prompt caching.** Codex's native 20/20 run served 9.65 M of its 9.89 M input tokens from cache;
    through the proxy ``cached_input_tokens`` is 0. Cost and latency are therefore not comparable to a
    native run, and only the ACTION/tier axis is.
  * **Harness preamble differs.** codex-cli sends ~12 K tokens of system prompt, tool and skill metadata
    before any task content. Against nemotron's 131 072-token window that is 9% consumed at step zero.

Run :func:`proxy_canary`-style controls (the same model through opencode-direct and through the proxy)
before attributing any difference to the harness.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path

import model_tiers as _MT

# ---------------------------------------------------------------------------------------------------
# Proxy endpoint. Host/port/key are overridable so a second campaign can run its own instance.
PROXY_HOST = os.environ.get("MERLIN_PROXY_HOST", "127.0.0.1")
PROXY_PORT = int(os.environ.get("MERLIN_PROXY_PORT", "4141"))
PROXY_KEY_ENV = "MERLIN_PROXY_KEY"
PROXY_BASE = f"http://{PROXY_HOST}:{PROXY_PORT}"

#: Context windows MEASURED from the provider's own 400 ("This model's maximum context length is N
#: tokens"), never from a vendor page. A model absent here is left on its harness's registry default --
#: we do not invent a window. Single source of truth; opencode_agent imports from here.
CONTEXT_WINDOWS: dict[str, int] = {
    "nvidia.nemotron-super-3-120b": 131072,
    "zai.glm-5": 202752,
}

#: The completion budget to ask for. MEASURED output is 200-400 tokens/step; the registry default of
#: 32_000 removed 24% of nemotron's window for a completion that never arrives.
# Transport-resilience knobs for the bridged codex provider (codex-cli 0.147.0 fields).
# Defaults are 5 retries / 300000 ms idle; a bridged round wants more headroom, see
# codex_config_fragment for why.
REQUEST_MAX_RETRIES = 8
STREAM_MAX_RETRIES = 12
STREAM_IDLE_TIMEOUT_MS = 900_000

DEFAULT_MAX_OUTPUT = int(os.environ.get("MERLIN_MAX_OUTPUT_TOKENS", "8000"))

#: Models the proxy serves. These are the ``model_name`` entries in proxy/litellm_config.yaml -- keep the
#: two in sync. Being listed here makes a model REACHABLE through the bridge; it does not by itself route
#: anything (see :func:`bridged_name`).
SERVED = ("nemotron", "glm5", "opus")


def _proxy_name(model: str) -> str | None:
    """The proxy ``model_name`` for *model*, or None when the proxy does not serve it.

    Accepts our alias (``nemotron``) or the concrete Bedrock id (``nvidia.nemotron-super-3-120b``), so a
    caller that has already resolved still routes correctly.
    """
    if model in SERVED:
        return model
    resolved = _MT.resolve(model)
    for name in SERVED:
        if _MT.resolve(name) == resolved or name == resolved:
            return name
    return None


_REGION_PREFIXES = ("us", "eu", "apac")


def _known_vendors() -> frozenset[str]:
    """Bedrock vendor slugs, DERIVED from the two registries we own rather than listed by hand.

    Sources: ``model_tiers.MODELS`` (alias -> Bedrock id) and the proxy config's ``litellm_params.model``
    entries (``bedrock/<vendor>.<model>``). Deriving matters here for the same reason it does everywhere
    else in this repo: a hand-written vendor list silently misclassifies the first model whose name
    happens to contain a dot -- ``gpt-5.6-sol`` parsed as vendor "gpt-5" and routed a native codex arm
    through the proxy.
    """
    vendors: set[str] = set()

    def add(rid: str) -> None:
        parts = rid.split("/")[-1].split(".")
        if len(parts) < 2:
            return
        head = parts[0]
        if head in _REGION_PREFIXES and len(parts) > 2:
            head = parts[1]
        vendors.add(head)

    for rid in _MT.MODELS.values():
        add(rid)
    try:
        import yaml
        cfg = yaml.safe_load(proxy_config_path().read_text()) or {}
        for entry in cfg.get("model_list") or []:
            add(str((entry.get("litellm_params") or {}).get("model", "")))
    except Exception:
        pass                      # config unreadable -> fall back to the registry-derived set
    return frozenset(vendors)


def _vendor(model: str) -> str:
    """Vendor of *model*, DERIVED from the id.

    Bedrock ids are ``<vendor>.<model>`` (optionally region-prefixed, ``us.anthropic....``), but so is
    any dotted version string, so the leading token counts as a vendor ONLY when it is one we actually
    know (:func:`_known_vendors`). Anything else is a harness-native model name -- codex's own catalogue
    -- and is reported as ``native``.
    """
    rid = _MT.resolve(model)
    parts = rid.split(".")
    if len(parts) < 2:
        return "native"
    head = parts[0]
    if head in _REGION_PREFIXES and len(parts) > 2:
        head = parts[1]
    return head if head in _known_vendors() else "native"


def bridged_name(model: str, harness: str, *, force: bool | None = None) -> str | None:
    """Proxy ``model_name`` to use for (*model*, *harness*), or None to take the harness's native path.

    Routing is HARNESS-DEPENDENT, and must be, because the same model is native to one harness and
    unreachable from another: ``gpt-5.6-sol`` is codex-cli's own model but has to be bridged to reach
    opencode or the claude CLI, while ``nemotron`` is native to opencode and has to be bridged for the
    other two. Deciding by model alone silently re-routed every existing codex arm through the proxy.

    ``force`` (or ``MERLIN_FORCE_BRIDGE=1``) bridges even a native pairing. That is the proxy-vs-direct
    CONTROL: the same model, same harness, once native and once through the bridge, so the bridge's own
    effect is measured instead of assumed.
    """
    if force is None:
        force = os.environ.get("MERLIN_FORCE_BRIDGE", "") == "1"
    name = _proxy_name(model)
    if name is None:
        return None                       # proxy does not serve it; nothing to route
    if force:
        return name
    vendor = _vendor(model)
    if harness == "opencode":
        return None                       # natively multi-provider
    if harness == "codex":
        # codex-cli reaches its own catalogue on the subscription seat; everything else needs Responses.
        return None if vendor in ("native", "openai") else name
    if harness == "claude":
        return None if vendor == "anthropic" else name
    return name


def context_window(model: str) -> int | None:
    """Measured context window for *model*, or None when we have not measured one."""
    return CONTEXT_WINDOWS.get(_MT.resolve(model))


def max_output_tokens(model: str) -> int:
    ctx = context_window(model)
    return min(DEFAULT_MAX_OUTPUT, ctx // 8) if ctx else DEFAULT_MAX_OUTPUT


def proxy_key() -> str:
    key = os.environ.get(PROXY_KEY_ENV, "")
    if not key:
        raise RuntimeError(f"{PROXY_KEY_ENV} is unset; the bridge cannot authenticate to the proxy")
    return key


# ---------------------------------------------------------------------------------------------------
# Lifecycle

def is_up(timeout: float = 2.0) -> bool:
    """True when something is listening on the proxy port."""
    try:
        with socket.create_connection((PROXY_HOST, PROXY_PORT), timeout=timeout):
            return True
    except OSError:
        return False


def proxy_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "proxy" / "litellm_config.yaml"


def proxy_venv_python() -> Path:
    from merlin.common.paths import repo_root
    return repo_root() / "build" / "proxy-venv" / "bin" / "litellm"


def start_proxy(log_path: Path, *, wait_s: int = 90) -> dict:
    """Start the LiteLLM proxy if it is not already up. Returns a record for the run manifest.

    Idempotent: a proxy already serving this port is adopted rather than duplicated, because several
    concurrent arms of one campaign legitimately share it.
    """
    if is_up():
        return {"proxy": PROXY_BASE, "started_by_us": False, "config": str(proxy_config_path())}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(proxy_venv_python()), "--config", str(proxy_config_path()),
           "--port", str(PROXY_PORT), "--host", PROXY_HOST]
    with open(log_path, "ab") as lf:
        subprocess.Popen(cmd, stdout=lf, stderr=lf, start_new_session=True)
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if is_up():
            return {"proxy": PROXY_BASE, "started_by_us": True, "config": str(proxy_config_path()),
                    "log": str(log_path)}
        time.sleep(2)
    raise RuntimeError(f"litellm proxy did not come up within {wait_s}s; see {log_path}")


# ---------------------------------------------------------------------------------------------------
# Per-harness wiring

CODEX_PROVIDER_ID = "merlinproxy"


def codex_config_fragment(model: str, *, force: bool | None = None) -> str:
    """The ``config.toml`` fragment routing codex-cli at the proxy, or "" for a native model.

    Declaring ``model_context_window`` matters: without it codex logs "Model metadata for `<model>` not
    found. Defaulting to fallback metadata", and its own context management then budgets against the
    wrong number -- the same class of defect that made opencode reserve 32 K of nemotron's window.
    """
    name = bridged_name(model, "codex", force=force)
    if not name:
        return ""
    lines = [f'model_provider = "{CODEX_PROVIDER_ID}"']
    ctx = context_window(model)
    if ctx:
        lines.append(f"model_context_window = {ctx}")
        lines.append(f"model_max_output_tokens = {max_output_tokens(model)}")
    lines += [
        "",
        f"[model_providers.{CODEX_PROVIDER_ID}]",
        'name = "merlin litellm bridge"',
        f'base_url = "{PROXY_BASE}/v1"',
        'wire_api = "responses"',
        f'env_key = "{PROXY_KEY_ENV}"',
        # Transport resilience. A bridged round rides a Bedrock capacity error ("We're currently
        # experiencing high demand") or an SSE stall the way a native round never does, and codex's
        # defaults give up after five reconnects and 300 s of silence -- short enough that a transient
        # provider hiccup ends the round with rc=1 and an empty usage record. These are the CLI's own
        # per-provider knobs; raising them costs nothing when the provider is healthy.
        f"request_max_retries = {REQUEST_MAX_RETRIES}",
        f"stream_max_retries = {STREAM_MAX_RETRIES}",
        f"stream_idle_timeout_ms = {STREAM_IDLE_TIMEOUT_MS}",
        "",
    ]
    return "\n".join(lines)


def codex_model_name(model: str, *, force: bool | None = None) -> str:
    """What to pass to ``codex --model``: the proxy's model_name when bridged, else the native id."""
    return bridged_name(model, "codex", force=force) or model


def claude_env(model: str, *, force: bool | None = None) -> dict:
    """Environment overrides pointing the ``claude`` CLI at the proxy, or {} for a native model.

    ``ANTHROPIC_BASE_URL`` redirects the Messages API; the token is the proxy's master key, NOT an
    Anthropic credential. CLAUDE_CODE_USE_BEDROCK must be OFF on this path -- Bedrock mode speaks the
    Bedrock runtime directly and would bypass the bridge (and only works for Anthropic models anyway).
    """
    name = bridged_name(model, "claude", force=force)
    if not name:
        return {}
    key = proxy_key()
    return {
        "ANTHROPIC_BASE_URL": PROXY_BASE,
        "ANTHROPIC_AUTH_TOKEN": key,
        "ANTHROPIC_API_KEY": key,
        "CLAUDE_CODE_USE_BEDROCK": "",
        "ANTHROPIC_MODEL": name,
    }


def claude_model_name(model: str, *, force: bool | None = None) -> str:
    return bridged_name(model, "claude", force=force) or _MT.resolve(model)


def sandbox_binds() -> list[str]:
    """bwrap args so an in-sandbox agent can reach the proxy on the loopback interface.

    bwrap shares the network namespace unless ``--unshare-net`` is passed, so loopback already works;
    this exists to make the dependency explicit and to carry the key into the sandbox environment.
    """
    return ["--setenv", PROXY_KEY_ENV, proxy_key()]


def record(model: str, harness: str) -> dict:
    """The provenance block a run manifest must carry when the bridge is in play."""
    name = bridged_name(model, harness)
    return {
        "harness": harness,
        "vendor": _vendor(model),
        "model_requested": model,
        "bridged": bool(name),
        "proxy_model_name": name,
        "proxy_base": PROXY_BASE if name else None,
        "context_window": context_window(model),
        "max_output_tokens": max_output_tokens(model),
        # Stated, not inferred: these are the two known asymmetries vs a native run.
        "caveats": ([] if not name else
                    ["no prompt caching through the proxy (native codex served 98% of input from cache)",
                     "harness system-prompt preamble differs per harness (~12K tokens for codex-cli)"]),
    }
