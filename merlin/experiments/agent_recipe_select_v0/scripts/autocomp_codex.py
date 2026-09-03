"""Let AutoComp drive the ChatGPT/codex subscription seat, installed by MONKEYPATCH.

WHY THIS EXISTS. AutoComp reaches models through provider APIs (`aws`, `google`, `openai`, `vllm`,
`together`), and its bridge in this repo gates on provider — so the seat cannot drive it, and a
head-to-head against a codex-driven arm would differ in BOTH harness and model. The harness is a
first-order variable here (the same model measured 0/20 on one harness and 15/20 on another), so an
uncontrolled model difference on top of that would make the comparison unreadable.

WHY A MONKEYPATCH AND NOT A PATCH TO THE CHECKOUT. `/scratch/agustin/projects/autocomp` is a clone
with the user's own fork remote and another study's results in it. Editing it would put this
experiment's needs into shared state. `run_autocomp.py` already establishes the pattern by
constructing AutoComp's agents and then DISCARDING its eval backend; this does the same one level
down. Nothing here writes to the AutoComp tree.

WHAT IT PATCHES, AND WHY THAT IS THE WHOLE SURFACE. `LLMClient.chat_async(prompts, num_samples,
temperature, reasoning_effort) -> list[list[str]]` is text-in/text-out with NO tool calling -- I
checked, because a text-only shim cannot serve structured tool calls. It already short-circuits on
`provider == "dummy"` before touching a client, which is exactly where a `codex` provider belongs.

⚠️ THREE DEVIATIONS THIS INTRODUCES, recorded rather than discovered later:

* **no temperature control.** `codex exec` exposes none, so sampling diversity comes only from
  independent invocations. AutoComp's beam search relies on diverse samples, so this is a real
  difference from its API path and must be stated beside any result.
* **num_samples costs num_samples SESSIONS.** Each is a fresh `codex exec` paying a fixed ~20k-token
  session overhead, so a call for 10 samples costs ~200k tokens where an API would batch it. Free in
  dollars on a seat, but it makes the token axis incomparable between arms unless reported per-arm.
* **dollars are notional.** A seat is not billed per token, so `cost_usd` is recorded as 0.0 and the
  list-price projection is kept in a separate field. Letting a projection enter AutoComp's cost
  ledger as real money would consume a budget ceiling nobody is being charged against.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

#: Model string prefix that routes a call to the seat, e.g. "codex/gpt-5.6-sol". Chosen so AutoComp's
#: own model->provider inference never claims it and the tiering stays readable in a run record.
PREFIX = "codex/"

#: The only model this seat actually serves. Probed 2026-09-03 on codex-cli 0.152.0 and again on
#: 0.153.0, where the provider still resolves to `codex` and AutoComp reports every metered key
#: as unavailable (OPENAI/ANTHROPIC/TOGETHER/AWS/GOOGLE) -- so a mis-routed call cannot bill,
#: it can only fail. Every alternate
#: ("gpt-5.6-codex", "gpt-5.6-codex-mini", "o4-mini") returns
#: `Model metadata for ... not found. Defaulting to fallback metadata` and never completes a turn.
#: So a two-tier plan/implement split is NOT available on codex alone -- pair it with a cheap API
#: model for the implementation tier.
SEAT_MODEL = "gpt-5.6-sol"

_STATE: dict = {"calls": 0, "tokens": 0, "notional_usd": 0.0, "home": None, "log": None}


def _codex_home() -> Path:
    """One isolated CODEX_HOME for the whole run. A fresh home changes the cached-token profile, so
    it must be constructed once and identically -- otherwise cache-hit rate varies for a reason
    unrelated to the treatment."""
    if _STATE["home"] is None:
        raise RuntimeError("install() was not called")
    return _STATE["home"]


def _one_sample(prompt: str, model: str, effort: str, timeout: int) -> tuple[str, dict]:
    home = _codex_home()
    ws = home / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CODEX_HOME"] = str(home)
    argv = ["codex", "exec", "--json", "--skip-git-repo-check",
            "-c", "approval_policy=never", "-c", f"model_reasoning_effort={effort}",
            "--sandbox", "read-only", "--cd", str(ws)]
    t0 = time.time()
    try:
        r = subprocess.run(argv, input=prompt, capture_output=True, text=True, env=env,
                           timeout=timeout)
        raw = r.stdout or ""
    except subprocess.TimeoutExpired:
        raw = ""
    dur = time.time() - t0

    text_parts: list[str] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
             "cache_write_tokens": 0}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "item.completed":
            it = e.get("item") or {}
            if it.get("type") == "agent_message" and it.get("text"):
                text_parts.append(it["text"])
        elif e.get("type") == "turn.completed":
            u = e.get("usage") or {}
            # `input_tokens` ALREADY CONTAINS the cached and cache-write buckets on this CLI, so
            # fresh input is a SUBTRACTION. Adding them overstated a measured round by 85%.
            cached = int(u.get("cached_input_tokens", 0) or 0)
            cw = int(u.get("cache_write_input_tokens", 0) or 0)
            usage = {"input_tokens": max(int(u.get("input_tokens", 0) or 0) - cached - cw, 0),
                     "output_tokens": int(u.get("output_tokens", 0) or 0),
                     "cache_read_tokens": cached, "cache_write_tokens": cw}
    usage.update({"model": model, "duration_s": round(dur, 3),
                  # A seat is not billed per token. Zero here is the TRUTH about money spent; the
                  # projection lives beside it and is never summed into a real budget.
                  "cost_usd": 0.0, "billing_mode": "subscription_notional"})
    return "\n".join(text_parts), usage


def install(*, home: Path, log: Path | None = None, effort: str = "high",
            timeout: int = 900, max_parallel: int = 4) -> None:
    """Teach AutoComp's ``LLMClient`` the ``codex`` provider. Idempotent."""
    from autocomp.common import llm_utils as LU

    _STATE.update({"home": Path(home), "log": Path(log) if log else None})
    Path(home).mkdir(parents=True, exist_ok=True)
    auth = Path.home() / ".codex" / "auth.json"
    if auth.exists():
        dst = Path(home) / "auth.json"
        if not dst.exists():
            dst.write_bytes(auth.read_bytes())

    if getattr(LU.LLMClient, "_codex_installed", False):
        return

    orig_init = LU.LLMClient.__init__
    orig_chat_async = LU.LLMClient.chat_async

    def __init__(self, model: str, provider: str | None = None):
        if str(model).startswith(PREFIX) or provider == "codex":
            # Bypass every API-client constructor: there is no client, only a subprocess.
            self.model = str(model)[len(PREFIX):] if str(model).startswith(PREFIX) else str(model)
            self.provider = "codex"
            self.client = self.async_client = None
            self._vllm_api_base = None
            self._last_usage = []
            self._usage_accumulator = []
            import asyncio
            self._loop = asyncio.new_event_loop()
            return
        orig_init(self, model, provider)

    def chat_async(self, prompts_lst, num_samples=10, temperature=None,
                   reasoning_effort="high"):
        if getattr(self, "provider", None) != "codex":
            return orig_chat_async(self, prompts_lst, num_samples=num_samples,
                                   temperature=temperature, reasoning_effort=reasoning_effort)
        eff = reasoning_effort or effort
        # Index BY POSITION, not by value: `prompts_lst.index(p)` misroutes every duplicate prompt
        # onto the first occurrence, and AutoComp's beam legitimately re-asks the same prompt.
        jobs = [(pi, si) for pi in range(len(prompts_lst)) for si in range(num_samples)]
        out: list[list[str]] = [[] for _ in prompts_lst]
        results: dict[tuple[int, int], str] = {}
        with ThreadPoolExecutor(max_workers=max_parallel) as ex:
            futs = {ex.submit(_one_sample, prompts_lst[pi], self.model, eff, timeout): (pi, si)
                    for (pi, si) in jobs}
            for f in futs:
                pi, si = futs[f]
                try:
                    text, usage = f.result()
                except Exception as exc:              # a failed sample is EMPTY, never fabricated
                    text, usage = "", {"model": self.model, "error": str(exc), "cost_usd": 0.0}
                usage.setdefault("phase", _STATE.get("phase") or "codex")
                self._usage_accumulator.append(usage)
                _STATE["calls"] += 1
                _STATE["tokens"] += (int(usage.get("input_tokens", 0) or 0)
                                     + int(usage.get("output_tokens", 0) or 0)
                                     + int(usage.get("cache_read_tokens", 0) or 0))
                if _STATE["log"]:
                    with _STATE["log"].open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"prompt_index": pi, "sample": si,
                                             "usage": usage, "chars": len(text)}) + "\n")
                results[(pi, si)] = text
        for (pi, si) in sorted(results):
            out[pi].append(results[(pi, si)])
        return out

    LU.LLMClient.__init__ = __init__
    LU.LLMClient.chat_async = chat_async
    LU.LLMClient._codex_installed = True


def stats() -> dict:
    return {"calls": _STATE["calls"], "tokens_total": _STATE["tokens"],
            "billed_usd": None, "billing_mode": "subscription_notional",
            "seat_model": SEAT_MODEL,
            "deviations": [
                "no temperature control: codex exec exposes none, so sample diversity comes only "
                "from independent invocations",
                "num_samples costs num_samples fresh sessions, each paying ~20k tokens of fixed "
                "session overhead, so the token axis is not comparable to a batched API arm",
                "dollars are notional: a seat is not billed per token, so cost_usd is 0.0 and the "
                "projection is kept separately",
            ]}
