#!/usr/bin/env python3
"""Every agent run goes through an approved route -- the Codex seat, Bedrock, or the Google API.

A standing constraint on this study, enforced here rather than remembered. Two reasons, and only the
first is about money:

**Anthropic and OpenAI models rate-cap quickly on this account.** A capped run does not fail loudly;
it produces short rounds and a small constant score, which reads as a weak agent rather than a
throttled one. That turns a capability comparison into a quota measurement, silently.

**The routes must stay separable in the ledger.** A Codex seat run is ``subscription_notional`` --
billed_usd is 0 and any dollar figure is a projection -- while Bedrock and the Google API are
``metered`` real spend, on two DIFFERENT credentials and budgets. Summing them makes a projection
consume a real budget and hides which account a cost landed on.

The check is DRIVER-AWARE, because the same model id means different things by route: `gpt-5.6-sol`
is the ChatGPT seat's own model on the codex driver (allowed, and the point of that arm) and an
OpenAI-on-Bedrock inference profile on the converse driver (refused, it caps).

Usage:
    check_method_models.py [--methods <dir>]     # exit 0 = every configured arm is allowed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
EXP = HERE.parent

#: Approved PROVIDERS, and the billing mode each MUST declare. Keyed on provider rather than driver
#: because a driver can reach more than one provider: opencode drives Bedrock and the Google API both,
#: so the driver alone does not say which budget a run spends.
ALLOWED_PROVIDERS = {
    "subscription": "subscription_notional",  # ChatGPT seat: billed 0, notional projection only
    "bedrock": "metered",                     # AWS, bearer auth
    "google": "metered",                      # Google API key -- a SEPARATE credential and budget
}

#: Which drivers can carry which provider. A mismatch here is a launch failure, not a policy call.
PROVIDER_DRIVERS = {
    "subscription": frozenset({"codex"}),
    "bedrock": frozenset({"converse", "opencode"}),
    "google": frozenset({"opencode"}),        # opencode is the multi-provider driver
}

#: Vendor tokens refused on a metered Bedrock route. Matched against the FIRST dotted segment of the
#: model id, which is where Bedrock puts the vendor -- `qwen.qwen3-...`, `us.anthropic.claude-...`.
#: A substring test would also reject a model whose name merely mentions one of these.
CAPPED_VENDORS = frozenset({"anthropic", "openai"})


def _vendor_of(model_id: str) -> str:
    """The vendor an id names, for either id style in use.

    Bedrock writes `us.anthropic.claude-x` (dotted, optional region first); opencode writes
    `google/gemini-x` (provider-slash). Taking the slash form first matters -- splitting
    `google/gemini-3.5-flash` on "." would yield `google/gemini-3` as the vendor and match nothing.

    A leading region qualifier (`us.`, `eu.`) is skipped, so the vendor is found either way.
    """
    if "/" in model_id:
        return model_id.split("/", 1)[0].strip().lower()
    parts = [p for p in model_id.split(".") if p]
    for p in parts:
        low = p.lower()
        if low in ("us", "eu", "apac", "global"):
            continue
        return low
    return ""


def violations(cfg: dict, *, where: str) -> list[str]:
    """Every way one method config breaks the routing policy."""
    out: list[str] = []
    driver = str(cfg.get("driver") or "")
    model = str(cfg.get("model") or "")
    billing = str(cfg.get("billing_mode") or "")
    provider = str(cfg.get("provider") or "")

    if provider not in ALLOWED_PROVIDERS:
        out.append(
            f"{where}: provider {provider!r} is not approved "
            f"(allowed: {sorted(ALLOWED_PROVIDERS)}). Every agent run uses the Codex seat, "
            f"Bedrock, or the Google API."
        )
        return out                      # nothing else can be judged against an unknown route

    if driver not in PROVIDER_DRIVERS[provider]:
        out.append(
            f"{where}: driver {driver!r} cannot carry provider {provider!r} "
            f"(drivers for it: {sorted(PROVIDER_DRIVERS[provider])})"
        )

    expected = ALLOWED_PROVIDERS[provider]
    if billing != expected:
        out.append(
            f"{where}: driver {driver!r} must declare billing_mode {expected!r}, got {billing!r}. "
            "A seat projection and metered spend must never share a total."
        )

    if provider == "bedrock":
        vendor = _vendor_of(model)
        if vendor in CAPPED_VENDORS:
            out.append(
                f"{where}: model {model!r} is a {vendor} model on a metered Bedrock route. "
                "Those rate-cap quickly here, and a capped run looks like a weak agent rather "
                "than a throttled one."
            )
        if not vendor:
            out.append(f"{where}: model {model!r} has no recognisable vendor segment")

    if not model:
        out.append(f"{where}: no model declared")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods", type=Path, default=EXP / "methods")
    a = ap.parse_args(argv)

    if not a.methods.is_dir():
        print(f"no methods dir at {a.methods} -- nothing to check")
        return 0

    found = sorted(a.methods.glob("*/method.yaml"))
    if not found:
        print(f"no method.yaml under {a.methods} -- nothing to check")
        return 0

    bad: list[str] = []
    for p in found:
        cfg = yaml.safe_load(p.read_text()) or {}
        # A methods dir outside the experiment tree is legitimate (a caller pointing at a scratch
        # copy). Falling over on relative_to made the check EXIT 1 BY CRASHING, which reads exactly
        # like a policy violation while reporting none of them.
        try:
            rel = p.relative_to(EXP)
        except ValueError:
            rel = p
        v = violations(cfg, where=str(rel))
        bad += v
        mark = "FAIL" if v else "ok  "
        print(f"  [{mark}] {cfg.get('method', p.parent.name):18s} "
              f"provider={cfg.get('provider')!s:13s} driver={cfg.get('driver')!s:10s} "
              f"model={cfg.get('model')}")

    if bad:
        print()
        for b in bad:
            print(f"  !! {b}")
        return 1
    print(f"\n  {len(found)} method(s): every agent run routes through an approved provider "
          f"({', '.join(sorted(ALLOWED_PROVIDERS))}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
