#!/usr/bin/env python3
"""Every agent run goes through the Codex subscription or Bedrock -- and nothing else.

A standing constraint on this study, enforced here rather than remembered. Two reasons, and only the
first is about money:

**Anthropic and OpenAI models rate-cap quickly on this account.** A capped run does not fail loudly;
it produces short rounds and a small constant score, which reads as a weak agent rather than a
throttled one. That turns a capability comparison into a quota measurement, silently.

**The two routes must stay separable in the ledger.** A Codex seat run is
``subscription_notional`` -- billed_usd is 0 and any dollar figure is a projection -- while a Bedrock
run is ``metered`` real spend. Mixing them into one total makes a projection consume a real budget.

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

#: Routes an agent run may take, and the billing mode each MUST declare.
ALLOWED_ROUTES = {
    "codex": "subscription_notional",   # ChatGPT seat: billed 0, notional projection only
    "converse": "metered",              # Bedrock Converse: real metered spend
    "opencode": "metered",              # Bedrock via OpenCode: real metered spend
}

#: Vendor tokens refused on a metered Bedrock route. Matched against the FIRST dotted segment of the
#: model id, which is where Bedrock puts the vendor -- `qwen.qwen3-...`, `us.anthropic.claude-...`.
#: A substring test would also reject a model whose name merely mentions one of these.
CAPPED_VENDORS = frozenset({"anthropic", "openai"})


def _vendor_of(model_id: str) -> str:
    """Bedrock's vendor segment. `us.anthropic.claude-x` -> anthropic; `qwen.q3` -> qwen.

    A leading region qualifier (`us.`, `eu.`) is skipped, so the vendor is found whether or not the
    id carries one.
    """
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

    if driver not in ALLOWED_ROUTES:
        out.append(
            f"{where}: driver {driver!r} is not an allowed route "
            f"(allowed: {sorted(ALLOWED_ROUTES)}). Every agent run uses the Codex subscription "
            f"or Bedrock."
        )
        return out                      # nothing else can be judged against an unknown route

    expected = ALLOWED_ROUTES[driver]
    if billing != expected:
        out.append(
            f"{where}: driver {driver!r} must declare billing_mode {expected!r}, got {billing!r}. "
            "A seat projection and metered spend must never share a total."
        )

    if expected == "metered":
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
              f"driver={cfg.get('driver')!s:10s} model={cfg.get('model')}")

    if bad:
        print()
        for b in bad:
            print(f"  !! {b}")
        return 1
    print(f"\n  {len(found)} method(s): every agent run routes through the Codex seat or Bedrock.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
