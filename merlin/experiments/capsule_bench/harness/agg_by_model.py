#!/usr/bin/env python3
"""Aggregate capsule-bench runs by MODEL — the 1:1 open-vs-commercial comparison.

Every other aggregator in this directory pivots on arm x condition, because the original experiment
varied how much Merlin help an agent got while holding the model fixed. The model-comparison campaign
inverts that: the arm is held fixed (arm-4) and the MODEL varies, so the arm-pivoted tables collapse
every model into one cell and the interesting axis disappears.

Nothing new has to be instrumented — `run_manifest.yaml::model` and
`cost_time_toolcalls.yaml::tokens_native_by_model` already record it. This is a regrouping of facts
already on disk, so it can be re-run over finished runs at any time, costs nothing, and never touches
a run directory.

Honesty rules this file enforces rather than documents:
  * a model whose price is unknown reports ``cost_usd: null`` and is counted as UNPRICED — never $0,
    which would silently understate a budget;
  * a Codex/ChatGPT-subscription run is ``subscription_notional``: its tokens are real but its dollars
    are not metered spend, so it is reported separately and never summed into the metered total;
  * a round whose provider never reported usage (a Codex turn killed by the round timeout) makes the
    run's token total a LOWER BOUND, and the flag says so.

Usage:
  agg_by_model.py [--tag SUBSTR] [--arm merlin_rtlchecks] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C                      # noqa: E402 — active target (descriptor-driven)
import agg_agentic_results as AAR        # noqa: E402 — reuse arm detection + the per-run loader

UNKNOWN_MODEL = "(unrecorded)"


def _y(p: Path) -> dict:
    """Load a YAML file, or {} when it is absent/unreadable. Never raises: a half-written run
    directory must degrade to 'unknown', not abort the whole rollup."""
    if not p.is_file():
        return {}
    try:
        return yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}


def _codex_facts(d: Path) -> dict:
    """Per-round Codex summaries -> billing mode + whether token usage is complete.

    codex-cli reports usage only on ``turn.completed``; a round killed by the round timeout carries
    none. Such a round is UNPRICED, not zero, so the run's totals are a lower bound.
    """
    summaries = sorted((d / "rounds").glob("*.codex_summary.json")) if (d / "rounds").is_dir() else []
    if not summaries:
        return {}
    billing, complete, incomplete = None, 0, 0
    for s in summaries:
        try:
            js = json.loads(s.read_text())
        except Exception:
            continue
        billing = js.get("billing_mode") or billing
        if js.get("usage_complete"):
            complete += 1
        else:
            incomplete += 1
    return {"driver": "codex", "billing_mode": billing,
            "rounds_with_usage": complete, "rounds_without_usage": incomplete,
            "tokens_are_lower_bound": incomplete > 0}


def billing_mode(env: dict, codex: dict) -> str:
    """How this run's dollar figure must be READ.

    ``metered``  — real spend against the budget ceiling (a Bedrock provider).
    ``subscription_notional`` — tokens are real, dollars are not billed per-token (Codex on a ChatGPT
        account, or the Claude subscription). Must never be summed into a metered total.
    ``unknown``  — the run predates provider recording. Reported separately rather than assumed to be
        metered (which would overstate spend) or free (which would understate it).
    """
    if (codex or {}).get("billing_mode") == "subscription_notional":
        return "subscription_notional"
    provider = env.get("provider")
    if provider == "bedrock":
        return "metered"
    if provider == "subscription":
        return "subscription_notional"
    return "unknown"


def _first_planes(man: dict) -> dict:
    for key in ("public_dev", "public"):
        blk = man.get(key)
        if isinstance(blk, dict) and blk.get("first_failure_planes"):
            return blk["first_failure_planes"]
    return {}


def _score(man: dict, key: str) -> dict:
    """The block's score as NUMBERS as well as its printed form.

    ``passed`` is written as the string "20/20", and comparing those lexicographically ranks "5/20"
    above "20/20" because "5" > "2". Parse it once here so every consumer compares counts."""
    blk = man.get(key)
    if not isinstance(blk, dict):
        return {"passed": None, "n": None, "total": None}
    raw = blk.get("passed")
    n = total = None
    if isinstance(raw, str) and "/" in raw:
        a, _, b = raw.partition("/")
        try:
            n, total = int(a.strip()), int(b.strip())
        except ValueError:
            n = total = None
    elif isinstance(raw, int):
        n = raw
    return {"passed": raw, "n": n,
            "total": total if total is not None else (blk.get("n_capsules") or blk.get("total"))}


def collect(tag: str | None, arm_filter: str | None) -> list[dict]:
    """One record per run directory under the active target's capsule-bench run root."""
    audit_p = C.REPORTS / "full_suite_audit.json"
    audit = json.loads(audit_p.read_text()) if audit_p.is_file() else {}
    rows: list[dict] = []
    for sub in AAR.RUN_DIRS:
        base = C.RUNS / sub
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            if not d.is_dir() or (tag and tag not in d.name):
                continue
            arm = AAR._arm_of(d)
            if arm is None or (arm_filter and arm != arm_filter):
                continue
            r = AAR.load_run(d, audit)
            if not r:
                continue
            man = _y(d / "run_manifest.yaml")
            ct = _y(d / "cost_time_toolcalls.yaml")
            env = _y(d / "environment.yaml")
            model = man.get("model") or env.get("model") or UNKNOWN_MODEL
            rows.append({
                "run_id": d.name, "arm": arm, "model": model,
                "bundle_id": env.get("bundle_id"),
                "driver": env.get("driver"), "provider": env.get("provider"),
                "converged": r["converged"], "n_rounds": r["n_rounds"],
                "public": _score(man, "public_dev"), "hidden": _score(man, "hidden"),
                "integrity_status": man.get("integrity_status"),
                "highest_tier": (man.get("public_dev") or {}).get("highest_tier"),
                "oracle_mode": man.get("oracle_mode"),
                "gradeable": man.get("gradeable"),
                "first_failure_planes": _first_planes(man),
                "wall_s": r["wall_s"], "active_wall_s": (ct.get("active_wall_s")),
                "rate_limit_wait_s": ct.get("rate_limit_wait_s"),
                "tool_calls": r["tool_calls"],
                "tokens_total": r["tokens_total"], "tokens_input": r["tokens_input"],
                "tokens_output": r["tokens_output"], "tokens_cached": r["tokens_cached"],
                "tokens_by_model": ct.get("tokens_native_by_model"),
                "cost_usd": r["cost_usd"],
                # A seat run leaves estimated_cost_usd null on purpose; its dollars live here. Without
                # reading it the table printed $0.00 notional for runs that cost $5-8 of equivalent
                # traffic, which reads as "free" rather than "not billed per token".
                "notional_usd": ct.get("subscription_notional_usd"),
                "codex": _codex_facts(d),
                "billing_mode": billing_mode(env, _codex_facts(d)),
            })
    return rows


def by_model(rows: list[dict]) -> dict:
    """Group per-run records by model, keeping metered and subscription spend separate."""
    out: dict[str, dict] = {}
    for r in rows:
        m = out.setdefault(r["model"], {
            "model": r["model"], "n_runs": 0, "runs": [],
            "best_public": None, "best_hidden": None, "best_public_n": -1, "best_hidden_n": -1,
            "metered_cost_usd": 0.0, "unpriced_runs": 0,
            "notional_cost_usd": 0.0, "unknown_billing_cost_usd": 0.0,
            "subscription_notional_runs": 0, "unknown_billing_runs": 0, "notional_unpriced_runs": 0,
            "lower_bound_token_runs": 0,
            "tokens_total": 0, "tool_calls": 0, "planes": {},
        })
        m["n_runs"] += 1
        m["runs"].append(r["run_id"])
        for k, sk in (("best_public", "public"), ("best_hidden", "hidden")):
            blk = r[sk] or {}
            n = blk.get("n")
            if n is None:
                continue
            if m[k] is None or n > m[k + "_n"]:
                m[k] = blk.get("passed")          # keep the printed "20/20" form
                m[k + "_n"] = n                   # rank on the COUNT, never the string
        bm = r["billing_mode"]
        if bm == "subscription_notional":
            m["subscription_notional_runs"] += 1
            m["notional_cost_usd"] += float(r.get("notional_usd") or r["cost_usd"] or 0.0)
            if r.get("notional_usd") is None:
                m["notional_unpriced_runs"] += 1
        elif r["cost_usd"] is None:
            m["unpriced_runs"] += 1          # cost unavailable - NOT zero
        elif bm == "metered":
            m["metered_cost_usd"] += float(r["cost_usd"])
        else:                                 # provider not recorded: keep it OUT of the metered total
            m["unknown_billing_runs"] += 1
            m["unknown_billing_cost_usd"] += float(r["cost_usd"])
        if (r["codex"] or {}).get("tokens_are_lower_bound"):
            m["lower_bound_token_runs"] += 1
        for f in ("tokens_total", "tool_calls"):
            if r[f]:
                m[f] += int(r[f])
        for plane, n in (r["first_failure_planes"] or {}).items():
            m["planes"][plane] = m["planes"].get(plane, 0) + int(n)
    return out


def markdown(models: dict, rows: list[dict], arm: str | None) -> str:
    L = [f"# capsule-bench by model - target `{C.TARGET}`" + (f", arm `{arm}`" if arm else ""), ""]
    L += ["| model | runs | best public | best hidden | rounds | tool calls | tokens | cost |",
          "|---|---|---|---|---|---|---|---|"]
    for name in sorted(models):
        m = models[name]
        rd = [r["n_rounds"] for r in rows if r["model"] == name and r["n_rounds"]]
        if m["subscription_notional_runs"] == m["n_runs"]:
            if m["notional_cost_usd"]:
                cost = f"${m['notional_cost_usd']:.2f} notional (subscription, not billed)"
                if m["notional_unpriced_runs"]:
                    cost += f" (+{m['notional_unpriced_runs']} run(s) predate the rate)"
            else:
                cost = "subscription — not billed per token"
        elif m["unpriced_runs"] == m["n_runs"]:
            cost = "unpriced (cost unavailable - not $0)"
        elif m["unknown_billing_runs"] == m["n_runs"]:
            cost = f"${m['unknown_billing_cost_usd']:.2f} (billing mode unrecorded)"
        else:
            cost = f"${m['metered_cost_usd']:.2f} metered"
            if m["notional_cost_usd"]:
                cost += f" + ${m['notional_cost_usd']:.2f} notional"
            if m["unknown_billing_cost_usd"]:
                cost += f" + ${m['unknown_billing_cost_usd']:.2f} unrecorded"
            if m["unpriced_runs"]:
                cost += f" (+{m['unpriced_runs']} unpriced)"
        tok = f"{m['tokens_total']:,}" + (" (lower bound)" if m["lower_bound_token_runs"] else "")
        L.append(f"| `{name}` | {m['n_runs']} | {m['best_public']} | {m['best_hidden']} | "
                 f"{max(rd) if rd else '-'} | {m['tool_calls']:,} | {tok} | {cost} |")
    L += ["", "## per run (a best-of row hides a run that diverged)", "",
          "| run | model | public | hidden | tier | rounds | tokens | cost |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (x["model"], x["run_id"])):
        c = (f"${r['notional_usd']:.2f} notional" if r.get("notional_usd")
             else (f"${r['cost_usd']:.2f}" if r.get("cost_usd") else "—"))
        L.append(f"| `{r['run_id']}` | `{r['model']}` | {(r['public'] or {}).get('passed')} | "
                 f"{(r['hidden'] or {}).get('passed')} | {r.get('highest_tier') or '—'} | "
                 f"{r['n_rounds']} | {(r['tokens_total'] or 0):,} | {c} |")
    L += ["", "## where capsules die (first failure plane, summed over runs)", ""]
    for name in sorted(models):
        planes = models[name]["planes"]
        L.append(f"- `{name}`: " + (", ".join(f"{k}={v}" for k, v in sorted(planes.items())) or "-"))
    metered = sum(m["metered_cost_usd"] for m in models.values())
    notional = sum(m["notional_cost_usd"] for m in models.values())
    unknown = sum(m["unknown_billing_cost_usd"] for m in models.values())
    L += ["", "## spend", "",
          f"- metered (counts against the budget ceiling): **${metered:.2f}**",
          f"- notional (subscription; tokens real, dollars not billed per-token): ${notional:.2f}",
          f"- billing mode unrecorded (older runs; NOT added to the metered total): ${unknown:.2f}"]
    L += ["", "## caveats", "",
          "- `notional (subscription)` = a ChatGPT/Claude-subscription run: tokens are real, dollars are notional.",
          "- `unpriced` means the price table has no rate for that model - it is NOT zero spend.",
          "- `(lower bound)` = at least one round ended without the provider reporting usage.", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default=None, help="filter run-ids by substring")
    ap.add_argument("--arm", default=None, help="restrict to one arm (e.g. merlin_rtlchecks)")
    ap.add_argument("--out-dir", default=None, help="write here instead of the target's report dir")
    a = ap.parse_args(argv)

    rows = collect(a.tag, a.arm)
    models = by_model(rows)
    out_dir = Path(a.out_dir) if a.out_dir else C.REPORTS
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"target": C.TARGET, "arm_filter": a.arm, "tag_filter": a.tag,
               "n_runs": len(rows), "models": models, "runs": rows}
    (out_dir / "by_model.json").write_text(json.dumps(payload, indent=2))
    md = markdown(models, rows, a.arm)
    (out_dir / "by_model.md").write_text(md)
    print(f"wrote {out_dir / 'by_model.json'}")
    print(f"wrote {out_dir / 'by_model.md'}")
    print()
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
