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


def _conformance(d: Path) -> dict:
    """The arm-4 conformance flags, as the LAST round recorded them plus whether they EVER held.

    These explained more of the 2026-08 comparison than the score did: the run that scored 20/20 used the
    RTL-derived tooling every round, and the run that scored 0/20 while passing the numeric and trace
    tiers on 17 capsules never touched it and hand-wrote the encoder those tools generate. A table with
    only pass/fail hides the one variable that separated them."""
    st = _y(d / "qa_loop_state.yaml")
    rounds = st.get("rounds") or []
    keys = ("no_regex_ok", "isa_tools_used", "cca_used", "full_selfcheck")
    ever, last = {k: False for k in keys}, {k: None for k in keys}
    for r in rounds:
        checks = ((r.get("conformance") or {}).get("checks") or {})
        for k in keys:
            if checks.get(k):
                ever[k] = True
            if k in checks:
                last[k] = checks.get(k)
    return {"last_round": last, "ever": ever,
            "conformant_rounds": sum(1 for r in rounds
                                     if (r.get("conformance") or {}).get("conformant"))}


def _tier_reach(d: Path) -> dict:
    """PARTIAL CREDIT: how far each GRADED capsule got, not just whether it passed.

    ``0/20`` is the same number for a run that never parsed and a run whose numeric and trace tiers passed
    on most capsules and that lost only on the hardware encoding. The ladder already knows the difference;
    this surfaces it.

    The graded set comes from ``score_capsule.json::per_capsule``, which is the grader's own record of what
    it scored. Walking the result files instead counts every capsule directory left under the run -- for one
    codex run that was 143 entries against a 20-capsule public set, because a wider grading pass had written
    123 more with empty tier blocks."""
    score = d / "grading_public" / "score_capsule.json"
    if not score.is_file():
        return {}
    try:
        per = (json.loads(score.read_text()).get("per_capsule") or [])
    except Exception:  # noqa: BLE001 — a run killed mid-write leaves this truncated
        return {}
    reach: dict = {}
    for c in per:
        tiers = c.get("tiers") or {}
        passed = [t for t, v in tiers.items()
                  if (v.get("status") if isinstance(v, dict) else v) == "pass"]
        top = max(passed) if passed else "none"
        reach[top] = reach.get(top, 0) + 1
    return reach


def _behaviour(d: Path) -> dict:
    """How the run was SPENT: reconnaissance before the first edit, and how much it rewrote afterwards.

    Measured across the campaign: the 20/20 run read for 33 of its 45 actions then made 4 edits; the runs
    that scored zero read for ~9% of their actions and one of them made 114 edits inside a single file.
    That contrast is invisible in tokens and tool-call totals, which were similar or larger for the runs
    that failed."""
    acts: list[str] = []
    rounds = d / "rounds"
    if not rounds.is_dir():
        return {}
    # Codex records actions as completed ITEMS in its own event stream; the CLI-style drivers record them
    # as tool_use blocks. Reading only one shape reported the codex runs as making zero edits, which is
    # both false and exactly backwards from the point the table exists to make.
    codex_streams = sorted(rounds.glob("*.codex_events.timestamped.jsonl"))
    if codex_streams:
        for f in codex_streams:
            for line in f.read_text(errors="ignore").splitlines():
                try:
                    ev = json.loads(line).get("event") or {}
                except Exception:  # noqa: BLE001 — truncated last line on a killed round
                    continue
                if ev.get("type") != "item.completed":
                    continue
                t = (ev.get("item") or {}).get("type")
                if t == "command_execution":
                    acts.append("bash")
                elif t == "file_change":
                    acts.append("edit")
    else:
        for f in sorted(rounds.glob("round_*.transcript.jsonl")):
            for line in f.read_text(errors="ignore").splitlines():
                try:
                    ev = json.loads(line)
                except Exception:  # noqa: BLE001 — a killed round leaves a truncated last line
                    continue
                for blk in (ev.get("message", {}) or {}).get("content", []) or []:
                    if isinstance(blk, dict) and blk.get("type") == "tool_use":
                        acts.append(str(blk.get("name")))
    if not acts:
        return {}
    writes = {"write", "edit", "patch", "apply_patch"}
    first = next((i for i, n in enumerate(acts) if n in writes), None)
    n_write = sum(1 for n in acts if n in writes)
    return {"actions": len(acts), "recon_before_first_write": first,
            "recon_fraction": round(first / len(acts), 3) if first is not None else None,
            "writes": n_write, "invalid_calls": sum(1 for n in acts if n == "invalid"),
            "distinct_tools": len(set(acts))}


def _sink_check(d: Path) -> dict:
    """Cross-check the run's own token accounting against the aet sink it wrote.

    Two surfaces report this campaign's spend and they have disagreed: this table read a GLM-5 run at
    $28.19 over 46.4 M tokens while ``aet spend`` reported the same run as unpriced with zero tokens. The
    sink was present and correct -- the disagreement was in how the other tool grouped the model -- but a
    silent disagreement between two cost surfaces is how a wrong dollar figure ends up quoted.

    So the check is local and does not depend on the other tool: read the sink's own
    ``gen_ai.usage.input_tokens`` and compare it to the input tokens this run recorded. A missing sink, or
    a sink that disagrees by more than a rounding margin, is reported rather than averaged away."""
    logs = d / "logs" / "metrics.jsonl"
    if not logs.is_file():
        return {"sink_present": False}
    total = 0
    for line in logs.read_text(errors="ignore").splitlines():
        try:
            m = json.loads(line)
        except Exception:  # noqa: BLE001 — a killed run truncates the last line
            continue
        if m.get("name") == "gen_ai.usage.input_tokens":
            total += int(m.get("value") or 0)
    return {"sink_present": True, "sink_input_tokens": total}


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
                "sink": _sink_check(d),
                "conformance": _conformance(d),
                "tier_reach": _tier_reach(d),
                "behaviour": _behaviour(d),
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
    # PARTIAL CREDIT. Two runs can both read 0/20 while one never parsed and the other passed the numeric
    # and trace tiers on most capsules and lost only on the hardware encoding. The ladder already knows the
    # difference, so print it rather than let a single pass/fail number flatten it.
    L += ["", "## how far capsules got (highest tier reached, summed over runs)", ""]
    for r in sorted(rows, key=lambda x: str(x.get("model"))):
        reach = r.get("tier_reach") or {}
        if reach:
            L.append(f"- `{r['run_id']}` ({r.get('model')}): "
                     + ", ".join(f"{k}={v}" for k, v in sorted(reach.items())))

    # CONFORMANCE. On this arm these flags separated the runs more sharply than the score did: the 20/20
    # run used the RTL-derived tooling every round; a 0/20 run that reached the simulator never touched it.
    L += ["", "## arm conformance (did the run use the RTL-derived tooling it was given?)", "",
          "| run | model | rounds conformant | isa_tools | cca | no-regex | full self-check |",
          "|---|---|---|---|---|---|---|"]
    def _m(v):
        return {True: "yes", False: "no", None: "—"}.get(v, str(v))
    for r in sorted(rows, key=lambda x: str(x.get("model"))):
        c = r.get("conformance") or {}
        ever = c.get("ever") or {}
        if not ever:
            continue
        L.append(f"| `{r['run_id']}` | `{r.get('model')}` | {c.get('conformant_rounds', 0)} | "
                 f"{_m(ever.get('isa_tools_used'))} | {_m(ever.get('cca_used'))} | "
                 f"{_m(ever.get('no_regex_ok'))} | {_m(ever.get('full_selfcheck'))} |")

    # HOW THE RUN WAS SPENT. Token and tool-call totals were similar or larger for the runs that failed;
    # what differed was the shape -- how long they investigated before editing, and how much they rewrote.
    L += ["", "## how the run was spent", "",
          "| run | model | actions | recon before 1st write | writes | invalid calls | distinct tools |",
          "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: str(x.get("model"))):
        b = r.get("behaviour") or {}
        if not b:
            continue
        frac = b.get("recon_fraction")
        L.append(f"| `{r['run_id']}` | `{r.get('model')}` | {b.get('actions')} | "
                 f"{b.get('recon_before_first_write')}"
                 f"{f' ({frac:.0%})' if isinstance(frac, float) else ''} | "
                 f"{b.get('writes')} | {b.get('invalid_calls')} | {b.get('distinct_tools')} |")

    L += ["", "## where capsules die (first failure plane, summed over runs)", ""]
    for name in sorted(models):
        planes = models[name]["planes"]
        L.append(f"- `{name}`: " + (", ".join(f"{k}={v}" for k, v in sorted(planes.items())) or "-"))
    metered = sum(m["metered_cost_usd"] for m in models.values())
    notional = sum(m["notional_cost_usd"] for m in models.values())
    unknown = sum(m["unknown_billing_cost_usd"] for m in models.values())
    # RECONCILIATION. Every run's telemetry sink is compared against the run's own accounting, so a
    # missing or disagreeing sink is visible before a dollar figure is quoted from either surface.
    bad = []
    for r in rows:
        sk = r.get("sink") or {}
        if not sk.get("sink_present"):
            bad.append((r["run_id"], "no aet sink written (MERLIN_AET_SINK unset?)"))
            continue
        own, sink = r.get("tokens_input") or 0, sk.get("sink_input_tokens") or 0
        if own and sink and abs(own - sink) > max(1000, 0.01 * own):
            bad.append((r["run_id"], f"sink {sink:,} vs run {own:,} input tokens"))
    L += ["", "## telemetry reconciliation", ""]
    L += ([f"- ⚠️ `{rid}`: {why}" for rid, why in bad] if bad
          else ["- every run wrote a sink and its token totals agree with the run's own accounting"])

    L += ["", "## spend", "",
          f"- metered (counts against the budget ceiling): **${metered:.2f}**",
          f"- notional (subscription; tokens real, dollars not billed per-token): ${notional:.2f}",
          f"- billing mode unrecorded (older runs; NOT added to the metered total): ${unknown:.2f}"]
    L += ["", "## caveats", "",
          "- `notional (subscription)` = a ChatGPT/Claude-subscription run: tokens are real, dollars are notional.",
          "- `unpriced` means the price table has no rate for that model - it is NOT zero spend.",
          "- `(lower bound)` = at least one round ended without the provider reporting usage.", ""]
    return "\n".join(L)



# ------------------------------------------------------------------------------------------------
# harness x model
#
# Every capsule-bench number before 2026-08-19 described a (model, HARNESS) PAIR, not a model: the open
# models were only ever driven by opencode and gpt-5.6-sol only ever by codex-cli, so a difference
# between them could not be attributed to either factor. With agent_bridge any model can be driven by
# any harness, and this is the view that separates them: rows are models, columns are harnesses, and
# reading DOWN a column isolates the model while reading ACROSS a row isolates the harness.

def by_cell(rows: list[dict]) -> dict:
    """Group per-run records by (model, driver). The key is the experimental CELL, not the model."""
    out: dict[str, dict] = {}
    for r in rows:
        key = f"{r['model']}::{r.get('driver') or UNKNOWN_MODEL}"
        c = out.setdefault(key, {"model": r["model"], "harness": r.get("driver"), "runs": [],
                                 "best_passed": 0, "n_capsules": None, "best_tier": None,
                                 "metered_usd": 0.0, "notional_usd": 0.0, "tool_calls": 0,
                                 "tokens_total": 0, "bridged": None})
        c["runs"].append(r["run_id"])
        # _score parses "20/20" into n (count passed) and total. Reading `passed` here would take the
        # raw STRING and compare it as an int -- which is how a 20/20 codex cell rendered as 0/20.
        pub = r.get("public") or {}
        got, tot = pub.get("n"), pub.get("total")
        if isinstance(got, int):
            c["best_passed"] = max(c["best_passed"], got)
        if isinstance(tot, int):
            c["n_capsules"] = tot
        tier = r.get("highest_tier")
        if tier and (c["best_tier"] is None or str(tier) > str(c["best_tier"])):
            c["best_tier"] = tier
        c["tool_calls"] += r.get("tool_calls") or 0
        c["tokens_total"] += r.get("tokens_total") or 0
        if r.get("billing_mode") == "subscription_notional":
            c["notional_usd"] += r.get("notional_usd") or 0.0
        else:
            c["metered_usd"] += r.get("cost_usd") or 0.0
        # provenance: was this cell reached through the LiteLLM bridge?
        bj = C.RUNS
        c["bridged"] = c["bridged"] if c["bridged"] is not None else _bridge_of(r["run_id"])
    return out


def _bridge_of(run_id: str) -> bool | None:
    """Read the run's bridge.json, written whenever a (model, harness) pairing needed the proxy."""
    for sub in AAR.RUN_DIRS:
        p = C.RUNS / sub / run_id / "bridge.json"
        if p.is_file():
            try:
                return bool(json.loads(p.read_text()).get("bridged"))
            except Exception:
                return None
    return False


def matrix_markdown(cells: dict) -> str:
    """The harness x model table, plus the two readings that make it evidence."""
    models = sorted({c["model"] for c in cells.values()})
    harnesses = sorted({c["harness"] for c in cells.values() if c["harness"]})
    L = ["## Harness x model", "",
         "Rows are models, columns are harnesses. Reading DOWN a column holds the harness fixed and",
         "varies the model; reading ACROSS a row holds the model fixed and varies the harness. A cell",
         "marked (b) was reached through the LiteLLM bridge, which carries its own caveats (no prompt",
         "caching, a different system-prompt preamble) -- see agent_bridge.", "",
         "| model | " + " | ".join(harnesses) + " |",
         "|---|" + "---|" * len(harnesses)]
    for m in models:
        cs = []
        for h in harnesses:
            c = cells.get(f"{m}::{h}")
            if not c:
                cs.append("—"); continue
            n = c["n_capsules"] or "?"
            tier = c["best_tier"] or "none"
            mark = " (b)" if c["bridged"] else ""
            cs.append(f"{c['best_passed']}/{n} @{tier}{mark}")
        L.append(f"| `{m}` | " + " | ".join(cs) + " |")
    L += ["", "### Cost and effort per cell", "",
          "| cell | runs | actions | tokens | metered | notional |", "|---|---|---|---|---|---|"]
    for k in sorted(cells):
        c = cells[k]
        L.append(f"| `{k}` | {len(c['runs'])} | {c['tool_calls']} | {c['tokens_total']:,} | "
                 f"${c['metered_usd']:.2f} | ${c['notional_usd']:.2f} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", default=None, help="filter run-ids by substring")
    ap.add_argument("--arm", default=None, help="restrict to one arm (e.g. merlin_rtlchecks)")
    ap.add_argument("--out-dir", default=None, help="write here instead of the target's report dir")
    a = ap.parse_args(argv)

    rows = collect(a.tag, a.arm)
    models = by_model(rows)
    cells = by_cell(rows)
    out_dir = Path(a.out_dir) if a.out_dir else C.REPORTS
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"target": C.TARGET, "arm_filter": a.arm, "tag_filter": a.tag,
               "n_runs": len(rows), "models": models, "cells": cells, "runs": rows}
    (out_dir / "by_model.json").write_text(json.dumps(payload, indent=2))
    md = markdown(models, rows, a.arm) + "\n" + matrix_markdown(cells)
    (out_dir / "by_model.md").write_text(md)
    print(f"wrote {out_dir / 'by_model.json'}")
    print(f"wrote {out_dir / 'by_model.md'}")
    print()
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
