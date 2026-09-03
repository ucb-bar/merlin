"""The agentic arm: an LLM selects recipes over the FROZEN compiler, one decision per round.

THE INVARIANT, ENFORCED BY CONSTRUCTION RATHER THAN BY INSTRUCTION. Each round is one `codex exec`
whose entire output is expected to be a JSON recipe object. The model is given no tools, a read-only
sandbox and no workspace to write into, so it cannot edit the compiler, cannot emit an instruction and
cannot run the oracle itself. The harness builds and evaluates. What is being measured is therefore
the quality and the token cost of DECISIONS, and nothing else.

WHY ONE CALL PER CANDIDATE instead of one long agentic session with shell access. Token accounting
becomes exact (usage is published per `turn.completed`, and one turn is one candidate), a malformed
reply is attributable to a single round, and the "agent may never modify the compiler" property stops
being a promise audited after the fact.

ACCOUNTING. The driver is codex on the ChatGPT subscription seat, so `billing_mode` is
`subscription_notional`: token COUNTS are real, `estimated_cost_usd` is `None` by construction, and any
dollar figure lives in `subscription_notional_usd` as what the same traffic WOULD have cost metered.
Contract RE-VERIFIED live against codex-cli 0.153.0 on this machine 2026-09-03, not assumed
from the 0.152.0 or 0.147.0 records -- a CLI whose JSONL event shape moved would not fail
loudly, it would return an empty reply or zero tokens and record a candidate that cost
nothing. The probe drove this module's own `ask_codex` and `parse_agent_transcript`:
rc=0, a well-formed recipe reply, and usage_complete=True with input 7895 / cached 11136 /
output 23, estimated_cost_usd=None and subscription_notional_usd only. Earlier note:
`turn.completed` carries `usage{input_tokens, cached_input_tokens, cache_write_input_tokens,
output_tokens, reasoning_output_tokens}` and `input_tokens` ALREADY INCLUDES the cached and
cache-write buckets, so fresh input is obtained by subtraction (adding them overstated one measured
round by 85%).

A DUPLICATE RECIPE STILL COSTS TOKENS. Re-proposing an evaluated point is served from the candidate
cache and charged no simulator time, but the round's tokens are counted anyway -- that is the real cost
of a search that revisits, and hiding it would flatter this arm.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _track as T                                                    # noqa: E402
import agent_compile as AC                                            # noqa: E402

from merlin.common.artifacts import start_run, finish_run             # noqa: E402
from merlin.targetgen import experiment_tokens as ET                  # noqa: E402

DRIVER = "codex"
BILLING = "subscription_notional"


def _utc() -> str:
    """UTC stamp in the repo's sortable form."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sims_running() -> int | None:
    """How many target simulators are executing right now, or None if it cannot be determined."""
    try:
        out = subprocess.run(["pgrep", "-fc", "emu_gemmini|simulator-chipyard"],
                             capture_output=True, text=True, timeout=10)
        return int((out.stdout or "0").strip() or 0)
    except Exception:
        return None


def _loadavg() -> float | None:
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except Exception:
        return None

TASK = """You are choosing compiler optimization settings for a frozen Gemmini accelerator backend.

You do NOT write code. You do NOT emit instructions. You choose a RECIPE; the compiler turns it into
machine code and cycle-accurate RTL simulation reports the cycles.

WORKLOAD
{workload}

MACHINE (derived from this target's own RTL)
{machine}

LEGAL CHOICES — you may only use these values. Each is annotated with what it does.
`legal_with_others_default` is that value's verdict holding every OTHER dimension at its default; the
dimensions interact, so a combination is only certainly legal once the compiler accepts it.
{choices}

BASELINE
The compiler's current default recipe is {default}, which measured {baseline} cycles on this workload.
Lower is better. Your goal is the fewest cycles.

WHAT YOU HAVE TRIED SO FAR ({n_tried} of {budget} evaluations used)
{history}

Reply with ONLY a JSON object naming your next recipe, e.g.
{{"activation_residency": "panel", "config_policy": "on_change", "drain": "inline",
  "block_m": "auto", "block_n": "256", "block_k": "128"}}
Every dimension listed above may be set; omitted ones take their default. The block extents are
whole-tile element counts (or "auto" for the compiler's derived cut) and they INTERACT -- the
accumulator bounds the PRODUCT of the M and N blocks -- so a pair that is individually legal can be
jointly illegal, and the build will tell you why.
No prose, no markdown fence, no explanation. Just the JSON object."""


def ask_codex(prompt: str, ws: Path, model: str, effort: str,
              timeout: int) -> tuple[str, Path, float, int]:
    """One codex turn. Returns (reply_text, transcript_path, wall_seconds, returncode)."""
    ws.mkdir(parents=True, exist_ok=True)
    transcript = ws / "turn.jsonl"
    env = dict(os.environ)
    # Isolated CODEX_HOME per arm: a fresh one changes the cached-token profile, so every arm must
    # construct it identically or cache-hit rate differs for a reason unrelated to the treatment.
    env["CODEX_HOME"] = str(ws / "codex_home")
    (ws / "codex_home").mkdir(parents=True, exist_ok=True)
    real_auth = Path.home() / ".codex" / "auth.json"
    if real_auth.exists():
        dst = ws / "codex_home" / "auth.json"
        if not dst.exists():
            dst.write_bytes(real_auth.read_bytes())
    argv = ["codex", "exec", "--json", "--skip-git-repo-check",
            "-c", "approval_policy=never", "-c", f"model_reasoning_effort={effort}",
            "--sandbox", "read-only", "--cd", str(ws)]
    t0 = time.time()
    r = subprocess.run(argv, input=prompt, capture_output=True, text=True, env=env, timeout=timeout)
    wall = time.time() - t0
    transcript.write_text(r.stdout, encoding="utf-8")
    text = []
    for line in r.stdout.splitlines():
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
                text.append(it["text"])
    return "\n".join(text), transcript, wall, r.returncode


def parse_recipe(reply: str) -> tuple[dict | None, str]:
    """Pull the JSON object out of a reply, tolerating a markdown fence but nothing more creative."""
    s = reply.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1] if s.count("```") >= 2 else s
        s = s[4:] if s.lower().startswith("json") else s
        s = s.strip()
    lo, hi = s.find("{"), s.rfind("}")
    if lo == -1 or hi <= lo:
        return None, f"no JSON object in the reply: {reply[:160]!r}"
    try:
        d = json.loads(s[lo:hi + 1])
    except json.JSONDecodeError as exc:
        return None, f"reply is not valid JSON ({exc}): {s[lo:hi + 1][:160]!r}"
    if not isinstance(d, dict):
        return None, "the JSON was not an object"
    return {k: str(v) for k, v in d.items()}, ""


def _codex_version() -> str | None:
    try:
        r = subprocess.run(["codex", "--version"], capture_output=True, text=True, timeout=30)
        return (r.stdout or "").strip() or None
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workload", required=True)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--model", default="gpt-5.6-sol")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--engine", default="gsim")
    ap.add_argument("--turn-timeout", type=int, default=900)
    ap.add_argument("--eval-timeout", type=int, default=3600)
    ap.add_argument("--method", default="recipe_agent_codex")
    args = ap.parse_args(argv)
    T.assert_frozen_intact()
    os.environ.update(T.gsim_env())

    mlir = Path(args.workload).resolve()
    insp = AC.v_inspect(mlir)
    choices = AC.v_choices(mlir)
    R = AC._recipe_mod()
    default = dict(R.DEFAULTS)

    h = start_run(suite=T.SUITE, method=args.method, target=T.TARGET, seed=0,
                  extra={"workload": str(mlir), "driver": DRIVER, "model": args.model,
                         "billing_mode": BILLING, "budget": args.budget,
                         "engine": args.engine, "llm_in_loop": True})
    ws_root = h.run_dir / "turns"

    # Candidate 0 is ALWAYS the frozen default, evaluated by the harness and not by the agent, so
    # every later number has a baseline measured on the same engine in the same run.
    b = AC.v_build(mlir, json.dumps(default), dump=False)
    base = AC.v_evaluate(b["candidate_id"], args.engine, args.eval_timeout)
    baseline = base.get("cycles")
    print(f"baseline (frozen default) = {baseline} cycles, correct={base.get('correct')}", flush=True)

    history: list[dict] = []
    best = baseline if isinstance(baseline, int) else None
    cum_tokens = 0
    cum_agent_s = cum_eval_s = 0.0
    cum_notional = 0.0

    def render_history() -> str:
        if not history:
            return "  (nothing yet — this is your first choice)"
        out = []
        for r in history:
            if not r.get("legal", True):
                out.append(f"  {json.dumps(r['recipe'])} -> REJECTED: {r.get('failure')}")
            elif r.get("cycles") is None:
                out.append(f"  {json.dumps(r['recipe'])} -> no result: {r.get('failure')}")
            else:
                tag = "" if not r.get("duplicate") else " (already tried — no new information)"
                out.append(f"  {json.dumps(r['recipe'])} -> {r['cycles']} cycles, "
                           f"correct={r['correct']}{tag}")
        return "\n".join(out)

    for i in range(args.budget):
        prompt = TASK.format(
            workload=json.dumps({k: insp[k] for k in ("M", "N", "K", "tiles", "macs")}, indent=1),
            machine=json.dumps(insp["machine"], indent=1),
            # `legal_with_others_default` is named in full rather than shortened to `legal`, because
            # the block extents INTERACT: the accumulator bounds the PRODUCT of the M and N blocks, so
            # a value flagged here can still be refused in combination. Calling it `legal` would tell
            # the model something the compiler does not promise.
            choices=json.dumps({d: [{"value": e["value"], "means": e["means"],
                                     "legal_with_others_default": e["legal_with_others_default"],
                                     "n_blocks": e.get("n_blocks")}
                                    for e in v] for d, v in choices["dimensions"].items()}, indent=1),
            default=json.dumps(default), baseline=baseline,
            n_tried=len(history), budget=args.budget, history=render_history())
        ws = ws_root / f"turn_{i:02d}"
        t_turn0 = _utc()
        reply, transcript, agent_s, rc = ask_codex(prompt, ws, args.model, args.effort,
                                                   args.turn_timeout)
        acct = ET.parse_agent_transcript(transcript, driver=DRIVER, model=args.model,
                                         billing_mode=BILLING) if transcript.exists() \
            else {"available": False}
        cum_tokens += int(acct.get("tokens_total") or 0)

        recipe, why = parse_recipe(reply)
        rec: dict = {"candidate": i, "recipe": recipe, "agent_seconds": round(agent_s, 2),
                     "codex_rc": rc, "accounting": acct, "cumulative_tokens": cum_tokens,
                     "prompt_chars": len(prompt), "reply_chars": len(reply),
                     # Harness-side timestamps: no codex event carries one, so this is the only
                     # source of wall-clock ordering.
                     "turn_started_utc": t_turn0, "turn_ended_utc": _utc(),
                     "transcript": str(transcript),
                     # The raw reply, so a malformed round can be audited rather than guessed at.
                     "reply_text": reply,
                     # Load AT THE MOMENT of measurement: cycles are concurrency-invariant, wall
                     # times are not, so a wall number is only interpretable beside this.
                     "sims_running_observed": _sims_running(),
                     "loadavg_1m": _loadavg()}
        if recipe is None:
            rec.update({"legal": False, "failure": f"malformed_reply: {why}", "cycles": None,
                        "correct": False})
        else:
            dup = next((r for r in history if r.get("recipe") == recipe), None)
            built = AC.v_build(mlir, json.dumps(recipe), dump=False)
            if not built.get("built"):
                rec.update({"legal": False, "failure": built.get("reason") or built.get("failure"),
                            "cycles": None, "correct": False})
            else:
                rec["instr_counts"] = built.get("instr_counts")
                rec["n_instructions"] = built.get("n_instructions")
                rec["artifact_digest"] = built.get("artifact_digest")
                rec["vs_default_code"] = built.get("vs_default")
                t0 = time.time()
                ev = AC.v_evaluate(built["candidate_id"], args.engine, args.eval_timeout)
                rec.update({"legal": True, "cycles": ev.get("cycles"),
                            "correct": ev.get("correct"), "failure": ev.get("failure"),
                            "eval_seconds": round(time.time() - t0, 2),
                            "served_from_cache": ev.get("served_from_cache"),
                            "duplicate": dup is not None,
                            "candidate_id": built["candidate_id"]})
                if isinstance(ev.get("cycles"), int) and ev.get("correct"):
                    if best is None or ev["cycles"] < best:
                        best = ev["cycles"]
        # Cumulative series, one value per candidate, so an anytime curve on ANY x axis is a
        # column read rather than a reconstruction.
        cum_agent_s += rec["agent_seconds"]
        cum_eval_s += float(rec.get("eval_seconds") or 0.0)
        cum_notional += float((acct or {}).get("subscription_notional_usd") or 0.0)
        rec["best_cycles_so_far"] = best
        rec["speedup_so_far"] = round(baseline / best, 4) if (best and baseline) else None
        rec["cumulative_agent_seconds"] = round(cum_agent_s, 2)
        rec["cumulative_eval_seconds"] = round(cum_eval_s, 2)
        rec["cumulative_wall_seconds"] = round(cum_agent_s + cum_eval_s, 2)
        rec["cumulative_notional_usd"] = round(cum_notional, 6)
        rec["cumulative_billed_usd"] = None    # a seat is never billed per token, by construction
        history.append(rec)
        print(f"[{i:02d}] {json.dumps(recipe) if recipe else 'MALFORMED':<86} "
              f"cycles={rec.get('cycles')} best={best} "
              f"tok={acct.get('tokens_total')} cum={cum_tokens}", flush=True)

    tot = {k: sum(int(r["accounting"].get(k) or 0) for r in history)
           for k in ("tokens_input", "tokens_cached", "tokens_cache_write", "tokens_output",
                     "tokens_reasoning", "tokens_total")}
    tot["notional_usd"] = round(sum(float(r["accounting"].get("subscription_notional_usd") or 0.0)
                                    for r in history), 4)
    tot["billed_usd"] = None            # a seat is not billed per token, by construction
    tot["agent_seconds"] = round(sum(r["agent_seconds"] for r in history), 1)
    tot["eval_seconds"] = round(sum(r.get("eval_seconds") or 0 for r in history), 1)
    n_invalid = sum(1 for r in history if not r.get("legal"))
    n_dup = sum(1 for r in history if r.get("duplicate"))

    summary = {"workload": str(mlir),
               "shape": {k: insp[k] for k in ("M", "N", "K", "tiles", "macs")},
               "baseline_recipe": default,
               "space_size": choices.get("n_total"), "space_legal": choices.get("n_legal"),
               "budget": args.budget, "effort": args.effort,
               "gsim_emu": str(T.GSIM_EMU), "gsim_sha256": T.GSIM_SHA,
               "codex_version": _codex_version(),
               "driver": DRIVER, "model": args.model,
               "billing_mode": BILLING, "engine": args.engine,
               "baseline_cycles": baseline, "best_cycles": best,
               "speedup_vs_default": round(baseline / best, 4) if (best and baseline) else None,
               "candidates": len(history), "invalid_candidates": n_invalid,
               "duplicate_candidates": n_dup, "totals": tot,
               "engine_note": T.ENGINE_NOTE, "history": history}
    (h.run_dir / "agent_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    print(f"\nbaseline {baseline} -> best {best} "
          f"({summary['speedup_vs_default']}x) in {len(history)} candidates "
          f"({n_invalid} invalid, {n_dup} duplicate)")
    print(f"tokens: {tot['tokens_total']} total ({tot['tokens_input']} fresh in / "
          f"{tot['tokens_output']} out / {tot['tokens_cached']} cached), "
          f"notional ${tot['notional_usd']} (billed: none — seat)")
    finish_run(h, "ok", summary={"best_cycles": best, "tokens_total": tot["tokens_total"],
                                 "candidates": len(history)})
    print(f"run: {h.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
