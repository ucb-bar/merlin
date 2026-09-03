"""Record what this experiment SPENT into the shared aet store, so `aet spend` can see it.

TWO KINDS OF SPEND, and conflating them is how a budget lies:

* **oracle spend** -- simulator invocations, wall seconds and simulated cycles. This is what the
  deterministic phase actually cost, and it is real: 24 GSIM/Verilator runs so far.
* **LLM spend** -- input/output tokens and dollars. For every phase run so far this is **exactly
  zero, measured**, because there is no model in the loop: the sweeps enumerate a 6-point recipe space
  by hand. That zero is logged with ``llm.in_loop=false`` and a stated reason rather than omitted,
  because "no LLM ran" and "we did not measure the LLM" are different claims and only the first is a
  result. This tree has a recurring bug class in which the unmeasured was reported as
  measured-and-zero; this is the inverse discipline.

WHEN THE AGENTIC ARM RUNS the numbers arrive from the driver's own transcript via
:func:`merlin.targetgen.experiment_tokens.parse_agent_transcript`, and the driver is **codex on the
ChatGPT subscription seat**. That has a consequence this module encodes rather than discovers later:
a seat is not billed per token, so ``billing_mode="subscription_notional"`` forces
``estimated_cost_usd`` to ``None`` and keeps any projection in ``cost.subscription_notional_usd``.
Token COUNTS stay real; DOLLARS become a projection, and ``gpt-5.6-sol`` has no entry in
``AET_PRICE_TABLE``, so it currently yields no dollar figure at all. **Tokens are therefore the
defensible axis of the headline plot, not dollars.**
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _track as T                                              # noqa: E402

from merlin.common.artifacts import artifacts_dir, start_run, finish_run   # noqa: E402

#: The driver the agentic arm uses. A seat, so its dollars are notional by construction.
AGENT_DRIVER = "codex"
AGENT_MODEL = "gpt-5.6-sol"
AGENT_BILLING = "subscription_notional"


def _products() -> list[Path]:
    """Every product this experiment has minted, newest first."""
    root = artifacts_dir() / "recipe-select" / T.TARGET
    if not root.exists():
        return []
    return sorted((p for p in root.glob("v*/*/") if (p / "manifest.yaml").exists()),
                  key=lambda p: p.name, reverse=True)


def _load(name: str) -> tuple[dict, Path] | tuple[None, None]:
    for prod in _products():
        f = prod / name
        if f.exists():
            return json.loads(f.read_text()), prod
    return None, None


def oracle_spend() -> dict:
    """Sim invocations, wall seconds and simulated cycles, summed over every measured phase."""
    phases: dict[str, dict] = {}

    sweep, sp = _load("recipe_sweep.json")
    if sweep:
        rows = [r for r in sweep["rows"] if isinstance(r.get("cycles"), int)]
        phases["wave_a_sweep"] = {
            "product": str(sp), "engine": sweep.get("engine"),
            "sim_invocations": len(rows),
            "wall_seconds": round(sum(r.get("wall_s") or 0 for r in rows), 1),
            "cycles_simulated": sum(r["cycles"] for r in rows),
            "all_bit_exact": all(r.get("correct") for r in rows),
        }

    pin, pp = _load("oracle_cost.json")
    if pin:
        rows = [r for r in pin["rows"] if isinstance(r.get("cycles"), int)]
        phases["oracle_pinning"] = {
            "product": str(pp), "engines": sorted({r["engine"] for r in rows}),
            "sim_invocations": len(rows),
            "wall_seconds": round(sum(r.get("wall_s") or 0 for r in rows), 1),
            "cycles_simulated": sum(r["cycles"] for r in rows),
        }

    ho, hp = _load("heldout_rule.json")
    if ho:
        cells = [a for r in ho["rows"] for a in r["arms"].values()
                 if isinstance(a.get("cycles"), int)]
        phases["heldout_generalization"] = {
            "product": str(hp), "engine": ho.get("engine"),
            "sim_invocations": len(cells),
            "wall_seconds": round(sum(c.get("wall_s") or 0 for c in cells), 1),
            "cycles_simulated": sum(c["cycles"] for c in cells),
            "generalizes": ho.get("generalizes"),
        }
    return phases


def agent_spend() -> list[dict]:
    """Per-run LLM spend for every agentic arm, read from its own summary.

    These were NOT in the aet rollup: `run_recipe_agent` opens an aet run for the layout but never
    logged token usage into it, so `aet spend` saw the deterministic phases at a measured zero and
    nothing else -- which reads as "this experiment used no model" when it used ~1M tokens. Rolled up
    here so the store is complete, and it works retroactively on runs already on disk.
    """
    out = []
    # The AUTOCOMP arm. Its runner cannot open an aet run itself (aet is not in AutoComp's venv), so
    # it writes a conventional run dir and this side registers it -- otherwise `aet spend` would show
    # the recipe arm's tokens and silently omit the arm it is being compared against.
    # VOIDING, not deleting. Five early AutoComp runs are invalidated by defects in THIS harness
    # (an injected `uint64_t` the agent's includes could not satisfy; a `solution()` fragment compiled
    # as a whole program; the missing gemmini dialect aliases). Their candidates are not evidence
    # about AutoComp, but they DID consume real tokens, so they stay listed and are excluded from
    # claims with a stated reason -- the pattern `status.py`'s VOID table uses.
    ac_runs = []
    for f in sorted(T.RUNS.glob("*/autocomp_summary.json")):
        d = json.loads(f.read_text())
        ac_runs.append((f, d))
    latest_ok = {}
    for f, d in ac_runs:
        if d.get("arm_completed") and (d.get("invalid_total") or 0) * 2 <= (
                d.get("candidates_evaluated") or 1):
            latest_ok[d.get("workload")] = f.parent.name
    for f, d in ac_runs:
        llm = d.get("llm") or {}
        cited = latest_ok.get(d.get("workload")) == f.parent.name
        void = None
        if not d.get("arm_completed"):
            void = f"search did not run: {(d.get('search_error') or 'unknown')[:70]}"
        elif not cited:
            void = ("superseded by a later run; its candidates were rejected by harness defects "
                    "(injected uint64_t / fragment-as-program / missing gemmini dialect), not by "
                    "AutoComp")
        out.append({
            "cited": cited, "void_reason": void,
            "run_id": f.parent.name, "run_dir": str(f.parent),
            "workload": d.get("workload", ""), "arm": "autocomp",
            "driver": "codex", "model": d.get("plan_model"),
            "code_model": d.get("code_model"),
            "billing_mode": llm.get("billing_mode", "subscription_notional"),
            "candidates": d.get("candidates_evaluated"),
            "invalid": d.get("invalid_total"),
            # AutoComp's per-call usage is aggregated by the codex provider; it does not split the
            # buckets per run, so only the total is claimed and the rest left EMPTY rather than 0.
            "tokens_input": 0, "tokens_cached": 0, "tokens_cache_write": 0, "tokens_output": 0,
            "tokens_total": llm.get("tokens_total", 0),
            "notional_usd": 0.0, "billed_usd": None,
            "agent_seconds": max((d.get("wall_seconds") or 0)
                                 - (d.get("oracle_seconds") or 0), 0.0),
            "eval_seconds": d.get("oracle_seconds", 0.0),
        })
    for f in sorted(T.RUNS.glob("*/agent_summary.json")):
        d = json.loads(f.read_text())
        t = d.get("totals") or {}
        out.append({
            "cited": True, "void_reason": None,
            "arm": "recipe_agent",
            "run_id": f.parent.name, "run_dir": str(f.parent),
            "workload": Path(d.get("workload", "")).stem,
            "driver": d.get("driver"), "model": d.get("model"),
            "billing_mode": d.get("billing_mode"),
            "candidates": d.get("candidates"), "invalid": d.get("invalid_candidates"),
            "tokens_input": t.get("tokens_input", 0), "tokens_cached": t.get("tokens_cached", 0),
            "tokens_cache_write": t.get("tokens_cache_write", 0),
            "tokens_output": t.get("tokens_output", 0),
            "tokens_total": t.get("tokens_total", 0),
            "notional_usd": t.get("notional_usd", 0.0),
            # A seat is not billed per token: None, never 0.0, so a budget rollup cannot count it.
            "billed_usd": None,
            "agent_seconds": t.get("agent_seconds", 0.0),
            "eval_seconds": t.get("eval_seconds", 0.0),
        })
    return out


def _log_agent_runs(rows: list[dict]) -> int:
    """Log each agentic run's REAL token usage into its OWN aet run dir, where it belongs."""
    from aet.tracking.run_logger import EvalRunLogger
    n = 0
    for r in rows:
        rd = Path(r["run_dir"])
        if (rd / "logs" / "metrics.jsonl").exists():
            continue                      # already logged; never double-count
        try:
            lg = EvalRunLogger.start(project="merlin", suite=T.SUITE, target=T.TARGET,
                                     method=("autocomp_codex" if r.get("arm") == "autocomp"
                                             else "recipe_agent_codex"),
                                     seed=0, run_id=r["run_id"],
                                     run_path=rd, tracking_mode="local")
            lg.log_token_usage(input_tokens=r["tokens_input"], output_tokens=r["tokens_output"],
                               cache_creation_tokens=r["tokens_cache_write"],
                               cache_read_tokens=r["tokens_cached"], model=r["model"] or "unknown")
            # cost 0 is the truth about MONEY on a seat; the projection stays a separate metric.
            lg.log_cost(0.0, model=r["model"] or "unknown")
            lg.log_param("billing_mode", r["billing_mode"] or "")
            lg.log_metric("cost.subscription_notional_usd", round(r["notional_usd"], 4))
            lg.log_agent_turns(r["candidates"] or 0)
            lg.log_metric("candidates.invalid", r["invalid"] or 0)
            lg.log_metric("time.agent_seconds", r["agent_seconds"] or 0.0)
            lg.log_metric("time.oracle_seconds", r["eval_seconds"] or 0.0)
            lg.log_param("workload", r["workload"] or "")
            lg.log_param("arm", r.get("arm") or "")
            if r.get("code_model"):
                lg.log_param("code_model", r["code_model"])
            lg.close()
            n += 1
        except Exception as exc:
            print(f"[aet] {r['run_id']}: {type(exc).__name__}: {exc}", file=sys.stderr)
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--method", default="recipe_sweep_deterministic",
                    help="the arm this spend belongs to")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    T.assert_frozen_intact()

    phases = oracle_spend()
    if not phases:
        raise SystemExit("no products found under out/artifacts/recipe-select/ -- nothing to record")

    total_sims = sum(p["sim_invocations"] for p in phases.values())
    total_wall = round(sum(p["wall_seconds"] for p in phases.values()), 1)
    total_cyc = sum(p["cycles_simulated"] for p in phases.values())

    print("=== ORACLE SPEND (real) ===")
    for name, p in phases.items():
        print(f"  {name:<26} sims={p['sim_invocations']:<4} wall={p['wall_seconds']:>8.1f}s "
              f"cycles={p['cycles_simulated']:>9}")
    print(f"  {'TOTAL':<26} sims={total_sims:<4} wall={total_wall:>8.1f}s cycles={total_cyc:>9}")
    ag = agent_spend()
    print(f"\n=== LLM SPEND ===")
    print(f"  deterministic phases above: 0 tokens, 0 turns -- MEASURED zero, no model in the loop.")
    if ag:
        print(f"  {'arm / workload / model':<52}{'cands':>6}{'inv':>5}{'tokens':>10}"
              f"{'notional$':>11}")
        for r in ag:
            label = ("" if r.get("cited") else "VOID ") + \
                f"{r.get('arm', '?')} / {r['workload']} / {r.get('model') or ''}"
            print(f"  {label[:52]:<52}{str(r['candidates']):>6}"
                  f"{str(r['invalid']):>5}{r['tokens_total']:>10}{r['notional_usd']:>11.4f}")
        for r in ag:
            if r.get("void_reason"):
                print(f"    VOID {r['run_id'][:34]}: {r['void_reason'][:100]}")
        cited = [r for r in ag if r.get("cited")]
        print(f"  {'TOTAL (CITED only)':<52}{sum(r['candidates'] or 0 for r in cited):>6}"
              f"{sum(r['invalid'] or 0 for r in cited):>5}"
              f"{sum(r['tokens_total'] for r in cited):>10}"
              f"{sum(r['notional_usd'] for r in cited):>11.4f}")
        print(f"  {'(incl. voided — spent but not evidence)':<52}"
              f"{sum(r['candidates'] or 0 for r in ag):>6}"
              f"{sum(r['invalid'] or 0 for r in ag):>5}"
              f"{sum(r['tokens_total'] for r in ag):>10}"
              f"{sum(r['notional_usd'] for r in ag):>11.4f}")
        print(f"  billed: NOTHING -- {AGENT_DRIVER}/{AGENT_MODEL} on a subscription seat is not "
              f"charged per token, so the dollar column is a projection of metered cost.")
        n = _log_agent_runs(ag) if not args.dry_run else 0
        if n:
            print(f"  logged {n} agentic run(s) into aet (they carried no token usage before)")

    if args.dry_run:
        return 0

    h = start_run(suite=T.SUITE, method=args.method, target=T.TARGET, seed=0,
                  extra={"phases": list(phases), "llm_in_loop": False})
    try:
        from aet.tracking.run_logger import EvalRunLogger
        lg = EvalRunLogger.start(project="merlin", suite=T.SUITE, target=T.TARGET,
                                 method=args.method, seed=0, run_id=str(h.run_id),
                                 run_path=h.run_dir, tracking_mode="local")
        # LLM cost is a MEASURED zero here, and the params say why so a reader of `aet spend`
        # cannot mistake it for an unrecorded run.
        lg.log_token_usage(input_tokens=0, output_tokens=0, cache_creation_tokens=0,
                           cache_read_tokens=0, model="none")
        lg.log_cost(0.0, model="none")
        lg.log_agent_turns(0)
        lg.log_param("llm_in_loop", "false")
        lg.log_param("llm_zero_reason",
                     "deterministic enumeration of a 6-point recipe space; no model was invoked")
        lg.log_param("agentic_arm_driver", AGENT_DRIVER)
        lg.log_param("agentic_arm_model", AGENT_MODEL)
        lg.log_param("agentic_arm_billing_mode", AGENT_BILLING)
        lg.log_param("engine_note", T.ENGINE_NOTE)
        lg.log_metric("oracle.sim_invocations", total_sims)
        lg.log_metric("oracle.wall_seconds", total_wall)
        lg.log_metric("oracle.cycles_simulated", total_cyc)
        for name, p in phases.items():
            lg.log_metric(f"oracle.{name}.sim_invocations", p["sim_invocations"])
            lg.log_metric(f"oracle.{name}.wall_seconds", p["wall_seconds"])
            lg.log_metric(f"oracle.{name}.cycles_simulated", p["cycles_simulated"])
        lg.close()
        emitted = True
    except Exception as exc:                      # telemetry must never fail the experiment
        print(f"[aet] could not log: {type(exc).__name__}: {exc}", file=sys.stderr)
        emitted = False

    (h.run_dir / "spend.json").write_text(json.dumps(
        {"phases": phases, "totals": {"sim_invocations": total_sims, "wall_seconds": total_wall,
                                      "cycles_simulated": total_cyc},
         "llm": {"in_loop": False, "tokens_total": 0, "turns": 0,
                 "reason": "deterministic enumeration; no model invoked",
                 "planned_driver": AGENT_DRIVER, "planned_model": AGENT_MODEL,
                 "planned_billing_mode": AGENT_BILLING,
                 "dollars_note": ("a subscription seat is not billed per token, so dollars are "
                                  "notional; gpt-5.6-sol has no AET_PRICE_TABLE rate, so no dollar "
                                  "figure is available at all -- cite TOKENS")},
         "agentic_arms": ag,
         "aet_emitted": emitted}, indent=1), encoding="utf-8")
    finish_run(h, "ok", summary={"sim_invocations": total_sims, "cycles_simulated": total_cyc,
                                 "llm_tokens": 0})
    print(f"\naet run: {h.run_dir}\n  (aet spend / aet runs will now see this experiment)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
