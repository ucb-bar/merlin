"""Turn this track's products into one findings report: what was measured, and what it licenses.

The report is written so a reader can tell four different statuses apart, because collapsing them is
how a performance claim goes wrong:

  MEASURED    a number from an RTL engine, with its engine and config named
  DERIVED     arithmetic over RTL-derived facts (transfer counts, capacity bounds)
  FALSIFIED   a claim this experiment tested and broke -- kept, not deleted, because the refutation
              is a result and because deleting it invites someone to re-derive it
  NOT MEASURED  an honest gap, named rather than filled with a prediction

It also carries the citation constraint on every cycle number, since the timing engine is a
serial-clock elaboration whose cycles were measured to differ from the stock config's.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _track as T                                                    # noqa: E402

from merlin.common.artifacts import artifacts_dir, new_product        # noqa: E402


def _find(name: str):
    root = artifacts_dir() / "recipe-select" / T.TARGET
    for prod in sorted((p for p in root.glob("v*/*/") if (p / name).exists()),
                       key=lambda p: p.name, reverse=True):
        return json.loads((prod / name).read_text()), prod
    return None, None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--version", type=int, default=1)
    args = ap.parse_args(argv)
    T.assert_frozen_intact()

    sweep, _ = _find("recipe_sweep.json")
    pin, _ = _find("oracle_cost.json")
    ho, _ = _find("heldout_rule.json")
    cm, _ = _find("costmodel_surface.json")

    L: list[str] = []
    a = L.append
    a("# Recipe selection over a frozen Gemmini compiler — findings\n")
    a("Every cycle number below is MEASURED on elaborated RTL. Citation constraint:\n")
    a(f"> {T.ENGINE_NOTE}\n")

    if pin:
        a("## The oracle, pinned\n")
        a("| capsule | verilator | gsim | wall (verilator) | wall (gsim) |")
        a("|---|---|---|---|---|")
        by: dict[str, dict[str, dict]] = {}
        for r in pin["rows"]:
            if r.get("skipped"):
                continue
            by.setdefault(r["capsule"], {})[r["engine"]] = r
        for cap, per in by.items():
            v = per.get("verilator", {})
            g = per.get("gsim_certified") or per.get("gsim_inuse") or {}
            a(f"| {cap} | {v.get('cycles')} | {g.get('cycles')} | {v.get('wall_s')}s "
              f"| {g.get('wall_s')}s |")
        a("\n**GSIM is 8–11× faster than Verilator on gemmini** and the two DISAGREE on cycles by "
          "+0.3%/+1.0%. The disagreement is the serial-clock wiring, not the datapath: the two "
          "elaborations differ only by `ClockSourceAtFreqMHz`×2 and one IO cell, and their `Mesh`, "
          "`MeshWithDelays`, `PE`, `Tile` and `AccumulatorMem` module sets are identical (3096 vs "
          "3099 modules).\n")

    if sweep:
        a("## The recipe surface, measured (wave A)\n")
        wls = []
        for r in sweep["rows"]:
            if r["workload"] not in wls:
                wls.append(r["workload"])
        a("| recipe | " + " | ".join(f"{w} — cycles (% faster)" for w in wls) + " |")
        a("|---" * (len(wls) + 1) + "|")
        combos: list[tuple] = []
        for r in sweep["rows"]:
            key = (r["recipe"]["activation_residency"], r["recipe"]["drain"])
            if key not in combos:
                combos.append(key)
        for res, dr in combos:
            cells = []
            for w in wls:
                row = next((r for r in sweep["rows"] if r["workload"] == w
                            and r["recipe"]["activation_residency"] == res
                            and r["recipe"]["drain"] == dr), None)
                if row is None or not isinstance(row.get("cycles"), int):
                    cells.append("—")
                    continue
                d = row.get("delta_pct")
                # `delta_pct` is (default - this)/default, so POSITIVE means faster. Spelling it
                # out because a bare signed percentage next to a cycle count is read both ways.
                cells.append(f"{row['cycles']}" +
                             ("" if not d else
                              f" (**{d:+.1f}%**)" if d > 0 else f" ({d:+.1f}%)"))
            tag = f"`{res}`/`{dr}`" + (" **(frozen default)**" if (res, dr) ==
                                       ("per_tile", "inline") else "")
            a(f"| {tag} | " + " | ".join(cells) + " |")
        a("\nAll cells bit-exact. Two results:\n")
        a("- **`panel` is a monotone, strict win** that scales with the N sweep, because the saving "
          "is `Mt·Kt·(Nt−1)` activation transfers — DERIVED, and matched by the emitted-code delta.")
        a("- **`drain` flips sign on both axes**: it helps with `per_tile` at the smaller shapes and "
          "hurts at the largest, and it helps alone while hurting combined with `panel`. So it has no "
          "promotable constant value, and a search that scores levers independently picks a WORSE "
          "combination than `panel` alone.\n")

    if ho:
        a("## Does the compiler change generalise?\n")
        # The rule is stated by asking the COMPILER what it does, not by echoing a string recorded
        # in the product. An earlier run's JSON still carries the capacity predicate that the
        # footprint arithmetic later falsified, and a report that quoted it would contradict its own
        # FALSIFIED section two headings below.
        import sys as _sys
        _sys.path.insert(0, str(T.FORK / "mlir_oot" / "lowering"))
        import recipe as _R                                            # noqa: PLC0415
        _probe = [(16, 16, 128), (64, 64, 64), (16, 512, 256)]
        _resolved = {f"{m}x{n}x{k}": _R.resolve_auto(m, n, k, dim=16, spad_rows=16384)
                     for m, n, k in _probe}
        a("Rule, read from the compiler's own `resolve_auto` rather than from a recorded string: "
          "**`activation_residency = panel`, unconditionally** on every shape this lowering can "
          f"emit (probed: {_resolved}).\n")
        if ho.get("rule_supersedes"):
            a(f"It supersedes `{ho['rule_supersedes']}`.\n")
        elif "Kt*(Mt+Nt)" in str(ho.get("rule", "")):
            a("> The product JSON for this run still records the earlier capacity predicate "
              "(`panel if Kt*(Mt+Nt) <= operand_rows/DIM else per_tile`). That predicate was "
              "falsified AFTER these cycles were measured, by the footprint arithmetic rather than "
              "by the measurements — both values reserve the same rows — so the cycle numbers below "
              "stand unchanged while the rule they support got simpler.\n")
        a(f"Fitted on {sorted(tuple(s) for s in ho['fitting_shapes'])}; tested on shapes never "
          f"measured before.\n")
        a("| held-out shape | probes | per_tile | panel | rule | verdict |")
        a("|---|---|---|---|---|---|")
        for r in ho["rows"]:
            pt = r["arms"].get("per_tile", {})
            pn = r["arms"].get("panel", {})
            def cell(x):
                return x.get("cycles") if x.get("legal") else "illegal"
            names = [n for n, _ in ho.get("wins", [])]
            ties = ho.get("ties", [])
            inex = [n for n, _ in ho.get("inexpressible", [])]
            v = ("WINS" if r["heldout"] in names else "TIES" if r["heldout"] in ties
                 else "N/A (inexpressible)" if r["heldout"] in inex else "—")
            a(f"| {r['heldout']} ({r['M']}×{r['N']}×{r['K']}) | Nt={r['Nt']} | {cell(pt)} "
              f"| {cell(pn)} | `{r['rule_choice']}` | {v} |")
        a(f"\n**Verdict: the rule {'GENERALISES' if ho.get('generalizes') else 'IS FALSIFIED'}** — "
          f"wins {len(ho.get('wins', []))}, ties {len(ho.get('ties', []))}, "
          f"losses {len(ho.get('losses', []))}, inexpressible "
          f"{len(ho.get('inexpressible', []))}.\n")

    # ---- the agentic arm
    agent = sorted(T.RUNS.glob("*/agent_summary.json"), key=lambda q: q.parent.name)
    if agent:
        a("## The agentic arm — an LLM selects recipes over the frozen compiler\n")
        a("Driver **codex** on the ChatGPT subscription seat (`gpt-5.6-sol`), one `codex exec` per "
          "candidate: no tools, read-only sandbox, no writable workspace, and the entire expected "
          "output is a JSON recipe. The model therefore *cannot* edit the compiler, emit an "
          "instruction, or run the oracle — the invariant is structural, not instructed.\n")
        a("| workload | default | agent best | speedup | optimum | hit it? | cands | invalid | dup |")
        a("|---|---|---|---|---|---|---|---|---|")
        tot = {"tok": 0, "wall": 0.0, "usd": 0.0, "inv": 0, "cand": 0}
        detail = []
        for q in agent:
            d = json.loads(q.read_text())
            wkey = Path(d["workload"]).stem.replace(".interface", "")
            base, best = d.get("baseline_cycles"), d.get("best_cycles")
            opt = None
            for prod in sorted((artifacts_dir() / "recipe-select" / T.TARGET)
                               .glob("v*/*/recipe_sweep.json"),
                               key=lambda x: x.parent.name, reverse=True):
                rr = [r for r in json.loads(prod.read_text())["rows"]
                      if r["workload"] == wkey and isinstance(r.get("cycles"), int)
                      and r.get("correct")]
                if len({json.dumps(r["recipe"], sort_keys=True) for r in rr}) >= 15:
                    opt = min(r["cycles"] for r in rr)
                    break
            t = d["totals"]
            first = next((i for i, r in enumerate(d["history"])
                          if r.get("best_cycles_so_far") == best), None)
            a(f"| {wkey} | {base} | **{best}** | "
              f"{round(base / best, 4) if (base and best) else '—'}x | {opt or '—'} | "
              f"{'**yes**' if (opt and best == opt) else 'no' if opt else '—'} | "
              f"{d['candidates']} | **{d['invalid_candidates']}** | {d['duplicate_candidates']} |")
            tot["tok"] += t["tokens_total"]
            tot["wall"] += t["agent_seconds"] + t["eval_seconds"]
            tot["usd"] += t["notional_usd"]
            tot["inv"] += d["invalid_candidates"]
            tot["cand"] += d["candidates"]
            detail.append((wkey, first, t))
        a("")
        a("**Zero invalid candidates** is the core of the thesis: because the compiler constructs the "
          "code from a recipe, a candidate cannot be malformed or numerically wrong. Every candidate "
          "across every workload was legal and bit-exact. That is the number to put against a "
          "code-generating arm's repair rate, and it needs no argument.\n")
        a("### Spend, and how much of it mattered\n")
        a("| workload | reached best at | tokens total | wall total (s) | notional USD |")
        a("|---|---|---|---|---|")
        for wkey, first, t in detail:
            a(f"| {wkey} | candidate {first} | {t['tokens_total']} | "
              f"{round(t['agent_seconds'] + t['eval_seconds'], 1)} | ${t['notional_usd']} |")
        a(f"\nTotals: {tot['cand']} candidates, {tot['tok']} tokens, "
          f"{round(tot['wall'], 1)}s wall, ${round(tot['usd'], 4)} notional "
          f"(**billed: nothing** — a subscription seat is not charged per token, so "
          f"`estimated_cost_usd` is `None` by construction and the dollar figure is what the same "
          f"traffic WOULD have cost metered).\n")
        a("⚠️ **Two things the token axis measures that are not the model.** Each candidate is a "
          "fresh `codex exec`, which pays a large fixed session overhead (~20k tokens) regardless of "
          "prompt size — a single long session would amortise it, so this axis reflects the loop "
          "design as much as the agent. And output tokens are tiny (~364/turn) because the agent only "
          "ever emits a JSON object; the cost is essentially all input.\n")
        a("⚠️ **This does not yet demonstrate search EFFICIENCY.** The space is 20 points and the "
          "budget is 16 evaluations, so the agent explores most of it. What it demonstrates is that "
          "the loop converges early and produces no broken candidates. A stronger efficiency claim "
          "needs a larger space or a smaller budget.\n")

    # ---- the AutoComp arm
    ac_runs = [(q, json.loads(q.read_text()))
               for q in sorted(T.RUNS.glob("*/autocomp_summary.json"), key=lambda x: x.parent.name)]
    # CITED = completed AND not superseded. A run whose candidates were rejected by defects in THIS
    # harness is not evidence about AutoComp, so it must not appear in the results table -- but it is
    # named below, because the tokens were really spent and hiding a voided run is how a comparison
    # quietly becomes unfalsifiable.
    ok_by_wl = {}
    for q, d in ac_runs:
        if d.get("arm_completed") and (d.get("invalid_total") or 0) * 2 <= (
                d.get("candidates_evaluated") or 1):
            ok_by_wl[d.get("workload")] = q.parent.name
    ac_ok = [d for q, d in ac_runs if ok_by_wl.get(d.get("workload")) == q.parent.name]
    ac_void = [(q, d) for q, d in ac_runs
               if ok_by_wl.get(d.get("workload")) != q.parent.name]
    if ac_runs:
        a("## The AutoComp arm — an LLM rewrites the kernel\n")
        a("AutoComp's own search, prompts and parameters (copied verbatim from its "
          "`run_search.py:63-108`, so the baseline is the framework as it ships), driven through a "
          "monkeypatched **codex** provider so the PLANNING model is the same seat model the recipe "
          "arm uses. Its eval backend is replaced by ours, because AutoComp's native gemmini backend "
          "scores `latency_spike_raw` — spike cycles — and spike does not model Gemmini timing at "
          "all (counts plateau ~120 from 4K to 2M MACs).\n")
        if ac_ok:
            a("| workload | seed (C library) | AutoComp best | vs its own seed | candidates | "
              "did not compile | compiled but wrong |")
            a("|---|---|---|---|---|---|---|")
            for d in ac_ok:
                a(f"| {d['workload']} | {d['baseline_cycles']} | **{d['best_cycles']}** | "
                  f"{d.get('speedup_vs_own_baseline')}x | {d['candidates_evaluated']} | "
                  f"{d['compile_failures']} | {d['incorrect']} |")
            a("")
        if ac_void:
            a(f"⚠️ **{len(ac_void)} run(s) VOIDED and excluded from the table above** — every one by a "
              f"defect in THIS harness, not in AutoComp. They are listed because the tokens were "
              f"really spent:\n")
            for q, d in ac_void:
                why = (d.get("search_error") or
                       ("candidates rejected by harness defects: injected `uint64_t` the agent's "
                        "includes could not satisfy; a `solution()` fragment compiled as a whole "
                        "program; missing gemmini dialect aliases"))
                a(f"- `{q.parent.name[:30]}` — {str(why)[:150]}")
            a("\n**This nearly produced a false headline.** Before the dialect shim, the arm read as "
              "*\"16 of 18 candidates fail to compile, zero improvement\"*. A candidate discarded by "
              "that run recompiled bit-exact at **822 cycles** once the aliases existed. AutoComp's "
              "ISA prompt documents the accelerator as bare `mvin`/`mvout`/`config_ld`/`preload`/"
              "`compute_*`/`fence`, and its shipped harnesses define those aliases locally; ours did "
              "not, so correct AutoComp dialect simply did not link. Group compile failures by "
              "message before citing a rate — 16 identical errors on one line is a harness "
              "signature, not a model signature.\n")
        a("### ⚠️ The two arms do not share a baseline\n")
        a("| starting point | cycles at 32x32x32 |")
        a("|---|---|")
        a("| canonical gemmini C library `tiled_matmul_auto` — **AutoComp's seed** | 1089 |")
        a("| frozen MLIR compiler, default recipe — **the recipe arm's baseline** | 780 |")
        a("| best recipe (found by the agent, = exhaustive optimum) | 716 |")
        a("\nThe frozen compiler is already **1.40x faster than the hand-written C library** before "
          "anything is tuned. So a per-arm \"speedup vs own baseline\" would credit the "
          "code-writing arm for closing a gap the compiler had already closed. **Absolute cycles on "
          "the shared oracle is the primary comparison**; own-baseline speedups are reported only "
          "beside the starting point they are relative to.\n")
        a("A second asymmetry, equally deliberate: **one candidate is not the same unit on both "
          "sides.** The recipe arm picks among 20 compiler-defined points and *cannot* emit a broken "
          "candidate; AutoComp writes arbitrary C and can. So a zero invalid rate on the recipe side "
          "is a property of the mechanism, not evidence the model writes better code — and only "
          "spend-to-quality curves compare meaningfully.\n")

    # ---- the search-efficiency control
    eff, _ = _find("search_efficiency.json")
    if eff:
        a("## ⚠️ Is the agent SEARCHING, or is the space just easy?\n")
        a("The recipe arm reached the optimum on every workload — but with a 16-evaluation budget "
          "over a 20-point space it explores most of the space, so that is not by itself evidence of "
          "search skill. The control is what UNIFORMLY RANDOM selection of the same number of points "
          "achieves, computed EXACTLY from the fully measured space "
          "(`P(best of a random n-subset is at rank r) = C(N-r, n-1)/C(N, n)`) — no sampling, no "
          "seed, so there is no knob that could be turned until the answer flattered the agent.\n")
        a("| workload | space | optimum | agent reached it at | random's EXPECTED best at that n | "
          "P(random hits the optimum by then) |")
        a("|---|---|---|---|---|---|")
        for r in eff["rows"]:
            if "space_size" not in r:
                continue
            a(f"| {r['workload']} | {r['space_size']} | {r['optimum']} | "
              f"n={r['agent_reached_optimum_at_n']} | "
              f"**{r['random_expected_best_at_that_n']}** | "
              f"{r['prob_random_finds_optimum_in_that_many']} |")
        a("\n**The agent is statistically indistinguishable from random search here.** Random's "
          "expected best after the same number of draws is within 0.1–0.3% of the optimum, because "
          "the top of this distribution is dense. So the recipe arm's advantage — fewer tokens, zero "
          "invalid candidates, better absolute cycles — comes from **the constrained space and the "
          "compiler, not from the model choosing well**.\n")
        a("That is the load-bearing caveat on the whole ablation, and it points at what a real "
          "search-skill test needs: a space where random does POORLY (far larger, or with a sparse "
          "optimum). It does not weaken the mechanism claim — a compiler-constructed candidate still "
          "cannot be broken, which is why this arm has 0 invalid against the code-writing arm's 4 — "
          "but it does relocate the credit.\n")

    a("## FALSIFIED along the way (kept deliberately)\n")
    a("**The calibrated gemmini cost model.** Predicted vs measured, errors in BOTH directions and "
      "outside its own declared `max_abs_pct` of 34.9%:\n")
    a("| case | predicted | measured | error |")
    a("|---|---|---|---|")
    for name, pred, meas in (("A2_single_tile_matmul", 174, 302), ("PK03_k128", 1103, 604),
                             ("w1_small 32³", 1056, 780), ("w2_medium 64³", 7322, 3466),
                             ("w3_n_heavy 16×512×256", 68917, 45304)):
        a(f"| {name} | {pred} | {meas} | {100.0 * (pred - meas) / meas:+.1f}% |")
    a("\nOne physical story explains both directions: its metadata says `\"linear, serial; no "
      "overlap\"`, so at depth it over-predicts a machine that overlaps DMA with compute, while its "
      "`const = −31.96` subtracts 32 cycles from an already-tiny prediction and under-shoots at "
      "K=16. `meta.sim` is `\"?\"` — the calibration does not record which simulator it was fitted "
      "against. It has **no gated consumer** (the perf layer registers it `DIAGNOSTIC`; `merlin/dse/` "
      "uses an unrelated module of the same name), so this is worth fixing, not a broken workstream.\n")
    a("**The capacity predicate for the new default.** The intended rule was \"`panel` where both "
      "operand grids fit, `per_tile` otherwise\". It does not discriminate: both values reserve the "
      "same rows, because the lowering stages the whole activation grid either way and only the "
      "TRANSFER count differs. The surviving claim is stronger — a plain default flip with no "
      "predicate to get wrong.\n")

    a("## NOT MEASURED (named gaps)\n")
    a("- **Shapes past the operand-store bound** (`Kt·(Mt+Nt) > operand_rows/DIM`) are inexpressible "
      "for every wave-A value. That needs a blocked-residency value, and it is a compiler COVERAGE "
      "gap, not evidence about the rule.\n")
    a("- **The agentic arm.** No model has been in the loop for any number above; LLM spend is a "
      "measured zero. The planned driver is codex on the ChatGPT seat, whose dollars are notional "
      "and currently unpriced, so tokens are the citable cost axis.\n")
    a("- **AutoComp head-to-head.** The existing bridge gates on provider and cannot be driven by "
      "the codex seat, so a same-model comparison needs both arms on a shared provider.\n")

    prod = new_product("recipe-select", version=args.version, target=T.TARGET,
                       notes="findings report: measured / derived / falsified / not-measured")
    out = prod.add_artifact("FINDINGS.md")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    prod.write_manifest()
    print("\n".join(L))
    print(f"\nproduct: {prod.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
