# Where the optimization opportunities are — abc4 synthesis

Pulling together the timing, cost, struggle, and CIRCT analyses. The central realization reframes
everything:

> **The agents are bottlenecked on REASONING, not on compute.** ~90% of each agent's active session is
> LLM think+generate; spike is ~0.3 s/run and the CIRCT screen ~7 ms — effectively free. Verilator is
> expensive (~222 s/capsule) but runs operator-side, *outside* the agent loop. So the two optimization
> targets are distinct: **(1) the agent's reasoning/generation time+tokens, and (2) the operator's
> verilator verification cost.** They need different levers.

## Session totals (clean numbers)
| arm | agent-operating (active, excl sleep) | think+gen share | cost | converged |
|---|---|---|---|---|
| baseline-C++ | 47.7 min | ~90% | $33.63 | 25/25 |
| merlin-xDSL | 55.2 min | ~78–90% | $32.11 | 25/25 |
| merlin+CIRCT | 52.8 min | ~93% | $51.25 | 25/25 |
(Calendar span 2.6–4 h is dominated by rate-limit windows + my mid-run restarts — not a clean metric.)

## Target 1 — agent reasoning/generation (the ~50 min + the $32–51 + the tokens)
Because cost ≈ O(tool-calls × growing cached context) and time ≈ think+generate, **anything that makes the
agent explore less and write less code reduces all three at once.** Ranked by measured impact:

| # | Opportunity | Evidence | Lever (built/proposed) | Expected effect |
|---|---|---|---|---|
| 1 | **Stop RTL/spec exploration** | round-1 was the cost peak; **34/51 baseline-merlin bash calls + 67 CIRCT-arm cmds were find/grep/cat** | per-arm `STARTER_PROMPT` (read README+1 example/family once) · **RTL_DIGEST** for CIRCT arm (1 doc vs 55 files) | fewer calls → less generation + smaller context every subsequent call pays for; biggest single token lever |
| 2 | **Stop rebuilding hw-agnostic plumbing** | xDSL wrote **1193 LOC; ~570** was input-parser + cmdbuf + dialect boilerplate that already exists (`interface_emit.parse`) | **OOT starter kit** (provided parser + schema cmdbuf builder + scaffold) | ~half the code not generated → less think+gen, fewer schema/parse failures |
| 3 | **Don't re-derive the ISA encoding** (CIRCT arm) | `rocc.py` ≈ 300 LOC; the 2 structural failure classes were encoding/ordering bugs | **`gen_isa_module`** (RTL-derived encoder; illegal-funct + use-before-config impossible) | removes the largest remaining chunk + its bug-rounds |
| 4 | **Front-load structure to kill fix-rounds** | merlin+CIRCT round-1 failed ALL 20 on config-ordering; merlin lost a round to conv-im2col | the checklist in every STARTER_PROMPT (config-before-use, im2col, movement≠compute, tile-to-DIM) | round 1 emits correct structure → fewer rounds → less of everything |
| 5 | **Curb over-verification** (CIRCT arm) | merlin+CIRCT did **23 self-checks vs 11**, +60% cost, same 25/25 | prompt: trust a clean CIRCT verdict; don't re-spike repeatedly | recover much of the CIRCT arm's cost premium |

## Target 2 — operator verilator verification (~834 s wall / 5541 s compute per full audit)
| # | Opportunity | Evidence | Lever | Expected effect |
|---|---|---|---|---|
| 6 | **CIRCT pre-screen replaces most verilator** | CIRCT 7 ms vs verilator 222 s/capsule (**~30,000×**); 0 false-clean on 119 pts; all 21 fails were structural | gate iteration on CIRCT; run verilator **once** when CIRCT-clean (numeric cert only) | abc4's 21 structural-fail iterations ≈ **78 min sim-compute avoided**; iteration becomes ~instant |
| 7 | **Don't re-verify already-clean capsules** | the audit re-runs all 25 every time | cache verilator pass per (dialect-hash, capsule); only re-sim changed lowerings | cut repeated full-suite audits to the delta |

## The honest headline
- **For agent speed/cost:** the win is *informational* — give the agent the right RTL facts + plumbing up
  front so it **reasons and writes less** (Targets 1.1–1.4). Faster sims would barely help; the agent
  already spends ~nothing on sims.
- **For verification cost:** CIRCT's ~30,000× speed edge + zero false-clean make it a near-ideal
  *structural* pre-screen → skip verilator on rejects, keep one verilator pass for numerics (Target 2.6).
  This is the quantified version of "we mostly only need the CIRCT tool."
- **Net path to beat C++ honestly:** Targets 1.2 (starter kit, both xDSL arms) + 1.3/2.6 (CIRCT moat,
  CIRCT arm only). C++ cannot get 1.3/2.6 (no RTL-facts pipeline) — that's the defensible margin.

## Validation (required before claiming magnitudes — N=1 today)
Run an N≥3 A/B: {baseline, merlin+kit, merlin+kit+CIRCT-tools} × {with, without starter prompt}. Measure
agent-operating time, tokens, rounds, and verilator-runs-avoided. Robust already (mechanism): the 7 ms-vs-
222 s ratio, 0 false-clean, think+gen-dominates, ~570 LOC rebuilt. Directional (needs N>1): the exact
time/cost deltas between arms.
