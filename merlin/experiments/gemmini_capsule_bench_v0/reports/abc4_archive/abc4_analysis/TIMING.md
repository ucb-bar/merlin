# abc4 timing decomposition — exact per-tool, and where to optimize

Two separate budgets (conflating them caused the "verilator was slow?" confusion):

## Budget 1 — AGENT ACTIVE SESSION (time to a working dialect, excl. rate-limit sleep)
| arm | active wall | think+generate (LLM) | spike (in-session) | CIRCT screen | verilator (in-session) |
|---|---|---|---|---|---|
| baseline-C++ | 47.7 min | ~1908 s | 13.5 s / 40 runs (**0.34 s/run**) | 0.3 s / 40 (**8 ms/run**) | **0** (didn't run it) |
| merlin-xDSL | 55.2 min | ~4051 s* | 10.6 s / 38 runs | 0.3 s (7 ms/run) | 0 |
| merlin+CIRCT | 52.8 min | ~4798 s* | 5.2 s / 20 runs | 0.3 s / 39 (7 ms/run) | 0 |

**The agent's active time is ~90% LLM think+generate.** spike is effectively free (~0.3 s/run), the
CIRCT static screen is **~7 ms/run**, and the agents **never ran verilator in-session** — they iterated
on spike and let the operator barrier do verilator. *(\*think+gen sums per-round `result.duration_api_ms`;
rounds that were retried/resumed double-count, so merlin's figure is an upper bound — the point that
think+gen ≫ all tools is robust regardless.)*

## Budget 2 — OPERATOR verilator barrier / audit (separate processes; the EXPENSIVE part)
| | suite wall (8 workers) | sim-compute | **per-capsule verilator** |
|---|---|---|---|
| full 25-capsule audit | 834 s | 5541 s | **≈ 222 s / capsule** |

This is the verilator cost you remembered — it's real and large, but it lives in the **operator-side
barrier + audits**, not in the agent's active session. Each capsule is ~222 s of cycle-accurate RTL.

## The decisive ratio for the CIRCT thesis
**CIRCT screen ≈ 7 ms; verilator ≈ 222 s per capsule → CIRCT is ~30,000× cheaper per check.** Since
CIRCT-reject ⟹ sim-fail held with zero false-clean on abc4, every structural-failure iteration you gate
on CIRCT instead of verilator saves ~222 s for 7 ms. The 21 structural failures CIRCT caught ≈ **78 min of
verilator sim-compute avoidable** by pre-screening — the concrete iteration-speed win.

## Where to optimize (two distinct levers)
1. **Agent wall (~50 min, ~90% think+generate):** the lever is **reduce reasoning/generation**, NOT sim
   speed (sims are already ~free in-session). Cut it by: the per-arm starter prompts (stop exploring), the
   OOT starter kit (don't regenerate the parser/cmdbuf/dialect ≈ half the LOC), and — CIRCT arm — the
   RTL digest + `gen_isa_module` encoder (don't crawl RTL, don't re-derive encoding). Less to explore and
   less to write ⇒ less to think/generate ⇒ shorter wall + fewer tokens.
2. **Verification cost (operator verilator, 222 s/capsule):** the lever is **CIRCT pre-screen** — skip the
   verilator run on every CIRCT-reject (7 ms vs 222 s), and run verilator once when CIRCT is clean (to
   certify numerics, which CIRCT can't). This is the "we mostly only need CIRCT" benefit, quantified.

## Instrumentation status / gap (for future runs)
- **Have, exact:** per-tier per-capsule `{build_s, sim_active_s, oracle_wait_s}` in every
  `capsule_result.json` (spike + verilator wall) + per-round `result.duration_api_ms` (think+gen).
- **Added here:** CIRCT screen wall (measured live via `rtl_check_runner.prescreen`; ~7 ms/run).
- **To add for clean future decomposition:** (a) log CIRCT screen wall *in-loop* (in `qa_check_rtlchecks`)
  rather than retroactively; (b) dedupe `result` events so retried rounds don't double-count think+gen;
  (c) emit a per-run `timing_detailed.json` automatically at run end.
