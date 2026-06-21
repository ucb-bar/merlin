# abc4 deep analysis — baseline-C++ vs merlin-xDSL vs merlin+CIRCT

**Run:** abc4 (one run per arm), realistic HW-bringup setup: Gemmini RTL + ISA headers + README + 4 example
kernels (matmul / mvin_mvout / conv / padded); self-check tool (spike/verilator/VCS, redacted golden);
"stop when correct" self-pacing; verilator the correctness barrier. Cheat-scan: **all 3 PASS** (no
prior-backend copy, no golden read, no oracle import). **N=1 per arm** — mechanism findings are robust;
cost/wall *magnitudes* are directional (need N>1).

## Headline
All three arms reach **functional + numerical correctness = 25/25 on verilator incl. 5 hidden.** The
must-have is met by every approach. The differences are in *effort* and in *how* they fail along the way.

| Arm | Approach | Verilator 25/25 | Cost | Tokens | Tool-calls | Rounds | Self-checks | Active wall |
|---|---|---|---|---|---|---|---|---|
| baseline | C++ from scratch, real MLIR passes | ✅ | **$33.63** | 15.9M | 115 | 2 | 1 | 47.7 min |
| merlin | xDSL + Merlin framework (IRDL dialect) | ✅ | **$32.11** | 15.1M | 139 | 3 | 11 | 55.2 min |
| merlin+CIRCT | xDSL + CIRCT RTL-checks | ✅ | **$51.25** | 27.5M | 227 | 3 | 23 | 52.8 min |

## A. Where each arm struggles / is strong
- **baseline (C++):** no recorded per-round failures in `_qa_work` — it authored a complete generic
  backend (`Backend.cpp`, 626 L: parses interface → shape-driven tiling, `else if (suf=="movement")`,
  conv/im2col, 0 hardcoded capsule names) and passed broadly on the first graded round. Strength:
  direct, mature MLIR infra. Self-checked only **once** — it was confident and right.
- **merlin (xDSL):** passed almost everything immediately; the **only** struggle was **conv2d (B3/B4)** —
  failed round 1 on `trace_check` (im2col instruction sequence), fixed by round 2. xDSL framework worked
  cleanly; conv is just the hardest op.
- **merlin+CIRCT:** failed **round 1 broadly** (every capsule, `trace_check`) — a single systemic bug
  the CIRCT checks pinpointed: *"use before config: COMPUTE before CONFIG_EX; MVOUT before CONFIG_ST"* +
  *"4 UNKNOWN instructions (fail-closed decode)"*. Round 1 `rtl_checks` = **0/19 ok**; it fixed the
  config-ordering wholesale → round 2 **20/20 ok** → converged. (One A1 movement round also failed
  `numeric_golden` pre-trace.) So CIRCT gave it a *broad, precise structural signal early* — but it then
  iterated 2× as much (23 self-checks) chasing structural cleanliness.

**The hard tail across all arms = conv2d/im2col + movement + config-ordering** — exactly the ops the
example kernels cover but that require non-trivial instruction sequencing.

## B. Progression / effort-to-correct
- **xDSL ≈ C++ on cost** ($32.11 vs $33.63) and tokens — the framework did **not** make authoring more
  expensive; if anything marginally cheaper. This refutes the earlier "xDSL is worse" impression (that was
  the *broken-setup artifact* from abc1–3 where the framework was unusable, not a real result).
- **xDSL did NOT clearly beat C++** either (the user's expectation). On this task the framework's leverage
  (structured IRDL dialect + passes) is offset by the overhead of *wiring* that dialect for a first working
  version, vs C++ targeting mature MLIR directly. Net: a wash on cost, ~equal active wall.
- **merlin+CIRCT cost ~60% more** ($51.25, 27.5M tok, 227 tool-calls, 23 self-checks). The CIRCT feedback
  drove **2× the iteration** — productive (it found the real config-ordering bug) but, since all arms
  converge anyway, on this task it was *extra cost without a correctness payoff*. Its value proposition is
  not "cheaper to converge" — it's the pre-screen speed-up below.

## C. ★ Can CIRCT replace the verilator/VCS check? (the thesis)
Replayed the **exact** live CIRCT screen (`rtl_check_runner.prescreen`) over **all 119 arm×round×capsule**
emitted traces and correlated its verdict against the actual sim outcome:

```
                     sim PASS    sim FAIL
   CIRCT ok/warn        98          0      <- false-clean = 0  (the decisive number)
   CIRCT reject          0         21      <- caught 100% of failures
```
**0 false-clean, 0 false-alarm — perfect separation on 119 points.** But the honest mechanism:
- **All 21 sim-failures were `trace_check` (structural)** — CIRCT's wheelhouse; it caught every one.
- The **single `numeric_golden` failure occurred pre-trace** (lowering failed before emitting a trace), so
  CIRCT couldn't run there — it is **out of CIRCT's scope**, not a CIRCT success.

**Verdict — refined version of the hypothesis:**
- ✅ **CIRCT is a perfect *pre-screen* for structural correctness.** CIRCT-reject ⟹ sim-fail held with no
  exceptions → you can **skip the expensive sim on every CIRCT-reject iteration** (catch the bug in ms via
  static checks instead of minutes of verilator). On abc4 that would skip the sim on **21 of 119**
  iterations with zero risk — the iteration-speed win the user is after.
- ⚠️ **CIRCT cannot *fully* replace the sim as the final correctness gate.** It is structural-only; it
  provably does not check numerics (the one numeric failure was outside its scope). A
  structurally-legal-but-numerically-wrong dialect would pass CIRCT and only the sim would catch it. abc4
  simply contained ~no such case (agents that got structure right got numerics right).
- **Practical recipe:** iterate against CIRCT (instant), skip sim while CIRCT rejects; once CIRCT is clean,
  run **one** sim pass to certify numerics. That's most of the speed benefit while staying sound.

## D. Why didn't xDSL win as expected?
Code-level: the C++ arm wrote a flat, shape-parameterized emitter directly on MLIR (`MlirOptMain` + 2 `.td`
dialects + a 626-line backend) — minimal ceremony, maximal directness. The xDSL arms built a proper
IRDL dialect + rewrite passes (merlin 1 dialect; merlin+CIRCT 3 dialects + a rewrite pattern) — more
*principled* but more *scaffolding* to reach a first passing version. On a single-target, 25-capsule task
that scaffolding cost ≈ what it saved. xDSL's advantage (reuse, multi-target, verified rewrites) would more
plausibly show across **multiple targets** or **larger op sets**, not one accelerator — a good next test.

## E. Recommendations
1. **Adopt CIRCT as the inner-loop pre-screen** (skip sim on reject) — robust, cheap iteration-speed win.
   Keep ≥1 sim pass for numeric certification. Consider adding a CIRCT *numeric-shape* sanity check to
   shrink (not eliminate) the residual numeric gap.
2. **Cost magnitudes need N>1.** Re-run ≥3 pairs to put error bars on the $32 vs $33 vs $51 deltas before
   any "cheaper" claim. (The *mechanism* findings here don't need it.)
3. **HW/accelerator-agnostic "best plan" starter prompt (user idea — strongly supported by the data):** the
   single biggest, arm-independent struggle was **instruction *ordering/config* + the conv-im2col
   sequence**. A starter prompt that front-loads a generic checklist — *"(1) emit all CONFIG_* before first
   use; (2) decode-clean: no UNKNOWN custom-3 forms; (3) movement = mvin→mvout, no compute; (4) conv ⇒
   lower to 2D im2col before matmul; (5) tile to the array DIM"* — would likely have prevented
   merlin+CIRCT's broad round-1 failure and merlin's conv miss, cutting rounds for every arm. This is
   exactly the structural knowledge CIRCT encodes; surfacing it up-front (target-agnostic) is a cheap win.

## Caveats
N=1/arm; abc4 was rate-limit-fragmented + restarted (wall/cost use **active-only** time). Robust:
struggle *locations*, failure *planes*, CIRCT-reject⟹fail, xDSL≈C++-not-worse. Directional (need N>1):
the exact cost deltas, "CIRCT costs 60% more," "xDSL marginally cheaper."
