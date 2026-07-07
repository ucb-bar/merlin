# Gaps worth closing — to improve the xDSL and xDSL+CIRCT arms

Beyond what's already built (OOT starter kit proposal, `gen_isa_module`, `gen_rtl_digest`, per-arm
prompts). Ordered by *expected value*, separating "demonstrates the tool's real advantage" from
"efficiency polish."

## A. xDSL (no-CIRCT) arm
1. **Start from a PASSING skeleton, not a blank page** (high value). The framework parser/cmdbuf exist but
   the agent rebuilt them — deeper fix: ship a `merlin.targetgen.oot_starterkit` whose scaffold compiles
   and **passes the matmul capsule out of the box** (interface parse + a minimal matmul lowering +
   schema-valid cmdbuf). The agent then only *adds* ops to a working baseline. Going from "blank→all ops"
   to "matmul-works→add movement/conv" should cut both rounds and generation tokens sharply.
2. **xDSL-native verifiers as a poor-man's pre-screen** (med-high). xDSL has IRDL `verify()`; if the agent
   writes strong op verifiers (operand ranks, tile-to-DIM, legal attrs), malformed IR is caught at
   construction — *before* spike — without CIRCT. This is the legitimate no-CIRCT analogue of structural
   checking; scaffold/encourage it in the starter kit + prompt.
3. **The real xDSL value is unproven here: cross-target reuse** (high, but it's an experiment gap). abc4
   showed xDSL≈C++ *because it's a single target* — the dialect/pass infra overhead ≈ its benefit. xDSL's
   actual advantage (reuse passes/dialect across accelerators) needs a **second target** (e.g. the
   `toynpu` DUT) to show. Single-target benchmarks structurally can't reveal it.

## B. xDSL+CIRCT arm
4. **Generate MORE than the encoder — a facts-driven conv/im2col + config-sequence skeleton** (high).
   `facts.json` already carries `LOOP_CONV_WS` + `LOOP_CONV_WS_CONFIG_1..6` and `LOOP_WS*` codes. conv was
   the single hardest op for *every* arm. A generated conv-im2col + WS-loop skeleton (from the RTL loop
   instructions) would crack it directly — the highest-leverage extension of `gen_isa_module`.
5. **Actually WIRE the CIRCT sim-skip into the live loop** (high — realizes the measured win). Today CIRCT
   is advisory and the in-loop gate is still spike; the 30,000× "skip verilator on CIRCT-reject" benefit
   was only shown *retroactively*. Wire `prescreen()` as a gate: CIRCT-reject ⇒ skip the sim this round,
   CIRCT-clean ⇒ run one sim. That converts the analysis into real iteration speed.
6. **Shrink CIRCT's numeric blind spot** (med). CIRCT is structural-only (the residual that stops it fully
   replacing the sim). Add a facts-derived *numeric-shape* sanity layer (accumulation width, scale/
   activation application, tile arithmetic) — won't certify numerics, but shrinks how often a sim is
   strictly required.
7. **Curb over-iteration** (med — recovers the +60% cost). merlin+CIRCT did 23 self-checks vs 11 for the
   same 25/25. Give an explicit stop signal: CIRCT-clean ∧ spike-pass ⇒ go to barrier, stop poking.

## C. Cross-cutting (validity, not just speed)
8. **N=1 → N≥3 A/B** (blocking for any magnitude claim). Everything cost/time is directional until this.
9. **Airtight isolation** (med). `sandbox=none` ⇒ isolation rests on `chmod 000` + post-run audit, not a
   hard sandbox (bwrap crashes `claude`). A separate-uid/container runner would make it airtight.
10. **Audit: flag oracle-source READS, not just imports** (low-med). The CIRCT arm read `reference.py`
    source in abc4 (cheat-scan passed since it wasn't imported, but reading the oracle *algorithm* is
    borderline). Extend the audit to flag reads of `runtime/{reference,simulator}.py`.
11. **In-loop instrumentation** (low). Log CIRCT-screen wall in-loop; dedup `result` events so retried
    rounds don't double-count think+gen; auto-emit `timing_detailed.json` per run.

## The two highest-leverage, honest moves
- **#4 + #5** (CIRCT arm): generate the conv/sequence skeleton *and* wire the sim-skip gate. Together they
  attack the hardest op and realize the verification-speed win — the clearest way the CIRCT arm beats C++
  on something C++ structurally cannot match.
- **#1** (both xDSL arms): a passing matmul skeleton — the cheapest large cut to rounds/tokens, inherited
  by the CIRCT arm.
Then **#8** (N≥3) to make the margins publishable, and ideally **#3** (a 2nd target) to show xDSL's
reuse — the value abc4's single-target design can't surface.
