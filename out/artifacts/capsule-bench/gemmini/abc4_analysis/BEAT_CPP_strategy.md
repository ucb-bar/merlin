# How to beat the from-scratch C++ arm — honestly — on cost + convergence

Goal restated: same must-have (functional+numerical correctness on verilator), but **fewer rounds and
fewer tokens** than a from-scratch C++ MLIR backend. Grounded in the abc4 traces. Two tracks: tooling that
helps the **xDSL (no-CIRCT)** arm (inherited by the CIRCT arm), and the **CIRCT-only** moat on top.

## Where the xDSL arm actually spent its budget (abc4, merlin_abc4: $32, 15.1M tok, 139 tool-calls)
- **Round-1 dominated** (~23.6M of the cache-read tokens). The sink was **exploration**: 34 of 51 round-1
  Bash calls were `find/grep/cat/ls` over the RTL repo + headers. Cost = O(tool-calls × growing context),
  so every exploration call re-pays for the whole accumulated context.
- **Rebuilt hw-agnostic plumbing it didn't need to:** 239 LOC input-IR parser (`iface_ir.py`), 83 LOC
  cmdbuf serializer (`cmdbuf.py`), ~250 LOC dialect boilerplate (`dialects.py`) — ~570 of 1193 LOC.
  Critically, **`interface_emit.parse_interface_mlir()` already exists** and does the parsing — the agent
  just didn't find/trust it.

## Track 1 — xDSL framework tooling (helps no-CIRCT arm; the CIRCT arm inherits it)
A small **provided, tested "merlin OOT starter kit"** — all **target-agnostic** (fixed input grammar +
frozen command-buffer ABI), so it encodes **nothing** about Gemmini (no funct table, no ops) → not
cheating, not overfitting:
1. **`parse_interface(mlir) -> model`** — wrap the existing `interface_emit.parse_interface_mlir`. Removes
   ~240 LOC the agent rebuilt and the `parse`-plane failures.
2. **`CommandBufferBuilder`** that emits **schema-valid** `command_buffer.json` (validates against the
   frozen bench-contract schema). Removes ~83 LOC and **eliminates the `command_buffer_schema` failure
   plane** outright (one of abc4's failure modes).
3. **Dialect + 4-entrypoint scaffold** (CLI wiring + IRDL skeleton) via `targetgen/generate/`. Removes
   ~250 LOC of boilerplate.
Net: ~half the hand-written code disappears → fewer tokens (less to write *and* less to re-read) and fewer
rounds (the removed code was a real bug source). The agent then writes only the **target lowering
algorithm + RoCC encoding** — the genuinely target-specific part.

**Why this beats C++ honestly:** from-scratch C++ must *also* write the parser, cmdbuf serializer, and
dialect scaffold every time (it did, in `Backend.cpp` + `Iface.td`). The framework amortizes exactly that
infra — that is the legitimate, classic reason a framework beats from-scratch. C++ has no equivalent
provided kit here.

**Anti-cheat guardrail (must hold):** the kit handles only the contract-fixed input/output formats +
generic scaffolding. It must contain **zero** target encoding (no funct codes, no Gemmini ops, no
goldens). The target encoding stays agent-derived (merlin) or CIRCT-facts-generated (merlin+CIRCT) — that
preserves both the no-cheat line and the merlin-vs-merlin+CIRCT distinction.

## Track 2 — the CIRCT moat (merlin+CIRCT only, on top of Track 1)
`gen_isa_module` (built + validated) turns CIRCT `facts.json` into a correct RTL-derived ISA encoder:
legal funct table + dims + capacities + a `Program` emitter that **rejects illegal functs** and **enforces
config-before-use** by construction. This removes the last big hand-written chunk (~300 LOC `rocc.py`) and
makes the two abc4 structural failure classes (UNKNOWN instruction, use-before-config) **impossible** —
not just caught. C++ structurally cannot get this (no RTL-facts pipeline). This is the part only the CIRCT
arm gets, and the expected source of a *significant* margin.

## Prompting (Track 0, cheapest)
Per-arm `STARTER_PROMPT.md` (in each bundle): stop spelunking the RTL (read README + one example kernel
per family once), derive/encode once, plan all op-families before round 1 (config-order, im2col,
movement≠compute, tile-to-DIM), iterate on spike + one verilator at the end.

## Expected ranking after these (to validate at N>1, not assert)
- **token/round cost:** merlin+CIRCT (Tracks 0+1+2) < merlin-xDSL (Tracks 0+1) < baseline-C++ < the naive
  abc4 runs. The starter kit should pull *both* xDSL arms below C++; the CIRCT encoder should pull the
  CIRCT arm clearly lowest.
- **correctness:** unchanged (all 25/25) — these reduce *effort*, not the bar.

## Honesty / risks
- N=1 abc4 → these are mechanism-grounded hypotheses; the *margins* need an N>1 A/B (with vs without the
  kit, and merlin vs merlin+CIRCT). Build the kit, re-run ≥3 pairs, then claim magnitudes.
- Watch that the starter kit doesn't quietly do target lowering — keep it format-only; audit the shipped
  package still contains the agent's own lowering + encoding (for merlin) so the comparison stays real.
