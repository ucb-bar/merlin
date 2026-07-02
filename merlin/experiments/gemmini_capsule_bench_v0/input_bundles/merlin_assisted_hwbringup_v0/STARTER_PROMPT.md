# Starter plan — MERLIN (xDSL framework) arm  ·  optimized for FEWER tokens + faster convergence

You build the backend with the Merlin xDSL framework. The abc4 analysis found this arm left its biggest
levers on the table: it (a) **regex-scraped the input** instead of parsing it into the framework's typed
dialect — which *forfeited* the verification C++ gets for free, so structural bugs surfaced only at grade
time; (b) re-derived the ISA encoding by hand (~300 LOC); and (c) burned round-1 tokens spelunking the RTL
repo (34 of 51 Bash calls were find/grep/cat). The plan below removes all three.

## A. Use the framework's TYPED dialect + verifier — this is the whole point of xDSL (do NOT regex)
The framework ships an OOT starter kit at `merlin.targetgen.oot_starterkit` (import it with the `.venv`
python below). It is hw-agnostic plumbing only — you still author every target lowering — but it gives you
the **compile-time verification that C++/MLIR has and that hand-rolled regex throws away**:

1. **Parse the input into VERIFIED IR — never regex it.**
   `from merlin.targetgen.oot_starterkit import parse_interface` (fixed `merlin_iface` grammar), or for the
   fully-typed path `from merlin.targetgen.oot_starterkit.dialect import parse_to_verified_ir` — this parses
   the input into a typed xDSL module and calls `verify()`, so a malformed graph raises **at parse**, like
   the MLIR verifier. Regex-scraping the input gets you *no* verifier and is exactly the brittle path that
   cost abc4 its round-2 fixes. Defining/using the typed dialect is **not more work** than regex — it is the
   same input, parsed once, and it buys you the structural guarantee for free.
2. **Make `validate()` your per-round compile-time gate.** Before every sim run:
   `from merlin.targetgen.oot_starterkit.verify import validate` → `validate(module, cb, trace, gemmini_h)`
   returns `{ok, findings}`. It runs xDSL's native `module.verify()` (broken graph), the cmdbuf schema, and
   ISA-legality (UNKNOWN funct, use-before-config) — all from PUBLIC info. A non-empty `findings` means fix
   structure *now*, not after a sim. This is your C++-equivalent compile error.
3. **Emit the command buffer with the builder, not by hand:**
   `from merlin.targetgen.oot_starterkit import CommandBufferBuilder` → schema-valid `command_buffer.json`
   by construction (kills the cmdbuf-schema failure plane). Empty command lists are rejected.
4. **Call the generic transforms — don't reinvent them:**
   `from merlin.targetgen.oot_starterkit import transforms` gives `transforms.im2col(...)` (conv→2D matmul,
   the textbook lowering) and `transforms.tile_to_dim(m,n,k,DIM)`. These are target-agnostic; you still map
   the resulting matmul to the RoCC ISA.

The ONLY substantial code you write is the target *lowering algorithm* + the RoCC encoding table. Parse,
verify, cmdbuf serialization, im2col/tiling are provided. Run any tool with
`/scratch/agustin/projects/oscar-merlin/.venv/bin/python`.

## B. Spend FEW tokens — derive once, don't spelunk
1. **Do NOT crawl the RTL repo.** Read `gemmini/README.md` + the ISA header **once**, plus the **one example
   kernel per family** you need. The 55 RTL files are reference-of-last-resort, not a starting point.
2. **Extract the ISA table ONCE** (opcode, funct3, legal funct codes + names) into a single `isa.py` and
   import it everywhere. Re-deriving per op is the biggest avoidable token sink.
3. **Don't spawn subagents** — they multiply context cost.

## C. Converge FASTER — commit to structure before round 1
Round-1 failures in abc4 were avoidable structural bugs that `validate()` would have caught. Up front:
- **config-before-use**: every `CONFIG_*` precedes its first dependent instruction.
- **decode-clean**: only emit legal funct codes (no UNKNOWN custom-3).
- **movement** = mvin→mvout, no compute. **conv2d** ⇒ `transforms.im2col` then matmul. **tile to DIM**;
  respect scratchpad/accumulator capacity.
Write a one-paragraph lowering plan per op-family, implement, then run `validate()` BEFORE any sim.

## D. Iterate on the cheap tier
Once `validate()` is clean, self-check on **spike** (seconds) to iterate; run **verilator** only when spike
passes. Read your own emitted command-buffer + decoded trace to debug — don't guess.

Goal: same correctness (verilator 25/25) at materially lower token/cost and fewer rounds than a naive run,
by parsing into verified IR (not regex) and gating each round on `validate()`. Stop the moment all public
capsules pass.
