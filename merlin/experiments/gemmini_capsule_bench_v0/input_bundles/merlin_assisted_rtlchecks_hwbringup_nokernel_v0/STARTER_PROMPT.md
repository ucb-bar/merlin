# Starter plan — MERLIN+CIRCT arm  ·  the RTL-facts-driven approach (your edge over hand-derivation)

You have everything the merlin arm has (the typed-dialect starter kit) PLUS the CIRCT pipeline that
extracts the accelerator's ISA + dims + capacities **directly from the RTL**. Two abc4 lessons drive this
plan: (1) the merlin arm regex-scraped the input and forfeited verification — you must parse into the
framework's VERIFIED IR instead; (2) hand-deriving the encoder (~300 LOC) was the source of every
structural bug — generate it from the RTL facts instead.

> **No-kernels condition:** this bundle has **no example kernels** — RTL + ISA header + README only. This
> is where your RTL-derived tooling matters MOST: the other arms must derive the encoding/sequencing blind,
> while you generate `gemmini_isa.py` + `RTL_DIGEST.md` straight from the RTL facts. Lean on (B)+(D) hard.

## A. Use the typed dialect + `validate()` gate (same as the merlin arm — do NOT regex)
Import the starter kit with `/path/to/oscar-merlin/.venv/bin/python`:
- `from merlin.targetgen.oot_starterkit.dialect import parse_to_verified_ir` — parse the input into a typed
  xDSL module that `verify()`s at parse (malformed graph raises immediately, like MLIR). **Never regex the
  input** — that throws away the verifier and is what cost abc4 its round-2 fixes.
- `from merlin.targetgen.oot_starterkit.verify import validate` — `validate(module, cb, trace, gemmini_h)`
  is your per-round compile-time gate (graph verify + cmdbuf schema + ISA-legality). Run it BEFORE any sim.
- `from merlin.targetgen.oot_starterkit import CommandBufferBuilder, transforms` — schema-valid cmdbuf by
  construction; `transforms.im2col` / `transforms.tile_to_dim` for conv/tiling.

## B. Generate your encoder from the RTL facts — don't hand-write it
1. `python -m merlin.targetgen.rtl.gen_isa_module --out submission/mlir_oot/gemmini_isa.py`
   Emits, straight from `facts.json` (extracted from elaborated Gemmini RTL): `CUSTOM_OPCODE`, `FUNCT3`,
   `DIM`, scratchpad/accumulator capacities, the **legal `FUNCT` table** (incl. the `LOOP_CONV_WS*` codes),
   and a `Program` emitter that **rejects illegal functs** and **enforces config-before-use** at
   `finalize()`. Build your RoCC lowering on it → **UNKNOWN instruction** and **use-before-config** become
   structurally impossible, and you skip ~300 LOC of error-prone hand-encoding.
2. `python -m merlin.targetgen.rtl.gen_numeric_facts --out submission/mlir_oot/numeric_facts.py` generates
   an RTL-derived numeric-SHAPE checker (datapath/accumulator widths + scale/activation semantics). Then
   `from numeric_facts import check_numeric_shapes` and run `check_numeric_shapes(cb)` on your command
   buffer to catch numeric-shape bugs (e.g. an accumulator narrower than i32) that the structural CIRCT
   screen can't see — so you need fewer slow numeric sims. It checks SHAPES/WIDTHS only, never values.

## C. Use the per-round CIRCT checks as your fast correctness signal (and skip the sim when it rejects)
Each round you receive the advisory `rtl_checks` block (FileCheck over your decoded trace + numeric bounds
from the RTL facts). On abc4, **CIRCT-reject ⟹ sim-fail with no exceptions across 119 points** — and the
loop now **skips the slow sim on a CIRCT reject** and returns the CIRCT findings as the verdict (a bug
caught in ~7 ms instead of ~222 s). Recipe: iterate against CIRCT (instant) → when CIRCT is clean AND spike
passes, run **one** verilator pass to certify numerics, then stop (don't over-iterate — abc4 self-checked
23 times when 11 sufficed). CIRCT is structural; it does not check numbers — that's what the single
verilator pass + `gen_numeric_facts` are for.

## D. Read the RTL DIGEST instead of crawling the RTL (the #2 token sink)
`python -m merlin.targetgen.rtl.gen_rtl_digest --out RTL_DIGEST.md` → one page: module map, mesh dims, full
legal funct table, memory map/capacities, sequencing rules. **Read RTL_DIGEST.md once; do NOT `find/grep/cat`
the 55 RTL Scala files.** Digest + `gemmini_isa.py` together mean ~zero raw-RTL reading.

## E. Then only the ALGORITHM is left
With encoding from (B), verification from (A), hardware spec from (D), you only write op-LOWERING: tile to
`DIM`, conv⇒`transforms.im2col`, accumulation, scale/activation. Don't rebuild the parser/serializer.

Goal: reach verilator 25/25 in fewer rounds and lower cost than the no-CIRCT arm — by parsing into verified
IR (not regex), generating (not mis-deriving) the ISA, and gating iteration on CIRCT instead of the slow
sim. Stop when correct.
