# Addendum — `merlin_rtlchecks` full suite (read together with the task above)

Everything in the generated task above applies **unchanged**: the runtime-generated scope block is
authoritative for the full admitted public/dev cohort, including its operator/slice cases and the
resource-bounded full-model capstones. The deliverable layout under `submission/`, the 4 CLI
entrypoints, the redacted QA gate, the integrity rules and the final status line are unchanged. This
addendum **only adds** tooling allowances and a provenance ask. It does not relax or change how you
are graded.

## Rules for this arm (in addition to the task's hard rules)

- This is a **fresh, measured `merlin_rtlchecks` run.** Your wall-time, tokens, cost, tool-calls and iteration
  count are recorded as the "effort to converge."
- **Use only files in your working directory.** Do not reach denied paths on the real filesystem and
  do not work around the sandbox. The reference/simulator oracle, hidden capsules, withheld goldens,
  the grader internals and all prior backends are off-limits — access is detected and invalidates
  the run.
- **Do not copy a prior backend.** An after-the-fact similarity audit compares your artifact against
  them.
- **Do not import hidden/reference/oracle functions into your final artifact.** Authoring tools are
  for *authoring*; the shipped package must be **self-contained and integrity-clean** (no
  `import merlin` / `from merlin`, no `merlin.runtime.reference`/`simulator`, no
  `reference_outputs`, no `pipeline.execute`). The integrity scan is the final gate; do not
  self-grade against the true oracle — the redacted `qa/verdict.json` is your only allowed feedback.
- Produce a **self-contained generated artifact** under `submission/` (manifest.yaml + your
  `mlir_oot/` sources + REPORT.md + docs/). It is invoked only through its CLI entrypoints.
- **Iterate** against the redacted public/dev QA verdict until the admitted cohort passes or the
  budget is exhausted. **Hidden grading is post-freeze and hidden repair is disabled** — you never
  see or repair against the hidden capsules.
- Write your final `REPORT.md` and `docs/iteration_notes.md` **from artifacts, not claims** (what the
  entrypoints actually emitted, what the verdict actually said).

## Required generated-package behavior (unchanged from the contract)

Your package must implement, via its declared CLI entrypoints:
- `parse` — parse + verify the `merlin_iface` interface MLIR.
- `lower_interface_to_target` — emit gemmini_universal-dialect MLIR (parses + `verify()`).
- `emit_command_buffer` — schema-valid `command_buffer.json`.
- **An instruction trace** the grader can read: either `emit_instruction_trace`, **or**
  `lower_target_to_llvm` whose RoCC `.insn r 0x7b` inline-asm the shared `rocc_decode` decodes.
- `lower_target_to_llvm` — `llvm.func @gemmini_kernel(...)` of RoCC instructions.

## What this arm adds: Merlin authoring tools

You additionally have read access to Merlin's **authoring** tools (see `ALLOWED_MERLIN_TOOLS.md` in
your working directory for the exact allowed/forbidden surface): `targetgen/synthesize/`,
`targetgen/generate/` (minus `runtime_adapter.py`), `xdsl_dialects/` (minus `lowering/`), and
`targetgen/contract/interface_emit.py`.

These tools are what distinguishes this arm from the raw baseline. **Before your first submission
edit** — once, not once per round — ground yourself with the two cheap discovery calls, then use
whatever else helps:

1. **CCA seam menu** — which compiler sections you may modify, and the next stronger lever per axis:
   - `python -c "from merlin.kernels.cca_contract import check_bijection; print(check_bijection('gemmini_universal'))"`
   - `python -c "from merlin.kernels.action_catalog import escalation_ladder; print(escalation_ladder('spatial.dataflow','gemmini_universal'))"`
2. **Scaffold generators** (`targetgen/synthesize/`, `targetgen/generate/`) — invoke one and print
   the generated paths, so the measured transcript records what the tooling produced.

Beyond that, use them where they help you author or debug faster. Do not perform discovery you do
not need: repeated ceremony is measured as effort and counts against this arm.

## What this arm adds on top: RTL-derived facts

You also have read access to the CIRCT RTL-fact tooling: `merlin/python/merlin/targetgen/rtl/` (the
generators) and `merlin/targets/gemmini_universal/contracts/rtl_facts/` (facts already extracted from THIS
target's elaborated RTL). Your verdict additionally carries an advisory `rtl_checks` block.

**Before your first submission edit** — once, not once per round — add these two to your discovery,
so the ISA you emit is grounded in the RTL rather than in documentation:

3. **RTL-derived levers** for this target:
   - `python -c "from merlin.targetgen import rtl_backend as R; print(R.derived_levers(R.target_profile('gemmini_universal')))"`
4. **RTL facts** (mesh DIM, opcode/funct legality, dtypes, memories) instead of guessing:
   - `python -c "from merlin.targetgen.rtl.facts import load_facts; import json; print(json.dumps(load_facts('gemmini_universal')['facts'],indent=1)[:2000])"`

Read the advisory `rtl_checks` block in `qa/verdict.json` when a capsule fails: it reports FileCheck
over your emitted MLIR and the decoded trace. It is advisory — it never changes your grade.

## THIS DEVICE'S READOUT RESTRICTION — read this before you design a single mvout

This is not a hint about any capsule. It is the machine you are compiling for, and it is stated up
front because the fast oracle you iterate against **cannot tell you about it**.

- **The accumulator readout is NARROW ONLY.** This elaboration sets `acc_read_full_width = false`.
  Its own attested ABI header (`gemmini_params.h`, the one in your `isa_include/` and the one the
  compiled kernel includes) defines `ACC_READ_SMALL_WIDTH` and does **not** define
  `ACC_READ_FULL_WIDTH`. Check it yourself; do not take this paragraph's word for it.
- **The store DMA is 128 bits** = `DIM x int8` (the StreamWriter's `data` port is `UInt<128>` in this
  design's own elaborated FIRRTL). On the sibling `gemmini` elaboration that same port is `UInt<512>`
  = `DIM x int32`. That difference is this whole track.
- **So the only dtype that can leave the accelerator is int8.** The mvout applies the accumulator
  scale, rounds, and clips to `elem_t`. There is no path by which an int32 accumulator row reaches
  memory on this device.
- ⚠️ **THE FUNCTIONAL (loop-tier) ORACLE DOES NOT ENFORCE THIS.** Its full-width accumulator read is
  implemented unconditionally off accumulator-address bit 29 and is *not* gated on
  `ACC_READ_FULL_WIDTH`. Measured 2026-09-09, one ELF on both planes of this device:
  the loop tier reported **PASSED, errors=0, bit-exact int32**; the elaborated RTL returned
  **all zeros, errors=256**. If you set bit 29, the fast oracle will tell you it worked and the
  hardware will hand you zeros. A loop-tier pass is not evidence that your readout is buildable.
- Three more facts derived from this same elaboration, each of which removes programs a sibling
  gemmini executes — so a lowering ported from one will violate them silently:
  - the spatial array is **weight-stationary only** (`dataflow = Dataflow.WS`); the
    output-stationary branch is gated off in the netlist and its propagate registers are `SInt<8>`,
    not accumulator-wide, so no OS program is expressible here;
  - the execute stage can **neither read the accumulator nor write the scratchpad**;
  - the **D operand is hardcoded to `GARBAGE_ADDR`** — do not supply a live one.

Ground every one of these in the artifacts you have been granted (the ABI header, and for the
RTL-facts arm `contracts/rtl_facts/facts.json`) rather than in this list.

## Provenance ask

After you converge (or stop), fill in `MERLIN_PROVENANCE_TEMPLATE.md` (in your working directory)
and save it as `submission/docs/merlin_provenance.md`. It records which tools you used, what you
generated with them, which failures they helped diagnose, and confirms your final artifact imports
**no** Merlin runtime code. It is not graded for correctness; it documents *how* the tooling did (or
did not) help, which is the whole point of the comparison.
