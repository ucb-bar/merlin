# Addendum — `merlin_assisted` full suite (read together with the task above)

Everything in the generated task above applies **unchanged**: the runtime-generated scope block is
authoritative for the full admitted public/dev cohort, including its operator/slice cases and the
resource-bounded full-model capstones. The deliverable layout under `submission/`, the 4 CLI
entrypoints, the redacted QA gate, the integrity rules and the final status line are unchanged. This
addendum **only adds** tooling allowances and a provenance ask. It does not relax or change how you
are graded.

## Rules for this arm (in addition to the task's hard rules)

- This is a **fresh, measured `merlin_assisted` run.** Your wall-time, tokens, cost, tool-calls and iteration
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
- `lower_interface_to_target` — emit gemmini-dialect MLIR (parses + `verify()`).
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
   - `python -c "from merlin.kernels.cca_contract import check_bijection; print(check_bijection('gemmini'))"`
   - `python -c "from merlin.kernels.action_catalog import escalation_ladder; print(escalation_ladder('spatial.dataflow','gemmini'))"`
2. **Scaffold generators** (`targetgen/synthesize/`, `targetgen/generate/`) — invoke one and print
   the generated paths, so the measured transcript records what the tooling produced.

Beyond that, use them where they help you author or debug faster. Do not perform discovery you do
not need: repeated ceremony is measured as effort and counts against this arm.

## Provenance ask

After you converge (or stop), fill in `MERLIN_PROVENANCE_TEMPLATE.md` (in your working directory)
and save it as `submission/docs/merlin_provenance.md`. It records which tools you used, what you
generated with them, which failures they helped diagnose, and confirms your final artifact imports
**no** Merlin runtime code. It is not graded for correctness; it documents *how* the tooling did (or
did not) help, which is the whole point of the comparison.
