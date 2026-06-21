# Addendum — `merlin_assisted` pilot (read together with the task above)

Everything in the task above (the shared `TASK_pilot.md`) applies **unchanged**: same four pilot
capsules (`A0_config_smoke`, `A2_single_tile_matmul`, `A4_acc_scale_i8`, `B0_quantized_linear_i8`),
same deliverable layout under `submission/`, same 4 CLI entrypoints, same redacted QA gate, same
integrity rules, same final status line. This addendum **only adds** Merlin-tooling allowances and a
provenance ask. It does not relax or change how you are graded.

## What's different for this arm

You additionally have read access to Merlin's **authoring** tools (see `ALLOWED_MERLIN_TOOLS.md` in
your working directory for the exact allowed/forbidden surface):
`targetgen/synthesize/`, `targetgen/generate/` (minus `runtime_adapter.py`), `xdsl_dialects/` (minus
`lowering/`), and `targetgen/contract/interface_emit.py`. Use them if they help you author or debug the
backend faster.

## Rules for this arm (in addition to the task's hard rules)

- This is a **fresh, measured `merlin_assisted` pilot run.** Your wall-time, tokens, cost, tool-calls,
  and iteration count are recorded as the "effort to converge."
- **Use only files in your working directory.** Do not reach denied paths on the real filesystem, do not
  work around the sandbox. The reference/simulator oracle, hidden capsules, withheld goldens, the grader
  internals, and all prior backends are off-limits — accessing them is detected and invalidates the run.
- **Do not copy a prior backend.** There is no `agent_spec_*`, `hand_smoke_oot`, or `merlin_native_v0`
  to copy, and an after-the-fact similarity audit compares your artifact against them.
- **Do not import hidden/reference/oracle functions into your final artifact.** Merlin tools are for
  *authoring*; the shipped package must be **self-contained and integrity-clean** (no `import merlin` /
  `from merlin`, no `merlin.runtime.reference`/`simulator`, no `reference_outputs`, no
  `pipeline.execute`). The integrity scan is the final gate; do not self-grade against the true oracle —
  the redacted `qa/verdict.json` is your only allowed feedback signal.
- Produce a **self-contained generated artifact** under `submission/` (manifest.yaml + your `mlir_oot/`
  sources + REPORT.md + docs/). It is invoked only through its CLI entrypoints, never imported.
- **Iterate** against the redacted public/dev QA verdict until `A0/A2/A4/B0` all pass or the round
  budget is exhausted. **Hidden grading is post-freeze and hidden repair is disabled** — you never see
  or repair against the hidden capsules; they only prove your backend computes rather than memorizes.
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

## Provenance ask (merlin-arm only)

After you converge (or stop), fill in `MERLIN_PROVENANCE_TEMPLATE.md` (in your working directory) and
save it as `submission/docs/merlin_provenance.md`. It records which Merlin tools you used, what you
generated with them, which failures Merlin tooling helped diagnose, and — importantly — confirms your
final artifact imports **no** Merlin runtime code and is self-contained. This is not graded for
correctness; it documents *how* the Merlin tooling did (or didn't) help, which is the whole point of the
comparison.
