# Addendum — `merlin_assisted` arm (read together with the task above)

This arm's whole point is to build the backend **WITH the Merlin xDSL framework** — a proper xDSL
dialect + lowering passes — not a hand-rolled string emitter. The shared task above (scope, self-check
tool, verilator/VCS barrier, "stop when correct") applies unchanged; this addendum adds what you may use
and what you must build.

## You have the Merlin xDSL framework — USE IT (at authoring AND runtime)
- `xdsl` (the library) **and** Merlin's `merlin.xdsl_dialects` (IRDL ops/types/verifiers + rewrite
  patterns), `merlin.targetgen.{synthesize,generate}`, and `merlin.targetgen.contract.interface_emit`
  are **available to import at runtime** — importing them is expected and legitimate (it is the
  framework, not the answer).
- **Run your tool with the Python that has the framework installed:**
  `/scratch/agustin/projects/oscar-merlin/.venv/bin/python` (has `xdsl` + `merlin`). Set your
  `manifest.yaml` `commands[*].argv` to invoke your tool with **that interpreter**, e.g.
  `["/scratch/agustin/projects/oscar-merlin/.venv/bin/python", "{tool}", "{input_mlir}", "{output_json}"]`.
  (The system `python3` does NOT have xdsl — using it will fail at runtime.)

## What you MUST build (the point of this arm)
A **proper xDSL backend**: define your input + Gemmini **target dialect** (IRDL ops/types with verifiers)
and implement **lowering/rewrite passes** that transform the parsed `merlin_iface` module into the
Gemmini target dialect and then to the RoCC command buffer. **A regex-parse + f-string-template emitter
is NOT an acceptable submission for this arm** — it does not exercise the framework and will be flagged.
Use `merlin.xdsl_dialects` patterns as your starting point.

## The ONLY things forbidden (the answer key)
You may import the framework freely. You may **not**:
- import or call the oracle: `merlin.runtime.reference`, `merlin.runtime.simulator`,
  `reference_outputs`, `outputs_match`, or `merlin.xdsl_dialects.lowering` (its `pipeline.execute`
  routes to the oracle);
- read any capsule `golden.yaml` / expected outputs, hidden capsules, the grader internals, or any
  prior backend.
The integrity scan enforces exactly this (oracle-only); the self-check tool's redacted verdict + your
own artifacts are your feedback. Self-grade against **your own** reference (you know the op semantics),
never the harness oracle.

## Required entrypoints (unchanged)
`parse` · `lower_interface_to_target` (emit gemmini-dialect MLIR, parses+`verify()`) ·
`emit_command_buffer` (schema-valid `command_buffer.json`) · `lower_target_to_llvm`
(`llvm.func @gemmini_kernel` of RoCC `.insn r 0x7b` the shared `rocc_decode` can decode).

## Provenance ask
After you converge (or stop), fill in `MERLIN_PROVENANCE_TEMPLATE.md` → `submission/docs/merlin_provenance.md`:
which framework pieces you used (which `xdsl_dialects` patterns, which passes), how they helped, and that
your artifact uses the framework but **no oracle**. This documents *how* the Merlin/xDSL approach did or
didn't help — the whole point of the comparison.
