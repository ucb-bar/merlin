# Addendum — Arm4 `merlin_assisted_rtlchecks` full suite (read together with the task above)

Everything in the generated task above applies **unchanged**: the runtime-generated scope block is
authoritative for the full admitted public/dev cohort, including its operator/slice cases and the
resource-bounded full-model capstones. The deliverable layout under `submission/`, 4 CLI entrypoints,
redacted QA gate, integrity rules, and final status line remain unchanged. This addendum **only adds**
Merlin-tooling requirements and a provenance ask. It does not relax or change how you are graded.

## What's different for this arm

You additionally have read access to Merlin's **authoring** tools (see `ALLOWED_MERLIN_TOOLS.md` in
your working directory for the exact allowed/forbidden surface):
`targetgen/synthesize/`, `targetgen/generate/` (minus `runtime_adapter.py`), `xdsl_dialects/` (minus
`lowering/`), and `targetgen/contract/interface_emit.py`. Use them if they help you author or debug the
backend faster.

## USE the arm-4 tooling — this is the whole point of this arm (actually RUN these, do not just read them)

These tools are the ONLY thing that distinguishes this arm from the raw baseline; if you author by hand
without them, the arm has no advantage. The `merlin` package is importable here **for authoring** (the
final package must still be self-contained + oracle-free). As your FIRST actions each round, actually run:

1. **CCA seam menu** — the machine-checked statement of which compiler sections you may modify and the
   next stronger lever for an axis (adapt args from the code — you can read these modules):
   - `python -c "from merlin.kernels.cca_contract import check_bijection; print(check_bijection('gemmini'))"`
   - `python -c "from merlin.kernels.action_catalog import escalation_ladder; print(escalation_ladder('spatial.dataflow','gemmini'))"`
2. **RTL-derived levers** for this target, extracted from the real RTL (not guesses):
   - `python -c "from merlin.targetgen import rtl_backend as R; print(R.derived_levers(R.target_profile('gemmini')))"`
3. **Ground the ISA in RTL facts** (mesh DIM, opcode/funct legality, dtypes, memories) instead of guessing:
   - `python -c "from merlin.targetgen.rtl.facts import load_facts; import json; print(json.dumps(load_facts('gemmini')['facts'],indent=1)[:2000])"`
4. **Scaffold generators** (`targetgen/synthesize/`, `targetgen/generate/`) — actually invoke one before
   your first submission edit and print the returned generated paths so the measured transcript contains
   a concrete generation witness. This zero-argument plan is a reliable starting point:
   - `python -c "from merlin.targetgen.generate import target_repo; a=target_repo.generate_skeleton('gemmini'); print([x.relpath for x in a])"`
   Use the generated shape as the scaffold for your xDSL package rather than hand-writing it from scratch.
5. **Every round, READ the real `rtl_checks` block** in your redacted `qa/verdict.json` (for example,
   `jq '.rtl_checks' qa/verdict.json`) and fix exactly what it
   flags (illegal funct/opcode, wrong tile count, missing instruction class). It is RTL-grounded truth the
   functional sim alone cannot give you — it is the reason this arm exists.

Do 1–4 before the first submission edit in each authoring round: derive the CCA seam, levers, facts, and
scaffold first, print their non-empty results, then author against them. Mentions in comments/`echo` do not
count; the workflow gate verifies the real API calls and their returned evidence.

## Rules for this arm (in addition to the task's hard rules)

- This is a **fresh, measured Arm4 `merlin_assisted_rtlchecks` full-suite run.** Your wall-time, tokens, cost, tool-calls,
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
- **Iterate** against the redacted public/dev QA verdict until every capsule in the authoritative
  runtime-generated public/dev scope passes or the round budget is exhausted. **Hidden grading is
  post-freeze and hidden repair is disabled** — you never see
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

---

## RTL-derived checks (ADVISORY feedback — this track only)

Each round, in addition to the redacted QA verdict, you receive an `rtl_checks` block. These are
**deterministic, RTL-grounded structural checks**, NOT golden values:

- They are compiled from the **actual Gemmini RTL** (facts extracted via CIRCT from the elaborated
  hardware: systolic mesh `DIM`, scratchpad/accumulator capacity, the legal RoCC funct decode table)
  plus the capsule's **declared** shape — never from any reference output.
- They run as `FileCheck` assertions over your emitted gemmini-dialect MLIR (`lowered.target.mlir`) and
  over the decoded RoCC trace. Each finding gives `expected` / `got` / a `fix_hint`, e.g.:
  - `MVOUT_COUNT` ≠ ⌈M/DIM⌉·⌈N/DIM⌉  → your output tiling does not cover the declared shape;
  - `ILLEGAL_FUNCT_COUNT > 0`         → you emitted a custom-3 funct the decoder rejects;
  - `COMPUTE_PRESENT no` on a matmul  → movement without the matrix compute;
  - scratchpad/accumulator address ≥ RTL capacity → resident footprint exceeds the hardware.

**These checks do NOT change pass/fail.** They are a fast, hardware-faithful signal so you can fix
encoding / tiling / capacity bugs *before* the expensive RTL oracle would catch them. A clean
`rtl_checks` does not guarantee numerical correctness — it means the ISA structure is hardware-legal.
Treat a `reject` verdict as a near-certain RTL-oracle failure worth fixing first.
