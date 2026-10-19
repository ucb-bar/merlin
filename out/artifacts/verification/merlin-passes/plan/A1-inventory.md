# A1 — Inventory, then evidence

**Branch:** `pv/inventory`. **Mission:** turn "how far are we?" into measured numbers on real
models, recommend the demonstration model by the end of D2, and then collect the paper's evidence.
You measure and report. You don't fix checkers or passes.

## You own

- `out/artifacts/verification/merlin-passes/inventory/` and `.../evidence/`.
- Small, read-only measurement scripts. If one is worth keeping, it goes in a library module with a
  test, never `build_tools/` one-offs. Ask A0.

## D1–D2: inventory (deadline: end of D2, Gate A)

1. **Environment.** Bring up the capture and compile environment: the model2MLIR environment,
   `MERLIN_M2M_PYTHON`, checkpoints, and the private W8A8 recipe if you have access, otherwise the
   public stand-in `--dtype int8 --scheme int8_dyn_act_int8_weight`. Record what works and what
   doesn't in `A1.md` within the first 4 hours. If you're blocked, say so immediately.
2. **Candidates.** The roster is `phase1_gates.private_full_models` (resnet50, smolvla
   prefix_encode/flow_denoise/action_decode, tiny_llama prefill/decode, int8 deployment). The
   build path is `packages/merlin-experiments/src/merlin_experiments/phase1/feedback/private_full_models.py`
   (`compile_saved_model(run="none", device=plan_before_build(..., granularity=BY_GROUP))` with the
   scalar int8 host package, `int8_compute=True`). Start with the two smallest programs. If private
   inputs are blocked, use public model capsules for the same shapes of work, and label the results
   "public stand-in".
3. **Record every boundary.** Build each candidate with `MERLIN_LEAN_STAGES=<new dir>`. Recording
   prints the module and must not change output.
   - Note that `apply_quant` is not recorded yet (A6 is adding it). Until then, diff
     `prepare-qinner-lift/after.mlir` against `prepare-lower-quant-ext/before.mlir` to see the W8A8
     change.
   - Also record the `lower_model` preprocessing boundaries and the upstream tensor passes (rows 4
     and 5).
4. **Measure each boundary.** Record:
   - whether it changed, and the changed op count;
   - the op census of its *before* text (op names and counts);
   - the existing checker verdict in report mode, `check_typed_stage(stage, index_bits=64)` (and
     `check_stage`), with its abstention reason;
   - wall time.
5. **Check the source.** Run `check_support` and `check_definedness` on `S0`, the preparation
   input. List every op `S0` uses that the Lean semantics doesn't model, with counts.
6. **Measure row 5.** Over the upstream-pass boundaries, report the percentage of changed sites the
   existing families verify. This is the paper's partial-validation row.
7. **Deliver** `inventory/<model>/inventory.json`, plus one `inventory/INVENTORY.md` comparing the
   candidates with:
   - a per-boundary table (stage, pass, changed?, sites, verdict, reason, time);
   - `S0` support and definedness verdicts and the unmodeled ops;
   - the checker time at model size;
   - **a recommendation**: the smallest program with the fewest gaps, and which row-2 passes fire on
     it.

## D3–D12: evidence

- **Rows 3, 7, 8 and 9.** Summarize what existing tests and L3 grading already cover for device
  routing, `ciface_inline`, runtime artifacts and link layout. These become "tested" rows in the
  trust table. Cite test names and the evidence paths.
- **Effort metrics, the way Axon reports them:**
  - Lean non-blank lines by category (semantics, checker framework, definedness,
    Merlin-pass rules, upstream rules, specs, assumptions);
  - commits per day;
  - agent count and active days.
  Recompute at freeze.
- **Runtime.** Checker time per boundary on the model, compared with compile time.
- **Findings list** for the paper: F6, S4, S6, plus anything new, each with a one-line description
  and its evidence path.
- Help A0 with the D10–D11 qualification run.

## Don't

- Don't change passes, checkers or specs.
- Don't fix gaps you find. Report them with an example.
- Don't let the inventory slide past D2. A partial inventory on time beats a complete one late.
