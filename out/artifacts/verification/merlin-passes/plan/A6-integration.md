# A6 — Pipeline integration

**Branch:** `pv/integration`. **Mission:** make the paper build produce one checkable chain for
row 2. That means:
- record the W8A8 boundary and admit the paper configuration;
- compose the chain across spec-transformer boundaries;
- generate the trust table;
- run the end-to-end check nightly.

## Read first

- `src/merlin/verify/lean/stages.py` (recording), `plans.py` (authored rosters, `preparation()` at
  line 151), `preparation.py` (`_admit` at line 44, `capture`, `validate`), `handoff.py`, and
  `check.py` (`compose_chain`, line 555).
- `src/merlin/runtime/backends/zephyr_model.py:466` (`_prepare_model_mlir`).
- `MerlinVerify/Check/Chain*.lean`, `TypedChain.lean`, `Check/Types.lean` (`ClaimType` already has
  `policy`, `equivArgs` and `contractSeam`).
- Design §5.10 (composition) and §8.4 (coverage matrix).

## You own

- `src/merlin/verify/lean/{stages,plans,preparation,handoff,check}.py` changes, and a new
  `coverage.py` plus its CLI subcommand.
- The single `_lean_stages.run` wrapper added in `zephyr_model.py`.
- `MerlinVerify/Check/ChainPolicy*.lean`, the new composition theorems. Statements go through A0.
- `merlin/tests/ir/test_lean_preparation_int8*.py`, `test_lean_coverage*.py`,
  `test_lean_chain_policy*.py`.
- `out/artifacts/verification/merlin-passes/nightly/`.

## Work, in order

1. **D1: record W8A8 as its own boundary.** Wrap the `apply_quant` call in `zephyr_model.py` in
   `_lean_stages.run("prepare-quantize-int8", ...)`, between `prepare-qinner-lift` and
   `prepare-lower-quant-ext`. This is a recorder call only: add a test that the emitted module is
   byte-identical with recording on and off. Tell A1 and A3 when it lands.
2. **D1–D2: admit the paper configuration.**
   - In `preparation._admit`, accept `int8_compute=True` with exactly `("contraction_int8",)`
     selected (the `respect_captured_quantization_scope` path). Keep refusing every other
     quantization pass and every optional feature.
   - Add the `prepare-quantize-int8` slot to `plans.preparation()` when int8 is selected. The
     roster is still authored independently and must match production before anything runs.
   - The paper build passes `device=plan_before_build(..., BY_GROUP)`. Instead of refusing, end the
     verified preparation chain at the routing boundary, and record routing as an explicitly
     **assumed** boundary (trust-table row 3). Keep refusing custom matrices, outlining and provider
     callbacks.
   - Tests: an omitted or reordered int8 slot must be refused, and so must a different pass
     selection. Unchanged boundaries must still be recorded.
3. **D2–D3: propose the composed-claim shape** in `nightly/CLAIM_SHAPE.md` for A0 and Jeremy to
   review. A policy or intent transformer isn't monotone, so the composed claim is a sequence of
   segments: each segment is a refinement chain, and segments join where a transformer applies.
   For example: `S_n ⊑ … ⊑ T_bool(S_j)`, `S_j ⊑ … ⊑ P_W8A8(S_k)`, `S_k ⊑ … ⊑ Args(S_m)`,
   `S_m ⊑ … ⊑ T_overrank(S_p)`, `S_p ⊑ … ⊑ S_0`, plus source definedness of `S_0`.
   - Define exactly what the paper states, and how receipts name each segment's claim type and
     spec.
   - Assumed boundaries (routing) appear as named hypotheses, never silently.
4. **D3–D6: implement the composition.**
   - Add Lean theorems for chaining across `policy`, `equivArgs` and assumed boundaries.
   - Extend `compose_chain` so it replays each receipt, checks that the claim type and spec name
     match the authored roster, and refuses substituted labels.
   - Keep the existing chain theorems and their pins unchanged; the new ones are additive.
5. **D3–D6: hints.** Let a recorded stage carry untrusted site hints in `meta.json`, taken from the
   `prov.*` roles the passes already emit, for checkers that want them. The checker treats them as
   candidates only, and a missing or wrong hint can only cause a refusal.
6. **D4–D8: trust-table generator.** Build `merlin-verify lean coverage <capture-dir> --plan
   <roster>`. It emits JSON and Markdown with one row per boundary:
   - claim type;
   - verdict;
   - sites verified, assumed and unsupported, with reasons;
   - method;
   - trust class (verified, assumed, tested, trusted).

   Include rows 1 and 3–9 from the plan's trust table as declared rows. It's generated and never
   hand-edited (design §8.4). Run `gen_cli_docs.py` after adding the subcommand.
7. **From D6: nightly end-to-end run.** Build the chosen model with recording at the current
   `feat/pass-verification` head. Run every checker, compose, and generate the trust table into
   `nightly/<TS>/`. Post the summary line (verified, assumed and abstained counts, plus the first
   failure) in `A6.md`.

## Don't

- Don't relax any existing refusal except the two above (int8 `contraction_int8`, and routing
  becoming an assumed boundary).
- Don't change pass behavior. The only edit to `zephyr_model.py` is the recorder wrapper.
- Don't edit `SPEC_LOCK`. Send statements to A0.
