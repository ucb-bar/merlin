# Rules for every agent

These apply to every agent in this campaign. Your brief adds to them and never relaxes them. If
your brief and these rules conflict, stop and ask A0.

## 1. Read before starting

- `CLAUDE.md` (repo conventions) and `merlin/lean/AGENT.md` (the Lean rules and layout).
- `docs/design/merlin_lean_verification.md`: §1.2 (non-negotiables), §4.5 (refinement and
  definedness), §5.1 (the rule checker), §5.5 (policy checking), §5.10 (composition).
- `merlin/lean/LEDGER.md` (assumptions) and `merlin/lean/findings.md` (bugs found so far).
- [`README.md`](README.md) (this plan) and your brief.

## 2. Lean non-negotiables (enforced by `build_tools/scripts/check_lean_guard.py`)

1. No `sorry`, `admit`, new `axiom` outside `MerlinVerify/Assumptions/`, `native_decide`,
   `@[implemented_by]`, `@[extern]`, `unsafe` or `partial def` under `MerlinVerify/`. `bv_decide`
   is effectively banned too, because it adds a native axiom per proof.
2. `#print axioms` of every claim is a subset of {`propext`, `Classical.choice`, `Quot.sound`} plus
   reviewed `Assumptions/` entries.
3. Floats are parametric. Never prove a float reassociation or constant fold. A float law exists
   only as an approved `Assumptions/` entry with a ledger row.
4. `Core/`, `Sem/Outcome|Float|Value|Refine|Program|Ops.lean` and `Check/Types.lean` are frozen.
   Changing them takes an interface change request to A0. Put additive lemmas in a file you own.

## 3. Verification discipline

1. **Write the soundness statement before the checker.** Send the statement to A0 for review
   before building on it.
2. **Certificates and hints are untrusted.** A wrong hint may only cause a refusal. The checker
   recomputes every fact its proof needs.
3. **Unsupported means abstained.** Never skip an unsupported construct and never report it as
   verified.
4. **Don't change the pass you verify**, except to add recorder or hint calls, and those must
   leave the emitted output byte-identical (test it). If you suspect a pass bug, write an entry in
   `merlin/lean/findings.md`, tell A0, and wait. The fix is a separate `fix(...)` commit with a
   regression test, as with F6.
5. **Don't weaken or delete tests.** A0 reviews test diffs.
6. **Don't narrow a statement or add an assumption without saying so.** If a proof won't go
   through, report it as Blocked (section 7) and propose alternatives: a stronger side condition in
   the checker, a narrower rule, or a split rule.
7. **Actual output counts; fixtures don't.** A boundary is done only when the checker accepts the
   chosen model's real boundary (plan §6).

## 4. Ownership

- You edit only the files your brief lists as yours. Anything else goes through A0.
- **`merlin/lean/SPEC_LOCK` is edited only by A0.** Propose statement pins in your status file.
- **`MerlinVerify/Spec/` is edited only by A2.** Other agents request changes through A0.
- `MerlinVerify/Assumptions/` changes need Jeremy's approval, and Agustin's for any float law.

## 5. Repo conventions (gate-enforced)

- No `import re` in library code. Parse structurally.
- No target-name literals (`"gemmini"`, `"atlas"`, …) in `src/merlin/**`, `packages/*/src/**` or
  `build_tools/scripts/**`.
- Generated output goes only under `out/`, using `merlin.common.paths` helpers. Evidence for this
  campaign goes under `out/artifacts/verification/merlin-passes/<your-area>/`.
- Lean tests go in `merlin/tests/ir/test_lean_<area>.py`. Resolve repo paths with
  `merlin.common.paths.repo_root()`, never `Path(__file__).parents[N]`.
- Commits: `type(scope): summary`, at most 72 characters, imperative mood, with
  `type ∈ feat|fix|docs|chore|refactor|test|perf|build`. Describe the change, not the process: no
  "phase", "step", "cohort", "round", "WIP" or "checkpoint". No attribution trailers. One coherent
  change per commit.

## 6. Git

- Work only in your own worktree and branch. Never switch branches in the main checkout.
- Rebase on `feat/pass-verification` at least daily. A0 merges.
- **Push your branch to `origin` at least daily.** The previous campaign lost three batches that
  existed only in local clones.
- Never force-push a branch someone else uses.

## 7. Status reporting

Keep `$STATUS_DIR/<agent>.md` current. Update it at least every 4 hours of work, at every gate, and
whenever you get blocked. Use this format and keep it short:

```markdown
# A<n> status — <UTC timestamp>
Done: <items, each with commit ID>
In progress: <item, expected finish>
Blocked: <what you need, from whom> (or "none")
Next: <items>
Evidence: <paths>
Assumptions added or proposed: <list or "none">
Statements awaiting review: <Lean names and file paths, or "none">
```

**Escalate** (set Blocked and tell A0) when:
- you've spent 3 hours on one theorem without progress;
- you need a file you don't own;
- a pass looks wrong;
- an actual boundary contains a construct your rule can't support.

## 8. Environment

- Lean v4.35.0 through elan. In `merlin/lean`, run `lake exe cache get`, then `lake build`.
- Compiler interpreter: `MERLIN_M2M_PYTHON`. Pinned PyTorch oracle: `MERLIN_TORCH_PYTHON`.
- Qualification runs set `MERLIN_LEAN_REQUIRED=1`, so missing tools fail instead of skipping.
- Stage recording: set `MERLIN_LEAN_STAGES=<new directory>` for a build, or use
  `merlin.verify.lean.stages.capture(...)`.

## 9. Stop list

See plan §8. If you find yourself working on upstream MLIR rewrite families, optional passes,
model2MLIR, the OOT, or the previous campaign's groups 13/14, stop and tell A0.
