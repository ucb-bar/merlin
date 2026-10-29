# A0 — Coordinator and integrator

**Who:** Jeremy, with one Codex agent working in the main checkout on `feat/pass-verification`.
**Mission:** keep the plan on schedule, merge the agents' work, own `SPEC_LOCK`, run the
integrated qualification and freeze the evidence. You are the only agent that sees the whole
picture.

## You own

- `merlin/lean/SPEC_LOCK` and the guard's pinned-statement list.
- Merges into `feat/pass-verification`.
- `$STATUS_DIR/A0.md`: a daily summary with gate tracking.
- `out/artifacts/verification/merlin-passes/qualification/` and `.../freeze/`.

## Every day

1. Read every `$STATUS_DIR/A*.md`. Act on each Blocked item within a few hours; unblocking others
   is your highest priority.
2. Merge the branches that are green, in this order: rebase onto `feat/pass-verification`, run
   `lake build`, run `build_tools/scripts/check_lean_guard.py`, run the agent's targeted tests,
   then merge. Never merge a branch whose guard fails.
3. Pin every statement Jeremy has approved in `SPEC_LOCK`, in its own commit.
4. Review test diffs for weakening (deleted asserts, broadened expected failures, removed
   mutations). Reject any.
5. Enforce the stop list (plan §8).
6. Write `A0.md`: per-gate status, what merged today, open decisions, risks.

## Gates

- **Gate A (end of D2).** Take A1's `INVENTORY.md`. With Jeremy and Agustin, confirm the model and
  configuration. Write the final row-2 list (the passes that actually fire) into `A0.md` and tell
  each agent which of their rows were dropped. Record the decisions from plan §7.
- **D3.** Review A6's composed-claim proposal with Jeremy. Pin the TorchAO and intent statements
  once Jeremy approves them.
- **Gate B (end of D5).** Is the W8A8 statement frozen and reviewed? If yes, pin it. If not,
  declare the fallback: W8A8 becomes "assumed, with differential tests against PyTorch", A3 moves
  to the leftover row-2 work, and A6 marks the boundary assumed in the plan.
- **Gate C (end of D9).** Every row-2 boundary is verified or explicitly assumed on the model, and
  the nightly end-to-end composes. List the gaps and decide: fix by D11, or mark assumed.
- **D10–D11: integrated qualification (one run).**
  1. Start from a fresh build of the chosen model with stage recording, at the commit you will
     freeze.
  2. Run every boundary checker and the source support and definedness checks, compose the chain,
     and generate the trust table (A6's generator).
  3. Run the mutation matrix from every workstream against the integrated checker.
  4. Measure checker runtime per boundary.
  5. Record the source, checker and binary hashes before and after the run. Don't edit or rebuild
     during the run. A failed or interrupted run is kept and labelled, never overwritten.
- **Gate D (D12): freeze.**
  1. Write `freeze/<TS>/manifest.yaml` with commit IDs, file SHA-256s, tool fingerprints, model and
     configuration identity, and verdicts.
  2. Add `.gitignore` negations for the curated evidence files, as the pause record did.
  3. Commit with a `docs(lean): ...` message.

## Escalation rules

- Three hours without progress on one theorem: discuss it with the agent. Usually the fix is to
  strengthen the checker's side conditions or split the rule (Axon's "widen the scope" lesson).
  Never let a statement be weakened silently.
- A suspected pass bug: the agent writes a findings entry, you approve a separate `fix(...)` commit
  with a regression test, and the evidence keeps the before and after.
- An agent drifting out of scope: stop it and point it at the stop list.

## Handing off to Jeremy

Jeremy reviews every `Spec/` statement, every new `Claims` or chain statement, and every
`Assumptions/` entry. Batch them, with a one-paragraph plain-English summary of what each
statement means and what it doesn't cover.
