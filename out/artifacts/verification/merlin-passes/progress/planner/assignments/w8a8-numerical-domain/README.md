# W8A8 assignment: discharge the result-1256 numerical blocker

**Historical assignment, superseded by the user-approved
[eight-day scoped paper goal](../../eight-day-paper/README.md).** Result 1256 now passes under
the explicit candidate domain/premises; the next retained refusal is 1533. Read the current
W8A8 numerical/rotary reports and continue the owned proof and qualification work. Do not
restart this completed milestone or interpret its old stopped-planner language as a new stop.

Assigned 2026-10-22 to the existing W8A8 worker. The user explicitly wants sustained W8A8
implementation work for several hours without returning for another assignment. The planner
is otherwise stopped; report progress and the completed deliverable in your own conversation.
Shared-branch staging, commits, integration and pushes remain on the user-controlled hold.

## Objective

Advance the actual `lower_contraction_int8` W8A8 proof through the retained TinyLlama source
quantizer at result 1256. Implement and qualify the necessary numerical reasoning and connect
it to the actual exported source and private W8A8 checker. This is W8A8 proof/integration work,
not a separate generic interval-analysis project, another diagnosis-only assignment, or LLVM
verification. Keep going through implementation, proof, actual-boundary exercise and tests.

Start from `w8a8/worker-status/README.md`, `w8a8/numerical-interface/README.md` and their raw
evidence. Those paths are relative to `out/artifacts/verification/merlin-passes/`.

## Numerical problem to solve

The actual chain is embedding result 979, RMSnorm reciprocal-square-root 1236, normalized
product 1241, gain product 1247, then source quantization result 1256. The captured gain is
an original input. Finite gain and normalized inputs can produce infinity. The native overflow
witness and the abstract law countermodel are retained negative controls.

Finiteness of every product is therefore not the required invariant by default. Work backward
from the exact source quantizer's successful, nonpoison conversion domain. Determine whether
infinite gain-product values can be handled soundly by its actual reciprocal, multiplication,
rounding, zero-point addition and ordered clipping. Account for NaN production, infinity times
zero, signed zero, underflow, reciprocal nonzeroness and the actual operation keys. Source
binary64 reciprocal followed by f32 conversion is distinct from dynamic-policy f32 reciprocal.

## Work sequence and deliverables

1. **State the W8A8 theorem before implementing its check.** State the exact successful-source
   quantization property, its float interpretation, input domain, numerical premises and actual
   SSA/prefix dependencies. Keep source success/nonpoison, reference support and policy matching
   separate. Identify which conditions are proved, checked or still hypotheses.
2. **Implement the smallest sound numerical extension needed by this chain.** Prefer reasoning
   that allows the observed infinities when source quantization is defined. Prove the necessary
   reciprocal/product/rounding/clamp facts for an explicit concrete model where feasible. Additive
   conditional lifting is useful, but a new hypothesis is an open obligation, not its discharge.
   Do not replace a failed finite-product premise with an equally unjustified non-NaN premise.
3. **Justify upstream facts from the actual source.** Handle the actual RMSnorm reduction,
   division, epsilon, rsqrt, broadcasting and gain wiring needed by your theorem. Track actual
   SSA bindings, captures and invalidation; parse constants and maps from source. Neither an
   operation-name pattern nor a plausible real-number RMSnorm inequality establishes rounded
   floating execution. Derive archive/constant facts from bound bytes where available; do not
   infer actual weight magnitudes from the model name or a synthetic example.
4. **Connect the result to the private W8A8 checker/receipt route.** Check the real retained
   before/after exports and exact input-contract snapshot. Report whether 1256 is discharged,
   what the next refusal is, and the theorem/premises supporting that result. Preserve the
   original attempted receipts and generate new evidence under a new identity/directory.
5. **Qualify the change.** Retain meaningful positive cases and compiler-valid adversarial
   cases: the existing finite-gain overflow witness, zero gain, both infinity signs, NaN,
   extreme/subnormal reciprocal behavior, altered epsilon/scale/zero-point, wrong broadcast
   maps and stale/rebound SSA facts as relevant to the implemented checker. Cases outside
   the claimed domain must refuse or have their precise modeled outcome recorded. Do not
   execute poison natively or count compiler rejection as Lean mutation rejection.
6. **Run the affected W8A8 regressions, fresh dependency/axiom audit and receipt replay** against
   one frozen private build. Verify identities afterward. Record tests separately from universal
   proofs and concrete-law admission; do not add overlapping historic test counts.
7. **Produce a reviewable worker report** with owned paths, exact theorem statements and remaining
   premises, original versus any candidate input domain, nonvacuity, actual changed-boundary
   coverage, next refusal, tests/audit/replay and preserved discrepancies. Leave coherent changes
   ready for user-controlled review/publication without staging or committing them.

## Autonomous handling of genuine blockers

Do not stop merely to ask which numerical route to try, for a selected final capture, for
independent review or for permission to edit owned additive W8A8 files. The user is choosing
the final capture with a teammate; retained development inputs remain explicitly development
evidence. Review and capture selection cannot be self-supplied and must not prevent independent
proof and checker work.

If the unchanged entry contract is insufficient, preserve a precise counterexample or a
proved insufficiency result and continue implementing a separately named, inhabited candidate
contract for the same W8A8 chain. Make every new restriction explicit and keep the original
domain/report intact. Prefer symbolic bounds with proved operational consequences over guessed
constants. A stronger domain remains proposed until reviewed, and success on it must not be
reported as solving the original domain. Keep working on source-derived fact extraction,
concrete primitive lemmas, tensor/SSA lifting and qualification rather than ending with only
another list of needed assumptions. Do not pursue a universal implication already refuted by
the retained witness.

If 1256 is genuinely discharged, continue into the next numerical refusal in the same retained
W8A8 source, reusing the proved facts and preserving the same domain. Do not expand into another
worker's pass. Finish at a coherent, qualified W8A8 deliverable, not at the first passing fixture.

## Ownership and invariants

- Own additive `Rules/W8A8*.lean`, `Check/PolicyW8A8*.lean`, W8A8-specific private drivers,
  W8A8-specific Python modules/tests/fixtures and W8A8 evidence/reports. Announce exact new paths
  in your own report. Leave general shared modules and exporter changes for coordination.
- Preserve the original six qualified files and 752-file runtime capsule as historical inputs.
  Build changes in a new private source/build snapshot; do not mutate the frozen capsule or
  another worker's package artifacts. Reuse large read-only dependencies without changing them.
- Keep the meanings of the existing seven source and three dynamic primitive laws fixed.
  New concrete lemmas/adapters must be explicit and independently audited. Do not add axioms,
  `sorry`, `admit`, `native_decide`, `bv_decide`, unsafe or substituted implementations.
- No production-transform changes. No edits to frozen Core/Sem/Typing/Types, `Spec/`, `Claims/`,
  `Assumptions/`, statement pins, `SPEC_LOCK`, shared Main/CLI, central LEDGER/findings, frontend,
  Qinner, scalar-square, overrank or bf16 files. Additive lemmas about frozen meanings belong
  in owned W8A8 files.
- Frontend retains `AbsReductionCorrect`, `ZeroPointElisionCorrect` and its captured-call proof.
  `W8A8ByteModel.extend` is not automatically equal to the frontend's float interpretation.
- Preserve F11/F12/F13, the numerical counterexamples, the unresolved original JSON failure
  and all old evidence. The now-successful refusal replay does not diagnose that old failure.
- Keep `whole_pass_verified`, `policy_reviewed` and `primitive_model_laws_verified` false unless
  their actual obligations are discharged. No self-approval of policy/domain/specification.

Use the preserved toolchain/environment guidance in `w8a8/handoff/README.md`. Required-mode
qualification uses `MERLIN_LEAN_REQUIRED=1` and retained successful pytest artifacts. Keep
ordinary user-facing progress updates in your own conversation, but do not return to the
planner for periodic reassignment. Do not create new agents for this task.
