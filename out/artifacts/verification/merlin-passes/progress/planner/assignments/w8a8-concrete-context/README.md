# W8A8 next assignment: construct the numerical context

Assigned after the qualified modular-policy milestone on 2026-10-22 (US/Pacific).
The existing W8A8 worker remains the implementation owner. Continue the
[eight-day goal](../../eight-day-paper/README.md); target October 30. Do not return another
planner handoff while this owned assignment is unfinished. A new explicit user stop applies.

## Concrete deliverable

Construct an explicit arithmetic base and prove, together, all existing premises required by
`PolicyW8A8NumericalContext.ContextOk`:

1. The seven fields of `Rules.W8A8FiniteLaws.Laws`.
2. The three additional fields of `Rules.W8A8DynamicLaws.Laws`.
3. Every field of `Rules.W8A8ArithmeticModel.Agreement`.
4. An actual context witness whose model is the corresponding `W8A8ByteModel.extend base`.

This establishes nonvacuity of the numerical context, separately from the already inhabited
finite/nonzero/exponent-cap input domains. It does not establish that actual captures meet
those domains, execute through attention, or match every native arithmetic implementation.

Use explicit pinned rounded operations, conversions and operation keys. Keep source f64
reciprocal followed by f32 narrowing distinct from dynamic f32 division. The normal-scale
guards in the existing laws are deliberate; tiny reciprocal overflow is not silently admitted.
Use the existing overflow-aware classification lemmas rather than finite-product closure.
State the exact sqrt/divide recipe used for `math.rsqrt`; native lowering fidelity is separate.

No new Agreement hypothesis, law record assumed inhabited, placeholder arithmetic returning
zero for constrained operations, new axiom, changed frozen semantics or production pass repair
discharges this task. The model may retain arbitrary shared base operations outside the
constrained slice, with that scope stated exactly. If a premise is false under the intended
explicit interpretation, preserve a kernel-checked counterexample and identify the exact
failed field instead of weakening the existing law silently.

## Shared numerical interface

Frontend owns the compatibility proof and receipt integration for the literal context
`W8A8ByteModel.extend (TorchAOIEEEPrimitives.model (withIEEEDivision base))` in its separate
byte-compat checkout. It has proved the two TorchAO laws for that named construction and
conversion equivalence, but qualification is still in progress. Its division uses the pinned
binary32 `UnpackedFloat.div`. Read its exact definitions; do not assume all model fields are
already globally equal. Prefer a W8A8 construction/adapter that can satisfy this common
interface without copying or changing frontend-owned code.

Quant-ext owns the f32 versus f64-then-f32 reciprocal identity and its `mixed-ieee` mode.
Do not duplicate that proof or treat its native samples as a proof of W8A8's arithmetic
Agreement. Any required equality between independently defined arithmetic implementations
remains an explicit integration obligation with one agreed owner.

## Execution sequence and qualification

- Read the modular-policy report and preserve the complete 68-file W8A8 delivery and every
  frozen runtime. Implement additive W8A8 modules and a **new** private source/build profile.
- Write the exact constructor/witness theorem first, compile all fields, and audit the actual
  imported closure and theorem statements. Do not rebuild the old qualified capsules.
- Add a separately named checker/receipt specialization for this explicit numerical instance,
  reusing the existing soundness result. Preserve generic/draft routes and flags. Report which
  hypotheses were proved and which remain about policy, native fidelity, domains and captures.
- Exercise actual retained prefixes and complete boundaries; compare the next refusal with
  quantizer 1533. Retain meaningful exceptional, signed-zero, subnormal and scale-bound tests,
  valid mutation refusals and current receipt replays against one fresh frozen identity.
- Record the exact model/domain witness and its numerical versus native scope, source hashes,
  tests, audit, replay and remaining obligations. Then continue the owned attention/1533 work:
  ordered multi-write bounds, sin/cos interpretation, masking with negative infinity, attained
  maxima and nonzero softmax denominators, without assuming the desired numerical result.

Independent review and final capture selection remain pending; continue authorized proofs
and qualification meanwhile. Preserve RMSnorm/rotary overflow witnesses, F11/F12/F13 and
the historical undiagnosed JSON failure. Broad flags stay false until their own obligations
are met; proving a named context exists does not prove native fidelity or the whole pass.

Only owned W8A8 additions and evidence are authorized. No staging, commits, integration,
pushes, frozen/spec/lock/shared Main/CLI/exporter changes or edits to other workers' modules.
After compaction reread this assignment and the current worker-status resume record, inspect
live processes and continue the next unfinished command. Update that record before compaction.
