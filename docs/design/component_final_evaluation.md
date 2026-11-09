---
title: "Protected component campaign final evaluation"
kind: design
status: current
owner: merlin-experiments
last_verified: 2026-11-09
related: [component_compiler_convergence, component_phase2_workflow]
code_refs:
  - src/merlin/experiments/phase2/component_final_policy.py
  - src/merlin/experiments/phase2/protected_final_evaluation.py
  - src/merlin/experiments/phase2/protected_final_observation.py
  - src/merlin/experiments/phase2/physical_final_admission.py
  - src/merlin/experiments/phase2/protected_verifier_qualification.py
  - src/merlin/experiments/phase2/numerical_readback.py
  - tests/targetgen/test_verification_protected_final_evaluation.py
  - tests/infra/test_verification_protected_verifier_qualification.py
---

# Protected final evaluation

The `merlin.component_final_gate.v2` policy requires the sealed compiler to
match or beat the frozen handwritten compiler for **every** final member. A
one-cycle regression fails, including at counters too large for exact floating
point comparison. Faster members cannot compensate for a slower member.
Convergence accounting is separate: comparable elapsed times produce a ratio;
missing history stays UNKNOWN and never changes the performance threshold.
Historical v1 receipts retain their original five-percent and twenty-times policy.

## Original ownership and qualification

The host selects `ProtectedFinalMember` records, the original campaign roster and
each `FinalExecutionBinding` through its private freeze lifecycle. They are never
candidate tool arguments. The original campaign JSON contains exactly:

```json
{
  "schema": "merlin.protected_final_campaign.v1",
  "members": [{"member": "<logical identity>", "binding_sha256": "<original binding digest>"}]
}
```

The evaluator verifies existing V4 ownership, the archived manifest and the exact
readonly snapshot. Binding, verifier control plan and implementation dependencies,
original budget, all readback/build files and execution witness records must be
private host inputs. Their live selected bytes must still equal the frozen bytes.
An omitted member, replaced binding, public record or changed source refuses.

The binding selects both compiler freezes, source program, inputs, original oracle
console, original numerical budget, runtime, toolchain, hardware and timer scope.
The host must freeze these selections before inspecting or compiling final
validation inputs. Snapshot integrity alone cannot prove that chronological rule
or the semantic validity of a reference.

`ProtectedExecutionVerifier.validate_selection` and `observe_arm` pin and inspect
an explicitly selected verifier without admitting it. `validate` and the final
observation path require an in-memory `ProtectedVerifierQualification` issued by
`qualify_protected_execution_verifier`. Its source, deterministic callable code,
dependency closure and original controls are rechecked around execution.
Qualification checks callback observations for compiler invocation, source
correspondence, original reference generation, dependency closure, declared hardware
execution, timer scope and whole-executable instruction audit. It does not qualify
the physical meaning of those observations. Returned arm
witnesses must bind actual executable/console bytes, staged executable, input,
hardware, timer, compiler and positive integer cycles to the original binding.
Each witness must name an actual reopened private evidence file.

The closed `merlin.protected_final_controls.v1` private plan contains exactly
`schema`, `scope`, the selected `dependencies` path/digest pairs and `controls`.
Every control records its name, witness kind, expected `observe` or `refuse`,
original typed binding digest and file, measured arm, complete readback paths,
original numerical reference paths, all input path/digest pairs and its actual
authority artifacts. The roster requires one independent positive and negative
case for each of the seven witness kinds. The host freezes this plan and every
input/dependency through the original V4 lifecycle before issuance.

Issuance executes the selected normal verifier on all fourteen cases. Positives
must return fully joined typed arm observations. Negatives must actually raise a
`ProtectedArmRefusal` naming the expected defective witness, original binding,
arm and reopened pre-frozen evidence. A crash, generic exception, wrong diagnosis
or accepted negative fails issuance. Original plan, inputs, dependency bytes and
current evaluator source membership are checked before and after every call.
All controls bind one exact runtime, toolchain, hardware and timer domain; the
issued verifier cannot observe an arm from a different declared execution domain.
The newly written readonly v3 receipt records `observation_qualified` and only the
`private_witness_observation` role. Replaying its
JSON or reconstructing a dataclass does not create an issued capability.

The evaluator owner supplies concrete source/build/invocation joins,
image/configuration, input execution, timer semantics and target instruction
audit controls. Their declared scope remains in the qualification. Synthetic
controls establish only lifecycle joins and refusals. Missing actual target
controls leave hardware qualification unavailable; no generic parser turns a
copied UART log into hardware authority. Selected files and their ancestors must
be direct paths; matching bytes through a symlink cannot establish ownership.

`observe_protected_final_comparison` executes the complete original snapshot,
callback and numerical lifecycle and returns `ProtectedFinalObservation`.
Its cycle fields are reported callback values and `physical_status` remains
**UNKNOWN**. Strict final comparison arithmetic refuses this distinct type.
`admit_protected_final_comparison` requires a separate independently issued
physical execution domain. Its production issuer is not implemented yet, so
admission and the final campaign explicitly refuse. Callback qualifications,
declared hashes, saved reports, success flags or supplied physical callbacks
cannot replace that issuer. Actual source-to-loaded-hardware, runtime, reset,
clock and timer correspondence must be implemented and observed before final
physical comparisons become available.

## Three numerical and performance roles

The original numerical oracle, frozen handwritten performance baseline and final
candidate have distinct roles. Both performance arms are compared directly to
the original numerical oracle using the existing full-output reconstruction and
the exact original `QualityBudget`. Comparing two approximate outputs to each
other can double the permitted error and is insufficient.

Full reconstruction checks every declared output word, shape and dtype; it keeps
the original exact or elementwise policy and rejects partial readbacks. Both
arms must pass. Admitted performance cycles must come from separately authenticated
execution witnesses; instruction counts, native wall time and analytical bounds
cannot substitute for them. The selected target verifier must apply the
original prohibited-instruction contract to every executable section.

`evaluate_protected_final_campaign` executes admission for the complete original
roster before calling strict comparison arithmetic. It accepts lifecycle inputs,
not precomputed comparisons or caller-supplied success flags. Final results stay
private until sealing. Exposing a final failure for repair ends that campaign;
any subsequent campaign must record the exposure.

## Verification scope

Regression tests execute independently compiled native programs and reconstruct
all numerical outputs. They cover tolerance doubling, original-byte drift,
replaced/private ownership, missing or unbound witnesses, staged/hardware
substitution, incomplete rosters and strict integer parity. The injected execution
verifier uses synthetic cycle records solely to test joins and refusals. These
tests do not qualify a FireSim verifier, an actual image, final workload parity,
fresh participant convergence or historical elapsed-time accounting.
They also check that every diagnostic qualification and supplied physical
declaration is refused by final physical admission.

`component_final_policy.py` owns strict comparison arithmetic.
`protected_final_observation.py` owns the separately typed observation lifecycle;
`physical_final_admission.py` continues to refuse final physical admission until
its independent issuer exists. Keeping these owners distinct prevents observed
callback cycles from being relabeled as authenticated hardware comparisons.
