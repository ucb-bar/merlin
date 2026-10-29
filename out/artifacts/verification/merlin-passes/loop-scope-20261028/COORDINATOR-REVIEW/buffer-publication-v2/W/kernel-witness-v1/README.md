# Actual changed-kernel nonvacuity — retained two_epochs

This additive witness uses the ORIGINAL four captured buffers and the already checked complete source/final terms. No example or raw IR was changed. `RAW-BINDINGS.json` names the raw files/hashes; `ACTUAL-PLAN.json` was freshly reconstructed using the SDK-free parser/projection. Python proposes constants/states, but Lean independently checks every instruction-key/specification binding and every operational transition.

## Concrete observation

| Item | Value / proof |
|---|---|
| Scalar carrier | `Int`, with the explicit common relation `result = left + right` |
| Literal relation | value `1`, for the actual module's scalar literal |
| Kernel inputs | seeds `3`, `5`; caller descriptor object `2`, shape `[1]` |
| Initial public heap | object `2`, cell `0` contains `7`; all other cells absent |
| Private allocation partition | logical objects below `2`; both initially uninitialized |
| First initializer | actual scoped 2×3 stores: all six cells contain `3` |
| First read/update/touch | private value `3` → `6`; caller `7` → `13` |
| Second initializer | actual scoped 2×3 stores: all six cells contain `5` |
| Second read/update/touch | private value `5` → `10`; caller `13` → `23` |
| Kernel return | scalar `36` and index `2` |
| Final public heap | caller cell is `23`; every other cell unchanged |
| User state | `99`, unchanged |

The retained touch order contributes the first result `13` and second result `23` to the returned `36`; it is not an empty or Unit-valued observer. The witness does not expose private object identities/clocks. Allocation totality/private isolation remain properties of the selected model, not native allocator conclusions. The preserved 8→16-byte peak-memory counterexample still rules out a general finite-capacity claim.

## Proof construction

- `KernelWitnessBase.lean`: actual typed arguments, nonprivate caller descriptor, inhabited entry domain, initialized caller cell, initially absent private cells.
- `KernelWitnessStates.lean`: explicit candidate intermediate memories; all retained specifications are checked against `plan.specifications`, and the entire 28-instruction list is proved equal to `kernel.instructions`.
- `KernelWitnessInitializers.lean`: derives BOTH actual SCF initializer executions using the existing constructive `uniform_heap` theorem, including all-six-cell coverage. No initializer-success premise is accepted at the final root.
- `KernelWitnessSteps.lean`: constructs every remaining allocation, free, load, store, scalar, observer and return transition in the existing SCF/epoch semantics. Only the two first loads require initializer facts, which the preceding module constructs.
- `KernelWitnessExecution.lean`: composes all 28 steps into a complete source run; derives original source success through the already proved typed/shape adapter; transfers the exact public observation through both selected-entry and all-entry checked endpoint theorems. `actual_kernel_nonvacuity` conjoins inhabited entry domain with successful actual source AND final observations under the SAME explicit interpretation. `ordered_observer_witness` retains the actual intermediate touch values and initializer coverage.

No whole-run equality, successful-source execution, arbitrary effect law, numerical theorem, allocator conformance or hidden world equality is assumed. Int addition is an explicit witness instantiation of the existing abstract scalar relation; this is NOT a proof that native i32 arithmetic/overflow implements it. The scalar/literal relations are chosen concretely, not universally promised total for other instances. The endpoint theorem itself remains polymorphic.

## Evidence and limits

Numbered build directories preserve the exact source at each build and first failures (missing Decidable instances were repaired by explicit constructor/domain proofs). Existing main theorem objects and all old capsules remain unchanged. `ROOTS.json`, `audit.stdout`, `types.stdout` and process records contain the completed new audit evidence: 76 witness root types, 2,441 transitively audited theorems, only propext/Classical.choice/Quot.sound, zero violations. `SOURCES.json` and `owned-source.patch` are additive source-only delivery. The one extra existing dependency is `Merlin.GraphLoopsMemorySCFWitness`, separately compiled unchanged; selected query/dependency binding remains `../candidate-review-v2/{PROJECT_CLOSURE,QUERY_BUNDLE}.json`.

This establishes nonempty outcomes for the retained changed kernel in the model. It does not establish all inputs terminate/succeed, categorized failure or divergence preservation, native MLIR/hardware fidelity, universal allocator behavior, broad aliases/async/tiling support, fresh gate qualification/replay, or human approval. Q owns generated-module admission/replay binding, F qualification, and root publication. W has changed no canonical source for this witness.
