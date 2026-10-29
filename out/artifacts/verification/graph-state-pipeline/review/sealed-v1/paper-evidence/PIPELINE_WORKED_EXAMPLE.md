# Actual five-stage worked example

Source: pinned4499 production APIs, i32 case in the frozen eight-member qualification population. All exact buffers/digests, retained query, checker/audit outputs, original receipt and fresh replay report are in `EVIDENCE.json:new_pipeline`. The checked relation is model-relative, complete defined-entry observation preservation; independent projection/assumptions review remains pending.

```mlir
// Source: inside for %iv = 0 to 24 step 1, carry %acc:
%invariant = arith.addi %bias, %bias : i32
%duplicate = arith.addi %bias, %bias : i32
%a = arith.addi %acc, %duplicate : i32
%a2 = arith.addi %a, %invariant : i32
%v = func.call @partial(%a2, %iv) : (i32,index) -> i32
scf.yield %v : i32
// A separate dead add is before the loop.

// CSE/DCE middle: the dead add disappears; %duplicate aliases %invariant.
// LICM middle: %invariant is computed once above the loop.

// Unroll middle: bulk for %g = 0 to 24 step 2, exact threaded carry:
%v0 = func.call @partial(add(add(%acc,%invariant),%invariant), %g)
%offset = arith.constant 1 : index
%iv1 = arith.addi %g, %offset : index
%v1 = func.call @partial(add(add(%v0,%invariant),%invariant), %iv1)
scf.yield %v1 : i32

// Segmented final: outer step 6, inner step 2, same two-lane body:
%sum = arith.addi %outerIV, %blockStep : index
%end = arith.minui %sum, %upper : index
%innerResult = scf.for %g = %outerIV to %end step %bulkStep
    iter_args(%acc = %outerCarry) -> i32 { /* actual two-lane body above */ }
scf.yield %innerResult : i32

// Retained after the loop in all stages:
func.call @final_observe(%result) : (i32) -> ()
func.return %result : i32
```

The `add(...)` lines above are explanatory shorthand, not submitted MLIR. The unabridged canonical actual stage buffers are authoritative. The source's external declarations and the untouched scalar function remain in the complete module.

Each meaningful edit is represented in the proof. CSE aliases equal `Expr.two(.add32)` trees under checked operand bindings; DCE removes an unused total pure node. LICM rebases the same expression through checked lexical `lift`/capture maps. Unrolling is not asserted as a matching trace string: the actual raw `B3` body is proved to execute the raw `B2` body twice, with exact incoming/outgoing carry, the concrete `%g+1` IV and first-error/terminal-world behavior. Segmentation's actual add/min/inner operands and complete raw `I4` body are tied to the bulk body by `Scope.check_sound`; extra private IV/min/sum/carry cells are mapped explicitly, not equated by number. The proved finite range and packed outcome folds instantiate W's grouping and segmentation roots. Exact continuations retain the post-loop observer; `complete` also binds the untouched function/entry roster.

The actual root `Merlin.GraphStatePipelineInstance.complete` and per-function roots are kernel checked. The i32 query's transitive audit covers 891 project theorems, uses only propext/Classical.choice/Quot.sound, and reports no violations. Raw compilation buffers are empty on success; retained audit/std3/statement output and receipt status provide the checker result. The frozen batch has 64 passing tests, eight originals and exactly one retained positive fresh-process replay batch of eight, with unchanged postcheck bindings. The model witness separately proves [3,5]→[17]/world14 and the IV4 fault error17/world16 under its explicit provider; it is not a runtime delivery claim.

Negative case: the frozen suite's compiler-valid `drop-observer` mutation removes the post-loop `@final_observe` from the actual final program. Complete continuation normalization cannot match the original effect skeleton, so no receipt is issued. EVIDENCE lists that exact refusal record and all other valid mutations/coherent forgeries. Operand-order changes are refusals of the strong ordered-primitive contract, not a claim that native integer addition is noncommutative.

The user-supplied panel-pipelining/gSIM figure is execution evidence for another boundary. It receives no Lean status from this scalar/SCF example, and this packet performs no performance remeasurement.
