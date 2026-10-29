# MACC proof review and precise remaining scope

No concrete soundness counterexample was found in the existing zero-init Exact/Sites/local qualification proof during this source review. This is not a fresh compilation or kernel audit. W has not changed Lean source.

The exposed checked result is to the **reference program**. The numerical explanation is factored through scalar `accRegion_sound`/`truncRegion_sound`, tensor accumulation/truncation/zero-fill, and operation dispatch theorems. Search over current Bf16 modules finds no checked-entry theorem directly concluding evaluation equals `Contract.result`; it is not honest to relabel the current checked root as that stronger packaged statement. The source-definedness claim remains local, not for the enclosing model.

An additive convenience theorem could close that packaging gap if needed for a stronger paper formula (not required to claim the precise current reference-program result):

1. From successful `siteE`, `emit`, fragment reconstruction and strict local checks, derive the actual same-format input/result types, license, maps, iteration extents and checked `StructuredOk` geometry. Do not accept these as unrelated caller premises.
2. Evaluate the actual six-operation replacement under its fresh SSA bindings. Compose empty/constant/fill, accumulating_op_sound and final_op_sound constructorwise. Use real operands and actual final result ID; no assumed whole-region equality or arithmetic laws.
3. Derive an initialized `Contract.result` on all typed initialized local inputs from the total frozen FloatModel operations and checked geometry. Handle zero reduction length and empty output explicitly. No source-success premise.
4. Combine actual fragment mapping with Exact/Sites root to present a statement about the submitted source/target pair. Keep source operation positions and coverage, rather than proving a template alone.

This theorem is **not implemented, compiled, or assumed** here. Existing complete-output/reference claims do not require pretending it exists. Current assignment forbids broad new proof families and retains frozen semantics, so numerical expansion/nonzero destinations/mixed fusion remain out of scope.

Concrete retained refusals needed for MACC qualification: wrong source dataflow (a*a vs a*b), nonzero or negative-zero initialization, named matmul map mismatch, altered multiply/add operands/order, missing fill/extension/truncation, extra flags/rounding attributes, dropped/duplicated site, and unrelated surrounding-operation mutation. All must be actual parser-valid cases when called semantic negative evidence; W's JSON mutation controls establish only host validation behavior.

F14 and F18 remain facts against the broader production claims: nonzero initialization was discarded; fused and unfused bf16 accumulation can differ at overflow. Nothing in this review upgrades arbitrary fusion or downstream numerical fidelity. Actual native outcomes and fresh proof audits are F's separate evidence.
