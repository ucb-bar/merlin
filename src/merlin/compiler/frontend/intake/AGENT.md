# src/merlin/compiler/frontend/intake

Keep original receipt, source SSA, type/argument/result and data-byte custody checks.
Runtime inputs, references and evaluator-only members must not enter compiler staging.
Execution-owned results remain an explicit selected ABI with independently rechecked
resource, type and original input restrictions. Provenance grants no omitted semantics.
Source-version, numeric/effect and target execution qualifications remain separate.

Standalone stages replay the same `prepare_kernel_inputs` builder from a closed
compiler projection. Keep evaluator fields outside that projection, preserve
the exact nested roster and source-derived policy, and recheck actual typed
source/ABI/effect relationships rather than trusting a resealed manifest.

Derived models replay `prepare_bound_model_inputs` with a compiler-only parent
receipt projection. Preserve original parent/call/constant relationships and
stream tensor archives. Public bindings join actual source placeholders to exact
stored parameters or explicit runtime state; a BUFFER snapshot is not immutable
consent. Keep the original staged ABI and the prepared runtime ABI distinct.
