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
