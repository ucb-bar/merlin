# src/merlin/compiler/frontend/intake

Keep original receipt, source SSA, type/argument/result and data-byte custody checks.
Runtime inputs, references and evaluator-only members must not enter compiler staging.
Execution-owned results remain an explicit selected ABI with independently rechecked
resource, type and original input restrictions. Provenance grants no omitted semantics.
Source-version, numeric/effect and target execution qualifications remain separate.
