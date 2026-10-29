# Independent confirmation of the typed-JSON binding repair

The exact false-ID probe that the earlier wrapper accepted is now refused against a fresh
`qualified-strict` initialized-kernel receipt. The source export's block argument id0 was
changed to JSON false and rehashed; source/target MLIR, report and proof audit were unchanged.

Replay reports `fresh=true`, `checked=false`, `matches_exports=false`,
`matches_dependencies=true`, `matches_recorded=true`, `audit_matches=true`. It does not
reproduce a recorded refusal. PROBE.json and the retained forged export/receipt/API snapshot
record the actual response and source identity. The original accepting witness remains in
quant-ext-json-binding-probe-v1. Session20445 is terminal and successful as a negative probe.

This is an independent end-to-end host binding check, not a new arithmetic proof, full
qualification, native-fidelity result, policy approval or clean-environment reproduction.
It modifies no worker files. Positive/current/refusal receipt replay is recorded separately.
