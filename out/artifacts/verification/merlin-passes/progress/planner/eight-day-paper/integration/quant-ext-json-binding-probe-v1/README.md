# Confirmed typed-JSON receipt association discrepancy

The planner independently replayed the corrected-wrapper initialized kernel receipt after
replacing the source export's block-argument `id: 0` with JSON `id: false` and supplying the
new export file's genuine hash. The source/target MLIR, report and audit were unchanged.

Replay returned `fresh=true`, `checked=true`, `matches_exports=true`, `matches_recorded=true`
and `audit_matches=true`. PROBE.json retains the exact JSON path, API identity and response;
forged-export.json, forged-receipt.json and api-reviewed.py retain the witness and implementation.
The executed helper is the planner's probe_quant_ext_json_binding.py. Session 99061 is terminal.

Cause: Python structural `==` treats integer zero and boolean false as equal. The exported
typed JSON documents differ. This demonstrates an artifact-association validation gap; it
does not refute the Lean arithmetic theorem or show that an incorrect native transformation
was accepted. The quant-ext worker has been asked to enforce typed comparison for exports,
reports and audits, retain an adversarial regression and regenerate affected qualification.
No worker files or original receipts were changed by this probe. No approval is inferred.

A direct invocation of the unchanged frozen Lean checker on that same forged source export
exits1 with `invalid source export`. Exact command/binary identity/output are retained in
direct-decoder-check.json. Thus replay marked checked an export the actual decoder refuses;
the replay regenerated a valid export from unchanged MLIR instead of rejecting the supplied
typed mismatch. The mismatch is not merely different whitespace or object-key order.
