# Confirmed HOST association gap — repair takes priority

Independent planner probe:
[PROBE.json](../../../progress/planner/eight-day-paper/integration/w8a8-ordered-contract-binding-probe-v2/PROBE.json).
The v1 environment failure remains preserved separately. Probe v2 changed only the report's
exponent_contracts.original record to a different valid cap254 file while the checked snapshot
remained cap127. After rehashing the report and receipt, ordered_policy.replay accepted and
returned requested_prefix_checks_passed=true. Original artifacts were untouched.

This is a **HOST original-to-snapshot association defect**. The receipt domain hash still
names the actual checked snapshot, so it is not an arithmetic, Lean theorem or numerical
bound counterexample. The claimed association to the original submitted contract is unsound.
No final qualification or release claim may silently omit it.

## Preserve the current run

The frozen ordered capsule remains unchanged: runtime d3ee5e2c80dd33f7269f8d5bb88e7e18c09da9c719b7b3c109961792841b731b,
1452records,26newsourcefiles. The109-case final suite55755 continues to completion unchanged,
with retained receipts/native comparisons. Its passing tests do not repair the new independent
counterexample. Keep all results, the26-file patch, complete115-file patch and frozen build
as historical/provisional evidence; never edit them into an apparently repaired release.

Root explicitly assigned: after that suite, create a NEW wrapper/profile for the repair,
covering BOTH input_contracts and exponent_contracts, strict JSON types and freshness after
replay. Trig qualification is deferred; its separately named mathematical scratch is retained.

## Concrete repair design

- Add a reusable owned file-reference/snapshot-binding helper. Validate exact field sets,
  path/hash string types and nonnegative integer byte lengths with bool explicitly rejected.
  Check each live file against its recorded reference. Require exact typed SHA256 AND byte
  equality between original and snapshot; JSON semantic equivalence is insufficient. Keep
  path identities separate from content identities and do not infer capture authenticity.
- A NEW receipt wrapper calls the unchanged sealed ordered checker and binds its base receipt,
  base report, both contract pairs and the new wrapper/helper source inventory. The wrapper
  cross-checks its binding records against the actual base report rather than trusting an
  independently edited duplicate. Use a new boundary/schema; never relabel an old receipt.
- Replay validates both bindings before invoking fresh checks, reconstructs the same domain
  from byte-identical contracts, compares regenerated evidence with strict typed JSON equality,
  and rechecks original/snapshot/source/report freshness AFTER the fresh checker and before
  publication. Preserve snapshot-domain theorem roots and all broad false flags.
- Tests must include independently valid cap254-for-cap127 original substitution; valid
  integer-contract substitution (or valid byte-different JSON) with an unchanged snapshot;
  substitutions in both copies of any repeated binding record; bool/int metadata confusion;
  changes to original contracts during replay; and genuine positive replay. Preserve the
  exact old counterexample rather than replacing it with an invalid-input failure.
- Use a NEW isolated build/runtime path, then qualify the repaired route with actual-pass
  positives, compiler-valid mutations, fresh audit/identity, retained TinyLlama166source/
  163reference demands, separate-process replay and post-identity checks. No old frozen source,
  spec, production, shared Main/CLI/ledger or foreign file edits; no commits/integration/push.

The final implementation should expose content association and lifetime checks as explicit
interfaces, rather than adding a special-case check only for the observed exponent record.
Artifact authenticity, independent policy review and capture admission remain separate.
