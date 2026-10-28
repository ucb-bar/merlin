# F MACC source delivery v2

Four owned test/qualification sources; exact snapshots and addition patch in SOURCES.json/owned.patch. Q integration nine-source v2 is separate. No production/Lean theorem/frozen semantics/spec/lock changes. v1 source cut and every preparatory error remain preserved.

Changes since F v1 are runner corrections only: canonicalize file-reference paths consistently with the gate, set the repository Lean working directory for independent segment audits, and reuse this task's already-completed exact native audit after binding unchanged source/tool identities. The full Main/project audit succeeded once; no failed native theorem audit was promoted. Runner formatting was finalized during the receipt job; the kernel checker, Q source/semantics, exports and proof/root bindings remain fixed. Treat generated receipts as direct candidate evidence with explicit audit provenance, not an approved live gate run.

Final outcomes, raw evidence and pending approval/build limits are in ../REPORT.md, ../FINAL_STATUS.json and ../HANDOFF.md once the already-running receipt/replay job completes. Root alone publishes; no SDK/build artifacts belong in this source patch.
