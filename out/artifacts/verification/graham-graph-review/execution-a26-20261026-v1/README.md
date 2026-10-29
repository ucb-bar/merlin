# Fresh a26 graph proof reproduction

Three workers completed source/claim reconciliation, independent SDK setup and
exact candidate compilation on 2026-10-26 Pacific (2026-10-27 UTC). The source
baseline is **c62f81a8a11347ff53b60101fa75280e69d002a6**. The review evidence was
read at **5a7f872db33d378d6b0c4a294e4099e80efd4956**, whose source base is
**58dbc7c00**. All jobs finished. No compiler implementation or frozen input was
changed, and no public compiler gate was activated.

## Source and claim review

The ten changed Python projection files add admission wiring and source-closure
binding. A new Python 3.12/xDSL 0.68.0 environment passes 32 independent controls
with zero failures or skips. The cases include raw resource metadata before
parsing, structured/nested resources and symbol references, indirect local calls,
direct-call signatures, parser-state repetition and positive controls. Exact
loaded-source hashes and raw commands/results are retained in
[agent-1](agent-1/CHANGED_SOURCE_REVIEW.json).

The historical 81-case runner and original producer closure were not transferred;
that replay is unavailable. The tested admission helper does not prove raw MLIR
meaning, the host matcher, primitive/effect/allocator interpretations or call-body
correspondence. The [scope table](agent-1/SUPPORTED_SCOPE.csv) states the admitted
variants and remaining assumptions. [The existing deliverable updates](../README.md)
explain the effect on coverage, IR correspondence and theorem-strength findings.

## Fresh SDK

The official Lean 4.35.0 Linux ZIP is **925899528 bytes**, SHA256
`c8531faaa4a93d5a4dae0c742cbc1bd42a2993594d02f93a9d629ebf299fb2c8`.
The intake checked 18571 ZIP entries and extracted/hashed 17938 regular files,
with declared expansion 3365301905 bytes. All eight finite diagnostic commands
passed: version/prefix and Init/List/Omega compilation, theorem/axiom inspection
and diagnostic evaluation. The [SDK admission report](agent-2/READY.json) records
official API/tag identities, complete member pins and fresh loader/runtime paths.

The first static `readelf` enumeration selected the official `libc++.so` text
linker script by suffix. Its failure was preserved, and the successor inspection
explicitly classified ELF magic. No member/hash/layout policy was relaxed and no
native diagnostic failed. This new context does not resolve the old runtime
incident, approve old contexts or establish native/MLIR semantic fidelity.

## Exact proof attempts

Every compilation ran serially with `-j1`, explicit fresh-only library/import
paths, a 180-second command limit and a 1200-second initial batch budget. The
327 frozen source/query/witness units still match their pins after execution.

| Profile | Dependencies | Exact case-query result | First actionable blocker |
| --- | --- | --- | --- |
| Q schedule | 7/7 compile | 3 compile and audit: two changed, one identity | None in this selected profile. |
| Q empty/inline | 23/23 compile | Empty compiles/audits; four nonempty queries fail | `Query.lean:51`: counted-continuation rewrite does not match concrete Int-index expressions. |
| Q tail | 24 compile; module 25 fails | All four queries unexecuted | `GraphStatePipelineTail.lean:33`: nil-prefix/negative-arity proof goal remains open. |
| W caller-context | 40 compile; module 41 fails | All 14 queries unexecuted | `GraphLoopsShapeAfterRename.lean:115` and later: renamed argument/environment/binding proof obligations. |
| W entries | 40 compile; module 41 fails independently | All 12 queries unexecuted | Same frozen renaming module and errors. |

Across **38 case queries**, four compile and pass exact root audits, four fail
and 30 are unexecuted. All five separate inline witness files were attempted and
failed. The empty witness's concrete evaluations fail at lines 119/131/143;
the other four also contain the nonempty query rewrite failure. No new
non-vacuity approval is inferred. Unexecuted expected-refusal cases are not
counted as refusals.

The four successful original queries have **76 requested final/support
root-audit occurrences**, all confined to `propext`, `Classical.choice` and
`Quot.sound`. Exact checked types, source identities, commands, stdout/stderr,
exits and per-profile first failures are in
[the reproduction summary](agent-3/REPRODUCTION_SUMMARY.json),
[the complete case roster](agent-3/CASE_ROSTER.csv) and its raw evidence.
The first staging/scanner and unsupported `pp.width` wrapper errors are preserved
as reviewer-tool failures. Only successor audits with exit zero and complete
root/type/axiom parsing contribute to the successful counts.

## Separate proof repair and next work

[TailLoopAppendProofSuccessor.lean](proposed-repairs/TailLoopAppendProofSuccessor.lean)
proves the exact original `loop_append` statement by adding an explicit remaining-
list case split in the failing branch. Its [audit](agent-3/TAIL_PROOF_SUCCESSOR_RESULT.json)
uses only `propext`; the statement SHA256 is
`4fd7176f33b33c2dee36618eb83f94070afdca793a473e01e3a14e71d4b319d4`.
It is a standalone proposed repair, not a repaired or qualified tail profile.

The next work is to create separate source successors for the inline cast
rewriting, inline witnesses and W renaming proofs, and incorporate/recheck the
tail repair through the entire module and four queries. Then run semantic
mutation/replay qualification and complete human review, actual IR/native
correspondence and compiler consumer integration. No all-graph, whole-model,
numerical, hardware or timing claim follows from this run.

## Publication and integrity

[SUMMARY.json](SUMMARY.json) joins the three worker results.
[PARENT_INTEGRITY_CHECKS.json](PARENT_INTEGRITY_CHECKS.json) checks the earlier
worker report manifests and all 469 baseline source/data pins;
[FINAL_REPORT_CHECK.json](FINAL_REPORT_CHECK.json) checks 662 proof-worker output
files and the 43-row case/witness roster. No mismatches were found.

[PATCH_RECOVERY.json](PATCH_RECOVERY.json) shows that 29 omitted prerequisite
source payloads reconstruct exactly from the retained patch; they were not
applied to primary. The integration documents and proposal generator remain
missing, and live enforcement remains unestablished.

Read [PUBLICATION.md](PUBLICATION.md) and [PUBLICATION_INDEX.json](PUBLICATION_INDEX.json)
for the transfer boundary. SDKs, dependency installations, staged sources and
compiled objects remain outside Git. Absolute local paths are preserved as
run provenance and do not grant portable runtime authority.
