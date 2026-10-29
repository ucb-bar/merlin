# Ordered-unroll variant coverage proposal — no aggregate correctness flag

All rows refer to the registered pinned4499 `ordered_loop_unrolling` mechanism.
They do not add a pass family or enlarge a historical qualified domain. Mode
selection uses actual static controls/count and selected factor, not model names.

| Variant | Actual producer form / selected domain | Retained actual evidence | Proof / qualification status and limits |
| --- | --- | --- | --- |
| Empty | count0, source lb=ub, positive stride; loop removed, initial carries substitute results | This packet: empty lb3/ub3/stride2, two carries, both final observer operands retained | New complete-entry candidate; arbitrary partial body never executes.25 roots/21 source modules. New Lean UNCOMPILED/native UNQUALIFIED. |
| Fully inline | 1<=count<=factor, factor2..4; actual constant IV plus ordered body clone per lane | This packet: count1/2/3; nonzero bounds/strides,32/64 bits, one/two carries; wide64 case | New complete-entry candidate preserves partial-body and observer failures.24 roots/21 modules per pair. New Lean UNCOMPILED/native UNQUALIFIED. Count4 is source-admitted but no actual retained case here; no qualification claim for it. |
| Bulk without remainder | count>factor and count divisible by factor; factor-wide bulk loop only | Historical sealed Pipeline/no-tail composition and larger-profiles-v1 retained source interfaces | Historical exact qualified proof/source/context remains separately pinned. New gate adapters are uncompiled/unqualified. No rerun or qualification transfer; no widening to tails/inline. |
| Bulk plus remainder | count>factor and remainder>0; ordered bulk loop followed by original-stride tail, carry forwarded | Preserved unroll-tail-v1: trip5/factor2; nonzero/nonunit/two carries; remainder2; wide64 | Source-only complete-entry candidate with structural finite-fold success/bulk-failure/tail-failure witnesses.34 roots/25 modules, UNCOMPILED/native UNQUALIFIED. Not yet composed/qualified with the historical full chain. |
| Other domains | Dynamic/negative controls, unsupported CFG/regions/internal or recursive calls, unproved flags, other widths/factors/carry geometry | No acceptance examples claimed in these new routes | Explicit refusal or a separate required proof/model bridge. Unchanged production boundaries are not changed-pass verification. |

Common modeled observation in the scalar rows is equality of complete entry
returns, modeled faults and terminal world under one registry/primitive/external
interpretation. It must not be conflated with memory possible-success relations,
physical native execution, poison absence or hardware completion. Private SSA
frames differ and are not themselves the observations.

Source-only current paper row: “We provide independently decoded actual empty,
full-inline and bulk/remainder ordered-unroll examples and complete operational
query candidates, with explicit mapping/model premises and bounded valid host
mutation refusals.” It must NOT be upgraded to “the checker proves” until the
new exact roots actually pass authorized checking, qualification and review.

Proposed post-checking sentence: “For the enumerated admitted ordered-unroll
boundaries, acceptance establishes equality of all defined-entry return, error
and terminal-world observations under the disclosed common scalar/SCF semantics;
empty loops skip their bodies and inline/bulk/tail forms retain ordered carries,
partial failures and the complete continuation.” Exact reviewed domain and
source/native trust limits must accompany that sentence.

FUTURE-CHECK-PRIORITY.json prioritizes existing empty/inline/tail source candidates
before any further expansion once an exact common native context is explicitly
authorized. It grants no runtime authority. Panel/double-buffer/gSIM performance
figures remain separate execution evidence. No covered-pass percentage or
all-graph flag is proposed, and no qualified ledger is edited.
