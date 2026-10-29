# Actual checked example: unroll then ordered 1D segmentation

This is the successful **Qv6/Fsource-cut-v3 qualification-v2** example. Exact
copies, original byte refs and current source cut are retained in BINDINGS.json
and batch-SOURCE-CUT.json. There are44exact root/type statements and a927-theorem
transitive standard-three-axiom audit, zero violations. The actual consumer
received exactly stage2.mlir; a receipt alone cannot grant delivery authority.

A complete function with two i32 arguments/carries executes indices
`5,8,11,14,17,20,23,26,29,32,35,38,41,44,47`. The original upper49 is deliberately
not aligned with stride3. Each iteration computes two carry updates, calls
`@step` with the actual IV, and executes two ordered `@observe` calls. After the
loop it calls `@watch` and returns both results. An unchanged function and all
four private declarations are retained in every complete module.

The actual pinned4499 producer unrolls by3 into group bases
`[5,14,23,32,41]`, group stride9 and aligned bulk cutoff50. Segmentation tile2
forms ordered groups `[[5,14],[23,32],[41]]`; outer stride18, inner upper
`min(base+18,50)`. The final one-group segment is short. All original lane
indices, carries, opaque operand order and final observer remain covered.

Read stage0.mlir, stage1.mlir and stage2.mlir for complete actual before/middle/
final IR. Original input is retained separately. LoopTransformQuery.lean is the
actual generated obligation. ROOTS.json and audit.stdout/types.stdout retain
kernel output, exact final/support types and complete audited imports. The
query.process.json/audit.process.json/types.process.json files give the original
native command transcripts and return codes. No fabricated output or rerun was
used to assemble this packet.

The final theorem is `Merlin.GraphStatePipelineInstance.complete` under the common
explicit index/word/external model: all entry observations preserve return values,
errors and terminal worlds. See ../paper-evidence/README.md for the full IR-field
map and premises. Word primitives are total pure functions; retained external
operations may fail and change the world, requiring result arity only on success.
MLIR-to-model projection, native/backend fidelity and operating-system transport
are trust boundaries. No source-success or whole-run-equality assumption appears.
The independent nonzero successful/failing model witnesses are in ../witness-v1;
they instantiate another retained actual example and are not numerical execution
of this example's opaque functions.

A parser-valid observer deletion is retained in
negative-observer-deleted.mlir; its actual Lean query
failed `F4.suffixAccepted34` with ordinary rc1. Swapped effect order and wrong inner
cutoff fail their own expected obligations. These direct component checks grant
no gate authority. Actual gate attribution refusals are recorded separately.

| Evidence | Status |
| --- | --- |
| Complete actual three-buffer producer and source binding | Checked in original Qv5 gate |
| Kernel complete root/types/transitive std3 audit | Passed; retained raw output |
| Real same-live gate → consumer exact final bytes | Passed in qualification-v2 |
| Source→term parser/projection correctness | Explicit Python trust |
| New template, original8positive roster,15mutations,12unsupported, fresh rebuild replay | Passed in qualification-v2 |
| Native semantics/backend/allocator/alias/completion | Not claimed |
| Independent human review/policy approval | Pending |

This packet is historical Linux-host evidence. Its absolute paths are not laptop
qualification. Rebuild/recheck in a fresh output with current exact source pins;
never treat copied JSON as a live completion.
