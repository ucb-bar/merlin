# Candidate verification text — independent review pending

This draft addresses the two manuscript FIXMEs. It describes the delivered
structural artifacts, not the intended all-shared-pass end state. Graham should
review it against the exact statements and IR mapping before it enters the
paper. Publication of this draft is not specification approval.

## Candidate main text

We use Lean to validate supported instances of Merlin's shared graph
transformations. The current artifacts cover static shape/view normalization,
exact scalar-integer simplification, common-subexpression and dead-code
elimination, loop-invariant motion,
ordered loop unrolling and segmentation, and initialized buffer reuse, each
within its explicitly checked domain. Trusted parsing and projection translate
the actual before/after IR into proof obligations; Lean checks preservation of
the modeled observations under recorded operation and environment contracts.
Unsupported boundaries are refused. The scalar and composed scalar/loop results
preserve returned values and terminal state on both success and failure. The
view and storage results equate possible complete outcomes, with descriptor and
allocation premises; they do not establish failure or divergence equivalence.
The appendix lists each admitted variant, theorem, mapping and assumption.

A worked example validates an actual four-pass CSE/DCE, loop-invariant motion,
unrolling and segmentation chain. The proof binds all five stage
programs, scoped SSA and loop carries, the unchanged entry roster, and a
post-loop observer. It derives the loop grouping and segmentation correspondence
under one interpretation, including the first failing external call and its
terminal state. The retained checker output and a compiler-valid mutation that
deletes the observer accompany the complete IR. This example is separate from
the panel-pipelining experiment in Figure 7.

These are conditional structural results. Primitive, control, effect, descriptor
and allocator fidelity, trusted IR projection, and independent contract review
remain explicit. The new standalone validators do not yet establish that every
ordinary compilation is gated by these proofs. They do not establish complete
shared-pass or selected-model coverage, numerical correctness, downstream
encoding, backend execution, or hardware correctness.

## Evidence for the first FIXME

Use [COVERAGE.md](COVERAGE.md) for the pass/variant denominator and open rows.
The source revision is `4499eebb959c0f92cefd92cc6d80646ba66474ed`; delivery is on
`feat/pass-verification`. The eight registered common transform kinds are not
the entire shared scheduling/fusion/transfer/storage inventory.

| Result | Authoritative relation and mapping packet | Limits to retain |
| --- | --- | --- |
| Scalar CSE/DCE and LICM | `graph-state-scalar/review/sealed-v1`, including its exact generated roots and projection | Typed SSA and local total-pure/common opaque contracts; host mapping and native fidelity remain separate. |
| Exact integer prefix and five-pass chain | `graph-state-integer/review/sealed-v2`; `Merlin.GraphStateIntegerInstance.complete` | Canonical initialized word entries and local successful-result representation contracts; reached value typing is derived.75 tests,17 originals and one17 replay. Native/poison fidelity and independent review remain separate. |
| Composed scalar/loop chain | `graph-state-pipeline/review/sealed-v1/PAPER_ROWS.md`; `Merlin.GraphStatePipelineInstance.complete` | Static nonnegative no-tail bulk mode, explicit widths/strides/carries and local external contracts; broader unroll modes need their own composed bridge. |
| Shape normalization | `graph-shape-normalization/paper-evidence/README.md` and `standalone-mode-v1/SEAL.json` under that root | Identical casts/static dimensions, descriptor typing and partial primitive meaning; whole typed heap/public observation. The new actual shape/memory composition is checked development, with qualification pending. |
| Initialized reuse and segmentation | `graph-memory-scf/memory-scf-mode-v1` and its paper packet | Complete public outcomes under total/private/unobservable allocation; the retained 8→16-byte counterexample excludes general resource/failure claims. |
| Declared-call/fusion correspondence | Historical `graph-regions-declared/qualification-v1` and its paper packet | Per-instruction, packing, ABI and Action contracts. The stronger wrapper is unqualified; newer capture candidates are development only. |

Packet paths in this table are relative to `out/artifacts/verification/`.
Exact integers now have a qualified scoped v2 row; its earlier v1 host diagnostic
stays unqualified. A remaining static stale-identity error-label path is recorded
in integer-v2-host-status-limitation.json; it is separate from positive theorem
acceptance. Sequential scheduling and broader mechanisms retain their
own narrower contracts/open obligations in COVERAGE. Do not describe the whole
table as reviewed or every graph pass as verified.

## Evidence for the second FIXME

Use the existing unabridged five-program example and explanatory excerpt at
`graph-state-pipeline/review/sealed-v1/paper-evidence/PIPELINE_WORKED_EXAMPLE.md`.
Its `EVIDENCE.json` identifies exact buffers/digests, generated Lean query,
checker/audit output, original receipt, fresh replay and observer-deletion refusal.
The final root preserves all defined entry observations under the stated model;
the separate inhabited provider witnesses do not claim physical delivery.

The frozen packet records 64 tests and eight actual chains changing every pass,
with one retained fresh batch of eight positive replays. Counts overlap earlier
suites and must not be added to them. The coordinator matched 4,001 indexed
evidence files and 20 original/retained source pairs. No test count substitutes
for the theorem or the independent projection/assumption review.

Retain [PAPER_TARGET.md](PAPER_TARGET.md)'s figure boundary: the serial→pipelined
gSIM example has separate execution and instruction evidence. None of the
above proofs certifies that exact asynchronous schedule, its encoding or cycles.
