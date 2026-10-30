# Full-size kernel handoff through Merlin's compiler gates — 2026-10-29

All 16 supplied cases passed using eight exclusive, low-priority CPU lanes. Every
case selected all ten compiler checks. All 106 input arrays stayed unchanged;
all 81 full output arrays matched the pinned reference exactly, with guards and
borrowed descriptors intact. Ordinary and gated LLVM bytes matched in every case.

The all-ten gate was pushed first as `b7626bbf4ec1df88c10dfb026a79e5e36101bf22`
on `feat/pass-verif-integration`; remote tip and GitHub attribution were verified
before any benchmark compilation/execution. The campaign then used that compiler
with the separately recorded, frozen test fixtures and harness.

## Results and proof scope

| Case | Complete output arrays | Accepted changing boundary | Functional result |
| --- | ---: | --- | --- |
| G01 | 1 | unchanged; zero credit | PASS |
| G02 | 1 | unchanged; zero credit | PASS |
| G03 | 1 | unchanged; zero credit | PASS |
| G04 | 1 | unchanged; zero credit | PASS |
| G05 | 1 | unchanged; zero credit | PASS |
| G06 | 1 | unchanged; zero credit | PASS |
| C01 | 1 | prune_dead_pure_tensor_ops | PASS |
| C02 | 1 | prune_dead_pure_tensor_ops | PASS |
| C03 | 1 | prune_dead_pure_tensor_ops | PASS |
| T01 | 32 | cse | PASS |
| T02 | 32 | cse | PASS |
| T03 | 1 | unchanged; zero credit | PASS |
| T04 | 2 | cse | PASS |
| T05 | 1 | unchanged; zero credit | PASS |
| T06 | 3 | cse | PASS |
| T07 | 1 | unchanged; zero credit | PASS |

There are **seven accepted typed rewrite checks**: DCE in C01–C03 and CSE in
T01, T02, T04 and T06. The other nine cases have unchanged DCE/QDQ/CSE captures
and earn zero rewrite-proof credit. Across the campaign: seven verified, 41
unchanged and zero unverified observed boundaries. All required typed checks,
source-support checks and typed chain compositions pass; their soundness is
proved in Lean and their reported axioms are the standard three logical axioms.
These tensor jobs use the audited executable checker and bound receipts; they
do not generate a new per-case Lean query/object tree.

Optional strengthenings remain distinct. The unrestricted boundary strengthening
is established for the three DCE cases and abstracted for all four CSE cases.
Definedness is established for C01–C03 and T01, and **not established for scaled
T02/T04/T06**. Their required typed refinement still passes. No stronger
no-UB, arbitrary-input, concrete IEEE or whole-kernel theorem is claimed.

Options() selects no extra transformations. Unroll, tiling, private buffer reuse,
MACC, configuration and host-fusion transformations were not invoked by these
fixtures; their selectors are requested but recorded `not_observed`, with zero
credit. Q/DQ's observed stage is unchanged because these fixtures express their
explicit scalar epilogues directly. Other per-pass tests are listed in
[TESTS.md](TESTS.md). Q/DQ remains proposed-model and MACC remains draft.

The measured scope is Merlin MLIR input, actual compiler boundaries and native
CPU functional output. LLVM/native implementation and allocator behavior remain
outside the formal proof boundary. IREE, TVM and Voyager pipelines, accelerator
execution, cycle comparisons and tuning require separate campaigns.

## Exact fixtures and provenance

The supplied handoff is pinned at comparison commit
`ebf58f2c11816370ee27f99557a6bd0b5363643f`, file
`examples/gemmini/comparisons/tvm/KERNEL_BENCHMARK_HANDOFF_IREE_TVM_VOYAGER_AGENTIC.md`.
Its unchanged canonical NumPy reference source is pinned by SHA-256
`2ece855c1438b0aceacb58313ee7f45430f6547d5ecef90f1ba76ebc089587b2`.

The test-only MLIR translation preserves original full dimensions and argument
order, signed INT8 products/INT32 accumulation, saturation versus raw exits,
VALID convolution with nonzero supplied borders and bias once, all 32 shared-K/V
heads, both T04 outputs and all three T06 outputs, and FP32-RNE scale 1/128.
G06 is M=784,K=256,N=1024. This fixture translation is a trusted test boundary,
qualified by parser/contract checks and finite full-output comparisons.
Explicit generic reductions avoid the installed xDSL named-matmul mixed-width
region-verifier incompatibility; no production compiler algorithm was changed.

The native test observes the actual selected Clang's 64-bit pointer ABI and raw
data layout, retains compiler hashes and probe records, and explicitly selects
MLIR index64 for both ordinary and gated builds. No default p0 field is invented
and no production layout policy is weakened. The first parallel attempt's 16
missing-layout harness failures remain in `campaign-v1`, with zero proof or
correctness credit. A failed strict host-observer preflight and the corrected
small real C probe are also retained. Campaign v2 is the successful run.

## Suite and reproduction

- [Full kernel tests and 26 host controls](../../../../merlin/tests/ir/test_kernel_benchmark_formal_gates.py)
- [Bounded parallel runner](../../../../merlin/tests/ir/kernel_benchmark_handoff_campaign.py)
- [MLIR fixture builder](../../../../merlin/tests/ir/kernel_benchmark_handoff_inputs.py)
- [Immutable reference helper](../../../../merlin/tests/ir/kernel_benchmark_handoff_oracle.py)
- [Portable contracts and complete array hashes](../../../../merlin/tests/data/kernel_benchmark_handoff_contracts.json)

With the compiler/Lean tools configured as in the [gate guide](../../../../docs/guides/lean_pass_gate.md),
use a fresh output directory and CPU IDs available on your host:

```sh
nice -n 19 taskset -c 28-35 python merlin/tests/ir/kernel_benchmark_handoff_campaign.py \
  --out out/build/kernel-handoff-rerun --jobs 8 --cpus 28..35 --case-timeout 1200
```

The runner sets the full-size opt-in and required-tool flags, limits each case
to one numerical thread and one Lean worker, and isolates cases in owned
subreaper processes. CPU leases return only after confirmed cleanup. Skips,
timeouts, drift and failed gate checks are failures, retained per case.
Normal pytest execution runs the small controls and explicitly skips the 16
expensive cases unless `MERLIN_RUN_KERNEL_HANDOFF=1` is set.

## Artifacts and other tests

- [Per-case results](RESULTS.csv): `elapsed_seconds` is full pytest-case walltime, not kernel performance
- [Compact source/array/coverage/claim records](summary.json)
- [Other integration tests, suite locations and corrections](TESTS.md)
- [All 248 identified retained JUnit cohorts](ALL-TEST-COHORTS.csv)

Raw process/JUnit logs, exact LLVM/native products, case reports and receipts are
retained locally under `out/build/kernel-benchmark-formal-gate-20261029/campaign-v2/`.
The independent audit is under
`out/build/validator-alignment-20261029/kernel-benchmark-results-audit-v1/`.
Saved records provide reproducibility information, never live proof authority.
Historical cohorts retain their exact source cuts and overlap; no grand total
across them is reported. This task did not run the full repository suite.

The consolidated inventory preserves 232 earlier JUnit records and adds the 16
successful campaign-v2 records. Five historical header/child inconsistencies are
explicitly flagged in [TESTS.md](TESTS.md); two interrupted zero-test files receive
zero pass credit. The original local inventory remains an unchanged historical
snapshot, while this report contains the count corrections and later campaign.

The separately [imported colleague audit](../frozen-followup-20261029/HANDOFF.md)
reports 207 distinct overlapping tests and a full Lean guard at `591687da8`,
alongside six unexpected admission-error regression failures, nine incomplete
handler-result protocol failures, and a bad UNROLL blocked with an `unavailable`
classification. These are not team reruns and are excluded from the 248-file
inventory; their raw JUnit is not bundled here. Those controller gaps remain open
for separate review and current-cut requalification. The 16 successful kernels
do not establish universal gate/controller hardening.
