# W8A8 verification handoff

## Briefing for a new planner

The user wants this conversation to remain the **W8A8 implementation worker** while a separate
agent plans and coordinates. The current approved [eight-day scoped paper goal](../../progress/planner/eight-day-paper/README.md)
targets **October 30, 2026, US/Pacific**. Preserve the complete-pass goal and coverage ledger;
the scoped paper milestone does not redefine the complete pass as already verified.

Read these current sources in order:

1. Root `AGENT.md`, `CLAUDE.md`, `docs/reference/architecture.md`, local `AGENT.md` files and
   `docs/design/merlin_lean_verification.md`. The filename here is singular `AGENT.md`.
2. [Planner onboarding](../../progress/planner/README.md), the active eight-day goal above,
   and its [combined-integration packet](../../progress/planner/eight-day-paper/integration/README.md).
3. This worker's [active status](../worker-status/README.md),
   [modular policy packet](../modular-policy/README.md),
   [rotary qualification](../rotary-domain/README.md) and
   [earlier numerical qualification](../numerical-domain/README.md).
4. The other workers' active records below. Old paused statements and historical result-1256
   blockers are superseded by later dated records. Do not count overlapping suites twice.

The verification interval is `model2MLIR -> Merlin-owned shared passes -> OOT compiler -> LLVM`.
Our target is Merlin's shared transformations, particularly `lower_contraction_int8`.
LLVM-owned MLIR optimizations, model2MLIR itself, OOT lowering and backend execution are outside
this proof claim. Use Lean-proved checkers over actual before/after bytes; tests and native
comparisons are qualification and fidelity evidence, never substitutes for soundness proofs.
W8A8 changes the numerical policy, so its relation is to the declared policy reference,
not literal equality to the original floating program.

| Owner | Checkout and current assignment | Authoritative resume source |
| --- | --- | --- |
| This W8A8 worker | `/home/jeremy/Projects/merlin`, `feat/pass-verification`: numerical/source proofs, modular policy correspondence, receipts and qualification | `w8a8/worker-status/README.md` and `w8a8/modular-policy/README.md` |
| Frontend/TorchAO worker | Canonical qualification in `/tmp/merlin-pv-frontend-resume`, `pv/frontend-captured-call`; active compatibility work in `/tmp/merlin-pv-frontend-byte-compat`, `pv/frontend-byte-compat` | `/tmp/merlin-pv-frontend-resume/out/artifacts/verification/merlin-passes/torchao-ieee/active/README.md` |
| Quant-ext worker, formerly Qinner | `/home/jeremy/Projects/merlin-quant-ext-pass`, `feat/quant-ext-lowering-verification`: per-tensor i8 dequant/mixed-pass qualification, host provenance and reciprocal-law specialization | `/home/jeremy/Projects/merlin-quant-ext-pass/out/artifacts/verification/merlin-passes/quant-ext/active/README.md` |
| New planner/coordinator | Assignment/coverage/assumption ledger, interface review, complete source deliveries, concrete combined candidate, independent review and clean reproduction | `progress/planner/README.md` and `progress/planner/eight-day-paper/README.md` |
| Independent teammate / Agustin | Capture production, independent policy/source-interface review, boundary census or clean-environment replay; frontend still owns its capture-binding proof | Planner coordinates these inputs without duplicating worker edits |

Read-only observations from the other workers on October 22: frontend reports a completed
canonical IEEE packet with 118 tests, an 11,357-theorem/zero-violation audit and four positive
replays, locally committed through `c29d71090`; native NaN fidelity and review remain separate.
It is now investigating compatibility with the W8A8 byte extension. Quant-ext reports 104
affected tests and ongoing frozen receipt/replay qualification; an integer reciprocal-word
proof exists but still needs its actual FloatModel/checker connection and semantic review.
These are worker-reported results, not independent requalification by this W8A8 worker.
Refresh their active records before assigning dependent work. Do not restart the historical
integer/reshape/splat LLVM-rule workers or give a teammate the already-owned frontend task.

W8A8's current qualified source-prefix result checks 163 bounds, including all 158 source
dequantizers, through 1,060 source and 1,087 reference operations. It proves positions `[0,8)`
and proposes exponent cap 250 on frequency argument 201. The input domain is explicitly
finite/nonzero f32 plus declared integer intervals/caps; its input witness is proved, but
no actual archive is claimed to satisfy it. Complete numerical execution still refuses
quantizer **1533**. The rotary packet records 334 passing tests and 9,974 audited theorems,
with eight positive boundary replays and the independently replayed retained-model prefix.

The newer modular checker separates that full-model obligation: matching the whole target
and checked suffix to the fixed draft reference proves equal outcomes in the named semantics,
without additional numerical laws or original-source-definedness premises. Successful and
nonpoison reference execution are separate transfer theorems. The complete retained TinyLlama
boundary matches and independently replays; **33 frozen tests pass**, with **9,984 theorem
audits and zero violations**. All 1,309 recorded files remain unchanged. The final packet is
`modular-policy/qualification-v1.json`, with both current and complete source-delivery patches.
This does not independently approve the draft policy, establish native semantics fidelity,
or make every production compilation automatically checked.

The remaining substantive obligations include independent recipe/source-interface/domain
review, selected-capture/version/scale binding, common numerical-model compatibility, the
seven source and three dynamic laws plus arithmetic agreement in the stronger numerical
routes, ordered floating accumulation/transcendental/softmax facts, remaining production
variants, and one fresh combined qualification. Review is **not** the only remaining task.
The frontend's canonical min/max model and W8A8's payload-preserving model differ on NaNs;
their local theorems cannot silently be composed. Frontend owns the compatibility proof.

Agustin is the nominated independent reviewer. The user approved only the explicit all-zero
row scale rule `0.00001`. References were found in `examples/gemmini` at pinned EL4 commit
`0cc289de52a10bed256f32c90cfbb54a21b3c81f`: historical TorchAO 0.16 for TinyLlama/ResNet and
0.15 for SmolVLA. This is not final selected-capture binding or blanket policy admission.

Main was observed at `229f42c627854c49476a101cb393d0b15044fab8`. The shared checkout contains
intentional foreign dirt, including Main/CLI and central ledgers. Do not reset/stash/switch,
prune old worktrees, broadly stage, modify protected/frozen definitions/specs/locks or repair
production passes. W8A8 and quant-ext staging/commit/integration/push holds remain; frontend
retains its already authorized local-commit permission, with no push/integration. A teammate
pulling the branch will not receive these uncommitted candidates and local artifacts.

The modular packet prepares a complete **68-file additive W8A8-owned source/test patch**
against that main base, in addition to the individual milestone patches. Private build-only
Main configuration is excluded. Preserve every old build/counterexample and regenerate
receipts for any combined or relocated identity. The planner must review the concrete full
source closure and shared interfaces before seeking release of the integration hold.

Never claim full verification from a test count or arbitrary percentage. Keep unsupported
boundaries explicit, all four broad verification/review/model-existence flags false, F11/F12/F13,
RMSnorm/rotary overflow witnesses, scalar-square/bf16/overrank counterexamples, native NaN
differences and the historical undiagnosed JSON failure intact. The unrelated structure gate
still fails on `zephyr_model.py`, 3557 > 3537 lines. No deadline permits weakening a theorem,
an invented Agreement premise or implementer self-approval. Do not send Slack/email messages
without explicit authorization; drafting a question is not authorization to send it.

**Current resume:** the user approved an [eight-day scoped paper goal](../../progress/planner/eight-day-paper/README.md)
on October 22, targeting October 30. Work is active; the older pause statement below is
historical. Read [worker status](../worker-status/README.md) and the
[rotary-domain qualification](../rotary-domain/README.md) before using the older numbers.
Latest modular result: 33 tests, 9,984 theorem audits with zero violations and independently
replayed complete TinyLlama target/reference correspondence. The preceding rotary result
retains 334 tests, eight positive receipt replays and all 163 prefix demands. These test suites
have different scopes; no cumulative unique-test total is claimed. Complete numerical W8A8
still refuses result 1533. Four broad verification/review/model-existence flags stay false.
Next: review the complete owned source delivery and scoped claim, coordinate policy/capture/
common-model obligations, and prepare held combined qualification. This worker retains the
attention/1533 numerical stretch track. No staging/commit/integration/push is authorized.

## Historical October 21 checkpoint

Paused at the user's request on 2026-10-21 (Pacific). No further implementation is running
from this session. **The complete W8A8 pass is not yet verified.** This report distinguishes
proved components, conditional numerical claims, passing engineering tests and unresolved
model attempts.

## Scope and working-tree ownership

Verify Merlin's shared `lower_contraction_int8` W8A8 pass between model2MLIR and the OOT
compiler. LLVM-owned passes are trusted infrastructure. Use one agent unless asked otherwise.
Do not edit production transforms while verifying them, or change frozen semantics,
specification locks, statement pins or numerical assumptions to make a case pass.

Branch: `feat/pass-verification`. Last checkpoint pushed by this session:
`7304d8e501c090782c81e25a8471038fc8a542ac`. Another agent subsequently committed
`229f42c62` (Qinner decoding); that work is independent and excluded from this qualification.
Inspect current history/status before resuming; another Codex instance is actively editing
the shared checkout. Do not switch branches, reset the tree, stage everything or rebuild its
checker during receipt tests.

The following **six untracked source/test files belong to this session** and remain
uncommitted:

- `merlin/lean/W8A8BoundedMain.lean`
- `src/merlin/verify/lean/interpreted.py`
- `src/merlin/verify/lean/bounded_numerics.py`
- `src/merlin/verify/lean/bounded_policy.py`
- `merlin/tests/ir/test_lean_w8a8_bounded_receipts.py`
- `merlin/tests/ir/test_lean_w8a8_bounded_protocol.py`

The modified parent [README](../README.md), this handoff and
`../bounded-receipts-qualification-v1.json` also belong to this session. The JSON qualification
records the exact qualified source hashes. Generated evidence is ignored; Markdown README
files follow the repository's retained-report convention. No checkpoint commit/push was made
for these latest six files before pausing.

Other dirty files are foreign. In particular, leave `Main.lean`, `LEDGER.md`, `findings.md`,
the existing Python `cli.py`, Qinner work and the BF16, overrank and scalar-square families
untouched. There are no active child agents from this session.

## Completed and qualified

| Work | Implemented behavior | Evidence |
|---|---|---|
| Bounded calls and suffixes | Function-indexed input domains; actual helper-body contracts at evaluator fuel; domain preservation through downstream policy suffixes | Pushed through `8081554421a33181442c29d75fa116cb5b194386`; 587 tests passed; 8971 theorem audits, zero violations |
| Dead producer normalization | Exact module/call equality after deleting a checked total unused producer; potentially failing producers cannot be discarded | Included in the same 587-test batch; mutation tests distinguish live and dead values |
| Projection contractions | Source-derived indexing geometry; actual reduction fiber cardinality; every signed-i32 accumulation prefix exact for the checked reduction bound; arbitrary interleaving and multiple reductions within the supported projection form | Pushed through `7304d8e50`; 1184 tests passed; 66 new cases, 42 complete native MLIR/Lean/ordered-recipe observations; 9037 theorem audits, zero violations |
| Bounded draft receipt route | Exact source function/argument contracts, authoritative exports, boundary joins, suffix/handoff binding, actual interpreter/proof artifact identity, fresh theorem axiom audit and fail-closed freshness checks | Latest uncommitted files; 239 tests passed in 299.640 seconds, including 53 new cases; no failures, errors or skips |

These regression batches overlap; do not sum them as unique tests. The last batch adds no
kernel theorem or numerical premise. Ruff check/format passed for its five Python files.
The unrelated structure ceiling failure remains `zephyr_model.py`: 3557 > 3537 lines;
that file was not changed.

The bounded route is independently callable with `python -m merlin.verify.lean.bounded_policy`.
It preserves the older structural diagnostics but runs the bounded theorem on the actual
exports itself. Input contracts reject duplicate JSON keys, duplicate functions, missing or
extra functions, Boolean bounds and malformed intervals. Intervals are nonnegative integers
with inclusive `lower` and exclusive `upper`; `null` supplies no range evidence.

The runtime pins the actual interpreter, auditor, standalone entry, configuration and local
MerlinVerify artifact inventory. It checks bytes and inventory before and after execution and
audits the actual theorem afresh rather than using an unrelated public-binary cache. Package
and toolchain trust remain in the existing ledger. All successful conditional draft attempts
remain `abstracted` with exit 1. They do not enable reviewed production admission.

## Latest qualification and preserved environment

All 1396 frozen records matched after the final tests. They include 368 selected compiled
proof modules and a conservative 752-file runtime capsule (744 local artifacts, including
hashes and stale artifacts). The isolated source manifest has 383 entries. The capsule's
SHA-256 is `260e2e7c3d24b758fbef605926b42adc13a1e82269cd98e1b401d19f3368354c`.

Evidence paths below are relative to this handoff's parent:

- `bounded-receipts-qualification-v1.json`: final results, source hashes and limitations.
- `bounded-receipts-qualified-v1.{xml,log}`: all 239 test identities/results.
- `bounded-receipts-{isolation,freeze,runtime}-v1.json`: exact inputs and execution capsule.
- `bounded-receipts-guard-v1.log`: 9037 theorem audits, zero violations.
- `projection-qualification-v1.json`, `projection-qualified-v2.{xml,log}`: prior broad suite.
- `bounded-integration-*`: previous bounded call/dead-module qualification.

Private Lean snapshot:
`out/build/lean-w8a8-source-integration/repo`. Its selected Lean sources are independent;
its Python source/tests link to the working checkout. The isolated Main excludes concurrent
foreign rules. Run Lake commands from its `merlin/lean` directory. Do not replace the snapshot
with the entire dirty main checkout. Shared package artifacts are large; reuse them.

The final regression used this environment from the repository root:

```sh
export ELAN_TOOLCHAIN="$PWD/out/build/lean-shared-passes/phase7-worktree/out/build/fresh-environment/toolchain"
export LEAN_SYSROOT="$ELAN_TOOLCHAIN" LEAN_NUM_THREADS=2
export PATH="/home/jeremy/.elan/bin:$PATH"
export MERLIN_LEAN_REQUIRED=1
export MERLIN_REPO_ROOT="$PWD/out/build/lean-w8a8-source-integration/repo"
export MERLIN_M2M_PYTHON="$PWD/out/artifacts/verification/lean-review/20261017T204007Z/integration/out/build/lean/mlir-venv/bin/python"
export MERLIN_TORCH_PYTHON="$PWD/out/artifacts/verification/lean-review/20261017T204007Z/integration/out/build/lean/torch-venv/bin/python"
export MERLIN_MLIR_RUNNER_UTILS="$PWD/out/build/lean-differential/libmlir_c_runner_utils.so"
```

The 11 test files, all under `merlin/tests/ir/`, were `test_lean_w8a8_` followed by:
`bounded_receipts.py`, `bounded_protocol.py`, `receipts.py`, `dynamic_receipts.py`, `chain.py`,
`handoff.py`, `cli.py`, `bounded_chain.py`, `bounded_calls.py`, `projection.py` and
`projection_differential.py`. Invocation used `.venv/bin/python -m pytest`, `-q -n 2`,
`-o tmp_path_retention_policy=all`, basetemp `/tmp/w8a8-bounded-receipts-qualified-v1` and the
JUnit path above. It has finished; no test process needs resuming.

## Unresolved retained TinyLlama public receipt failure

The separate actual model attempt exited **2**, published **unmeasured**, and produced no
bounded certificate. It is not part of the 239 passing regressions. The reason was:

```text
JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

Its outer attempt is `2d48497fd44b42bd92b2e44aa76b0fab`. The nested structural preflight
completed with draft/refusal diagnostics. The precise subprocess responsible for malformed
or empty JSON has **not been diagnosed**. Do not label this a numerical counterexample or
claim the retained model passes the new public route.

Reproduction with the environment above (use a new destination to preserve evidence):

```sh
PYTHONPATH=src .venv/bin/python -m merlin.verify.lean.bounded_policy \
  out/artifacts/verification/merlin-passes/w8a8/model-boundaries/20261020T215617Z/SY_model_tiny_llama/lean-preparation/55b1a0827d7445abb67966a3c8c40195/stages/0006_prepare-quantize-int8 \
  --destination out/artifacts/verification/merlin-passes/w8a8/bounded-receipts-tinyllama-retry \
  --input-contracts out/artifacts/verification/merlin-passes/w8a8/bounded-inputs-tinyllama-contracts-v1.json \
  --m2m-python "$MERLIN_M2M_PYTHON" --scope captured
```

Preserved original attempt: `../bounded-receipts-tinyllama-v1.json` and
`../bounded-receipts-tinyllama-v1/`. Existing authoritative exports are already available at
`bounded-receipts-tinyllama-v1/lean/291fbf5dc41c4f1f92957f5febe5a61c/structural/lean/53cb6e8e470a4821992a514ba9d1a44e/{0_before,0_after}.json`.
The same attempt's `lean/291fbf5dc41c4f1f92957f5febe5a61c/input-contracts.json` is the exact
contract snapshot. First replay the standalone interpreter/auditor on these inputs and retain
raw stdout, stderr and exit status; avoid repeating authoritative export unnecessarily.
Then diagnose the underlying cause and add a regression and useful process diagnostics.
Do not guess a JSON result or turn an execution failure into acceptance.

## Remaining proof obligations and next actions

1. Diagnose the public retained-model failure above. Review the six uncommitted files against
   their qualified hashes, then commit coherent changes with the documented limitation. Before
   pushing, fetch and review every unpublished commit and final diff; another agent may advance
   this branch. Never include its dirty work accidentally.
2. Resume numerical definedness at the older standalone TinyLlama boundary. It still first
   refuses `quant_ext.quantize_per_tensor` result **1256**, after 902 supported prefix operations.
   Earlier embedding/slice/concat/gather blockers were discharged. RMS-normalization and
   floating products now need justified numerical facts; finite inputs alone do not prove
   finite floating products. This is separate from the public receipt error.
3. Complete the concrete primitive floating-point model obligations. The existing seven source
   and three dynamic primitive numerical laws are explicit premises; no concrete IEEE model
   instance proving them has been admitted. Conditional proofs do not discharge these laws.
4. Obtain Agustin's independent recipe, floating-point interpretation and valid-input-domain
   review, then freeze the approved contracts. The user approved the explicit zero-row scale
   rule `0.00001`. TorchAO references were located at `examples/gemmini` on pinned
   `origin/feat/el4-experiment` commit `0cc289de52a10bed256f32c90cfbb54a21b3c81f`:
   Torch 2.10, TorchAO 0.16 for TinyLlama/ResNet and 0.15 for SmolVLA. The existing review packet
   remains at `../policy-review/README.md`; reference discovery is not independent signoff.
5. Bind the newly selected EL4 capture configuration and finish remaining source/feature-path
   coverage and reviewed production admission. The retained eight-token TinyLlama development
   capture is not that newer capture. Calibrated reuse, f16/BF16 and named variants are not
   automatically covered by these projection theorems.

The flags `whole_pass_verified`, `policy_reviewed` and `primitive_model_laws_verified` remain
**false**. Existing suspected pass issues F11 (square-body accumulator), F12 (per-channel
conversion order) and F13 (per-channel zero-point/axis checks) remain reports; this session
did not fix production code or claim those paths correct.

Optional unfinished research: `/tmp/W8A8ProjectionGrouping.lean` contains checked grouping
lemmas, but is unintegrated, uncommitted, outside qualification and not durable. Re-prove and
review any desired lemmas before integrating; do not count this scratch file as coverage.

## Resume note, 2026-10-22

The new planner assigned this agent W8A8 ownership. The six source/test files still match
their qualified hashes. The preserved 752-file runtime capsule is available and byte-identical.
Read-only direct and instrumented public-route replays returned valid JSON; the latter produced
an honest `abstracted` receipt with numerical qualification false. The old JSON failure was
not reproduced or isolated and is not marked fixed. Its original evidence is preserved.
The parent [progress report](../README.md) links the new raw process evidence and
[numerical-interface proposal](../numerical-interface/README.md). No source fix/regression,
law/domain change or rebuild was made. Commit/push operations await planner serialization.

## Later implementation milestone, 2026-10-22

The [numerical-domain handoff](../numerical-domain/README.md) now contains the implemented,
qualified continuation. It uses a **new private build**, preserving the runtime and six files
described above. The old finite-only domain/checker still has its documented limits; the new
separately named finite/nonzero-f32 candidate certifies source quantizer 1256 and 158 finite
source dequantizers, then refuses quantizer 1533. It has proved input-domain nonvacuity and
separate complete-boundary/prefix receipt paths. Its rotary-input-201 witness demonstrates
why further domain/capture facts are needed; full W8A8 is not claimed.

Qualification: 292 passed, 9,663 audited theorems with zero violations, 13 curated positive
receipt replays and an independent retained-model refusal replay. `numerical-domain/freeze-v2.json`
records the 1,230 unchanged records; the original 752-file capsule also remains unchanged.
`numerical-domain/implementation-v1.patch` contains the 30 owned additive source/test files.
The new private root is `out/build/lean-w8a8-numerical-domain/repo`. No files were staged,
committed, integrated or pushed. Shared/frozen, frontend and QuantExtLowering source remains
outside this worker's edits.
