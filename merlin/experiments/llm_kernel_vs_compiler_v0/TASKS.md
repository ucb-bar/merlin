# Task register — what "fully realized" means for this study

Every task the plan requires, with its current state. `DONE` means implemented **and** verified by a
test or a measured run; `PARTIAL` means the mechanism exists but a required property is unproven;
`OPEN` means not started. A task is not `DONE` because code exists — three things that looked done
this week were defects that flattered the result.

Blocking order is noted where one task's output is another's input. Costed tasks say so, because the
Bedrock ceiling is shared with prior work.

---

## Phase 0 — Honest measurement (no agent spend)

| id | task | state |
|---|---|---|
| 0.1 | Whole-model execution: weights-blob fix in `muon_harness` | **DONE** — 45 min → 14.2 s, bit-exact, 0 verdict changes over 36 capsules; 4 tests |
| 0.2 | Kernel provenance audit → contamination fence | **DONE** — `audit_kernel_provenance.py`; hand / agent_generated / compiler_generated / artifact_only / unknown |
| 0.2b | Register `radiance-kernels` as an answer surface so `bwrap.coverage_gap` masks it | **OPEN** — the new bwrap box masks all of `/scratch`, which covers it today, but it is not *declared*, so a future grant could re-expose it silently |
| 0.3 | Codex credential rotation (`--bind` auth.json) | **DONE** — consented; now applied per-run inside the sandbox |
| 0.4 | Pick + track the authoritative radiance contract, register pins | **PARTIAL** — 9 pins declared and green; the chosen residual is still untracked, so the ARR denominator has no version history |
| 0.5 | Inherit readiness fixes; work in a worktree | **DONE** — `feat/kernel-vs-compiler` |
| 0.6 | Phase-0 exit gate: `readiness_check` GO, `preflight` GO_FOR_PILOT, oracle available, TinyLlama visible | **PARTIAL** — oracle + TinyLlama confirmed; the two gate scripts have not been re-run since the merge |
| 0.7 | Tests must exercise the tree they live in | **DONE** — several worktrees share one venv, which installs `merlin` from the primary checkout, so `import merlin` inside a worktree resolved to the *other* tree and a green suite proved nothing about the edit under test. `merlin/tests/conftest.py` now prepends its own tree; a no-op in the primary checkout. It immediately exposed the worktree's real gaps: 10 of 661 infra tests now fail here and pass in the primary checkout, all of them reading **untracked** inputs a worktree does not carry — the live-bwrap golden-masking cases (goldens are deliberately untracked) and one CLI smoke test reading a generated `out/artifacts/` file. Those are environment gaps made visible, not regressions; the library tests are green |

## Phase 1 — Library modules (`merlin/python/merlin/benchharness/`)

| id | task | state |
|---|---|---|
| 1.1 | `evaluation.py` — `EvaluationResult` + whitelist `redact()` | **DONE** — 14 tests incl. the golden-leak test |
| 1.2 | `kernel_library.py` — signature-keyed store + reuse ladder L0–L3 | **DONE** — 21 tests; L1 is a proposal the evaluator must confirm, a failed claim demotes to L3 and is charged |
| 1.3 | `task_basis.py` — weighted set cover over a census + eligibility map | **DONE** — 29 tests; runs on the real TinyLlama census (2 tasks, 97.77% cover) |
| 1.4 | `adaptation_ledger.py` — cumulative-cost curves | **PARTIAL** — per-run totals land in `summary.json.cost`; no cross-run accumulator, no reuse-aware ledger |
| 1.5 | Experiment scaffolding: `AGENT.md`, `README.md`, `METHODOLOGY.md`, `study.yaml`, `spec/radiance_bench.py`, `spec/reveal_order.yaml`, `spec/ladder.yaml`, `schemas/*.schema.json` | **OPEN** — `METHODOLOGY.md` is where the "cannot be honoured as written" list must live |

## Phase 2 — The common evaluator

| id | task | state |
|---|---|---|
| 2.1 | One `evaluate()` both arms call | **DONE** — `kvc_eval.py`; same oracle, fidelity and definition of correct for every arm |
| 2.2 | Kernel shim delegating scaffolding to the reference tool | **DONE** — `shim/kernelshim`, own integrity scan, `artifact_provenance` stamped |
| 2.3 | Honesty rule: `certifying_tiers = {L2}`; execution-only tier cannot certify | **DONE** — tested |
| 2.4 | Honesty rule: MX unwinnable = `unsupported`, out of both sides of the ratio | **DONE** — tested |
| 2.5 | Honesty rule: unavailable ≠ zero; unpriced turn ≠ $0 | **DONE** — tested both in the evaluator and the token reader |
| 2.6 | Redaction audited against `answer_surfaces.audit_tokens` | **PARTIAL** — whitelist is tested directly; not yet passed through the repo's own auditor |
| 2.7 | Latency provenance (`cycles_tier`, `cycles_cycle_accurate`) | **DONE** — GSIM certifies execution but reports no cycles, so cert yields RTL-backed correctness + a model latency |

## Phase 3 — Inventory and deterministic task derivation

| id | task | state |
|---|---|---|
| 3.1 | Weight-free census of all five workloads (`work_dir=` mandatory) | **DONE** — run across all five captured models; 0 opaque ops, so `priced_fraction` is 1.0 everywhere. Contraction is 97.2–99.9% of FLOPs in every one |
| 3.2 | Captures for the reveal set | **PARTIAL** — TinyLlama / Gemma-2 / LSTMNetVIT have full bundles; DeepSeek + SmolVLA have `model.mlir` only (census-capable, not executable); SmolVLA has 1 opaque op |
| 3.3 | Task basis: signature key, eligibility filter, cost weight, greedy cover ≥95%, family floor | **PARTIAL** — derivation lands and runs on TinyLlama (2 of 5 groups, 97.77% of the eligible denominator); the basis is not yet materialised into capsules, so **every number measured so far is still three pilot capsules, not TinyLlama** |
| 3.4 | `basis_certificate.json` | **PARTIAL** — emitted by `derive_basis`, incl. `families_declared_not_evidenced` vs `families_outside_census_scope`; not yet written to a tracked path |
| 3.5 | Config ladder C0–C4 (~5 per task) | **OPEN** |
| 3.6 | `holdout_certificate.json` proving shape-tuple disjointness | **OPEN** — prior "hidden" sets on other targets were renames; a nominal hidden score is not transfer |

## Phase 4 — The arms

| id | task | state |
|---|---|---|
| 4.1 | Arm A1 Codex seat, from the specification | **DONE** — running; sandboxed; notional-only accounting |
| 4.2 | Arm A2 Bedrock qwen3-coder-480b | **DONE** — running; metered; uncached by construction and reported as such |
| 4.3 | Arm A3 Google gemini-3.5-flash | **DONE** — running; solved R0 from scratch under sandbox |
| 4.4 | Mutual blindness enforced, not asserted | **DONE** — bwrap per agent, empty grant bundle, leak paths verified closed |
| 4.5 | Add the blindness + contamination checks to `verify_no_cheat` | **OPEN** — enforced at launch today, but not asserted by the repo's own cheat checker |
| 4.6 | Multi-seed repeats (≥3) | **DONE** — 3 tasks × 3 seeds on all three kernel arms; s4 is the valid codex/bedrock matrix, s3 the valid gemini one |
| 4.7 | Arm G AutoComp, real framework, our oracle | **DONE** — bridge + `--start scratch` self-bootstrap |
| 4.8 | AutoComp historical re-accounting + contaminated-by-construction label | **PARTIAL** — figure measured ($267.37 / 6882 calls, vs $223 in the plan and $178.54 in the rollup); not yet written into `METHODOLOGY.md` |
| 4.9 | Arm F kernel-library reuse + generalization matrix | **OPEN** — 1.2 is done, so this is now unblocked and next. Without it kernel-gen pays full price for every config and the crossover arrives too early |
| 4.10 | Arm B `Merlin-Base` — generate the backend, no optimized kernels in scope | **OPEN** — the other half of the comparison; nothing here yet |
| 4.11 | Arm B `Merlin+Seed` — mine TinyLlama trajectories through the promotion ladder | **OPEN** |
| 4.12 | Anti-specialization audit (5 mechanical tests) | **OPEN** — blocks any claim that mined policy is general rather than a lookup table |
| 4.13 | Freeze + sequential reveal, zero-LLM **enforced** (no creds bound, ledger asserts 0 calls) | **OPEN** |
| 4.14 | Does the optimization stage pay for itself? | **OPEN** — measured and non-monotone: gemini's best R4 kernel was **round 0**, and five optimization rounds made it worse; codex improved to round 3 then regressed. The loop keeps the best so no result is wrong, but most of the token spend currently buys nothing |

## Phase 5 — Accounting, metrics, plots

| id | task | state |
|---|---|---|
| 5.1 | Token/cost telemetry per round incl. failures | **DONE** — `parse_agent_transcript` for codex + opencode; 17 tests. Was reading **zero** for every arm |
| 5.2 | Preserve all intermediates | **DONE** — per-round candidate kernels archived; transcripts already kept |
| 5.3 | AET emission (`emit_to_aet`, `MERLIN_AET_SINK=1`) + `agg_by_model` | **OPEN** |
| 5.4 | Coverage metric: % independently eligible execution cost accelerated | **OPEN** — manifest exists; the per-run number is not computed. CPU fallback for an eligible region must count *against*. ⚠️ **The denominator is FLOPs, and FLOPs flatter it**: no measured tick profile exists for any reveal model on radiance, so cost falls back to `work_share`, where contraction is 99.93% of TinyLlama and 87.2% of its ops weigh 0.07%. Prior measurement on another substrate found elementwise/scalar work dominating wall time at a comparable FLOP share — see `elementwise-runs-scalar-f32`. Quote coverage as FLOP-weighted until a tick profile exists |
| 5.5 | Performance metric vs reference and best-observed, geomeans, %peak | **PARTIAL** — cycles/gflops/%peak captured per run; no aggregation |
| 5.6 | Break-even per reveal prefix, independently for tokens / time / notional $ | **OPEN** — the headline result |
| 5.7 | Statistics: median, geomean, bootstrap CIs; failures stay in the distribution | **OPEN** — needs 4.6 to land |
| 5.8 | Six 2D curves + 3D Pareto, PDF/SVG/PNG + CSV/JSON | **OPEN** |
| 5.9 | Time-to-first-correct and time-to-within-X%-of-best curves | **OPEN** — *addition to the plan*; inputs already recorded |
| 5.10 | Model × family capability matrix | **OPEN** — *addition to the plan*; may be a headline finding if model choice dominates the kernel arm |
| 5.11 | Cert (GSIM) pass over accepted kernels | **PARTIAL** — tier validated at 84 s; not yet run across the accepted set. Yields RTL-backed correctness, **not** cycle-accurate latency |
| 5.12 | Reference baseline every arm is relative to | **DONE** — `scripts/baseline.py`; reference_v0 at L2/fast: R0 631,721 · R4 284,694 · R3 673,923. Until it existed the study recorded absolute cycle counts and nothing else |
| 5.13 | Failure taxonomy: an unparseable submission is not a `tool_crash` | **OPEN** — it currently reads as our infrastructure breaking, which would understate the model's failure and overstate ours |

## Phase 6 — Pilot gate

| id | task | state |
|---|---|---|
| 6.1 | Three structurally different tasks end to end | **DONE** — R0 / R4 / R3 through generation, grading, feedback, optimization |
| 6.2 | Full chain incl. AET trajectory, `usage_complete=true`, `freeze.json` | **PARTIAL** — accounting now real; AET and freeze missing |
| 6.3 | 3×5 reuse matrix at zero LLM cost | **OPEN** — blocked on 1.2 / 4.9 |
| 6.4 | Same pilot on every arm, in parallel and blind | **DONE** — 45 sandboxed jobs across s3/s4, per-provider concurrency caps |

---

## How to check this file against reality

`scripts/status.py` reads the register here and, independently, the run dirs, the matrix files and the
test tree. The states below are hand-maintained and therefore drift -- three rows this week said
`OPEN` for work that had already landed. Every *number* comes from disk at the moment you ask:

    .venv/bin/python merlin/experiments/llm_kernel_vs_compiler_v0/scripts/status.py

A disagreement between the two halves means one of them is wrong. Fix the register; do not adjust the
measurement to match it.

---

## Measured so far (three pilot capsules, not yet TinyLlama)

R0 gemm fp32 / R4 rmsnorm fp32 / R3 attention-qk fp16, from the specification, three seeds each,
every agent in its own bwrap box with an empty grant bundle.

| arm | model | solved | tokens | cost | agent time |
|---|---|---|---|---|---|
| codex_kernel | gpt-5.6-sol (seat) | **9/9** | 6.05 M | $12.66 notional, $0 billed | 1.19 h |
| gemini_kernel | gemini-3.5-flash | **9/9** | 117.7 M | $54.11 billed | 3.05 h |
| bedrock_kernel | qwen3-coder-480b | **0/9** | 1.69 M | $0.50 billed | 0.19 h |

### Performance, against a reference for the first time

`reference_v0` is the hand-curated, correct, unoptimized lowering — by its own manifest the ceiling an
agentic backend has to re-derive and beat. Measured through the same runner, oracle and fidelity as
every arm:

| task | reference | codex median | gemini median | best seen |
|---|---|---|---|---|
| R0_gemm_fp32 | 631,721 | 526,151 (1.20x) | 549,377 (1.15x) | 517,301 (1.22x) |
| R4_rmsnorm_fp32 | 284,694 | 243,968 (1.17x) | 243,670 (1.17x) | 243,670 (1.17x) |
| R3_attention_qk_fp16 | 673,923 | 557,531 (1.21x) | 571,776 (1.18x) | 470,727 (1.43x) |
| **geomean** | — | **1.192x** | **1.166x** | — |

Every accepted kernel beats the reference, by 1.15–1.43x. Two qualifications that must travel with
those numbers:

- **They are estimates.** All of it is the L2 functional model. Only the cycle-accurate tier may be
  quoted as a measurement, and it has not been run over the accepted set (5.11).
- **The metric does discriminate, which was worth checking.** Re-running stored candidates through
  the same path reproduces every recorded count exactly, and within a single run the counts move with
  the kernel: codex spans 284,694 -> 243,723 across its rounds, gemini 243,670 -> 303,822. An earlier
  read that the arms had converged was an artifact of comparing each arm's *best* round.

Three things the solved/cost table already says, and one it does not:

- **The arms are three orders of magnitude apart in token cost for the same result.** Gemini and codex
  both solve 9/9; gemini spends ~20x the tokens and is the only arm with a real dollar bill. If the
  kernel arm's cost curve is quoted without saying which model drew it, the curve is meaningless.
- **The bedrock 0/9 is a real model failure, not a harness one.** Checked round by round: the agent
  received the verdict, read it, and rewrote the kernel each round (three distinct candidates), but
  emitted pre-opaque-pointer LLVM dialect (`!llvm.ptr<!llvm.array<256 x f32>>`, `llvm.constant`) every
  time and never got past the parser. Recorded as a capability result.
- **The failure taxonomy mislabels it.** An unparseable *submission* is recorded
  `failure_category: tool_crash`, which reads as our infrastructure breaking. It needs its own
  category before any failure breakdown is published.
- **It does not say anything about TinyLlama.** These are three pre-existing pilot capsules. The task
  basis (3.3) is derived but not materialised, so no measured number here is about a workload yet.

---

## Kernel and coverage inventory (measured 2026-08-29)

**Kernels.** 226 kernel files, 168 distinct texts, across 58 agent run dirs. Of those, **18 are
claim-bearing accepted kernels** — 9 codex (`s4`) + 9 gemini (`s3`), one per (capsule x seed) over
3 capsules x 3 seeds. Bedrock accepted none. The rest are 141 per-round candidates (kept on purpose,
see 5.2), plus voided-round and exploratory dirs. Accepted kernels span 76 to 13,335 lines; size does
not track cycles (a 336-line codex R4 and a 2,410-line gemini R4 land 0.1% apart).

**Model inventory** (weight-free, `model.mlir` only; 0 opaque ops anywhere, so `priced_fraction` = 1.0):

| model | linalg ops | total work (FLOP) | contraction share |
|---|---:|---:|---:|
| gemma2_2b (int8) | 2,300 | 673,182,195,592 | 99.93% |
| tiny_llama (fp32) | 1,557 | 16,573,632,084 | 99.93% |
| spectformer (int8) | 762 | 3,754,493,304 | 98.03% |
| lstmnetvit (int8) | 387 | 114,073,736 | 97.24% |
| small_llama (fp32) | 198 | 6,994,320 | 98.85% |

**Coverage of the seed model (TinyLlama), the number the study will be asked for:**

| step | value |
|---|---|
| census scope | 200 of 1,557 ops (12.8%), carrying 99.93% of FLOPs |
| eligible / census | 99.93% (2 rank-4 `batch_matmul` groups ineligible: legal ranks are [2, 3]) |
| chosen cover / eligible | 97.77% — 2 of 5 signature groups |
| **basis vs whole model** | **97.63% of FLOPs; 87.2% of ops unsearched, weighing 0.07%** |
| basis vs model, *measured* | **0%** — no basis capsule exists yet (3.5) |

⚠️ **Read the 97.63% as FLOP-weighted, never as execution cost.** `kernels.census` walks
`observe_contractions`, so contraction is the only family it can see, and no measured tick profile
exists for any reveal model on radiance — so `derive_basis` falls back to `work_share`. On another
substrate, elementwise and scalar-f32 work at a comparable FLOP share dominated measured wall time
(`elementwise-runs-scalar-f32`, `wholemodel-slowness-is-memory-not-mix`). A coverage claim built on
FLOPs credits the compiler for the 99.93% that is easy and stays silent about the 87.2% of ops that
decide the clock. 5.4 must state which weighting it used.

**Fixed while measuring this.** `derive_basis` reported TinyLlama's seven unsearched families as
`families_declared_not_evidenced` — a claim about the *model* ("it never normalizes") derived from a
fact about the *census*. The full inventory disproves it directly: TinyLlama has 66 normalization ops
and 542 elementwise ops. `Census` now carries the scope it walked (`census.CENSUS_SCOPE`) and
`derive_basis` reads it, so those seven land in `families_outside_census_scope` instead.

---

## Critical path to a defensible headline

The break-even number (5.6) is the paper's claim. It needs, in order:

1. **3.3 -> 3.5 -> 3.4/3.6** -- materialise the derived basis into capsules and a config ladder, with
   its certificate. The derivation exists; until the capsules do, *everything measured is three pilot
   capsules, not TinyLlama.*
2. **4.9** -- the kernel library wired into the runner, so kernel-gen is not charged full price for
   every config. Unblocked now that 1.2 has landed. Without it the comparison is unfair **to the
   kernel arm** and the crossover arrives too early.
3. **4.10 -> 4.13** -- Merlin-Base, frozen, revealed sequentially with zero-LLM enforced. There is
   currently **no compiler arm at all**; only the kernel-generation side exists. This is the largest
   remaining block of work and nothing downstream of it can be reported without it.
4. **5.4** -- the coverage denominator, independent of Merlin's own declarations.
5. **5.6 / 5.7** -- the curves and their confidence intervals.

Nothing on that path is blocked by a missing measurement; it is all buildable from what is on disk.

## Standing risks

- **A harness limit reads as a model defect.** Eight occurrences on this project, five in the last
  three days (kernel misroute, zero token accounting, cross-arm reads, codex's sandbox nested inside
  ours, worktree tests importing another tree's library). Any new number gets an adversarial pass over the *trace* before it is quoted, not just the
  score. The bedrock 0/9 above survived that pass; the earlier codex 0/9 did not.
- **The harness is a first-order variable.** Codex runs on its own CLI, the other two on opencode.
  Cross-driver comparisons are a harness band, never agent capability.
- **`out/artifacts/targets/radiance/` ships no capsule-contract reference package**, so muon
  `reference_v0` is being graded under `target="radiance"`. Record it or author one.
- **AutoComp saw all four reveal models in prior campaigns** -- its reveal results are an upper bound,
  never transfer.
- **Spend.** $56.30 billed so far, essentially all of it gemini ($55.35 incl. voided runs); codex is seat-notional and bedrock is
  under a dollar. The gemini arm is the one that can exhaust a budget.
