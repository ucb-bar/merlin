# STATUS — LLM kernel generation vs. compiler generation on Radiance

Branch `feat/kernel-vs-compiler`, worktree `$KVC_WORKTREE`. Last updated 2026-08-27.

The study asks where two ways of bringing workloads to a new accelerator cross over: repeatedly
LLM-generating kernels per workload, versus spending LLM effort once to generate a compiler, freezing
it, and compiling unseen workloads with no further agentic adaptation.

This file records what is true right now, including the parts that are not ready. Nothing below is a
projection.

---

## Implemented and verified

### The whole-model wall is gone (the gate on every model-level result)

The harness emitted **one C statement per operand element**, so a 2048x2048 f32 operand became
4,194,305 statements / 124.72 MB of C and clang ran 45+ minutes without finishing. No whole-model
capsule had ever been graded because of it.

That emission was not naive — it keeps the object relocation-free, which the *inline* build path
requires. But model capsules take the **object** path, which already transcodes reloc-preserving and
already resolves HI20/LO12, exactly what a `.rodata` reference costs. The constraint never applied to
the path that was blocked.

| | before | after |
|---|---|---|
| 2048x2048 build | 45+ min, never finished | **14.2 s** |
| `main.c` | 124.72 MB / 4,194,305 lines | **1.4 KB / 30 lines** |

Verified, not asserted: a 40x40 build whose operands cross the threshold is **bit-exact against numpy
on cyclotron**, and an A/B of the blob threshold across all 36 public radiance capsules **changed no
verdict** (18/36 either way).

### Kernel provenance — the contamination is real but narrower than the directory count suggests

| checkout | kernels | verdicts |
|---|---|---|
| `/scratch2/.../radiance-kernels` @ `399757f` (**pinned**) | 99 | 57 hand · 40 compiler-generated · 2 unknown |
| `/scratch/.../radiance-kernels` @ `4de7ef3` (unpinned) | 896 | 57 hand · **802 agent-generated** · 34 artifact-only · 3 unknown |

The 802 agent-generated dirs contain **no source at all** — ELFs, objects, logs, traces — and exist
only in the *unpinned* checkout. Still an answer surface (an ELF disassembles), so excluded, but not
the source leak the count implied. The audit fails closed: `unknown` is excluded exactly like
`agent_generated`.

It records its own limit: a `hand` verdict cannot separate source a human typed from agent output a
human committed. **No reference number built on these kernels may be called "expert."**

### Eligibility denominator, grounded independently

Coverage is `accelerated / eligible`, so whoever sets the denominator sets the score. It comes from
kernels a human wrote against this hardware, never from Merlin's own declaration.

Which way the bias runs matters: a **smaller** denominator flatters the compiler, and Merlin's own
derivation is the most conservative of the three views — it leaves `movement`, `reduction` and
`synchronization` *undetermined*. The hand kernels evidence all three. Deferring to Merlin would have
shrunk the denominator in Merlin's favour, so the manifest takes the larger independent one:
**8 families in, 0 excluded**.

### Workload inventory (Phase A)

Reads `model.mlir` only, so a 4.4 GB capture inventories in seconds. Four ways it silently *shrank* a
workload were found and fixed — each produced a plausible number rather than an error:

- an opaque `func.call` (an op the importer gave up on) was invisible, though still fully typed;
- `provenance()` keys carry a `prov.` prefix, and reading without it emptied **all 3121** rows, so the
  model read as having no semantic families at all;
- walking every op counted linalg *body* ops as ops, ~2x inflation;
- `work_of` returns **FLOPs, not MACs**, so pricing an opaque op in MACs weighted it half as heavily
  as an identical op the importer happened to handle — a bias tracking importer quality, not workload.

Cross-checked against the repo's independent census: totals agree **exactly** on four models.

### Shared evaluation schema

One schema for every arm, because two arms scored by different oracles produce a join no statistics
repair. Three rules are encoded rather than left to callers, each of which fails in the flattering
direction: an execution-only tier cannot certify numerics; an unwinnable task is `unsupported` and
leaves *both* sides of the ratio; an incorrect kernel gets no performance credit.

Redaction is a **whitelist** — the grader's own report embeds `first_mismatch = {output, index,
expected, observed}`, and `expected` is the answer key. Validated against real grades of the three
pilot capsules, not only synthetic dicts.

### Workloads — all five captured

| model | ops | FLOPs | note |
|---|---|---|---|
| gemma2_2b (int8) | 2300 | 673.2 G | 13 captures were on disk but invisible — unregistered base name |
| smolvla | 4133 | 263.9 G | **1 opaque op**, priced at 1.21 GFLOP |
| deepseek_r1_qwen | 2055 | 24.7 G | **authored here**; 197 matmuls = 28 layers x 7 + lm_head |
| tiny_llama (seed) | 1557 | 16.6 G | **re-captured**; 155 matmuls = 22 layers x 7 + lm_head |
| spectformer (int8) | 762 | 3.75 G | |
| lstmnetvit (int8) | 387 | 114.1 M | |

**The seed was not TinyLlama.** `capture.toml` pinned `M2M_LLAMA_LAYERS=2`, which the loader treats
as a smoke path with **random init** — right architecture, right shapes, meaningless weights, and
nothing in the filenames saying so. It also held 15 of 155 matmuls, which would have under-counted
every per-layer op 11x against the embedding and `lm_head`, skewing task selection toward the head.
Re-captured at full depth with 4.4 GB of pretrained weights and a golden.

---

## Measured findings that change the design

**TinyLlama is 99.86% matmul by FLOPs** (16.55 G of 16.57 G; next is batch_matmul at 0.07%). So a
purely cost-weighted "cover 95% of eligible cost" basis **collapses to a single family** and would
never exercise norms, softmax or elementwise. Two consequences:

- the **family floor**, not the cost cover, is what actually populates the task basis;
- the cost weight should prefer **measured cycles** over FLOPs, since FLOPs are blind to the
  memory-bound ops that dominate real time.

---

## Agent routing — a standing constraint

**Every agent run goes through the Codex seat, Bedrock, or the Google API — nothing else.** Anthropic
and OpenAI models rate-cap quickly on this account, and a capped run does not fail loudly: it yields
short rounds and a small constant score, which reads as a weak agent rather than a throttled one. A
comparison run that way measures quota, not capability.

| arm | provider | driver | model | billing |
|---|---|---|---|---|
| `codex_kernel` | subscription | codex | `gpt-5.6-sol` | `subscription_notional` — billed 0, any figure is a projection |
| `bedrock_kernel` | bedrock (**us-west-2**) | converse | `qwen.qwen3-coder-480b-a35b-v1:0` | `metered`, cap $100 |
| `gemini_kernel` | google | opencode | `google/gemini-3.5-flash` | `metered`, cap $60, **separate credential** |

Both open-model picks are campaign-proven rather than guesses — they are the two most-called models of
the prior Radiance campaign (qwen 2955 calls, gemini 1400).

Three facts that would each have failed at launch, all verified against the live account:

- **Gemini is not on Bedrock.** Google's Bedrock presence is Gemma only (3-4b/12b/27b, open-weight
  and far smaller). Checked with `list_foundation_models` in both regions. Matching the prior campaign
  means adding Google's own API as a third route, not pretending Bedrock has it.
- **`qwen3-coder-480b` exists only in us-west-2.** `us-east-1` — where prior capsule-bench runs
  authenticated — offers `qwen3-coder-next` and `-30b` but not the 480b. Region is part of this
  model's identity, so the config pins it.
- **`model_tiers` aliases `qwen-coder` to qwen3-coder-next**, a different model the campaign never
  used. The id is pinned explicitly rather than taken from the alias.

The arms run **in parallel and blind to each other**. Enforced by `scripts/check_method_models.py`,
keyed on **provider** rather than driver because one driver reaches several — opencode carries both
Bedrock and Google, so the driver alone does not say which budget a run spends.

**Codex is live** (verified 2026-08-27): a real `codex exec --json` round returned `PONG` with
`auth_mode: chatgpt` and usage `input_tokens 17929 / cached 11008 / output 6`, which also confirms the
accounting mandate that `input_tokens` already contains the cached count. Note `--skip-git-repo-check`
is required outside a git repo.

## Not ready

- **Task derivation (Phase B), kernel library (F), agent runners (E/G), accounting (D), plots.**
  Not started.
- **SmolVLA has one opaque op.** An m2m defect, not a shape limitation: the identical conv decomposes
  to `im2col_matmul` in *both* backends in isolation and in SmolVLA's own venv; only the full-model
  export bails. The inventory prices it rather than chasing it upstream.
- **`out/artifacts/targets/radiance/` ships no capsule-contract reference package.** The muon
  `reference_v0` package is being graded under `target="radiance"` and refuses some capsules at the
  **parse** plane (`RP7_attn_full_fp16_pt`: "unsupported or missing merlin_iface.version"). This is a
  package gap, unrelated to the tensor wall, and it bounds what the reference arm can currently score.
- **The dse suite does not complete** within 25 minutes on either this branch **or an unmodified
  baseline**, so it is unverified either way. Failure positions are identical between the two, which
  is why the `gemma2_2b` registration was judged safe — but that is a comparison, not a green suite.

## Known limitations that could weaken the claim

1. **AutoComp's onboarding cost cannot be measured on this box.** **$267.37 over 6882 calls** is
   already spent across both ledgers, and its campaign docs cover **all four reveal models**. Its
   reveal results are contaminated-by-construction and must be reported as an upper bound with dual
   accounting, never as transfer. By model: `us.anthropic.claude-sonnet-4-6` $124.11/2429 calls,
   `qwen.qwen3-coder-480b-a35b-v1:0` $76.10/2955, `gemini-3.5-flash` $57.19/1400, rest $9.97.
2. **"Hand-written" is only "human-committed."** Stated in the artifact itself.
3. **Only cyclotron@L2 certifies numerics.** Verilator L3 is completion-only; VCS L3 crashes upstream.
   Every result must quote `tier_reached`, never a bare fraction.
4. **An oracle can run and still be wrong.** On another target, L2 ran cleanly yet disagreed with RTL
   on 1011/1024 elements. Cyclotron should be difftested against RTL before being trusted as a
   *correctness certifier*, not merely as a runner.
5. **The harness is a first-order variable** — the same model scored 0/20 on one harness and 15/20 on
   another. Comparisons stay within-driver unless the same model runs on two.

## Commands

```sh
cd $KVC_WORKTREE && export PYTHONPATH=$PWD/merlin/python TMPDIR=$TMPDIR
E=merlin/experiments/llm_kernel_vs_compiler_v0

.venv/bin/python $E/scripts/audit_kernel_provenance.py \
  --repo ${MERLIN_RADIANCE_KERNELS} --repo ${MERLIN_RADIANCE_KERNELS_E2E} \
  --agent-tree ${MERLIN_EXT_AUTOCOMP} \
  --out $E/eligibility/provenance/kernel_provenance.yaml

.venv/bin/python $E/scripts/build_eligibility_manifest.py --target radiance \
  --out $E/eligibility/radiance_eligibility_manifest.yaml

.venv/bin/python $E/scripts/inventory_models.py --all --out out/artifacts/kvc-inventory/v1

# every agent arm must route through the Codex seat or Bedrock -- exit 0 required before a run
.venv/bin/python $E/scripts/check_method_models.py

MERLIN_MUON_SKIP_RTL_L3=1 .venv/bin/python -m pytest \
  merlin/tests/infra/test_kvc_inventory.py merlin/tests/infra/test_evaluation_schema.py \
  merlin/tests/infra/test_kvc_method_routing.py \
  merlin/tests/runtime/test_muon_harness_blobs.py merlin/tests/infra/test_codex_driver.py -q
```

⚠️ Always `git commit ... < /dev/null` here — the hooks otherwise hang waiting on an inherited stdout
pipe, and stale attempts stack up behind each other.
