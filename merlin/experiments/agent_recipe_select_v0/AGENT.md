# AGENT.md — agent_recipe_select_v0

## Purpose

The complementary half of merlin's performance story. The established path is *agent **builds** the
compiler*: an agent edits compiler source and the deliverable is a better reusable compiler
(`experiments/performance_contract`, driven by `gemmini_perf_bench/scripts/perf_agent_stage.py`).
This experiment tests *agent **uses** the compiler*: freeze a certified backend, expose a small set of
compiler-defined optimization choices, and have an LLM pick a **recipe** per workload.

The agent produces a recipe and nothing else — never source, never an instruction. The compiler turns
the decision into code; elaborated RTL turns the code into cycles.

**The deliverable is a better compiler, not a per-workload lookup table.** Every finding has to
terminate in either a changed default or a rule the compiler derives itself. A recipe that wins
everywhere is a wrong default, not a selection opportunity — see `FINDINGS.md` in the product dir.

## ⚠️ This is a PARALLEL track — read before touching anything

The certified backend and the agentic perf experiment are live work owned by other sessions in this
shared checkout. The rule:

> READ the frozen artifacts. COPY anything that has to change. WRITE only under paths this
> experiment owns exclusively.

`scripts/_track.py` states that boundary once and is imported by every script. It carries
`assert_frozen_intact()`, which every measuring script calls **before** producing a number, so a
drifted champion fails loudly instead of silently invalidating the equivalence gate.

| | path | mode |
|---|---|---|
| certified champion | `out/artifacts/targets/gemmini/gemmini_xdsl_rtl_v0/` | **read-only**, byte-pinned by its own `SHA256SUMS` |
| our fork | `out/artifacts/targets/gemmini/gemmini_xdsl_recipe_v0/` | owned (a copy) |
| runs | `out/runs/gemmini/recipe-select/` | owned, via `benchharness.runs_root(target, suite)` |
| products | `out/artifacts/recipe-select/gemmini/v<N>/` | owned, via `new_product` |
| GSIM emulator | `/scratch/agustin/tmp/gsim_cert_serialclk_v1/…_filtered_final` | **read-only**, another session's campaign uses it concurrently |

Do **not** pass `runs_root=out/runs`: `certify` appends `runs/<suite>/`, which produced
`out/runs/runs/gemmini-contract/` and put this track's runs in a SHARED suite dir. Use `T.RUNS`.

## Scripts

| script | what it does |
|---|---|
| `_track.py` | the isolation boundary + `assert_frozen_intact()` + the GSIM env and its citation constraint |
| `agent_compile.py` | the agent-facing surface: `inspect` / `choices` / `build` / `evaluate`, compact JSON, no MCP |
| `costmodel_surface.py` | milestone 0: price the surface analytically. **Its gate is void** — the cost model was falsified (see below); kept because the refutation is the result |
| `pin_oracle_cost.py` | same ELF on both engines: cycles agreement + measured wall cost |
| `sweep_recipes.py` | emitted-code deltas + measured cycles per point (`--all-points` enumerates the compiler's own catalog, so a new value cannot escape the bit-exactness gate) |
| `heldout_rule.py` | does the compiler change generalise? Shapes never used in fitting |
| `run_recipe_agent.py` | the agentic arm: one `codex exec` per candidate, read-only sandbox, JSON recipe only |
| `record_spend.py` | oracle + LLM spend into aet |
| `gen_report.py` | `FINDINGS.md`, separating MEASURED / DERIVED / FALSIFIED / NOT MEASURED |

## Invariants

- **The default recipe must emit the certified compiler byte for byte.** `merlin/tests/gemmini/test_recipe_surface.py`
  asserts it per capsule. Without it, a measured delta is attributable to the refactor rather than the
  recipe.
- **A lever is real only if the EMITTED CODE changes** — counts for a deletion, *order* for a
  reordering. Judging a reordering by a histogram reports a real change as inert; that mistake is
  already encoded as a test.
- **Cycles name their engine.** GSIM here simulates `GemminiGsimSerialClkConfig`, whose accelerator
  modules are identical to stock `GemminiRocketConfig` but whose cycles were **measured** to differ
  (302 vs 303, 604 vs 610). Never quote them as Verilator-equivalent.
- **A run id must name the WHOLE recipe.** It once spelled two of three dimensions, collapsing 20
  points onto 10 run dirs while still reporting them as measured. Content-address it.
- **The agent may never modify the compiler**, and that is structural, not instructed: no tools, a
  read-only sandbox, no writable workspace, and the whole expected output is a JSON object.

## Two things this experiment FALSIFIED (kept, not deleted)

1. **`merlin/python/merlin/cost_model/gemmini_cost_coeffs.json`.** Predicted vs measured: A2 174/302,
   PK03_k128 1103/604, w1 1056/780, w2 7322/3466 — both directions, outside its own declared
   `max_abs_pct` of 34.9%. Its metadata says `"linear, serial; no overlap"` and `meta.sim` is `"?"`.
   It has **no gated consumer** (`perf` registers it `DIAGNOSTIC`; `merlin/dse/` imports an unrelated
   module of the same name), so this is worth fixing, not a broken workstream.
2. **The capacity predicate for the new default.** Both residency values reserve the *same* rows, so
   capacity cannot discriminate between them. The surviving claim is stronger: a plain default flip.

## Accounting

Spend is recorded to aet (`aet spend out/runs/gemmini/recipe-select`). The deterministic phases log a
**measured zero** for LLM tokens with a stated reason — never an omission, because "no model ran" and
"we did not measure the model" are different claims. The agentic arm is **codex on the ChatGPT seat**,
so `billing_mode=subscription_notional`: token counts are real, `estimated_cost_usd` is `None` by
construction, and dollars live in `subscription_notional_usd` as a projection. Contract verified
against codex-cli **0.152.0** on this machine (the repo's record is 0.147.0): `input_tokens` already
includes the cached and cache-write buckets, so fresh input is a subtraction.

⚠️ **AutoComp cannot be driven by the codex seat** (its bridge gates on provider). A same-model
head-to-head therefore needs both arms on a shared provider — an open decision, recorded rather than
papered over, because the harness is a first-order variable here.
