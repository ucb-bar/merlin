---
title: "Running the Gemmini recipe-select agent campaigns"
kind: guide
status: current
owner: core
last_verified: 2026-09-03
related: [agent_uses_the_compiler_gemmini]
code_refs: [merlin/experiments/agent_recipe_select_v0/scripts/run_census_campaign.py, merlin/experiments/agent_recipe_select_v0/scripts/census_workloads.py, merlin/experiments/agent_recipe_select_v0/scripts/compare_arms.py, merlin/experiments/agent_recipe_select_v0/scripts/agent_compile.py]
---

# Running the recipe-select agent campaigns

Two agent arms over a frozen, RTL-certified Gemmini backend, measured on the ResNet-50 and TinyLlama
kernel census. For *why* the experiment is shaped this way, read
[the design note](../design/agent_uses_the_compiler_gemmini.md) first — in particular the section on
why the two arms cannot be ranked by their own speedups.

## Prerequisites

* `.venv/bin/python` in the repo (the project interpreter — see `venv-python`).
* AutoComp's **own** interpreter for that arm, at `$MERLIN_EXT_AUTOCOMP/.venv/bin/python`
  (default `/scratch/agustin/projects/autocomp`). It is named explicitly and never inherited:
  running the AutoComp arm under merlin's interpreter once made a known-good reference kernel read as
  numerically wrong.
* A codex seat (`auth_mode: chatgpt`). Runs are billed as `subscription_notional`; `billed_usd` stays
  empty on seat rows and empty never means zero.

> ⚠️ **The shared venv can import the wrong merlin.** Sibling checkouts on this host have shadowed
> `merlin` in the past, so a campaign silently measured another tree's compiler. `_track.py` calls
> `assert_right_merlin()` at import and `py_env()` prepends the correct `merlin/python`. If you
> invoke a script by hand, go through those rather than around them.

## 1. Build the workloads

```bash
.venv/bin/python merlin/experiments/agent_recipe_select_v0/scripts/census_workloads.py \
    --emit-workloads out/build/recipe_select_workloads
```

K is preserved on every shape — it is where accumulation, residency and spill behaviour live — and
only the parallel extents are clamped to the per-candidate GSIM wall budget. Sub-tile extents
(ResNet-50's `M=1` classifier, TinyLlama's `M=8` at sequence 8) are left exactly as they are: a
one-tile floor would round them *up*, which is a different workload, not a smaller one.

## 2. Run an arm

```bash
# recipe arm
.venv/bin/python merlin/experiments/agent_recipe_select_v0/scripts/run_census_campaign.py \
    --workloads out/build/recipe_select_workloads --arm recipe \
    --budget 16 --slots 6 --exclude .interface. \
    --log-dir out/runs/gemmini/recipe-select/campaign_recipe_$(date -u +%Y%m%dT%H%M%SZ)

# autocomp arm (fewer slots: it evaluates serially inside one Gemmini checkout)
... --arm autocomp --slots 3 --log-dir .../campaign_autocomp_<TS>
```

Each shape runs in its own session (`setsid`), so a campaign survives the parent shell exiting.
`--slots` bounds concurrency for the machine's sake only — shapes are independent.

### Resuming an interrupted campaign

A shape costs 30–150 minutes of cycle-accurate simulation, so never restart from scratch:

```bash
... --resume --log-dir out/runs/gemmini/recipe-select/campaign_recipe_<NEW_TS>
```

`--resume` scans sibling `campaign_*` directories and skips shapes that a prior campaign **of the
same arm** completed with `rc == 0`. The arm is part of the key: both arms log into one directory and
measure the same shapes differently, so a recipe rerun must not skip a shape only AutoComp finished.
A shape whose driver died mid-flight has no manifest row and is correctly re-run. Use
`--resume-from DIR` to name prior campaigns explicitly, and `--dry-run` to see the skip list first.

## 3. Compare the arms

```bash
.venv/bin/python merlin/experiments/agent_recipe_select_v0/scripts/compare_arms.py
.venv/bin/python merlin/experiments/agent_recipe_select_v0/scripts/compare_arms.py --json --all
```

By default it prints only shapes **both** arms have finished. The `recipe vs ac` column is computed
from **absolute GSIM cycles** — the per-arm `×own baseline` columns sit beside it and are not
comparable to each other, because the recipe arm's baseline is the compiler default while AutoComp's
is a hand-written C seed that starts ahead on real layers.

## Model tiering and token accounting

The AutoComp arm splits planning from implementation:

```bash
--plan-model codex/gpt-5.6-sol:high  --code-model codex/gpt-5.3-codex-spark:low
```

Both are passed to codex with `--model`. (They were once recorded but not passed, which made the
tiering cosmetic — two model strings in the record, one model answering both.) Other reachable slugs:
`gpt-5.6-terra`, `gpt-5.6-luna`, `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`.

Tokens are recorded in **disjoint buckets** per model and tier — fresh input, output, cache-read,
cache-write. Note that this CLI's `input_tokens` **already contains** the cached and cache-write
buckets, so fresh input is a *subtraction*; adding them overstated a measured round by 85%.

> ⚠️ **A tier can fail silently.** When a model hits its usage limit, codex emits
> `{"type":"error", ...}` on stdout and the run records a call with 0 tokens and no reason — 43% of
> code-tier calls in one campaign. Before trusting a run, check for iterations with `calls > 0` and
> `output_tokens == 0`, and check `eval-results-iter-*/` for candidate counts of zero.

## Oracles

`GSIM` (`GemminiGsimSerialClkConfig`) is the search oracle. FireSim is the whole-model verdict oracle
and takes the `fsq.py` queue lock, then `kill → infrasetup → runworkload → kill`, every run. **The
two never share an axis** — the same capsule reads 510 cycles on FireSim and 317 under Verilator — so
`oracle_engine` is mandatory on every row carrying a cycle count.
