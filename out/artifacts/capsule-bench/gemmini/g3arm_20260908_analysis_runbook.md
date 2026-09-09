# Batch `g3arm_20260908` — what it is, and how to analyse it honestly

Three arms of the tooling ladder, launched in parallel 2026-09-08T08:25:18Z on the codex
subscription seat, cold-start, at the full/public info set.

| arm | run dir | bundle | adds over the arm above |
|---|---|---|---|
| arm-1 | `raw_baseline/rb_g3arm_20260908` | `raw_baseline_public_v0` | — |
| arm-2 | `cpp_merlininfra/rbinfra_g3arm_20260908` | `cpp_merlininfra_public_v0` | `cpp_oot_generators` |
| arm-3 | `merlin_assisted/merlin_g3arm_20260908` | `merlin_assisted_public_v0` | `merlin_infra, xdsl_kit, cca_spine, isa_tools, cca_tools` |

Held identical across arms (verified from each `environment.yaml`): model `gpt-5.6-sol`, effort high,
driver codex, provider subscription, sandbox bwrap, `repo_sha 2c547c68`, `--schedule continuous`,
`max_wall_s 43200`, `round_timeout_s 14400`, `qa_timeout_s 1200`, `grade_interval_s 900`,
`sim_max_jobs 4`, scope 97 public/dev + 15 held out, golden mask OK (1291 masked, 0 leaked).

## Commands

    H=merlin/experiments/capsule_bench/harness
    .venv/bin/python $H/agg_ab_results.py --tag g3arm_20260908
    .venv/bin/python $H/agg_agentic_results.py
    .venv/bin/python $H/agg_by_model.py
    .venv/bin/python $H/plots/make_plots.py \
      --runs raw_baseline/rb_g3arm_20260908,cpp_merlininfra/rbinfra_g3arm_20260908,merlin_assisted/merlin_g3arm_20260908 \
      --compare-run raw_baseline/rb_g3arm_20260908 \
      --out out/artifacts/capsule-bench/gemmini/g3arm_20260908_plots
    .venv/bin/aet spend out/runs
    .venv/bin/python $H/abc_status.py --tag g3arm_20260908     # live, while running

## Six things that will mislead you if you skip them

1. **This batch does NOT compare to the 93/97 run** (`merlincirct_gemmini_universal_smallread_resnet50_20260907`,
   arm-4). That run was SEEDED from a 190 MB pre-developed package and opened at 83/97 in round 0;
   these arms cold-start. Its driver kickoff prompt also differed — see
   `out/runs/gemmini/capsule-bench/ab_batch_g3arm_20260908.tree_state.diff`, which records the
   uncommitted `codex_agent.py` prompt edits live at launch (identical across all three arms here, so
   the WITHIN-batch contrast is unaffected).
2. **The ladder is not strictly nested at arm-2 -> arm-3.** `tool_registry.ARM_TOOLS` gives arm-3
   `_ASSISTED` WITHOUT `cpp_oot_generators`, so arm-3 SUBSTITUTES the xDSL kit for the C++ scaffold
   rather than adding to it. arm-1->arm-2 is a clean +1 tool and arm-3->arm-4 a clean +2, but
   arm-2->arm-3 is a switch. `AGENT.md`'s "raw_baseline < cpp_merlininfra < merlin_assisted" claim
   does not hold at that step. Do not attribute the arm-2->arm-3 delta to "one addition".
3. **Quote `tier_reached["L3"]` beside every `n_passed`.** Under bwrap the materializer caps
   `required_oracle_tiers` at L2, so the elaborated-RTL engine runs and is recorded but does not gate.
   A prior submission travelled as 20/20 with `tier_reached[L3] = 1/20`. `highest_tier` is a suite
   MINIMUM — one failing capsule pins it for the whole run.
4. **Two different denominators.** The fast first grade and the interval grades report 95 capsules;
   the authoritative round verdict reports 97. Never plot them on one axis without saying which.
5. **Drop `stage: first_grade_loop_tier` rows.** The fast grade legitimately reports `n_capsules: 0`
   before a gradeable manifest exists (measured: arm-3's first verdict was `0/0`). Those rows chart as
   a collapse to zero.
6. **Cost is notional, not spent.** `billing_mode: subscription_notional` on all three arms:
   `estimated_cost_usd` is None by design and the projection lives in
   `cost.subscription_notional_usd`. Never sum it into a metered budget.
7. **A mid-run aet record has turns but ZERO tokens, and that is honest.** Codex emits token usage only
   at `turn.completed`; a turn still in flight has `turn.started` and nothing else. MEASURED on this
   batch at 08:50Z: arm-3's `logs/metrics.jsonl` carried `num_turns: 82` and every token field 0, with
   the raw stream showing 1 `turn.started` and 0 `turn.completed`. Tokens appear once the turn closes.
   So a zero token count on an in-flight run means "not reported yet", NOT "no tokens" — and on a run
   killed mid-turn it means "unrecoverable", not zero. Turns and tool calls do survive a kill.

## Check the treatment actually fired before believing a null

`abc_status.py`'s `tools` column counts Bash calls invoking THAT arm's granted tooling. A rung sitting
at 0 has an untreated treatment and its score is not evidence about the tool. arm-1 shows `n/a` (it
grants none). Cross-check any tool-call figure against an independent count of the same transcripts: a
`tool_use` block missing its `id` records ZERO calls, and `subagent_tool_calls_tracked` is False by
design, so `tool_calls` is top-level-agent only while `tokens` are subagent-inclusive.

## ⚠️ arm-1 needs `--resume --seal-current` to get an official grade

MEASURED 2026-09-08T~10:20Z: `rb_g3arm_20260908` has `feedback_health.healthy = false` from
**1 expired request out of 58** (57 completed). The expired one is `req_7913_...`, a `sim: spike`
self-check on `M3_host_island_seam_gemmini` with a 420 s client deadline that went unanswered — a
capsule heavy enough that spike plausibly overran while 18 concurrent sims and a full test-suite run
were competing for CPU (the test run has since been stopped).

Two consequences:

1. **The agent asked for feedback and got none once.** That is a real, if small, degradation of arm-1's
   treatment, not just bookkeeping. Note it beside arm-1's score.
2. **The final official public+hidden grade will be SKIPPED.** `run_baseline_qa_loop.py:4372` gates it
   on `workflow_conformant and feedback_health["healthy"]`, and `:4421` then sets
   `official_grade["failures"] = ["feedback_channel_unhealthy"]`. `_feedback_health` is recomputed from
   the `.qa_channel` files each time, and the unanswered `req_*` file persists, so this does NOT heal on
   its own.

**Remedy — the designed one, not a workaround.** Do NOT delete the `req_` file and do NOT fabricate a
`resp_`/`done_` pair; either would destroy the evidence that the promised feedback was not delivered.
Instead finalize the arm with the operator seal:

    .venv/bin/python merlin/experiments/capsule_bench/harness/run_baseline_qa_loop.py \
      --run-id rb_g3arm_20260908 --resume --seal-current   # (+ the original flags)

`--seal-current` stops authoring at the last completed checkpoint and runs the ordinary official
public/hidden grade and immutable freeze, recording an INCOMPLETE operator seal. It never reports
convergence and never bypasses integrity, and `formal_complete` stays False
(`run_baseline_qa_loop.py:4447`) — which is the honest outcome here.

**Design observation, for later, not to change mid-run:** one expiry in 58 requests (98.3% delivered)
condemns the whole official grade. A 420 s synchronous deadline is also tight for a host-island or
whole-model capsule when N arms run in parallel; consider scaling the self-check deadline with arm
count, or making the health verdict proportional and named rather than binary.

## ⚠️ Regenerate `full_suite_audit.json` before quoting fig4's top two panels

`plots/make_plots.py` reads ONE `full_suite_audit.json` from the reports dir — whatever ran last left
behind. MEASURED 2026-09-08: the copy on disk was from **2026-07-28**, a **25-capsule** corpus, backend
`rb_abc4`, with no `generated_at` field, and `fig4_ab_summary.png` rendered its coverage and cycle
panels from it under this batch's title. The corpus grew 11 -> 97 over the study, so a coverage
fraction from another cell is not comparable to this one.

fig4's coverage panel now names the audit's capsule count and backend list in its own title, so a
stale audit can no longer masquerade as this batch. Before quoting those panels, run
`full_suite_audit.py` for THIS batch and confirm the title says 97 capsules and lists this batch's
three run ids. Until then, read only the bottom-left effort panel as belonging to this batch.

## Pipeline proven before the runs finished

Dry-run 2026-09-08 against the in-flight run dirs: `agg_ab_results.py --tag g3arm_20260908`,
`agg_agentic_results.py` and `make_plots.py` (all 5 figures) complete end-to-end with the three arms.
Two defects were found and fixed by doing this early rather than at the end: `agg_ab_results.plot`
raised `KeyError: 'cpp_merlininfra'` on its colour map AFTER writing `ab_results.json` (numbers written,
figures missing), and its bar geometry hardcoded three arms so a fourth overlapped and sat off-centre.

## ⚠️ The batch was interrupted at ~2.5 h and RESUMED — read the wall-clock accordingly

All three drivers were launched with plain `nohup &` from an agent Bash call. When that Claude Code
process exited they were **killed mid-grade** — no traceback, no refusal, the launch logs just stop at
2026-09-08T10:40/10:41/11:03Z. The monitor watching them was torn down by the same event, so nothing
reported it; the loss was found by `pgrep` in the next session, ~4.7 h later.

Scores at the interruption (the last verdicts of the first attempt):

| arm | score | turns / tool calls |
|---|---|---|
| arm-1 `rb` | 83/97 | 343 / 261 |
| arm-2 `rbinfra` | 91/97 | 269 / 185 |
| arm-3 `merlin` | 9/95 | 420 / 336 |

Those turn and tool figures exist ONLY because the telemetry sink runs on the grader cadence; the
former end-of-run-only sink would have recorded nothing for any of the three.

**Resumed 2026-09-08T15:44Z** via `scratchpad/resume_g3arm.sh`, `setsid`-detached so a further session
teardown cannot take the arms with it. All three printed
`[resume] reusing existing workspace + submission`, and submission file counts were unchanged
(17 / 56 / 19), so no authored work was lost. `verify_bundle_snapshot` checks the snapshot against
itself rather than the live repo, so each arm kept its ORIGINAL frozen inputs even though HEAD moved
from `2c547c68` to `fde5ce4d` under it — the treatment is preserved across the interruption.

**Two consequences for reading the numbers.**

1. **Wall-clock is not continuous.** There is a ~4.7 h gap between the first attempt and the resume,
   and `qa_loop_state.yaml` was never written (`_checkpoint()` fires on round completion and all three
   died inside round 0), so `active_wall_s` restarts at zero. The resume was given
   `--max-wall-s 34200` — the REMAINDER of the declared 12 h, not a second full budget — so total agent
   wall stays comparable to the batch's declared config and across the three arms. Any wall-clock or
   cost-per-hour figure must be computed from the per-round records, never from first-launch to finish.
2. **Round numbering restarts, and the resumed verdicts OVERWRITE the originals.** The resume
   re-enters at round 0, so it re-uses the same `verdict_inturn_r0000_tNNNNNN.json` filenames. This is
   not two generations side by side -- the second generation CLOBBERS the first. Caught in the act:
   `merlin_g3arm_20260908/qa_history/verdict_inturn_r0000_t000001.json` read 0/95 before the resume and
   51/95 after, with the original value recoverable only from the launch log.

   The surviving pre-interruption verdicts were copied to
   `<run_dir>/qa_history/preinterruption_gen1/` (deliberately not matching `verdict_*.json`, so no
   harness glob sees it). Contents:

   | arm | fast | t000001 | t000002 | t000003 |
   |---|---|---|---|---|
   | `rb` | 0/95 | 48/95 | 83/97 | — |
   | `rbinfra` | 0/95 | 57/95 | 91/97 | — |
   | `merlin` | 0/0 | (lost: 0/95) | 0/95 | 9/95 |

   Read the first attempt from that directory and the second from `qa_history/` itself. An mtime sort
   over `qa_history/` alone will silently show only the resumed generation.

## Keeping the batch alive, and the disk headroom it needs

**Auto-resume guard (detached).** `scratchpad/watchdog_g3arm.sh` runs `harness/watchdog.py --tag
g3arm_20260908` under `setsid` (ppid=1, own session), so it survives a Claude Code teardown -- the very
event that killed the arms the first time. It relaunches any arm whose process has exited while
`converged` is not true, with a resume command byte-identical to the running one (verified flag by
flag against `LB._arm_cmd`).

Two defects in that watchdog had to be fixed before it could be trusted with this batch:

* **`codex` was not among its `--driver` choices**, while being `launch_ab_batch`'s DEFAULT driver, and
  its default was `auto`. A resume therefore passed no `--driver` at all, and `auto` can never resolve
  to codex: it routes a non-Anthropic model id to the Bedrock Converse loop. A guarded codex batch
  would have come back as a different agent on a metered account.
* **It defined none of the eight run-shape knobs** that `_arm_cmd` reads via `getattr(..., default)`.
  Most damaging was `max_wall_s`, whose fallback is `0` -- no wall cap -- so a guarded continuous run
  would have returned unbounded. `--model` had also drifted to `claude-opus-4-8` against the launcher's
  `gpt-5.6-sol`.

**Disk.** Reclaimed 48 G on 2026-09-08 by deleting `bundle_inputs/` from FOUR finished runs
(`merlincirct_g4p1_20260905`, `_g4p1_biasabi_20260906`, `_qdq_cleanroom_20260907`,
`_qdq_focused_build9_20260907`), keeping each one's `workspace/submission`. Those snapshots are real
full copies (`nlink=1`, apparent == on-disk), not hardlinks, so the space is genuinely returned. They
are sealed read-only, so `chmod -R u+w` is needed before `rm`. /scratch went 220 G -> 268 G free.

Consumption runs about 9.5 G/h with three arms live, so the remaining budget needs roughly 90 G.

If it gets tight again, in the order I would take them:

| candidate | size | cost of taking it |
|---|---|---|
| `_qa_ws/*/bundle_inputs` of any newly finished run | 13 G each | forecloses `--resume`/`--seal-current` on that run |
| `out/artifacts/cache/weight_panel` + `weight_prepack` | 6.8 G | regenerable; may belong to another session's model work |
| `_qa_ws/merlincirct_gemmini_universal_smallread_resnet50_20260907/bundle_inputs` | 13 G | ⚠️ the 93/97 run could then never be sealed; its submission lives ONLY in that workspace |
| `out/artifacts/recaptures` | 52 G | declared PURGEABLE, but confirm no model work depends on it |
| `out/artifacts/cache/elf-builds` | 24 G | ⚠️ DO NOT while arms are live -- it is the ELF/tier-cert cache they reuse, so purging it makes them redo L3 work |

Never touch a LIVE arm's `bundle_inputs`: `verify_bundle_snapshot` checks it on every resume, and the
run cannot restart without it.
