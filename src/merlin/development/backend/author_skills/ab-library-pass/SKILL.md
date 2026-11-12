---
name: ab-library-pass
description: "A/B-test one catalogue library mechanism against your current compiler by cycles. Use once test cases pass and a self-check performance_brief shows low array utilization or names suggested_mechanisms for a case's busiest unit: apply the mechanism in variants/NAME/ or behind a tuning knob, check both on the same cases, compare cycles and numerics with scripts/ab_compare.py, keep it only if cycles fall with numerics unchanged."
---

`$SKILL` is this directory. Run from the workspace root.

1. **Pick the case and mechanism.** In your newest submission self-check response
   (`.qa_channel/resp_ID.json` or its `--out` file) read `performance_brief.largest_headroom` (or
   `lowest_utilization`). Note `test_case`, `dominant_modeled_resource`, `suggested_mechanisms`.
   Prefer `workload_check: satisfied`; otherwise run its `optimization_suggest.py` command.
2. **Find its API.** `python $SKILL/scripts/mechanism_api.py ID` prints the module and symbol, the
   signature and docstring from `submission/compiler_library`, the applicability, the adapter you
   still need (`remaining_adapter_or_feedback`) and nearby callers. `--list [--facet F]` lists ids.
3. **Connect and apply.** Supply the author adapter in [reference/adapter.md](reference/adapter.md).
   It binds source/effects to shared passes and your target emitter with completion callbacks.
   `heavy-pass-recipes` has scheduling examples. Run:
   ```
   python $SKILL/scripts/apply_library_pass.py --submission submission --variant variants/NAME --adapter tools/library_pass_adapter.py --mechanism ID --case CASE --receipt ab/NAME-application.json
   ```
   Only `APPLIED_EMITTED_CHANGE` permits the next step. `CONSTRUCTION_REQUIRED` names a missing
   author hook/mapping; `REFUSED` includes unchanged emission. Retry failed variants with a new name.
   The tool checks actual compiler output, supplies no target implementation or legality proof, and
   requires current selected facts (missing facts stay UNKNOWN). Declare existing knobs with `merlin.compiler.selection.knobs`
   and run its `check submission`; `performance_brief.tuning_knob_candidates` lists declared choices.
4. **Check both on SAME cases** (the entry's case plus 2-3 related shapes, one request each):
   ```
   python agent_selfcheck.py --capsules A,B,C --wait-budget 5      # submission/: prints an id
   python agent_selfcheck.py --submission variants/NAME --capsules A,B,C --wait-budget 5
   mkdir -p ab; python agent_selfcheck.py --attach ID --out ab/base.json   # ab/NAME.json for the variant
   ```
   The default sim gives measured certifying-tier cycles. `--suite scalability` gives timing-model
   estimates for the large public shapes. Use the same suite and sim on both sides.
   Parallel: `python agent_selfcheck.py --sweep variants/a,variants/b --capsules A,B,C --wait-budget 5`;
   `--attach SWEEP_ID` collects. [Sweeps](reference/sweeps.md) covers knobs, bundles and identities.
5. **Compare.**
   `python $SKILL/scripts/ab_compare.py ab/NAME.json ab/base.json` (or
   `ab_compare.py variants/NAME`, which uses the newest submission/ rows from `.qa_channel`) prints
   cycles before/after, delta and numerics, then a verdict:
   - `KEEP` (exit 0): total cycles fall, no case slower, numerics unchanged. Carry the change into
     `submission/` (or set the knob's value in its settings file). Then re-check
     `--capsules changed` and see that the cases still pass.
   - `MIXED`: restrict where the pass applies to the facts of the cases it helped, then re-check.
   - `DO NOT KEEP`: numerics changed or no gain. Read the mechanism's `evidence_to_request`, fix
     the adapter or try the next suggested mechanism.
   - `INCONCLUSIVE`: no comparable cycles. Run both sides with the same suite, sim and cases.
6. **Let idle CPUs keep timing.** The harness explores changed variants and declared single-knob
   alternatives on high-headroom cases. Read `qa/idle_tuning.json` and
   `performance_brief.idle_tuning` (`delta_pct`, `numerics`, `how_to_apply`). These are model
   estimates; confirm with step 4. Missing adapters or alternatives are construction tasks.
   [Sweeps](reference/sweeps.md#idle-construction-and-baselines) explains source identities,
   baseline/candidate progress and existing-emitter selection requirements.

Compile-only: `merlin.compiler.transforms.pipeline_policy.evaluate_variants`. See `explore-schedules`
and `shape-schedule-selection`.
