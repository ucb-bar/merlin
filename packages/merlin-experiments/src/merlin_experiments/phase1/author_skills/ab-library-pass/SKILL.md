---
name: ab-library-pass
description: "A/B-test one catalogue library mechanism against your current compiler by cycles. Use once test cases pass and a self-check performance_brief shows low array utilization or names suggested_mechanisms for a case's busiest unit: apply the mechanism in variants/NAME/ or behind a tuning knob, check both on the same cases, compare cycles and numerics with scripts/ab_compare.py, keep it only if cycles fall with numerics unchanged."
---

`$SKILL` = the directory holding this file. Run commands from the workspace root.

1. **Pick the case and mechanism.** In your newest submission self-check response
   (`.qa_channel/resp_ID.json` or its `--out` file) read `performance_brief.largest_headroom` (or
   `lowest_utilization`). Take the first entry; note `test_case`, `dominant_modeled_resource` and
   `suggested_mechanisms`. Those naming the busy unit come first. Prefer one whose
   `workload_check` is `satisfied`; otherwise run the `optimization_suggest.py` command it prints.
   The ranked list over all cases is in `qa/latest_completed_grade.json`.
2. **Find its API.** `python $SKILL/scripts/mechanism_api.py ID` prints the module and symbol, the
   signature and docstring from `submission/compiler_library`, the applicability, the adapter you
   still have to supply (`remaining_adapter_or_feedback`) and the library modules that already call
   it (your nearest examples). `--list [--facet F]` lists all ids.
3. **Apply it in a copy.** Either:
   - variant: `cp -r submission variants/NAME` (letters, digits, `._-`), then call the mechanism in
     `variants/NAME/` where your compiler makes that choice today; or
   - knob: in `submission/` call
     `knob("NAME", default="off", values=("off", "on"))` from `merlin.compile.tuning_knobs`
     (in `compiler_library`; module docstring has the details) and apply the pass only for `"on"`.
     Declare it once with `declare("submission", {"NAME": Knob(("off", "on"), default="off",
     mechanism="ID")})`, then check:
     `PYTHONPATH=submission/compiler_library python -m merlin.compile.tuning_knobs check submission`.
     To time a value yourself, copy to a variant and run `... tuning_knobs set variants/NAME NAME=on`.
   Gate the pass on the facts it needs (shapes, capacities) so other cases compile as before.
   Switches you already have: `performance_brief.tuning_knob_candidates` (declare them the same way).
4. **Check both on the SAME cases** (the entry's case plus 2-3 related shapes, one request each):
   ```
   python agent_selfcheck.py --capsules A,B,C --wait-budget 5      # submission/: prints an id
   python agent_selfcheck.py --submission variants/NAME --capsules A,B,C --wait-budget 5
   mkdir -p ab; python agent_selfcheck.py --attach ID --out ab/base.json   # ab/NAME.json for the variant
   ```
   The default sim gives measured certifying-tier cycles. `--suite scalability` gives timing-model
   estimates for the large public shapes. Use the same suite and sim on both sides.
5. **Compare.**
   `python $SKILL/scripts/ab_compare.py ab/NAME.json ab/base.json` (or
   `ab_compare.py variants/NAME`, which uses the newest submission/ rows from `.qa_channel`) prints
   per-case cycles before/after, delta and numerics, then a verdict:
   - `KEEP` (exit 0): total cycles fall, no case slower, numerics unchanged. Carry the change into
     `submission/` (or set the knob's value in its settings file). Then re-check
     `--capsules changed` and see that the cases still pass.
   - `MIXED`: restrict where the pass applies to the facts of the cases it helped, then re-check.
   - `DO NOT KEEP`: numerics changed or no gain. Read the mechanism's `evidence_to_request`, fix
     the adapter or try the next suggested mechanism.
   - `INCONCLUSIVE`: no comparable cycles. Run both sides with the same suite, sim and cases.
6. **Let idle CPUs keep timing.** While cores are free, the harness times every `variants/NAME/`
   that differs from `submission/`, and every single-knob change declared in
   `submission/TUNING_KNOBS.json`, on the cases with the most headroom. A variant that imports a
   catalogue module is labelled with that mechanism. A knob carrying `mechanism` goes first when it
   matches the brief. The results go to `qa/idle_tuning.json` and `performance_brief.idle_tuning`
   (`delta_pct`, `numerics`, `how_to_apply`). They are bound to the current submission digest. They
   are timing-model estimates, so confirm a win with step 4 before you adopt it.

Compile-only, no cycles: `merlin.transforms.pipeline_policy.evaluate_variants`. Sweeps: the
`explore-schedules` skill; per-shape cutoffs: `shape-schedule-selection` skill.
