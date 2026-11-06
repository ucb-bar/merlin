# Public non-model fullgrade instrument

This directory grades **all 431 non-model public/dev capsules** in `fin6w2-m27b`: 320 core and 111 scalability. `SY_micro_model` is the remaining capsule in the 432-capsule roster. Model requests fail explicitly; omitted models are listed in the report. A successful report qualifies only the selected non-model scope. `full_release_passed` is always null.

No grader, simulator, compiler, model capture or draft generator was run while developing this instrument. No Merlin, template or live-run files were modified. No operator-private model specification, answer/expected-output file or hidden profile was read. The prior fullgrade directory was absent in this workspace; the saved prompts and installed `regrade.py` were the starting material.

## Run a smoke

Set `CPUS` to **four allocated CPUs** (comma list or taskset ranges). The orchestrator should run:

```bash
bash /scratch2/loren/aws/m27-fix/fullgrade/regrade_full_eval.sh \
  --corpus fin6w2-m27b \
  --cand /scratch2/loren/aws/m27-fix/wt-s1 \
  --out "/scratch/loren/eval/results/fullgrade-smoke-$(date -u +%Y%m%dT%H%M%SZ)" \
  --cpus "${CPUS:?set CPUS to four allocated CPUs}" --workers 2 --sim-jobs 2 \
  --names G_matmul_m32k32n32_raw,SY_movement_fill_i8_aligned,PM00_m16n16
```

The two core capsules declare **L0/L1/L2/L3**, with `oracle_tier_ceiling: L3`; they require Verilator L3 to pass. `PM00_m16n16` is a scalability capsule and remains Spike L0–L2. Native fail-fast can stop a bad candidate before L3; that does not qualify it at L3. Check `results/core-score.json`, the durable native capsule results, and `results/fullgrade-summary.json` for actual tier outcomes. No `--covering-only` option is offered.

For the whole non-model roster on **24 allocated cores**, remove `--names`, use `--workers 12 --sim-jobs 4`, and supply all 24 CPUs. `--workers` is per group; workers times groups may not exceed the CPU count. `--groups core` and `--groups scalability` are supported. `--limit N` selects a sorted prefix per group and marks the report as a subset. Unknown names and explicitly requested model capsules are errors. Names may span both selected groups.

Use a fresh `--out` for every invocation. Exit 0 means every requested non-model group passed; any non-pass, incomplete measurement, exception or missing child report exits nonzero. `results/exit.log` records child exits. The JSON summary embeds the group reports and explicitly withholds a whole-release verdict. A group with no names requested is recorded as `not_requested`, never passed.

## Exact policy and deliberate scope

All source references below are in the read-only stage:

`M=/scratch/loren/p12-unsealed/stage/m27b/merlin`

`P=$M/packages/merlin-experiments/src/merlin_experiments/phase1`

`G=$M/packages/merlin-experiments/src/merlin/targetgen`

There are two different release paths. **`joint_final_coverage.validate()` is explicitly a cheap functional drain**, with `_adapters("spike")`, only the L2 adapter, and `functional_public_tier="L2"` (`$P/joint_final_coverage.py:184`, `:226`). Its public-grade branch rewrites mandatory tiers down to L2 (`$G/capsule_grade.py:1488`). It does not run RTL. Merely copying that function would reproduce the limitation this task is fixing.

This instrument instead uses **the release checkpoint/full ladder for core**, through `CR.qa_checkpoint_adapters` (`$G/capsule_runner.py:1259`). That function delegates directly to `oracle_adapters` (`$G/candidate_scope.py:71`), which is also the formal public grader's factory (`$P/feedback/formal.py:450`). It passes the context's native oracle kwargs, including any explicit readback/transport selection. The wrapper sources the same `env.sh`, which pins `MERLIN_REQUIRED_RTL_ENGINE=verilator`. The driver checks that core L2 is Spike and L3 is Verilator; it never substitutes another engine. Runtime native oracle results must establish their own fidelity; an adapter name alone supplies no execution credit. The chipyard engine-selection policy is `$M/src/merlin/targetgen/oracle_policy.py:151`.

Scalability uses the release's exact `_adapters("spike", ...)` path (`$P/feedback/public_scalability.py:152`; factory `$P/feedback/selfcheck.py:363`), with only L2 supplied. A native `tier_policy.floor_exempt` scope is applied to the **copied** case directories, matching `$P/joint_final_coverage.py:218` and `$P/feedback/public_scalability.py:191`. It never receives L3 or a fabricated RTL obligation. The declared scalability tiers are already L0–L2, so no functional-tier rewrite is needed.

Capsule bytes are copied unchanged: mandatory tiers, `oracle_tier_ceiling`, separate correctness/timing ceilings, `extends`, expected refusals, host guards and trace requirements are all left to the native runner. `tier_policy.required_tiers` owns floor application (`$M/src/merlin/targetgen/tier_policy.py:409`); the runner enforces execution ceilings before ordering adapters (`$G/capsule_runner.py:5355`). In this template, **every core capsule declares L0–L3 and an L3 execution ceiling**. No hardcoded capsule list chooses L3.

Each complete group is submitted in **one `CG.grade` call**, using the same public/dev labels, frozen contract and target experiment. Native eligibility filtering and the greedy covering set remain inside `run_suite` (`$G/capsule_runner.py:6637`). Covering-set priority is **not** a rule to omit mandatory L3 from capsules outside the cover. Native budgets, fail-fast, tier ordering and ceiling decisions remain in control. Unlike the old wrapper, there is no process-sharding of a group's roster: that would recompute its covering set over arbitrary slices and split dependency scheduling/budget state. Both groups run concurrently, and the native scheduler uses threads/operator children within each group.

The base per-capsule budget is 900 seconds, matching the unexpired final-drain ceiling (`$P/joint_final_coverage.py:236`); `--timeout` can explicitly change it. Native `CG.grade` raises it from the copied run's own history (`$G/capsule_grade.py:1552`; `$G/capsule_budget.py:63`). `timing_history(_history_runs_roots(run_dir))` supplies that history rather than dropping known slow L3 observations. With a 900-second request, this stage's measurement-derived cap is 3600 seconds. Queue, step and tier deadlines remain native; the instrument does not invent an outer 900-second process kill.

Package integrity, emission, numeric, trace, source/offload and refusal decisions all come from `CG.grade`. Exact cohort closure uses the release's `_original_cohort_complete` (`$P/feedback/qa.py:47`), plus native gradeability, numeric and structural flags and an exact selected-name match. This follows the checkpoint closure (`$P/feedback/qa.py:1107`) without incorrectly requiring JFC's **Spike-only** `functional_public_evidence` claim (`$P/joint_final_coverage.py:72`). A full-ladder score does not produce that claim. Nine focused tests include this distinction and ensure denominator shrinkage cannot yield success.

Both original run-scoped memos are retained unchanged from `regrade.py`: `_toolchain` resolves/verifies once per manifest/snapshot/marker identity; `_payload_content` reuses a digest only while all file stat stamps remain identical. The original compiler-library/shared-pass contexts and `functional_service`/`compiler_execution` are retained. No source monkeypatch is persisted to disk. `--no-memo` disables these two in-process memos; the existing snapshot lease cache remains enabled, as in the original wrapper.

The wrapper keeps the original bwrap same-path run/workspace binds, staged root/Merlin/contract/evidence/facts mounts, taskset, nice 10 and shared CPU pool. It copies the template run, candidate, and workspace into its output. **Writable bundle views use the independent workspace copy**, because m27b needs them; it never binds the template workspace writable. Candidate bytes are read-only in the namespace and checked before/after grading. Script/settings snapshots prevent mid-run edits from changing an invocation. CPU and simulator counts are explicit and all paths/arguments use shell arrays or `%q` quoting.

The instrument does not run `prepared.verify_inputs()`, reproduce campaign admission/deadline/group-share/reclaim mechanics, grade hidden capsules, close the model gate, run visible full-program broker checks, or perform the separate formal resource/performance/paper-accuracy/full-model gates. It checks the native selected scalability input binding and both original group rosters, including original `coverage.json` if present and any declared final-only core capsules. Removing the model creates an explicitly scoped cohort of 320 core capsules; the report does not claim that this is the descriptor's full 321-capsule core cohort. This is a public non-model verification instrument, not a finished full-release verifier.

## Timing and validation

On 24 cores with four Verilator slots, plan provisionally for **several hours, roughly 3–12+ hours** for a mostly passing full non-model roster. A useful estimate is `copy/setup + max(total functional worker time / workers, total RTL slot time / 4)`. The tier-policy documentation gives an illustrative expensive-oracle cost around 132 seconds and separately describes substantial cycle-dependent cost (`tier_policy.py:16`, `:252`); at 132 seconds each, 320 runs would already take about three hours of wall time with four slots. Those source measurements are not a calibration for this candidate or this host. Deep cases, cold setup, queueing, historical budget raises and timeouts can extend that considerably. This is a planning estimate, not a completed benchmark; there is no completed final-validation log on garden. Cheap Spike-only evaluation timings do not predict full L3 wall time.

Performed: Python compilation; `bash -n` on both shell scripts; stage-venv import/signature checks for the native grade/context/selection interfaces; and nine synthetic-data tests. No `CG.grade`, adapter/oracle invocation, bwrap run, full-model gate, capture or draft generation was executed.

```bash
PYTHONDONTWRITEBYTECODE=1 \
/scratch/loren/core-aten-snap/venvs/merlin-wtC/bin/python -B \
  /scratch2/loren/aws/m27-fix/fullgrade/test_fullgrade.py -v
```

The separate model investigation, field-by-field draft format and acceptance limits are in [MODEL_SPEC_REPORT.md](MODEL_SPEC_REPORT.md). `draft_public_model_spec.py` writes only public projection review drafts below this directory; it has not been run against the stage or private inputs.
