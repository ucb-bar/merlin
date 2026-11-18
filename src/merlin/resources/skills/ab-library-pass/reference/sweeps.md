# Parallel model comparisons

Run from the workspace root:

```
python agent_selfcheck.py --sweep variants/a,variants/b --capsules A,B,C --wait-budget 5
python agent_selfcheck.py --sweep --knob ready_readout=false,true --bundle REPRESENTATIVE --wait-budget 5
python agent_selfcheck.py --attach SWEEP_ID
```

The sweep includes a submission baseline, submits ordinary scalability requests together, and returns
compact rankings of Spike timing-model estimates with comparable-case counts, slower cases and
numerical failures. Shared CPU leases and existing operator caps determine concurrency. Author work
keeps its scheduling priority over automatic background experiments.

`--knob` values are JSON scalars declared in `submission/TUNING_KNOBS.json`. Each named value is an
independent candidate; the broker changes only its private copy. Repeat `--knob` for another setting.
`--bundle` selects the representative and siblings recorded in `qa/idle_tuning.json`, instead of
explicit `--capsules`. Full comparisons and response references remain in the printed `all` file.

Compare only matching case scopes, model identities and epochs. Each sweep result retains both
executed source digests and ELF identities. Numerical failures never enter cycle rankings. A geometric
mean ratio summarizes comparable passing cases; inspect individual slower cases before adopting it.
Sweep results are advisory and confer no certification or promotion.

Automatic experiments run beside missing baseline timing. Their results survive source edits in
`qa/idle_tuning.json`, with exact candidate/baseline/case/model identities in `comparison_history`.
`baseline_status` means `current`, `older_source` (source moved but no changed emission observed), or
`superseded` (new emission or model). Superseded and failing comparisons stay available but do not
enter the applicable ranking. Carry a winning change into submission/ and confirm with the ordinary
correctness and certifying checks.

## Idle construction and baselines

While cores are free, the harness times every `variants/NAME/`
that differs from `submission/`, and every single-knob change declared in
`submission/TUNING_KNOBS.json`, on the cases with the most headroom. A variant that imports a
catalogue module is labelled with that mechanism. A knob carrying `mechanism` goes first when it
matches the brief. The results go to `qa/idle_tuning.json` and `performance_brief.idle_tuning`
(`delta_pct`, `numerics`, `how_to_apply`). Results survive edits with baseline labels (`current`,
`older_source`, `superseded`) and immutable `comparison_history`. Baselines and candidates run
beside each other. These are model estimates; confirm with the same-case self-check comparison.
One complete passing performance baseline is enough to begin; missing baseline cases continue
alongside bounded parallel candidate/case requests through the ordinary shared broker. The
`idle_tuning.progress` fields show ready baselines, declared candidates, outstanding candidate/case
checks and requests in flight. Missing adapter versus missing declared alternatives is an explicit
construction task. If your SAME emitter already reads `LIBRARY_PASS_SELECTION.json`, its
`mechanism` field can be the settings knob naming your finite registered alternatives; preserve
the incumbent and expose only actions whose source/effect prerequisites your adapter checks.
The coordinator neither imports your backend nor supplies a target encoder or legality law.
