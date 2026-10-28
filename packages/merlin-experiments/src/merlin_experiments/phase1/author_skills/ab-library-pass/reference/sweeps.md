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
