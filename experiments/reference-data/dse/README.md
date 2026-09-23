# DSE reference evidence

`case_study/` contains the 265 committed analysis files previously located at
`out/artifacts/dse-guidance/case_study/`. The move preserves every payload byte.
`MIGRATION.json` records each relative filename and SHA-256 digest. This inventory
establishes migration fidelity, not the scientific validity of historical claims.
Historical paths and statements inside the snapshot remain unchanged intentionally.

Read this input through `merlin.dse_guidance.reference_data.case_study_dir()`.
The resolver prefers this checkout-owned snapshot, then the old checkout path for
historical workspaces. It does not follow `MERLIN_OUT_ROOT`. Explicit
`--case-study-dir` or `MERLIN_DSE_REFERENCE_DIR` selects a different input; query
mode also retains its legacy `--out` input alias. This reference is not bundled
into the core wheel: installed users must supply the input directory.

```sh
merlin-dse-guidance --query summary
merlin-dse-guidance --insight-mining --out out/artifacts/dse-guidance/analysis
merlin-dse-guidance --case-study --out out/artifacts/dse-guidance/case_study
merlin-dse-guidance --query summary --case-study-dir out/artifacts/dse-guidance/case_study
```

Generation remains separate from reference consumption. New outputs belong under
the configured output root. Do not use this directory as `--out`, regenerate it
in tests, or mechanically rewrite the historical payload to reflect new paths.
