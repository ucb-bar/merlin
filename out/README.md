# Generated output

Everything under `out/` is generated and ignored by Git, except these four READMEs.
Source, experiment definitions and curated inputs never live here.

| Directory | Holds |
| --- | --- |
| [runs/](runs/README.md) | Experiment runs: inputs, logs, grades, checkpoints, accounting |
| [artifacts/](artifacts/README.md) | Named products: reports, measurements, packages, figures, caches |
| [build/](build/README.md) | Build trees, tool environments and agent workspaces |

Tools find this root through `merlin.common.paths` (`out_dir()`, `runs_dir()`,
`artifacts_dir()`, `build_dir()`); `MERLIN_OUT_ROOT` moves it outside the checkout.
Do not add a fourth directory.

Inspect before changing anything:

```sh
merlin storage layout        # where things are, and anything undeclared
merlin storage experiments   # storage per experiment group
merlin storage report        # what it costs, including duplicated bytes
```

Reports and manifests quote paths under `out/`. Do not rename directories by hand
or delete by age; `merlin storage organize` and `merlin storage prune` default to dry
runs and keep citations working.
