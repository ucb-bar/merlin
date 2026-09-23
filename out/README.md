# Generated output

This directory contains outputs, not Merlin's source or experiment definitions.
Start experiments from [the catalog](../experiments/README.md).

| Directory | Use it for |
| --- | --- |
| [runs/](runs/README.md) | Experiment execution records: inputs, logs, grades, checkpoints and accounting |
| [artifacts/](artifacts/README.md) | Named products: reports, measurements, published packages and paper assets |
| [build/](build/README.md) | Build trees, tool environments, agent workspaces and disposable diagnostics |

Tools resolve this root through `merlin.common.paths`; `MERLIN_OUT_ROOT` can place
it outside the checkout. Do not create another top-level output directory.

To understand existing output without changing it:

```sh
merlin storage layout
merlin storage experiments
```

The [storage contract](../merlin/contract/storage.yaml) owns product categories;
the [storage guide](../docs/guides/storage.md) explains inspection and cleanup.
Reports and frozen manifests may cite old paths. Do not manually rename those
directories, infer liveness from age, or delete a run because its name looks stale.
Organization and retention commands default to dry runs; inspect their plan first.
