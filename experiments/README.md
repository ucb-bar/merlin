# Experiments

Inspect the authored catalog without launching an agent:

```sh
merlin experiment list
```

The catalog selects [target workflows](../examples/compiler-generation/README.md)
and the baseline definition. Each workflow has test generation and backend
generation; optimization belongs to backend generation. See the
[experiment guide](../docs/guides/12-experiments.md) for required inputs and budgets.

| Material | Home |
| --- | --- |
| Stable experiment IDs | [`catalog.yaml`](catalog.yaml) |
| Generic authored definitions | `definitions/` |
| Shared authored templates | `templates/` |
| Authored target/workload inputs | [`examples/targets/`](../examples/targets/README.md) and [`examples/workloads/`](../examples/workloads/README.md) |
| New runs, captures, corpora, bundles and reports | Configured `out/runs/`, `out/build/` and `out/artifacts/` |

Definitions resolve paths relative to their own files. A new backend experiment
requires an explicitly selected, reviewed test-generation release and its source
inputs. Generated corpora, input bundles, compiler snapshots and dated reports
are not committed defaults. Keep the original model, calibration and independent
reference selections with the run that produced them.

Previous run payloads and study snapshots are preserved locally under ignored
`out/artifacts/retained/`. They are available for explicit historical replay with
their original source snapshot; current workflows never discover them as fresh
inputs. Some historical captures require unavailable upstream assets and tools.

Private holdouts, golden answers, credentials and machine assets stay out of public
examples. FireSim provisioning and launch remain external.
