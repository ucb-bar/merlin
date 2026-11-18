---
title: 12. Run backend-generation experiments
kind: guide
status: current
owner: experiments
last_verified: 2026-11-09
related: [10-compiler-generation, 11-optimization, ../reference/repo_structure]
code_refs: [experiments/catalog.yaml, src/merlin/experiments/definition.py, src/merlin/experiments/cli.py, src/merlin/experiments/options.py, src/merlin/experiments/runs/plans.py, src/merlin/experiments/runs/plan_sources.py, src/merlin/experiments/runs/plan_execution.py, src/merlin/experiments/runs/adapters.py, src/merlin/experiments/resources/experiment.schema.json, src/merlin/development/backend/optimization/whole_model_measured/cli.py, src/merlin/development/backend/optimization/whole_model_measured/scoring/cell_runs.py, examples/compiler-generation/gemmini/experiment.yaml]
---

# 12. Run backend-generation experiments

[Previous](11-optimization.md) | [Guide index](../README.md)

Use `merlin experiment` for the selected target definition, inputs, budgets and
run records. Its two named generation workflows are **test-generation** and
**backend-generation**. Backend generation develops correctness and performance
in one continuous native session.

## Find a definition

```sh
merlin experiment list
merlin experiment levels
merlin experiment run --help
```

`list` reads metadata without launching an agent. `levels` explains the EL1-EL4
tool-access treatments; they describe what an author receives, not a result.
`inspect` resolves an executable plan and therefore requires the selected inputs.

The [catalog](../../experiments/catalog.yaml) maps stable IDs to authored YAML.
Target definitions live under `examples/compiler-generation/<target>/`.
The small root `experiments/` directory contains only the catalog, baseline
configuration and shared authored template. Generated study trees are not source.

Definitions use `schema_version: 2` and named `stages`. They select known adapters
and typed settings, not arbitrary shell commands or Python modules. Paths are
relative to the definition file. Descriptors and source grants keep their own
explicit path conventions.

## Configure and check a backend run

Copy a target template, supply its operator inputs, and set `kind: experiment`
when it is ready for execution. First generate and review a test corpus, then
select its release seal and generated input bundle. Run the two workflows
separately: orchestration preflight refuses an unreviewed test-to-backend handoff.
No retained old corpus or checked-in bundle is selected implicitly.

For a configured definition, preflight the exact selections:

```sh
merlin experiment preflight path/to/experiment.yaml \
  --workflow backend-generation \
  --corpus-seal /absolute/release/private/seal.json \
  --bundle-manifest /absolute/release/input_bundle_manifest.yaml \
  --backend-joint-campaign /absolute/policy.json \
  --backend-oracle-timing /absolute/timing.json
```

Use the actual release's file paths. These are placeholders, not supplied data.
The joint policy and timing input must meet the native engine's existing checks.
Fresh backend selection and a continuous schedule are explicit in the template.
`preflight` checks configuration; it does not establish hardware qualification.

Use `run` with the same arguments to execute. An actual run also needs the
selected toolchains, independent references, runtime, agent credentials, budgets
and Linux sandbox facilities. FireSim setup and launch remain external.

## Use the same boundary from Python

```python
from pathlib import Path
from merlin.experiments.definition import load_spec
from merlin.experiments.runs.plans import preflight, resolve_plan, run

spec = load_spec("path/to/experiment.yaml")
plan = resolve_plan(
    spec,
    workflow="backend-generation",
    corpus_seal=Path("/absolute/release/private/seal.json"),
    bundle_manifest=Path("/absolute/release/input_bundle_manifest.yaml"),
    backend_joint_campaign=Path("/absolute/policy.json"),
    backend_oracle_timing=Path("/absolute/timing.json"),
)
# Check configuration and selected input identities without starting engines:
readiness = preflight(plan)
# Execute only after reviewing readiness and supplying the native prerequisites:
# run(plan, workflow="backend-generation")
```

Resolution selects commands, source membership and input paths. `preflight`
checks their current identities; `run` repeats those checks and writes the frozen
`resolved-plan.json` and orchestration record before execution. It does not
silently resolve another workflow. Legacy numeric Python selectors cannot be
combined with `workflow`; named YAML definitions use `--workflow` on the CLI.

## Read results and resume

```sh
merlin experiment runs --target TARGET
merlin experiment status /absolute/run
merlin experiment lineage /absolute/run
merlin experiment resume /absolute/run
```

New runs use named suites under `out/runs/<target>/test-generation/` or
`backend-generation/`. Historical records remain discoverable at their original
locations. Resume retains the frozen inputs, source identities and policy.

Generated captures, corpora, bundles, compiler snapshots and reports belong under
`out/`. Keep authored workload sources, policies and configuration in source.
The process status is different from numerical correctness, device execution and
measurement validity; read those reports separately as described in
[verification](09-verification.md).

The optional `whole_model_measured` workflow remains separate from the two
generation stages. It optimizes a selected compiler against a measured
whole-model objective on declared machines; cell runs use selected model groups.
Its commands and configuration are described in the
[optimization example](../../examples/compiler-generation/gemmini/backend-generation/optimization/README.md):

```sh
merlin experiment measured --help
merlin experiment cell --help
```

Capsule feedback and final-model checks in backend generation do not provide the
same whole-model measured objective or campaign lifecycle.

## Open the implementation

| Responsibility | Source under `src/merlin/experiments/` |
| --- | --- |
| Definitions and named selection | `definition.py` |
| Closed adapter argument construction | `runs/adapters.py` |
| Plan freezing and preflight | `runs/plans.py` |
| Source selection and corpus binding | `runs/plan_sources.py` |
| Frozen process execution | `runs/plan_execution.py` |
| Public CLI | `cli.py` |
| Dashboard and terminal entrypoints | `tracking/__init__.py` |
| Record readers, page rendering and live terminal display | `tracking/readers/`, `tracking/pages/`, `tracking/terminal/` |

The actual test generator and authoring engine live under `development/`;
experiment orchestration does not duplicate their implementations.

Operator launch configuration is in
[`backend/session/launch_config.py`](../../src/merlin/development/backend/session/launch_config.py).
Its records, variable resolution and rendering are separate from
[`launch_drift.py`](../../src/merlin/development/backend/session/launch_drift.py),
which compares prepared inputs with the reviewed configuration. Reading an
optional live process checks its argv, environment and affinity; it does not
start a run or establish numerical correctness.
