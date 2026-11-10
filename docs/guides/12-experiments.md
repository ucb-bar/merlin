---
title: "12. Run and inspect experiments"
kind: guide
status: current
owner: experiments
last_verified: 2026-11-09
related: [09-verification, 10-compiler-generation, 11-optimization]
code_refs: [experiments/catalog.yaml, src/merlin/experiments/definition.py, src/merlin/experiments/cli.py, src/merlin/experiments/runner.py, src/merlin/experiments/adapters.py, src/merlin/experiments/resources/experiment.schema.json, examples/gemmini/experiment.yaml]
---

# 12. Run and inspect experiments

[Previous](11-optimization.md) | [Guide index](../README.md)

Use `merlin experiment` to select a study, inspect its inputs, launch its configured
phase and find the results. Start with discovery:

```sh
merlin experiment list
merlin experiment inspect gemmini-functional --phase 1
merlin experiment preflight gemmini-functional --phase 1
```

`list` shows catalog entries. `inspect` resolves the definition into a plan.
`preflight` checks configuration and available inputs. None starts an authoring
agent or establishes that the compiler, hardware or experiment has passed.

The experiment commands use the `experiments` extra. Follow the repository setup
instructions before using an execution worker. Actual runs also need their
selected toolchains, runtime/oracles, source and corpus inputs, agent credentials,
budgets and the native sandbox facilities required by the engine.

## Catalog, definition and runner

[`experiments/catalog.yaml`](../../experiments/catalog.yaml) maps stable IDs to
definition files. Most target entries currently point to
`examples/<target>/experiment.yaml`; generic templates live in
`experiments/definitions/`. A definition can also be selected by its explicit path.

Definitions declare the target, input paths, phase adapters and typed settings.
They do not embed shell commands or choose arbitrary Python modules to execute.
The adapter registry validates the selected configuration and constructs the
actual command.

Paths inside a definition are relative to that file. For example, the Gemmini
definition selects `target/descriptor.yaml` and `phase0/recipe.yaml` beside itself,
while shared templates are referenced under `experiments/templates/`.
The descriptor has its own resource/grant conventions; do not assume that every
path in every selected YAML file has the definition's base directory.

| Owner under `src/merlin/experiments/` | Responsibility |
| --- | --- |
| `definition.py` | Load the catalog and validate definitions against the schema. |
| `adapters.py` | Resolve known phase adapters and their settings. |
| `runner.py` | Build the plan, check inputs, execute, resume and record attempts. |
| `cli.py` | Expose those operations through `merlin experiment`. |

Python callers use these same owners:

```python
import json
from merlin.experiments.definition import catalog, load_spec
from merlin.experiments.runner import resolve_plan

definition = load_spec(catalog()["gemmini-functional"])
plan = resolve_plan(definition, phase="1")
print(json.dumps(plan, indent=2))
```

This resolves a plan; it does not call `runner.run`. Outside a checkout, supply an
explicit catalog path or definition and the selected external inputs.

## Choose one phase and its handoff

| Phase | Job | Input for the next phase |
| --- | --- | --- |
| 0: hardware-guided test generation | Derive requirements, programs and reference data from selected inputs. | An explicitly reviewed and sealed corpus. |
| 1: functional compiler generation | Author a target compiler and evaluate it independently. | The frozen compiler and its required functional evidence. |
| 2: performance optimization | Improve that compiler using the selected workloads and measurement policy. | Compiler revisions and measurements tied to those exact inputs. |

Use [compiler generation](10-compiler-generation.md) for the Phase 0/1 flow and
[optimization](11-optimization.md) for Phase 2. A definition need not implement all
three phases. The catalog's template entries also require operator selections;
their presence is not a ready-to-run target.

Run Phase 0 and Phase 1 separately. Review and seal the generated corpus before
selecting it for functional grading. The current runner refuses a combined route
that would bypass that review or silently grade a retained older corpus.
Phase 2 must select the frozen functional compiler and matching evidence.

## Launch only after selecting the inputs

An operator with a reviewed corpus and bundle can use:

```sh
merlin experiment run gemmini-functional --phase 1 \
  --corpus-seal /absolute/reviewed/corpus-seal.json \
  --bundle-manifest /absolute/reviewed/input_bundle_manifest.yaml
```

These paths are placeholders for real selected inputs, not files supplied by
this guide. Use `preflight` with the same flags first. `run --help` lists the
supported driver/model/provider and Phase 0 input overrides. The run records
selected values; resume must retain the original identities and constraints.

`merlin experiment levels` explains Phase 1 information levels EL1 through EL4.
They describe what the author receives, not whether a compiler is correct or fast.
Old bundle IDs remain in historical records so those experiments can be replayed.

## Find a run and read its records

```sh
merlin experiment runs --target gemmini
merlin experiment status /absolute/path/to/run
merlin experiment lineage /absolute/path/to/run
merlin experiment resume /absolute/path/to/run
```

The first three commands read stored state. `resume` continues an existing run.
`status` describes process attempts; `lineage` connects the frozen definition,
selected inputs and phase outputs. Read the native engine's numerical and
measurement evidence for the scientific result. A process exit code or dashboard
alone cannot establish that result; see [verification](09-verification.md).

## Know which files are inputs and which are outputs

The current repository still contains both live study inputs and retained material:

| Material | Current home |
| --- | --- |
| Catalog and generic definitions | `experiments/catalog.yaml`, `experiments/definitions/` |
| Target workflows and authored public recipes | `examples/<target>/` |
| Shared experiment templates | `experiments/templates/` |
| Retained public benchmark capsules | `experiments/corpora/`; its manifest distinguishes generated and hand-authored origins. |
| Study definitions, harness selections and reproduction inputs | `experiments/studies/` and `experiments/benchmarks/` |
| Reusable reference inputs and historical analysis records | `experiments/reference-data/`; inspect each record's purpose and status. |
| New run state, build products and derived artifacts | Configured `out/runs/`, `out/build/` and `out/artifacts/` |

Retained generated capsules can be versioned benchmark inputs for an existing
study. New generation writes to its run; it must not overwrite those inputs.
Likewise, a dated inactive candidate report under `reference-data/` is historical
analysis, not a currently selected hardware model. Its dates, status and original
hashes remain meaningful even when its location is later reorganized.

Keep private holdouts, golden answers, credentials and machine assets out of public
examples. Select their operator-owned locations explicitly. FireSim provisioning
and launch remain external: Merlin can prepare guest files and collect outputs.

For a specific study, start with its catalog entry and target workflow map.
For old paper results, use the retained input/archive manifests under
`experiments/reference-data/paper-artifact/` and `paper-studies/`, with the original
source revision and tools. Current commands and new results do not rewrite an
older experiment's evidence.

[Back to optimization](11-optimization.md) | [Compiler generation](10-compiler-generation.md) |
[Guide index](../README.md)
