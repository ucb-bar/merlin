# Examples

Choose the kind of work you want to do:

| Start here | Purpose |
| --- | --- |
| [Tutorials](tutorials/README.md) | Learn contracts/facts, test generation, scheduling, transformations and CCA with small interactive inputs. |
| [Compilation](compilation/README.md) | Compile and run programs with an existing backend. Start with the scalar quickstart. |
| [Compiler generation](compiler-generation/README.md) | Select target inputs and run the test-generation and backend-authoring workflow. |
| [Target inputs](targets/README.md) | Inspect target-only declarations and retained provider dependencies. |
| [Workloads](workloads/README.md) | Reusable small model/function inputs for frontend iteration. |

The experiment catalog remains at [`experiments/catalog.yaml`](../experiments/catalog.yaml).
Target implementations are OOT packages; provider snapshots retained with a target
are dependencies, not an authored candidate or a successful experiment result.
Their `SOURCE.yaml` records pin the original bytes.

[`board-catalog.yaml`](board-catalog.yaml) is shared public board data. Select it
explicitly when a runtime needs it; an OOT target can provide its own catalog.
The scripts in `lib/` provide shared shell/preflight helpers for these examples.

Generated captures, capsules, binaries, weights and reports belong under the
configured `out/` root. Keep private grading inputs and credentials outside public
examples. FireSim setup and launch are external; examples can prepare guest files
and inspect returned output.
