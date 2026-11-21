# src/merlin/experiments

`definition.py` reads experiment definitions and selects test generation,
backend generation, and the supported whole-model measured workflow.
`cli.py` exposes those workflows and read-only run navigation.

`runs/adapters.py` declares the supported options and exact installed/native
commands. `runs/plans.py` resolves and freezes inputs and performs preflight.
`runs/plan_sources.py` binds source and corpus membership.
`runs/plan_execution.py` launches the frozen command, records process attempts,
and retains ownership when descendants might remain. A successful process exit
is an execution observation; numerical and performance verdicts belong to the
engine's independent evidence.

`runs/execution.py` owns shared environment selection. Frozen test generation
uses its selected-only environment. Installed children and inspection exclude
an ambient board catalog until a declared frozen input owns it. Preserve this
policy when reorganizing launch code.

`runs/catalogs.py` provides stored-run discovery, lineage, and compiler-history
export. Export consumes existing scoped evidence; it does not synthesize a
backend, import a candidate, install a service, or publish remotely.

The measured-claims and model-portfolio execution adapters are retired. Do not
restore their coordinators, deployment commands, checkpoint-segment execution,
or mode definitions. Historical records remain inspectable. The paired tracking
reader owns only immutable record-format constants and reads stored observations;
it must not import a retired execution engine or infer scientific qualification.

Inspection reaches target-specific compiler and functional-model hooks through
the selected backend. Missing hooks remain unavailable; no target name selects a
fallback implementation.

Source snapshots retain original ownership and full membership. Old seals can
be inspected but cannot be relabeled as a new verifier's execution evidence.
Managed execution services and console panels are loaded dynamically; absence
from a static import graph is not evidence that they are unused.
