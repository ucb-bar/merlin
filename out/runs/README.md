# Experiment runs

Runs are grouped as `<target>/<suite>/<run-id>/`. Start with a run's recorded
definition, environment and native-engine paths rather than guessing its purpose
from a timestamp. AET records accounting; phase engines own grades and checkpoints.
An orchestration exit code alone is not a compiler correctness or speedup result.

Use `merlin experiment status /absolute/path/to/run` for catalog-managed runs.
Use `merlin storage experiments` to inspect storage across experiment groups.
Do not rename frozen runs or edit their receipts to reorganize output.

New run locations come from the run/storage APIs, not hand-built path strings.
Historical native runs can have different internal layouts; their recorded paths
remain authoritative. See the [experiment guide](../../experiments/README.md)
and [storage guide](../../docs/guides/storage.md).
