# Experiment runs

One directory per run: `<target>/<suite>/<run-id>/`, with `run_record.json` as the
source of truth. Run ids are timestamp-first (`<UTC-TS>_<method>_seed<NNN>_<sha7>`).
Create runs only through `merlin.common.artifacts.start_run(..., target=...)`.

- `merlin experiment status /absolute/path/to/run` shows a catalog-managed run.
- `aet runs` (pointed at the `out/` root) lists runs.
- An orchestration exit code alone is not a correctness or speedup result; read the
  run's grades and receipts.

Never rename a frozen run or edit its receipts. Historical runs may use other
internal layouts; their recorded paths stay authoritative.
