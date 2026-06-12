# Role: evidence_extractor

## Purpose
Extract evidence records from selected source files and write `evidence_graph.jsonl`
under the run's `contracts/` directory. Used by v3_evidence_graph and v6_full methods.

## Allowed inputs (read-only)
- `datasets/gemmini/source_snapshot/`
- `datasets/gemmini/selected_docs/`
- `datasets/gemmini/selected_rtl/`
- `datasets/gemmini/selected_kernels/`
- `datasets/gemmini/selected_traces/`

## Allowed outputs (write)
- `<run_dir>/contracts/evidence_graph.jsonl`

## Forbidden modifications
- `datasets/` source files
- Any file outside `<run_dir>/contracts/`

## Validation command
```bash
python -c "
import json, pathlib
lines = pathlib.Path('<run_dir>/contracts/evidence_graph.jsonl').read_text().splitlines()
records = [json.loads(l) for l in lines if l.strip()]
assert all('concept' in r and 'source' in r for r in records), 'missing required fields'
print(f'{len(records)} evidence records OK')
"
```

## Success criteria
- `evidence_graph.jsonl` contains ≥ 10 records covering core Gemmini concepts
- Every record has `concept`, `source`, and `quote` fields
- No records reference files outside the selected_* directories
