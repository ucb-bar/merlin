# Reported Merlin results

[`registry.json`](registry.json) connects seven result families to their original
run IDs, saved input files, SHA-256 hashes, statistical rules and producer commands.
[`supplemental-archive.json`](supplemental-archive.json) inventories the additional
small-study records that are not in the original frozen-kernel or figure archives.
Archive files are distributed with the artifact package, not as source code.

| Recipe | Saved-data calculation |
| --- | --- |
| `frozen-performance` | Per-case cycles and useful peak utilization, construction-to-refinement ratios, main-workload and shape-sweep summaries |
| `single-source-cca` | Family geometric means relative to seed, Raw and earlier CCA-off arms |
| `scaffolding` | E1-E3 sixty-minute pass counts and uncached tokens, four-hour failures and tokens, self-check median/IQR |
| `checked-scheduling` | Time-to-improvement ratios for all five plotted scheduling studies |
| `convolution-library` | Library/seed and library/oracle contrasts for four shapes across four image layouts |
| `controlled-cca` | p1r/p2r sample medians and contrasts over all four boots |
| `fsm-controls` | Current automatic-library comparison and manually selected tiles/grouped-head controls, retaining both orders |

Run a recipe from the source checkout after restoring the artifact data:

```sh
python3 build_tools/scripts/recompute_paper_results.py \
  --registry experiments/reference-data/paper-studies/registry.json \
  --artifact-root "$ARTIFACT_DATA" --recipe scaffolding \
  --output "$MERLIN_OUT_ROOT/artifacts/paper-recomputation/scaffolding.json"
```

Use Python 3.12 or newer. Set `ARTIFACT_DATA` to the restored `data/` directory
and `MERLIN_OUT_ROOT` to a separate generated-output root. Omit `--recipe` to run
all seven calculations. The program verifies each consumed input hash and size,
rejects different medians and mismatched same-image controls, and refuses to
overwrite an existing receipt. It never runs the original producer scripts,
compilers, agents, simulators or FPGA jobs. Output must remain outside the
immutable input tree.

The [reproduction guide](../../../docs/guides/reproducing_paper_results.md)
describes the data layout, formulas and the remaining execution requirements.
