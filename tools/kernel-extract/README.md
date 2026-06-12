# kernel-extract — kernel feature/policy extractor

Thin CLI entrypoint. Aggregates kernel indexes into abstraction candidates, policy rules, a
feature table, and a markdown report.

## What it does

Loads one or more `kernel-index` outputs, aggregates motifs across the corpus, applies the
promotion gate (≥2 sources OR ≥`--min-kernels`), and emits schema-valid
`abstraction_candidates.yaml` + `policy_rules.yaml`, a `kernel_features.jsonl` table
(optional `.parquet`), and `kernel_mining_report.md`.

## Backing module

`merlin.kernels.cli_extract:main` (installed as the `kernel-extract` console script).

## Usage

```bash
kernel-extract \
  --inputs   "output/kernels/*_index.json" \
  --out      output/kernels/abstraction_candidates.yaml \
  --policies output/kernels/policy_rules.yaml \
  --report   output/kernels/kernel_mining_report.md \
  --min-kernels 10            # single-source promotion gate
  # --plots                   # 7 evaluation PNGs under <report dir>/plots (needs .[kernels-plots])
  # --json                    # machine-readable summary JSON on stdout (human text on stderr)
  # --strict                  # exit 2 if a consistency invariant is violated (CI gate)
  # --parquet                 # also write kernel_features.parquet (needs .[kernels-parquet])
  # --llm-summary             # advisory one-shot summary over the aggregated motif table
```

## Notes

Aggregation is deterministic; kernels vendored verbatim across sources are deduplicated by
content hash before counting. Besides abstractions/policies it emits interface candidates (L5),
runtime candidates (L7), dialect requirements (L6, input to TargetGen) and LLVM requirements
(L8, always `requires_llvm_fork: false` until Stages F/G pass). The report includes Stage-D
validation (benchmarks + capacity sweep + shape-regime matrix), an actionability scorecard,
and consistency invariants. `--llm-summary` is the only optional LLM touch here and runs once
over the small motif table (never per kernel; needs `ANTHROPIC_API_KEY`). See
`tools/kernel-audit` for marker-precision auditing. Artifacts are written under `output/`
(gitignored).
