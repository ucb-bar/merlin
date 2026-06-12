# Kernel abstraction mining (Workstream 2)

Extract optimization **decisions** (not constants) from existing kernels and turn the ones that
recur across many kernels into compiler-consumable policies and abstraction candidates.

```
XNNPACK RVV / OpenBLAS RVV / Autocomp (Gemmini) / Exo (compiled C + schedule .py)
                          / Triton + triton-cpu (@triton.jit)
  -> kernel_record        (per kernel: L1 op_sequence, L2 per-tensor memory roles +
                           measured reuse, L3/L4 features, shape regime, motifs, evidence)
  -> content-hash dedup   (verbatim copies across sources never inflate the signal)
  -> motif aggregation    (counts per motif x source)
  -> abstraction_candidate + interface_candidate (L5, 4 lowering variants)
   + policy_rule (L4) + runtime_candidate (L7)
   + dialect_requirement (L6, feeds TargetGen) + llvm_requirement (L8, always "no fork yet")
  -> Stage-D validation    (benchmarks + capacity sweep + shape-regime matrix)
  -> evaluation layer      (report + plots + invariants + kernel-audit)
```

## Design principles

- **Deterministic, cheap, scalable.** Per-kernel extraction is regex / filename / C-signature /
  Exo-lowered-C — **zero LLM calls** over thousands of kernels. Reproducible and diff-friendly.
- **Decisions, not constants.** Record "packed RHS reused", "accumulator lives across the
  epilogue", "vector-length-agnostic loop" — never `tile=64` or `LMUL=4`.
- **Insight from cross-source frequency.** A motif seen in XNNPACK *and* OpenBLAS *and* Gemmini
  *and* Triton is a real abstraction candidate; a one-source fluke is not. Content-hash dedup
  keeps this honest (triton-cpu vendors the triton tutorials verbatim).
- **Regimes, not shapes.** Generalization is stated over `capacity_fit / tail_heavy /
  memory_bound / skinny` regimes, never exact `(M,N,K)` triples.

## Modules

`merlin/python/merlin/kernels/`:
- `ingest/` — per-source adapters → `NormalizedKernel` (`types.py`, incl. `content_hash`).
  XNNPACK parses the ukernel symbol; OpenBLAS parses `kernel/riscv64` filenames (vector
  kernels only, scalar fallbacks skipped); Autocomp parses the `void test(...)` signature;
  Exo **compiles specs to C** and also mines schedule `.py`; Triton extracts one record per
  `@triton.jit` function (default subtrees `python/tutorials` + `python/triton_kernels`),
  with `source="triton_cpu"` for the CPU fork.
- `markers.py` — the `(ISA-family, motif) → regex` table; the heart of extraction.
- `features/` — pure `extract_*` functions, incl. `shape_regime.py` (working-set bytes,
  arithmetic intensity, regime labels) and `roles.py` (L2 memory roles, **measured** reuse).
- `classify.py` / `evidence.py` — features → canonical motif set + evidence ids/markers.
- `emit/` — schema-shaped `kernel_record` / `abstraction_candidate` / `policy_rule` /
  `interface_candidate` / `runtime_candidate` / `dialect_requirement` / `llvm_requirement`.
- `policy.py` — dedup + aggregation + the promotion ladder (≥2 sources OR ≥N kernels);
  promoted interfaces also emit the L6/L8 requirements. `validate.py` — Stage-D: benchmark
  verdicts, capacity sweep, and the symbolic **shape-regime matrix** (reuse × K × tail grid
  + mutable-RHS / no-reuse negative controls).
- **Evaluation layer:** `report.py` (report incl. actionability scorecard), `plots.py`
  (7 evaluation PNGs), `invariants.py` (consistency checks + surprise list), `audit.py`
  (`kernel-audit` marker-precision spot-check).

Sources are external repos passed by path / `MERLIN_<SOURCE>_REPO` env var, never vendored.

## Tools

```bash
kernel-index   --source {xnnpack|autocomp|exo|openblas|triton|triton_cpu} \
               --repo <path> [--target T] [--json] --out <index.json>
kernel-extract --inputs "output/kernels/*_index.json" \
               --out abstraction_candidates.yaml --policies policy_rules.yaml \
               --report kernel_mining_report.md \
               [--plots] [--json] [--strict] [--min-kernels 10] [--parquet] [--llm-summary]
kernel-audit   --inputs "output/kernels/*_index.json" [--motif M] [--n 8] [--seed 0] \
               [--llm-judge] [--json] --out audit_samples.md
```

Installed via `[project.scripts]`. Extras: `.[kernels-exo]` (Exo ingest), `.[kernels-parquet]`
(columnar table), `.[kernels-plots]` (matplotlib). Artifacts land in `output/kernels/`
(gitignored). All CLIs support `--json` (machine-readable summary on stdout, human text on
stderr) so agents and CI can compose them.

## Evaluating the results

Each artifact answers a specific "is this actionable / does it make sense?" question:

| Question | Artifact |
|---|---|
| Is the cross-source signal real (not corpus-size bias)? | `plots/motif_source_heatmap.png` (per-source *fractions*) |
| How broadly attested is each motif? | `plots/motif_prevalence.png` + motif table |
| How hard does the ladder filter? | `plots/promotion_funnel.png` |
| Is the measured L2 reuse metric sane? | `plots/reuse_distribution.png` |
| Is the L7 runtime candidate justified? | `plots/dispatch_scatter.png` |
| Which decisions travel together (composites)? | `plots/motif_cooccurrence.png` |
| Does any marker over-fire on the wrong ops? | `plots/motif_op_heatmap.png` + invariants surprise list |
| Do markers mean what they claim? | `kernel-audit` samples (real snippets, ±context) |
| Are the artifacts internally consistent? | report "Consistency invariants" section (`--strict` gates CI) |
| What should I do with each policy? | report "Actionability scorecard" (evidence breadth, Stage-D, regime sweep, downstream consumer, falsifier, next step) |

**LLM / agent escalation (optional, bounded):** set `ANTHROPIC_API_KEY` (+ optionally
`MERLIN_LLM_MODEL`) to enable `kernel-extract --llm-summary` (one call over the motif table)
and `kernel-audit --llm-judge` (one verdict per sampled snippet → marker-precision estimate).
For deeper triage, run headless Claude Code over the small artifacts — see
`tools/kernel-audit/README.md` for the recipe. Every artifact exists without any key;
deterministic outputs remain the source of truth.

## Promotion ladder

`Observation` (marker in one kernel) → `Motif` (marker→decision) → `Policy candidate`
(≥2 sources OR ≥`--min-kernels`) → `Validated` (fires on benchmark positives **and across the
regime grid where reuse ≥ 2**, silent on negative controls). Promoted interfaces emit L6
dialect requirements (`status: proposed`, input to TargetGen) and L8 LLVM requirements that
always say `requires_llvm_fork: false` until Stage F (target lowering) and Stage G
(exploitability) pass. Emitted `evidence` is always the real set of kernel ids that fired.

## Must not

Build a large classifier before a small validated corpus; claim automatic abstraction discovery;
hard-code XNNPACK-only assumptions into the generic kernel schema; treat the Autocomp `score` or
an unexecuted kernel as a correctness/perf signal (no kernel is run or timed); read any plot as
a speedup (they visualize evidence frequency only); modify LLVM from kernel evidence alone.
