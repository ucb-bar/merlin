# kernel-audit — marker-precision spot-check

Thin CLI entrypoint. Answers the question every mined motif must survive: **"do I believe
this?"** For each motif it samples kernels (stratified across sources, seed-deterministic),
re-fires the marker table on the original source, and writes the matched snippet with ±context
lines so a human or an agent can judge whether the code really evidences the claimed
optimization *decision*.

## Backing module

`merlin.kernels.audit:main` (installed as the `kernel-audit` console script).

## Usage

```bash
kernel-audit \
  --inputs "output/kernels/*_index.json" \
  --out    output/kernels/audit_samples.md \
  --n 8 --seed 0                # samples per motif, deterministic
  # --motif packed_rhs          # repeatable; default: every observed motif
  # --llm-judge                 # one bounded LLM verdict per sample (see below)
  # --json                      # machine-readable summary on stdout (human text on stderr)
```

## LLM escalation (optional, bounded)

Two levels, both strictly optional — every artifact exists without any API key:

1. **`--llm-judge`** — one Anthropic call per *sampled snippet* (≤ motifs × `--n` calls,
   never per kernel) asking "does this code evidence the claimed decision?" and recording
   `confirms / unclear / refutes` per sample plus a marker-precision estimate per motif.
   Setup:

   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...
   export MERLIN_LLM_MODEL=claude-opus-4-8   # optional; this is the default
   ```

   Without a key the audit runs identically and notes that verdicts were skipped.

2. **Agentic loop (Claude Code)** — the pipeline's artifacts (report, audit MD, invariants,
   features JSONL, `--json` summaries) are deliberately small and self-contained so a coding
   agent can consume *results* instead of re-reading thousands of kernels. Typical headless
   escalation after an extract run:

   ```bash
   claude -p "Read output/kernels/kernel_mining_report.md and \
   output/kernels/audit_samples.md. Triage every consistency-invariant violation and \
   surprise-list entry as marker-bug vs genuine insight; for marker bugs, propose concrete \
   regex fixes in merlin/python/merlin/kernels/markers.py."
   ```

   The same pattern works for deeper questions ("which promoted policy has the weakest
   evidence?") because all evidence ids and counts are in the artifacts.

## Notes

Sampling is deterministic for a given `--seed`; context re-reads use the `repo` recorded in
each index, so audits keep working as long as the corpus checkout exists. Output goes under
`output/` (gitignored).
