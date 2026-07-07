# AGENT.md — merlin/prompts

## Purpose
Versioned **agent-instruction templates** (prose preambles) fed to the kernel-mining / RVV-tuning
agentic loops. Curated inputs (cannot be generated).

## What lives here
- `rvv_mining_v1.md`, `rvv_tuning_v1.md` — RVV-specific (their only consumers are RVV modules).

## Used by (the LIBRARY, not experiments)
- `merlin.kernels.agent_mine` → `rvv_mining_v{V}.md`
- `merlin.rvvgen.tuning_agent` → `rvv_tuning_v{V}.md`
Both resolve the dir via `merlin.common.paths.merlin_dir()/"prompts"` (robust — not `parents[N]`).

## Invariants
Prose templates only; version with a `_v{N}` suffix (never edit a shipped version in place). If a new
prompt is target-family-specific, prefix it accordingly (like `rvv_`).
