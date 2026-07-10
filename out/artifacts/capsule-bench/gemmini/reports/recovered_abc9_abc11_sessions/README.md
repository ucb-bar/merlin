# Recovered abc9 / abc11 agent sessions (partial)

## What this is

Salvaged **nested-agent session transcripts** for the abc9 and abc11 capsule-bench A/B arms,
copied here on 2026-07-02 from `~/.claude/projects/*--qa-ws-<arm>-workspace/` (the per-run bwrap
workspaces spawned nested Claude sessions whose transcripts live in the experimenter's home
`~/.claude`, outside the repo).

| Arm folder | Original run | Content |
|---|---|---|
| `rb_abc9`         | `raw_baseline/rb_abc9`          | 24 files, 256M (incl. a nested workspace snapshot — may hold the agent's built submission) |
| `merlin_abc9`     | `merlin_assisted/merlin_abc9`   | agent transcripts (5.8M) |
| `merlincirct_abc9`| merlin+CIRCT arm                | agent transcripts (1.9M) |
| `rb_abc11`        | `raw_baseline/rb_abc11`         | agent transcripts (6.6M) |
| `rbinfra_abc11`   | `cpp_merlininfra/rbinfra_abc11` | agent transcripts (9.5M) |

58 `.jsonl` transcripts total. Timestamps confirm the abc9 era (2026-06-21).

## Why it's only partial — provenance

The authoritative **run outputs** for these arms lived under
`experiments/gemmini_capsule_bench_v0/runs/` (`perf_results.json`, grades, instruction traces,
oracle timings, aggregated results). That tree was **untracked** and was deleted during the
2026-07-01 repo reorg (`rm -rf experiments/gemmini_capsule_bench_v0/runs`, the #20 "purge stale
experiment runs" step) — the plan treated `experiments/*/runs` as regenerable leftovers, but abc9/abc11
were live results. Because they were never git-tracked, they are not recoverable from history.

Searched and **not found anywhere**: git history, filesystem snapshots (none on this host),
open-file-descriptor recovery, tarballs, other worktrees, `/scratch{,2}/agustin`. The only other
surviving capsule-bench runs are the three abc4 archives under `../abc4_archive/`.

**Lost (unrecoverable as authoritative artifacts):** the computed run outputs above.
**Survives here:** the full agent trajectories — from which final submissions and the grades the
agent observed may be partially reconstructed.

## Preservation status

- Untracked (like `../abc4_archive/`), so `artifacts/` cleanups can still reach it — do NOT purge.
- The originals remain in `~/.claude/projects/` as a second copy.
- To make this durable across clones it would need to be git-tracked; held off pending owner
  decision (~280M, and agent transcripts may contain benchmark-sensitive content).

## Lesson wired back into the process

`experiments/**/runs/` is now gitignored and new runs route to `runs/<target>/<suite>/`. The purge
that lost these should have archived non-regenerable run outputs first; future run-tree cleanups must
snapshot `perf_results.json`/grades/aggregates before deleting.
