# Working-tree & branch rules for this checkout

## ⚠️ This is a SHARED working directory — one HEAD for ALL Claude Code sessions
Every Claude Code session running in `/scratch/agustin/projects/oscar-merlin` shares the **same** `.git`
and the **same** checked-out HEAD. There is no per-session branch isolation. Therefore:

- **DO NOT switch branches** (`git checkout <branch>` / `git switch <branch>`) to a *different* branch.
  Switching flips HEAD for **every** concurrent session — it will silently move another session onto a
  different branch mid-task. This has already caused work to land on an unexpected branch once.
- **Commit to whatever branch is currently checked out.** Do not create new long-lived branches for
  parallel work in this same checkout.
- The two active branch names — `feature/kernel-policy-mining` and `feature/rtl-derived-checks` — are kept
  pointing at the **same** commit (one linear history; one is not "ahead" of the other in content). Keep
  committing on the current branch; if you must, fast-forward the *other* name to match afterward so they
  never drift. Never `git reset`/force-push either in a way that drops commits.
- If you genuinely need branch isolation, use a **git worktree** (separate directory) instead of switching
  HEAD in this shared tree.

Rationale: this repo is worked on by multiple concurrent agents. The only safe invariant in a shared
working tree is "everyone is always on the current HEAD; nobody flips it."

# Generated-output convention — three roots only

All generated/produced output lives in **exactly three top-level roots**. Nothing generated goes
anywhere else (the old `output/`, `results/`, `selfcheck_out/`, `docs/presentation/`, and
per-experiment `runs/`/`reports/` locations are retired and gitignored).

- **`runs/<target>/<suite>/<run-id>/`** — aet-managed experiment runs. Create via
  `merlin.common.artifacts.start_run(..., target=...)` (never hand-build a run path). Query with
  `aet runs`.
- **`artifacts/`** — every other generated product:
  - versioned products at `artifacts/<topic>/<target>/v<ver>/<topic>_<target>_v<ver>_<TS>_<sha7>/`
    (+ `manifest.yaml`, relative `latest` symlink) via `new_product(..., target=...)`;
  - regenerable caches under `artifacts/cache/<ns>/` via `cache_dir(...)` (PURGEABLE);
  - the 130 GB model recaptures under `artifacts/recaptures/` (PURGEABLE) via `recaptures_dir()`.
- **`build/`** — compiled / CMake / codegen scaffolds (unchanged).

**Naming ("sortable + provenance")**: timestamp token `YYYYMMDDTHHMMSSZ` (UTC, no `:`); runs are
timestamp-first (`<TS>_<method>_seed<NNN>_<sha7>`), products are topic-first
(`<topic>_<target>_v<ver>_<TS>_<sha7>`). The folder name is convenience; `run_record.json`/`manifest.yaml`
(git_sha, timestamp, version) is the source of truth.

**Target at folder level**: pass `target=` so it becomes a folder component
(`runs/<target>/<suite>/...`, `artifacts/<topic>/<target>/...`) and everything for a target groups
together. Keep inner file names identical across targets (e.g. `perf_results.json`, `findings.csv`,
`manifest.yaml`) so target-vs-target diffs are trivial.

**Enforcement** (do not bypass without cause): a PreToolUse hook
(`scripts/hooks/guard_artifact_writes.py`) blocks generated writes outside the three roots;
`scripts/check_artifact_layout.py` lints tracked-file violations (pre-commit / Stop hook). Helper
API and examples: `.claude/skills/artifact-layout/SKILL.md` and `merlin/python/merlin/common/artifacts.py`.
Escape hatch for a genuine one-off: `export MERLIN_ALLOW_ARTIFACT_WRITE=1` or add a prefix to
`scripts/hooks/artifact_allowlist.txt`.
