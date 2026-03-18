# Merlin Plugin and Patch Model

This document defines the maintenance model for Merlin with a small team.

## Goals

- Keep Merlin behavior in this repository and out-of-tree whenever possible.
- Track unavoidable in-tree IREE edits as commits on a dedicated fork branch.
- Fail fast in CI when the submodule drifts from the expected upstream base.

## Directory Contracts

- `compiler/src/merlin/`: Merlin-owned compiler logic (dialects, transforms, heuristics).
- `compiler/plugins/target/*/`: IREE plugin registration and target glue.
- `third_party/iree_bar`: Submodule pointing at `ucb-bar/main` branch of our IREE fork.
- `build_tools/patches/manifest.env`: Pins the upstream IREE commit that `ucb-bar/main` is based on.
- `build_tools/patches/upstream/`: Hand-curated `git format-patch` exports for upstream PR preparation.

## Plugin IDs and Backend IDs

Registered plugin IDs:

- `merlin_target_gemmini`
- `merlin_target_saturn`
- `merlin_target_spacemit`

Registered HAL backend IDs:

- `gemmini`
- `saturn`
- `spacemit`

Current baseline behavior: each backend aliases to `llvm-cpu` while Merlin-specific
pipeline wiring is migrated out-of-tree.

## Branch-Based Patch Management

### Branch layout

```
github.com/iree-org/iree              (vanilla upstream)
  └── main

github.com/ucb-bar/iree               (our fork)
  ├── main                             (tracks upstream vanilla)
  └── ucb-bar/main                     (default = upstream + merlin commits)
       ├── commit: "RISC-V ukernel tiles and codegen"
       ├── commit: "Bare-metal runtime: alignment + status propagation"
       └── commit: "RISC-V mmt4d ukernel architecture files"
```

The `third_party/iree_bar` submodule points at `ucb-bar/main`. There are no
diff-based patch files to apply — the submodule already contains all changes.

### Why commits instead of diffs

- **Rebasing upstream** is `git rebase upstream/main` — git resolves at the
  commit level, not per-file. Each Merlin commit rebases independently.
- **Conflicts are isolated** — if the ukernel commit conflicts but bare-metal
  doesn't, you only fix the ukernel commit.
- **History is preserved** — `git log`, `git blame`, `git rebase -i` all work
  normally.
- **No apply step** — `git submodule update --init` is all you need.

### Commit conventions

Each Merlin-specific commit on `ucb-bar/main` should:

- Have a clear, descriptive commit message prefixed with `[Merlin]`.
- Be atomic — one concern per commit (e.g., bare-metal fixes separate from
  ukernel tiles).
- Be ordered so that independent changes don't depend on each other's context
  lines (makes rebasing easier).

### Bumping IREE upstream

```bash
cd third_party/iree_bar

# Fetch latest upstream
git fetch https://github.com/iree-org/iree main

# Rebase our commits onto the new upstream
git rebase FETCH_HEAD

# Resolve per-commit conflicts, then continue
# git rebase --continue

# Push (force-with-lease for safety)
git push origin ucb-bar/main --force-with-lease

# Update submodule pointer in Merlin
cd ../..
git add third_party/iree_bar
```

After rebasing, update `IREE_UPSTREAM_BASE` in `build_tools/patches/manifest.env`.

### Verification commands

```bash
# Verify submodule is a clean rebase of pinned upstream base
merlin patches verify

# Show Merlin-specific commits
merlin patches log

# Check how far behind upstream we are
merlin patches drift
```

## Upstream PR Preparation

The `build_tools/patches/upstream/` directory contains hand-curated `git format-patch`
exports with READMEs explaining what to include/exclude for upstream PRs.

To prepare an upstream PR:

```bash
# Start from clean upstream main in the fork
cd third_party/iree_bar
git checkout main
git pull https://github.com/iree-org/iree main

# Cherry-pick just the relevant commit(s)
git checkout -b upstream-pr/my-feature
git cherry-pick <commit-hash-from-ucb-bar-main>

# Export as format-patch for documentation
merlin patches export-upstream <commit-hash>

# Push and open PR
git push origin upstream-pr/my-feature
```

## Policy for New Changes

- Prefer out-of-tree Merlin plugin/core changes first.
- If in-tree IREE change is unavoidable:
  - Make the edit in `third_party/iree_bar` on the `ucb-bar/main` branch.
  - Commit with a `[Merlin]` prefix and clear description.
  - Push to the fork.
  - Update the submodule pointer in Merlin.
- Keep commits atomic and independent where possible.

## CI Gate Expectations

The `merlin ci patch-gate` command runs all submodule checks:

```bash
conda run -n merlin-dev uv run tools/merlin.py ci patch-gate
```

This verifies:

1. **Ancestry** — submodule HEAD descends from `IREE_UPSTREAM_BASE` in
   `manifest.env`. Catches missed rebases or accidental force-pushes.
2. **Clean working tree** — no uncommitted changes in the submodule.
   Catches ad hoc edits that weren't committed to `ucb-bar/main`.
3. **Commit count** — reports how many Merlin commits sit on top of upstream
   (informational, not a gate).

A full CI pipeline should run:

```bash
uv run tools/merlin.py ci cli-docs-drift
uv run tools/merlin.py ci patch-gate
```
