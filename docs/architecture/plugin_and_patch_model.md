Here’s an updated version that reflects the workflow we just used in practice, including backups, LLVM-first rebases, submodule-pointer conflicts, temp branches, and post-force-push local reset behavior.

````text
# Merlin Plugin and Patch Model

This document defines the maintenance model for Merlin with a small team and
documents the practical rebase workflow for both IREE and its pinned
`llvm-project`.

## Goals

- Keep Merlin behavior in this repository and out-of-tree whenever possible.
- Track unavoidable in-tree IREE edits as commits on a dedicated fork branch.
- Track unavoidable in-tree LLVM edits as commits on a dedicated fork branch.
- Rebase downstream branches in a way that preserves a recoverable backup at all
  times.
- Fail fast in CI when the submodule drifts from the expected upstream base.

## Directory Contracts

- `compiler/src/merlin/`: Merlin-owned compiler logic (dialects, transforms, heuristics).
- `compiler/plugins/target/*/`: IREE plugin registration and target glue.
- `third_party/iree_bar`: Submodule pointing at `ucb-bar/main` branch of our IREE fork.
- `third_party/iree_bar/third_party/llvm-project`: Nested submodule pointing at our LLVM fork commit used by IREE.
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

```text
github.com/iree-org/iree              (vanilla upstream)
  └── main

github.com/ucb-bar/iree               (our fork)
  ├── main                             (should stay close to upstream vanilla)
  └── ucb-bar/main                     (default downstream branch = upstream + merlin commits)

github.com/iree-org/llvm-project      (IREE's LLVM fork)
  └── <pinned commit used by iree-org/iree main>

github.com/ucb-bar/llvm-project       (our LLVM fork)
  └── ucb-bar/main                     (default downstream LLVM branch = IREE-pinned LLVM + merlin LLVM commits)
````

The `third_party/iree_bar` submodule points at `ucb-bar/main`. Inside that
submodule, `third_party/llvm-project` points at the downstream LLVM commit
needed by our rebased IREE branch.

There are no diff-based patch files to apply for normal downstream development.
The submodule branches themselves contain the patch stack.

### Why commits instead of diffs

* **Rebasing upstream** happens at the commit level. Each Merlin commit replays
  independently.
* **Conflicts are isolated**. If one codegen commit conflicts but a bare-metal
  runtime commit does not, only the conflicting commit needs manual resolution.
* **History is preserved**. `git log`, `git blame`, `git range-diff`, and
  interactive rebases work normally.
* **Upstream PR prep is cleaner**. Individual downstream commits can be
  cherry-picked or exported selectively.
* **No patch apply step**. Normal operation is just submodule checkout and update.

## Commit Conventions

Each Merlin-specific commit on downstream branches should:

* Have a clear, descriptive message.
* Prefer a `[Merlin]` prefix for new downstream-only commits.
* Be atomic: one concern per commit.
* Avoid mixing unrelated subsystems in one commit.
* Be ordered so that independent changes do not unnecessarily depend on each
  other's context lines.

Examples of good granularity:

* RVV/SpacemiT encoding model changes
* LLVMCPU kernel dispatch policy
* vector contract custom kernel lowering
* bare-metal runtime build gating
* LLVM pointer update commit in IREE

## Rebase Policy

### Golden rule

Rebase in dependency order:

1. **Rebase LLVM first**
2. **Then rebase IREE**
3. **Then update the IREE LLVM submodule pointer to the rebased LLVM commit**

Reason: IREE depends on a specific LLVM state. Rebasing IREE before its
underlying LLVM branch is settled creates avoidable conflicts and ambiguity.

### Rebase target for LLVM

Do **not** rebase our downstream LLVM branch onto raw `llvm/llvm-project main`
unless that is explicitly the intended base.

For Merlin's IREE maintenance workflow, the LLVM base is:

* the exact `third_party/llvm-project` commit pinned by `iree-org/iree main`

That SHA can be obtained from the IREE repo with:

```bash
cd third_party/iree_bar
git fetch upstream
git ls-tree upstream/main third_party/llvm-project
```

Use the reported gitlink SHA as the LLVM rebase base.

### Rebase target for IREE

Rebase `ucb-bar/main` in `ucb-bar/iree` onto:

```bash
upstream/main
```

where `upstream` is `https://github.com/iree-org/iree.git`.

## Backup Policy

Before any history rewrite, create both:

* a backup branch
* a backup tag

Do this for **both** repos:

* `ucb-bar/llvm-project`
* `ucb-bar/iree`

Recommended names:

```text
backup/ucb-bar-main-pre-rebase-YYYY-MM-DD
backup-llvm-ucb-bar-main-pre-rebase-YYYY-MM-DD
backup-iree-ucb-bar-main-pre-rebase-YYYY-MM-DD
```

Push the backups to the remote before rebasing. These refs remain valid even
after force-pushing `ucb-bar/main`.

Example pattern:

```bash
git branch backup/ucb-bar-main-pre-rebase-2026-03-23
git tag backup-iree-ucb-bar-main-pre-rebase-2026-03-23
git push origin backup/ucb-bar-main-pre-rebase-2026-03-23
git push origin backup-iree-ucb-bar-main-pre-rebase-2026-03-23
```

## Practical Rebase Workflow

### 0. Helpful one-time config

In each repo:

```bash
git config rerere.enabled true
git config rebase.autoStash true
```

`rerere` is valuable when similar conflicts recur across multiple rebase stops.

### 1. Rebase LLVM first

In `third_party/iree_bar/third_party/llvm-project`:

```bash
git fetch origin
git fetch iree-org
git fetch upstream

git switch ucb-bar/main
git pull --ff-only origin ucb-bar/main

git branch backup/ucb-bar-main-pre-rebase-YYYY-MM-DD
git tag backup-llvm-ucb-bar-main-pre-rebase-YYYY-MM-DD
git push origin backup/ucb-bar-main-pre-rebase-YYYY-MM-DD
git push origin backup-llvm-ucb-bar-main-pre-rebase-YYYY-MM-DD

git switch -c rebase/ucb-bar-main-YYYY-MM-DD
```

Decide whether to preserve merges:

```bash
git log --oneline --merges <IREE_PINNED_LLVM_SHA>..HEAD
```

* If this prints nothing: use plain `git rebase <IREE_PINNED_LLVM_SHA>`
* If it prints meaningful downstream merges that should be preserved:
  use `git rebase --rebase-merges <IREE_PINNED_LLVM_SHA>`

After the rebase succeeds:

```bash
git push origin HEAD:rebase/ucb-bar-main-YYYY-MM-DD
git rev-parse HEAD
```

Record the resulting rebased LLVM SHA. That is the LLVM commit IREE should point to.

### 2. Rebase IREE second

In `third_party/iree_bar`:

```bash
git fetch origin
git fetch upstream

git switch ucb-bar/main
git pull --ff-only origin ucb-bar/main

git branch backup/ucb-bar-main-pre-rebase-YYYY-MM-DD
git tag backup-iree-ucb-bar-main-pre-rebase-YYYY-MM-DD
git push origin backup/ucb-bar-main-pre-rebase-YYYY-MM-DD
git push origin backup-iree-ucb-bar-main-pre-rebase-YYYY-MM-DD

git switch -c rebase/ucb-bar-main-YYYY-MM-DD
```

Check for merge commits:

```bash
git log --oneline --merges upstream/main..HEAD
```

* If empty: use plain `git rebase upstream/main`
* Otherwise: `git rebase --rebase-merges upstream/main`

### 3. Conflict handling during rebase

General loop:

```bash
git status
# resolve files manually
git add <resolved-files>
git rebase --continue
```

Do **not** create a normal commit during a rebase. Use `git rebase --continue`.

Use `git rebase --skip` only if you are certain the replayed commit is already
fully subsumed by the new base and should not be kept.

### 4. Submodule pointer conflicts during IREE rebase

IREE rebases may stop on commits that update `third_party/llvm-project`.

When Git reports a submodule conflict in `third_party/llvm-project`, resolve it
by checking out the **final rebased LLVM SHA** inside the submodule and staging
the gitlink in the IREE superproject:

```bash
git -C third_party/llvm-project fetch origin
git -C third_party/llvm-project checkout <REBASED_LLVM_SHA>
git add third_party/llvm-project
git rebase --continue
```

Important:

* This may happen more than once if the historical IREE branch contains multiple
  LLVM pointer update commits.
* In practice, repeated pointer conflicts can usually be resolved to the same
  final rebased LLVM SHA.

### 5. Push rebased branches to temporary remote branches first

Before replacing the real branch names, push the rebased result to a temp branch:

```bash
# LLVM
git push origin HEAD:rebase/ucb-bar-main-YYYY-MM-DD

# IREE
git push origin HEAD:rebase/ucb-bar-main-YYYY-MM-DD
```

This gives a stable remote ref for inspection, CI, and recovery.

### 6. Validate, then replace `ucb-bar/main`

Once satisfied with the rebased result:

```bash
# LLVM
git push --force-with-lease origin rebase/ucb-bar-main-YYYY-MM-DD:ucb-bar/main

# IREE
git push --force-with-lease origin rebase/ucb-bar-main-YYYY-MM-DD:ucb-bar/main
```

Always use `--force-with-lease`, never plain `--force`.

## Important Local-State Caveats

### Force-pushing the remote branch does not move the local branch automatically

After force-pushing:

* `origin/ucb-bar/main` points at the rebased history
* local `ucb-bar/main` may still point at the old pre-rebase tip

This is normal.

To align the local branch with the rewritten remote:

```bash
git switch ucb-bar/main
git reset --hard origin/ucb-bar/main
```

Do this in both:

* `third_party/iree_bar`
* `third_party/iree_bar/third_party/llvm-project`

### Detached HEAD in submodules is normal

When `llvm-project` is checked out as a submodule, it may end up in detached HEAD
state at the exact pinned commit. This is expected.

If you need to work on the named branch again:

```bash
git switch ucb-bar/main
```

or, if needed:

```bash
git switch -c ucb-bar/main --track origin/ucb-bar/main
```

### Dirty submodules and local scratch files

A rebased branch can be correct even if the working tree is dirty.

Examples:

* modified submodules such as `third_party/benchmark`, `third_party/stablehlo`,
  `third_party/torch-mlir`, `third_party/tracy`
* local notes, scratch MLIR files, test directories, logs, or helper scripts

Guidance:

* Do not accidentally include unrelated local dirt in rebased commits.
* `git reset --hard origin/ucb-bar/main` in the superproject does **not** remove
  top-level untracked files.
* `git submodule update --init --recursive` may move submodule checkouts back to
  recorded commits and may complain if a submodule contains local modifications.

If local scratch files are important, back them up outside Git before aggressive
cleanup. A simple `rsync` backup is sufficient.

## Recommended Verification

### Fast repo sanity checks

```bash
# Show current branch and cleanliness
git status -sb

# Show current commit
git rev-parse HEAD

# Show current pinned LLVM commit from IREE
git -C third_party/llvm-project rev-parse HEAD

# Show submodule states
git submodule status
```

### Compare old vs rebased patch stack

Use `git range-diff`:

```bash
# LLVM
git range-diff <IREE_PINNED_LLVM_SHA>...backup/ucb-bar-main-pre-rebase-YYYY-MM-DD \
               <IREE_PINNED_LLVM_SHA>...rebase/ucb-bar-main-YYYY-MM-DD

# IREE
git range-diff upstream/main...backup/ucb-bar-main-pre-rebase-YYYY-MM-DD \
               upstream/main...rebase/ucb-bar-main-YYYY-MM-DD
```

This is one of the best checks that the intended downstream patch stack survived
the rebase.

### Backup verification

Verify backup refs still exist both locally and remotely:

```bash
git branch --list 'backup/*'
git tag --list 'backup*'
git ls-remote --heads origin 'backup/*'
git ls-remote --tags origin 'backup*'
```

## Bumping IREE Upstream (Condensed Procedure)

```bash
# 1. Determine the LLVM SHA pinned by upstream IREE
cd third_party/iree_bar
git fetch upstream
git ls-tree upstream/main third_party/llvm-project

# 2. Rebase downstream LLVM onto that pinned LLVM SHA
cd third_party/llvm-project
git fetch origin
git fetch iree-org
git switch ucb-bar/main
git switch -c rebase/ucb-bar-main-YYYY-MM-DD
git rebase <IREE_PINNED_LLVM_SHA>
git push origin HEAD:rebase/ucb-bar-main-YYYY-MM-DD

# 3. Rebase downstream IREE onto upstream/main
cd ..
git fetch origin
git fetch upstream
git switch ucb-bar/main
git switch -c rebase/ucb-bar-main-YYYY-MM-DD
git rebase upstream/main

# 4. Resolve LLVM submodule pointer conflicts to the final rebased LLVM SHA
git -C third_party/llvm-project checkout <REBASED_LLVM_SHA>
git add third_party/llvm-project
git rebase --continue

# 5. Push the rebased IREE temp branch
git push origin HEAD:rebase/ucb-bar-main-YYYY-MM-DD

# 6. After validation, replace the real branches
# LLVM repo:
git push --force-with-lease origin rebase/ucb-bar-main-YYYY-MM-DD:ucb-bar/main

# IREE repo:
git push --force-with-lease origin rebase/ucb-bar-main-YYYY-MM-DD:ucb-bar/main
```

After rebasing, update `IREE_UPSTREAM_BASE` in `build_tools/patches/manifest.env`.

## Verification Commands

```bash
# Verify submodule is a clean rebase of pinned upstream base
merlin patches verify

# Show Merlin-specific commits
merlin patches log

# Check how far behind upstream we are
merlin patches drift
```

## Upstream PR Preparation

The `build_tools/patches/upstream/` directory contains hand-curated
`git format-patch` exports with READMEs explaining what to include or exclude
for upstream PRs.

To prepare an upstream PR:

```bash
# Start from clean upstream main in the IREE fork
cd third_party/iree_bar
git checkout main
git fetch upstream
git reset --hard upstream/main

# Create topic branch
git checkout -b upstream-pr/my-feature

# Cherry-pick the relevant downstream commit(s)
git cherry-pick <commit-hash-from-ucb-bar-main>

# Export format-patch for documentation
merlin patches export-upstream <commit-hash>

# Push and open PR
git push origin upstream-pr/my-feature
```

Do the analogous process in `llvm-project` for LLVM-only upstream work.

## Policy for New Changes

* Prefer out-of-tree Merlin plugin/core changes first.
* If an in-tree IREE change is unavoidable:

  * make it in `third_party/iree_bar` on `ucb-bar/main`
  * commit it clearly and atomically
  * push to the fork
  * update the Merlin superproject submodule pointer as needed
* If an in-tree LLVM change is unavoidable:

  * make it in `third_party/iree_bar/third_party/llvm-project` on `ucb-bar/main`
  * commit it clearly and atomically
  * keep IREE's LLVM submodule pointer consistent with it
* Keep commits atomic and independent where possible.

## CI Gate Expectations

The `merlin ci patch-gate` command runs all submodule checks:

```bash
conda run -n merlin-dev uv run tools/merlin.py ci patch-gate
```

This verifies:

1. **Ancestry** — IREE submodule HEAD descends from `IREE_UPSTREAM_BASE` in
   `manifest.env`. Catches missed rebases or accidental force-pushes.
2. **Clean working tree** — no uncommitted changes in the IREE submodule.
   Catches ad hoc edits that were not committed to `ucb-bar/main`.
3. **Commit count** — reports how many downstream commits sit on top of upstream
   (informational, not a gate).

A full CI pipeline should run:

```bash
uv run tools/merlin.py ci cli-docs-drift
uv run tools/merlin.py ci patch-gate
```

## Recovery

If a rebase goes wrong before pushing:

```bash
git rebase --abort
```

If a bad rewritten branch was already pushed, restore from the backup branch:

```bash
git switch ucb-bar/main
git reset --hard backup/ucb-bar-main-pre-rebase-YYYY-MM-DD
git push --force-with-lease origin ucb-bar/main
```

Because backup branches and tags are pushed before the rebase, recovery remains
possible even after force-pushing the main downstream branches.
