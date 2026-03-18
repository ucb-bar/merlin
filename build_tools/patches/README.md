# Patch Management

Merlin maintains downstream changes to IREE as commits on the `ucb-bar/main`
branch of our fork (`github.com/ucb-bar/iree`). The `third_party/iree_bar`
submodule points at this branch.

**There are no diff-based patch files to apply.** The submodule already
contains all Merlin changes. Just `git submodule update --init` and build.

## Branch Layout

```
github.com/iree-org/iree              (vanilla upstream)
  └── main

github.com/ucb-bar/iree               (our fork)
  ├── main                             (tracks upstream vanilla)
  └── ucb-bar/main                     (default branch = upstream + merlin commits)
       ├── commit: "RISC-V ukernel tiles and codegen"
       ├── commit: "Bare-metal runtime: alignment + status propagation"
       └── commit: "RISC-V mmt4d ukernel architecture files"
```

## Bumping IREE Upstream

When a new upstream IREE release is available:

```bash
cd third_party/iree_bar

# Fetch latest upstream
git fetch upstream    # or: git fetch https://github.com/iree-org/iree main

# Rebase our commits onto the new upstream
git rebase upstream/main

# Resolve any conflicts (per-commit, not per-file)
# git rebase --continue after each resolution

# Push the rebased branch
git push origin ucb-bar/main --force-with-lease

# Update the submodule pointer in Merlin
cd ../..
git add third_party/iree_bar
```

After rebasing, update `manifest.env` with the new `IREE_UPSTREAM_BASE`.

## CI Verification

CI checks that the submodule HEAD is a descendant of the pinned upstream base:

```bash
source build_tools/patches/manifest.env
git -C third_party/iree_bar merge-base --is-ancestor \
  "$IREE_UPSTREAM_BASE" HEAD
```

This catches accidental force-pushes or missed rebases.

## Upstream PR Preparation

The `patches/upstream/` directory contains hand-curated `git format-patch`
exports and READMEs for preparing PRs against `iree-org/iree`. These are
documentation artifacts, not part of the build system.

To prepare an upstream PR:

```bash
# Start from clean upstream main
git checkout main
git pull upstream main

# Cherry-pick just the relevant commit(s) from ucb-bar/main
git checkout -b upstream-pr/my-feature
git cherry-pick <commit-hash>

# Push and open PR against iree-org/iree
git push origin upstream-pr/my-feature
```

## Contents

- `manifest.env` — Pins the upstream IREE commit that `ucb-bar/main` is based on.
- `upstream/` — Hand-curated format-patch exports for upstream PR preparation.
