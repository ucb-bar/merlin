# Merlin Plugin and Patch Model

This document defines the maintenance model for Merlin with a small team.

## Goals

- Keep upstream trees (`third_party/iree_bar` and nested `llvm-project`) near-clean.
- Keep Merlin behavior in this repository and out-of-tree whenever possible.
- Track unavoidable in-tree edits as a versioned patch stack.
- Fail fast in CI when drift or untracked edits appear.

## Directory Contracts

- `compiler/src/merlin/`: Merlin-owned compiler logic (dialects, transforms, heuristics).
- `compiler/plugins/target/*/`: IREE plugin registration and target glue.
- `build_tools/patches/iree/`: patch files applied to `third_party/iree_bar`.
- `build_tools/patches/llvm/`: patch files applied to `third_party/iree_bar/third_party/llvm-project`.
- `build_tools/patches/series.iree`, `build_tools/patches/series.llvm`: ordered patch manifests.
- `build_tools/patches/manifest.env`: pinned base commits for drift checks.

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

## Required Workflow

Preferred entrypoint (wraps the scripts below):

```bash
uv run tools/merlin.py --help
```

1. Ensure submodules are at pinned commits.
2. Export or refresh patch files from local in-tree edits:

```bash
./build_tools/patches/tools/refresh_all.sh
```

3. Apply patch stack to fresh trees:

```bash
./build_tools/patches/tools/apply_all.sh
```

4. Verify patched state is exactly as expected:

```bash
./build_tools/patches/tools/verify_clean.sh
```

5. Check drift against pinned upstream commits:

```bash
./build_tools/patches/tools/check_upstream_drift.sh
```

## Policy for New Changes

- Prefer out-of-tree Merlin plugin/core changes first.
- If in-tree change is unavoidable:
  - add tests upstream-side where possible,
  - export to patch stack,
  - update series files only when intentionally adding/removing patches.
- Do not leave ad hoc edits in upstream trees without corresponding patch files.

## CI Gate Expectations

A strict CI gate should run at least:

- `uv run tools/merlin.py ci cli-docs-drift`
- `build_tools/patches/tools/apply_all.sh`
- `build_tools/patches/tools/verify_clean.sh`
- `build_tools/patches/tools/check_upstream_drift.sh`

This catches upstream drift and silent local edits before demo/release branches.
