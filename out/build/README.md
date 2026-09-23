# Build and working state

Generated build trees, tool environments, buildable out-of-tree packages, test
scratch space and agent workspaces live here. Source belongs in `src/`,
`packages/`, `build_tools/` or its out-of-tree repository, never in a build tree.

Common owners include `cmake/` for compiler builds, `python/` for distribution
builds and install checks, `generated/` for generated buildable target trees, and
`agent-workspaces/` for active compiler-authoring workspaces. A workspace may have
frozen input siblings and an active lease; being beneath `build/` does not make
it safe to delete while a run exists.

Preserve release/qualification logs needed to reproduce results. Use the
[storage guide](../../docs/guides/storage.md) and supported dry-run commands
before reclamation. Most contents are ignored; tracked README/AGENT files describe
the layout rather than storing build implementation.
