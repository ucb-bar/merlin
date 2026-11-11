# Build and working state

Disposable and rebuildable state: `cmake/` compiler builds, `python/` distribution
builds and install checks, `toolchains/` baseline toolchains, `generated/` buildable
out-of-tree target repositories, `agent-workspaces/` active compiler-authoring
workspaces.

Being under `build/` does not make something safe to delete: an agent workspace can
have frozen input siblings and an active lease. Use `merlin storage prune` (dry run
by default) rather than deleting by hand.
