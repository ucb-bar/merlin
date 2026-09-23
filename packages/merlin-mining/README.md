# Merlin mining extension

Install alongside the core:

```sh
python -m pip install ./packages/merlin-mining
```

This distribution owns search, beam/fork campaigns, optional LLM proposers,
measurement orchestration, and mining reports. Its eleven `merlin-kernel-*`,
`merlin-rvv-*`, and `merlin-cca-route` command names retain their previous behavior.
Historical `merlin.mining.*` imports remain unchanged. Core owns the single package
initializer and extends its path; the extension contributes modules, not another
copy of core classes or registries. `merlin.rvvgen` remains an identity-preserving
compatibility alias; importing a search module through it still needs this extension.

Core retains these compiler/runtime primitives under their historical names during
staged cutover:

| Module | Why ordinary compilation still needs it |
| --- | --- |
| `registry` | Validates and loads isolated schedule packages; owns dtype vocabulary |
| `apply` | Adapts compiler features to workload and per-hart contraction shapes |
| `from_strategy` | Renders schedules and registers the microkernel realization |
| `host_actions` | Validates host-policy legality and mechanism requirements |
| `lever_applicability` | Rejects compiler features inapplicable to the input IR |
| `k1` | Executes the existing whole-model board build/run path |

The compatibility `from_strategy.mint_fork` call delegates lazily to this extension.
Loading a compiler package or rendering its schedule does not import search or LLM
proposers. Core also retains `merlin.kernels`: its CCA/measurement/shape/ISA modules
are used by the compiler and trusted graders. Physically extracting that subtree
requires a further semantic split, not declaring mining mandatory in core.

This does not make hardware or compiler tools available automatically. Native
toolchain/board discovery and absence behavior are unchanged. Pass real target
packages and configured external tools; the package neither downloads them nor
creates sibling repositories. Optional section-build compilation uses the
`compiler` extra. Existing core kernel-ingest extras remain core-owned for now.

For source development without installation, include both `src` and
`packages/merlin-mining/src` on `PYTHONPATH`. Do not install into another worktree's
shared virtual environment implicitly.
