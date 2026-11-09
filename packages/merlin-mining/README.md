# Merlin mining workflows

Mining workflows ship in the main `merlin` package. From the repository root:

```sh
python -m pip install '.[mining]'
```

This directory retains tests and documentation; active implementation lives under
`src/merlin/mining`. The mining logical component owns search, beam/fork campaigns, optional LLM proposers,
measurement orchestration, and mining reports. Its eleven `merlin-kernel-*`,
`merlin-rvv-*`, and `merlin-cca-route` command names retain their previous behavior.
Historical `merlin.mining.*` imports remain unchanged inside the single main package.
Shared compiler classes and registries have one implementation. `merlin.rvvgen` remains
an identity-preserving compatibility alias; using optional search or rendering engines
through it still requires their selected dependencies.

Core retains these compiler/runtime primitives under their historical names during
staged cutover:

| Module | Why ordinary compilation still needs it |
| --- | --- |
| `registry` | Validates and loads isolated schedule packages; owns dtype vocabulary |
| `apply` | Adapts compiler features to workload and per-hart contraction shapes |
| `from_strategy` | Compatibility selection/delegation to OOT schedule and microkernel policy |
| `host_actions` | Validates host-policy legality and mechanism requirements |
| `lever_applicability` | Rejects compiler features inapplicable to the input IR |
| `k1` | Executes the existing whole-model board build/run path |

The compatibility `from_strategy.mint_fork` call delegates lazily to the mining research owner.
Loading a compiler package or rendering its schedule does not import search or LLM
proposers. Core also retains `merlin.kernels`: its CCA/measurement/shape/ISA modules
are used by the compiler and trusted graders. Physically extracting that subtree
requires a further semantic split, not declaring mining mandatory in core.

Vector rendering now requires the OOT compiler dependency (`pip install
'merlin[vector]'`, or an explicitly selected support checkout).
Pass `codegen_provider` or `support_target` to select a policy directly. The
historical explicitly RVV APIs can use the installed named RVV service; arbitrary
SoC/package target names do not imply that provider. Scalar package loading stays
independent. Importing `from_strategy` registers no resolver; callers using the
generic resolver registry must opt in explicitly. Generated forks retain their
lineage and record/recheck the selected renderer's actual source bytes.

This does not make hardware or compiler tools available automatically. Native
toolchain/board discovery and absence behavior are unchanged. Pass real target
packages and configured external tools; the package neither downloads them nor
creates sibling repositories. Optional section-build compilation uses the
`compiler` extra. Existing core kernel-ingest extras remain core-owned for now.

For source development without installation, select this checkout's `src` on
`PYTHONPATH`. Do not install into another worktree's shared virtual environment
implicitly. Keep the independent `merlin-timing` package and frozen historical
runtime environments separate.
