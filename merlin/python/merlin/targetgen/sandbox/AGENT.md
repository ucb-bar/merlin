# AGENT.md — merlin/python/merlin/targetgen/sandbox

## Purpose

Shared, descriptor+manifest-driven agentic bwrap sandbox — a new target gets a correct, continuously-guarded isolation sandbox from its `target_experiment.yaml` (+ capability manifest) with ZERO copied scripts. Routing is by compute-unit KIND / sim FAMILY, never a target name.

## Modules

- `answer_surfaces.py` — The DERIVED answer-surface mask set (goldens/hidden/prior/oracle/grader/memory), the single declared oracle+grader registry, and the transcript-audit tokens derived from it.
- `toolchain.py` — The legit tools bound back over the deny-by-default masks: universal + the descriptor's `sim_via` family, cross-checked by `kind` via `merlin.targetgen.families`.
- `bwrap.py` — Deny-by-default argv assembly + the hermetic mount-table replay that PROVES no answer surface is reachable (coverage guard), without launching bwrap.
- `__init__.py` — `build_sandbox(descriptor, ws, bundle)` → a `Sandbox` facade (argv / env / wrap / coverage_gap); `resolve_kind` for family routing.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->

## Invariants / notes

- DERIVE, never per-target-branch. The mask set comes from the descriptor's corpus/`answer_surfaces` + the declared `ORACLE_MODULES`/`GRADER_MODULES` registry; the toolchain routes on `sim_via` (declarative `SIM_TOOLCHAINS` table) + `kind`. There is no `if target ==` anywhere.
- The coverage guard is the drift/cheat gate: `bwrap.coverage_gap(argv, surfaces)` replays the ordered mount table and returns any answer surface still reachable — it MUST be empty. The historical cheat gap (a hard-coded slug that left the `~/.claude` memory dir unmasked) is exactly a non-empty gap; the memory dir is derived from the current repo path so it can never go stale.
- `apply_answer_masks` only masks a surface a bind would otherwise RE-EXPOSE (so it never `/dev/null`-overlays a path whose parent is an empty tmpfs — that mount would fail). Surfaces already hidden by deny-by-default need no overlay.
- Continuously guarded by `merlin/tests/infra/test_sandbox_isolation.py` (hermetic policy assertions for every roster target + a guarded live bwrap probe).
