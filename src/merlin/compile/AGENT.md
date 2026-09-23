# AGENT.md — merlin/python/merlin/compile

## Purpose

Compile machinery behind ``merlin-compile``: bundles, the host lane, capacity, and mesh execution.

## Modules

- `bundles.py` — Capture bundles for the RVV whole-model lane, and the scalar datatype a bundle's IR carries.
- `capacity.py` — Operand and accumulator capacity of a target's matrix unit, and the tile that fits it.
- `host_lane.py` — The RVV host lane: which package a compile uses for each datatype, and its provenance pin.
- `mesh.py` — Execute and certify matmul layers on a target's accelerator mesh.
- `mesh_backend.py` — Plumbing for running a layer through a target's out-of-tree backend package.
- `mesh_model.py` — Drivers that run every matmul layer of a whole model, or of an int8 layer chain, on the mesh.
- `mesh_reference.py` — The host-side reference a mesh tile is checked against.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->

## Invariants

- `merlin.compile_cli` is the front door: `compile_rvv`, `compile_model`, `compile_oot`, `main`, and the
  steps only they use (`_ensure_bundle`, `_workload_features`, `_session_correctness_gate`,
  `_summarize_route_plan`) stay DEFINED there, because other modules look them up through it at call
  time (`compare.study`, `targetgen.capsule_runner`) and tests replace them there. Everything this
  package defines is re-exported by `compile_cli` for callers.
- Patch a name in the module that DEFINES it. A re-export (in `compile_cli`, or a sibling imported by
  name, e.g. `mesh` importing `capacity`'s helpers) is a separate binding the defining module's callers
  never read, so a patch there passes without testing anything.
  `merlin/tests/infra/test_compile_cli_patch_targets.py` fails on such a patch.
- Keep a patched helper in the same module as the callers a test drives through it: `mesh` holds
  `_default_oot_package` and `_mesh_tile_binding` because both `_mesh_verify` and `run_matmul_on_mesh`
  resolve them, and `run_matmul_on_mesh`/`_mesh_rows`/`_certify_tile_via_executor` call each other.
- No import cycles: `mesh` imports `capacity`, `mesh_backend` and `mesh_reference`; `mesh_model` imports
  `mesh`; `host_lane` imports `bundles`. Library imports stay inside functions, as they were.
- Mesh compilation/execution takes `numeric_policy` explicitly. It carries the author's complete
  numerical regime (comparison, tolerances, scaling and subnormal handling), not hardware evidence.
  Core never discovers a profile by target name. Floating execution without that declaration refuses;
  callers must forward it through recursive tiling, certification, runtime dispatch and device builds.
