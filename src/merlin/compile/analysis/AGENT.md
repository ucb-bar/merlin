# AGENT.md — src/merlin/compile/analysis

## Purpose

Target-agnostic analyses that derive facts from IR, shapes and a target's own derived tables.

## Modules

- `compute_endpoints.py` — Bind the closed role vocabulary to each target's OWN derived encoding table.
- `contraction_shapes.py` — Observe the CONTRACTION SHAPES a module carries, and the extents a cover reads off them.
- `disassembly.py` — Generic, ISA-agnostic disassembly tokenizer.
- `engine_facets.py` — The ENGINE model — the one place that says which CCA facet a compute datapath is described by.
- `instruction_roles.py` — The closed ROLE vocabulary — what an instruction DOES to a compute endpoint.
- `intrinsics_header.py` — Derive a custom-instruction table from the target's OWN intrinsics header.
- `rocc_commands.py` — Decode a RoCC command stream against the target's OWN derived funct table.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->

## Invariants

- Public compiler source. `merlin_experiments.phase1.full_compiler_workspace` stages this package
  into a full-compiler author's tree while `merlin.kernels` and `merlin.mining` stay withheld, so
  nothing here may import either of them, a reference runtime, a grader or an oracle -- not even
  lazily. `packages/merlin-experiments/tests/test_full_compiler_workspace_imports.py` imports
  every staged module with those prefixes refused.
- Analyses only: each module reads its subject (IR, a disassembly, the target's RTL-derived or
  contract-declared tables) and reports what it found. No kernel, schedule, tuned block, measured
  choice or baseline enters it; a fact that cannot be derived is reported as unknown or refused.
- `contraction_extents` is the one `id(op) -> (m, k, n, rank)` table; consumers (region
  coverage, compute grouping, device offload, output fusion) call it instead of re-deriving which
  axes are parallel and which reduce. Its windowed reader does not depend on the observer.
- The old `merlin.kernels.{shapes,roles,engines,endpoints}` and
  `merlin.kernels.decode.{rocc,objdump,insn_header}` paths are `sys.modules` aliases of these
  modules (same objects, so patches and identities agree). New code imports from here.
