# AGENT.md — merlin/python/merlin/compile

## Purpose

Compile machinery behind ``merlin-compile``: bundles, the host lane, capacity, and mesh execution.

## Modules

- `application_plan.py` — Select complete compatible source regions from provider-admitted alternatives.
- `c_sources.py` — Package explicitly selected C sources into a standalone C/H pair.
- `c_parameters.py` — Embed declared immutable model parameters in a C call.
- `c_bucketize.py` — Compile the checked f32 boundary-count reduction as integer-only C.
- `c_conversions.py` — Render checked source float conversions and operand axis order.
- `c_tiling.py` — Render byte-preserving C packing for checked contraction and row-softmax tiling.
- `bundles.py` — Capture bundles for the RVV whole-model lane, and the scalar datatype a bundle's IR carries.
- `capacity.py` — Operand and accumulator capacity of a target's matrix unit, and the tile that fits it.
- `command.py` — The `merlin-compile` command line: its arguments, the combinations it refuses, and its report.
- `debug.py` — Compile debugging options (`--list-stages`, `--dump-ir-after/-before`, `--stop-after`, `--trace-dir`), shared by `merlin-compile`, the whole-model builder CLI and launch configs.
- `host_lane.py` — The RVV host lane: which package a compile uses for each datatype, and its provenance pin.
- `mesh.py` — Execute and certify matmul layers on a target's accelerator mesh.
- `mesh_backend.py` — Plumbing for running a layer through a target's out-of-tree backend package.
- `mesh_model.py` — Drivers that run every matmul layer of a whole model, or of an int8 layer chain, on the mesh.
- `mesh_reference.py` — The host-side reference a mesh tile is checked against.
- `route_before_build.py` — Source-derived placement and device routing selected before whole-model compilation.

<!-- Purpose/Modules derived from docstrings via build_tools/scripts/gen_package_docs.py.
     Add hand-written notes (invariants, gotchas) below. -->

## Invariants

- `merlin.compile_cli` is the front door: `compile_rvv`, `compile_model`, `compile_oot`, `main`, and the
  steps only they use (`_ensure_bundle`, `_workload_features`, `_session_correctness_gate`,
  `_summarize_route_plan`) stay DEFINED there, because other modules look them up through it at call
  time (`compare.study`, `targetgen.capsule_runner`) and tests replace them there. The
  exact historical facade imports from the extracted modules are re-exported by `compile_cli`
  for existing callers. New module APIs keep their helpers local rather than growing the CLI
  facade's import surface. The compatibility test pins those historical names explicitly;
  its monkeypatch scan still discovers all compile modules, including nested packages.
  `command.py` holds only the argument parsing, validation and report `main` calls; it dispatches
  nothing itself, so no test needs to patch it.
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
- Source float casts use integer carriers for the declared IEEE rounding. Reject
  unqualified NaNs before any output write and preserve disjoint-buffer checks.
  These CPU implementations require explicit placement; they are not a native
  device fallback. Logical identity copying does not establish a physical alias.
- Pre-encoded BF16 input copying preserves exact code bytes, including signed
  zero, after finite/code-range validation. Reject policy flags that would
  round, scale, clip or canonicalize the input. Check every lane before writing.
- Binary64 multiplication uses a full 106-bit integer product and one RNE step.
  Keep shared operands on one C ABI port; check all invalid lanes and overlap
  before any output write. This route requires explicit CPU placement.
- BF16 source add/sub uses exact signed integer magnitudes before one RNE
  step. Preserve subnormals and both zero signs. Reject NaNs and invalid
  infinity cancellation before output writes. It is explicit CPU computation,
  not a qualification of any target's truncated arithmetic.
- Minimum/first-index CPU reductions compare signed-i64 values without signed
  overflow and retain the first equal minimum. Keep exact axis, output shape
  and source policy checks.
- State initialization transports raw float code bytes unchanged. Preserve
  the declared input/update mapping and caller ownership; byte copying does
  not qualify arithmetic or initialize caller-provided state.
- Source sine/cosine requires a separately selected library/numerical policy.
  Archive identity alone is not a mathematical or framework-equivalence grant.
  Preserve finite-input/RNE guards and exact f32/f64 call signatures. Merge
  actual symbol requirements into the existing application interface; final
  linking must check the supplier. Never use a verification helper at runtime.
