# Allowed Merlin tooling — generated for `merlin_assisted_verify_public_v0`

This is the human-readable view of the **authoring** grants for arm `merlin_verify`. It is generated from
`input_bundle_manifest.yaml`; the manifest remains the machine-readable authority.

## Runtime isolation (launch-derived)

This bundle does **not** select a sandbox. Read `TASK.md` → **Runtime scope** (`Active sandbox`) and the
run's `environment.yaml` → `sandbox` for the mode actually used. Both fields are written from the
launcher's real `--sandbox` argument. A scored, trusted run requires deny-by-default `bwrap` plus its
frozen input snapshot. An explicit `none` run is diagnostic only and supports no trusted isolation claim.
If older static prose names another mode, those run artifacts win.

## Granted tools — what they are and how to invoke them

### `merlin_infra`

Answer-free support modules imported transitively by the xDSL, CCA and RTL-profile authoring tools. Not an oracle, grader or answer surface — granting them widens no moat, and withholding one disables the advertised tool that imports it.

### `xdsl_kit`

The xDSL authoring kit: dialect synthesis, the generators, the dialect definitions, the interface emitters and the out-of-tree starter kit.

### `cca_spine`

The Common-Compute-Abstraction spine — the where/how of modifying a compiler: extract a CCA, diff two, check the CCA<->action bijection, walk the escalation ladder, author a microkernel.

### `isa_tools`

Derived assembler, disassembler, static linter and lite debugger for the target's own ISA. Oracle-free: it encodes the syntax you chose and inspects the words you emitted.

### `cca_tools`

The two mandated CCA introspection calls — check_bijection and escalation_ladder — reachable as plain imports inside the sandbox. Oracle-free: public schema against public routes.

### `rtl_generators`

The CIRCT RTL-fact generators: derive an ISA encoder module, a distilled RTL digest and a numeric-shape checker from the target's elaborated RTL rather than from its documentation.

### `rtl_facts`

The facts already extracted from THIS target's RTL — the generators' output, granted directly.

### `verify_seam`

The compiler-verification seam. `python -m merlin.verify.cli compile --interface <f.mlir> --command-buffer <cb.json>` proves whether YOUR emitted command buffer computes what the interface program specified, for every integer input at that shape: exit 0 verified, 1 refuted with the counterexample inputs printed, 2 abstained. An abstention is a limit of the checker, not a defect in your buffer. Also: the fault/detection matrix that scores what a check would have caught, a sweep of the target's derived extent lattice, and `python -m merlin.xdsl_dialects.opt`, which runs a single pass on a single module. Advisory only -- it does not gate your submission.

## Allowed authoring inputs

- `merlin/contract/` — frozen ABI v0.1
- `merlin/contract/capsules/isa/` — capsule corpus
- `merlin/contract/capsules/layers/` — declared by the generated bundle manifest
- `merlin/contract/capsules/model/` — declared by the generated bundle manifest
- `merlin/contract/capsules/model_slices/` — declared by the generated bundle manifest
- `experiments/capsule_bench/targets/gemmini/contracts/hwbringup_gemmini_v0/isa_include/gemmini.h` — ISA header (shared hardware spec)
- `experiments/capsule_bench/targets/gemmini/contracts/hwbringup_gemmini_v0/isa_include/gemmini_params.h` — ISA header (shared hardware spec)
- `experiments/capsule_bench/targets/gemmini/task/` — declared by the generated bundle manifest
- `third_party/llvm-install/` — LLVM/MLIR 23 toolchain
- `experiments/capsule_bench/targets/gemmini/contracts/hwbringup_gemmini_v0` — shared hardware spec: RTL + ISA headers + README + example (ALL arms)
- `experiments/capsule_bench/targets/gemmini/scripts/agent_selfcheck.py` — redacted self-check
- `out/artifacts/targets/rvv/impr_tuned_wholemodel_vf_int8/` — frozen host lane (pinned infrastructure, read-only)
- `merlin/python/merlin/common/` — ALLOWED tool: xDSL kit / CCA spine
- `merlin/python/merlin/targetgen/families.py` — ALLOWED tool: xDSL kit / CCA spine
- `merlin/python/merlin/targetgen/compute_units.py` — ALLOWED tool: xDSL kit / CCA spine
- `merlin/python/merlin/targetgen/semantic_families.py` — ALLOWED tool: xDSL kit / CCA spine
- `merlin/python/merlin/targetgen/target_experiment.py` — ALLOWED tool: xDSL kit / CCA spine
- `merlin/python/merlin/targetgen/evidence/store.py` — ALLOWED tool: xDSL kit / CCA spine
- `merlin/python/merlin/runtime/commandbuffer.py` — ALLOWED tool: xDSL kit / CCA spine
- `merlin/python/merlin/runtime/tensor.py` — ALLOWED tool: xDSL kit / CCA spine
- `merlin/python/merlin/kernels/endpoints.py` — ALLOWED tool: xDSL kit / CCA spine
- `merlin/python/merlin/kernels/roles.py` — ALLOWED tool: xDSL kit / CCA spine
- `merlin/python/merlin/targetgen/synthesize/` — ALLOWED tool: xDSL kit / CCA spine
- `merlin/python/merlin/targetgen/generate/` — ALLOWED tool: xDSL kit / CCA spine
- `merlin/python/merlin/xdsl_dialects/` — ALLOWED tool: xDSL kit / CCA spine
- `merlin/python/merlin/targetgen/contract/interface_emit.py` — ALLOWED tool: xDSL kit / CCA spine
- `merlin/python/merlin/targetgen/contract/linalg_iface.py` — ALLOWED tool: xDSL kit / CCA spine
- `merlin/python/merlin/targetgen/oot_starterkit/` — ALLOWED tool: xDSL kit / CCA spine
- `merlin/python/merlin/kernels/cca.py` — ALLOWED tool: xDSL kit / CCA spine
- `merlin/python/merlin/kernels/cca_compare.py` — ALLOWED tool: xDSL kit / CCA spine
- `merlin/python/merlin/kernels/cca_contract.py` — ALLOWED tool: xDSL kit / CCA spine
- `merlin/python/merlin/kernels/action_catalog.py` — ALLOWED tool: xDSL kit / CCA spine
- `merlin/python/merlin/kernels/microkernel.py` — ALLOWED tool: xDSL kit / CCA spine
- `merlin/python/merlin/targetgen/rtl_backend.py` — ALLOWED tool: xDSL kit / CCA spine
- `merlin/python/merlin/targetgen/rtl/` — ALLOWED (CIRCT arm): RTL-facts generators
- `merlin/targets/gemmini/contracts/rtl_facts/` — ALLOWED (CIRCT arm): RTL-extracted facts
- `merlin/python/merlin/verify/` — ALLOWED (verify arm): the compiler-verification seam
- `merlin/python/merlin/xdsl_dialects/opt.py` — ALLOWED (verify arm): the compiler-verification seam

## Denied inputs

- `merlin/python/merlin/runtime/reference.py` — oracle-callable route
- `merlin/python/merlin/runtime/simulator.py` — oracle-callable route
- `merlin/python/merlin/targetgen/generate/runtime_adapter.py` — oracle-callable route
- `merlin/python/merlin/xdsl_dialects/lowering/` — oracle-callable route
- `out/artifacts/targets/gemmini/merlin_native_v0/` — prior backend / exemplar (answer surface)
- `out/artifacts/targets/gemmini/hand_smoke_oot/` — prior backend / exemplar (answer surface)
- `out/artifacts/targets/gemmini/agent_spec_v0_mlir_oot/` — prior backend / exemplar (answer surface)
- `out/artifacts/targets/gemmini/agent_spec_v1_mlir_oot/` — prior backend / exemplar (answer surface)
- `merlin/contract/capsules/hidden/` — hidden capsules + goldens
- `experiments/capsule_bench/targets/gemmini/input_bundles/grader_private_v0/` — grader-private
- `experiments/capsule_bench/targets/gemmini/runs/` — prior submissions
- `merlin/python/merlin/llvmlower/` — frozen host lane: the experiment measures the target lane, not a second CPU backend

## Submission boundary

The grants above may help author and debug the package. The submitted package must remain self-contained
and integrity-clean: it is graded only through its declared CLI entrypoints and may not import Merlin,
call an oracle/reference implementation, copy a prior backend or kernel, or embed expected outputs.
Denied paths and answer surfaces remain masked even when a broader parent directory is allowed.
