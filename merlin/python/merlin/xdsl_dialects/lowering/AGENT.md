# AGENT.md — merlin/python/merlin/xdsl_dialects/lowering

## Purpose

The staged lowering that proves the core dialects compose: linalg input → `contract` facts → `schedule` decisions → `interface` ops → target dialect (`toynpu`) → `runtime` command-buffer IR → executable command-buffer dict for the `merlin.runtime` engine.

Plus the **whole-model dialect plane** that brings real model2MLIR modules into this world: `outline.py` (`merlin-outline-dispatches`) splits the monolithic `func @forward` into one kernel func per compute dispatch + a driver, and `dispatch_program.py` (`merlin-emit-dispatch-program`) flattens that driver into a serializable runtime dispatch DAG. `passes.py` is the catalog of every Merlin-**authored** pass (vs the upstream passes orchestrated in `llvmlower/`) and the `run_dialect_plane` entry point.

## What belongs here

- One module per synthetic-workload stage (`contract_facts.py`, `schedule_decisions.py`, `interface_lowering.py`, `target_lowering.py`, `runtime_lowering.py`, `emit_command_buffer.py`), the workload builder (`input_workload.py`), cross-op analyses (`analyses.py`), and the orchestrator (`pipeline.py`).
- The whole-model dialect plane: `outline.py`, `dispatch_program.py`, `schedule_dispatch.py` (multicore partitioning), `passes.py`.

## What does not belong here

- Dialect definitions (those are the sibling modules).
- Runtime execution semantics (that is `merlin/python/merlin/runtime/`).
- Target-specific codegen beyond the dialect-plan-driven op mapping.

## Interfaces

- Consumes `merlin/targets/<t>/contracts/{target_contract,dialect_plan}.yaml`.
- Produces dicts conforming to `merlin/schemas/command_buffer.schema.yaml`, executed by `merlin.runtime.simulate` and checked against `reference_outputs`.

## Invariants

- **Outliner**: each compute op (`linalg.*` except `fill`/`yield`/`index`) becomes one kernel func; cheap pure producers (`arith.constant`/`tensor.empty`/`linalg.fill`) are *cloned into* the kernel; view/glue ops stay in the driver. Region-captured free values (model2MLIR `linalg.generic` gather bodies) MUST be lifted to kernel params — miss one and the kernel fails `IsolatedFromAbove`. The rewrite is value-preserving: inlining the kernels reproduces the original dataflow (proven bit-identical on host).
- **Dispatch program** is a DAG: every node input is an earlier node output or a model arg (`verify_program`). `prune_dead_nodes` drops the dead cloned-accumulator view copies. The whole thing is JSON-serializable (runtime ABI).
- Outlined kernel funcs are clean linalg — no `extract_slice` — so they round-trip through the xDSL printer (the whole model does not). This is what makes per-kernel compile + check work (`llvmlower/kernel_backend.py`).
- Stages are plain module→module transforms returning fresh/cloned modules; never mutate the caller's module. Every stage output must pass `module.verify()`.
- Local checks live in op `verify_`; anything needing program order or SSA chains lives in `analyses.py` and is run by the pipeline.
- The emitted command buffer must execute such that `simulate(cb) == reference_outputs(cb)` — residency must never change results.
- Deterministic tensor naming (`W`, `A0..`, `W_res`, `acc{i}`, `Y{i}`); same input → byte-identical command buffer.

## Testing expectations

`merlin/python/tests/test_xdsl_lowering_e2e.py` (full descent + execution + metric assertions) and the analysis cases in the per-dialect tests.

## Notes for future agents

To add a target: a reference dialect under `../targets/`, a lowering table in its `dialect_plan.yaml`, an opcode map in `runtime_lowering.py` (or a target-supplied encoding), and a backend in `merlin/python/merlin/runtime/`. Keep stages wrappable as xDSL `ModulePass`es — don't grow hidden state.
