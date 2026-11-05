# IR stages, printers and contracts

## How the grader runs the opt tool

These rules come from `merlin.targetgen.package_runtime` (`load_package`, `_resolve_argv`,
`run_entrypoint`, `analysis_emission_entrypoints`) and `merlin.targetgen.capsule_common.lower_interface`.

- **Tool path.** It is `build.tool_output` if the manifest has a `build` block, else
  `entrypoints.tool`. Either is resolved against the package directory (`submission/` or
  `variants/NAME/`).
- **Substitution.** Only `{tool}`, `{input_mlir}` and `{output_json}` are substituted, all as
  absolute paths. Any other `{...}` placeholder is refused before the tool runs.
- **Working directory and environment.** Every command runs with the package root as its working
  directory and `PYTHONDONTWRITEBYTECODE=1`.
- **Interpreter.** A Python tool without the execute bit is run through the interpreter.
- **Order of commands:**
  1. The interface is copied to `generated/input.interface.mlir`.
  2. `parse` runs. A nonzero exit is a tool crash.
  3. `lower_interface_to_target` runs. It must exit 0 with non-empty stdout, which is kept as
     `lowered.target.mlir`.
  4. The final commands depend on what the manifest declares.
     - With `emit_analysis_bundle` declared, that one process writes `{output_json}` and prints
       the artifact.
     - Without it, `emit_command_buffer` writes `{output_json}`, and then `lower_target_to_llvm`
       prints the artifact. `emit_target_artifact` is an accepted alias.
- **What is kept.** The command buffer is schema-checked, and a `declined` entry needs a non-empty
  `reason`. The artifact is kept as `lowered.llvm.mlir`.

`dump_stages.py run` follows this procedure, with your own `--append` arguments at the end of every
argv.

## The stages you can print

| Stage | Where it comes from | How to read or print it |
|---|---|---|
| interface input | the case's `capsule.interface.mlir` | `merlin.targetgen.contract.interface_emit.parse_interface_mlir(text)` gives tensors and commands; `undefined_op_mnemonics(text)` lists ops the grammar lacks |
| linalg-on-tensors source | `public_sources/*.mlir`, model cases | `merlin.targetgen.contract.linalg_iface`: `is_linalg_on_tensors(text)`, `parse_linalg_mlir(text)` gives `{level, entry, args, results, returns, ops[]}`; `matmul_records(parsed)`; `make_linalg_context()` |
| dense contraction IR | your planner | `merlin.transforms.dense_ir.DenseScheduleOp` (`schedule.block_contraction`) and `dense_schedule_op(...)`; `merlin.transforms.dense_planning` (`dense_blocks`, `choose_dense_block`, `verify_dense_emission`) |
| schedule plan | your scheduler | `merlin.compile.scheduling.schedule_ir.dense_schedule_text(plan)` prints it; `parse_dense_schedule_plan(text)` parses it back |
| loop-nest kernel | `merlin.sched.ir` | `Kernel.text()`, `Kernel.digest()`; `merlin.sched.ir.parse.parse_kernel(text)`; `merlin.sched.ir.kernel.check_structure(kernel)` gives the problems found |
| any xDSL module | your passes | `merlin.xdsl_dialects._common.make_context(*dialects)`, `text(module, generic=True)` (generic form keeps every attribute); `merlin.xdsl_dialects.ir_inspection.compact_text(module, elements_limit=64)` elides large constants |
| target IR | stdout of `lower_interface_to_target` | parse with an xDSL `Context(allow_unregistered=True)` plus the upstream dialects, or with your own dialect loaded |
| lowered artifact | stdout of the artifact command | see the `inspect-kernel` skill |

A diff between two generic-form prints compares every attribute. A diff between two custom-form
prints hides some of them, so use `generic=True` when bisecting.

## Contracts

- **`interface_dialect_contract.yaml`.**
  - `dialect.namespace` names the dialect. `dialect.module_attributes` lists the attributes,
    including `must_equal` for the version.
  - `dialect.types` lists the types. `dialect.required_ops` gives each op's operands, result,
    attrs and `maps_to`, the command it becomes.
  - `acceptance` requires every example to parse and requires unknown versions and mnemonics to be
    rejected.
- **`interface_grammar.md`.** This is the prose grammar, with a worked example and the attribute
  encoding.
- **`target_dialect_contract.yaml`.**
  - `target_dialect` holds a recommended namespace, ops and types. It is a recommendation, not a
    gate.
  - `required_outputs` lists the three products. The target IR must be non-empty and parseable,
    the command buffer must be schema-valid and agree with simulation, and the lowered artifact
    must compile to an object.
- **`mlir_oot_backend_contract.yaml`.**
  - `lowered_ir_task` carries each plan `task_index` as `merlin.global_task` on the operations that
    task owns.
  - `device_command_source` attributes device work through `merlin.source_op_index` and
    `prov.region_id`.
  - `host_compute_boundary` says what host code may compute inside a device task.

`dump_stages.py check` covers the interface attributes and op names, the target IR's non-empty
parse, the command buffer's JSON and declined reason, and the lowered module having a function.
Schema validity and numerical agreement need a self-check.

## Bisect checklist

1. Get two dumps of the same case, good and bad, and run `compare`. It names the first stage that
   differs.
2. Check whether the input is read right. Compare `parse_interface_mlir(input)["commands"]` with
   your `command_buffer.json` `commands`: opcode, operands, and attributes such as layout, stride,
   padding and epilogue.
3. Check whether the buffer is right but the emission wrong. Run `isa_tools.py debug` on the
   buffer; it shows per-command state and withholds the output values. Then run
   `isa_tools.py disasm lowered.llvm.mlir`, which decodes what you actually emitted. Compare the
   two.
4. Split your own pipeline. Print it after half its passes in both runs, and keep the half where
   they first diverge. Run `module.verify()` after each pass; the first one that fails to verify is
   the culprit.
5. Shrink the case. A smaller case of the same family with the same failure, among the public
   isa, layers or model-slice cases, is cheaper to bisect.
