# Kernel products, the object build and its gates

## What the grader builds from your lowered IR

The steps below come from `merlin.targetgen.contract.compile` (the object and link steps),
`merlin.targetgen.contract.stack_usage` (the preflight) and `merlin.llvmlower.codegen` (the compiler
call). All files land in the case's `generated/` directory.

| Step | Input to output | Notes |
|---|---|---|
| copy | `lowered.llvm.mlir` to `kernel.llvm.mlir` | the bytes your artifact command printed |
| translate | `mlir-translate --mlir-to-llvmir` to `kernel.ll` | an LLVM-dialect module that does not translate fails here |
| compile | `clang --target=TRIPLE -march -mabi -mcmodel -O2 -ffreestanding -fno-builtin -fstack-usage -c kernel.ll -o kernel.o` | march and mabi come from the target's build recipe, the `toolchain.cflags` in `target_contract.yaml` |
| preflight | `kernel.su` to `kernel.stack_frame.json` | the rule is below; a failure triggers one generic stack repair, and if the repair also fails the refusal stands |
| ABI receipt | to `kernel.abi.json` | the object digest and the ABI it was built for |
| link | `harness.c` + `kernel.o` + the runtime units to `package_kernel.elf` | the runtime units are listed in `toolchain.runtime_units`, the linker script in `toolchain.link_script` |

`build_kernel.py` repeats translate, compile and preflight with the same flags and keeps every file
in a scratch directory. It does not link: the harness source and runtime units are not staged.

## Stack-frame preflight rule

The policy comes from `toolchain.kernel_stack_frame` in `target_contract.yaml`:
`entry_symbol` and `max_static_bytes`.

Each line of `kernel.su` is `SOURCE:FUNCTION<TAB>BYTES<TAB>KIND`, with KIND one of `static`,
`dynamic` or `dynamic,bounded`. The grader refuses the object when any of these holds:

- a row names a source other than `kernel.ll`;
- any row in the unit is not `static` (`contains dynamic allocation for: ...`);
- the entry symbol does not have exactly one row (`contains N rows for entrypoint 'ENTRY'; exactly one is required`);
- the entry frame exceeds the budget (`entrypoint 'ENTRY' needs N static stack bytes, exceeding the target-declared B-byte budget by D bytes`).

The author-facing detail reports this as `kernel stack-frame preflight failed: ...`, with digits
masked. `build_kernel.py` prints the same refusals, unmasked.

## Link failures

A link error reaches you as a failure detail containing `link failed:` and the linker's tail. That
tail names the object (`generated/kernel.o`), the section, and `first defined here` or
`undefined reference`. Common causes:

- You define a symbol that the runtime units also define. The common case is a C-library or
  startup name you emitted as an external helper. Make the helper `internal`, or rename it.
- Two of your functions share a name. That can happen when one lowering emits a helper per use
  instead of once.
- You call a function nothing defines.

`build_kernel.py` lists `external_defined` and `undefined`. `--also OBJ` adds other objects and
reports every external symbol defined more than once.

## Lowered IR conventions the tools rely on

- Accelerator instructions are `llvm.inline_asm` ops. `instruction_trace.json` decodes them in text
  order, one row per site: `index`, `class`, `funct`, `rs1`/`rs2` (`raw`, `kind`, `arg_index`,
  `offset`) and `decoded`.
- `instruction_trace.json` `summary.counted_loops` describes the loops, with basis
  `original_llvm_cfg_constant_induction_v1`. Each function lists `loops[]`, with fields
  `header_block`, `latch_block`, `body_blocks`, `initial`, `bound`, `step`, `trip_count`,
  `predicate` and `induction_argument_index`, and `unavailable[]` for loops whose bounds are not
  constant. Block numbers are positions in the function: the entry block is 0 and the block
  printed `^bbN` is N.
- Ownership attributes are written by your emitter and read by the grader's audits:
  - `merlin.global_task` is the plan `task_index` (`lowered_ir_task` in
    `mlir_oot_backend_contract.yaml`).
  - `merlin.source_op_index` and `prov.region_id` attribute device work to its source
    (`device_command_source`).
  - Untagged arithmetic is not attributable host work.
- A latch can carry `loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>`, which
  keeps clang from unrolling a loop your schedule already shaped.

## Tools on the author's PATH

- **LLVM tools.** `mlir-translate`, `clang`, `llc`, `llvm-mc`, `llvm-objdump`, `llvm-nm`,
  `llvm-readelf`, `llvm-size` and `llvm-symbolizer` all come from the granted LLVM toolchain. It is
  also linked as `install/bin` in the workspace.
- **`isa_tools.py`.**
  - `disasm FILE` decodes the accelerator instructions in a lowered module.
  - `lint FILE` reports illegal or undeclared instructions, class coverage and unused declared
    instructions.
  - `debug CB --capsule CASE` runs a command buffer on the derived model; see the
    `read-command-buffer` skill.

## Relating a loop to its schedule

The lowered IR carries no schedule id. The join goes through ownership:

1. The loop's blocks give the `merlin.global_task` value.
2. That task is a plan task, or a command index if your compiler numbers tasks that way.
3. The task leads to the schedule decision your compiler recorded for it.

Record the decision where you can print it. Examples are a `DenseScheduleDecision.plan_text` /
`plan_sha256`, or the `Kernel.digest()` of a loop nest. Then compare its extents, tiles and unroll
factors with the counted trip counts. If one decision produces several loops, their trip counts
multiply to the iteration count of that decision's tile loop nest.
