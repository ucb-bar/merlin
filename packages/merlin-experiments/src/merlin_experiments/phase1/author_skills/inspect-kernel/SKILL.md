---
name: inspect-kernel
description: Inspect the compiled kernel of one case, meaning its lowered LLVM-dialect IR, rebuilt object, disassembly, stack-usage report and symbols, and relate a hot loop back to the task and schedule that produced it. Open it when a failure detail mentions the kernel stack-frame preflight, a link failure with a duplicate or missing symbol, or an object that does not build; when cycles concentrate in a loop you want to identify; or when you want to see the machine code your lowering became. Covers rebuilding kernel.o exactly as the grader does from selfcheck_out, the preflight rule, symbol checks, llvm-objdump, and a script that ranks loops by issued instructions and names their owning tasks.
---

# Inspect a compiled kernel

`SKILL` below is this skill's directory. Run commands from the workspace root. Build steps, file
formats and failure messages are in `reference/products.md`.

## 1. Start from the case's lowered IR

A self-check returns `lowered.llvm.mlir`, `instruction_trace.json` and `command_buffer.json` for
each case; the `inspect-artifacts` skill shows where. The grader keeps the object, the ELF and the
stack report it built, so rebuild them yourself. For a tree you have not checked yet, dump its
stages first with the `inspect-ir` skill.

## 2. Rebuild the object and run the preflight

```
python SKILL/scripts/build_kernel.py selfcheck_out/requests/RID/CASE      # or an inspect-ir dump dir
python SKILL/scripts/build_kernel.py DIR --also other.o                   # duplicate symbols across objects
```
The script repeats the grader's object step:
- `mlir-translate --mlir-to-llvmir`, then `clang -c -fstack-usage`;
- the triple and the `-march/-mabi/-mcmodel/-O` flags come from the `toolchain` block of `target_contract.yaml`, which also holds `kernel_stack_frame`;
- tools are found on PATH, then under `install/bin`.

It writes `kernel.ll`, `kernel.o`, `kernel.su`, `kernel.dis` and `report.json` to
`$TMPDIR/kernel-build/CASE`, and prints:
- `stack`: entry frame, budget, headroom, rows largest first, and `refusals` under the grader's rule (every row `static`, one row for the entry symbol, frame within `max_static_bytes`);
- `symbols`: external definitions, undefined references, and `defined_more_than_once`;
- `instructions_per_function`.

Exit 1 means the preflight refuses the object or a symbol is defined twice.

## 3. Fix what the preflight reports

- **Over budget.** The entry is keeping too much live: large local arrays or many spills. Move per-call scratch to a module-level global, or split the work into helper functions. Each helper's frame is measured separately and must also be `static`.
- **Dynamic allocation.** A variable-size or in-loop `llvm.alloca` gives a `dynamic` row. Hoist it to the entry block with a constant size.
- **Wrong number of entry rows.** Define the entry symbol exactly once, and give every function in the module a distinct name.

## 4. Fix link failures

The failure detail quotes the linker (`first defined here`, `undefined reference`); compare it with
`symbols.external_defined`. The harness links your object with its own startup, console and small
C-library units. Give every helper `internal` linkage (`llvm.func internal @name`) unless the ABI
needs it visible. For an undefined reference, emit the definition.

## 5. Map hot loops to tasks and the schedule

```
python SKILL/scripts/loop_map.py selfcheck_out/requests/RID/CASE --top 10
```
For each counted loop the script prints:
- its header `^bbN`, trip count and depth;
- its static instruction sites;
- an issued estimate: each site weighted by the trip counts of the loops around it;
- the decoded classes;
- the owners from `merlin.global_task` / `prov.region_id`, joined to `params.global_program_plan.tasks` or to `commands[N]`.

Uncounted loops are listed as notes. The estimate is static; measured cycles come from the verdict
row and the `read-command-buffer` skill.

To go from a loop to its schedule:
1. Take the task and command `loop_map.py` names.
2. Print the schedule decision your compiler made for that command (tile, loop order, unroll) with `dense_schedule_text` or `Kernel.text()`.
3. Check that the trip counts follow from it; extent divided by tile is the usual relation.

A loop whose trip count does not match is a lowering bug. A matching loop that dominates the work
is a scheduling opportunity; see the `scheduling-api` and performance skills. Tag every op you emit
with its task, so attribution stays complete.

## 6. Read machine code

`kernel.dis` is `llvm-objdump -d -r kernel.o`. Find the function, then the branch back to the loop
header. `python isa_tools.py disasm FILE` decodes the accelerator instructions in your lowered IR.
