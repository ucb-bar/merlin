"""Diagnostic task-entry counters for already emitted, grouped LLVM CFGs."""

from __future__ import annotations

import copy
import hashlib
import re
from collections.abc import Mapping

from xdsl.parser import Parser

from merlin.common.ir_lock import IR_LOCK
from merlin.perf.host_cfg_index import prepare_host_cfg
from merlin.perf.task_cfg_evidence import analyze_task_cfg
from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context


def _owner(operation):
    attribute = operation.attributes.get("merlin.global_task")
    return getattr(getattr(attribute, "value", None), "data", None)


def _insert_markers(text, function, markers, prefix):
    """Keep original LLVM bytes: the stock and xDSL custom printers can differ."""
    lines = text.splitlines(keepends=True)
    tagged = [(op.name, _owner(op)) for block in function.body.blocks for op in block.ops if _owner(op) is not None]
    observed, first_lines = [], {}
    tag = re.compile(r"merlin\.global_task\s*=\s*(-?\d+)\s*:\s*i64")
    operation = re.compile(r'^\s*(?:%[^=\n]+=\s*)?(?:"(llvm\.[^"]+)"|(llvm\.[\w.]+))')
    for index, line in enumerate(lines):
        owners = tag.findall(line)
        if not owners:
            continue
        match = operation.match(line)
        if len(owners) != 1 or match is None:
            raise ValueError("profiling requires one complete tagged LLVM operation per line")
        owner = int(owners[0])
        observed.append((match[1] or match[2], owner))
        if owner >= 0:
            first_lines.setdefault(owner, index)
    if observed != tagged:
        raise ValueError("text task tags differ from the independently parsed LLVM operation roster")
    if "%" + prefix + "_task_" in text:
        raise ValueError("profiling SSA names overlap the original program")
    entry_lines = [index for index, line in enumerate(lines)
                   if re.match(r"\s*llvm\.func\s+@" + re.escape(function.sym_name.data) + r"\(", line)]
    if len(entry_lines) != 1:
        raise ValueError("profiling requires an explicit one-line LLVM entry declaration")
    insertions = {entry_lines[0]: f"  llvm.func @{prefix}_mark(i64)\n"}
    for marker in markers:
        task_id = marker["task_id"]
        insertions[first_lines[task_id]] = (
            f"    %{prefix}_task_{task_id} = llvm.mlir.constant({task_id}) : i64\n"
            f"    llvm.call @{prefix}_mark(%{prefix}_task_{task_id}) "
            f"{{merlin.profile_marker = {task_id} : i64}} : (i64) -> ()\n"
        )
    return "".join(insertions.get(index, "") + line for index, line in enumerate(lines))


def _support(prefix, count):
    return f'''#include <stdint.h>
#include "htif.h"
static volatile uint64_t {prefix}_cycles[{count + 1}] __attribute__((aligned(64)));
static volatile uint64_t {prefix}_instructions[{count + 1}] __attribute__((aligned(64)));
static volatile uint64_t {prefix}_hits[{count + 1}] __attribute__((aligned(64)));
static volatile uint64_t {prefix}_invalid __attribute__((aligned(64)));
__attribute__((noinline)) void {prefix}_mark(uint64_t task_id) {{
  uint64_t cycles, instructions;
  __asm__ volatile("csrr %0, mcycle\\n\\tcsrr %1, minstret"
                   : "=r"(cycles), "=r"(instructions) : : "memory");
  if (task_id < UINT64_C({count + 1})) {{
    {prefix}_cycles[task_id] = cycles;
    {prefix}_instructions[task_id] = instructions;
    {prefix}_hits[task_id]++;
  }} else {{ {prefix}_invalid++; }}
  __asm__ volatile("" : : : "memory");
}}
void {prefix}_dump(void) {{
  for (uint64_t task_id=0; task_id<UINT64_C({count + 1}); task_id++) {{
    htif_puts("TASK_PROFILE id="); htif_putd((long)task_id);
    htif_puts(" mcycle="); htif_putd((long){prefix}_cycles[task_id]);
    htif_puts(" minstret="); htif_putd((long){prefix}_instructions[task_id]);
    htif_puts(" hits="); htif_putd((long){prefix}_hits[task_id]);
    htif_puts("\\n");
  }}
  htif_puts("TASK_PROFILE invalid="); htif_putd((long){prefix}_invalid); htif_puts("\\n");
}}
'''


def instrument_task_windows(
    lowered_mlir_text: str,
    command_buffer: Mapping,
    *,
    entry_symbol: str = "kernel_entry",
    symbol_prefix: str = "merlin_profile",
) -> dict:
    """Insert one entry call per exact task, or refuse unsupported task CFGs.

    The first static task operation must dominate all operations of that task
    and all returns, and its block must belong to no directed CFG cycle. Thus
    a nested task loop cannot execute the marker repeatedly. Original operations,
    attributes, operands and branches are preserved. The caller adds the final
    mark immediately after its existing completion call, then dumps outside its
    timing window. These instrumented windows include counter-hook perturbation.
    """
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", symbol_prefix):
        raise ValueError("profiling symbol prefix must be a C/LLVM identifier")
    plan = (command_buffer.get("params") or {}).get("global_program_plan")
    tasks = plan.get("tasks") if isinstance(plan, Mapping) else None
    if not isinstance(tasks, list) or not tasks or len(tasks) > 1024:
        raise ValueError("profiling needs 1..1024 explicitly declared global tasks")
    ids = [row.get("task_index") for row in tasks if isinstance(row, Mapping)]
    if ids != list(range(len(tasks))) or any(type(value) is not int for value in ids):
        raise ValueError("profiling needs exactly ordered contiguous task IDs")
    with IR_LOCK:
        module = Parser(make_llvm_context(), lowered_mlir_text).parse_module()
        module.verify()
        functions = [op for op in module.body.block.ops if op.name == "llvm.func" and op.body.blocks]
        if len(functions) != 1 or functions[0].sym_name.data != entry_symbol:
            raise ValueError("profiling needs one defined selected LLVM entry")
        if any(getattr(op.properties.get("sym_name"), "data", "").startswith(symbol_prefix)
               for op in module.body.block.ops):
            raise ValueError("profiling symbols overlap the original program")
        function = functions[0]
        cfg = prepare_host_cfg(function)
        evidence = analyze_task_cfg(function, ids, prepared_cfg=cfg)
        if evidence["problems"]:
            raise ValueError("unsupported task CFG: " + "; ".join(evidence["problems"]))
        tagged = [op for block in cfg.blocks for op in block.ops if _owner(op) is not None and _owner(op) >= 0]
        observed_order = []
        for op in tagged:
            if not observed_order or observed_order[-1] != _owner(op):
                observed_order.append(_owner(op))
        if observed_order != ids:
            raise ValueError("task operations are missing, noncontiguous or disjoint")
        markers = []
        for task_id in ids:
            operations = [op for op in tagged if _owner(op) == task_id]
            first = operations[0]
            block = first.parent_block()
            pending, seen = list(cfg.successors[block]), set()
            while pending:
                successor = pending.pop()
                if successor is block:
                    raise ValueError(f"task {task_id} entry marker would repeat in a CFG loop")
                if successor not in seen:
                    seen.add(successor)
                    pending.extend(cfg.successors[successor])
            if any(not cfg.dominance.dominates(block, op.parent_block()) for op in operations):
                raise ValueError(f"task {task_id} first operation does not dominate its complete task")
            if any(not cfg.dominance.dominates(block, end)
                   for end in cfg.blocks if end.last_op.name == "llvm.return"):
                raise ValueError(f"task {task_id} entry marker is bypassable")
            markers.append(dict(task_id=task_id, block_index=cfg.blocks.index(block), first_operation=first.name))
        instrumented = _insert_markers(lowered_mlir_text, function, markers, symbol_prefix)
        Parser(make_llvm_context(), instrumented).parse_module().verify()
    return dict(
        lowered_mlir_text=instrumented,
        c_support=_support(symbol_prefix, len(tasks)),
        completion_statement=f"{symbol_prefix}_mark({len(tasks)});",
        dump_statement=f"{symbol_prefix}_dump();",
        manifest=dict(
            schema="merlin.task_window_profile.v1",
            qualification="UNQUALIFIED",
            symbol_prefix=symbol_prefix,
            unprofiled_lowered_sha256=hashlib.sha256(lowered_mlir_text.encode()).hexdigest(),
            instrumented_lowered_sha256=hashlib.sha256(instrumented.encode()).hexdigest(),
            tasks=copy.deepcopy(tasks),
            markers=markers,
            completion_marker=len(tasks),
            once_per_entry="marker block acyclic; dominates all task operations and returns",
            scope="instrumented CPU mcycle/minstret windows; includes profiling overhead; source effects pending",
        ),
    )
