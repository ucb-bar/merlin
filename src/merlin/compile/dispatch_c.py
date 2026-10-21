"""Emit a serial C driver for explicitly compiled dispatch implementations.

The driver consumes the existing program and checked arena offsets. It owns no
operator interpreter, device instructions, startup code or platform services.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

from merlin.runtime.program import MerlinProgram
from merlin.xdsl_dialects.lowering.arena_plan import check_bounded_arena
from merlin.xdsl_dialects.lowering.global_plan_emission import dispatch_digest

from .c_sources import c_source_pair


@dataclass(frozen=True)
class DispatchCall:
    """A compiled pointer-ABI wrapper that returns zero only after completion.

    ``placement`` names the selected execution endpoint, not output storage.
    A wrapper must preserve the source semantics and report failure directly.
    ``alias_safe`` explicitly admits the graph's declared overlapping views.
    ``context_index`` selects a slot in the emitted context table. With no slot,
    the wrapper receives the original context pointer as in the single-call ABI.
    """

    symbol: str
    placement: str
    completion: str = "synchronous"
    alias_safe: bool = False
    context_index: int | None = None


def _identifier(value: str) -> bool:
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_"
    return (
        isinstance(value, str)
        and bool(value)
        and value[0] in letters
        and all(char in letters + "0123456789" for char in value)
    )


def emit_program_c(
    program: MerlinProgram,
    name: str,
    calls: Mapping[int, DispatchCall],
    *,
    read_only_inputs: tuple[str, ...] = (),
) -> tuple[dict, dict[str, str]]:
    """Return a standalone C/H driver and its ordered buffer/call description.

    Every node, including support operations, needs an implementation binding.
    Caller-owned arguments/constants/results stay outside the temporary arena.
    The caller reserves and aligns that arena and supplies the wrapper context.
    This path is explicit; it does not change monolithic LLVM compilation.
    """
    if not _identifier(name) or not name[0].isalpha() or len(name) > 64:
        raise ValueError("driver name must be a C identifier beginning with a letter")
    graph, memory, requirements = program.dispatch, program.memory_plan, program.memory_requirements
    if requirements is None:
        raise ValueError("C dispatch export requires explicit bounded memory requirements")
    check_bounded_arena(graph, requirements, memory)
    if program.global_plan is not None and (
        program.global_plan_emission is None
        or program.global_plan.digest != program.global_plan_emission.plan_digest
        or dispatch_digest(graph) != dispatch_digest(program.global_plan_emission.dispatch)
    ):
        raise ValueError("a shadow or changed global plan cannot be exported")
    if any(item.get("mode", "none") != "none" for item in program.parallel.values()):
        raise ValueError("C dispatch export requires serial execution")
    if set(calls) != set(range(len(graph.nodes))) or any(type(i) is not int for i in calls):
        raise ValueError("every emitted node needs exactly one compiled implementation")
    for call in calls.values():
        if (
            not _identifier(call.symbol)
            or not isinstance(call.placement, str)
            or not call.placement.strip()
            or call.completion != "synchronous"
            or type(call.alias_safe) is not bool
            or (call.context_index is not None and (type(call.context_index) is not int or call.context_index < 0))
        ):
            raise ValueError("compiled calls need valid symbols, placement and synchronous completion")
    if any(key != buf.id or buf.kind not in ("arg", "const", "intermediate") for key, buf in graph.buffers.items()):
        raise ValueError("invalid dispatch buffer table")
    args = [key for key, buf in graph.buffers.items() if buf.kind == "arg"]
    indices = [graph.buffers[key].arg_index for key in args]
    if (
        any(type(i) is not int or i < 0 for i in indices)
        or len(set(indices)) != len(indices)
        or len(set(graph.args)) != len(graph.args)
        or set(indices) != set(graph.args)
    ):
        raise ValueError("argument indices do not match the dispatch signature")
    args.sort(key=lambda key: graph.args.index(graph.buffers[key].arg_index))
    if (not isinstance(read_only_inputs, tuple) or len(set(read_only_inputs)) != len(read_only_inputs)
            or not set(read_only_inputs) <= set(args)):
        raise ValueError("read-only input bindings must name distinct source arguments")
    external = list(
        dict.fromkeys(
            [
                *args,
                *(key for key, buf in graph.buffers.items() if buf.kind == "const"),
                *graph.results,
            ]
        )
    )
    placements = {}
    if program.global_plan_emission is not None and program.global_plan is not None:
        selected = {item.id: item for item in program.global_plan.selected}
        for component in program.global_plan_emission.regions:
            for index in component.node_indices:
                placements[index] = selected[component.plan_id].placement
    defined = {key for key, buf in graph.buffers.items() if buf.kind in ("arg", "const")}
    for index, node in enumerate(graph.nodes):
        if node.kind not in ("dispatch", "view") or any(key not in defined for key in node.inputs):
            raise ValueError("unsupported node or input without a producer")
        for key in node.outputs:
            if key not in graph.buffers or key in defined:
                raise ValueError("dispatch outputs need unique non-external definitions")
            defined.add(key)
        if index in placements and calls[index].placement != placements[index]:
            raise ValueError("compiled implementation differs from selected placement")
        if not calls[index].alias_safe:
            for source in node.inputs:
                for target in node.outputs:
                    if (
                        source in memory.offsets
                        and target in memory.offsets
                        and (
                            memory.offsets[source] < memory.offsets[target] + memory.sizes[target]
                            and memory.offsets[target] < memory.offsets[source] + memory.sizes[source]
                        )
                    ):
                        raise ValueError("overlapping view needs an alias-safe compiled implementation")
    if not set(graph.results) <= defined:
        raise ValueError("missing program result")
    slots = {key: index for index, key in enumerate(external)}

    def pointer(key):
        if key in slots:
            return f"buffers[{slots[key]}].data"
        return f"((unsigned char *)arena + {memory.offsets[key]}u)"

    upper = name.upper()
    context_count = max(
        (call.context_index + 1 for call in calls.values() if call.context_index is not None),
        default=0,
    )
    header = f"""#include <stddef.h>
#include <stdint.h>
typedef struct {{ void *data; size_t bytes; }} {name}_buffer;
typedef struct {{ int active; size_t completed; int kernel_status; }} {name}_state;
enum {{ {upper}_OK = 0, {upper}_INVALID = -1,
       {upper}_BUSY = -2, {upper}_KERNEL_ERROR = -3 }};
#define {upper}_ARENA_BYTES {memory.arena_bytes}u
#define {upper}_ARENA_ALIGNMENT {requirements.alignment}u
#define {upper}_BUFFER_COUNT {len(external)}u
"""
    if context_count:
        header += f"""
/* Pass this table as context when calls select distinct compiled regions. */
typedef struct {{ void *const *items; size_t count; }} {name}_contexts;
#define {upper}_CONTEXT_COUNT {context_count}u
"""
    declarations = f"""
/* Zero-initialize state. Buffers follow the manifest order, not SSA name order.
 * On a kernel error state stays active: drain/reset the endpoint before clearing
 * state or reusing its buffers. Outputs may be partial after a kernel error. */
int {name}_run({name}_state *state, void *arena, size_t arena_bytes,
               const {name}_buffer *buffers, size_t buffer_count, void *context);
"""
    body = ["#include <stdint.h>"]
    for symbol in sorted({call.symbol for call in calls.values()}):
        if symbol == name + "_run":
            raise ValueError("kernel symbol conflicts with generated entry")
        body.append(f"extern int {symbol}(void *, const void *const *, void *const *);")
    body.append(f"""
int {name}_run({name}_state *state, void *arena, size_t arena_bytes,
               const {name}_buffer *buffers, size_t buffer_count, void *context) {{
  if (!state) return {upper}_INVALID;
  if (state->active) return {upper}_BUSY;
  if (buffer_count != {len(external)}u || (buffer_count && !buffers)) return {upper}_INVALID;
  (void)context;
  uintptr_t begin = (uintptr_t)arena;
  uintptr_t state_begin = (uintptr_t)state;
  if (state_begin > UINTPTR_MAX - sizeof(*state)) return {upper}_INVALID;
  (void)arena_bytes; (void)begin;
""")
    if context_count:
        body.append(f"""  const {name}_contexts *contexts = context;
  if (!contexts || !contexts->items || contexts->count < {context_count}u) return {upper}_INVALID;
""")
    if memory.arena_bytes:
        body.append(f"""  if (arena_bytes < {memory.arena_bytes}u || !begin ||
      begin % {requirements.alignment}u || begin > UINTPTR_MAX - {memory.arena_bytes}u)
    return {upper}_INVALID;
  if (state_begin < begin + {memory.arena_bytes}u &&
      begin < state_begin + sizeof(*state)) return {upper}_INVALID;
""")
    for index, key in enumerate(external):
        body.append(f"""  if (!buffers[{index}].data || buffers[{index}].bytes < {memory.sizes[key]}u ||
      (uintptr_t)buffers[{index}].data > UINTPTR_MAX - {memory.sizes[key]}u) return {upper}_INVALID;
  if (state_begin < (uintptr_t)buffers[{index}].data + {memory.sizes[key]}u &&
      (uintptr_t)buffers[{index}].data < state_begin + sizeof(*state)) return {upper}_INVALID;
""")
        if memory.arena_bytes:
            body.append(f"""  if ((uintptr_t)buffers[{index}].data < begin + {memory.arena_bytes}u &&
      begin < (uintptr_t)buffers[{index}].data + {memory.sizes[key]}u) return {upper}_INVALID;
""")
        for other in range(index):
            if key in read_only_inputs and external[other] in read_only_inputs:
                continue
            other_size = memory.sizes[external[other]]
            body.append(f"""  if ((uintptr_t)buffers[{index}].data <
      (uintptr_t)buffers[{other}].data + {other_size}u &&
      (uintptr_t)buffers[{other}].data < (uintptr_t)buffers[{index}].data + {memory.sizes[key]}u)
    return {upper}_INVALID;
""")
    body.append("  state->active = 1; state->completed = 0; state->kernel_status = 0;")
    for index, node in enumerate(graph.nodes):
        inputs = ", ".join(pointer(key) for key in node.inputs) or "0"
        outputs = ", ".join(pointer(key) for key in node.outputs) or "0"
        context = "context" if calls[index].context_index is None else f"contexts->items[{calls[index].context_index}]"
        body.append(f"""  {{
    const void *inputs[{max(1, len(node.inputs))}] = {{{inputs}}};
    void *outputs[{max(1, len(node.outputs))}] = {{{outputs}}};
    state->kernel_status = {calls[index].symbol}({context}, inputs, outputs);
    if (state->kernel_status) return {upper}_KERNEL_ERROR;
    state->completed = {index + 1}u;
  }}""")
    body.append(f"  state->active = 0; return {upper}_OK;\n}}")
    files = c_source_pair(
        name,
        public_headers={"dispatch.h": header},
        private_headers={},
        sources={"dispatch.c": "\n".join(body)},
        declarations=declarations,
    )
    description = {
        "schema": "merlin.dispatch_c.v1",
        "entry": name + "_run",
        "execution": "serial",
        "dispatch_digest": dispatch_digest(graph),
        "memory_plan": memory.to_dict(),
        "global_plan_emission": program.global_plan_emission.receipt() if program.global_plan_emission else None,
        "buffers": [
            {
                "id": key,
                "bytes": memory.sizes[key],
                "dtype": graph.buffers[key].dtype,
                "shape": graph.buffers[key].shape,
            }
            for key in external
        ],
        "inputs": args,
        "outputs": list(graph.results),
        "calls": [
            {
                "node": index,
                **{k: v for k, v in vars(calls[index]).items() if k != "context_index" or v is not None},
                "inputs": node.inputs,
                "outputs": node.outputs,
            }
            for index, node in enumerate(graph.nodes)
        ],
        "files": {member: hashlib.sha256(text.encode()).hexdigest() for member, text in files.items()},
        "linked": False,
        "executed": False,
    }
    if context_count:
        description["context_table"] = {"type": name + "_contexts", "minimum_count": context_count}
    if read_only_inputs:
        description["read_only_inputs"] = list(read_only_inputs)
    # Validate JSON serializability before returning any generated source.
    json.dumps(description, sort_keys=True)
    return description, files
