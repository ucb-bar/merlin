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


@dataclass(frozen=True)
class DispatchLoop:
    """An optional counted window authenticated against the entire flat driver.

    This only compresses the schedule: each iteration retains its original
    wrapper, placement, context and pointers. It implies no body equivalence,
    parameter substitution, allocation change or operand residency.
    """

    offset: int
    period: int
    trip_count: int
    authority_sha256: str


def _loop_authority(program, calls):
    record = dict(
        dispatch=dispatch_digest(program.dispatch),
        memory=program.memory_plan.to_dict(),
        plan=program.global_plan_emission.receipt() if program.global_plan_emission else None,
        calls=[dict(node=i, **vars(calls[i])) for i in range(len(program.dispatch.nodes))],
    )
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prepare_dispatch_loop(program, calls, *, offset, period, trip_count):
    """Capture current source, selected plan, memory and ordered call bindings."""
    if (any(type(v) is not int for v in (offset, period, trip_count))
            or offset < 0 or period < 1 or trip_count < 2
            or offset + period * trip_count > len(program.dispatch.nodes)):
        raise ValueError("invalid dispatch loop bounds")
    return DispatchLoop(offset, period, trip_count, _loop_authority(program, calls))


def prepare_dispatch_loop_from_plan(program, calls, record, *, requests, source_request_digest):
    """Bind all trips of a current application loop plan to original C calls.

    The analysis signature only locates a window. Exact request expansion and
    the current dispatch, selected plan, memory and call table remain authority.
    """
    from .region_loop import expand_region_loop

    if (not isinstance(record, dict)
            or record.get("schema") != "merlin.application_loop_plan.v1"
            or record.get("source_request_digest") != source_request_digest):
        raise ValueError("application loop plan differs from current source")
    region_loop = record.get("region_loop")
    if (not isinstance(region_loop, dict)
            or region_loop.get("schema") != "merlin.region_loop.v1"
            or region_loop.get("mode") != "analysis_only"):
        raise ValueError("application loop plan has no repeated region")
    requests = list(requests)
    if expand_region_loop(region_loop) != requests or len(requests) != len(program.dispatch.nodes):
        raise ValueError("application loop plan differs from current region requests")
    return prepare_dispatch_loop(
        program, calls, offset=region_loop["offset"], period=region_loop["period"],
        trip_count=region_loop["trip_count"],
    )


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
    loop: DispatchLoop | None = None,
) -> tuple[dict, dict[str, str]]:
    """Return a standalone C/H driver and its ordered buffer/call description.

    Every node, including support operations, needs an implementation binding.
    Caller-owned arguments/constants/results stay outside the temporary arena.
    The caller reserves and aligns that arena and supplies the wrapper context.
    This path is explicit; it does not change monolithic LLVM compilation.
    With an authenticated ``loop``, the additive ``_run_loop`` entry first asks
    a target admission hook, then runs the original binding table. The ordinary
    ``_run`` entry uses flat order. No target hook runs after dispatch begins.
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
    if (
        not isinstance(read_only_inputs, tuple)
        or len(set(read_only_inputs)) != len(read_only_inputs)
        or not set(read_only_inputs) <= set(args)
    ):
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
    if loop is not None:
        if not isinstance(loop, DispatchLoop) or loop != prepare_dispatch_loop(
            program, calls, offset=loop.offset, period=loop.period, trip_count=loop.trip_count
        ):
            raise ValueError("dispatch loop differs from current source, plan or bindings")

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
    if loop is not None:
        header += f"""
typedef int (*{name}_wrapper)(void *, const void *const *, void *const *);
typedef struct {{
  {name}_wrapper function;
  void *context;
  const void *const *inputs;
  void *const *outputs;
  size_t input_count, output_count;
}} {name}_call;
typedef struct {{
  const {name}_call *calls;
  size_t count, offset, period, trip_count;
}} {name}_loop;
/* Admission only, before any call: return 1 to admit or 0 for flat fallback.
 * The hook must not dispatch, mutate bindings or retain this invocation's table. */
typedef int (*{name}_loop_hook)(void *, const {name}_loop *);
"""
        declarations += f"""
int {name}_run_loop({name}_state *state, void *arena, size_t arena_bytes,
                    const {name}_buffer *buffers, size_t buffer_count, void *context,
                    {name}_loop_hook hook, void *hook_context);
"""
    body = ["#include <stdint.h>"]
    if loop is not None:
        body.append("#include <stdlib.h>")
    for symbol in sorted({call.symbol for call in calls.values()}):
        if symbol == name + "_run" or (loop is not None and symbol == name + "_run_loop"):
            raise ValueError("kernel symbol conflicts with generated entry")
        body.append(f"extern int {symbol}(void *, const void *const *, void *const *);")
    signature = f"""int {name}_run({name}_state *state, void *arena, size_t arena_bytes,
               const {name}_buffer *buffers, size_t buffer_count, void *context)"""
    if loop is not None:
        signature = f"""int {name}_run_loop({name}_state *state, void *arena, size_t arena_bytes,
               const {name}_buffer *buffers, size_t buffer_count, void *context,
               {name}_loop_hook hook, void *hook_context)"""
    body.append(f"""
{signature} {{
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
    if external:
        sizes = ",".join(f"{memory.sizes[key]}u" for key in external)
        readonly = ",".join(str(int(key in read_only_inputs)) for key in external)
        # Keep validation bounded in generated source size. Expanding every
        # buffer pair makes ordinary host optimization quadratic in IR size.
        body.append(f"""  static const size_t required[{len(external)}] = {{{sizes}}};
  static const unsigned char readonly[{len(external)}] = {{{readonly}}};
  for (size_t i=0; i<{len(external)}u; ++i) {{
    uintptr_t address=(uintptr_t)buffers[i].data;
    if (!buffers[i].data || buffers[i].bytes<required[i] || address>UINTPTR_MAX-required[i])
      return {upper}_INVALID;
    if (state_begin<address+required[i] && address<state_begin+sizeof(*state))
      return {upper}_INVALID;
""")
        if memory.arena_bytes:
            body.append(f"""    if (address<begin+{memory.arena_bytes}u && begin<address+required[i])
      return {upper}_INVALID;
""")
        body.append(f"""    for (size_t j=0; j<i; ++j) {{
      if (readonly[i] && readonly[j]) continue;
      uintptr_t other=(uintptr_t)buffers[j].data;
      if (address<other+required[j] && other<address+required[i]) return {upper}_INVALID;
    }}
  }}
""")
    if loop is None:
        body.append("  state->active = 1; state->completed = 0; state->kernel_status = 0;")
    else:
        input_count = sum(max(1, len(node.inputs)) for node in graph.nodes)
        output_count = sum(max(1, len(node.outputs)) for node in graph.nodes)
        body.append(f"""  struct {{
    {name}_call calls[{len(graph.nodes)}];
    const void *inputs[{input_count}];
    void *outputs[{output_count}];
  }} *bindings = malloc(sizeof(*bindings));
  if (!bindings) return {upper}_INVALID;""")
    input_offset, output_offset = 0, 0
    for index, node in enumerate(graph.nodes):
        inputs = ", ".join(pointer(key) for key in node.inputs) or "0"
        outputs = ", ".join(pointer(key) for key in node.outputs) or "0"
        context = "context" if calls[index].context_index is None else f"contexts->items[{calls[index].context_index}]"
        if loop is not None:
            for slot, key in enumerate(node.inputs):
                body.append(f"  bindings->inputs[{input_offset + slot}] = {pointer(key)};")
            if not node.inputs:
                body.append(f"  bindings->inputs[{input_offset}] = 0;")
            for slot, key in enumerate(node.outputs):
                body.append(f"  bindings->outputs[{output_offset + slot}] = {pointer(key)};")
            if not node.outputs:
                body.append(f"  bindings->outputs[{output_offset}] = 0;")
            # Assign fields directly: a compound literal for every call can
            # itself produce a large automatic frame in unoptimized builds.
            body.append(f"""  bindings->calls[{index}].function = {calls[index].symbol};
  bindings->calls[{index}].context = {context};
  bindings->calls[{index}].inputs = bindings->inputs + {input_offset}u;
  bindings->calls[{index}].outputs = bindings->outputs + {output_offset}u;
  bindings->calls[{index}].input_count = {len(node.inputs)}u;
  bindings->calls[{index}].output_count = {len(node.outputs)}u;""")
            input_offset += max(1, len(node.inputs))
            output_offset += max(1, len(node.outputs))
            continue
        body.append(f"""  {{
    const void *inputs[{max(1, len(node.inputs))}] = {{{inputs}}};
    void *outputs[{max(1, len(node.outputs))}] = {{{outputs}}};
    state->kernel_status = {calls[index].symbol}({context}, inputs, outputs);
    if (state->kernel_status) return {upper}_KERNEL_ERROR;
    state->completed = {index + 1}u;
  }}""")
    if loop is not None:
        body.append(f"""  const {name}_loop schedule = {{bindings->calls, {len(graph.nodes)}u, {loop.offset}u, {loop.period}u, {loop.trip_count}u}};
  int admitted = hook ? hook(hook_context, &schedule) : 0;
  if (admitted != 0 && admitted != 1) {{ free(bindings); return {upper}_INVALID; }}
  state->active = 1; state->completed = 0; state->kernel_status = 0;
  size_t next = 0;
  /* Flat and counted paths use the same immutable original call table. */
  for (size_t group = 0; group < (admitted ? {loop.trip_count + 2}u : 1u); ++group) {{
    size_t stop = {len(graph.nodes)}u;
    if (admitted) {{
      if (!group) stop = {loop.offset}u;
      else if (group <= {loop.trip_count}u) stop = {loop.offset}u + group * {loop.period}u;
      else stop = {len(graph.nodes)}u;
    }}
    for (; next < stop; ++next) {{
      const {name}_call *call = &bindings->calls[next];
      state->kernel_status = call->function(call->context, call->inputs, call->outputs);
      if (state->kernel_status) {{ free(bindings); return {upper}_KERNEL_ERROR; }}
      state->completed = next + 1u;
    }}
  }}
  free(bindings);""")
    body.append(f"  state->active = 0; return {upper}_OK;\n}}")
    if loop is not None:
        body.append(f"""int {name}_run({name}_state *state, void *arena, size_t arena_bytes,
               const {name}_buffer *buffers, size_t buffer_count, void *context) {{
  return {name}_run_loop(state, arena, arena_bytes, buffers, buffer_count, context, 0, 0);
}}""")
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
    if loop is not None:
        description["loop"] = dict(schema="merlin.dispatch_loop.v1", **vars(loop),
                                   entry=name + "_run_loop", admission="target_hook_before_execution")
    # Validate JSON serializability before returning any generated source.
    json.dumps(description, sort_keys=True)
    return description, files
