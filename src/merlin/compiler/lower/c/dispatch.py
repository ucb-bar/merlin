"""Emit a serial C driver for explicitly compiled dispatch implementations.

The driver consumes the existing program and checked arena offsets. It owns no
operator interpreter, device instructions, startup code or platform services.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from math import lcm
from pathlib import PurePosixPath
from typing import Any

from merlin.compiler.partition.dispatch_program import DispatchProgram
from merlin.compiler.partition.global_plan import GlobalPlan, verify_global_plan
from merlin.compiler.partition.global_plan_emission import (
    GlobalPlanEmission,
    GlobalPlanEmitter,
    dispatch_digest,
    emit_global_plan,
)
from merlin.compiler.partition.memory_plan import MemoryPlan, plan_arena
from merlin.compiler.schedule.storage.arena import ArenaRequirements, check_bounded_arena, plan_bounded_arena

# ---- sources -----------------------------------------------------------------------------------
# Combine explicitly selected C sources into a standalone source/header pair.
#
# This is packaging, not a C preprocessor. Providers supply headers in dependency
# order. Literal project includes must resolve to an already supplied header;
# system includes and preprocessor conditions remain for the consumer's compiler.


def c_source_pair(
    name: str,
    *,
    public_headers: Mapping[str, str],
    private_headers: Mapping[str, str],
    sources: Mapping[str, str],
    declarations: str = "",
    implementation: str = "",
) -> dict[str, str]:
    """Render C/H files without checkout includes or a generated main function.

    File names are logical source names used in diagnostics, not local paths.
    The caller owns source selection, symbol compatibility and platform ABI.
    The consumer still supplies any required C library and platform services.
    """
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    identifier_chars = letters + "0123456789_"
    if not name or len(name) > 64 or name[0] not in letters or any(c not in identifier_chars for c in name):
        raise ValueError("package name must be a C identifier beginning with a letter")
    members = (*public_headers, *private_headers, *sources)
    if not sources or len(set(members)) != len(members):
        raise ValueError("C package needs sources and distinct logical member names")
    for member in members:
        path = PurePosixPath(member)
        if (
            not member
            or path.is_absolute()
            or path.as_posix() != member
            or ".." in path.parts
            or any(c not in identifier_chars + "./-" for c in member)
        ):
            raise ValueError("invalid logical C source name")

    available: set[str] = set()

    def flatten(member: str, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("empty C package member")
        lines = [f"#line 1 {json.dumps(member)}"]
        for line in text.splitlines():
            directive = line.lstrip()
            if directive.startswith("#"):
                directive = directive[1:].lstrip()
                end = 0
                while end < len(directive) and directive[end] in identifier_chars:
                    end += 1
                keyword, operand = directive[:end], directive[end:].strip()
            else:
                keyword, operand = "", ""
            if keyword in ("include", "include_next"):
                # A trailing line comment, as in pinned vendor sources, is not part of the operand.
                if operand.startswith("<") and ">" in operand and operand.split(">", 1)[1].lstrip().startswith("//"):
                    operand = operand.split(">", 1)[0] + ">"
                if keyword != "include" or len(operand) < 3 or operand[0] not in ('"', "<"):
                    raise ValueError(f"nonliteral or unsupported include in {member}")
                closing = '"' if operand[0] == '"' else ">"
                if not operand.endswith(closing) or closing in operand[1:-1]:
                    raise ValueError(f"nonliteral or unsupported include in {member}")
                if operand[0] == '"':
                    target = posixpath.normpath(str(PurePosixPath(member).parent / operand[1:-1]))
                    if target not in available:
                        raise ValueError(f"unresolved or out-of-order include {operand!r} in {member}")
                    # Keep line numbering for diagnostics after removing it.
                    lines.append("")
                    continue
            lines.append(line)
        return "\n".join(lines) + "\n"

    public = []
    for member, text in public_headers.items():
        public.append(flatten(member, text))
        available.add(member)
    guard = "MERLIN_C_PACKAGE_" + name.upper() + "_H"
    header = (
        f"#ifndef {guard}\n#define {guard}\n\n"
        + "\n".join(public)
        + '\n#ifdef __cplusplus\nextern "C" {\n#endif\n'
        + declarations
        + "\n#ifdef __cplusplus\n}\n#endif\n#endif\n"
    )
    body = [f'#include "{name}.h"\n']
    # Clang normally suppresses unused inline helpers in included headers.
    # Preserve that behavior for private headers embedded in a C translation
    # unit, without suppressing warnings in the implementation below them.
    body.append(
        '#ifdef __clang__\n#pragma clang diagnostic push\n#pragma clang diagnostic ignored "-Wunused-function"\n#endif'
    )
    for member, text in private_headers.items():
        body.append(flatten(member, text))
        available.add(member)
    body.append("#ifdef __clang__\n#pragma clang diagnostic pop\n#endif")
    for member, text in sources.items():
        body.append(flatten(member, text))
    body.append(implementation)
    return {name + ".h": header, name + ".c": "\n".join(body) + "\n"}


# ---- program -----------------------------------------------------------------------------------
# ``MerlinProgram`` — the target-independent, lean whole-model PROGRAM the HW-agnostic runtime
# replays.
#
# This is the superset of :class:`~merlin.compiler.partition.dispatch_program.DispatchProgram`
# (the DAG-verified command buffer of dispatch/view nodes over stable buffer ids) plus the two things
# a LEAN runtime needs that the bare dispatch program lacks:
#
# * a compile-time **static memory plan** (:mod:`~merlin.compiler.partition.memory_plan`) — one
#   arena + per-intermediate-buffer offsets, so the runtime binds ``arena_base()+offset`` with ZERO
#   per-op allocation (vs the ~4391 per-op mallocs today); and
# * **target/dispatch metadata** — a kernel table (id → compiled symbol + fused roots + required
#   capability), per-node parallel annotation (``none|forall|launch``), an extensible/namespaced
#   opcode set, and per-program capability flags — the hooks that let custom-ISA / SIMT / research
#   HW plug in via an adapter without changing the program format.
#
# The program is serializable data. The explicitly selected C dispatcher in
# :mod:`merlin.compiler.lower.c.dispatch` consumes a checked bounded-memory plan
# and compiled wrappers. It supplies no operator interpreter or automatic target
# lowering. The ordinary monolithic model path remains separate.
#
# The arena the measured whole-model build actually gets does NOT come through here. It is planned
# directly over the emitted LLVM IR by :mod:`merlin.compiler.lower.host.arena`, which shares this module's
# placement core (``arena_plan.pack_disjoint``) but derives liveness from the ``malloc``/``free`` pairs
# the build emits rather than from a dispatch DAG.
# Default-off and additive — the existing monolithic/outlined paths are untouched.

PROGRAM_ABI_VERSION = "0.1"


@dataclass
class KernelEntry:
    id: str  # stable kernel id (== dispatch symbol)
    symbol: str  # compiled C symbol the replay invokes
    roots: list[str] = field(default_factory=list)  # fused root op names (provenance)
    capability: str = "scalar"  # required target capability (scalar|rvv|systolic|simt|...)
    # Model-layer provenance carried from the dispatch node's ``prov.*`` (see the outliner). The
    # same key the section slicer joins on. ``role``
    # is intentionally NOT stored here — it is derived from ``fqn`` downstream (``dse_guidance``) to
    # keep ``runtime`` free of an analysis-layer import (``dse_guidance`` already imports ``runtime``).
    region_id: str = ""  # prov.region_id (e.g. "matmul_3")
    fqn: str = ""  # prov.fqn (deepest nn.Module path, e.g. "blocks.0.attn.q")


@dataclass
class MerlinProgram:
    abi_version: str
    entry: str
    dispatch: DispatchProgram  # the DAG (buffers, nodes, args, results)
    memory_plan: MemoryPlan  # static arena + offsets
    kernels: dict[str, KernelEntry]  # symbol -> KernelEntry
    capabilities: list[str]  # capabilities this program requires of a target
    parallel: dict[int, dict] = field(default_factory=dict)  # node index -> {mode, grid, ...}
    opcodes: list[str] = field(default_factory=list)  # namespaced opcode set used
    # Optional because the legacy program ABI remains byte-identical unless the production compiler
    # explicitly elects the global-plan path.  ``shadow`` mode can therefore analyse a model without
    # changing the command buffer that executes it; ``emit`` attaches the exact plan lowering consumed.
    global_plan: GlobalPlan | None = None
    global_plan_emission: GlobalPlanEmission | None = None
    memory_requirements: ArenaRequirements | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "abi_version": self.abi_version,
            "entry": self.entry,
            "dispatch": self.dispatch.to_dict(),
            "memory_plan": self.memory_plan.to_dict(),
            "kernels": {k: vars(v) for k, v in self.kernels.items()},
            "capabilities": list(self.capabilities),
            "parallel": {str(k): v for k, v in self.parallel.items()},
            "opcodes": list(self.opcodes),
        }
        if self.global_plan is not None:
            result["global_plan"] = self.global_plan.to_dict()
        if self.global_plan_emission is not None:
            result["global_plan_emission"] = self.global_plan_emission.receipt()
        return result

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @property
    def stats(self) -> dict[str, Any]:
        d = self.dispatch
        return {
            "n_nodes": len(d.nodes),
            "n_dispatches": d.n_dispatches,
            "n_views": sum(1 for n in d.nodes if n.kind == "view"),
            "n_buffers": len(d.buffers),
            "n_kernels": len(self.kernels),
            "arena_bytes": self.memory_plan.arena_bytes,
            "arena_reuse_factor": self.memory_plan.stats.get("reuse_factor"),
            "mallocs_eliminated": self.memory_plan.stats.get("n_intermediate_buffers"),
            "capabilities": list(self.capabilities),
            "global_plan": self.global_plan.digest if self.global_plan is not None else None,
            "global_plan_emitted": self.global_plan_emission is not None,
        }


def build_program(
    dispatch: DispatchProgram,
    *,
    capability: str = "rvv",
    abi_version: str = PROGRAM_ABI_VERSION,
    global_plan: GlobalPlan | None = None,
    global_plan_emitter: GlobalPlanEmitter | None = None,
    memory_requirements: ArenaRequirements | Callable[[DispatchProgram], ArenaRequirements] | None = None,
    memory_plan: MemoryPlan | None = None,
) -> MerlinProgram:
    """Assemble a :class:`MerlinProgram` from a dispatch program: plan the arena, build the kernel
    table from the dispatch nodes, and tag the required capability. The opcode set is the distinct
    node ops (namespaced as needed by a target adapter); v1 parallel annotation is empty (the replay
    runs nodes serially; multi-core is layered in via ``parallel`` later, node-level).
    """
    if global_plan is not None:
        problems = verify_global_plan(dispatch, global_plan)
        if problems:
            raise ValueError("cannot attach invalid global plan: " + "; ".join(problems))
    if global_plan_emitter is not None and global_plan is None:
        raise ValueError("a global-plan emitter requires a selected global plan")
    emission = (
        emit_global_plan(dispatch, global_plan, global_plan_emitter)
        if global_plan is not None and global_plan_emitter is not None
        else None
    )
    # An emitter changes the program that is actually replayed. Without one, attaching a plan is
    # explicitly shadow analysis and preserves the legacy dispatch byte-for-byte.
    if emission is not None:
        dispatch = emission.dispatch
    # Recompute requirements after outlining so lifetimes bind the actual graph.
    if callable(memory_requirements):
        memory_requirements = memory_requirements(dispatch)
    if memory_plan is not None:
        if memory_requirements is None:
            raise ValueError("reusing a memory plan requires explicit memory requirements")
        check_bounded_arena(dispatch, memory_requirements, memory_plan)
        mem = memory_plan
    else:
        mem = (
            plan_bounded_arena(dispatch, memory_requirements)
            if memory_requirements is not None
            else plan_arena(dispatch)
        )
    kernels: dict[str, KernelEntry] = {}
    opcodes: set[str] = set()
    for node in dispatch.nodes:
        opcodes.add(node.op if node.kind == "view" else f"dispatch:{node.op}")
        if node.kind == "dispatch" and node.op not in kernels:
            # The dispatch node's prov keys are the model2MLIR ``prov.*`` spellings (``prov.op`` /
            # ``prov.fqn`` / ``prov.region_id``), NOT bare ``root``/``op``/``name`` — the old filter
            # matched none of them, so ``roots`` was silently always ``[]`` and the layer identity was
            # dropped when building the kernel table. Read the real keys and carry region_id/fqn through.
            prov = node.prov or {}
            roots = [prov[k] for k in ("prov.op", "prov.name") if k in prov]
            kernels[node.op] = KernelEntry(
                id=node.op,
                symbol=node.op,
                roots=roots,
                capability=capability,
                region_id=prov.get("prov.region_id", ""),
                fqn=prov.get("prov.fqn", ""),
            )
    return MerlinProgram(
        abi_version=abi_version,
        entry=dispatch.entry,
        dispatch=dispatch,
        memory_plan=mem,
        kernels=kernels,
        capabilities=[capability],
        opcodes=sorted(opcodes),
        global_plan=global_plan,
        global_plan_emission=emission,
        memory_requirements=memory_requirements,
    )


# ---- dispatch ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class DispatchCall:
    """A compiled pointer-ABI wrapper that returns zero only after completion.

    ``placement`` names the selected execution endpoint, not output storage.
    A wrapper must preserve the source semantics and report failure directly.
    ``alias_safe`` explicitly admits the graph's declared overlapping views.
    ``context_index`` selects a slot in the emitted context table. With no slot,
    the wrapper receives the original context pointer as in the single-call ABI.
    ``buffer_alignment`` is the required byte alignment of every input/output
    pointer passed to this wrapper, independent of element storage width.
    """

    symbol: str
    placement: str
    completion: str = "synchronous"
    alias_safe: bool = False
    context_index: int | None = None
    buffer_alignment: int = 1
    workspace_offset: int | None = None
    workspace_bytes: int = 0
    workspace_alignment: int = 1


@dataclass(frozen=True)
class DispatchLoop:
    """A counted window authenticated against the entire flat driver.

    The window regroups the original ordered call table: each iteration keeps
    its wrapper, placement, context, and pointers. It proves no body equivalence
    by itself. Any shared body or resident parameter binding is a separate
    provider-checked fact carried by those calls and their contexts.
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
    """Capture current source, selected plan, memory, and ordered call bindings."""
    if (
        any(type(v) is not int for v in (offset, period, trip_count))
        or offset < 0
        or period < 1
        or trip_count < 2
        or offset + period * trip_count > len(program.dispatch.nodes)
    ):
        raise ValueError("invalid dispatch loop bounds")
    return DispatchLoop(offset, period, trip_count, _loop_authority(program, calls))


def prepare_dispatch_loop_from_plan(program, calls, record, *, requests, source_request_digest):
    """Bind all trips of a current application loop plan to original C calls.

    The analysis signature only locates a window. Exact request expansion and
    the current dispatch, selected plan, memory, and call table remain authority.
    """
    from merlin.compiler.schedule.region_loop import expand_region_loop

    if (
        not isinstance(record, dict)
        or record.get("schema") != "merlin.application_loop_plan.v1"
        or record.get("source_request_digest") != source_request_digest
    ):
        raise ValueError("application loop plan differs from current source")
    region_loop = record.get("region_loop")
    if (
        not isinstance(region_loop, dict)
        or region_loop.get("schema") != "merlin.region_loop.v1"
        or region_loop.get("mode") != "analysis_only"
    ):
        raise ValueError("application loop plan has no repeated region")
    requests = list(requests)
    if expand_region_loop(region_loop) != requests or len(requests) != len(program.dispatch.nodes):
        raise ValueError("application loop plan differs from current region requests")
    return prepare_dispatch_loop(
        program, calls, offset=region_loop["offset"], period=region_loop["period"], trip_count=region_loop["trip_count"]
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
    resident_only_inputs: tuple[str, ...] = (),
    loop: DispatchLoop | None = None,
) -> tuple[dict, dict[str, str]]:
    """Return a standalone C/H driver and its ordered buffer/call description.

    Every node, including support operations, needs an implementation binding.
    Caller-owned arguments/constants/results stay outside the temporary arena.
    The caller reserves and aligns that arena and supplies the wrapper context.
    This path is explicit; it does not change monolithic LLVM compilation.
    With an authenticated ``loop``, the additive ``_run_loop`` entry first asks
    the caller's hook to admit the complete call table, then runs the same calls
    in counted groups. The ``_run`` entry stays available for flat dispatch.
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
            or type(call.buffer_alignment) is not int
            or call.buffer_alignment <= 0
            or call.buffer_alignment & (call.buffer_alignment - 1)
        ):
            raise ValueError(
                "compiled calls need valid symbols, placement, synchronous completion and power-of-two buffer alignment"
            )
        if call.workspace_offset is not None:
            if (
                type(call.workspace_offset) is not int
                or call.workspace_offset < memory.arena_bytes
                or type(call.workspace_bytes) is not int
                or call.workspace_bytes <= 0
                or type(call.workspace_alignment) is not int
                or call.workspace_alignment <= 0
                or call.workspace_alignment & (call.workspace_alignment - 1)
                or call.workspace_offset % call.workspace_alignment
                or call.context_index is not None
            ):
                raise ValueError("source workspace requires aligned disjoint arena storage without a context slot")
        elif call.workspace_bytes != 0 or call.workspace_alignment != 1:
            raise ValueError("workspace requirements need an actual arena offset")
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
    if (
        not isinstance(resident_only_inputs, tuple)
        or len(set(resident_only_inputs)) != len(resident_only_inputs)
        or not set(resident_only_inputs) <= set(read_only_inputs)
        or set(resident_only_inputs) & set(graph.results)
    ):
        raise ValueError("resident-only inputs must be distinct read-only arguments without public outputs")
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
    alignments = dict.fromkeys(graph.buffers, 1)
    for index, node in enumerate(graph.nodes):
        for key in (*node.inputs, *node.outputs):
            alignments[key] = max(alignments[key], calls[index].buffer_alignment)
    arena_alignment = requirements.alignment
    arena_bytes = memory.arena_bytes
    for call in calls.values():
        if call.workspace_offset is not None:
            arena_alignment = lcm(arena_alignment, call.workspace_alignment)
            arena_bytes = max(arena_bytes, call.workspace_offset + call.workspace_bytes)
    if arena_bytes > requirements.capacity_bytes:
        raise ValueError("actual source workspace and graph arena exceed the selected arena capacity")
    for key, offset in memory.offsets.items():
        if offset % alignments[key]:
            raise ValueError(f"arena offset for {key!r} cannot satisfy its compiled buffer alignment")
        arena_alignment = lcm(arena_alignment, alignments[key])
    slots = {key: index for index, key in enumerate(external)}
    if loop is not None and (
        not isinstance(loop, DispatchLoop)
        or loop != prepare_dispatch_loop(
            program, calls, offset=loop.offset, period=loop.period, trip_count=loop.trip_count
        )
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
#define {upper}_ARENA_BYTES {arena_bytes}u
#define {upper}_ARENA_ALIGNMENT {arena_alignment}u
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
/* body numbers the call's kernel among this program's kernels; calls run by direct call, so no kernel's address
 * is taken (a static stack bound would follow any function pointer to every address-taken function). */
typedef struct {{
  unsigned body;
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
    kernels = sorted({call.symbol for call in calls.values()})
    for symbol in kernels:
        if symbol == name + "_run" or (loop is not None and symbol == name + "_run_loop"):
            raise ValueError("kernel symbol conflicts with generated entry")
        body.append(f"extern int {symbol}(void *, const void *const *, void *const *);")
    if loop is not None:
        cases = "".join(f"  case {i}u: return {symbol}(call->context, call->inputs, call->outputs);\n"
                        for i, symbol in enumerate(kernels))
        body.append(f"""static int {name}_call_body(const {name}_call *call) {{
  switch (call->body) {{
{cases}  default: return -1;
  }}
}}""")
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
    if arena_bytes:
        body.append(f"""  if (arena_bytes < {arena_bytes}u || !begin ||
      begin % {arena_alignment}u || begin > UINTPTR_MAX - {arena_bytes}u)
    return {upper}_INVALID;
  if (state_begin < begin + {arena_bytes}u &&
      begin < state_begin + sizeof(*state)) return {upper}_INVALID;
""")
    has_workspace = any(call.workspace_offset is not None for call in calls.values())
    if has_workspace and external:
        body.append(f"""  uintptr_t table_begin=(uintptr_t)buffers;
  size_t table_bytes=sizeof(*buffers)*{len(external)}u;
  if (table_begin>UINTPTR_MAX-table_bytes ||
      (state_begin<table_begin+table_bytes && table_begin<state_begin+sizeof(*state)) ||
      (table_begin<begin+{arena_bytes}u && begin<table_begin+table_bytes))
    return {upper}_INVALID;
""")
    if external:
        sizes = ",".join(f"{memory.sizes[key]}u" for key in external)
        readonly = ",".join(str(int(key in read_only_inputs)) for key in external)
        resident = (
            f"  static const unsigned char resident_only[{len(external)}] = {{"
            + ",".join(str(int(key in resident_only_inputs)) for key in external)
            + "};\n"
            if resident_only_inputs
            else ""
        )
        absent = "    if (!buffers[i].data && resident_only[i]) continue;\n" if resident_only_inputs else ""
        absent_previous = "      if (!buffers[j].data && resident_only[j]) continue;\n" if resident_only_inputs else ""
        alignment_check = any(alignments[key] != 1 for key in external)
        if alignment_check:
            required_alignment = ",".join(f"{alignments[key]}u" for key in external)
            body.append(f"  static const size_t alignment[{len(external)}] = {{{required_alignment}}};")
        # Keep validation bounded in generated source size. Expanding every
        # buffer pair makes ordinary host optimization quadratic in IR size.
        body.append(f"""  static const size_t required[{len(external)}] = {{{sizes}}};
  static const unsigned char readonly[{len(external)}] = {{{readonly}}};
{resident}  for (size_t i=0; i<{len(external)}u; ++i) {{
    uintptr_t address=(uintptr_t)buffers[i].data;
{absent}    if (!buffers[i].data || buffers[i].bytes<required[i] || address>UINTPTR_MAX-required[i])
      return {upper}_INVALID;
    if (state_begin<address+required[i] && address<state_begin+sizeof(*state))
      return {upper}_INVALID;
""")
        if has_workspace:
            body.append(f"""    if (table_begin<address+required[i] && address<table_begin+table_bytes)
      return {upper}_INVALID;
""")
        if alignment_check:
            body.append(f"    if (address % alignment[i]) return {upper}_INVALID;")
        if arena_bytes:
            body.append(f"""    if (address<begin+{arena_bytes}u && begin<address+required[i])
      return {upper}_INVALID;
""")
        body.append(f"""    for (size_t j=0; j<i; ++j) {{
      if (readonly[i] && readonly[j]) continue;
{absent_previous}      uintptr_t other=(uintptr_t)buffers[j].data;
      if (address<other+required[j] && other<address+required[i]) return {upper}_INVALID;
    }}
  }}
""")
    if loop is None:
        body.append("  state->active = 1; state->completed = 0; state->kernel_status = 0;")
    else:
        input_count = sum(max(1, len(node.inputs)) for node in graph.nodes)
        output_count = sum(max(1, len(node.outputs)) for node in graph.nodes)
        # One static table, not the heap (a board host declares none) and not the stack (it grows with the calls).
        body.append(f"""  static struct {name}_bindings {{
    {name}_call calls[{len(graph.nodes)}];
    const void *inputs[{input_count}];
    void *outputs[{output_count}];
  }} table;
  static unsigned char table_busy;
  if (table_busy) return {upper}_BUSY;
  table_busy = 1;
  struct {name}_bindings *bindings = &table;""")
    input_offset, output_offset = 0, 0
    for index, node in enumerate(graph.nodes):
        inputs = ", ".join(pointer(key) for key in node.inputs) or "0"
        outputs = ", ".join(pointer(key) for key in node.outputs) or "0"
        context = "context" if calls[index].context_index is None else f"contexts->items[{calls[index].context_index}]"
        if calls[index].workspace_offset is not None:
            context = f"((unsigned char *)arena + {calls[index].workspace_offset}u)"
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
            body.append(f"""  bindings->calls[{index}].body = {kernels.index(calls[index].symbol)}u;
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
  if (admitted != 0 && admitted != 1) {{ table_busy = 0; return {upper}_INVALID; }}
  state->active = 1; state->completed = 0; state->kernel_status = 0;
  size_t next = 0;
  /* Flat and counted paths use the same immutable original call table. */
  for (size_t group = 0; group < (admitted ? {loop.trip_count + 2}u : 1u); ++group) {{
    size_t stop = {len(graph.nodes)}u;
    if (admitted) {{
      if (!group) stop = {loop.offset}u;
      else if (group <= {loop.trip_count}u) stop = {loop.offset}u + group * {loop.period}u;
    }}
    for (; next < stop; ++next) {{
      const {name}_call *call = &bindings->calls[next];
      state->kernel_status = {name}_call_body(call);
      if (state->kernel_status) {{ table_busy = 0; return {upper}_KERNEL_ERROR; }}
      state->completed = next + 1u;
    }}
  }}
  table_busy = 0;""")
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
                **({"alignment": alignments[key]} if alignments[key] != 1 else {}),
            }
            for key in external
        ],
        "inputs": args,
        "outputs": list(graph.results),
        "calls": [
            {
                "node": index,
                **{
                    k: v
                    for k, v in vars(calls[index]).items()
                    if (k != "context_index" or v is not None)
                    and (k != "buffer_alignment" or v != 1)
                    and (k != "workspace_offset" or v is not None)
                    and (k != "workspace_bytes" or v != 0)
                    and (k != "workspace_alignment" or v != 1)
                },
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
    if resident_only_inputs:
        description["resident_only_inputs"] = list(resident_only_inputs)
    if loop is not None:
        description["loop"] = dict(
            schema="merlin.dispatch_loop.v1", **vars(loop),
            entry=name + "_run_loop", admission="target_hook_before_execution",
        )
    if arena_alignment != requirements.alignment:
        description["arena_alignment"] = arena_alignment
    # Validate JSON serializability before returning any generated source.
    json.dumps(description, sort_keys=True)
    return description, files
