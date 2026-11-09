"""Bind compiled regions to the existing checked serial C dispatcher.

This consumes the original selected partition, never a self-reported replacement
plan. Products remain C sources plus explicit link/payload obligations; no device
binary is treated as a callable function merely because it compiled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace

from merlin.compiler.artifact import Compilation
from merlin.compiler.backend.cabi import TensorPort
from merlin.compiler.backend.stateful import CRegionABI2, context_reservations, read_c_abi, selected_context
from merlin.compiler.backend.workspace import CWorkspaceABI
from merlin.compiler.lower.c.dispatch import DispatchCall, emit_program_c
from merlin.compiler.schedule.arena import ArenaRequirements
from merlin.runtime.program import build_program
from merlin.xdsl_dialects.lowering.arena_plan import _buf_bytes
from merlin.xdsl_dialects.lowering.global_plan_emission import dispatch_digest


@dataclass(frozen=True)
class RegionProduct:
    entry: str
    compilation: Compilation
    _identity: tuple = field(init=False, repr=False)

    def __post_init__(self):
        if type(self.entry) is not str or not self.entry or type(self.compilation) is not Compilation:
            raise ValueError("region products require an explicit source entry and Compilation")
        object.__setattr__(self, "_identity", self._current())

    def _current(self):
        return (self.entry, self.compilation.directory,
                json.dumps(self.compilation.to_dict(), sort_keys=True, allow_nan=False))

    def revalidate(self):
        if self._current() != self._identity:
            raise ValueError("compiled region receipt changed after invocation")
        loaded = Compilation.load(self.compilation.manifest)
        if json.dumps(loaded.to_dict(), sort_keys=True, allow_nan=False) != self._identity[2]:
            raise ValueError("compiled region manifest differs from its original invocation")


@dataclass(frozen=True)
class _PartitionEmitter:
    partition: object

    def emit_global_plan(self, program, plan):
        self.partition.revalidate()
        selection = self.partition.selection
        if program is not selection.source.program or plan is not selection.plan:
            raise ValueError("C control requires the original selected graph and plan")
        return self.partition.emission


def control_entry(options, source_entry, deployment=None):
    """Resolve only the implemented public control-emission options."""
    if deployment is None:
        if any(value is not None for value in (options.system, options.platform, options.harts)):
            raise ValueError("planned region/control emission does not select an image deployment")
    else:
        for key in ("system", "platform", "harts"):
            value = getattr(options, key)
            if value is not None and value != getattr(deployment, key):
                raise ValueError(f"planned image differs from explicitly selected {key}")
        if not options.parameters:
            raise ValueError("planned image linking requires merlin.emit=c-dispatch")
    if not options.parameters:
        return None
    if (set(options.parameters) - {"merlin.emit", "merlin.entry"} or
            options.parameters.get("merlin.emit") != "c-dispatch"):
        raise ValueError("planned compiler accepts only merlin.emit=c-dispatch and optional merlin.entry")
    name = options.parameters.get("merlin.entry", source_entry)
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if (type(name) is not str or not name or len(name) > 64 or name[0] not in letters or
            any(c not in letters + "_0123456789" for c in name)):
        raise ValueError("C dispatcher entry must be a C identifier beginning with a letter (at most 64 characters)")
    return name


def emit_selected_c(partition, products: tuple[RegionProduct, ...], *, name: str, directory):
    """Check actual artifact ABIs and emit serial control with bounded memory.

    Every operation needs a synchronous wrapper. Additional host scratch remains
    zero; explicit source-backed tensor workspace is materialized in the caller
    arena and independently bound to original values. Inline CPU operations stay
    supported by the ordinary partition route; this C route refuses to drop them.
    """
    partition.revalidate()
    selection = partition.selection
    if len(partition.regions) != len(selection.selected):
        raise ValueError("C control requires a real compiled wrapper for every selected region; inline CPU remains")
    if type(products) is not tuple or len(products) != len(partition.regions):
        raise ValueError("C control products must exactly cover the original partition")
    graph = partition.emission.dispatch
    components = {row.plan_id: row for row in partition.emission.regions}
    alternatives = {item.alternative.id: item for item in selection.selected}
    memory = selection.topology.memory(selection.memory_domain)
    calls, links, payloads, symbols, bindings = {}, [], [], {}, []
    contexts = {}
    workspaces = {}
    workspace_source_buffers = {}
    reservations = context_reservations(selection.selected, selection.topology)
    for region, product in zip(partition.regions, products, strict=True):
        if type(product) is not RegionProduct:
            raise ValueError("C control requires typed region products")
        product.revalidate()
        compiled = product.compilation
        expected = region.request_source(product.entry)
        if compiled.source_sha256 != expected.sha256 or compiled.backend != region.provider:
            raise ValueError("compiled C region differs from the exact selected source/provider")
        if not compiled.directory.is_relative_to(directory):
            raise ValueError("compiled C region lies outside the selected build directory")
        abi = read_c_abi(compiled)
        function = region.source_module.body.block.first_op
        if (abi.inputs != tuple(TensorPort.from_type(t) for t in function.function_type.inputs) or
                abi.outputs != tuple(TensorPort.from_type(t) for t in function.function_type.outputs)):
            raise ValueError("compiled C ABI differs from the ordered original tensor signature")
        item = alternatives[region.alternative]
        if abi.resource_kind != selection.topology.resource(item.proposal.resource).kind:
            raise ValueError("compiled implementation resource kind differs from the selected resource")
        if item.alternative.scratch_bytes != 0:
            raise ValueError("this C control route cannot materialize a nonzero region scratch reservation")
        if abi.buffer_alignment > memory.alignment_bytes:
            raise ValueError("C ABI alignment exceeds the selected memory contract; replan explicitly")
        indices = components[region.alternative].node_indices
        if len(indices) != 1 or graph.nodes[indices[0]].kind != "dispatch":
            raise ValueError("C wrapper must correspond to one actual outlined invocation")
        index = indices[0]
        context_index = None
        requirement = selected_context(item.alternative)
        if isinstance(abi, CWorkspaceABI):
            from merlin.compiler.lower.llvm.workspace import plan

            expected_workspace, _ = plan(tuple(function.body.block.ops)[:-1], function.get_return_op().arguments)
            declared = dict(item.alternative.metadata)
            expected_record = json.dumps(abi.workspace.record(), sort_keys=True, separators=(",", ":"))
            if (expected_workspace != abi.workspace or declared.get("source_workspace") != expected_record
                    or declared.get("workspace_status") != "source_backed_native_admission_pending"
                    or abi.workspace.alignment > memory.alignment_bytes):
                raise ValueError("compiled source workspace differs from exact selected original tensor storage")
            names = []
            for (ordinal, result), port in zip(abi.workspace.sources, abi.workspace.ports, strict=True):
                original = item.alternative.nodes[ordinal]
                logical = selection.source.program.nodes[original].outputs[result]
                buffer = selection.source.program.buffers[logical]
                if (tuple(buffer.shape) != port.shape or buffer.dtype != port.dtype or buffer.kind != "intermediate"):
                    raise ValueError("source workspace is not the same original tensor charged by planning")
                names.append(logical)
            workspaces[index] = abi.workspace
            workspace_source_buffers[index] = names
        if isinstance(abi, CRegionABI2):
            if requirement != abi.context:
                raise ValueError("compiled context differs from the original selected host reservation")
            declaration = next(a for a in compiled.artifacts
                               if a.path.relative_to(compiled.directory).as_posix() == abi.context.declaration)
            identity = (abi.context, compiled.source_sha256, declaration.sha256,
                        tuple((a.kind, a.role, a.sha256) for a in compiled.artifacts))
            previous = contexts.get(abi.context.identity)
            if previous is not None and previous[1] != identity:
                raise ValueError("shared context identity has different compiled initialization or products")
            if previous is None:
                previous = (len(contexts), identity)
                contexts[abi.context.identity] = previous
            context_index = previous[0]
        elif requirement is not None:
            raise ValueError("selected caller-owned context was not discharged by a stateful C ABI")
        calls[index] = DispatchCall(abi.symbol, item.alternative.placement, buffer_alignment=abi.buffer_alignment,
                                    context_index=context_index)
        artifacts = {a.path.relative_to(compiled.directory).as_posix(): a for a in compiled.artifacts}
        # Identical stateless implementations may serve multiple source sites.
        # A repeated symbol with different source, ABI or products must not win
        # by link order. Retain one definition only after exact identity checks.
        identity = (compiled.backend, compiled.source_sha256, abi,
                    tuple(sorted((key, a.kind, a.role, a.sha256) for key, a in artifacts.items())))
        if abi.symbol in symbols and symbols[abi.symbol] != identity:
            raise ValueError("compiled regions conflict on a C wrapper symbol")
        if abi.symbol not in symbols:
            links.extend((compiled.directory / path).relative_to(directory).as_posix() for path in abi.link_inputs)
            payloads.extend((compiled.directory / path).relative_to(directory).as_posix() for path in abi.payloads)
            symbols[abi.symbol] = identity
        bindings.append({"alternative": region.alternative, "provider": region.provider,
                         "resource": item.proposal.resource, "memory_domain": item.proposal.memory_domain,
                         "node": index, "source_sha256": compiled.source_sha256, "abi": abi.record(),
                         "compilation": compiled.manifest.relative_to(directory).as_posix()})
    if set(calls) != set(range(len(graph.nodes))):
        raise ValueError("C control cannot omit an emitted source operation")
    external = {key for key, value in graph.buffers.items() if value.kind in {"arg", "const"}} | set(graph.results)
    alignment = memory.alignment_bytes
    external_bytes = sum(((_buf_bytes(graph.buffers[key].shape, graph.buffers[key].dtype) + alignment - 1)
                          // alignment) * alignment for key in external)
    reserved_bytes = reservations["domains"].get(memory.id, {}).get("total_bytes", 0)
    workspace_bytes = max((((layout.bytes + alignment - 1) // alignment) * alignment
                           for layout in workspaces.values()), default=0)
    available = memory.capacity_bytes - external_bytes - reserved_bytes - workspace_bytes
    if available < 0:
        raise ValueError("external C buffers exceed selected memory capacity")
    requirements = ArenaRequirements(
        max(1, available + workspace_bytes), tuple(2 * i for i in range(len(graph.nodes))),
        tuple(2 * i + 1 for i in range(len(graph.nodes))), alignment=alignment,
        dispatch_digest=dispatch_digest(graph))
    program = build_program(selection.source.program, capability="compiled_pointer_calls", global_plan=selection.plan,
                            global_plan_emitter=_PartitionEmitter(partition), memory_requirements=requirements)
    if program.memory_plan.arena_bytes > available:
        raise ValueError("C arena and external buffers exceed selected memory capacity")
    if workspaces:
        workspace_offset = (program.memory_plan.arena_bytes + alignment - 1) // alignment * alignment
        if external_bytes + reserved_bytes + workspace_offset + workspace_bytes > memory.capacity_bytes:
            raise ValueError("materialized source workspace and C arena exceed selected memory capacity")
        calls = {index: replace(call, workspace_offset=workspace_offset, workspace_bytes=workspaces[index].bytes,
                                workspace_alignment=workspaces[index].alignment) if index in workspaces else call
                 for index, call in calls.items()}
    read_only = tuple(key for key, value in graph.buffers.items() if value.kind == "arg")
    report, files = emit_program_c(program, name, calls, read_only_inputs=read_only)
    report.update(bindings=bindings, link_inputs=links, payloads=payloads,
                  topology_sha256=selection.topology.sha256, memory_domain=memory.id,
                  external_bytes=external_bytes, total_tensor_bytes=external_bytes + program.memory_plan.arena_bytes,
                  backend_correctness="not_established", context="caller_owned_table" if contexts else "unused")
    if contexts:
        report.update(host_context_reservations=reservations,
                      context_indices={identity: row[0] for identity, row in contexts.items()},
                      context_initialization="caller_obligation_not_observed",
                      payload_loading="checked_embedding_bytes; loader_invocation_not_observed",
                      completion_visibility="runtime_obligation; no_execution_observed",
                      error_lifetime="retain_context_workspace_and_device_buffers_until_drained_or_reset",
                      total_reserved_host_bytes=reserved_bytes)
        if workspaces:
            report["context_arena_disjointness"] = (
                "caller_obligation; generated_image_entry_checks_before_initialization")
            report["error_lifetime"] = "retain_context_workspace_arena_external_and_device_buffers_until_quiescent"
    if workspaces:
        report.update(source_workspace={"offset": workspace_offset, "reserved_bytes": workspace_bytes,
                                        "source_values": {str(index): layout.record()
                                                          for index, layout in workspaces.items()},
                                        "original_buffers": workspace_source_buffers,
                                        "accounting": "original source values counted by source lifetime bound"},
                      total_tensor_bytes=external_bytes + workspace_offset + workspace_bytes)
    partition.revalidate()
    for product in products:
        product.revalidate()
    return report, files
