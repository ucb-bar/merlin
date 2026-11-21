"""Bind compiled regions to the existing checked serial C dispatcher.

This consumes the original selected partition, never a self-reported replacement
plan. Products remain C sources plus explicit link/payload obligations; no device
binary is treated as a callable function merely because it compiled.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

from merlin.common.storage.content_store import digest_file
from merlin.compiler.backend.c_abi import (
    CRegionABI2,
    CWorkspaceABI,
    TensorPort,
    context_reservations,
    selected_context,
)
from merlin.compiler.compilation import Compilation
from merlin.compiler.build.completion import admit_call
from merlin.compiler.lower.c.application import emit_application_c
from merlin.compiler.lower.c.dispatch import DispatchCall, build_program, c_source_pair, emit_program_c
from merlin.compiler.partition.global_plan import GlobalPlan
from merlin.compiler.partition.global_plan_emission import dispatch_digest
from merlin.compiler.partition.memory_plan import ArenaPlanError, MemoryPlan, _buf_bytes
from merlin.compiler.partition.semantic_regions import outline_semantic_regions
from merlin.compiler.planning.cover_selection import application_plan
from merlin.compiler.schedule.storage.arena import ArenaRequirements
from merlin.compiler.semantic.model import KernelRequest
from merlin.compiler.semantic.source_preparation import preparation_filename, verify_source_preparation
from merlin.compiler.semantic.target_binding import NativeCompilationError, verify_native_publication
from merlin.development.targetgen.execution.host.admission import validate_linkage_contract

# ---- control -----------------------------------------------------------------------------------


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
        return (
            self.entry,
            self.compilation.directory,
            json.dumps(self.compilation.to_dict(), sort_keys=True, allow_nan=False),
        )

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
    if (
        set(options.parameters) - {"merlin.emit", "merlin.entry"}
        or options.parameters.get("merlin.emit") != "c-dispatch"
    ):
        raise ValueError("planned compiler accepts only merlin.emit=c-dispatch and optional merlin.entry")
    name = options.parameters.get("merlin.entry", source_entry)
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if (
        type(name) is not str
        or not name
        or len(name) > 64
        or name[0] not in letters
        or any(c not in letters + "_0123456789" for c in name)
    ):
        raise ValueError("C dispatcher entry must be a C identifier beginning with a letter (at most 64 characters)")
    return name


def emit_selected_c(partition, products: tuple[RegionProduct, ...], *, name: str, directory, validate_completion=None):
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
        item = alternatives[region.alternative]
        abi, completion = admit_call(
            compiled,
            topology=selection.topology,
            resource=item.proposal.resource,
            memory_domain=item.proposal.memory_domain,
            validate_completion=validate_completion,
        )
        function = region.source_module.body.block.first_op
        if abi.inputs != tuple(TensorPort.from_type(t) for t in function.function_type.inputs) or abi.outputs != tuple(
            TensorPort.from_type(t) for t in function.function_type.outputs
        ):
            raise ValueError("compiled C ABI differs from the ordered original tensor signature")
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
            from merlin.compiler.lower.native.caller_buffers import plan

            expected_workspace, _ = plan(tuple(function.body.block.ops)[:-1], function.get_return_op().arguments)
            declared = dict(item.alternative.metadata)
            expected_record = json.dumps(abi.workspace.record(), sort_keys=True, separators=(",", ":"))
            if (
                expected_workspace != abi.workspace
                or declared.get("source_workspace") != expected_record
                or declared.get("workspace_status") != "source_backed_native_admission_pending"
                or abi.workspace.alignment > memory.alignment_bytes
            ):
                raise ValueError("compiled source workspace differs from exact selected original tensor storage")
            names = []
            for (ordinal, result), port in zip(abi.workspace.sources, abi.workspace.ports, strict=True):
                original = item.alternative.nodes[ordinal]
                logical = selection.source.program.nodes[original].outputs[result]
                buffer = selection.source.program.buffers[logical]
                if tuple(buffer.shape) != port.shape or buffer.dtype != port.dtype or buffer.kind != "intermediate":
                    raise ValueError("source workspace is not the same original tensor charged by planning")
                names.append(logical)
            workspaces[index] = abi.workspace
            workspace_source_buffers[index] = names
        if isinstance(abi, CRegionABI2):
            if requirement != abi.context:
                raise ValueError("compiled context differs from the original selected host reservation")
            declaration = next(
                a
                for a in compiled.artifacts
                if a.path.relative_to(compiled.directory).as_posix() == abi.context.declaration
            )
            identity = (
                abi.context,
                compiled.source_sha256,
                declaration.sha256,
                tuple((a.kind, a.role, a.sha256) for a in compiled.artifacts),
            )
            previous = contexts.get(abi.context.identity)
            if previous is not None and previous[1] != identity:
                raise ValueError("shared context identity has different compiled initialization or products")
            if previous is None:
                previous = (len(contexts), identity)
                contexts[abi.context.identity] = previous
            context_index = previous[0]
        elif requirement is not None:
            raise ValueError("selected caller-owned context was not discharged by a stateful C ABI")
        calls[index] = DispatchCall(
            abi.symbol, item.alternative.placement, buffer_alignment=abi.buffer_alignment, context_index=context_index
        )
        artifacts = {a.path.relative_to(compiled.directory).as_posix(): a for a in compiled.artifacts}
        # Identical stateless implementations may serve multiple source sites.
        # A repeated symbol with different source, ABI or products must not win
        # by link order. Retain one definition only after exact identity checks.
        identity = (
            compiled.backend,
            compiled.source_sha256,
            abi,
            tuple(sorted((key, a.kind, a.role, a.sha256) for key, a in artifacts.items())),
        )
        if abi.symbol in symbols and symbols[abi.symbol] != identity:
            raise ValueError("compiled regions conflict on a C wrapper symbol")
        if abi.symbol not in symbols:
            links.extend((compiled.directory / path).relative_to(directory).as_posix() for path in abi.link_inputs)
            payloads.extend((compiled.directory / path).relative_to(directory).as_posix() for path in abi.payloads)
            symbols[abi.symbol] = identity
        bindings.append(
            {
                "alternative": region.alternative,
                "provider": region.provider,
                "resource": item.proposal.resource,
                "memory_domain": item.proposal.memory_domain,
                "node": index,
                "source_sha256": compiled.source_sha256,
                "abi": abi.record(),
                "completion_admission": completion,
                "compilation": compiled.manifest.relative_to(directory).as_posix(),
            }
        )
    if set(calls) != set(range(len(graph.nodes))):
        raise ValueError("C control cannot omit an emitted source operation")
    external = {key for key, value in graph.buffers.items() if value.kind in {"arg", "const"}} | set(graph.results)
    alignment = memory.alignment_bytes
    external_bytes = sum(
        ((_buf_bytes(graph.buffers[key].shape, graph.buffers[key].dtype) + alignment - 1) // alignment) * alignment
        for key in external
    )
    reserved_bytes = reservations["domains"].get(memory.id, {}).get("total_bytes", 0)
    workspace_bytes = max(
        (((layout.bytes + alignment - 1) // alignment) * alignment for layout in workspaces.values()), default=0
    )
    available = memory.capacity_bytes - external_bytes - reserved_bytes - workspace_bytes
    if available < 0:
        raise ValueError("external C buffers exceed selected memory capacity")
    requirements = ArenaRequirements(
        max(1, available + workspace_bytes),
        tuple(2 * i for i in range(len(graph.nodes))),
        tuple(2 * i + 1 for i in range(len(graph.nodes))),
        alignment=alignment,
        dispatch_digest=dispatch_digest(graph),
    )
    program = build_program(
        selection.source.program,
        capability="compiled_pointer_calls",
        global_plan=selection.plan,
        global_plan_emitter=_PartitionEmitter(partition),
        memory_requirements=requirements,
    )
    if program.memory_plan.arena_bytes > available:
        raise ValueError("C arena and external buffers exceed selected memory capacity")
    if workspaces:
        workspace_offset = (program.memory_plan.arena_bytes + alignment - 1) // alignment * alignment
        if external_bytes + reserved_bytes + workspace_offset + workspace_bytes > memory.capacity_bytes:
            raise ValueError("materialized source workspace and C arena exceed selected memory capacity")
        calls = {
            index: replace(
                call,
                workspace_offset=workspace_offset,
                workspace_bytes=workspaces[index].bytes,
                workspace_alignment=workspaces[index].alignment,
            )
            if index in workspaces
            else call
            for index, call in calls.items()
        }
    read_only = tuple(key for key, value in graph.buffers.items() if value.kind == "arg")
    report, files = emit_program_c(program, name, calls, read_only_inputs=read_only)
    report.update(
        bindings=bindings,
        link_inputs=links,
        payloads=payloads,
        topology_sha256=selection.topology.sha256,
        memory_domain=memory.id,
        external_bytes=external_bytes,
        total_tensor_bytes=external_bytes + program.memory_plan.arena_bytes,
        backend_correctness="not_established",
        context="caller_owned_table" if contexts else "unused",
    )
    if contexts:
        report.update(
            host_context_reservations=reservations,
            context_indices={identity: row[0] for identity, row in contexts.items()},
            context_initialization="caller_obligation_not_observed",
            payload_loading="checked_embedding_bytes; loader_invocation_not_observed",
            completion_visibility="runtime_obligation; no_execution_observed",
            error_lifetime="retain_context_workspace_and_device_buffers_until_drained_or_reset",
            total_reserved_host_bytes=reserved_bytes,
        )
        if workspaces:
            report["context_arena_disjointness"] = (
                "caller_obligation; generated_image_entry_checks_before_initialization"
            )
            report["error_lifetime"] = "retain_context_workspace_arena_external_and_device_buffers_until_quiescent"
    if workspaces:
        report.update(
            source_workspace={
                "offset": workspace_offset,
                "reserved_bytes": workspace_bytes,
                "source_values": {str(index): layout.record() for index, layout in workspaces.items()},
                "original_buffers": workspace_source_buffers,
                "accounting": "original source values counted by source lifetime bound",
            },
            total_tensor_bytes=external_bytes + workspace_offset + workspace_bytes,
        )
    partition.revalidate()
    for product in products:
        product.revalidate()
    return report, files


# ---- application -------------------------------------------------------------------------------
# Compile a semantic application through selected target region implementations.
#
# The existing GlobalPlan, bounded arena, C dispatcher and native program-set
# publication remain authoritative. Providers supply legality and concrete region
# compilation; this module accounts for every source operation and boundary.


@dataclass(frozen=True)
class CompiledCRegion:
    symbol: str
    header: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    files: dict[str, bytes]
    shared_files: tuple[str, ...]
    programs: tuple[str, ...]
    description: dict


@dataclass(frozen=True)
class ApplicationCBinding:
    declarations: str
    implementation: str
    interface: dict


class SemanticApplicationAdapter(Protocol):
    def plan(self, source: KernelRequest) -> GlobalPlan: ...
    def compile_region(
        self, snapshot, region, alternative, name, segments, target_source, limits
    ) -> CompiledCRegion: ...
    def export_region(self, region, alternative, name, segments, target_source) -> CompiledCRegion: ...


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _render(
    source,
    adapter,
    name,
    capacity,
    compiled,
    *,
    plan,
    memory_plan=None,
    state=None,
    parameters=None,
    native_root=None,
):
    # Materialization consumes the choice already used to compile/export each
    # region. Re-running provider enumeration here could silently select a
    # different implementation with the same source boundary.
    outlined = outline_semantic_regions(source, entry=name, plan=plan)
    try:
        program = build_program(
            outlined.logical,
            global_plan=plan,
            global_plan_emitter=outlined,
            memory_plan=memory_plan,
            memory_requirements=lambda graph: ArenaRequirements(
                capacity,
                tuple(2 * i for i in range(len(graph.nodes))),
                tuple(2 * i + 1 for i in range(len(graph.nodes))),
                dispatch_digest=dispatch_digest(graph),
            ),
        )
    except ArenaPlanError as error:
        raise NativeCompilationError("resource_limit", "application buffer placement: " + str(error)) from error
    if len(compiled) != len(outlined.regions):
        raise ValueError("compiled regions do not cover the selected source plan")
    files, shared, calls, summaries, wrappers, headers, link_sources = {}, {}, {}, [], [], [], []
    placements = {item.id: item.placement for item in plan.selected}
    for index, ((key, region), lowered) in enumerate(zip(outlined.regions, compiled, strict=True)):
        node = outlined.emission.dispatch.nodes[index]
        if (
            set(lowered.outputs) != set(node.outputs)
            or len(lowered.outputs) != len(node.outputs)
            or not set(lowered.inputs) <= set(node.inputs)
            or any(region.node(value).effect != "constant" for value in set(node.inputs) - set(lowered.inputs))
        ):
            raise ValueError("compiled C ports omit a runtime source value or a required result")
        label = f"region_{index:03d}"
        for member, data in lowered.files.items():
            relative = Path(member)
            if relative.is_absolute() or relative.as_posix() != member or ".." in relative.parts:
                raise ValueError("region C export contains an unsafe path")
            path = f"regions/{label}/{member}"
            files["c-package/" + path] = data
            if member in lowered.shared_files:
                if member in shared and shared[member] != data:
                    raise ValueError("compiled regions disagree on a shared runtime source")
                first = member not in shared
                shared[member] = data
                if not first:
                    continue
            if member.endswith(".c"):
                link_sources.append(path)
        headers.append(f'#include "regions/{label}/{lowered.header}"')
        inputs = ",".join(f"inputs[{node.inputs.index(value)}]" for value in lowered.inputs) or "0"
        outputs = ",".join(f"outputs[{node.outputs.index(value)}]" for value in lowered.outputs) or "0"
        symbol = f"{name}_call_{index}"
        wrappers.append(f"""int {symbol}(void *context,const void *const *inputs,void *const *outputs) {{
  (void)inputs;
  const void *ordered_inputs[{max(1, len(lowered.inputs))}]={{{inputs}}};
  void *ordered_outputs[{max(1, len(lowered.outputs))}]={{{outputs}}};
  return {lowered.symbol}(context,ordered_inputs,ordered_outputs);
}}
""")
        calls[index] = DispatchCall(symbol, placements[key], context_index=index)
        summaries.append(
            {
                "id": key,
                "source_request_digest": region.digest(),
                "placement": placements[key],
                "context_index": index,
                "description": lowered.description,
                "native_programs": list(lowered.programs),
                "inputs": list(lowered.inputs),
                "outputs": list(lowered.outputs),
            }
        )
    if parameters is not None:
        parameters.check(source)
    driver, driver_files = emit_program_c(
        program,
        name + "_driver",
        calls,
        read_only_inputs=parameters.node_ids if parameters is not None else (),
    )
    interface, interface_files = emit_application_c(
        source,
        driver,
        driver_files,
        name,
        capacity=capacity,
        parameters=parameters,
    )

    host_linkage = merge_linkage_requirements(
        row["description"]["host_linkage"] for row in summaries if "host_linkage" in row["description"]
    )
    if host_linkage:
        interface["host_linkage"] = host_linkage
    for filename, text in {**driver_files, **interface_files}.items():
        files["c-package/" + filename] = text.encode()
        if filename.endswith(".c"):
            link_sources.append(filename)
    files[f"c-package/{name}_calls.c"] = ("\n".join(headers + wrappers)).encode()
    link_sources.append(name + "_calls.c")
    # Keep region sources as inspectable stages, but link one self-contained
    # translation unit in the platform application. Include guards retain one
    # copy of shared declarations; only the selected runtime source is linked.
    if state is not None:
        state.check(source)
        initialization, state_files = state.emit_initializers(name, input_order=interface["inputs"])
        state_record = {**state.record(), **initialization}
        if parameters is not None:
            state_record["ports"] = [
                {
                    **row,
                    "source_input_index": row["input_index"],
                    "input_index": interface["inputs"].index(row["node_id"]),
                }
                for row in state_record["ports"]
            ]
        interface["model_state"] = state_record
        files["application-state.json"] = _json(state_record)
        for filename, text in state_files.items():
            files["c-package/" + filename] = text.encode()
            if filename.endswith(".c"):
                link_sources.append(filename)
    region_link_sources = tuple(link_sources)
    combined_name = name + "_combined"
    binding_factory = getattr(adapter, "application_binding", None)
    binding = binding_factory(name, summaries, interface) if binding_factory else None
    if binding is not None and not isinstance(binding, ApplicationCBinding):
        raise TypeError("application binding must contain C declarations, implementation and interface")
    combined = c_source_pair(
        combined_name,
        public_headers={
            path.removeprefix("c-package/"): data.decode() for path, data in files.items() if path.endswith(".h")
        },
        private_headers={},
        sources={path: files["c-package/" + path].decode() for path in region_link_sources},
        declarations=binding.declarations if binding else "",
        implementation=binding.implementation if binding else "",
    )
    files.update({"c-package/combined/" + path: text.encode() for path, text in combined.items()})
    link_sources = ["combined/" + combined_name + ".c"]
    interface.update(source=link_sources[0], header="combined/" + combined_name + ".h")
    if binding:
        interface["platform_binding"] = binding.interface
    files.update(
        {
            "source-request.json": _json(source.record()),
            "selected-plan.json": _json(plan.to_dict()),
            "program.json": _json(program.to_dict()),
            "dispatch-c.json": _json(driver),
            "application-c.json": _json(interface),
            "compiled-regions.json": _json(summaries),
        }
    )
    files["c-package/README.md"] = f"""# {name}

Compile the C sources listed in the parent manifest.json's link_sources, relative to this
directory. combined/{combined_name}.c and combined/{combined_name}.h are a standalone pair for
the platform application; no region source files or checkout includes are needed.
Include {combined_name}.h from that pair and call {name}_invoke with a zero-initialized
{name}_driver_state, aligned workspace, ordered source-float input/output arrays
and a {name}_driver_contexts table. compiled-regions.json in the parent directory
lists each context's compiler-provided binding interface and scratch sizes.

Constants and immutable parameters are embedded. Every model result is copied
out only after all calls succeed. Preserve the state, all contexts and scratch
after an error until outstanding device work has completed or the platform has
recovered it. Do not reuse a timed-out invocation's buffers. No eager framework,
reference evaluator, driver, startup, linker script or final host ELF is included.

The application workspace covers tensor temporaries and output staging; private
region scratch is separate. This diagnostic precision choice does not qualify
original-model accuracy or execution on an integrated SoC or FPGA.
""".encode()
    if interface.get("pointer_abi") == "opaque_native_dense_buffers":
        files["c-package/README.md"] = files["c-package/README.md"].replace(
            b"ordered source-float input/output arrays",
            b"ordered dense input/output pointer arrays with the types in application-c.json",
        )
    if parameters is not None:
        files["c-package/README.md"] = files["c-package/README.md"].replace(
            b"Constants and immutable parameters are embedded.",
            b"Small constants are embedded. Immutable model parameters are in ../handoff/parameters.bin. "
            b"Bind that file once with the generated parameter binding, and keep its bytes immutable. "
            b"Parameter ports are excluded from the invocation input array.",
        )
    if host_linkage:
        files["c-package/README.md"] += (
            b"\nSelected CPU math requires the archives in application-c.json's host_linkage list. "
            b"Check their SHA256 values and link those exact files. Per-region descriptions "
            b"record the required CPU ISA, ABI, rounding mode and effects. Library identity "
            b"alone does not establish framework numerical equivalence.\n"
        )
    if native_root is not None:
        from merlin.compiler.lower.c.application import application_handoff

        files.update(application_handoff(source, interface, summaries, files, native_root))
    return outlined, summaries, interface, link_sources, files


def compile_semantic_application(
    source,
    *,
    adapter,
    snapshot,
    target_source,
    destination,
    limits,
    name,
    capacity,
    state=None,
    source_folding=None,
    parameters=None,
):
    """Compile every selected region; fail without substituting a different route."""
    if state is not None:
        state.check(source)
    if parameters is not None:
        parameters.check(source)
    destination.mkdir(parents=True, exist_ok=False)
    segments = destination / "segments"
    segments.mkdir()
    (destination / "source-request.json").write_bytes(_json(source.record()))
    if source_folding is not None:
        verify_source_preparation(source, source_folding)
        (destination / preparation_filename(source_folding)).write_bytes(_json(source_folding))
    plan = application_plan(source, adapter)
    outlined = outline_semantic_regions(source, entry=name, plan=plan)
    (destination / "selected-plan.json").write_bytes(_json(plan.to_dict()))
    selected = {item.id: item for item in plan.selected}
    lowered = [
        adapter.compile_region(snapshot, region, selected[key], f"region_{i:03d}", segments, target_source, limits)
        for i, (key, region) in enumerate(outlined.regions)
    ]
    _, summaries, interface, links, files = _render(
        source, adapter, name, capacity, lowered, plan=plan, state=state, parameters=parameters, native_root=destination
    )
    if parameters is not None:
        (destination / "handoff").mkdir(exist_ok=True)
        parameters.write_payload(destination / "handoff/parameters.bin")
    for filename, data in files.items():
        path = destination / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    rows = []
    for region in summaries:
        for filename in region["native_programs"]:
            path = destination / filename
            manifest_bytes = (path / "manifest.json").read_bytes()
            manifest = json.loads(manifest_bytes)
            rows.append(
                {
                    "index": len(rows),
                    "path": filename,
                    "manifest_sha256": _sha(manifest_bytes),
                    "request_digest": manifest["request_digest"],
                    "binary_sha256": manifest["binary_sha256"],
                }
            )
    if not rows:
        raise ValueError("native application has no accelerator program")
    manifest = {
        "schema": "merlin.semantic_application.v1",
        "engine": "merlin_native",
        "artifact_kind": "program_set",
        "compile_mode": source.lowering_policy,
        "qualification_scope": "diagnostic_only",
        "request_digest": source.digest(),
        "native_snapshot_sha256": _sha((snapshot.root / "manifest.json").read_bytes()),
        "target_identity": source.target_identity,
        "segments": rows,
        "application_name": name,
        "application_memory_bytes": capacity,
        "application_interface": interface,
        "link_sources": links,
        "regions": summaries,
        "stage_artifacts": {
            path.relative_to(destination).as_posix(): digest_file(path)[0]
            for path in sorted(destination.rglob("*"))
            if path.is_file()
        },
    }
    (destination / "manifest.json").write_bytes(_json(manifest))
    verify_semantic_application(
        source,
        adapter=adapter,
        snapshot=snapshot,
        target_source=target_source,
        artifact=destination,
        state=state,
        source_folding=source_folding,
        parameters=parameters,
    )
    return manifest


def verify_semantic_application(
    source,
    *,
    adapter,
    snapshot,
    target_source,
    artifact,
    state=None,
    source_folding=None,
    parameters=None,
):
    """Replay source partitioning, region exports, buffer planning and C generation."""
    manifest = json.loads((artifact / "manifest.json").read_bytes())
    if manifest.get("diagnostics_only"):
        raise NativeCompilationError(manifest["status"], manifest["reason"])
    if source_folding is not None:
        receipt = artifact / preparation_filename(source_folding)
        if not receipt.is_file() or json.loads(receipt.read_bytes()) != source_folding:
            raise ValueError("application constant folding differs from source intake")
        verify_source_preparation(source, source_folding)
    elif any((artifact / name).exists() for name in ("source-constant-folding.json", "source-preparation.json")):
        raise ValueError("application verification needs its source constant-folding receipt")
    if manifest.get("native_snapshot_sha256") != _sha((snapshot.root / "manifest.json").read_bytes()):
        raise ValueError("application snapshot differs from its compiler identity")
    members = {
        path.relative_to(artifact).as_posix()
        for path in artifact.rglob("*")
        if path.is_file() and path != artifact / "manifest.json"
    }
    if members != set(manifest["stage_artifacts"]):
        raise ValueError("application file inventory differs from its publication")
    verify_native_publication(
        artifact,
        manifest,
        engine="merlin_native",
        request_digest=source.digest(),
        target_identity=source.target_identity,
    )
    if parameters is not None and digest_file(artifact / "handoff/parameters.bin")[0] != parameters.payload_sha256:
        raise ValueError("model parameter blob differs from the staged weights")
    plan = application_plan(source, adapter)
    if (artifact / "selected-plan.json").read_bytes() != _json(plan.to_dict()):
        raise ValueError("application selected plan differs from source replay")
    outlined = outline_semantic_regions(source, entry=manifest["application_name"], plan=plan)
    selected = {item.id: item for item in plan.selected}
    compiled = [
        adapter.export_region(region, selected[key], f"region_{i:03d}", artifact / "segments", target_source)
        for i, (key, region) in enumerate(outlined.regions)
    ]
    _, summaries, interface, links, files = _render(
        source,
        adapter,
        manifest["application_name"],
        manifest["application_memory_bytes"],
        compiled,
        plan=plan,
        memory_plan=MemoryPlan(**json.loads((artifact / "program.json").read_bytes())["memory_plan"]),
        state=state,
        parameters=parameters,
        native_root=artifact,
    )
    if (
        manifest["regions"] != summaries
        or manifest["application_interface"] != interface
        or manifest["link_sources"] != links
    ):
        raise ValueError("application manifest differs from replayed region compilation")
    for filename, data in files.items():
        if (artifact / filename).read_bytes() != data:
            raise ValueError("application artifact differs from replayed source: " + filename)
    return {
        "status": "checked",
        "request_digest": source.digest(),
        "regions": len(compiled),
        "native_programs": len(manifest["segments"]),
        "source_wiring": "checked",
        "execution": "not_run",
    }


# ---- linkage -----------------------------------------------------------------------------------
# Merge explicit region library requirements without choosing their suppliers.


def merge_linkage_requirements(values):
    """Retain checked archive/runtime identities and reject conflicting symbols.

    These are unresolved linking obligations, not numerical or execution proof.
    Preserve the complete selected-runtime recipe rather than treating every
    accepted contract as an archive contract.
    """
    identities, symbols, suppliers = {}, {}, {}
    for value in values:
        selected = validate_linkage_contract(value)
        identity = {key: item for key, item in selected.items() if key != "symbols"}
        key = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        for symbol in selected["symbols"]:
            if symbol in suppliers and suppliers[symbol] != key:
                raise ValueError("compiled regions select different suppliers for the same symbol")
            suppliers[symbol] = key
        identities[key] = identity
        symbols.setdefault(key, set()).update(selected["symbols"])
    return [{**identities[key], "symbols": sorted(symbols[key])} for key in sorted(identities)]
