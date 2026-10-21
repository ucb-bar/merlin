"""Compile a semantic application through selected target region implementations.

The existing GlobalPlan, bounded arena, C dispatcher and native program-set
publication remain authoritative. Providers supply legality and concrete region
compilation; this module accounts for every source operation and boundary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from merlin.runtime.program import build_program
from merlin.semantic_compiler.model import KernelRequest
from merlin.semantic_compiler.target_binding import NativeCompilationError, verify_native_publication
from merlin.xdsl_dialects.lowering.arena_plan import ArenaPlanError, ArenaRequirements, MemoryPlan
from merlin.xdsl_dialects.lowering.global_plan import GlobalPlan
from merlin.xdsl_dialects.lowering.global_plan_emission import dispatch_digest

from .application_c import emit_application_c
from .application_plan import application_plan
from .c_sources import c_source_pair
from .dispatch_c import DispatchCall, emit_program_c
from .semantic_regions import outline_semantic_regions


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


def _render(source, adapter, name, capacity, compiled, *, memory_plan=None, state=None, native_root=None):
    plan = application_plan(source, adapter)
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
    driver, driver_files = emit_program_c(program, name + "_driver", calls)
    interface, interface_files = emit_application_c(source, driver, driver_files, name, capacity=capacity)
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
        initialization, state_files = state.emit_initializers(name)
        state_record = {**state.record(), **initialization}
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
    if native_root is not None:
        from .application_handoff import application_handoff

        files.update(application_handoff(source, interface, summaries, files, native_root))
    return outlined, summaries, interface, link_sources, files


def compile_semantic_application(
    source, *, adapter, snapshot, target_source, destination, limits, name, capacity, state=None, source_folding=None
):
    """Compile every selected region; fail without substituting a different route."""
    if state is not None:
        state.check(source)
    destination.mkdir(parents=True, exist_ok=False)
    segments = destination / "segments"
    segments.mkdir()
    (destination / "source-request.json").write_bytes(_json(source.record()))
    if source_folding is not None:
        from merlin.semantic_compiler.constant_folding import verify_constant_folding

        verify_constant_folding(source, source_folding)
        (destination / "source-constant-folding.json").write_bytes(_json(source_folding))
    plan = application_plan(source, adapter)
    outlined = outline_semantic_regions(source, entry=name, plan=plan)
    (destination / "selected-plan.json").write_bytes(_json(plan.to_dict()))
    selected = {item.id: item for item in plan.selected}
    lowered = [
        adapter.compile_region(snapshot, region, selected[key], f"region_{i:03d}", segments, target_source, limits)
        for i, (key, region) in enumerate(outlined.regions)
    ]
    _, summaries, interface, links, files = _render(
        source, adapter, name, capacity, lowered, state=state, native_root=destination
    )
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
            path.relative_to(destination).as_posix(): _sha(path.read_bytes())
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
    )
    return manifest


def verify_semantic_application(source, *, adapter, snapshot, target_source, artifact, state=None, source_folding=None):
    """Replay source partitioning, region exports, buffer planning and C generation."""
    manifest = json.loads((artifact / "manifest.json").read_bytes())
    if manifest.get("diagnostics_only"):
        raise NativeCompilationError(manifest["status"], manifest["reason"])
    receipt = artifact / "source-constant-folding.json"
    if source_folding is not None:
        from merlin.semantic_compiler.constant_folding import verify_constant_folding

        if not receipt.is_file() or json.loads(receipt.read_bytes()) != source_folding:
            raise ValueError("application constant folding differs from source intake")
        verify_constant_folding(source, source_folding)
    elif receipt.exists():
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
    plan = application_plan(source, adapter)
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
        memory_plan=MemoryPlan(**json.loads((artifact / "program.json").read_bytes())["memory_plan"]),
        state=state,
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
