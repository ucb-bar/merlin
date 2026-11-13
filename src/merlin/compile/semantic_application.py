"""Compile a semantic application through selected target region implementations.

The existing GlobalPlan, bounded arena, C dispatcher and native program-set
publication remain authoritative. Providers supply legality and concrete region
compilation; this module accounts for every source operation and boundary.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from merlin.common.content_store import digest_file
from merlin.runtime.program import build_program
from merlin.semantic_compiler.model import KernelRequest
from merlin.semantic_compiler.codegen import current_codegen_options
from merlin.semantic_compiler.materialization import (
    MaterializationCache, canonical_request, materialization_key, materialization_region, materialization_session,
)
from merlin.semantic_compiler.source_preparation import preparation_filename, verify_source_preparation
from merlin.semantic_compiler.target_binding import NativeCompilationError, verify_native_publication
from merlin.xdsl_dialects.lowering.arena_plan import ArenaPlanError, ArenaRequirements, MemoryPlan
from merlin.xdsl_dialects.lowering.global_plan import GlobalPlan
from merlin.xdsl_dialects.lowering.global_plan_emission import dispatch_digest

from .application_c import emit_application_c
from .application_plan import application_plan
from .c_sources import c_source_pair
from .dispatch_c import DispatchCall, emit_program_c, prepare_dispatch_loop_from_plan
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


def _render(
    source, adapter, name, capacity, compiled, *, memory_plan=None, state=None, parameters=None, native_root=None,
    emit_region_loop=False,
):
    if type(emit_region_loop) is not bool:
        raise TypeError("region loop emission needs an explicit Boolean")
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
    if parameters is not None:
        parameters.check(source)
    loop_plan, dispatch_loop = None, None
    if emit_region_loop:
        from .region_loop import analyze_region_loop

        loop_parameters = getattr(adapter, "region_loop_parameters", None)
        loop = analyze_region_loop(
            [region.record() for _, region in outlined.regions],
            parameter_attributes=loop_parameters or (),
        )
        loop_plan = {
            "schema": "merlin.application_loop_plan.v1",
            "source_request_digest": source.digest(),
            "region_loop": loop,
        }
        files["application-loop-plan.json"] = _json(loop_plan)
    if emit_region_loop:
        dispatch_loop = prepare_dispatch_loop_from_plan(
            program, calls, loop_plan, requests=[region.record() for _, region in outlined.regions],
            source_request_digest=source.digest(),
        )
    driver, driver_files = emit_program_c(
        program, name + "_driver", calls,
        read_only_inputs=parameters.node_ids if parameters is not None else (),
        loop=dispatch_loop,
    )
    interface, interface_files = emit_application_c(
        source, driver, driver_files, name, capacity=capacity, parameters=parameters,
    )
    from merlin.targetgen.host_linkage_contract import merge_linkage_requirements

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
            state_record["ports"] = [{**row, "source_input_index": row["input_index"],
                                      "input_index": interface["inputs"].index(row["node_id"])}
                                     for row in state_record["ports"]]
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
    if emit_region_loop:
        files["c-package/README.md"] += (
            b"\nThe invocation uses the authenticated counted dispatch window in application-loop-plan.json "
            b"for every recorded trip. It regroups the original call table and does not yet share programs. "
            b"The low-level driver _run entry remains available for flat dispatch.\n"
        )
    if host_linkage:
        files["c-package/README.md"] += (
            b"\nSelected CPU math requires the archives in application-c.json's host_linkage list. "
            b"Check their SHA256 values and link those exact files. Per-region descriptions "
            b"record the required CPU ISA, ABI, rounding mode and effects. Library identity "
            b"alone does not establish framework numerical equivalence.\n"
        )
    if native_root is not None:
        from .application_handoff import application_handoff

        files.update(application_handoff(source, interface, summaries, files, native_root))
    return outlined, summaries, interface, link_sources, files


def _region_materialization_contract(adapter, region, alternative):
    identity_hook = getattr(adapter, "materialization_identity", None)
    if identity_hook is None:
        # Without a provider proof retain the complete selected
        # alternative and source identity, sacrificing reuse.
        contract = {
            "target_choice": alternative.to_dict(),
            "source_identity_policy": "preserve",
            "compilation_identity": {
                "adapter_module": type(adapter).__module__,
                "adapter_type": type(adapter).__qualname__,
            },
        }
    else:
        contract = identity_hook(region, alternative)
    if not isinstance(contract, dict) or set(contract) != {
        "target_choice", "compilation_identity", "source_identity_policy"
    }:
        raise ValueError("provider materialization identity contract differs")
    # Snapshot the JSON contract so later provider callbacks cannot change
    # normalization policy after its identity has been checked and keyed.
    return json.loads(json.dumps(contract, sort_keys=True, allow_nan=False))


def _region_materialization_key(adapter, snapshot, region, alternative, target_source, limits, *, contract=None):
    if contract is None:
        contract = _region_materialization_contract(adapter, region, alternative)
    return materialization_key(
        region, target_choice=contract["target_choice"], options=current_codegen_options(),
        source_identity_policy=contract["source_identity_policy"],
        compilation_identity={
            "provider": contract["compilation_identity"],
            "snapshot_manifest_sha256": _sha((snapshot.root / "manifest.json").read_bytes()),
            "snapshot_root": str(snapshot.root.resolve()),
            "target_source": str(target_source.resolve()),
            "limits": limits.record() if hasattr(limits, "record") else limits,
        },
    )


def compile_semantic_application(
    source, *, adapter, snapshot, target_source, destination, limits, name, capacity, state=None, source_folding=None,
    parameters=None, allow_host_only=False, reuse_materializations=False, materialization_stats=None,
    reuse_region_materializations=False, emit_region_loop=False, deterministic_receipts=False,
):
    """Compile and authenticate every original publication in one cache session.

    Targets explicitly opt into canonical materialization. Unsupported regions
    retain original compilation; accepted regions compile an alpha-normalized
    source once and bind each original through target-owned publication checks.
    """
    arguments = locals().copy()
    arguments.pop("materialization_stats")
    arguments.pop("deterministic_receipts")
    if type(deterministic_receipts) is not bool:
        raise TypeError("deterministic receipts need an explicit Boolean")
    publication_options = getattr(adapter, "publication_options", None)
    if deterministic_receipts and publication_options is None:
        raise ValueError("target does not support deterministic receipts")
    publication_scope = (publication_options(deterministic_receipts=deterministic_receipts)
                         if publication_options is not None else nullcontext())
    if type(reuse_materializations) is not bool:
        raise TypeError("materialization reuse needs an explicit Boolean")
    if type(reuse_region_materializations) is not bool:
        raise TypeError("region materialization reuse needs an explicit Boolean")
    enabled = reuse_materializations and getattr(adapter, "supports_materialization_cache", False) is True
    stats = {}
    try:
        with publication_scope, materialization_session(enabled=enabled) as stats:
            return _compile_semantic_application(**arguments, _stats=stats)
    finally:
        if materialization_stats is not None:
            materialization_stats.clear()
            materialization_stats.update(stats)


def _compile_semantic_application(
    source, *, adapter, snapshot, target_source, destination, limits, name, capacity, state=None, source_folding=None,
    parameters=None, allow_host_only=False, reuse_materializations=False, _stats=None,
    reuse_region_materializations=False, emit_region_loop=False,
):
    """Compile selected regions and replay every original publication.

    Explicitly opted-in adapters may memoize deterministic internal tool calls.
    An optional ``materialization_identity(region, alternative)`` provider hook
    supplies JSON target choices and provider compilation identity plus an
    explicit source identity policy. Canonical region identity then partitions
    exact tool reuse. Optional target materialize/bind hooks may additionally
    reuse checked canonical compilation, preserving every original publication.
    Both reuse options default off. ``reuse_materializations`` enables exact
    tool reuse and ``reuse_region_materializations`` enables canonical bodies.
    Verification always runs fresh tools in its own session. Optional caller
    statistics stay outside published application bytes.
    """
    if type(reuse_materializations) is not bool:
        raise TypeError("materialization reuse needs an explicit Boolean")
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
    lowered = []
    cache = MaterializationCache("merlin.application_region.v1")
    declined = object()
    # Canonical artifacts are invocation-local and never part of an original
    # authenticated publication. Only the target binding hook publishes them.
    with tempfile.TemporaryDirectory(prefix="canonical-regions-", dir=destination.parent) as temporary:
        canonical_root = Path(temporary)
        for i, (key, region) in enumerate(outlined.regions):
            alternative = selected[key]
            _stats["region_calls"] += 1
            contract = _region_materialization_contract(adapter, region, alternative)
            region_key = _region_materialization_key(
                adapter, snapshot, region, alternative, target_source, limits, contract=contract)
            with materialization_region(region_key):
                supports = getattr(adapter, "supports_region_materialization", None)
                if (reuse_region_materializations
                        and getattr(adapter, "supports_materialization_cache", False) is True and supports is not None
                        and supports(region, alternative)):
                    materialized = cache.get(region_key)
                    if materialized is None:
                        canonical = canonical_request(region, source_identity_policy=contract["source_identity_policy"])
                        _stats["region_compiles"] += 1
                        _stats["region_materializations"] += 1
                        materialized = adapter.materialize_region(
                            snapshot, canonical, alternative, f"canonical_{i:03d}",
                            canonical_root, target_source, limits)
                        if materialized is None:
                            # A target may find a construction it cannot bind
                            # after checking the canonical materialization.
                            # Cache the refusal, then compile original requests.
                            materialized = declined
                        cache.put(region_key, materialized)
                    elif materialized is not declined:
                        _stats["region_hits"] += 1
                    if materialized is declined:
                        _stats["region_compiles"] += 1
                        lowered.append(adapter.compile_region(
                            snapshot, region, alternative, f"region_{i:03d}", segments, target_source, limits))
                    else:
                        lowered.append(adapter.bind_materialization(
                            materialized, region, alternative, f"region_{i:03d}", segments, target_source))
                else:
                    _stats["region_compiles"] += 1
                    lowered.append(adapter.compile_region(
                        snapshot, region, alternative, f"region_{i:03d}", segments, target_source, limits))
    _, summaries, interface, links, files = _render(
        source, adapter, name, capacity, lowered, state=state, parameters=parameters, native_root=destination,
        emit_region_loop=emit_region_loop,
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
    if not rows and (not allow_host_only or not summaries
            or any(row["placement"] != "host" or row["native_programs"] for row in summaries)):
        raise ValueError("native application has no accelerator program")
    manifest = {
        "schema": "merlin.semantic_application.v1",
        "engine": "merlin_native",
        "artifact_kind": "program_set" if rows else "host_application",
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
    if emit_region_loop:
        manifest["dispatch_loop"] = True
    (destination / "manifest.json").write_bytes(_json(manifest))
    compilation_stats = {key: value for key, value in _stats.items() if key != "elapsed_seconds"}
    verification_stats = {}
    verify_semantic_application(
        source,
        adapter=adapter,
        snapshot=snapshot,
        target_source=target_source,
        artifact=destination,
        state=state,
        source_folding=source_folding,
        parameters=parameters,
        allow_host_only=allow_host_only,
        limits=limits,
        materialization_stats=verification_stats,
    )
    # Account for the independent fresh replay in total invocation counters.
    for key, value in verification_stats.items():
        if key != "elapsed_seconds":
            _stats[key] += value
    _stats["compilation"] = compilation_stats
    _stats["verification"] = verification_stats
    return manifest


def verify_semantic_application(
    source, *, adapter, snapshot, target_source, artifact, state=None, source_folding=None, parameters=None,
    allow_host_only=False, limits=None, reuse_materializations=False, materialization_stats=None,
):
    """Authenticate original exports with fresh tools in an independent session.

    ``reuse_materializations`` is accepted for API compatibility. Verification
    always executes tools fresh, including when nested in an enabled compile.
    """
    arguments = locals().copy()
    arguments.pop("materialization_stats")
    if type(reuse_materializations) is not bool:
        raise TypeError("materialization reuse needs an explicit Boolean")
    try:
        with materialization_session(enabled=False) as stats:
            return _verify_semantic_application(**arguments, _stats=stats)
    finally:
        if materialization_stats is not None:
            materialization_stats.clear()
            materialization_stats.update(stats)


def _verify_semantic_application(
    source, *, adapter, snapshot, target_source, artifact, state=None, source_folding=None, parameters=None,
    allow_host_only=False, limits=None, reuse_materializations=False, _stats=None,
):
    """Replay source partitioning, region exports, buffer planning and C generation."""
    manifest = json.loads((artifact / "manifest.json").read_bytes())
    emit_region_loop = manifest.get("dispatch_loop", False)
    if type(emit_region_loop) is not bool:
        raise ValueError("application dispatch loop option must be Boolean")
    if manifest.get("artifact_kind") == "host_application" and not allow_host_only:
        raise ValueError("host-only application requires explicit allow_host_only selection")
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
    outlined = outline_semantic_regions(source, entry=manifest["application_name"], plan=plan)
    selected = {item.id: item for item in plan.selected}
    compiled = []
    for i, (key, region) in enumerate(outlined.regions):
        alternative = selected[key]
        region_key = _region_materialization_key(
            adapter, snapshot, region, alternative, target_source, limits)
        with materialization_region(region_key):
            _stats["region_replays"] += 1
            compiled.append(adapter.export_region(
                region, alternative, f"region_{i:03d}", artifact / "segments", target_source))
    _, summaries, interface, links, files = _render(
        source,
        adapter,
        manifest["application_name"],
        manifest["application_memory_bytes"],
        compiled,
        memory_plan=MemoryPlan(**json.loads((artifact / "program.json").read_bytes())["memory_plan"]),
        state=state,
        parameters=parameters,
        native_root=artifact,
        emit_region_loop=emit_region_loop,
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
