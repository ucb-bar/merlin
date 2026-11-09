import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin import api
from merlin.compiler.artifact import Artifact, Compilation, Deployment
from merlin.compiler.backend import layout
from merlin.compiler.backend.sourceproposal import (
    SourceProposal,
    admit_proposal,
    source_numerical_identity,
)
from merlin.compiler.build.image import CImageLink
from merlin.compiler.pipeline import PlannedCompiler, RegionBackend
from merlin.compiler.planning.execution import select_execution
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.offload_plan import bind_source
from merlin.system.topology import select_topology
from merlin.transforms.certificates import ir_text
from merlin.transforms.pipeline_state import PipelineContext

SOURCE = "module {func.func @forward(%x:tensor<2x3xf32>)->tensor<2x3xf32> {\n %a=tensor.collapse_shape %x [[0,1]] : tensor<2x3xf32> into tensor<6xf32>\n %b=tensor.expand_shape %a [[0,1]] output_shape [2,3] : tensor<6xf32> into tensor<2x3xf32>\n func.return %b:tensor<2x3xf32>\n}}"


def selected(tmp_path):
    path = tmp_path / "hardware.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema": "merlin.execution_topology.v1",
                "resources": [{"id": "cpu0", "kind": "cpu", "features": [], "shared_group": "cpu0"}],
                "memory_domains": [{"id": "ram", "capacity_bytes": 192, "alignment_bytes": 64}],
                "access": [
                    {
                        "resource": "cpu0",
                        "domain": "ram",
                        "read": True,
                        "write": True,
                        "addressability": "shared",
                        "visibility": "coherent",
                        "completion": "synchronous",
                    }
                ],
            }
        )
    )
    hardware = select_topology(path)
    source = bind_source(parse_mlir_text(SOURCE), entry="forward", context=PipelineContext(), graph_mode="ssa")

    def approve_layout(bound, proposal):
        assert len(proposal.alternative.nodes) == 1
        assert bound.operations[proposal.alternative.nodes[0]].name in {"tensor.collapse_shape", "tensor.expand_shape"}
        bound.module.verify()

    candidates = []
    for index, original in enumerate(source.cpu_alternatives):
        alternative = replace(original, id=f"layout_{index}", implementation="dense-layout", placement="cpu0")
        claim = SourceProposal(
            alternative,
            layout.name,
            "cpu0",
            "ram",
            "logical_tensor_v1",
            source.stamp.source_sha256,
            (source.operation_ordinals[index],),
            source_numerical_identity(source, (index,)),
        )
        candidates.append(admit_proposal(source, claim, hardware, validate_implementation=approve_layout))
    return (source, hardware, tuple(candidates))


OPTIONS = api.CompileOptions(parameters={"merlin.emit": "c-dispatch", "merlin.entry": "model"})


def compile_control(tmp_path, *, mutate=None, image=None, options=OPTIONS):
    source, _, candidates = selected(tmp_path)
    provider = layout
    if mutate:

        def changed(request):
            result = layout.compile(request)
            mutate(result)
            return result

        provider = SimpleNamespace(name=layout.name, validate=layout.validate, compile=changed)
    backend = PlannedCompiler(
        select_execution(source, candidates), {layout.name: RegionBackend(provider, "kernel")}, image=image
    )
    return api.compile(ir_text(source.module), backend=backend, output=tmp_path / "compiled", options=options)


def image_selection(tmp_path):
    gcc = shutil.which("gcc")
    if not gcc:
        pytest.skip("GCC required for actual C image execution")
    directory = tmp_path / "startup"
    directory.mkdir()
    path = directory / "main.c"
    path.write_text(
        '#include <stdint.h>\n#include <stdio.h>\n#include "model.h"\nint main(void) {\n  _Alignas(64) unsigned char arena[64];\n  uint32_t incoming[6] = {0x7f800001u, 0x7fc12345u, 0x80000000u, 0x7f800000u, 1u, 0xff800001u};\n  uint32_t outgoing[6] = {0};\n  model_buffer buffers[2] = {{incoming, sizeof incoming}, {outgoing, sizeof outgoing}};\n  model_state state = {0};\n  if (model_run(&state, arena, sizeof arena, buffers, 2, 0)) return 1;\n  if (state.completed != 2 || state.active) return 2;\n  for (int i = 0; i < 6; ++i) if (incoming[i] != outgoing[i]) return 3;\n  puts("selected-layout-bits-preserved");\n  return 0;\n}\n'
    )
    product = Artifact.from_file(path, kind="c-source", role="entry")
    startup = Compilation(directory, "caller.startup", product.sha256, (product,))
    startup.write()
    return CImageLink(
        compiler=Path(gcc).resolve(),
        flags=("-std=c11", "-O2", "-Wall", "-Wextra", "-Werror"),
        startup=startup,
        link_inputs=("main.c",),
        deployment=Deployment("fixture-cpu", "linux", "x86_64", "sysv", 1),
    )


def test_selected_image_links_actual_wrappers_runs_and_prepares_existing_external_handoff(tmp_path):
    from merlin.runtime.service import RunOptions, RunResult

    image = image_selection(tmp_path)
    options = replace(OPTIONS, system="fixture-cpu", platform="linux", harts=1)
    result = compile_control(tmp_path, image=image, options=options)
    assert result.deployment == image.deployment and result.metadata["stage"] == "linked_image"
    report = json.loads((result.directory / "plan.json").read_text())
    assert "link" not in report["remaining"] and "execution_evidence" in report["remaining"]
    linked = json.loads((result.directory / "image/link.json").read_text())
    assert linked["placement_execution"] == "not_observed" and linked["elf"]["machine"] == 62
    assert len(list((result.directory / "image/invocations").rglob("invocation.json"))) == 1

    def validate(request):
        assert request.deployment == image.deployment and request.artifact.kind == "elf"

    def run(request):
        actual = subprocess.run([request.artifact.path], check=True, capture_output=True, text=True)
        return RunResult(
            "fixture.host",
            request.artifact.sha256,
            "completed",
            actual.stdout,
            {"scope": "actual host CPU layout test; no accelerator or FireSim execution"},
        )

    executed = api.run(result, runner=SimpleNamespace(name="fixture.host", validate=validate, run=run))
    assert executed.stdout == "selected-layout-bits-preserved\n"
    prepared = api.run(result, runner="firesim", options=RunOptions(mode="prepare", output=tmp_path / "handoff"))
    request = json.loads((prepared.directory / "request.json").read_text())
    assert request["entry"] == "payload/image/program.elf" and request["execution"] == "external_operator_required"
    assert (prepared.directory / "payload/image/startup/main.c").is_file()
    actual = subprocess.run(["sh", str(prepared.directory / "invoke.sh")], capture_output=True, text=True, check=True)
    assert actual.stdout == executed.stdout
    api.inspect(result.manifest)


def stateful_control(
    tmp_path, *, corrupt=None, share=False, split_host=False, topology_edit=None, platform=None, image_factory=None
):
    """Source-bound software ABI fixture; no device math/execution claim."""
    from merlin.compiler.backend.cabi import CRegionABI
    from merlin.compiler.backend.stateful import CContext, CRegionABI2, EmbeddedPayload, render_u32_payload

    source, hardware, original = selected(tmp_path)
    document = yaml.safe_load(hardware.path.read_text())
    document["memory_domains"][0]["capacity_bytes"] = 1024
    if split_host:
        document["memory_domains"].append({"id": "host", "capacity_bytes": 256, "alignment_bytes": 64})
        document["access"].append(
            {
                "resource": "cpu0",
                "domain": "host",
                "read": True,
                "write": True,
                "addressability": "shared",
                "visibility": "coherent",
                "completion": "synchronous",
            }
        )
    if topology_edit is not None:
        topology_edit(document)
    hardware.path.write_text(yaml.safe_dump(document))
    hardware = select_topology(hardware.path)
    requirements = []
    admitted = []
    for index, item in enumerate(original):
        requirement = CContext(
            "shared" if share else f"context_{index}",
            "a" * 64 if platform is None else next((a.sha256 for a in platform.artifacts if a.kind == "json")),
            "cpu0",
            "host" if split_host else "ram",
            64,
            16,
            64,
            16,
            "context.h",
            "software_context",
            "software_context_init",
        )
        requirements.append(requirement)
        alternative = replace(item.alternative, metadata=tuple(sorted((requirement.selection_metadata(),))))
        proposal = replace(item.proposal, alternative=alternative)

        def admit(bound, claim):
            assert bound is source and claim.alternative.nodes in ((0,), (1,))
            assert claim.alternative.scratch_bytes == 0

        admitted.append(admit_proposal(source, proposal, hardware, validate_implementation=admit))

    def compile_region(request):
        result = layout.compile(request)
        index = int(request.output.name)
        context = requirements[index]
        original_call = CRegionABI.from_compilation(result)
        header = request.output / "context.h"
        header.write_text(
            '#include <stddef.h>\n#include <stdint.h>\ntypedef struct { unsigned calls; uint32_t loaded; void *workspace; size_t bytes; } software_context;\n_Static_assert(sizeof(software_context)<=64 && _Alignof(software_context)<=16, "context reservation");\nstatic inline int software_context_init(software_context *c, void *workspace, size_t bytes) {\n if(!c || !workspace || bytes<64) return 1;\n *c=(software_context){0,0,workspace,bytes}; return 0;\n}\n'
        )
        binary = request.output / "program.bin"
        binary.write_bytes(bytes.fromhex("0403020108070605"))
        embedded = request.output / "program.h"
        embedded.write_text(render_u32_payload("software_program", binary.read_bytes()))
        wrapper = request.output / "context.c"
        symbol = original_call.symbol + "_context"
        wrapper.write_text(
            f'#include "context.h"\n#include "program.h"\nextern int {original_call.symbol}(void *, const void *const *, void *const *);\nint {symbol}(void *opaque, const void *const *inputs, void *const *outputs) {{\n software_context *c=opaque;\n if(!c || !c->workspace || c->bytes<64) return 37;\n c->loaded=software_program[0];\n int result={original_call.symbol}(0,inputs,outputs);\n if(!result) ++c->calls;\n return result;\n}}\n'
        )
        call = replace(
            original_call,
            symbol=symbol,
            link_inputs=(*original_call.link_inputs, "context.c"),
            payloads=("program.bin",),
        )
        abi = CRegionABI2(call, context, (EmbeddedPayload("program.bin", "program.h", "software_program"),))
        artifacts = (
            *result.artifacts,
            Artifact.from_file(header, kind="c-header", role="context"),
            Artifact.from_file(binary, kind="device-binary", role="device-program"),
            Artifact.from_file(embedded, kind="c-header", role="embedded-program"),
            Artifact.from_file(wrapper, kind="c-source", role="context-wrapper"),
        )
        result = replace(result, artifacts=artifacts, metadata={**result.metadata, "c_call": abi.record()})
        if corrupt:
            corrupt(result)
        return result

    provider = SimpleNamespace(name=layout.name, validate=layout.validate, compile=compile_region)
    selection = select_execution(source, tuple(admitted))
    backend = PlannedCompiler(
        selection,
        {layout.name: RegionBackend(provider, "kernel")},
        image=image_factory(tuple(requirements)) if image_factory is not None else None,
    )
    return api.compile(ir_text(source.module), backend=backend, output=tmp_path / "compiled", options=OPTIONS)


def test_selected_stateful_contexts_reserve_host_storage_embed_exact_bytes_and_execute(tmp_path):
    gcc = shutil.which("gcc")
    if not gcc:
        pytest.skip("GCC required for actual stateful C control execution")
    result = stateful_control(tmp_path, split_host=True)
    report = json.loads((result.directory / "control/control.json").read_text())
    assert report["context_table"]["minimum_count"] == 2
    assert [row["context_index"] for row in report["calls"]] == [0, 1]
    assert report["context_indices"] == {"context_0": 0, "context_1": 1}
    assert report["host_context_reservations"]["domains"] == {
        "host": {"state_bytes": 128, "workspace_bytes": 64, "total_bytes": 192}
    }
    assert report["total_reserved_host_bytes"] == 0
    assert report["context_initialization"] == "caller_obligation_not_observed"
    assert report["completion_visibility"] == "runtime_obligation; no_execution_observed"
    assert len(report["payloads"]) == 2 and result.deployment is None
    first = result.directory / "regions/0"
    startup = tmp_path / "stateful.c"
    startup.write_text(
        '#include <stdint.h>\n#include <string.h>\n#include "model.h"\n#include "context.h"\nint main(void) {\n _Alignas(64) unsigned char arena[128], workspace0[64], workspace1[64];\n _Alignas(64) uint32_t input[6]={0x7f800001,0x7fc12345,0x80000000,0x7f800000,1,0xff800001};\n _Alignas(64) uint32_t output[6]={0};\n software_context first,second;\n if(software_context_init(&first,workspace0,sizeof(workspace0)) ||\n    software_context_init(&second,workspace1,sizeof(workspace1))) return 1;\n void *items[]={&first,&second}; model_contexts contexts={items,2};\n model_buffer buffers[]={{input,sizeof(input)},{output,sizeof(output)}};\n model_state state={0};\n if(model_run(&state,arena,sizeof(arena),buffers,2,&contexts)) return 2;\n if(first.calls!=1 || second.calls!=1 || first.loaded!=0x01020304u || second.loaded!=0x01020304u ||\n    memcmp(input,output,sizeof(input)) || state.active || state.completed!=2) return 3;\n /* Missing context is a real wrapper error. Driver remains active and cannot\n  * reuse buffers or reenter until its owner drains/resets the endpoint. */\n state=(model_state){0}; items[0]=0;\n if(model_run(&state,arena,sizeof(arena),buffers,2,&contexts)!=MODEL_KERNEL_ERROR || !state.active ||\n    state.completed || state.kernel_status!=37) return 4;\n if(model_run(&state,arena,sizeof(arena),buffers,2,&contexts)!=MODEL_BUSY) return 5;\n return 0;\n}\n'
    )
    executable = tmp_path / "stateful"
    subprocess.run(
        [
            gcc,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I" + str(result.directory / "control"),
            "-I" + str(first),
            str(startup),
            *(str(result.directory / name) for name in report["link_inputs"]),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
