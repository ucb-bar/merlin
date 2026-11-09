"""Selected mixed ABI execution; fake MMIO is independent of CPU numerical checks."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin import api
from merlin.compiler.artifact import Artifact, Compilation, Deployment
from merlin.compiler.backend import scalar
from merlin.compiler.backend.cabi import CRegionABI, TensorPort
from merlin.compiler.backend.sourceproposal import (
    SourceProposal,
    admit_proposal,
    proposal_boundary,
    source_numerical_identity,
)
from merlin.compiler.backend.stateful import CContext, CRegionABI2, EmbeddedPayload, render_u32_payload
from merlin.compiler.backend.workspace import CWorkspaceABI
from merlin.compiler.build.contextentry import CImageContext, CImageEntry, ReservedImageInterval
from merlin.compiler.build.image import CImageLink
from merlin.compiler.lower.llvm.workspace import plan
from merlin.compiler.pipeline import PlannedCompiler, RegionBackend
from merlin.compiler.planning.execution import select_execution
from merlin.compiler.request import CompileOptions
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower import target_data_layout, toolchain
from merlin.llvmlower.offload_plan import bind_source
from merlin.runtime import linux
from merlin.system.topology import select_topology
from merlin.transforms.certificates import ir_text
from merlin.transforms.pipeline_state import PipelineContext

SOURCE = """module {func.func @compute(%x:tensor<2x2xf32>,%unused:tensor<4xf32>)->tensor<4xf32> {
 %w=arith.constant dense<[[1.0,2.0],[3.0,4.0]]>:tensor<2x2xf32>
 %z=arith.constant 0.0:f32
 %e=tensor.empty():tensor<2x2xf32>
 %f=linalg.fill ins(%z:f32) outs(%e:tensor<2x2xf32>)->tensor<2x2xf32>
 %a=linalg.matmul ins(%x,%w:tensor<2x2xf32>,tensor<2x2xf32>)
 outs(%f:tensor<2x2xf32>)->tensor<2x2xf32>
 %one=arith.constant 1.0:f32
 %e2=tensor.empty():tensor<2x2xf32>
 %f2=linalg.fill ins(%one:f32) outs(%e2:tensor<2x2xf32>)->tensor<2x2xf32>
 %b=linalg.matmul ins(%a,%w:tensor<2x2xf32>,tensor<2x2xf32>)
 outs(%f2:tensor<2x2xf32>)->tensor<2x2xf32>
 %r=tensor.collapse_shape %b [[0,1]]:tensor<2x2xf32> into tensor<4xf32>
 func.return %r:tensor<4xf32>
}}"""

DEVICE_HEADER = """#ifndef SOFTWARE_DEVICE_H
#define SOFTWARE_DEVICE_H
#include <stddef.h>
#include <stdint.h>
typedef struct { void *workspace; size_t bytes; unsigned failed,effects,starts;
 volatile unsigned control,status; uint32_t words[2]; } software_device;
int software_initialize(software_device *,void *,size_t);
int software_call(void *,const void *const *,void *const *);
void *software_context_storage(void);
void *software_workspace_storage(void);
void software_fail(unsigned);
unsigned software_effects(void);
#endif
"""
DEVICE = """#include "device.h"
#include "program.h"
#include <string.h>
static software_device *owner;
static unsigned fail;
_Static_assert(sizeof(software_device)+sizeof(owner)+sizeof(fail)<=128,"selected device context");
int software_initialize(software_device *c,void *workspace,size_t bytes) {
 if(!c || !workspace || bytes<64 || owner) return 1;
 *c=(software_device){.workspace=workspace,.bytes=bytes}; owner=c; return 0;
}
void *software_context_storage(void) { return owner; }
void *software_workspace_storage(void) { return owner ? owner->workspace : 0; }
void software_fail(unsigned value) { fail=value; }
unsigned software_effects(void) { return owner ? owner->effects : 0; }
int software_call(void *opaque,const void *const *input,void *const *output) {
 software_device *c=opaque;
 if(c!=owner || !c || c->failed || !c->workspace || c->bytes<64) return 9;
 /* Test-only fake MMIO protocol; these opaque words are not target instructions. */
 c->words[0]=software_program[0]; c->words[1]=software_program[1]; c->effects+=2;
 memcpy(c->workspace,input[0],16); c->control=1; ++c->effects; ++c->starts;
 if(fail) { c->failed=1; return 10; }
 c->status=1; ++c->effects;
 memcpy(output[0],c->workspace,16); c->control=0; return 0;
}
"""

MAIN = """#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "entry.h"
#include "regions/1/device.h"
int main(void) {
 _Alignas(64) unsigned char arena[MODEL_ARENA_BYTES], output_space[64];
 _Alignas(64) float input[4]={1,2,-1,-2},unused[4]={0};
 float *output=(float *)output_space;
 float expected[4]={38,55,-36,-53},before[4]={1,2,-1,-2};
 model_buffer buffers[]={{input,sizeof(input)},{unused,sizeof(unused)},{output,16}};
 model_state state={0};
 if(image_run(&state,arena,sizeof(arena),buffers,3) || memcmp(expected,output,16) ||
    memcmp(input,before,16) || state.active || state.completed!=2) return 1;
 unsigned effects=software_effects();
 /* All persistent state/workspace aliases refuse before init or another call. */
 model_buffer invalid[3]={{input,16},{software_context_storage(),16},{output,16}};
 state=(model_state){0};
 if(image_run(&state,arena,sizeof(arena),invalid,3)!=MODEL_INVALID || software_effects()!=effects) return 2;
 invalid[1].data=software_workspace_storage();
 if(image_run(&state,arena,sizeof(arena),invalid,3)!=MODEL_INVALID || software_effects()!=effects) return 3;
 if(image_run(&state,software_context_storage(),sizeof(arena),buffers,3)!=MODEL_INVALID ||
    software_effects()!=effects) return 4;
 /* Even an unused argument and the descriptor table cannot alias private storage. */
 if(image_run(&state,arena,sizeof(arena),(model_buffer *)software_context_storage(),3)!=MODEL_INVALID ||
    software_effects()!=effects) return 9;
 /* A descriptor table cannot occupy the arena or any external data bytes. */
 model_buffer *table=(model_buffer *)arena; memcpy(table,buffers,sizeof(buffers));
 void *items[]={software_context_storage()}; model_contexts contexts={items,1};
 if(model_run(&state,arena,sizeof(arena),table,3,&contexts)!=MODEL_INVALID || state.active ||
    software_effects()!=effects) return 10;
 if(image_run(&state,arena,sizeof(arena),table,3)!=MODEL_INVALID || software_effects()!=effects) return 5;
 table=(model_buffer *)output_space; memcpy(table,buffers,sizeof(buffers));
 if(image_run(&state,arena,sizeof(arena),table,3)!=MODEL_INVALID || software_effects()!=effects) return 6;
 memcpy(output,expected,16);
 /* A later device error retains caller arena plus all context/workspace state. */
 software_fail(1); state=(model_state){0};
 if(image_run(&state,arena,sizeof(arena),buffers,3)!=MODEL_KERNEL_ERROR || !state.active ||
    state.completed!=1 || memcmp(input,before,16) || memcmp(output,expected,16)) return 7;
 effects=software_effects(); state=(model_state){0};
 if(image_run(&state,arena,sizeof(arena),buffers,3)!=MODEL_BUSY || software_effects()!=effects) return 8;
 puts("native-cpu-workspace-and-fake-mmio-plan"); return 0;
}
"""


@pytest.mark.parametrize("second_seed", [0.0, 1.0])
def test_native_scalar_workspace_and_persistent_fake_mmio_share_serial_image(tmp_path, second_seed):
    compiler = Path(toolchain.clang())
    if toolchain._env("MERLIN_COMPILER_DRIVER", "python") != "stock-cli" or not compiler.is_file():
        pytest.skip("explicit actual LLVM/MLIR toolchain required")
    flags = ["--target=x86_64-unknown-linux-gnu"]
    observed = target_data_layout.observe_index_width(compiler, flags)
    text = SOURCE.replace("arith.constant 1.0:f32", f"arith.constant {second_seed}:f32")
    source = bind_source(parse_mlir_text(text), entry="compute", context=PipelineContext(), graph_mode="ssa")
    hardware = tmp_path / "hardware.yaml"
    hardware.write_text(
        yaml.safe_dump(
            {
                "schema": "merlin.execution_topology.v1",
                "resources": [
                    {"id": kind, "kind": kind, "features": [], "shared_group": kind} for kind in ("cpu", "accelerator")
                ],
                "memory_domains": [{"id": "ram", "capacity_bytes": 65536, "alignment_bytes": 64}],
                "access": [
                    {
                        "resource": resource,
                        "domain": "ram",
                        "read": True,
                        "write": True,
                        "addressability": "shared",
                        "visibility": "coherent",
                        "completion": "synchronous",
                    }
                    for resource in ("cpu", "accelerator")
                ],
            }
        )
    )
    topology = select_topology(hardware)
    platform_dir = tmp_path / "platform"
    platform_dir.mkdir()
    contract = platform_dir / "contract.json"
    contract.write_text('{"scope":"fake MMIO; no device arithmetic or hardware qualification"}')
    product = Artifact.from_file(contract, kind="json", role="platform-contract")
    platform = Compilation(platform_dir, "caller.fake-mmio", product.sha256, (product,))
    platform.write()
    context = CContext(
        "endpoint", product.sha256, "cpu", "ram", 128, 64, 64, 64, "device.h", "software_device", "software_initialize"
    )
    split = len(source.operations) - 1
    workspace, _ = plan(source.operations[:split], (source.operations[split - 1].results[0],))
    assert workspace.bytes == 16
    candidates = []
    for nodes, provider, metadata, resource in (
        (
            tuple(range(split)),
            scalar.name,
            (
                ("native_abi_admission", "required_before_control_or_link"),
                ("source_workspace", json.dumps(workspace.record(), sort_keys=True, separators=(",", ":"))),
                ("tensor_workspace_limit", "64"),
                ("workspace_status", "source_backed_native_admission_pending"),
            ),
            "cpu",
        ),
        ((split,), "fixture.mmio", (context.selection_metadata(),), "accelerator"),
    ):
        incoming, outgoing = proposal_boundary(source, nodes)
        variant = replace(
            source.cpu_alternatives[0],
            id=provider,
            nodes=nodes,
            implementation=provider,
            placement=resource,
            inputs=incoming,
            outputs=outgoing,
            metadata=tuple(sorted(metadata)),
        )
        claim = SourceProposal(
            variant,
            provider,
            resource,
            "ram",
            "logical_tensor_v1",
            source.stamp.source_sha256,
            tuple(source.operation_ordinals[n] for n in nodes),
            source_numerical_identity(source, nodes),
        )

        def admit(bound, proposed):
            assert bound is source
            operations = tuple(bound.operations[n] for n in proposed.alternative.nodes)
            if proposed.provider == scalar.name:
                assert operations[-1].name == "linalg.matmul" and len(operations) == split
            else:
                assert len(operations) == 1 and operations[0].name == "tensor.collapse_shape"
            bound.module.verify()

        candidates.append(admit_proposal(source, claim, topology, validate_implementation=admit))

    def validate(request):
        module = parse_mlir_text(request.source.text)
        assert [op.name for op in module.body.block.first_op.body.block.ops] == ["tensor.collapse_shape", "func.return"]

    def compile_device(request):
        validate(request)
        request.output.mkdir()
        files = {
            "device.h": DEVICE_HEADER,
            "device.c": DEVICE,
            "program.h": render_u32_payload("software_program", bytes.fromhex("0403020108070605")),
        }
        for name, text in files.items():
            (request.output / name).write_text(text)
        (request.output / "program.bin").write_bytes(bytes.fromhex("0403020108070605"))
        call = CRegionABI(
            "software_call",
            (TensorPort((2, 2), "f32"),),
            (TensorPort((4,), "f32"),),
            4,
            ("device.c",),
            payloads=("program.bin",),
            resource_kind="accelerator",
        )
        abi = CRegionABI2(call, context, (EmbeddedPayload("program.bin", "program.h", "software_program"),))
        artifacts = tuple(
            Artifact.from_file(request.output / name, kind=kind, role="fake-mmio")
            for name, kind in (
                ("device.h", "c-header"),
                ("device.c", "c-source"),
                ("program.h", "c-header"),
                ("program.bin", "device-binary"),
            )
        )
        return Compilation(
            request.output, "fixture.mmio", request.source.sha256, artifacts, metadata={"c_call": abi.record()}
        )

    device = SimpleNamespace(name="fixture.mmio", validate=validate, compile=compile_device)
    startup_dir = tmp_path / "startup"
    startup_dir.mkdir()
    main = startup_dir / "main.c"
    expected = "{37,54,-37,-54}" if second_seed == 0.0 else "{38,55,-36,-53}"
    main.write_text(MAIN.replace("{38,55,-36,-53}", expected))
    artifact = Artifact.from_file(main, kind="c-source", role="entry")
    startup = Compilation(startup_dir, "caller.startup", artifact.sha256, (artifact,))
    startup.write()

    def approve(binding, region, image, options):
        assert binding.context == context and binding.platform is platform
        assert image.startup is startup and options == {"scope": "fake MMIO"}
        assert CRegionABI2.from_compilation(region).context == context

    entry = CImageEntry(
        contexts=(
            CImageContext(
                context,
                platform,
                "contract.json",
                "int(context*,void*,size_t)",
                "contents_not_retained",
                "host_state_only",
            ),
        ),
        symbol="image_run",
        validate_platform=approve,
        options={"scope": "fake MMIO"},
        reservations=(ReservedImageInterval("physical", 0x90000000, 4096),),
        entry_state_bytes=128,
        entry_state_pre_reserved=True,
    )
    image = CImageLink(
        compiler.resolve(),
        (*flags, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-no-pie"),
        startup,
        ("main.c",),
        Deployment("software-mixture", "linux", "x86_64", "sysv", 1),
        context_entry=entry,
    )
    scalar_options = CompileOptions(
        parameters={
            "scalar.data-layout": observed["data_layout"],
            "scalar.index-bits": observed["index_bits"],
            "scalar.emit": "object",
            "scalar.target-flags": flags,
            "scalar.call-entry": "kernel",
            "scalar.workspace-bytes": 64,
        }
    )
    backend = PlannedCompiler(
        select_execution(source, tuple(candidates)),
        {scalar.name: RegionBackend(scalar, "kernel", scalar_options), "fixture.mmio": RegionBackend(device, "kernel")},
        image=image,
    )
    result = api.compile(
        ir_text(source.module),
        backend=backend,
        output=tmp_path / "compiled",
        options=CompileOptions(parameters={"merlin.emit": "c-dispatch", "merlin.entry": "model"}),
    )
    report = json.loads((result.directory / "control/control.json").read_text())
    assert report["source_workspace"]["reserved_bytes"] == 64
    assert report["host_context_reservations"]["domains"]["ram"]["total_bytes"] == 192
    assert report["total_tensor_bytes"] + report["total_reserved_host_bytes"] <= topology.memory("ram").capacity_bytes
    assert report["error_lifetime"] == "retain_context_workspace_arena_external_and_device_buffers_until_quiescent"
    assert {binding["abi"]["schema"] for binding in report["bindings"]} == {"merlin.c_call.v2", "merlin.c_call.v3"}
    CWorkspaceABI.from_compilation(Compilation.load(result.directory / "regions/0"))
    ran = api.run(result, runner=linux)
    assert ran.status == "completed" and ran.stdout == "native-cpu-workspace-and-fake-mmio-plan\n"
    assert result.metadata["execution_verified"] is False
