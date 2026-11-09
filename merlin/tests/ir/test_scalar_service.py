"""Scalar service admission and legacy object-command preservation.

Diagnostic compiler products below exercise request plumbing, not native or
numerical qualification. Real tool execution remains a separate installed gate.
"""

import json
from types import SimpleNamespace

import pytest

from merlin import api
from merlin.compiler.backend import scalar
from merlin.compiler.request import CompileOptions, CompileRequest, Source

SOURCE = "module {func.func @identity(%x: i32) -> i32 {func.return %x : i32}}"
LAYOUT = "e-p:64:64-i64:64-n32:64-S128"


def request(tmp_path, **parameters):
    return CompileRequest(Source(SOURCE, kind="upstream-mlir"), tmp_path / "out", CompileOptions(parameters={
        "scalar.data-layout": LAYOUT, "scalar.index-bits": 64, **parameters,
    }))


@pytest.mark.parametrize("parameters", [
    {"scalar.index-bits": True}, {"scalar.data-layout": ""}, {"scalar.extra": True},
    {"scalar.emit": "image"}, {"scalar.emit": []}, {"scalar.emit": "object"},
    {"scalar.emit": "object", "scalar.target-flags": ["-march=native"]},
    {"scalar.emit": "object", "scalar.target-flags": ["--target=unknown", "-o=injected"]},
    {"scalar.target-flags": ["--target=unknown"]},
    {"scalar.workspace-bytes": True}, {"scalar.workspace-bytes": -1}, {"scalar.workspace-bytes": 32},
])
def test_invalid_selection_refuses_before_tools_and_outputs(tmp_path, monkeypatch, parameters):
    monkeypatch.setattr(scalar, "_tools", lambda *args: pytest.fail("native tools reached"))
    with pytest.raises(ValueError):
        scalar.validate(request(tmp_path, **parameters))
    assert not (tmp_path / "out").exists()


def test_deployment_and_vector_inputs_refuse_before_tools(tmp_path, monkeypatch):
    monkeypatch.setattr(scalar, "_tools", lambda *args: pytest.fail("native tools reached"))
    value = request(tmp_path)
    with pytest.raises(ValueError, match="deployment"):
        scalar.validate(CompileRequest(value.source, value.output, CompileOptions(platform="selected")))
    source = Source("module {func.func @f(%x: vector<4xi32>) -> vector<4xi32> {func.return %x : vector<4xi32>}}")
    with pytest.raises(ValueError, match="vector"):
        scalar.validate(CompileRequest(source, value.output, value.options))


def test_layout_mismatch_cannot_reach_native_lowering(tmp_path, monkeypatch):
    from merlin.llvmlower import pipeline, target_data_layout

    monkeypatch.setattr(scalar, "_tools", lambda *args: {"clang": {"path": "diagnostic", "sha256": "0" * 64}})
    monkeypatch.setattr(target_data_layout, "observe_index_width", lambda *args: {
        "data_layout": "e-p:32:32", "index_bits": 32,
    })
    monkeypatch.setattr(pipeline, "lower_to_llvm_ir", lambda *a, **k: pytest.fail("lowering reached"))
    value = request(tmp_path, **{"scalar.emit": "object", "scalar.target-flags": ["--target=unknown"]})
    with pytest.raises(ValueError, match="actual object compiler"):
        scalar.compile(value)
    assert not value.output.exists()


def test_prepared_and_model_sources_use_existing_owners_and_publish_no_image(tmp_path, monkeypatch):
    from merlin.llvmlower import lower, pipeline

    observed = []
    monkeypatch.setattr(scalar, "_tools", lambda *args: {"compiler-python": {"path": "diagnostic", "sha256": "0" * 64}})

    def prepared(text, **options):
        observed.append((text, options))
        assert options["vectorize"] is False and options["parallel"] is False
        assert options["features"] == frozenset()
        options["workdir"].mkdir()
        (options["workdir"] / "diagnostic.json").write_text('{"native_execution": false}')
        return "; diagnostic nonexecuted LLVM product\n"

    def model(text, work, **options):
        assert options["targets"] == ()
        prepared(text, workdir=work, **{key: value for key, value in options.items() if key != "targets"})
        path = work / "model.ll"
        path.write_text("; diagnostic model product\n")
        return SimpleNamespace(ll_path=path, stats={"native_execution": False})

    monkeypatch.setattr(pipeline, "lower_to_llvm_ir", prepared)
    monkeypatch.setattr(lower, "lower_model", model)
    for kind in ("mlir", "upstream-mlir"):
        result = api.compile(Source(SOURCE, kind=kind), backend=scalar, output=tmp_path / kind,
                             options=request(tmp_path).options)
        reopened = api.Compilation.load(result.manifest)
        assert reopened.deployment is None
        record = json.loads((reopened.directory / "scalar.json").read_text())
        assert record["execution_observed"] is False and record["source_kind"] == kind
        assert not any(product.kind in {"elf", "object"} for product in reopened.artifacts)
    assert [item[0] for item in observed] == [SOURCE, SOURCE]


def test_legacy_object_command_preserves_flags_runner_and_timeout(tmp_path, monkeypatch):
    from merlin.llvmlower import codegen

    calls = []
    monkeypatch.setattr(codegen, "clang", lambda: "selected-clang")
    monkeypatch.setattr(codegen, "_run", lambda command, **options: calls.append((command, options)))
    source, output = tmp_path / "model.ll", tmp_path / "model.o"
    returned = codegen.compile_ll(source, output, "host", extra_flags=("-fno-vectorize",), timeout_s=17)
    command, options = calls.pop()
    assert command == ["selected-clang", *codegen.X86_FLAGS, "-fno-vectorize", "-c", source, "-o", output]
    assert options == {"timeout_s": 17, "inputs": (source,), "outputs": (output,)}
    assert returned == output


def test_native_failure_leaves_diagnostic_without_success_manifest(tmp_path, monkeypatch):
    from merlin.llvmlower import pipeline

    monkeypatch.setattr(scalar, "_tools", lambda *args: {})
    failure = RuntimeError("diagnostic native failure")

    def failed(*args, **kwargs):
        raise failure

    monkeypatch.setattr(pipeline, "lower_to_llvm_ir", failed)
    value = request(tmp_path)
    with pytest.raises(RuntimeError, match="diagnostic native failure") as observed:
        api.compile(value.source, backend=scalar, output=value.output, options=value.options)
    assert observed.value is failure
    assert not (value.output / "compilation.json").exists()
    assert json.loads((value.output / "failure.json").read_text())["type"] == "RuntimeError"


WEIGHTED = """module {func.func @compute(%x:tensor<2x3xf32>) -> tensor<2x2xf32> {
 %w=arith.constant dense<[[1.0,2.0],[3.0,4.0],[5.0,6.0]]> : tensor<3x2xf32>
 %b=arith.constant dense<[1.0,-2.0]> : tensor<2xf32>
 %z=arith.constant 0.0:f32
 %e=tensor.empty():tensor<2x2xf32>
 %f=linalg.fill ins(%z:f32) outs(%e:tensor<2x2xf32>) -> tensor<2x2xf32>
 %m=linalg.matmul ins(%x,%w:tensor<2x3xf32>,tensor<3x2xf32>) outs(%f:tensor<2x2xf32>) -> tensor<2x2xf32>
 %o=tensor.empty():tensor<2x2xf32>
 %r=linalg.generic {indexing_maps=[affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d1)>,
 affine_map<(d0,d1)->(d0,d1)>],iterator_types=["parallel","parallel"]}
 ins(%m,%b:tensor<2x2xf32>,tensor<2xf32>) outs(%o:tensor<2x2xf32>) {
 ^bb0(%v:f32,%bias:f32,%old:f32):
 %s=arith.addf %v,%bias:f32
 %a=arith.maximumf %s,%z:f32
 linalg.yield %a:f32
 } -> tensor<2x2xf32>
 func.return %r:tensor<2x2xf32>
}}"""


def test_native_weighted_scalar_region_joins_selected_image_and_preserves_input(tmp_path, monkeypatch):
    from dataclasses import replace
    from pathlib import Path

    import yaml

    from merlin.compiler.artifact import Artifact, Compilation, Deployment
    from merlin.compiler.backend.cabi import CRegionABI
    from merlin.compiler.backend.sourceproposal import (
        SourceProposal,
        admit_proposal,
        proposal_boundary,
        source_numerical_identity,
    )
    from merlin.compiler.build.image import CImageLink
    from merlin.compiler.pipeline import PlannedCompiler, RegionBackend
    from merlin.compiler.planning.execution import select_execution
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower import target_data_layout, toolchain
    from merlin.llvmlower.offload_plan import bind_source
    from merlin.runtime import linux
    from merlin.system.topology import select_topology
    from merlin.transforms.certificates import ir_text
    from merlin.transforms.pipeline_state import PipelineContext

    compiler = toolchain.clang()
    if toolchain._env("MERLIN_COMPILER_DRIVER", "python") != "stock-cli" or not Path(compiler).is_file():
        pytest.skip("select actual stock MLIR/LLVM tools for numerical scalar image execution")
    flags = ["--target=x86_64-unknown-linux-gnu"]
    observed = target_data_layout.observe_index_width(compiler, flags)
    options = CompileOptions(parameters={"scalar.data-layout": observed["data_layout"], "scalar.index-bits": 64,
                                         "scalar.emit": "object", "scalar.target-flags": flags,
                                         "scalar.call-entry": "kernel"})
    source = bind_source(parse_mlir_text(WEIGHTED), entry="compute", context=PipelineContext(), graph_mode="ssa")
    hardware = tmp_path / "hardware.yaml"
    hardware.write_text(yaml.safe_dump({
        "schema": "merlin.execution_topology.v1",
        "resources": [{"id": "cpu0", "kind": "cpu", "features": [], "shared_group": "cpu0"}],
        "memory_domains": [{"id": "ram", "capacity_bytes": 65536, "alignment_bytes": 64}],
        "access": [{"resource": "cpu0", "domain": "ram", "read": True, "write": True,
                    "addressability": "shared", "visibility": "coherent", "completion": "synchronous"}],
    }))
    topology = select_topology(hardware)
    nodes = tuple(range(len(source.operations)))
    incoming, outgoing = proposal_boundary(source, nodes)
    variant = replace(source.cpu_alternatives[0], id="whole", nodes=nodes, implementation="scalar-native",
                      placement="cpu0", inputs=incoming, outputs=outgoing)
    claim = SourceProposal(variant, scalar.name, "cpu0", "ram", "logical_tensor_v1", source.stamp.source_sha256,
                           tuple(source.operation_ordinals[n] for n in nodes), source_numerical_identity(source, nodes))
    def original_validator(bound, proposed):
        assert proposed.alternative.nodes == tuple(range(len(bound.operations)))
        bound.module.verify()
    proposal = admit_proposal(source, claim, topology, validate_implementation=original_validator)
    startup = tmp_path / "startup"
    startup.mkdir()
    main = startup / "main.c"
    main.write_text('''#include <stdio.h>
#include "model.h"
int main(void) {
  _Alignas(64) unsigned char arena[64];
  float x[6] = {1,2,3,-1,-2,-3}, before[6] = {1,2,3,-1,-2,-3};
  float y[4] = {-100,-100,-100,-100}, expected[4] = {23,26,0,0};
  model_buffer buffers[2] = {{x,sizeof x},{y,sizeof y}};
  model_state state = {0};
  if (model_run(&state, arena, sizeof arena, buffers, 2, 0)) return 1;
  if (state.completed != 1 || state.active) return 2;
  for (int i=0;i<6;++i) if (x[i]!=before[i]) return 3;
  for (int i=0;i<4;++i) if (y[i]!=expected[i]) return 4;
  puts("weighted-scalar-plan-executed");
  return 0;
}
''')
    artifact = Artifact.from_file(main, kind="c-source", role="entry")
    entry = Compilation(startup, "caller.startup", artifact.sha256, (artifact,))
    entry.write()
    image = CImageLink(Path(compiler).resolve(), tuple([*flags, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror"]),
                       entry, ("main.c",), Deployment("fixture-cpu", "linux", "x86_64", "sysv", 1))
    backend = PlannedCompiler(select_execution(source, (proposal,)),
                              {scalar.name: RegionBackend(scalar, "kernel", options)}, image=image)
    compiled = api.compile(ir_text(source.module), backend=backend, output=tmp_path / "compiled",
                           options=CompileOptions(parameters={"merlin.emit": "c-dispatch", "merlin.entry": "model"}))
    region = Compilation.load(compiled.directory / "regions/0")
    abi = CRegionABI.from_compilation(region)
    assert len(abi.inputs) == len(abi.outputs) == 1 and abi.scratch_bytes == 0
    record = json.loads((region.directory / "scalar.json").read_text())
    assert record["call"]["memory"]["writes"] == "selected_outputs_only"
    assert record["call"]["memory"]["inputs"] == "read_only"
    assert record["execution_observed"] is False and compiled.metadata["execution_verified"] is False
    result = api.run(compiled, runner=linux)
    assert result.status == "completed" and result.metadata["exit_code"] == 0
    assert len(result.metadata["initial_cpu_affinity"]) == image.deployment.harts
    assert result.stdout == "weighted-scalar-plan-executed\n"
    api.inspect(compiled.manifest)
    from io import StringIO

    from xdsl.dialects import builtin, llvm
    from xdsl.printer import Printer

    from merlin.compiler.lower.llvm import call as call_owner

    generic = (region.directory / "call.native.mlir").read_text()
    selected = call_owner.ScalarCall("", "kernel", record["call"]["compute_entry"], abi.symbol, abi.inputs, abi.outputs)
    with pytest.raises(ValueError, match="descriptor"):
        call_owner.admit(generic, selected, index_bits=32, data_layout=observed["data_layout"])
    for fault in ("input_write", "workspace"):
        module = call_owner._parse_native(generic)
        compute = next(op for op in module.body.block.ops
                       if type(op) is llvm.FuncOp and op.sym_name.data == selected.compute)
        if fault == "input_write":
            store = next(op for op in compute.walk() if type(op) is llvm.StoreOp)
            store.operands = (store.operands[0], compute.body.blocks.first.args[1])
        else:
            count = llvm.ConstantOp(builtin.IntegerAttr(1, builtin.i64), builtin.i64)
            allocation = llvm.AllocaOp(count, builtin.i8)
            compute.body.blocks.first.insert_ops_before([count, allocation], compute.body.blocks.first.first_op)
        printed = StringIO()
        Printer(stream=printed, print_generic_format=True).print_op(module)
        with pytest.raises(ValueError, match="mutate inputs|workspace or external"):
            call_owner.admit(printed.getvalue(), selected, index_bits=64, data_layout=observed["data_layout"])
    from merlin.llvmlower import codegen

    original_run = codegen._run
    def mutate_after_translation(command, **kwargs):
        result = original_run(command, **kwargs)
        if kwargs.get("stage") == "scalar_call_translation":
            selected_source = kwargs["inputs"][0]
            selected_source.write_text(selected_source.read_text() + "\n")
        return result
    with monkeypatch.context() as mutation:
        mutation.setattr(codegen, "_run", mutate_after_translation)
        direct = replace(options, parameters={**options.parameters, "scalar.call-entry": "compute"})
        with pytest.raises(ValueError, match="interface/source changed"):
            api.compile(WEIGHTED, backend=scalar, output=tmp_path / "mutated", options=direct)
    assert not (tmp_path / "mutated/compilation.json").exists()


def test_scalar_call_selection_cannot_turn_prepared_scalars_or_effects_into_tensor_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(scalar, "_tools", lambda *args: pytest.fail("native tools reached"))
    value = request(tmp_path, **{"scalar.call-entry": "identity"})
    with pytest.raises(ValueError, match="object emission"):
        scalar.validate(value)
    value = request(tmp_path, **{"scalar.emit": "object", "scalar.target-flags": ["--target=unknown"],
                                 "scalar.call-entry": "identity"})
    with pytest.raises(ValueError, match="original tensor MLIR"):
        scalar.validate(value)
    from merlin.compiler.lower.llvm.call import prepare

    with pytest.raises(ValueError, match="semantic or ABI attributes"):
        prepare(WEIGHTED.replace(
            " -> tensor<2x2xf32> {", " -> tensor<2x2xf32> attributes {llvm.readnone} {", 1),
            "compute", "wrapper")


def test_source_workspace_is_bounded_private_storage_without_new_arithmetic(tmp_path, monkeypatch):
    from merlin.compiler.lower.llvm.call import prepare
    from merlin.frontends.linalg_mlir import parse_mlir_text

    source = WEIGHTED.replace(" %o=tensor.empty()", ''' %w2=arith.constant dense<[[1.0,2.0],[3.0,4.0]]>:tensor<2x2xf32>
 %e2=tensor.empty():tensor<2x2xf32>
 %f2=linalg.fill ins(%z:f32) outs(%e2:tensor<2x2xf32>) -> tensor<2x2xf32>
 %m2=linalg.matmul ins(%m,%w2:tensor<2x2xf32>,tensor<2x2xf32>)
 outs(%f2:tensor<2x2xf32>) -> tensor<2x2xf32>
 %o=tensor.empty()''').replace("ins(%m,%b:", "ins(%m2,%b:")
    original = parse_mlir_text(source)
    selected = prepare(source, "compute", "private_source", workspace_bytes=16)
    assert selected.workspace.bytes == 16 and len(selected.outputs) == 1
    private = parse_mlir_text(selected.source)
    function = private.body.block.first_op
    assert len(function.function_type.outputs) == 2
    assert [op.name for op in original.walk()] == [op.name for op in private.walk()]
    # Remove only private returns: every original operation and public outcome
    # must remain exactly equivalent, with no introduced copy/recomputation.
    function.get_return_op().operands = function.get_return_op().operands[:1]
    function.update_function_type()
    function.properties["sym_name"] = original.body.block.first_op.properties["sym_name"]
    assert original.is_structurally_equivalent(private)
    monkeypatch.setattr(scalar, "_tools", lambda *args: pytest.fail("native tools reached"))
    monkeypatch.setenv("MERLIN_COMPILER_DRIVER", "stock-cli")
    options = {"scalar.data-layout": LAYOUT, "scalar.index-bits": 64, "scalar.emit": "object",
               "scalar.target-flags": ["--target=x86_64-unknown-linux-gnu"], "scalar.call-entry": "compute",
               "scalar.workspace-bytes": 15}
    with pytest.raises(ValueError, match="workspace.*ceiling"):
        scalar.validate(CompileRequest(Source(source), tmp_path / "too-small", CompileOptions(parameters=options)))
    alias = source.replace(" %m2=", ''' %view=tensor.cast %m : tensor<2x2xf32> to tensor<2x2xf32>
 %m2=''').replace("ins(%m,%w2", "ins(%view,%w2")
    # The source still has a real materialization barrier via the second use;
    # unknown/view aliases do not acquire workspace permission by shape alone.
    with pytest.raises(ValueError, match="view/alias"):
        prepare(alias, "compute", "refused_alias", workspace_bytes=64)
    dynamic = source.replace("tensor<2x3xf32>", "tensor<?x3xf32>")
    with pytest.raises(ValueError, match="static positive"):
        prepare(dynamic, "compute", "refused_dynamic", workspace_bytes=64)


def test_public_export_system_compilation_plans_bound_weights_runs_and_refuses_unknown_workspace(tmp_path):
    import platform
    import subprocess
    from pathlib import Path

    from merlin.compiler.artifact import Artifact, Compilation, Deployment
    from merlin.compiler.build.image import CImageLink
    from merlin.compiler.system import SystemCompiler, scalar_cpu
    from merlin.integrations.model2mlir import capture_pin, capture_python_path, root
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.target_data_layout import observe_index_width
    from merlin.runtime import linux
    from merlin.system.topology import select_topology

    checkout = root()
    producer = capture_python_path(checkout=checkout)
    compiler = toolchain.clang()
    if (not producer.is_file() or not checkout.is_dir() or not Path(compiler).is_file() or
            platform.machine() != "x86_64" or toolchain._env("MERLIN_COMPILER_DRIVER") != "stock-cli"):
        pytest.skip("select pinned Torch capture and actual native MLIR/LLVM on Linux x86_64")
    assert capture_pin(checkout)["ok"]
    script = tmp_path / "export.py"
    script.write_text('''import json,sys,torch
from pathlib import Path
class Weighted(torch.nn.Module):
    def __init__(self, workspace=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[1.,2.],[3.,4.],[5.,6.]]))
        self.bias = torch.nn.Parameter(torch.tensor([1.,-2.]))
        if workspace:
            self.weight = torch.nn.Parameter(torch.arange(12,dtype=torch.float32).reshape(3,4))
            self.second = torch.nn.Parameter(torch.arange(8,dtype=torch.float32).reshape(4,2))
    def forward(self,x):
        value = x @ self.weight
        if hasattr(self,"second"): value = value @ self.second
        return torch.relu(value + self.bias)
x=torch.tensor([[1.,2.,3.],[-1.,-2.,-3.]])
root=Path(sys.argv[1])
for name,workspace in (("model",False),("workspace",True)):
    model=Weighted(workspace).eval()
    exported=torch.export.export(model,(x,),strict=False)
    torch.export.save(exported,root/(name+".pt2"))
    if not workspace:
        reference=exported.module()(x).detach().contiguous().view(torch.int32).flatten().tolist()
        (root/"reference.json").write_text(json.dumps(reference))
    else:
        reference=exported.module()(x).detach().contiguous().view(torch.int32).flatten().tolist()
        (root/"workspace-reference.json").write_text(json.dumps(reference))
''')
    subprocess.run([str(producer), str(script), str(tmp_path)], check=True, capture_output=True, text=True)
    hardware = tmp_path / "hardware.yaml"
    hardware.write_text('''schema: merlin.execution_topology.v1
resources:
- {id: cpu0, kind: cpu, features: [], shared_group: cpu0}
memory_domains:
- {id: ram, capacity_bytes: 65536, alignment_bytes: 64}
access:
- {resource: cpu0, domain: ram, read: true, write: true, addressability: shared,
   visibility: coherent, completion: synchronous}
''')
    topology = select_topology(hardware)
    flags = ["--target=x86_64-unknown-linux-gnu"]
    observed = observe_index_width(compiler, flags)
    component = scalar_cpu(resource="cpu0", memory_domain="ram", options=api.CompileOptions(parameters={
        "scalar.data-layout": observed["data_layout"], "scalar.index-bits": observed["index_bits"],
        "scalar.target-flags": flags,
    }))
    startup = tmp_path / "startup"
    startup.mkdir()
    main = startup / "main.c"
    main.write_text('''#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "model.h"
int main(void) {
  _Alignas(64) unsigned char arena[MODEL_ARENA_BYTES ? MODEL_ARENA_BYTES : 1];
  float input[6]={1,2,3,-1,-2,-3}, before[6]={1,2,3,-1,-2,-3}, output[4]={0};
  model_buffer buffers[2]={{input,sizeof input},{output,sizeof output}};
  model_state state={0};
  if(MODEL_ARENA_BYTES) {
    if(model_run(&state,arena,MODEL_ARENA_BYTES-1,buffers,2,0)!=MODEL_INVALID || state.active) return 3;
    buffers[1].data=arena;
    if(model_run(&state,arena,sizeof arena,buffers,2,0)!=MODEL_INVALID || state.active) return 4;
    buffers[1].data=output;
  }
  if(model_run(&state,arena,sizeof arena,buffers,2,0)||state.completed!=1||state.active) return 1;
  if(memcmp(input,before,sizeof input)) return 2;
  for(int i=0;i<4;++i) {uint32_t bits;memcpy(&bits,&output[i],sizeof bits);printf("%u\\n",bits);}
  return 0;
}
''')
    artifact = Artifact.from_file(main, kind="c-source", role="entry")
    entry = Compilation(startup, "caller.startup", artifact.sha256, (artifact,))
    entry.write()
    image = CImageLink(Path(compiler).resolve(), tuple([*flags, "-std=c11", "-O2"]), entry, ("main.c",),
                       Deployment("fixture-cpu", "linux", "x86_64", "sysv", 1))
    backend = SystemCompiler(topology=topology, components=(component,),
                             preferred_providers=("merlin.scalar",), image=image)
    options = api.CompileOptions(parameters={"merlin.emit": "c-dispatch", "merlin.entry": "model"})
    result = api.compile(tmp_path / "model.pt2", backend=backend, output=tmp_path / "compiled", options=options)
    system = json.loads((result.directory / "system.json").read_text())
    frontend = result.metadata["frontend"]["argument_binding"]
    assert len(frontend["initializers"]) == 2 and len(frontend["runtime_inputs"]) == 1
    assert system["request_source_sha256"] == result.source_sha256 == frontend["bound_program_sha256"]
    assert system["selected_obligations"][0]["requirements"]["workspace_status"] == "conditional_not_observed"
    scalar_record = json.loads((result.directory / "regions/0/scalar.json").read_text())
    normalization = scalar_record["call"]["initialization"]
    assert any("fill_ordinal" in item for item in normalization["replacements"])
    assert any("empty_metadata_ordinal" in item and "prov.origin_node_ids" in item["attributes"]
               for item in normalization["replacements"])
    assert scalar_record["call"]["memory"]["tensor_scratch_bytes"] == 0
    assert len(system["selection"]["execution_order"]) == 1
    assert result.metadata["stage"] == "linked_image" and not result.metadata["execution_verified"]
    executed = api.run(result, runner=linux)
    assert executed.status == "completed"
    assert list(map(int, executed.stdout.splitlines())) == json.loads((tmp_path / "reference.json").read_text())
    assert not any(p.name in {"golden.npy", "inputs.npz"} for p in result.directory.rglob("*"))
    with pytest.raises(ValueError, match="workspace|external|interface|signature"):
        api.compile(tmp_path / "workspace.pt2", backend=backend, output=tmp_path / "refused", options=options)
    failure = json.loads((tmp_path / "refused/system-failure.json").read_text())
    assert failure["status"] == "failed" and failure["fallback"].startswith("not_attempted")
    actual = (tmp_path / "refused/regions/0/call.native.mlir").read_text()
    workspace_component = scalar_cpu(resource="cpu0", memory_domain="ram", options=api.CompileOptions(parameters={
        **component.binding.options.parameters, "scalar.workspace-bytes": 256,
    }))
    workspace_backend = SystemCompiler(topology=topology, components=(workspace_component,),
                                       preferred_providers=("merlin.scalar",), image=image)
    with_workspace = api.compile(tmp_path / "workspace.pt2", backend=workspace_backend,
                                 output=tmp_path / "with-workspace", options=options)
    workspace_run = api.run(with_workspace, runner=linux)
    assert workspace_run.status == "completed"
    assert list(map(int, workspace_run.stdout.splitlines())) == json.loads(
        (tmp_path / "workspace-reference.json").read_bytes())
    workspace_record = json.loads((with_workspace.directory / "regions/0/scalar.json").read_bytes())
    assert workspace_record["call"]["memory"]["source_workspace_bytes"] == 32
    assert workspace_record["call"]["memory"]["additional_tensor_scratch_bytes"] == 0
    control = json.loads((with_workspace.directory / "control/control.json").read_bytes())
    assert control["source_workspace"]["reserved_bytes"] == 64  # Source32 rounded by selected memory alignment.
    assert control["source_workspace"]["original_buffers"]["0"]
    assert control["source_workspace"]["reserved_bytes"] < 256  # Ceiling is not an automatic allocation.
    assert control["total_tensor_bytes"] <= topology.memory("ram").capacity_bytes
    assert with_workspace.metadata["execution_verified"] is False
    assert '"llvm.alloca"' in actual or '"llvm.call"' in actual
    assert not (tmp_path / "refused/compilation.json").exists()

