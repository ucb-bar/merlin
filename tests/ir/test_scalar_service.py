import json

import pytest

from merlin import api
from merlin.compiler.backend import scalar
from merlin.compiler.request import CompileOptions, CompileRequest, Source

SOURCE = "module {func.func @identity(%x: i32) -> i32 {func.return %x : i32}}"
LAYOUT = "e-p:64:64-i64:64-n32:64-S128"


def request(tmp_path, **parameters):
    return CompileRequest(
        Source(SOURCE, kind="upstream-mlir"),
        tmp_path / "out",
        CompileOptions(parameters={"scalar.data-layout": LAYOUT, "scalar.index-bits": 64, **parameters}),
    )


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
    if (
        not producer.is_file()
        or not checkout.is_dir()
        or (not Path(compiler).is_file())
        or (platform.machine() != "x86_64")
        or (toolchain._env("MERLIN_COMPILER_DRIVER") != "stock-cli")
    ):
        pytest.skip("select pinned Torch capture and actual native MLIR/LLVM on Linux x86_64")
    assert capture_pin(checkout)["ok"]
    script = tmp_path / "export.py"
    script.write_text(
        'import json,sys,torch\nfrom pathlib import Path\nclass Weighted(torch.nn.Module):\n    def __init__(self, workspace=False):\n        super().__init__()\n        self.weight = torch.nn.Parameter(torch.tensor([[1.,2.],[3.,4.],[5.,6.]]))\n        self.bias = torch.nn.Parameter(torch.tensor([1.,-2.]))\n        if workspace:\n            self.weight = torch.nn.Parameter(torch.arange(12,dtype=torch.float32).reshape(3,4))\n            self.second = torch.nn.Parameter(torch.arange(8,dtype=torch.float32).reshape(4,2))\n    def forward(self,x):\n        value = x @ self.weight\n        if hasattr(self,"second"): value = value @ self.second\n        return torch.relu(value + self.bias)\nx=torch.tensor([[1.,2.,3.],[-1.,-2.,-3.]])\nroot=Path(sys.argv[1])\nfor name,workspace in (("model",False),("workspace",True)):\n    model=Weighted(workspace).eval()\n    exported=torch.export.export(model,(x,),strict=False)\n    torch.export.save(exported,root/(name+".pt2"))\n    if not workspace:\n        reference=exported.module()(x).detach().contiguous().view(torch.int32).flatten().tolist()\n        (root/"reference.json").write_text(json.dumps(reference))\n    else:\n        reference=exported.module()(x).detach().contiguous().view(torch.int32).flatten().tolist()\n        (root/"workspace-reference.json").write_text(json.dumps(reference))\n'
    )
    subprocess.run([str(producer), str(script), str(tmp_path)], check=True, capture_output=True, text=True)
    hardware = tmp_path / "hardware.yaml"
    hardware.write_text(
        "schema: merlin.execution_topology.v1\nresources:\n- {id: cpu0, kind: cpu, features: [], shared_group: cpu0}\nmemory_domains:\n- {id: ram, capacity_bytes: 65536, alignment_bytes: 64}\naccess:\n- {resource: cpu0, domain: ram, read: true, write: true, addressability: shared,\n   visibility: coherent, completion: synchronous}\n"
    )
    topology = select_topology(hardware)
    flags = ["--target=x86_64-unknown-linux-gnu"]
    observed = observe_index_width(compiler, flags)
    component = scalar_cpu(
        resource="cpu0",
        memory_domain="ram",
        options=api.CompileOptions(
            parameters={
                "scalar.data-layout": observed["data_layout"],
                "scalar.index-bits": observed["index_bits"],
                "scalar.target-flags": flags,
            }
        ),
    )
    startup = tmp_path / "startup"
    startup.mkdir()
    main = startup / "main.c"
    main.write_text(
        '#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n#include "model.h"\nint main(void) {\n  _Alignas(64) unsigned char arena[MODEL_ARENA_BYTES ? MODEL_ARENA_BYTES : 1];\n  float input[6]={1,2,3,-1,-2,-3}, before[6]={1,2,3,-1,-2,-3}, output[4]={0};\n  model_buffer buffers[2]={{input,sizeof input},{output,sizeof output}};\n  model_state state={0};\n  if(MODEL_ARENA_BYTES) {\n    if(model_run(&state,arena,MODEL_ARENA_BYTES-1,buffers,2,0)!=MODEL_INVALID || state.active) return 3;\n    buffers[1].data=arena;\n    if(model_run(&state,arena,sizeof arena,buffers,2,0)!=MODEL_INVALID || state.active) return 4;\n    buffers[1].data=output;\n  }\n  if(model_run(&state,arena,sizeof arena,buffers,2,0)||state.completed!=1||state.active) return 1;\n  if(memcmp(input,before,sizeof input)) return 2;\n  for(int i=0;i<4;++i) {uint32_t bits;memcpy(&bits,&output[i],sizeof bits);printf("%u\\n",bits);}\n  return 0;\n}\n'
    )
    artifact = Artifact.from_file(main, kind="c-source", role="entry")
    entry = Compilation(startup, "caller.startup", artifact.sha256, (artifact,))
    entry.write()
    image = CImageLink(
        Path(compiler).resolve(),
        tuple([*flags, "-std=c11", "-O2"]),
        entry,
        ("main.c",),
        Deployment("fixture-cpu", "linux", "x86_64", "sysv", 1),
    )
    backend = SystemCompiler(
        topology=topology, components=(component,), preferred_providers=("merlin.scalar",), image=image
    )
    options = api.CompileOptions(parameters={"merlin.emit": "c-dispatch", "merlin.entry": "model"})
    result = api.compile(tmp_path / "model.pt2", backend=backend, output=tmp_path / "compiled", options=options)
    system = json.loads((result.directory / "system.json").read_text())
    frontend = result.metadata["frontend"]["argument_binding"]
    assert len(frontend["initializers"]) == 2 and len(frontend["runtime_inputs"]) == 1
    assert system["request_source_sha256"] == result.source_sha256 == frontend["bound_program_sha256"]
    assert system["selected_obligations"][0]["requirements"]["workspace_status"] == "conditional_not_observed"
    scalar_record = json.loads((result.directory / "regions/0/scalar.json").read_text())
    normalization = scalar_record["call"]["initialization"]
    assert any(("fill_ordinal" in item for item in normalization["replacements"]))
    assert any(
        (
            "empty_metadata_ordinal" in item and "prov.origin_node_ids" in item["attributes"]
            for item in normalization["replacements"]
        )
    )
    assert scalar_record["call"]["memory"]["tensor_scratch_bytes"] == 0
    assert len(system["selection"]["execution_order"]) == 1
    assert result.metadata["stage"] == "linked_image" and (not result.metadata["execution_verified"])
    executed = api.run(result, runner=linux)
    assert executed.status == "completed"
    assert list(map(int, executed.stdout.splitlines())) == json.loads((tmp_path / "reference.json").read_text())
    assert not any((p.name in {"golden.npy", "inputs.npz"} for p in result.directory.rglob("*")))
    with pytest.raises(ValueError, match="workspace|external|interface|signature"):
        api.compile(tmp_path / "workspace.pt2", backend=backend, output=tmp_path / "refused", options=options)
    failure = json.loads((tmp_path / "refused/system-failure.json").read_text())
    assert failure["status"] == "failed" and failure["fallback"].startswith("not_attempted")
    workspace_component = scalar_cpu(
        resource="cpu0",
        memory_domain="ram",
        options=api.CompileOptions(parameters={**component.binding.options.parameters, "scalar.workspace-bytes": 256}),
    )
    workspace_backend = SystemCompiler(
        topology=topology, components=(workspace_component,), preferred_providers=("merlin.scalar",), image=image
    )
    with_workspace = api.compile(
        tmp_path / "workspace.pt2", backend=workspace_backend, output=tmp_path / "with-workspace", options=options
    )
    workspace_run = api.run(with_workspace, runner=linux)
    assert workspace_run.status == "completed"
    assert list(map(int, workspace_run.stdout.splitlines())) == json.loads(
        (tmp_path / "workspace-reference.json").read_bytes()
    )
    workspace_record = json.loads((with_workspace.directory / "regions/0/scalar.json").read_bytes())
    assert workspace_record["call"]["memory"]["source_workspace_bytes"] == 32
    assert workspace_record["call"]["memory"]["additional_tensor_scratch_bytes"] == 0
    control = json.loads((with_workspace.directory / "control/control.json").read_bytes())
    assert control["source_workspace"]["reserved_bytes"] == 64
    assert control["source_workspace"]["original_buffers"]["0"]
    assert control["source_workspace"]["reserved_bytes"] < 256
    assert control["total_tensor_bytes"] <= topology.memory("ram").capacity_bytes
    assert with_workspace.metadata["execution_verified"] is False
    assert not (tmp_path / "refused/compilation.json").exists()
