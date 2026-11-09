"""Native saved-export capture and compiler-only staging with selected m2m."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.compiler.frontend.inputs import stage_compile_inputs
from merlin.integrations.model2mlir import capture_pin, capture_python_path, root


def test_native_export_worker_preserves_state_and_refuses_incomplete_output_reference(tmp_path):
    checkout = root()
    python = capture_python_path(checkout=checkout)
    if not python.is_file() or not checkout.is_dir():
        pytest.skip("configure the actual pinned model2MLIR capture environment")
    assert capture_pin(checkout)["ok"]
    script = tmp_path / "exports.py"
    script.write_text('''import sys
from pathlib import Path
import torch
class Layout(torch.nn.Module):
    def forward(self, x): return x.reshape(6)
class Weighted(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4))
        self.register_buffer("bias", torch.arange(4, dtype=torch.float32))
    def forward(self, x): return x @ self.weight + self.bias
class Multi(torch.nn.Module):
    def forward(self, x): return x.reshape(6), x + 1
for name, model in [("layout", Layout()), ("weighted", Weighted()), ("multi", Multi())]:
    exported = torch.export.export(model.eval(), (torch.arange(6, dtype=torch.float32).reshape(2, 3),))
    torch.export.save(exported, Path(sys.argv[1]) / (name + ".pt2"))
''', encoding="utf-8")
    subprocess.run([str(python), str(script), str(tmp_path)], check=True, capture_output=True, text=True)
    worker = repo_root() / "src/merlin/targetgen/_pt2_capture_worker.py"
    for name in ("layout", "weighted", "multi"):
        capture = tmp_path / name
        completed = subprocess.run([
            str(python), str(worker), "--exported", str(tmp_path / (name + ".pt2")),
            "--out", str(capture), "--m2m-dir", str(checkout),
        ], capture_output=True, text=True)
        if name == "multi":
            assert completed.returncode != 0
            assert "one tensor output" in completed.stderr
            assert not (capture / "capture_receipt.json").exists()
            assert not (capture / "ingest_meta.json").exists()
            continue
        assert completed.returncode == 0, completed.stderr
        assert "__PT2_INGEST__" in completed.stdout
        staged = tmp_path / (name + "-staged")
        manifest = stage_compile_inputs(capture, staged)
        assert "golden.npy" not in manifest["members"]
        assert "inputs.npz" not in manifest["members"]
        trace = json.loads((staged / "frontend-trace.json").read_text())
        assert trace["status"] == "complete"
        assert not trace["blockers"]
        weights = json.loads((staged / "weights.safetensors.manifest.json").read_text())
        assert [weights[str(i)]["kind"] for i in range(len(weights))] == (
            ["param", "buffer", "input"] if name == "weighted" else ["input"]
        )

    from merlin import api
    from merlin.compiler.backend import layout
    from merlin.compiler.frontend.inputs import load_compile_inputs

    result = api.compile(
        api.SavedExport(tmp_path / "layout.pt2", capture=tmp_path / "public-reference"),
        backend=layout, output=tmp_path / "public-compiled",
    )
    reopened = api.Compilation.load(result.manifest)
    load_compile_inputs(reopened.directory / "frontend")
    assert reopened.metadata["frontend"]["compiler_mlir_sha256"] == reopened.source_sha256
    assert not any(path.name in {"golden.npy", "inputs.npz"} for path in reopened.directory.rglob("*"))
    assert (tmp_path / "public-reference/reference/golden.npy").is_file()
    with pytest.raises(ValueError, match="buffer binding requires"):
        api.compile(
            api.SavedExport(tmp_path / "weighted.pt2", capture=tmp_path / "weighted-reference"),
            backend=layout, output=tmp_path / "weighted-rejected",
        )
    assert not (tmp_path / "weighted-rejected").exists()


def test_python_program_inputs_freeze_original_and_use_the_public_compile_service(tmp_path):
    """Real exporter/producer/compiler calls, including shared state and refusals."""
    checkout = root()
    python = capture_python_path(checkout=checkout)
    if not python.is_file() or not checkout.is_dir():
        pytest.skip("configure the actual pinned model2MLIR capture environment")
    assert capture_pin(checkout)["ok"]
    script = tmp_path / "programs.py"
    script.write_text('''import ctypes, json, subprocess, sys
from pathlib import Path
import torch
from merlin import api
from merlin.compiler.backend import layout
from merlin.compiler.frontend.torch.program import freeze
root = Path(sys.argv[1])
class Layout(torch.nn.Module):
    def forward(self, x): return x.reshape(6)
def function(x): return x.reshape(6)
model = Layout().train()
inputs = (torch.arange(6, dtype=torch.float32).reshape(2, 3),)
exported = torch.export.export(model, inputs, strict=False)
torch.export.save(exported, root/"source.pt2")
for name, program in (("module", model), ("fx", torch.fx.symbolic_trace(model)),
                      ("function", function), ("exported", exported), ("archive", root/"source.pt2")):
    result = api.compile(program, backend=layout, output=root/name,
                         example_inputs=None if name in {"exported", "archive"} else inputs)
    if name != "archive":
        record = result.metadata["frontend"]["original_program"]
        saved = root/(name + ".capture")/"original/program.pt2"
        assert record["archive_sha256"] == api.Artifact.from_file(saved, kind="export", role="original").sha256
    assert model.training and torch.equal(inputs[0], torch.arange(6).reshape(2, 3))
    assert not any(p.name in {"golden.npy", "inputs.npz"} for p in result.directory.rglob("*"))
    if name == "module":
        abi = result.metadata["c_call"]
        library = root/"layout.so"
        subprocess.run(["gcc", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                        *(str(result.directory/p) for p in abi["link_inputs"]), "-o", str(library)], check=True)
        call = getattr(ctypes.CDLL(str(library)), abi["symbol"])
        call.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
        call.restype = ctypes.c_int
        actual = torch.empty(6, dtype=torch.float32)
        assert call(None, (ctypes.c_void_p*1)(inputs[0].data_ptr()),
                    (ctypes.c_void_p*1)(actual.data_ptr())) == 0
        reference = torch.export.load(saved).module()(*inputs)
        assert torch.equal(actual.view(torch.int32), reference.view(torch.int32))
class Shared(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(3, 3, bias=False)
    def forward(self, x): return self.layer(x) + self.layer(x)
shared = Shared().train()
before = shared.layer.weight.detach().clone()
frozen, record = freeze(shared, example_inputs=inputs, workspace=root/"shared")
loaded = torch.export.load(frozen.path)
assert len(loaded.graph_signature.parameters) == 1
linear_calls = [n for n in loaded.graph.nodes
                if n.op == "call_function" and n.target == torch.ops.aten.linear.default]
assert len(linear_calls) == 2
assert torch.equal(before, shared.layer.weight) and shared.training
for program, args, message in ((model, None, "requires example_inputs"),
                               (exported, inputs, "already carries"),
                               (model, (3,), "flat positional tensor")):
    try:
        api.compile(program, backend=layout, output=root/"rejected", example_inputs=args)
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError("invalid input was accepted")
assert not (root/"rejected").exists() and not (root/"rejected.capture").exists()
print(json.dumps({"input_kinds": 5, "actual_scalar_c_execution": True, "shared_calls": 2}))
''', encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [
        str(repo_root() / "src"), environment.get("PYTHONPATH"),
    ]))
    completed = subprocess.run(
        [str(python), str(script), str(tmp_path)], env=environment,
        capture_output=True, text=True, check=True,
    )
    assert json.loads(completed.stdout.splitlines()[-1]) == {
        "input_kinds": 5, "actual_scalar_c_execution": True, "shared_calls": 2,
    }


def test_weighted_export_native_scalar_execution_matches_separate_reference(tmp_path):
    """Exercise real capture, immutable binding, upstream lowering and execution."""
    import ctypes
    import platform

    import numpy as np

    from merlin import api
    from merlin.compiler.backend import scalar
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.abi import descriptor
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.target_data_layout import observe_index_width

    checkout = root()
    python = capture_python_path(checkout=checkout)
    tools = (toolchain.llvm_install() / "bin/mlir-opt", toolchain.mlir_translate(), toolchain.clang())
    if (not python.is_file() or not checkout.is_dir() or not all(path.is_file() for path in tools)
            or platform.machine() != "x86_64" or toolchain._env("MERLIN_COMPILER_DRIVER") != "stock-cli"):
        pytest.skip("requires selected pinned capture and native stock MLIR/clang on x86_64")
    assert capture_pin(checkout)["ok"]
    source = tmp_path / "original.py"
    source.write_text('''import sys, torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
        with torch.no_grad():
            self.linear.weight.copy_(torch.arange(12).reshape(4, 3) / 8)
            self.linear.bias.copy_(torch.arange(4) / 4)
    def forward(self, x): return torch.relu(self.linear(x))
program = torch.export.export(Model().eval(), (torch.arange(-3, 3, dtype=torch.float32).reshape(2, 3),),
                              strict=False)
torch.export.save(program, sys.argv[1])
''', encoding="utf-8")
    archive = tmp_path / "model.pt2"
    subprocess.run([str(python), str(source), str(archive)], check=True, capture_output=True, text=True)
    flags = ("--target=x86_64-unknown-linux-gnu",)
    selected = observe_index_width(tools[2], flags)
    result = api.compile(archive, backend=scalar, output=tmp_path / "compiled", options=api.CompileOptions(parameters={
        "scalar.data-layout": selected["data_layout"], "scalar.index-bits": selected["index_bits"],
        "frontend.max-initialization-bytes": 64,
    }))
    result.verify()
    binding = result.metadata["frontend"]["argument_binding"]
    assert len(binding["initializers"]) == 2 and len(binding["runtime_inputs"]) == 1
    assert binding["unique_initialization_bytes"] == binding["initialization_byte_budget"] == 64
    assert "frontend.max-initialization-bytes" not in result.metadata["request"]["options"]["parameters"]
    library = tmp_path / "model.so"
    subprocess.run([str(tools[2]), *flags, "-O2", "-fno-vectorize", "-fno-slp-vectorize", "-fPIC", "-shared",
                    str(result.directory / "module.ll"), str(mlir_runtime_c()), "-lm", "-o", str(library)], check=True)
    call = ctypes.CDLL(str(library))._mlir_ciface_forward
    call.argtypes, call.restype = [ctypes.c_void_p] * 2, None
    inputs = np.arange(-3, 3, dtype=np.float32).reshape(2, 3)
    output = np.full((2, 4), np.nan, dtype=np.float32)
    arguments = [descriptor(value.ctypes.data, value.shape) for value in (inputs, output)]
    call(*(ctypes.byref(value) for value in arguments))
    # This evaluator reads the separate reference after compilation has finished.
    reference = np.load(Path(result.metadata["frontend"]["capture_workspace"]) / "reference/golden.npy")
    assert output.tobytes() == reference.tobytes()
    assert not any(path.name in {"inputs.npz", "golden.npy"} for path in result.directory.rglob("*"))
