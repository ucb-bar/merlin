"""Live runtime packaging preserves source identity and authenticated C artifacts."""

import ctypes as ct
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

import prepare_live_execution as packaging
from merlin.compile.semantic_application import CompiledCRegion, _render
from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
from merlin.xdsl_dialects.lowering.global_plan import (
    BufferRepresentation, CycleInterval, GlobalPlan, RegionAlternative, ValueRepresentation,
)


@pytest.fixture
def case(tmp_path, monkeypatch):
    # Exercise the shared semantic compiler's actual application/dispatch C.
    # The synthetic region copies bytes and has no target implementation.
    text = """module { func.func @forward(%x: tensor<2xbf16>) -> tensor<2xbf16> {
      %e = tensor.empty() : tensor<2xbf16>
      %r = linalg.copy ins(%x : tensor<2xbf16>) outs(%e : tensor<2xbf16>) -> tensor<2xbf16>
      return %r : tensor<2xbf16>
    } }"""
    source = translate_linalg_text(text, entry="forward", target_identity="synthetic").request
    x, y = source.input_storages[0][0], source.outputs[0]
    rep = ValueRepresentation("memory", "dense", "bf16")
    plan = GlobalPlan((RegionAlternative(
        "copy", (0,), "copy", "host", CycleInterval.unknown("test"),
        (BufferRepresentation(x, rep),), (BufferRepresentation(y, rep),),
    ),), (), CycleInterval.unknown("test"))
    adapter = SimpleNamespace(plan=lambda _: plan)
    lowered = CompiledCRegion("copy_bytes", "copy.h", (x,), (y,), {
        "copy.h": b"int copy_bytes(void *,const void *const *,void *const *);\n",
        "copy.c": b"#include <string.h>\nint copy_bytes(void *c,const void *const *i,void *const *o) "
                  b"{ (void)c; memcpy(o[0],i[0],4); return 0; }\n",
    }, (), (), {})
    _, regions, interface, links, files = _render(source, adapter, "capsule", 4096, [lowered])
    root, capture = tmp_path / "compiled", tmp_path / "capture"
    application = root / "application"
    for member, data in files.items():
        path = application / member
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest = dict(schema="merlin.semantic_application.v1", application_name="capsule",
                    application_interface=interface, application_memory_bytes=4096,
                    request_digest=source.digest(), regions=regions, segments=[], link_sources=links,
                    stage_artifacts={key: sha256(data).hexdigest() for key, data in files.items()})
    (application / "manifest.json").write_text(json.dumps(manifest))
    capture.mkdir()
    (capture / "model.mlir").write_text(text)
    tensor = torch.tensor([1., -0.], dtype=torch.bfloat16)
    np.savez(capture / "inputs.npz", in0=tensor.float().numpy())
    inputs, boundary = tmp_path / "inputs.safetensors", tmp_path / "source.json"
    save_file({"hidden": tensor}, str(inputs))
    boundary_record = dict(kind="synthetic", model_sha256="a" * 64, input_keys=["hidden"],
                           tensors={"hidden": dict(shape=[2], dtype="torch.bfloat16",
                             sha256=sha256(packaging.raw_bytes(tensor)).hexdigest())})
    boundary.write_text(json.dumps(boundary_record))
    staged = SimpleNamespace(original_source_sha256=sha256(text.encode()).hexdigest())
    from merlin.frontends import compile_inputs
    from merlin.targetgen import application_inventory

    monkeypatch.setattr(compile_inputs, "load_compile_inputs", lambda _: staged)
    monkeypatch.setattr(application_inventory, "verify_capture_receipt",
                        lambda _: {"status": "verified_materialized"})
    events, real_run = [], subprocess.run

    def run(command, **kwargs):
        if "native-verify" in command:
            events.append("verify")
            assert Path(command[command.index("--artifact") + 1]) == application
            assert command[command.index("--compile-inputs") + 1] == str(tmp_path / "staged")
            return subprocess.CompletedProcess(command, 0)
        events.append("cc")
        assert events[0] == "verify"
        return real_run(command, **kwargs)

    monkeypatch.setattr(packaging.subprocess, "run", run)
    argv = ["prepare_live_execution", "--compiled", str(root), "--capture", str(capture),
            "--staged", str(tmp_path / "staged"), "--snapshot", str(tmp_path / "snapshot"),
            "--atlas-root", str(tmp_path / "target"), "--boundary-source", str(boundary),
            "--expected-boundary-source-sha256", sha256(boundary.read_bytes()).hexdigest(),
            "--inputs", str(inputs)]
    monkeypatch.setattr(packaging.sys, "argv", argv)
    return SimpleNamespace(root=root, capture=capture, application=application, boundary=boundary,
                           inputs=inputs, staged=staged, events=events, manifest=manifest)


def test_real_emitted_host_library_preserves_words_and_native_inventory(case):
    if not shutil.which("cc"):
        pytest.skip("C compiler required")
    before = {p.relative_to(case.application): p.read_bytes()
              for p in case.application.rglob("*") if p.is_file()}
    packaging.main()
    assert {p.relative_to(case.application): p.read_bytes()
            for p in case.application.rglob("*") if p.is_file()} == before
    assert case.events == ["verify", "cc"]
    record = json.loads((case.root / "compiled.json").read_text())
    assert record["source_model_sha256"] == "a" * 64
    assert record["capture_mlir_sha256"] == sha256((case.capture / "model.mlir").read_bytes()).hexdigest()
    assert record["source_model_sha256"] != record["capture_mlir_sha256"]
    library = case.root / record["host_library"]
    assert record["host_library_sha256"] == sha256(library.read_bytes()).hexdigest()
    package = library.parent
    for member in case.manifest["stage_artifacts"]:
        if member.startswith("c-package/"):
            assert (package / Path(member).relative_to("c-package")).read_bytes() == before[Path(member)]
    assert not (package / "source-request.json").exists()
    with np.load(case.root / "tensors.npz", allow_pickle=False) as arrays:
        assert arrays["input_0"].tolist() == arrays["hidden"].tolist() == [0x3F80, 0x8000]
    # Actual C compilation and invocation exercise the extracted host-only shim.
    compiled = ct.CDLL(str(library))
    compiled.run.argtypes = [ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_void_p)]
    incoming = np.array([0x3F80, 0x8000], dtype=np.uint16)
    outgoing = np.zeros(2, dtype=np.uint16)
    assert compiled.run((ct.c_void_p * 1)(incoming.ctypes.data),
                        (ct.c_void_p * 1)(outgoing.ctypes.data)) == 0
    assert outgoing.tobytes() == incoming.tobytes()


@pytest.mark.parametrize("corruption,reason", [
    ("staged", "staged source differs"),
    ("model", "staged source differs"),
    ("captured_input", "worker inputs differ"),
    ("boundary", "boundary source digest differs"),
    ("tensor", "boundary tensor changed"),
])
def test_corrupt_or_unrelated_source_refuses_before_host_compilation(case, corruption, reason):
    if corruption == "staged":
        case.staged.original_source_sha256 = "b" * 64
    elif corruption == "model":
        (case.capture / "model.mlir").write_text("unrelated source")
    elif corruption == "captured_input":
        np.savez(case.capture / "inputs.npz", in0=np.array([2., -0.], np.float32))
    elif corruption == "boundary":
        record = json.loads(case.boundary.read_text())
        record["model_sha256"] = "b" * 64
        case.boundary.write_text(json.dumps(record))
    else:
        save_file({"hidden": torch.tensor([2., -0.], dtype=torch.bfloat16)}, str(case.inputs))
    with pytest.raises(ValueError, match=reason):
        packaging.main()
    assert "cc" not in case.events
    assert not (case.root / "compiled.json").exists()
    assert not (case.root / "runtime").exists()
