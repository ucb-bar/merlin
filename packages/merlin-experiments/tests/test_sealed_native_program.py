"""Complete signed outputs and exact source/metadata adapters stay separate from sealing."""

import json

import numpy as np
import pytest
from merlin_experiments.capture_execution import native_integer as N
from merlin_experiments.capture_execution.sealed_static import _file_digest
from merlin_experiments.phase0 import sealed_native_program as P


def test_native_program_reads_all_integer_goldens_and_refuses_source_or_numeric_drift(tmp_path, monkeypatch):
    stage = tmp_path / "capture/stages/first"
    stage.mkdir(parents=True)
    source = tmp_path / "snapshots/source/workload/leaf.py"
    source.parent.mkdir(parents=True)
    source.write_text("explicit original public source\n")
    (
        stage / "model.mlir"
    ).write_text("""module { func.func @forward(%x: tensor<2xi32>) -> (tensor<2xi32>, tensor<2xi32>) {
        func.return %x, %x : tensor<2xi32>, tensor<2xi32>
    } }""")
    values = np.asarray([-16777217, 16777217], dtype=np.int32)
    np.savez(stage / "inputs.npz", in0=values)
    (stage / "golden.json").write_text(json.dumps([values.tolist(), values.tolist()]))
    # A lossy historical output-zero file must not supply any of these references.
    np.save(stage / "golden.npy", values.astype(np.float32))
    trace = dict(
        schema="m2m.frontend_trace.v1", graphs=dict(original=dict(schema="m2m.frontend_graph.v1", status="complete"))
    )
    (stage / "frontend-trace.json").write_text(json.dumps(trace))
    (stage / "pytorch-opset.json").write_text("{}")
    meta = dict(
        capture_quantization="already_materialized",
        source_mode=N.MODE,
        quantization_stats=dict(applied=False),
        ok=True,
        opaque=0,
        input_abi=[dict(shape=[2], dtype="i32")],
        output_abi=[dict(shape=[2], dtype="i32")] * 2,
        frontend_trace=dict(path="frontend-trace.json", sha256=_file_digest(stage / "frontend-trace.json")),
        framework_catalog=dict(path="pytorch-opset.json", sha256=_file_digest(stage / "pytorch-opset.json")),
    )
    (stage / "meta.json").write_text(json.dumps(meta))
    plan = dict(
        dtype="int8",
        recipe=None,
        worker_options=dict(source_mode=N.MODE),
        m2m_root=str(tmp_path),
        native_integer_verifier_sha256=_file_digest(__import__("pathlib").Path(N.__file__)),
    )
    # This unit test exercises the reader only. Real isolated seal/replay remains a separate integration gate.
    monkeypatch.setattr(P, "require_verified_execution", lambda attestation: None)
    monkeypatch.setattr(P.capture_selection, "load", lambda *a, **k: dict(plan=plan, run_dir=str(tmp_path)))
    att = dict(
        selection=dict(path="selection", sha256="a" * 64),
        capture=dict(kind="session", capture_path=str(tmp_path / "capture"), programs=[dict(name="first")]),
    )
    art = P.read(att, program="first", source_member="leaf.py")
    assert art.golden == [values.tolist(), values.tolist()]
    assert art.meta["frontend_trace"]["path"] == str(stage / "frontend-trace.json")
    assert json.loads((stage / "meta.json").read_bytes())["frontend_trace"]["path"] == "frontend-trace.json"
    with pytest.raises(ValueError, match="exact program"):
        P.read(att, program="absent", source_member="leaf.py")
    with pytest.raises(ValueError, match="relative"):
        P.read(att, program="first", source_member="../leaf.py")
    adapter = P.Source(att, program="first", source_member="leaf.py")
    with pytest.raises(ValueError, match="transformation"):
        adapter.capture_loader(source, "int8", already_quantized=True, recipe={"forbidden": True})
    with pytest.raises(ValueError, match="transformation"):
        adapter.capture_loader(source, "int8", already_quantized=True, agreement_tolerance=(0.1, 0))
    copied = tmp_path / "changed.py"
    copied.write_text("changed source")
    with pytest.raises(ValueError, match="source request"):
        adapter.capture_loader(copied, "int8", already_quantized=True)
    assert (
        adapter.capture_loader(source, "int8", already_quantized=True, agreement_tolerance=(0, 0)).golden == art.golden
    )
    assert adapter.attestations == [att]
    (stage / "frontend-trace.json").write_text("{}")
    with pytest.raises(ValueError, match="original frontend"):
        P.read(att, program="first", source_member="leaf.py")
