"""Complete signed outputs and exact source/metadata adapters stay separate from sealing."""

import json

import numpy as np
import pytest
from merlin_experiments.capture_execution import native_integer as N
from merlin_experiments.capture_execution.sealed_static import _file_digest
from merlin_experiments.phase0 import sealed_native_program as P


@pytest.mark.parametrize("layout", ["legacy", "single", "session"])
def test_native_program_reads_all_integer_goldens_and_refuses_source_or_numeric_drift(tmp_path, monkeypatch, layout):
    stage = tmp_path / ("capture/stages/first" if layout == "session" else "capture")
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
    capture = (
        dict(kind="session", capture_path=str(tmp_path / "capture"), programs=[dict(name="first")])
        if layout == "session"
        else dict(kind="single", capture_path=str(stage))
        if layout == "single"
        else dict(model_path=str(stage / "model.mlir"))
    )
    att = dict(selection=dict(path="selection", sha256="a" * 64), capture=capture)
    program = "first" if layout == "session" else None
    art = P.read(att, program=program, source_member="leaf.py")
    assert art.golden == [values.tolist(), values.tolist()]
    assert art.meta["frontend_trace"]["path"] == str(stage / "frontend-trace.json")
    assert json.loads((stage / "meta.json").read_bytes())["frontend_trace"]["path"] == "frontend-trace.json"
    with pytest.raises(ValueError, match="exact program" if layout == "session" else "session program"):
        P.read(att, program="absent", source_member="leaf.py")
    with pytest.raises(ValueError, match="relative"):
        P.read(att, program=program, source_member="../leaf.py")
    adapter = P.Source(att, program=program, source_member="leaf.py")
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
        P.read(att, program=program, source_member="leaf.py")


def test_v3_single_root_uses_actual_complete_capture_attestation_layout(tmp_path):
    # Shape emitted by attest_sealed_m2m_v3 for a genuine complete single
    # program: capture_path is the DIRECTORY; model_path is absent. This
    # adapter-only test neither creates nor verifies an execution attestation.
    capture = dict(
        run_dir=str(tmp_path),
        capture_path=str(tmp_path / "capture"),
        kind="single",
        capture_tree_sha256="a" * 64,
        sealed_receipt_sha256="b" * 64,
        integer_contractions=3,
        model_sha256="c" * 64,
        receipt_sha256="d" * 64,
    )
    assert P._root(dict(capture=capture), None) == tmp_path / "capture"


@pytest.mark.parametrize(
    "capture, program, message",
    [
        ({"kind": "unknown", "model_path": "/capture/model.mlir"}, None, "kind"),
        ({"kind": None, "model_path": "/capture/model.mlir"}, None, "kind"),
        ({"kind": [], "capture_path": "/capture"}, None, "kind"),
        ({"kind": "single"}, None, "root"),
        ({"kind": "single", "capture_path": ""}, None, "root"),
        ({"kind": "single", "capture_path": 1}, None, "root"),
        ({}, None, "root"),
        ({"capture_path": "/capture"}, None, "root"),
        ({"kind": "single", "model_path": "/capture/model.mlir"}, None, "root"),
        ({"model_path": "/capture/model.mlir", "capture_path": "/capture"}, None, "ambiguous"),
        ({"kind": "single", "model_path": "/capture/model.mlir", "capture_path": "/capture"}, None, "ambiguous"),
        ({"kind": "session", "model_path": "/capture/model.mlir", "capture_path": "/capture"}, "first", "ambiguous"),
        ({"kind": "session", "programs": [{"name": "first"}]}, "first", "root"),
        ({"kind": "session", "capture_path": "/capture", "programs": [{"name": "first"}]}, None, "exact program"),
        (
            {"kind": "session", "capture_path": "/capture", "programs": [{"name": "first"}, {"name": "first"}]},
            "first",
            "exact program",
        ),
    ],
)
def test_native_root_refuses_unknown_or_ambiguous_layout(capture, program, message):
    with pytest.raises(ValueError, match=message):
        P._root(dict(capture=capture), program)


def test_v3_single_reader_requires_original_attestation_before_selection_or_arrays(monkeypatch):
    observed = []

    def refuse(attestation):
        observed.append(attestation)
        raise ValueError("original attestation required")

    monkeypatch.setattr(P, "require_verified_execution", refuse)
    monkeypatch.setattr(P.capture_selection, "load", lambda *_a, **_k: pytest.fail("selection before attestation"))
    monkeypatch.setattr(np, "load", lambda *_a, **_k: pytest.fail("arrays before attestation"))
    attestation = dict(capture=dict(kind="single", capture_path="/capture"))
    with pytest.raises(ValueError, match="original attestation required"):
        P.read(attestation, program=None, source_member="loader.py")
    assert observed == [attestation]


def test_explicit_public_payload_bound_is_checked_before_any_array_or_source_loading(tmp_path, monkeypatch):
    stage = tmp_path / "capture/stages/first"
    stage.mkdir(parents=True)
    (stage / "inputs.npz").write_bytes(b"x" * 2048)
    (stage / "golden.json").write_text("[]")
    (stage / "capture_receipt.json").write_text(json.dumps(dict(artifacts={"inputs.npz": {}})))
    monkeypatch.setattr(P, "require_verified_execution", lambda _attestation: None)
    monkeypatch.setattr(P.capture_selection, "load", lambda *_a, **_k: dict(plan={}, run_dir=str(tmp_path)))
    monkeypatch.setattr(np, "load", lambda *_a, **_k: pytest.fail("denied reference may not allocate arrays"))
    att = dict(
        selection=dict(path="selection", sha256="a" * 64),
        capture=dict(kind="session", capture_path=str(tmp_path / "capture"), programs=[dict(name="first")]),
    )
    with pytest.raises(ValueError, match="bound before array loading"):
        P.read(att, program="first", source_member="absent.py", max_payload_bytes=1024)
