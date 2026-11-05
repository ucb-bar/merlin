"""Publication failures must leave the requested output destination absent."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType


def test_missing_success_metadata_fails_before_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from merlin.semantic_compiler import snapshot, target_binding
    from merlin.targetgen import cli

    tensor = TensorType((1,), "i32", "exact-i32")
    request = KernelRequest(
        (SemanticNode("x", "input", (), tensor, effect="input"),),
        ("x",),
        ("external",),
        (("x", "external"),),
        "synthetic-publication",
        lowering_policy="diagnostic",
    )
    profile = SimpleNamespace(target_identity=request.target_identity, digest=lambda: "profile")

    def compile_files(_snapshot, source, *, destination, **_kwargs):
        # Plumbing fixture only: valid file identities, missing report metadata.
        destination.mkdir()
        binary, plan = b"\x13\0\0\0", b"{}"
        (destination / "program.bin").write_bytes(binary)
        (destination / "execution_plan.json").write_bytes(plan)
        manifest = {
            "engine": "merlin_native",
            "request_digest": source.digest(),
            "target_identity": source.target_identity,
            "binary_sha256": hashlib.sha256(binary).hexdigest(),
            "execution_plan_sha256": hashlib.sha256(plan).hexdigest(),
            "stage_artifacts": {},
        }
        (destination / "manifest.json").write_text(json.dumps(manifest))
        return manifest

    binding = SimpleNamespace(profile=lambda: profile, compile=compile_files)
    monkeypatch.setattr(snapshot, "open_native_snapshot", lambda _root: SimpleNamespace(profile=profile))
    monkeypatch.setattr(target_binding, "load_native_target_binding", lambda _name: binding)
    precision = SimpleNamespace(record=lambda: {"schema": "synthetic.precision_plumbing.v1"})
    monkeypatch.setattr(
        cli, "_load_selected_native_request", lambda *_args, **_kwargs: (request, "semantic", (), precision)
    )
    abi = tmp_path / "abi.json"
    abi.write_text(json.dumps({"fixed_inputs": {"x": 0}, "fixed_outputs": [1]}))
    output, status = tmp_path / "compiled", tmp_path / "status.json"
    returned = cli.main(
        [
            "native-compile",
            "--engine",
            "merlin_native",
            "--support",
            "synthetic",
            "--snapshot",
            str(tmp_path / "snapshot"),
            "--request",
            str(tmp_path / "request.json"),
            "--abi",
            str(abi),
            "--target-source",
            str(tmp_path),
            "--mode",
            "diagnostic",
            "--out",
            str(output),
            "--status-file",
            str(status),
        ]
    )
    assert returned == 2
    assert not output.exists()
    report = json.loads(status.read_text())
    assert report["status"] == "compile_error"
    assert "constant_inputs" in report["reason"]
