"""Scope planning never manufactures session execution evidence."""

import importlib.util
import json

import yaml

from merlin.common.paths import repo_root


def tool():
    source = repo_root() / "examples/gemmini/phase1/launch/prepare_full_chain_scope.py"
    spec = importlib.util.spec_from_file_location("full_chain_scope", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected(tmp_path, *, session_steps):
    bundle = tmp_path / "bundle"
    (bundle / "configuration").mkdir(parents=True)
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "session_contract.yaml").write_text(
        yaml.safe_dump(
            {
                "kind": "image_stream",
                "stage_schedule": [{"name": "classify", "steps": session_steps}],
                "bindings": [],
                "states": [],
            }
        )
    )
    (bundle / "configuration/models.yaml").write_text(
        yaml.safe_dump(
            {
                "models": {
                    "sample": {"chain": {"kind": "single"}, "programs": [{"name": "model"}]},
                }
            }
        )
    )
    (bundle / "MANIFEST.json").write_text(json.dumps({"models": {"sample": {}}}))
    # Payload deliberately absent: preparation consumes metadata, never goldens.
    (bundle / "SHA256SUMS").write_text("a" * 64 + "  sample/visible/v00/int8.npz\n")
    private = tmp_path / "private.yaml"
    private.write_text(
        yaml.safe_dump(
            {
                "numerics_bundle": {"path": str(bundle)},
                "models": [
                    {
                        "id": "sample",
                        "capture": str(capture),
                        "deployment_dtype": "int8",
                        "capture_tree_sha256": "b" * 64,
                        "selected_input_bindings": [],
                        "capture_selection": "/metadata/selection.json",
                        "capture_selection_sha256": "c" * 64,
                        "capture_execution_attestation": "/metadata/attestation.json",
                        "capture_execution_attestation_sha256": "d" * 64,
                    }
                ],
            }
        )
    )
    return private


def test_session_scope_cannot_be_silently_reduced(tmp_path):
    report = tool().prepare(selected(tmp_path, session_steps=256))
    mismatch = next(x for x in report["blockers"] if x["code"] == "session_reference_scope_mismatch")
    assert (mismatch["declared_steps"], mismatch["selected_reference_steps"]) == (256, 1)
    assert report["status"] == "blocked_not_selected"
    assert report["execution_performed"] is False
    assert "numerical_qualification" not in report


def test_matching_stage_counts_are_not_session_proof(tmp_path):
    private = selected(tmp_path, session_steps=1)
    session_path = tmp_path / 'capture' / 'session_contract.yaml'
    session = yaml.safe_load(session_path.read_text())
    session['paper_ready'] = False
    session_path.write_text(yaml.safe_dump(session))
    report = tool().prepare(private)
    assert len(report["blockers"]) == 2
    assert [row['code'] for row in report['limitations']] == ['declared_not_paper_ready']
    assert report["payload_bytes_verified"] is False
    assert report["reference_payload_declarations"][0]["sha256"] == "a" * 64
    assert report["criterion_preview"]["output_scope"] == "all_declared"
    assert "deadline_unix" not in report["criterion_preview"]
    assert report["deferred_until_compiler_freeze"]
