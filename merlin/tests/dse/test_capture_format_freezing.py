"""Installed fixed format consumer controls; no model, runtime or package qualification."""

import copy
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from capture_contraction_inputs import _pin, _selection

from merlin.capture.contraction_formats import ContractionFormatError
from merlin.compare import paper_measurement_freeze as M


def _capture(root, form="integer"):
    selection = _selection(root, formats=(form,))
    inputs = np.asarray([[2.0], [3.0]], dtype=np.float32)
    reference = inputs * 2
    np.savez(root / "inputs.npz", stream=inputs)
    np.savez(root / "reference.npz", output=reference)
    contract = {
        "version": 1,
        "streams": [{"key": "stream"}],
        "inputs": "inputs.npz",
        "quality": {
            "scope": "trajectory",
            "reference": "eager_fp32",
            "metric": "exact",
            "golden": "reference.npz",
            "key": "output",
            "reference_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
        },
    }
    path = root / "session_contract.yaml"
    path.write_text(yaml.safe_dump(contract))
    return replace(selection, session_contract=_pin(path))


def _sources(root, selection=None):
    return M.write_capture_measurement_source_receipt(
        root, model="synthetic", precision="int8", observations=2, contraction_selection=selection
    )


def _constructor(root, tmp_path, monkeypatch, selection):
    # These unrelated package placeholders are unit-test dispatch data; this
    # test never claims package/source closure or executes a backend.
    package = tmp_path / "package.json"
    package.write_text(json.dumps({"object_recipe": "diagnostic"}))
    resource = {
        "package_receipt": {"path": str(package), "sha256": _pin(package).sha256},
        "runtime_artifact": {"path": str(package), "sha256": _pin(package).sha256},
    }
    backend = {"name": "diagnostic", "adapter": "diagnostic"}
    model = {"session": {"observations": 2, "kind": "synthetic"}, "artifacts": {"int8": {"sha256": "a" * 64}}}
    monkeypatch.setattr(M, "_resource_sets", lambda _: [(backend, "synthetic", model, "int8", resource, [])])
    monkeypatch.setattr(M, "_validate_package_before_private_io", lambda *a, **kw: None)
    selections = None if selection is None else {("synthetic", "int8"): selection}
    return M.construct_measurement_evidence(
        {"backends": [backend], "freeze": {}, "target": "diagnostic"},
        capture_roots={("synthetic", "int8"): root},
        output_path=tmp_path / "frozen.yaml",
        toolchain_authority_path=package,
        toolchain_authority_sha256=_pin(package).sha256,
        contraction_selections=selections,
    )


def test_default_mixed_receipt_meaning_is_unchanged(tmp_path, monkeypatch):
    root = tmp_path / "capture"
    _capture(root, "float")
    document = json.loads(_sources(root).read_text())
    assert document["kind"] == "paper_measurement_capture_sources_v1"
    assert "contraction_formats" not in document
    result, _ = _constructor(root, tmp_path, monkeypatch, None)
    assert result["diagnostic"]["synthetic"]["int8"]["reference_output"]


def test_explicit_policy_is_recomputed_at_both_fixed_boundaries(tmp_path, monkeypatch):
    root = tmp_path / "capture"
    selection = _capture(root)
    path = _sources(root, selection)
    doc = json.loads(path.read_text())
    assert doc["schema_version"] == 3 and doc["contraction_formats"]["census"]["eligible"] is True
    derived, retained = _constructor(root, tmp_path, monkeypatch, selection)
    assert derived["diagnostic"]["synthetic"]["int8"]["reference_output"]
    assert all(
        pin.path in retained
        for pin in (
            selection.session_contract,
            *[pin for row in selection.programs for pin in (row.original_graph, row.frontend_trace, row.final_mlir)],
        )
    )


def test_complete_integer_policy_refuses_plain_float_before_registration(tmp_path):
    root = tmp_path / "capture"
    selection = _capture(root, "float")
    with pytest.raises(ContractionFormatError, match="coverage is not established"):
        _sources(root, selection)
    assert not (root / "paper_measurement_sources.json").exists()


@pytest.mark.parametrize("change", ["no_selection", "saved_census", "bool_count", "mlir", "original"])
def test_saved_eligibility_or_changed_source_cannot_freeze(tmp_path, monkeypatch, change):
    root = tmp_path / "capture"
    selection = _capture(root)
    path = _sources(root, selection)
    if change in {"saved_census", "bool_count"}:
        doc = json.loads(path.read_text())
        doc["contraction_formats"]["census"]["counts"]["integer"] = 999 if change == "saved_census" else True
        path.write_text(json.dumps(doc))
    elif change in {"mlir", "original"}:
        pin = selection.programs[0].final_mlir if change == "mlir" else selection.programs[0].original_graph
        pin.path.write_bytes(pin.path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="receipt|source bytes changed"):
        _constructor(root, tmp_path, monkeypatch, None if change == "no_selection" else selection)


def test_unknown_explicit_registration_cell_is_not_silently_skipped(tmp_path):
    selection = _capture(tmp_path / "capture")
    with pytest.raises(ValueError, match="membership is incomplete"):
        M.construct_measurement_evidence(
            {},
            capture_roots={},
            output_path=tmp_path / "frozen.yaml",
            toolchain_authority_path=tmp_path / "unused",
            toolchain_authority_sha256="a" * 64,
            contraction_selections={("unselected", "int8"): selection},
        )


def test_selected_registration_cell_must_be_consumed_by_requested_backend(tmp_path):
    root = tmp_path / "capture"
    selection = _capture(root)
    with pytest.raises(ValueError, match="membership is incomplete"):
        M.construct_measurement_evidence(
            {"models": [], "backends": []},
            capture_roots={("synthetic", "int8"): root},
            output_path=tmp_path / "frozen.yaml",
            toolchain_authority_path=tmp_path / "unused",
            toolchain_authority_sha256="a" * 64,
            contraction_selections={("synthetic", "int8"): selection},
        )


def test_receipt_writer_cannot_follow_preexisting_dangling_symlink(tmp_path):
    root = tmp_path / "capture"
    selection = _capture(root)
    excluded = tmp_path / "excluded-owned-file"
    (root / "paper_measurement_sources.json").symlink_to(excluded)
    with pytest.raises(FileExistsError):
        _sources(root, selection)
    assert not excluded.exists()


def test_normal_freeze_study_forwards_selection_to_fixed_constructor(tmp_path, monkeypatch):
    from merlin.compare import freeze
    from merlin.compare import paper_toolchain_authority as authority

    root = tmp_path / "capture"
    selection = _capture(root, "float")
    source = tmp_path / "study.yaml"
    (tmp_path / "merlin/python/merlin").mkdir(parents=True)
    (tmp_path / "merlin/python/merlin/owned.py").write_text("# diagnostic source\n")
    (tmp_path / "pyproject.toml").write_text("# diagnostic project\n")
    paper_inputs = tmp_path / "inputs"
    paper_inputs.write_bytes(b"owned diagnostic inputs")
    raw = {
        "target": "diagnostic",
        "models": [
            {"name": "synthetic", "artifacts": {"int8": {"path": str(root), "variant": "int8", "sha256": "unresolved"}}}
        ],
        "backends": [{"name": "diagnostic", "adapter": "diagnostic", "kind": "frozen_baseline", "options": {}}],
        "freeze": {},
        "paper_inputs": {"path": str(paper_inputs), "sha256": freeze.sha256_paths([paper_inputs])},
    }
    source.write_text(yaml.safe_dump(raw))
    model = SimpleNamespace(name="synthetic", capture="synthetic", session=object(), expected_provenance={})
    spec = SimpleNamespace(
        source_path=source,
        target="diagnostic",
        models=(model,),
        paper_inputs=raw["paper_inputs"],
        canonical_dict=lambda: copy.deepcopy(raw),
    )
    # Only the ordinary forwarding seam is under test, not upstream admission.
    monkeypatch.setattr(freeze.PaperStudySpec, "parse", lambda *a, **kw: spec)
    monkeypatch.setattr(freeze, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(freeze, "_git_sha", lambda _: "a" * 40)
    monkeypatch.setattr(
        freeze, "_require_external_package_registration", lambda *a: (source, "a" * 64, {"packages": []})
    )
    monkeypatch.setattr(authority, "load_toolchain_authority", lambda *a, **kw: None)
    monkeypatch.setattr(M, "validate_packages_before_private_io", lambda *a, **kw: None)
    monkeypatch.setattr(freeze, "validate_paper_input_binding", lambda *a: [])
    monkeypatch.setattr(freeze, "validate_capture_session", lambda *a, **kw: ({}, []))
    monkeypatch.setattr(freeze.bundle.CaptureBundle, "require", lambda value: value)
    from merlin.baselines import executorch_session

    monkeypatch.setattr(executorch_session, "capture_session_identity", lambda _: {})
    monkeypatch.setattr(executorch_session, "session_identity_sha256", lambda _: "a" * 64)
    # Use the actual fixed consumer, before any measurement/package construction.
    original = M.construct_measurement_evidence

    def consume(raw, **kwargs):
        assert kwargs["contraction_selections"] == {("synthetic", "int8"): selection}
        return original(raw, **kwargs)

    monkeypatch.setattr(M, "construct_measurement_evidence", consume)
    monkeypatch.setattr(
        M,
        "_resource_sets",
        lambda _: [
            (
                {"name": "diagnostic", "adapter": "diagnostic"},
                "synthetic",
                {"session": {"observations": 2}},
                "int8",
                {},
                [],
            )
        ],
    )
    monkeypatch.setattr(M, "_validate_package_before_private_io", lambda *a, **kw: None)
    with pytest.raises(ContractionFormatError, match="coverage is not established"):
        freeze.freeze_study(
            spec,
            policy_path=source,
            runtime_paths=[source],
            toolchain_authority_path=source,
            output_path=tmp_path / "frozen.yaml",
            contraction_selections={("synthetic", "int8"): selection},
        )
    assert not (tmp_path / "frozen.yaml").exists()
