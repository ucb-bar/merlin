"""Descriptor folding retains exact capture authority after its query disappears."""

from types import SimpleNamespace

import pytest

from merlin.targetgen import core_aten_capture as capture


@pytest.mark.parametrize(
    "proof,expected",
    [
        (
            {
                "authority": "static_capture_input_descriptors",
                "queries": [{"operator": "aten.sym_numel.default", "value": 6}],
            },
            "captured_exact",
        ),
        (
            {"authority": "unknown", "queries": [{"operator": "aten.sym_numel.default", "value": 6}]},
            "captured_lost_exact_provenance",
        ),
        ({}, "captured_lost_exact_provenance"),
    ],
)
def test_constant_capture_requires_descriptor_specialization_proof(tmp_path, monkeypatch, proof, expected):
    artifact = SimpleNamespace(
        linalg_mlir="builtin.module {}",
        inputs={},
        golden={},
        meta={"static_metadata_specialization": proof},
    )
    monkeypatch.setattr(
        capture,
        "PytorchRefSource",
        lambda **kwargs: SimpleNamespace(capture_loader=lambda *args, **kwargs: artifact, m2m_dir=tmp_path),
    )
    monkeypatch.setattr(capture, "_git_revision", lambda path: None)
    corpus = dict(
        pytorch_version="test",
        cases=[
            dict(
                overload="aten.sym_numel.default",
                arguments={"args": []},
                expected=6,
            )
        ],
    )
    result = capture.capture_case_corpus(corpus, tmp_path)
    assert result["results"]["aten.sym_numel.default"]["status"] == expected


def test_descriptor_specialization_binding_refuses_changed_input_or_missing_proof():
    import copy

    case = {
        "overload": "aten.sym_numel.default",
        "arguments": {"args": [{"kind": "tensor", "shape": [2, 3], "stride": [3, 1], "storage_offset": 0}]},
    }
    proof = {
        "authority": "static_capture_input_descriptors",
        "inputs": [{"shape": [2, 3], "stride": [3, 1], "storage_offset": 0}],
        "queries": [{"operator": "aten.sym_numel.default", "value": 6}],
    }
    receipt = {"capture_meta": {"static_metadata_specialization": proof}}
    assert capture.descriptor_specialization_matches(case, receipt)
    for key, value in (("shape", [3, 2]), ("stride", [6, 2]), ("storage_offset", 1)):
        changed = copy.deepcopy(case)
        changed["arguments"]["args"][0][key] = value
        assert not capture.descriptor_specialization_matches(changed, receipt)
    # The capture proof, not the case's overload name, is the authority: renaming
    # the case neither grants nor refuses the fold; a proof without folded
    # queries refuses.
    assert capture.descriptor_specialization_matches(dict(case, overload="aten.clone.default"), receipt)
    assert not capture.descriptor_specialization_matches(
        case, {"capture_meta": {"static_metadata_specialization": dict(proof, queries=[])}}
    )
    proof["authority"] = "unknown"
    assert not capture.descriptor_specialization_matches(case, receipt)
