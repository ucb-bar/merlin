"""Installed prompt artifact integrity and unchanged acceptance-template enforcement."""

from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import stage_prompt as SP
from merlin_experiments.phase2.claims import dispatch as CD
from merlin_experiments.phase2.contracts import StageGateError


def test_exact_prompt_bytes_and_immutable_materialization(tmp_path, monkeypatch):
    monkeypatch.setattr(SP, "render_stage_prompt", lambda inputs: "exact prompt\n")
    path = tmp_path / "prompt.md"
    artifact = SP.materialize_canonical_prompt(None, path)
    assert artifact.text == "exact prompt\n"
    assert artifact.n_bytes == len(path.read_bytes())
    with pytest.raises(FileExistsError):
        SP.materialize_canonical_prompt(None, path)
    assert path.read_bytes() == b"exact prompt\n"
    link = tmp_path / "linked.md"
    link.symlink_to(path)
    with pytest.raises(StageGateError):
        SP.load_prompt(link)


def test_acceptance_template_is_exact_and_engine_parameter_is_retained():
    observed = []
    declaration = {"evidence": {"timing_simulator": "fixture"}, "threshold": 3}

    def supported(engine):
        observed.append(engine)
        return declaration

    module = SimpleNamespace(supported_acceptance=supported)
    CD.verify_supported_acceptance(module, declaration, "family")
    assert observed == ["fixture"]
    with pytest.raises(StageGateError, match="differs"):
        CD.verify_supported_acceptance(module, {**declaration, "threshold": 4}, "family")
    assert CD.supported_acceptance(SimpleNamespace(), declaration, "family") is None
