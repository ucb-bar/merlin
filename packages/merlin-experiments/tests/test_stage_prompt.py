"""Installed prompt artifact integrity and unchanged acceptance-template enforcement."""

from types import SimpleNamespace
import hashlib

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


def test_public_author_guidance_is_frozen_and_cannot_expand_authority(monkeypatch):
    text = "Inspect full-K panel reuse on the public tall contraction."
    record = SP.freeze_author_guidance(text)
    assert record["sha256"] == hashlib.sha256(text.encode()).hexdigest()
    rendered = SP.render_author_guidance(record)
    assert text in rendered and record["sha256"] in rendered
    assert "does not change" in rendered and "numerical checks" in rendered
    monkeypatch.setattr(SP, "render_capsule_development_prompt", lambda inputs: "canonical capsule prompt")
    inputs = SimpleNamespace(capsule_development={}, author_guidance=record, continue_queued_author=False)
    assert SP.render_stage_prompt(inputs) == "canonical capsule prompt" + rendered
    inputs.author_guidance = None
    assert SP.render_stage_prompt(inputs) == "canonical capsule prompt"
    with pytest.raises(StageGateError, match="frozen text identity"):
        SP.render_author_guidance({**record, "text": "changed"})
    with pytest.raises(StageGateError):
        SP.render_author_guidance({**record, "tool_grant": "extra"})


@pytest.mark.parametrize("value", ["", " ", "x\0y", "x" * ((256 << 10) + 1), 4],
                         ids=["empty", "blank", "nul", "oversized", "nontext"])
def test_public_author_guidance_refuses_unbounded_or_nontext_inputs(value):
    with pytest.raises(StageGateError):
        SP.freeze_author_guidance(value)
    assert SP.freeze_author_guidance(None) is None
