"""Only a structured frontend refusal authorizes unchanged host fallback."""

import json
from types import SimpleNamespace

import pytest

from merlin.targetgen.capsule_source import M2MUnavailable, NoEligibleCaptureOperations, PytorchRefSource


@pytest.mark.parametrize("typed", [True, False])
def test_capture_refusal_is_typed_independently_of_message(tmp_path, monkeypatch, typed):
    source = PytorchRefSource(m2m_dir=tmp_path)
    message = "different diagnostic wording" if typed else "the model has no operator the recipe's families cover"

    def worker(*args, **kwargs):
        if typed:
            (tmp_path / "refusal.json").write_text(
                json.dumps(
                    dict(
                        schema="merlin.capture.refusal.v1",
                        code="no_eligible_recipe_operations",
                        reason=message,
                    )
                )
            )
        return SimpleNamespace(returncode=4 if typed else 1, stderr=message)

    monkeypatch.setattr(source, "_launch_worker", worker)
    expected = NoEligibleCaptureOperations if typed else M2MUnavailable
    with pytest.raises(expected) as observed:
        source._run_capture(
            tmp_path / "loader.py",
            "model",
            "int8",
            workdir=tmp_path,
            src="",
            scheme=None,
            declared_env={},
            interpreter=tmp_path / "python",
            recipe=None,
            recipe_sha256="",
            slot=None,
        )
    assert isinstance(observed.value, NoEligibleCaptureOperations) is typed


def test_removed_readout_fold_cannot_traverse_add_or_arbitrary_reshape():
    from merlin.capture import integerization

    assert not hasattr(integerization, "combine_static_readout_scales")
