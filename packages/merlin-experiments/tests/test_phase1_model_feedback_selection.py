"""Descriptor capabilities select optional feedback; target names grant nothing."""

from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1.providers.execution import automatic_model_gate_interval


@pytest.mark.parametrize("target", ["first_accelerator", "second_accelerator"])
def test_declared_model_feedback_is_target_neutral(tmp_path, target):
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text(
        yaml.safe_dump(
            {
                "target": target,
                "phase1_gates": {
                    "private_full_models": {
                        "required": True,
                        "models": ["network"],
                        "programs": {"network": ["encode", "step"]},
                    }
                },
            }
        )
    )
    context = SimpleNamespace(target=target, descriptor=descriptor)
    options = SimpleNamespace(experiment="full", grade_interval=137, validation_scope="full")
    assert automatic_model_gate_interval(context, options) == 137
    options.validation_scope = "capsules"
    assert automatic_model_gate_interval(context, options) == 0
    options.validation_scope, options.experiment = "full", "pilot"
    assert automatic_model_gate_interval(context, options) == 0
    descriptor.write_text(yaml.safe_dump({"target": target}))
    options.experiment = "full"
    assert automatic_model_gate_interval(context, options) == 0


def test_malformed_model_declaration_is_not_silently_enabled(tmp_path):
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text(
        yaml.safe_dump({"phase1_gates": {"private_full_models": {"required": True, "models": ["network", "network"]}}})
    )
    with pytest.raises(ValueError, match="repeats"):
        automatic_model_gate_interval(
            SimpleNamespace(descriptor=descriptor), SimpleNamespace(experiment="full", grade_interval=137)
        )
