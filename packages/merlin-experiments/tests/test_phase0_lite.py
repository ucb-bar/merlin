"""The selected spec/capture flow has no advisory-evidence dependency."""

import pytest
import yaml
from merlin_experiments.phase0 import profiles


def test_synthesis_binds_selected_inputs_without_record_or_campaign(tmp_path):
    recipe, descriptor, requirement, synthesis = [tmp_path / name for name in (
        "recipe.yaml", "descriptor.yaml", "requirements.yaml", "synthesis.yaml"
    )]
    recipe.write_text("{}\n")
    descriptor.write_text(yaml.safe_dump({
        "target": "toy",
        "workload_spec": {
            "phase0_spec_validation": "required",
            "fusion_validation": {"path": str(tmp_path / "absent-campaign.json"), "sha256": "0" * 64},
        },
    }))
    requirement.write_text("name: toy\n")
    arguments = dict(recipe=recipe, descriptor=descriptor, conformance_spec=requirement)
    identity = profiles.synthesis_input_identity(**arguments)
    synthesis.write_text(yaml.safe_dump({"provenance": {"selected_inputs": identity}}))
    assert profiles.verify_selected_synthesis(synthesis, **arguments)["status"] == "verified"
    # Removing the advisory gate must retain the upstream input-byte binding.
    recipe.write_text("{}\n# changed selected bytes\n")
    with pytest.raises(ValueError, match="recipe_sha256 changed"):
        profiles.verify_selected_synthesis(synthesis, **arguments)
