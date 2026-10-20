"""Original target pass selection survives ordinary whole-model callbacks."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_selected_schedule_contracts import inputs

from merlin.compile.scheduling.configuration_ir import ConfigurationIRSelection
from merlin.llvmlower.device_build import (
    DeviceRouting,
    freeze_device_target_configuration,
    require_device_target_configuration,
)


def route(tmp_path, *, target="neutral"):
    selected, *_ = inputs(tmp_path)
    return DeviceRouting(
        target,
        tmp_path,
        "i8",
        "i32",
        target_configuration=ConfigurationIRSelection(selected, "independent-neutral-test"),
    )


def test_absent_selection_keeps_ordinary_route():
    assert freeze_device_target_configuration(None) is None
    assert require_device_target_configuration(None, None) is None


def test_untyped_configuration_cannot_enter_route():
    with pytest.raises(ValueError, match="original ConfigurationIRSelection"):
        freeze_device_target_configuration(SimpleNamespace(target_configuration={}, device="neutral"))


def test_different_target_refuses_original_selection(tmp_path):
    with pytest.raises(ValueError, match="different target"):
        freeze_device_target_configuration(route(tmp_path, target="other"))


def test_replacement_after_preparation_refuses_even_equal_controls(tmp_path):
    original = route(tmp_path)
    frozen = freeze_device_target_configuration(original)
    assert require_device_target_configuration(original, frozen) is original.target_configuration
    with pytest.raises(ValueError, match="selection changed"):
        require_device_target_configuration(
            replace(original, target_configuration=replace(original.target_configuration)),
            frozen,
        )


def test_late_injection_refuses(tmp_path):
    with pytest.raises(ValueError, match="selection changed"):
        require_device_target_configuration(route(tmp_path), None)


def test_original_facts_change_refuses_before_emission(tmp_path):
    original = route(tmp_path)
    frozen = freeze_device_target_configuration(original)
    original.target_configuration.selected_contracts.facts_path.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        require_device_target_configuration(original, frozen)


def test_selected_pass_cannot_silently_miss_the_consumed_ir(tmp_path):
    from merlin.llvmlower.device_build import revalidate_device_target_artifacts

    original = route(tmp_path)
    with pytest.raises(ValueError, match="no consumed package IR"):
        revalidate_device_target_artifacts(original, freeze_device_target_configuration(original), None)


@pytest.mark.parametrize("role", ["receipt", "selected_target", "target_artifact"])
def test_late_provider_mutation_cannot_leave_successful_target_evidence(tmp_path, monkeypatch, role):
    from test_target_ir_consumer_protocol import run, selection

    from merlin.llvmlower.device_build import revalidate_device_target_artifacts
    from merlin.targetgen import package_runtime

    selected = selection(tmp_path)
    package, generated, _, _ = run(tmp_path, selected)
    original = DeviceRouting("neutral", tmp_path, "i8", "i32", target_configuration=selected)
    frozen = freeze_device_target_configuration(original)
    monkeypatch.setattr(package_runtime, "load_package", lambda _path: package)
    import hashlib
    import json

    path = generated / "target_configuration.json"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    artifacts = {
        "kernels": {
            "entry": {
                "custody": {
                    "target_configuration": {
                        "path": str(path),
                        "sha256": digest,
                    }
                }
            }
        }
    }
    revalidate_device_target_artifacts(original, frozen, artifacts)
    if role != "receipt":
        from pathlib import Path

        path = Path(json.loads(path.read_text())["files"][role]["path"])
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed"):
        revalidate_device_target_artifacts(original, frozen, artifacts)
