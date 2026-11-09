"""Legacy vector policy delegates to OOT while keeping generic/scalar isolation."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common.yaml import load_yaml
from merlin.kernels import microkernel
from merlin.mining import from_strategy as legacy


@pytest.fixture
def selected():
    try:
        return legacy._codegen()
    except RuntimeError as exc:
        if "requires its OOT package" not in str(exc):
            raise
        pytest.skip(str(exc))


def test_import_has_no_registration_and_custom_target_keeps_its_resolver(monkeypatch):
    monkeypatch.setitem(microkernel._RESOLVERS, "custom-policy", lambda spec: [f"custom_{spec.MR}"])
    before = dict(microkernel._RESOLVERS), dict(microkernel._SHAPE_POLICIES)
    importlib.reload(legacy)
    assert (microkernel._RESOLVERS, microkernel._SHAPE_POLICIES) == before
    monkeypatch.setattr(legacy, "_codegen", lambda *a, **k: pytest.fail("custom target selected another compiler"))
    assert legacy.microkernel_features({"MR": 3}, target="custom-policy") == ["custom_3"]
    with pytest.raises(microkernel.UnsupportedAxis):
        legacy.microkernel_features({}, target="unregistered-target")


def test_real_oot_renderer_policy_and_discovery_are_the_selected_owners(selected):
    root = Path(selected.microkernel_policy_api().__file__).parents[3]
    knobs = load_yaml(root / "schedules/hand_v0/knobs.yaml")
    rendered = legacy.render_schedule(knobs, codegen_provider=selected)
    assert rendered == (root / "schedules/hand_v0/schedule.mlir").read_text()
    assert legacy.microkernel_features({"MR": 4, "NR": 16, "KC": 16}, codegen_provider=selected) == [
        "accumulator_resident_microkernel_v3",
        "erase_self_copy",
    ]
    assert legacy._rvv_best_block(4, 16, [(8, 344)], codegen_provider=selected) == (4, 8)
    from merlin.kernels import action_catalog, regions

    source = str(selected.microkernel_policy_api().__file__)
    assert action_catalog.seam_location("schedule:op_match", codegen_provider=selected)["seam_file"] == source
    points = [
        point
        for _, point in regions.all_edit_points(codegen_provider=selected)
        if point.seam in {"schedule:op_match", "schedule:vector_sizes"}
    ]
    assert points and all(point.file == source and point.forkable_now for point in points)
    record = legacy.policy_source_record(selected)
    assert any(row["path"].endswith("policy/blocking_cost.py") for row in record["members"])
    assert any(row["path"].endswith("service.py") for row in record["members"])
    assert record["complete_dependency_identity"] is False


def test_explicit_invalid_selection_never_falls_through_to_installed_backend(monkeypatch, selected):
    from merlin.compiler.backend import service
    from merlin.targetgen.plugins import PluginError

    monkeypatch.setattr(service, "resolve", lambda *a, **k: pytest.fail("invalid explicit selection fell through"))
    with pytest.raises(PluginError):
        legacy.render_schedule({}, support_target="nonexistent-support")
    assert "transform.named_sequence" in legacy.render_schedule({}, codegen_provider=selected)
    with pytest.raises(ValueError, match="API_VERSION"):
        legacy.render_schedule({}, codegen_provider=object())


def test_missing_installed_policy_and_policy_source_mutation_refuse(monkeypatch, tmp_path):
    from merlin.common import paths
    from merlin.compiler.backend import service

    monkeypatch.setattr(paths, "env", lambda name: None)

    def absent(*args, **kwargs):
        raise ValueError("no installed backend")

    monkeypatch.setattr(service, "resolve", absent)
    with pytest.raises(RuntimeError, match="requires its OOT package"):
        legacy.render_schedule({})
    member = tmp_path / "selected-policy.py"
    member.write_text("before")

    def changed(knobs):
        member.write_text("after")
        return "candidate schedule"

    provider = SimpleNamespace(API_VERSION=1, render_schedule=changed, policy_source_files=lambda: [member])
    with pytest.raises(ValueError, match="source changed"):
        legacy.render_schedule({}, codegen_provider=provider)


def test_fork_retains_lineage_and_actual_selected_renderer_sources(selected, tmp_path):
    pytest.importorskip("merlin.mining.fork")
    from merlin.compiler.backend.host import load_host_package
    from merlin.mining.route_report import _ladder_for, _regions_view

    root = Path(selected.microkernel_policy_api().__file__).parents[3]
    parent = load_host_package(root / "schedules/hand_v0", codegen_provider=selected)
    original = parent.schedule_text
    result = legacy.mint_fork(
        parent,
        {"contraction_strategy": "outerproduct"},
        version=1,
        depth=1,
        timestamp="test",
        source_evidence=["policy-owner-test"],
        lever="lowering_pattern",
        out_root=tmp_path,
    )
    fork = load_host_package(result, codegen_provider=selected)
    assert fork.manifest["lineage"]["parent_run_id"] == parent.run_id
    assert fork.manifest["lineage"]["source_evidence"] == ["policy-owner-test"]
    assert "outerproduct" in fork.schedule_text and parent.schedule_text == original
    assert fork.manifest["authoring"]["renderer_sources"] == legacy.policy_source_record(selected)
    source = str(selected.microkernel_policy_api().__file__)
    views = _regions_view(codegen_provider=selected)
    points = [
        point
        for region in views
        for point in region["edit_points"]
        if point["seam"] in {"schedule:op_match", "schedule:vector_sizes"}
    ]
    assert points and all(point["file"] == source for point in points)
    ladder = _ladder_for("compute.nr_is_vsetvlmax", "rvv", codegen_provider=selected)["ladder"]
    assert any(step["seam_file"] == source for step in ladder if step["target_seam"].startswith("schedule:"))


def test_copied_provider_and_service_refuse_source_drift(selected, tmp_path):
    import shutil

    from merlin.targetgen.plugins import load_module

    root = Path(selected.microkernel_policy_api().__file__).parents[3]
    copied = tmp_path / "selected-provider"
    shutil.copytree(root / "src", copied / "src")
    for relative in ("backend.py", "service.py", "provider.yaml"):
        shutil.copyfile(root / relative, copied / relative)
    provider = load_module(copied, "backend.py", package_name="copied-policy")
    assert provider.policy_source_files()
    member = copied / "src/rvv/policy/blocking_cost.py"
    member.write_text(member.read_text() + "\n# changed copied policy\n")
    with pytest.raises(ValueError, match="changed after import"):
        legacy._rvv_best_block(4, 16, [(8, 344)], codegen_provider=provider)
    service = load_module(copied, "service.py", package_name="copied-policy")
    member = copied / "service.py"
    member.write_text(member.read_text() + "\n# changed copied service\n")
    with pytest.raises(ValueError, match="service changed after import"):
        service.codegen_provider()
