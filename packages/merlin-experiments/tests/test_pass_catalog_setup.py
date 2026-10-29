"""Exercise treatment selection without launching an author or target runtime."""

from types import SimpleNamespace

import pytest
from merlin_experiments.adapters import ADAPTERS
from merlin_experiments.cli import main
from merlin_experiments.corpus import release
from merlin_experiments.phase1.levels import level_for_phase1
from merlin_experiments.phase1.task_staging import TaskStagingConfig, arm_from_bundle_id, task_tool_inventory_block
from merlin_experiments.spec import SpecError, catalog, load_spec


def test_authored_task_receives_catalog_instruction_without_generated_prompt(tmp_path):
    selection = TaskStagingConfig(
        context=SimpleNamespace(),
        bundle_id="merlin_assisted_pass_catalog_public_v0",
        bundle_dir=tmp_path,
        experiment="full",
    )
    assert "pass_catalog.yaml" in task_tool_inventory_block(selection)
    assert "production pipeline invocation" in task_tool_inventory_block(selection)
    ablated = TaskStagingConfig(
        context=selection.context,
        bundle_id=selection.bundle_id,
        bundle_dir=tmp_path,
        experiment="full",
        drop_tools=("pass_catalog",),
    )
    assert "pass_catalog.yaml" not in task_tool_inventory_block(ablated)


def test_definition_adds_only_catalog_treatment_to_control():
    definitions = catalog()
    control = load_spec(definitions["gemmini-functional"]).document
    candidate = load_spec(definitions["gemmini-pass-catalog"]).document
    assert set(candidate["phases"]) == {"1"}
    assert candidate["policy"] == control["policy"]
    before = control["phases"]["1"]["config"]
    after = candidate["phases"]["1"]["config"]
    assert {key for key in before.keys() | after.keys() if before.get(key) != after.get(key)} == {
        "level",
        "bundle",
        "bundle_manifest",
    }
    assert level_for_phase1(after) == "EL4-P"
    assert arm_from_bundle_id(after["bundle"]) == "merlin_pass_catalog"
    for changed in ({"treatment": "baseline"}, {"bundle": before["bundle"]}):
        with pytest.raises(SpecError):
            ADAPTERS["capsule_bench"].validate({**after, **changed})


def test_prepare_cli_threads_explicit_arms_without_running(monkeypatch, tmp_path, capsys):
    observed = {}

    def prepare(run_dir, output, **kwargs):
        observed.update(kwargs)
        return {"state": "not_executed"}

    monkeypatch.setattr(release, "prepare", prepare)
    assert (
        main(
            [
                "corpus",
                "prepare",
                str(tmp_path),
                "--bundle-arm",
                "merlin_rtlchecks",
                "--bundle-arm",
                "merlin_pass_catalog",
            ]
        )
        == 0
    )
    assert observed["bundle_arms"] == ("merlin_rtlchecks", "merlin_pass_catalog")
    assert "not_executed" in capsys.readouterr().out


@pytest.mark.parametrize("arms", [("typo",), ("merlin_pass_catalog", "merlin_pass_catalog")])
def test_unknown_or_duplicate_arms_refuse_before_release_io(tmp_path, arms):
    with pytest.raises(SpecError, match="distinct registered arms"):
        release.prepare(tmp_path / "absent", bundle_arms=arms)
