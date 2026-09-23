"""Catalog selection and replay admission; no experiment process is executed."""

import copy
import json
import sys

import pytest
import yaml
from merlin_experiments import load_spec
from merlin_experiments import portfolio_catalog as P
from merlin_experiments.measured_launch import execution_environment
from merlin_experiments.phase2 import portfolio_cli as CLI
from merlin_experiments.runner import resolve_plan
from merlin_experiments.spec import SpecError


@pytest.fixture
def plan(tmp_path, monkeypatch):
    storage = tmp_path / "storage"
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(storage))
    source = tmp_path / "source"
    source.mkdir()
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("target: synthetic\n")
    campaign = tmp_path / "campaign.json"
    campaign.write_text(json.dumps({"descriptor": str(descriptor)}))
    deployment = tmp_path / "deployment.json"
    deployment.write_text("{}")
    selection = {
        "target": "synthetic",
        "output_root": str(storage),
        "source_root": str(source),
        "python_roots": ["python"],
        "lease_path": str(tmp_path / "lease"),
    }
    # CLI schema/source admission is exercised by test_portfolio_cli. This seam
    # isolates the catalog's mapping, overlap checks and immutable replay contract.
    monkeypatch.setattr(CLI, "load_deployment", lambda path: selection)
    monkeypatch.setattr(CLI, "deployment_source_inputs", lambda path: {"source": str(source)})
    document = {
        "schema_version": 1,
        "id": "synthetic-portfolio",
        "target": "synthetic",
        "phases": {
            2: {
                "adapter": "model_portfolio",
                "mode": "model_portfolio",
                "config": {
                    "campaign_config": str(campaign),
                    "deployment": str(deployment),
                    "candidate": str(tmp_path / "candidate"),
                    "round_seconds": 60,
                    "max_tool_calls": 5,
                    "max_rounds": 1,
                    "total_authoring_seconds": 60,
                },
            }
        },
    }
    definition = tmp_path / "definition.yaml"
    definition.write_text(yaml.safe_dump(document))
    return resolve_plan(load_spec(definition), run_dir=storage / "run")


def test_catalog_uses_installed_module_and_pins_nested_inputs(plan):
    command = plan["phases"]["2"]
    assert command["argv"][:3] == [sys.executable, "-m", P.MODULE]
    assert command["resume_policy"] == "checkpoint_segment"
    assert plan["input_paths"][P.PREFIX + "campaign:descriptor"].endswith("descriptor.yaml")
    assert P.PREFIX + "source" in plan["input_paths"]
    P.verify_plan(plan)


@pytest.mark.parametrize("mutation", ["argv", "env", "membership", "template", "owner", "fingerprints"])
def test_replay_refuses_changed_selection(plan, mutation):
    changed = copy.deepcopy(plan)
    command = changed["phases"]["2"]
    if mutation == "argv":
        command["argv"].append("--analysis-only")
    elif mutation == "env":
        command["env"]["PYTHONPATH"] = "/unselected"
    elif mutation == "membership":
        del changed["input_paths"][P.PREFIX + "source"]
    elif mutation == "template":
        command["portfolio_launch"]["template"] = True
    elif mutation == "owner":
        changed["target"] = "another"
    else:
        changed["inputs"] = {}
    with pytest.raises(SpecError):
        P.verify_plan(changed)


def test_execution_discards_ambient_python_selection(plan, monkeypatch):
    monkeypatch.setenv("PYTHONHOME", "/unselected")
    monkeypatch.setenv("PYTHONUSERBASE", "/unselected")
    monkeypatch.setenv("PYTHONPATH", "/unselected")
    command = plan["phases"]["2"]
    environment = execution_environment(command)
    assert "PYTHONHOME" not in environment and "PYTHONUSERBASE" not in environment
    assert environment["PYTHONPATH"] == command["env"]["PYTHONPATH"]


def test_replay_refuses_candidate_inside_pinned_source(plan):
    record = plan["phases"]["2"]["portfolio_launch"]
    record["values"]["candidate"] = plan["input_paths"][P.PREFIX + "source"] + "/candidate"
    plan["phases"]["2"] = P._command(
        **{
            key: record[key]
            for key in (
                "values",
                "target",
                "run_dir",
                "storage_root",
                "template",
            )
        }
    )
    with pytest.raises(SpecError, match="overlaps"):
        P.verify_plan(plan)
