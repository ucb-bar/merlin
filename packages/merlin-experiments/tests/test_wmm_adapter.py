"""The whole_model_measured mode is registered like the other Phase 2 modes: a catalog template, a typed
adapter that carries the experiment's prohibited roles into the run command, and the target's machine
registry and objective config as example-owned data."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from merlin_experiments.adapters import ADAPTERS
from merlin_experiments.phase2.whole_model_measured import cli
from merlin_experiments.phase2.whole_model_measured import config as C
from merlin_experiments.phase2.whole_model_measured import registry as R
from merlin_experiments.spec import SpecError, catalog, load_spec

from merlin.common.paths import repo_root

TEMPLATE = Path("experiments/definitions/whole-model-measured-template.yaml")


def test_the_template_is_catalogued_and_validates_against_its_adapter(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: pytest.fail("inspection launched a process"))
    root = repo_root()
    assert catalog()["whole-model-measured-template"].resolve() == (root / TEMPLATE).resolve()
    spec = load_spec(root / TEMPLATE)
    adapter = ADAPTERS["whole_model_measured"]
    assert adapter.mode == "whole_model_measured" and adapter.resume == "native_flag"
    config = spec.document["phases"]["2"]["config"]
    adapter.validate(config)
    assert {name for name, option in adapter.options.items() if option.required} <= config.keys()
    with pytest.raises(SpecError):
        adapter.validate({**config, "chia_wrapper": "x"})


def test_the_run_command_carries_the_declared_roles(tmp_path):
    import yaml

    root = repo_root()
    document = yaml.safe_load((root / TEMPLATE).read_text())
    document["policy"]["prohibited_instruction_roles"] = ["loop_descriptor"]
    definitions = tmp_path / "definitions"
    definitions.mkdir()
    (definitions / "measured.yaml").write_text(yaml.safe_dump(document))
    spec = load_spec(definitions / "measured.yaml")
    config = dict(spec.document["phases"]["2"]["config"])
    command = ADAPTERS["whole_model_measured"].resolve(spec, config, root, tmp_path / "run")
    argv = command["argv"]
    assert argv[argv.index("-m") + 1] == "merlin_experiments.phase2.whole_model_measured" and "run" in argv
    assert argv[argv.index("--prohibited-instruction-role") + 1] == "loop_descriptor"
    assert command["instruction_policy"]["prohibited_instruction_roles"] == ["loop_descriptor"]
    assert any(token.startswith("model_capsule=") for token in argv)
    assert command["mode"] == "whole_model_measured"


def test_the_targets_machine_registry_is_valid_data():
    document = R.load(repo_root() / "examples" / "gemmini" / "phase2" / "whole-model-machines.yaml")
    machines = document["machines"]
    assert machines["batched_board"]["kind"] == "batched" and machines["batched_board"]["timing"] in machines
    assert machines["batched_board"]["timing"] == "full_u250_board"
    board = machines[machines["batched_board"]["timing"]]
    assert board["hw_config"] == "alveo_u250_firesim_gemmini_rocket_30mhz"
    assert board["program_header_sha256"] == "3758ae967af3a179497660970201093a7fb624be00173990ce33d3f5c38da924"
    assert machines["lean_u250_board"]["hw_config"] != board["hw_config"]
    assert board["queue_command"][:7] == ["env", "-u", "HOME", "-u", "USER", "-u", "LOGNAME"]
    assert "cycle_count" in board["adjudicates"]


def test_the_stock_board_is_one_registered_full_width_device():
    """The stock board's hw-config names exactly one registered bitstream, whose declared ABI header is
    the one the board's programs are built against -- and not the lean board's narrow-readout header."""
    from merlin.common import provenance
    from merlin.perf import design_identity

    machines = R.load(repo_root() / "examples" / "gemmini" / "phase2" / "whole-model-machines.yaml")["machines"]
    board = machines[machines["stock_batched_board"]["timing"]]
    assert machines["stock_batched_board"]["kind"] == "batched" and board["kind"] == "firesim"
    (name,) = design_identity.hw_config_index()[board["hw_config"]]
    artifact = provenance.load_artifacts()[name]
    assert artifact.abi_header_sha256 == board["program_header_sha256"]
    assert artifact.abi_header_sha256 != machines["lean_u250_board"]["program_header_sha256"]
    assert board["queue_command"][:7] == ["env", "-u", "HOME", "-u", "USER", "-u", "LOGNAME"]
    assert board["lock_path"] == machines["lean_u250_board"]["lock_path"]  # one physical board


def test_the_example_objective_config_names_a_registry_machine_and_a_frozen_input():
    document = json.loads((repo_root() / "examples" / "gemmini" / "phase2" / "whole-model-objective.json").read_text())
    assert document["screen"]["machine"]["name"] == "batched_board"
    assert document["screen"]["build_options"]["model_capsule"] == "{input:model_capsule}"
    assert "builder" not in document and C.DEFAULT_BUILDER == "merlin.perf.whole_model_builder:build"
    assert document["mechanism_policy"] == C.DERIVED_MECHANISMS
    # The run is graded under the target's reviewed exactness contract, which lives beside this config.
    assert document["exactness"].endswith("examples/gemmini/phase2/exactness.yaml")


def test_a_new_run_needs_its_config_and_seed():
    with pytest.raises(SystemExit, match="objective-config"):
        cli.main(["prepare", "--target", "toy", "--method", "m", "--why", "w"])
