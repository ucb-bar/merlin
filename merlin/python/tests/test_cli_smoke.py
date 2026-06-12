"""Smoke tests for the design-pressure and dse CLIs."""
from merlin.design_pressure import cli as dp_cli
from merlin.dse import cli as dse_cli


def test_design_pressure_cli_writes_artifacts(tmp_path):
    rc = dp_cli.main(["--workload", "vla_action_chunk_decode", "--H", "8", "--out", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "design_pressure.json").is_file()
    assert (tmp_path / "candidate_contracts.yaml").is_file()


def test_dse_cli_no_experiment(tmp_path):
    rc = dse_cli.main(["--workload", "vla_action_chunk_decode", "--no-experiment",
                       "--out", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "resident_packed_tensor" / "dse_result.yaml").is_file()
