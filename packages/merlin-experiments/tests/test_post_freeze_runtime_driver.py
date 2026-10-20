"""Exercise the actual driver boundary locally without AWS or an FPGA."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from merlin_experiments.post_freeze import elf_custody as C
from merlin_experiments.post_freeze import manager_result as M
from merlin_experiments.post_freeze import runtime_driver as R


@pytest.fixture
def case(tmp_path):
    runtime = tmp_path / "sim_slot_0"
    runtime.mkdir()
    boot = runtime / "boot.elf"
    boot.write_bytes(b"local selected target code")
    staged = runtime / "rsyncdir"
    staged.mkdir()
    (staged / "boot.elf").write_bytes(b"a different staging copy")
    script = runtime / "sim-run.sh"
    script.write_text("# actual outer driver script\n")
    driver = Path(sys.executable).resolve()
    model = {"id": "model", "assets": [{"role": "boot_elf", **C.artifact(boot)}]}
    request = {
        "schema": "merlin.post_freeze.firesim.request.v1", "status": "prepared",
        "deployment": {"mode": "direct_elf_ssm_on_demand"}, "models": [model],
        "budget": {"job_wall_s": 10},
        "manager_contract_definition": {
            "schema": C.CONTRACT_SCHEMA, "boot_argument_prefix": "+prog0=",
            "helpers": {name: C.identity(C.artifact(C.__file__))
                        for name in ("elf_custody", "chain_observation", "manager_result")},
            "runtime_execution": {"helper": C.identity(C.artifact(R.__file__)),
                                  "driver": C.identity(C.artifact(driver))},
        },
    }
    request_path = tmp_path / "request.json"
    C.write(request_path, request)
    download = tmp_path / "download.json"
    C.write(download, {"schema": C.SCHEMA, "kind": "versioned_download",
                       "version_id": "actual-version", "expected": C.identity(C.artifact(boot)),
                       "downloaded_elf": C.artifact(boot)})
    return {"request": request_path, "request_sha256": C.artifact(request_path)["sha256"],
            "model": "model", "download_record": download, "driver_script": script,
            "argv": [str(driver), "-c", "pass", "+prog0=boot.elf"], "cwd": runtime,
            "output": tmp_path / "observation", "wall_seconds": 5}


def run(case, code="pass"):
    case["argv"][2] = code
    result = R.execute(**case)
    assert C.read(case["output"] / "runtime-driver.json") == result
    return result


def test_actual_driver_argv_and_working_directory_select_boot_bytes(case):
    result = run(case)
    assert result["status"] == "driver_finished"
    assert result["returncode"] == 0
    assert result["scientific_qualification"] is False
    before = C.read(case["output"] / "elf-before.json")
    assert before["simulator_argv"] == case["argv"]
    assert before["consumed_elf"]["path"] == str(case["cwd"] / "boot.elf")
    assert (case["output"] / "elf-after.json").exists()


def test_staging_copy_cannot_substitute_for_actual_boot_operand(case):
    case["argv"][-1] = "+prog0=rsyncdir/boot.elf"
    result = run(case, "raise RuntimeError('must not execute')")
    assert result["status"] == "refused"
    assert result["returncode"] is None
    assert "ELF bytes differ" in result["error"]["reason"]


def test_post_run_boot_mutation_is_retained_as_failed_custody(case):
    result = run(case, "from pathlib import Path; Path('boot.elf').write_bytes(b'changed')")
    assert result["status"] == "custody_failed"
    assert result["returncode"] == 0
    assert not (case["output"] / "elf-after.json").exists()


def test_nonzero_driver_exit_preserves_after_observation(case):
    result = run(case, "raise SystemExit(7)")
    assert result["status"] == "driver_finished"
    assert result["returncode"] == 7
    assert result["after"]


def test_timeout_kills_and_reaps_local_driver(case):
    case["wall_seconds"] = 1
    result = run(case, "import time; time.sleep(30)")
    assert result["timed_out"] is True
    assert result["returncode"] == -9
    assert result["after"]
    assert result["elapsed_wall_seconds"] < 5


@pytest.mark.parametrize("field", ["helper", "driver"])
def test_unselected_runtime_sources_refuse_before_execution(case, field):
    request = json.loads(case["request"].read_bytes())
    request["manager_contract_definition"]["runtime_execution"][field]["sha256"] = "0" * 64
    case["request"].write_text(json.dumps(request))
    case["request_sha256"] = C.artifact(case["request"])["sha256"]
    result = run(case)
    assert result["status"] == "refused"
    assert result["returncode"] is None


def test_ambiguous_boot_argv_refuses(case):
    case["argv"].append("+prog0=another.elf")
    result = run(case)
    assert result["status"] == "refused"
    assert "exactly one" in result["error"]["reason"]


def test_over_budget_runtime_refuses(case):
    case["wall_seconds"] = 11
    result = run(case)
    assert result["returncode"] is None
    assert "remaining job budget" in result["error"]["reason"]


@pytest.mark.parametrize("code,complete", [("pass", True), ("raise SystemExit(7)", False)])
def test_manager_reopens_runtime_exit_and_custody(case, code, complete):
    run(case, code)
    directory = case["output"]
    records = {name: C.artifact(directory / name) for name in ("elf-before.json", "elf-after.json")}
    selected = C.read(case["request"])["manager_contract_definition"]
    before = C.read(directory / "elf-before.json")
    assert M._runtime_observation(selected, directory, records, before) is complete
    assert "runtime-driver.json" in records


@pytest.mark.parametrize("field", ["runtime", "argv", "working_directory", "after", "observer"])
def test_manager_refuses_runtime_record_substitution(case, field):
    run(case)
    directory = case["output"]
    row = C.read(directory / "runtime-driver.json")
    row[field] = {} if field not in {"argv", "working_directory"} else "different"
    (directory / "runtime-driver.json").write_text(json.dumps(row))
    records = {name: C.artifact(directory / name) for name in ("elf-before.json", "elf-after.json")}
    selected = C.read(case["request"])["manager_contract_definition"]
    with pytest.raises((ValueError, KeyError)):
        M._runtime_observation(selected, directory, records, C.read(directory / "elf-before.json"))


def test_standalone_deployment_does_not_import_experiment_dependencies(tmp_path):
    # The installed experiment package requires a newer Python. The manager
    # receives only these standard-library helpers under empty namespaces.
    package = tmp_path / "merlin_experiments"
    modules = package / "post_freeze"
    modules.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"")
    (modules / "__init__.py").write_bytes(b"")
    for name in ("elf_custody", "chain_observation", "manager_result", "runtime_driver"):
        shutil.copyfile(Path(R.__file__).with_name(name + ".py"), modules / (name + ".py"))
    interpreter = shutil.which("python3") or sys.executable
    program = (
        "import sys, runpy; sys.path.insert(0, sys.argv[1]); "
        "sys.argv=['runtime_driver', '--help']; "
        "runpy.run_module('merlin_experiments.post_freeze.runtime_driver', run_name='__main__')"
    )
    result = subprocess.run([interpreter, "-I", "-S", "-c", program, str(tmp_path)],
                            check=True, capture_output=True, text=True, timeout=10)
    assert "--download-record" in result.stdout
