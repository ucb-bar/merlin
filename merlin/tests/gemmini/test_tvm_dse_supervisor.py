"""External worker ownership and incremental evidence tracking."""

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest

from merlin.common.paths import repo_root


def load_supervisor():
    path = repo_root() / "examples/gemmini/comparisons/tvm/launch_dse.py"
    spec = importlib.util.spec_from_file_location("tvm_dse_supervisor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_progress_waits_for_complete_records(tmp_path):
    supervisor = load_supervisor()
    path = tmp_path / "trials.jsonl"
    progress = dict(offset=0, trials=0, valid_timings=0)
    path.write_text('{"cycles": 17}\n{"cycles":')
    assert supervisor.read_progress(path, progress) == dict(trials=1, valid_timings=1)
    with path.open("a") as stream:
        stream.write(" null}\n")
    assert supervisor.read_progress(path, progress) == dict(trials=2, valid_timings=1)
    assert supervisor.read_progress(path, progress) == dict(trials=2, valid_timings=1)


@pytest.mark.parametrize("scoped", [False, True, "deadline"])
def test_supervisor_collects_all_child_outcomes(tmp_path, monkeypatch, scoped):
    supervisor = load_supervisor()
    app = tmp_path / "tvm/apps/gemmini"
    app.mkdir(parents=True)
    (app / "tune_models.py").write_text("""
from pathlib import Path
import sys
root = Path(sys.argv[sys.argv.index("--work-dir") + 1])
root.mkdir()
(root / "trials.jsonl").write_text('{"cycles": 17}\\n')
if "--deadline-epoch" in sys.argv:
    import time
    time.sleep(30)
raise SystemExit(3 if root.name == "tinyllama-decode" else 0)
""")
    chain = tmp_path / "toolchain.json"
    chain.write_text(
        json.dumps(
            {
                key: "unused"
                for key in (
                    "gcc",
                    "adapter_receipt",
                    "spike_receipt",
                    "hardware_receipt",
                    "memory_variant_receipt",
                    "firesim_memory_artifact",
                    "memory_pin_registry",
                )
            }
        )
    )
    original_sleep = supervisor.time.sleep
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: original_sleep(0.02))
    monkeypatch.setattr(supervisor, "memory", lambda: {"MemAvailable": 16 * 2**30})
    root = tmp_path / "run"
    args = [
        "--tvm",
        str(tmp_path / "tvm"),
        "--python",
        sys.executable,
        "--toolchain",
        str(chain),
        "--resnet",
        "unused.pt2",
        "--work-dir",
        str(root),
        "--cpus",
        "1",
    ]
    if scoped:
        manifest = tmp_path / "campaigns.json"
        manifest.write_text(
            json.dumps(
                [dict(name="resnet-scoped", source_args=["--program", "unused.pt2"], scope="test scoped campaign")]
            )
        )
        shared = tmp_path / "shared-pool"
        shared.mkdir()
        shared_config = dict(cpus=[min(os.sched_getaffinity(0))], build_slots=1, min_available_bytes=4 * 2**30)
        (shared / "config.json").write_text(json.dumps(shared_config))
        args += ["--campaign-manifest", str(manifest), "--execution-pool", str(shared)]
    if scoped == "deadline":
        args += ["--deadline-epoch", str(time.time() + 2.4)]
    result = supervisor.main(args)
    assert result == (1 if scoped == "deadline" or not scoped else 0)
    outcome = json.loads((root / "exit.json").read_text())
    assert outcome["returncodes"] == (
        {"resnet-scoped": -15}
        if scoped == "deadline"
        else {"resnet-scoped": 0}
        if scoped
        else {"resnet-real": 0, "tinyllama-decode": 3, "smolvla-expert": 0}
    )
    status = json.loads((root / "status.json").read_text())
    assert status["status"] == ("deadline_reached" if scoped == "deadline" else "complete" if scoped else "failed")
    assert all(row["trials"] == 1 for row in status["campaigns"].values())
    if scoped:
        assert json.loads((shared / "config.json").read_text()) == shared_config
    if scoped == "deadline":
        assert outcome["deadline_reached"] and (root / "STOP").exists()


def test_frozen_deadline_cannot_be_extended(tmp_path, capsys):
    supervisor = load_supervisor()
    deadline = time.time() + 100
    (tmp_path / "policy.json").write_text(json.dumps(dict(deadline_epoch=deadline)))
    with pytest.raises(SystemExit) as error:
        supervisor.main(
            [
                "--tvm",
                "unused",
                "--python",
                sys.executable,
                "--toolchain",
                "unused",
                "--resnet",
                "unused",
                "--work-dir",
                str(tmp_path),
                "--deadline-epoch",
                str(deadline + 1),
            ]
        )
    assert error.value.code == 2
    assert "Cannot extend" in capsys.readouterr().err
