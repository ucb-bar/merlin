"""Supplement production follows a real recurrent chain, including unjudged outputs."""

from __future__ import annotations

import importlib.util
import json
import os
import struct
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.model_check import bundle as B
from merlin.model_check import catalog as CAT
from merlin.model_check import chain as CH
from merlin.runtime import linalg_numpy as LN


@pytest.fixture
def operator():
    source = repo_root() / "examples/gemmini/phase1/launch/produce_all_outputs_reference.py"
    spec = importlib.util.spec_from_file_location("test_all_outputs_reference_operator", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def case(tmp_path, monkeypatch, operator):
    bundle = tmp_path / "bundle"
    stage = tmp_path / "capture"
    stage.mkdir()
    (stage / "model.mlir").write_text("""
module {
  func.func @forward(%x: tensor<1xf32>) -> (tensor<1xf32>, tensor<1xf32>) {
    %one = arith.constant dense<1.0> : tensor<1xf32>
    %y = arith.addf %x, %one : tensor<1xf32>
    return %y, %x : tensor<1xf32>, tensor<1xf32>
  }
}
""")
    header = b"{}"
    (stage / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    (stage / "weights.safetensors.manifest.json").write_text(json.dumps({"0": {"kind": "input", "name": "x"}}))
    np.savez(stage / "inputs.npz", in0=np.zeros(1, np.float32))
    configuration = bundle / "configuration/models.yaml"
    configuration.parent.mkdir(parents=True)
    configuration.write_text(
        yaml.safe_dump(
            {
                "schema": "merlin.model_check.models.v1",
                "target": "t",
                "models": {
                    "m": {
                        "workload": "w",
                        "chain": {"kind": "single"},
                        "programs": [{"name": "p", "steps": 3, "inputs": [{"from": "input", "key": "x", "carry": 0}]}],
                        "judged": {"kind": "logits", "outputs": [{"program": "p", "output": 0}]},
                    }
                },
            }
        )
    )
    for name in ("MANIFEST.json", "SHA256SUMS"):
        (bundle / name).write_text("{}")
    catalog = CAT.ProgramCatalog("t", stage, stage / "model.mlir", stage / "model.mlir", {}, {}, {})
    directory = bundle / "m/programs/p"
    directory.mkdir(parents=True)
    (directory / "catalog.json").write_text(json.dumps(catalog.record()))
    monkeypatch.setattr(B.Bundle, "catalog", lambda *_args: catalog)
    here = bundle / "m/visible/v00"
    here.mkdir(parents=True)
    np.savez(here / "int8.npz", **{f"out{i}": np.array([i + 2], np.float32) for i in range(3)})
    inputs = tmp_path / "derived/m/visible/v00"
    inputs.mkdir(parents=True)
    np.savez(inputs / "inputs.npz", x=np.ones(1, np.float32))
    config, launcher = tmp_path / "config.yaml", tmp_path / "launcher.py"
    config.write_text("selected: fixture\n")
    launcher.write_text("# selected fixture\n")
    out = tmp_path / "supplement"
    out.mkdir()
    args = SimpleNamespace(
        repo=str(repo_root()),
        model=["m"],
        worker_model="m",
        config=str(config),
        launcher=str(launcher),
        input="visible/v00",
        out=str(out),
        timeout=10,
    )
    monkeypatch.setattr(
        operator, "selected", lambda _: (None, SimpleNamespace(configured_environment=lambda _: nullcontext()))
    )
    monkeypatch.setenv("MERLIN_MODEL_CHECK_BUNDLE", str(bundle))
    monkeypatch.delenv("MERLIN_COMPUTE_CPUS", raising=False)
    return args, out / "m", here, inputs / "inputs.npz", stage


def test_real_recurrent_reference_persists_every_declared_output(operator, case):
    args, out, sealed, _inputs, stage = case
    original = (sealed / "int8.npz").read_bytes()
    operator.worker(args)
    manifest = json.loads((out / "reference.json").read_text())
    assert manifest["schema"] == "merlin.operator.full_declared_output_reference.v1"
    assert manifest["judged_crossjoin"]["bit_exact"] is True
    assert manifest["judged_crossjoin"]["outputs"] == 3
    assert manifest["implementation"]["numeric_contract"] == LN.numeric_contract()
    (program,) = manifest["programs"]
    assert program["steps_declared"] == 3
    assert len(program["declared_outputs"]) == 2
    for index, step in enumerate(program["steps"]):
        with np.load(step["outputs_file"]["path"], allow_pickle=False) as data:
            assert data.files == ["out0", "out1"]
            np.testing.assert_array_equal(data["out0"], [index + 2])
            np.testing.assert_array_equal(data["out1"], [index + 1])
        assert step["outputs_file"] == operator.pin(out / f"p.{index}.npz")
    assert (sealed / "int8.npz").read_bytes() == original
    assert any(row["path"] == str(stage / "weights.safetensors") for row in manifest["implementation"]["source_pins"])
    assert (out / "SHA256SUMS").is_file()


@pytest.mark.parametrize("mutation", ["judged", "policy", "input", "nonfinite", "arity"])
def test_no_manifest_is_published_under_a_changed_or_inconsistent_reference(operator, case, monkeypatch, mutation):
    args, out, sealed, input_path, _stage = case
    original = CH.run_reference

    def changed(*positional, **keywords):
        run = original(*positional, **keywords)
        if mutation == "judged":
            np.savez(sealed / "int8.npz", **{f"out{i}": np.zeros(1, np.float32) for i in range(3)})
        elif mutation == "policy":
            policy = LN.numeric_contract()
            monkeypatch.setattr(LN, "numeric_contract", lambda: {**policy, "version": "changed"})
        elif mutation == "input":
            np.savez(input_path, x=np.zeros(1, np.float32))
        elif mutation == "nonfinite":
            run.steps["p"][0][1] = np.array([np.nan], np.float32)
        else:
            run.steps["p"][0].append(np.ones(1, np.float32))
        return run

    monkeypatch.setattr(CH, "run_reference", changed)
    with pytest.raises(ValueError):
        operator.worker(args)
    assert not (out / "reference.json").exists()
    assert not (out / "SHA256SUMS").exists()


def test_roster_input_and_deadline_are_explicit_and_safe(operator, tmp_path):
    base = []
    for field in ("repo", "config", "launcher", "task-root", "prep", "run-id", "out"):
        base.extend(["--" + field, str(tmp_path)])
    base.extend(["--deadline", "2099-01-01T00:00:00+00:00", "--input", "visible/v00", "--model", "m"])
    assert operator.arguments(base).model == ["m"]
    for invalid in (
        ["--model", "m"],
        ["--model", "../m"],
        ["--input", "visible/../v00"],
        ["--deadline", "2099-01-01T00:00:00"],
        ["--worker", "--worker-model", "other"],
    ):
        with pytest.raises(SystemExit):
            operator.arguments([*base, *invalid])


def test_process_birth_identity_does_not_invent_missing_process_ticks(operator):
    assert operator.process_identity(2**30)["start_ticks"] is None


def test_whole_operator_reaps_failed_workers_and_preserves_launch_receipts(operator, tmp_path):
    out = tmp_path / "failed-operator"
    argv = [sys.executable, str(Path(operator.__file__).resolve())]
    for field in ("repo", "config", "launcher", "task-root", "prep", "run-id"):
        argv.extend(["--" + field, str(tmp_path / "intentionally-absent")])
    argv.extend(
        [
            "--out",
            str(out),
            "--deadline",
            "2099-01-01T00:00:00+00:00",
            "--input",
            "visible/v00",
            "--timeout",
            "10",
            "--min-headroom-gib",
            "0",
            "--model",
            "m",
            "--model",
            "n",
        ]
    )
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=20, check=False)
    assert completed.returncode != 0
    assert json.loads((out / "status.json").read_text())["status"] == "failed"
    assert json.loads((out / "cleanup.json").read_text())["complete"] is True
    jobs = json.loads((out / "jobs.json").read_text())
    assert set(jobs["children"]) == {"m", "n"}
    assert jobs["controller"]["start_ticks"] is not None
    assert jobs["controller"]["owner_uid"] == os.getuid()
    for model, child in jobs["children"].items():
        assert child["argv"][-3:] == ["--worker", "--worker-model", model]
        assert not Path(f"/proc/{child['pid']}").exists()
