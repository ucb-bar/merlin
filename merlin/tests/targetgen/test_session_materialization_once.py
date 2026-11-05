"""The ordinary worker publishes each stage once and retains full trajectories."""

import json
import os
import subprocess

import numpy as np
import pytest

from merlin.common.paths import repo_root

LOADER = """
import torch
from m2m.capture.external_runtime import ExternalRuntimeProgram, make_external_runtime_session

class Prefix(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 4)
    def forward(self, x):
        return self.layer(x)

class Step(torch.nn.Module):
    def forward(self, context, state):
        value = context + state
        return value, value

class Capture:
    def external_runtime_session(self):
        first, step = Prefix().eval(), Step().eval()
        x = torch.ones(1, 4)
        context = first(x).detach()
        child = dict(kind="recurrent", steps=5, stages=["step"], streams=[],
                     states=[dict(name="state", input_index=1, output_index=1)],
                     quality=dict(output_index=0), paper_ready=False)
        programs = (ExternalRuntimeProgram("prefix", first, (x,), 1),
                    ExternalRuntimeProgram("step", step, (context, torch.zeros_like(context)), 5, child))
        return make_external_runtime_session(version=2, programs=programs, metadata=dict(
            kind="recurrent", stages=["prefix", "step"], states=["state"], quality_program="step",
            stage_schedule=[dict(name=p.name, steps=p.steps) for p in programs],
            bindings=[dict(name="context", **{"from":dict(program="prefix", output_index=0),
                          "to":dict(program="step", input_index=0)})]))

def get_model_and_inputs():
    return Capture(), ()
"""

RUNNER = """
import functools, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
import m2m.capture.bundle as B

calls, sessions, writes = [], [], []
original_bundle, original_session, original_savez = B.write_bundle, B.write_session_artifacts, np.savez
if len(sys.argv) > 7 and sys.argv[7] == "mutate":
    original_inventory, original_multi = B._shared_tensor_inventory, B.write_multi_program_bundle
    owners = []
    @functools.wraps(original_inventory)
    def inventory(model):
        owners.append(model)
        return original_inventory(model)
    @functools.wraps(original_multi)
    def multi(*args, **kwargs):
        result = original_multi(*args, **kwargs)
        import torch
        with torch.no_grad():
            next(owners[0].parameters()).add_(1)
        return result
    B._shared_tensor_inventory, B.write_multi_program_bundle = inventory, multi
@functools.wraps(original_bundle)
def bundle(*args, **kwargs):
    start = time.perf_counter()
    result = original_bundle(*args, **kwargs)
    calls.append(dict(stage=Path(args[2]).name, capture_regions=kwargs.get("capture_regions", True),
                      seconds=time.perf_counter()-start))
    return result
@functools.wraps(original_session)
def session(*args, **kwargs):
    start = time.perf_counter()
    result = original_session(*args, **kwargs)
    sessions.append(dict(stage=Path(args[2]).name, steps=result["session_steps"],
                         seconds=time.perf_counter()-start))
    return result
@functools.wraps(original_savez)
def savez(path, *args, **kwargs):
    writes.append(dict(file=Path(path).name, bytes=sum(np.asarray(v).nbytes for v in (*args, *kwargs.values()))))
    return original_savez(path, *args, **kwargs)
B.write_bundle, B.write_session_artifacts, np.savez = bundle, session, savez
spec=importlib.util.spec_from_file_location("worker_under_test", sys.argv[1])
W=importlib.util.module_from_spec(spec); spec.loader.exec_module(W)
start=time.perf_counter()
rc=W.main(["--m2m-dir",sys.argv[2],"--loader",sys.argv[3],"--dtype","int8",
           "--recipe",sys.argv[4],"--materialize-bundle","--seed","3","--out",sys.argv[5]])
Path(sys.argv[6]).write_text(json.dumps(dict(returncode=rc, calls=calls, sessions=sessions,
    writes=writes, seconds=time.perf_counter()-start)))
raise SystemExit(rc)
"""


def recipe():
    from merlin.targetgen.quant_recipe import digest

    tensor = dict(dtype="int8", granularity="tensor", symmetric=True, block=None, mode="static")
    value = dict(
        schema="quant_recipe_v1",
        target="t",
        unit="matrix",
        status="derived",
        weight={**tensor, "quant_min": -127, "quant_max": 127},
        activation={**tensor, "quant_min": -128, "quant_max": 127},
        families=["contraction"],
        bias_domain="accumulator",
        software_numerical_engine="integer_reference",
    )
    value["recipe_sha256"] = digest(value)
    return value


@pytest.mark.parametrize("mutation", [False, True])
def test_materialize_each_selected_stage_and_carried_trajectory_once(tmp_path, mutation):
    python, m2m = os.environ.get("MERLIN_M2M_PYTHON"), os.environ.get("MERLIN_M2M_DIR")
    if not python or not m2m:
        pytest.skip("an explicit capture interpreter is required")
    loader, runner, selected = (tmp_path / name for name in ("loader.py", "runner.py", "recipe.json"))
    loader.write_text(LOADER)
    runner.write_text(RUNNER)
    selected.write_text(json.dumps(recipe()))
    output, metrics = tmp_path / "capture", tmp_path / "metrics.json"
    command = [
        python,
        "-B",
        str(runner),
        str(repo_root() / "src/merlin/targetgen/_m2m_capture_worker.py"),
        m2m,
        str(loader),
        str(selected),
        str(output),
        str(metrics),
    ]
    if mutation:
        command.append("mutate")
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=120,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
    )
    if mutation:
        assert result.returncode != 0
        assert "source weights changed during bundle publication" in result.stderr
        assert not (output / "session-receipt.json").exists()
        return
    assert result.returncode == 0, result.stderr[-6000:]
    observed = json.loads(metrics.read_bytes())
    assert [(row["stage"], row["capture_regions"]) for row in observed["calls"]] == [("prefix", True), ("step", True)]
    assert [(row["stage"], row["steps"]) for row in observed["sessions"]] == [("step", 5)]
    for stage in ("prefix", "step"):
        assert (output / "stages" / stage / "capture_receipt.json").is_file()
    assert (output / "session-receipt.json").is_file()
    with np.load(output / "stages/step/session_goldens.npz") as saved:
        values = saved["output0"]
        assert values.shape == (5, 1, 4)
        np.testing.assert_allclose(values, values[0:1] * np.arange(1, 6)[:, None, None], atol=1e-6)
    meta = json.loads((output / "stages/prefix/meta.json").read_bytes())
    assert meta["integerization_receipt"]["golden_agreement"]["reference"] == "pt2e_integer"
    assert meta["integerization_receipt"]["golden_agreement"]["max_abs"] == 0
