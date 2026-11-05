"""The selected projection recipe reaches real capture and target readout owners."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import repo_root

_LOADER = """import torch
class Siblings(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.left=torch.nn.Linear(16,16,bias=False)
        self.right=torch.nn.Linear(16,32,bias=False)
    def forward(self,x):
        return self.left(x).sin(),self.right(x).cos()
def get_model_and_inputs():
    torch.manual_seed(177)
    return Siblings().eval(),(torch.randn(3,16),)
def get_calibration_inputs(model,inputs):
    yield inputs
    yield (inputs[0]*.5,)
"""


@pytest.fixture(scope="module")
def capture_request(tmp_path_factory):
    python, m2m = os.environ.get("MERLIN_M2M_PYTHON"), os.environ.get("MERLIN_M2M_DIR")
    if not python or not Path(python).is_file() or not m2m:
        pytest.skip("requires selected TorchAO capture interpreter and Model2MLIR checkout")
    directory = tmp_path_factory.mktemp("projection-capture")
    loader = directory / "loader.py"
    loader.write_text(_LOADER)
    recipe = repo_root() / "examples/gemmini/protocol/recipes/tiny_llama-native-speed.json"
    return directory, loader, recipe, python, m2m


def _check(meta):
    assert meta["ok"] and meta["opaque"] == 0
    stats = meta["quantization_stats"]
    assert stats["logical_annotated_contractions"] == 2
    assert stats["annotated_contractions"] == 1
    assert stats["integer_epilogue"]["by_exit"] == {"requant": 2, "accumulator": 0}
    projection = stats["projection_input_reuse"]
    assert projection["contractions_removed"] == 1 and projection["native_i8_exits_preserved"]
    assert projection["groups"][0]["output_columns"] == [16, 32]
    agreement = meta["integerization_receipt"]["golden_agreement"]
    assert agreement["status"] == "passed" and agreement["reference"] == "pt2e_integer"
    assert agreement["max_abs"] == 0
    assert meta["integerization_receipt"]["exported_integer_mm_count"] == 1


@pytest.fixture(scope="module")
def ordinary_capture(capture_request):
    from merlin.targetgen.capsule_source import PytorchRefSource

    directory, loader, recipe, python, m2m = capture_request
    source = PytorchRefSource(python=Path(python), m2m_dir=Path(m2m), timeout=90)
    # Explicit output ownership; this test checks the ordinary fresh worker,
    # rather than borrowing any persistent artifact or cached verdict.
    source._cache_slot = lambda *_args, **_kwargs: None
    result = source.capture_loader(
        loader,
        "int8",
        workdir=directory / "ordinary",
        recipe=json.loads(recipe.read_bytes()),
        env={"M2M_SESSION_STEPS": "2"},
        agreement_tolerance=(2e-6, 1e-5),
    )
    return result


def test_ordinary_capture_consumes_transformed_graph_and_real_integer_reference(ordinary_capture):
    _check(ordinary_capture.meta)


def test_stock_materialized_worker_retains_physical_contraction_census(capture_request):
    directory, loader, recipe, python, m2m = capture_request
    output = directory / "materialized"
    result = subprocess.run(
        [
            python,
            "-B",
            str(repo_root() / "src/merlin/targetgen/_m2m_capture_worker.py"),
            "--loader",
            str(loader),
            "--dtype",
            "int8",
            "--recipe",
            str(recipe),
            "--m2m-dir",
            m2m,
            "--out",
            str(output),
            "--materialize-bundle",
            "--agreement-atol",
            "0.000002",
            "--agreement-rtol",
            "0.00001",
        ],
        env={**os.environ, "M2M_SESSION_STEPS": "2"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    _check(json.loads((output / "meta.json").read_bytes()))
    assert (output / "model.mlir").is_file()
    assert (output / "weights.safetensors").is_file()
    assert (output / "capture_receipt.json").is_file()


def test_selected_target_proves_actual_per_command_i8_readout(ordinary_capture):
    if not os.environ.get("MERLIN_TARGET_CONTRACT") or not os.environ.get("MERLIN_RTL_FACTS"):
        pytest.skip("requires explicitly selected real target contract and RTL facts")
    from merlin.common import mlir_query as mq
    from merlin.llvmlower.source_compute_groups import form_exact_groups, form_source_groups
    from merlin.xdsl_dialects.lowering import compute_groups as groups
    from merlin.xdsl_dialects.lowering import group_command

    module = mq.parse(ordinary_capture.linalg_mlir)
    # No synthetic oracle or readout override: the selected target declares
    # its own physical width, readout formats and proof requirements.
    formed, _ = form_exact_groups(
        module,
        "gemmini",
        Path(ordinary_capture.weights_path).parent,
        form_source_groups,
    )
    (contraction,) = [g for g in formed if groups.CONTRACTION in g.stages]
    assert contraction.placement != groups.HOST
    assert contraction.fusion["cut"] == "requant"
    proof = contraction.fusion["proof"]
    assert proof["granularity"] == "per_command" and proof["columns_per_scale"] == 16
    assert proof["channels"] == 48
    program = group_command.program(contraction).entry
    assert program["output_dtype"] == "i8" and program["epilogue"] == ["acc_scale"]
    gains = program["acc_scale_by_column"]
    assert len(gains) == 48 and len(set(gains)) > 1
    assert all(len(set(gains[i : i + 16])) == 1 for i in range(0, 48, 16))
