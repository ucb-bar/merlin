"""Augmented residual inputs retain the ordinary source routing proof."""

import json
import os
import subprocess

import pytest

from merlin.common.paths import repo_root
from merlin.tests.targetgen.test_integer_accumulator_residual import SCRIPT


def test_components_reach_ordinary_integer_source_groups(tmp_path):
    python, m2m = os.environ.get("MERLIN_M2M_PYTHON"), os.environ.get("MERLIN_M2M_DIR")
    if not python or not m2m:
        pytest.skip("selected capture interpreter and frontend required")
    artifact = tmp_path / "components.mlir"
    result = subprocess.run(
        [python, "-B", "-c", SCRIPT, str(repo_root() / "src/merlin/targetgen"), "lowering", m2m, str(artifact)],
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    from fake_quant_layer import Oracle

    from merlin.common import mlir_query as mq
    from merlin.frontends.capture_normalization import normalize_capture_mlir
    from merlin.xdsl_dialects.lowering import compute_groups as CG
    from merlin.xdsl_dialects.lowering import group_command

    text, receipt = normalize_capture_mlir(artifact.read_text())
    assert not receipt["remaining_opaque_detail"]
    groups = CG.form_groups(mq.parse(text), "synthetic", oracle=Oracle())
    devices = [g for g in groups if g.root is not None and g.placement != CG.HOST]
    assert len(devices) == 3
    weights = json.loads(artifact.with_name(artifact.name + ".weights.json").read_text())
    assert all(group_command.program(g, weight_args=weights).entry["op"] == "matmul" for g in devices)
